import io
import os
import numpy as np
import soundfile as sf
import importlib.util
import wave
import json
import requests
from tqdm import tqdm
try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional in tests
    OpenAI = None
try:
    from groq import Groq
except Exception:  # pragma: no cover - optional in tests
    Groq = None

from utils import ConfigManager
from keyring_manager import KeyringManager
from text_processor import TextProcessor
from model_registry import AZURE_MODEL_FAMILY_AUTO, resolve_transcription_capabilities
from openai_clients import (
    AZURE_MODE_LEGACY,
    build_azure_client,
    build_openai_client,
    resolve_azure_api_mode,
)
from vocabulary import get_transcription_keywords
from whisper_languages import normalize_whisper_language

# (connect, read) timeouts for transcription HTTP calls made without the SDK. The
# read budget has to cover uploading and transcribing a full recording, so it is
# deliberately generous.
REQUEST_TIMEOUT = (10, 180)
TRANSCRIPTION_TIMEOUT = 180.0

# Azure audio still lives on the dated api-version path; this version is the oldest
# one confirmed to serve gpt-transcribe.
DEFAULT_AZURE_AUDIO_API_VERSION = '2025-03-01-preview'

VOSK_MODEL_URLS = {
    'vosk-model-small-en-us-0.15': 'https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip',
    'vosk-model-en-us-0.22': 'https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip'
}

# Check if GPU packages are available
HAS_FASTER_WHISPER = importlib.util.find_spec("faster_whisper") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None

# Add check for Vosk availability
HAS_VOSK = importlib.util.find_spec("vosk") is not None

def get_recording_sample_rate() -> int:
    """Return the configured recording sample rate, falling back to 16 kHz."""
    return ConfigManager.get_config_section('recording_options').get('sample_rate', 16000)

def get_transcription_hints() -> dict:
    """Return optional transcription hints configured for the current model."""
    common_options = ConfigManager.get_config_section('model_options').get('common', {})
    return {
        'language': normalize_whisper_language(common_options.get('language')),
        'prompt': common_options.get('initial_prompt'),
        'temperature': common_options.get('temperature')
    }

def get_configured_languages() -> list:
    """Languages for models taking the plural `languages` field.

    Falls back to the single configured language, and to nothing at all when that
    is 'auto', which leaves the model free to detect the language itself.
    """
    common_options = ConfigManager.get_config_section('model_options').get('common', {})
    configured = common_options.get('languages')

    if configured:
        codes = [part.strip() for part in str(configured).replace(';', ',').split(',')]
        normalized = [normalize_whisper_language(code) for code in codes if code.strip()]
        return [code for code in normalized if code]

    single = normalize_whisper_language(common_options.get('language'))
    return [single] if single else []

def apply_transcription_hints(request_data: dict) -> dict:
    """Populate supported transcription hint parameters into an API request payload."""
    hints = get_transcription_hints()
    language = hints.get('language')
    prompt = hints.get('prompt')
    temperature = hints.get('temperature')

    if language:
        request_data['language'] = language
    if prompt:
        request_data['prompt'] = prompt
    if temperature is not None:
        request_data['temperature'] = temperature

    ConfigManager.console_print(
        (
            "Transcription hints: "
            f"language={language or 'auto'}, "
            f"prompt={'set' if prompt else 'not set'}, "
            f"temperature={temperature if temperature is not None else 'default'}"
        ),
        verbose=True
    )
    return request_data

def build_transcription_request(model: str, family: str = AZURE_MODEL_FAMILY_AUTO) -> dict:
    """Build the request arguments a specific transcription model accepts.

    Models differ in ways that fail quietly rather than loudly: gpt-transcribe
    ignores the singular `language` field instead of rejecting it, so sending the
    wrong one means the language hint is simply lost.
    """
    capabilities = resolve_transcription_capabilities(model, family)
    hints = get_transcription_hints()
    request = {}

    if capabilities.language_param == 'languages':
        languages = get_configured_languages()
        if languages:
            request['languages'] = languages
    elif capabilities.language_param == 'language' and hints.get('language'):
        request['language'] = hints['language']

    if capabilities.supports_prompt and hints.get('prompt'):
        request['prompt'] = hints['prompt']

    if capabilities.supports_temperature and hints.get('temperature') is not None:
        request['temperature'] = hints['temperature']

    if capabilities.supports_keywords:
        keywords = get_transcription_keywords()
        if keywords:
            request['keywords'] = keywords

    ConfigManager.console_print(
        "Transcription request: "
        + ", ".join(
            f"{key}={len(value) if isinstance(value, list) else value!r}"
            for key, value in sorted(request.items())
        ),
        verbose=True
    )
    return request

def encode_wav(audio_data) -> bytes:
    """Encode recorded samples as an in-memory WAV file for upload."""
    byte_io = io.BytesIO()
    sf.write(byte_io, audio_data, get_recording_sample_rate(), format='wav')
    return byte_io.getvalue()

def check_upload_size(wav_bytes: bytes, capabilities) -> bool:
    """Warn and refuse when a recording exceeds the API's hard file-size limit."""
    if len(wav_bytes) <= capabilities.max_file_bytes:
        return True
    ConfigManager.console_print(
        f"Recording is {len(wav_bytes) / 1024 / 1024:.1f} MB, over the "
        f"{capabilities.max_file_bytes / 1024 / 1024:.0f} MB limit for the transcription API. "
        "Record shorter clips or switch to a local model."
    )
    return False

def read_transcription_text(result) -> str:
    """Pull the text out of an SDK transcription result, logging detected languages."""
    detected = getattr(result, 'languages', None)
    if detected:
        codes = [
            item.get('code') if isinstance(item, dict) else getattr(item, 'code', None)
            for item in detected
        ]
        ConfigManager.console_print(
            f"Detected language(s): {', '.join(code for code in codes if code)}", verbose=True
        )
    return getattr(result, 'text', '') or ''

def is_vosk_model(model_name: str) -> bool:
    """Check if the model name is a Vosk model"""
    return model_name in VOSK_MODEL_URLS

def get_model_path(model_name: str) -> str:
    """Get the path where the model should be stored"""
    base_path = os.path.join(os.path.expanduser('~'), '.whisperwriter', 'models')
    if is_vosk_model(model_name):
        return os.path.join(base_path, 'vosk', model_name)
    return os.path.join(base_path, 'whisper', model_name)

def download_vosk_model(model_name: str) -> bool:
    """Download a Vosk model if not present"""
    if model_name not in VOSK_MODEL_URLS:
        ConfigManager.console_print(f"Error: Unknown Vosk model {model_name}")
        return False
        
    model_path = get_model_path(model_name)
    url = VOSK_MODEL_URLS[model_name]
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    try:
        ConfigManager.console_print(f"Downloading Vosk model {model_name}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        zip_path = model_path + '.zip'
        with open(zip_path, 'wb') as f, tqdm(
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
                
        ConfigManager.console_print("Extracting model...")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get the name of the directory inside the zip
            root_dir = zip_ref.namelist()[0].split('/')[0]
            zip_ref.extractall(os.path.dirname(model_path))
            
            # If the extracted directory name doesn't match our expected path, rename it
            extracted_path = os.path.join(os.path.dirname(model_path), root_dir)
            if extracted_path != model_path:
                if os.path.exists(model_path):
                    import shutil
                    shutil.rmtree(model_path)
                os.rename(extracted_path, model_path)
            
        os.remove(zip_path)
        ConfigManager.console_print(f"Vosk model {model_name} downloaded and extracted successfully")
        return True
        
    except Exception as e:
        ConfigManager.console_print(f"Error downloading Vosk model: {str(e)}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return False

def get_optimal_device():
    """
    Determine the best available device for Whisper inference.
    Returns device string: 'mps', 'cuda', 'rocm', or 'cpu'
    """
    if not HAS_TORCH:
        ConfigManager.console_print("Torch not available, defaulting to API mode")
        return None
        
    import torch
    ConfigManager.console_print(f"PyTorch version: {torch.__version__}")
    ConfigManager.console_print(f"PyTorch CUDA version: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'Not available'}")
    
    # Check CUDA availability with detailed logging
    if torch.cuda.is_available():
        cuda_device_count = torch.cuda.device_count()
        cuda_device_name = torch.cuda.get_device_name(0) if cuda_device_count > 0 else "Unknown"
        ConfigManager.console_print(f"CUDA is available. Found {cuda_device_count} device(s)")
        ConfigManager.console_print(f"CUDA device name: {cuda_device_name}")
        ConfigManager.console_print(f"CUDA capability: {torch.cuda.get_device_capability()}")
        ConfigManager.console_print(f"CUDA arch list: {torch.cuda.get_arch_list() if hasattr(torch.cuda, 'get_arch_list') else 'Not available'}")
    else:
        ConfigManager.console_print("CUDA is not available. Checking why...")
        if not hasattr(torch, 'cuda'):
            ConfigManager.console_print("PyTorch was not built with CUDA support")
        else:
            ConfigManager.console_print("PyTorch has CUDA support but no CUDA devices were found")
    
    # Device selection logic
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        ConfigManager.console_print("Using Apple Silicon GPU (MPS)")
        return "mps"
    elif torch.cuda.is_available():
        ConfigManager.console_print("Using NVIDIA GPU (CUDA)")
        return "cuda"
    elif hasattr(torch, 'hip') and torch.hip.is_available():
        ConfigManager.console_print("Using AMD GPU (ROCm)")
        return "rocm"
    else:
        ConfigManager.console_print("Using CPU")
        return "cpu"

def create_local_model():
    """Create a local model using Whisper."""
    if not HAS_FASTER_WHISPER and not HAS_VOSK:
        ConfigManager.console_print("Neither Faster Whisper nor Vosk available, defaulting to API mode")
        return None
        
    local_model_options = ConfigManager.get_config_section('model_options')['local']
    model_name = local_model_options['model']
    
    if is_vosk_model(model_name):
        if not HAS_VOSK:
            ConfigManager.console_print("Vosk not available, defaulting to API mode")
            return None
            
        # Import Vosk components only when needed
        from vosk import Model, SetLogLevel
        
        SetLogLevel(-1)  # Reduce Vosk logging noise
        model_path = get_model_path(model_name)
        
        if not os.path.exists(model_path):
            if not download_vosk_model(model_name):
                return None
                
        try:
            model = Model(model_path)
            ConfigManager.console_print('Vosk model created.')
            return ('vosk', model)
        except Exception as e:
            ConfigManager.console_print(f'Error initializing Vosk model: {e}')
            return None
    else:
        if not HAS_FASTER_WHISPER:
            ConfigManager.console_print("Faster Whisper not available, defaulting to API mode")
            return None
            
        # Import Whisper components only when needed
        from faster_whisper import WhisperModel
        
        ConfigManager.console_print('Creating local model...')
        compute_type = local_model_options['compute_type']
        model_path = local_model_options.get('model_path')

        if compute_type == 'int8':
            device = 'cpu'
            ConfigManager.console_print('Using int8 quantization, forcing CPU usage.')
        else:
            device = local_model_options.get('device', 'auto')
            if device == 'auto':
                device = get_optimal_device()

        try:
            if model_path:
                ConfigManager.console_print(f'Loading Whisper model from: {model_path}')
                model = WhisperModel(model_path,
                                   device=device,
                                   compute_type=compute_type,
                                   download_root=None)
            else:
                model = WhisperModel(local_model_options['model'],
                                   device=device,
                                   compute_type=compute_type)
            ConfigManager.console_print('Whisper model created.')
            return ('whisper', model)
        except Exception as e:
            ConfigManager.console_print(f'Error initializing WhisperModel: {e}')
            ConfigManager.console_print('Falling back to CPU.')
            model = WhisperModel(model_path or local_model_options['model'],
                               device='cpu',
                               compute_type=compute_type,
                               download_root=None if model_path else None)
            return ('whisper', model)

def transcribe_local(audio_data, local_model=None):
    """Transcribe audio using a local model (Whisper or Vosk)."""
    if not local_model:
        local_model = create_local_model()
    if not local_model:
        return ''
        
    model_type, model = local_model
    model_options = ConfigManager.get_config_section('model_options')
    
    if model_type == 'vosk':
        if not HAS_VOSK:
            ConfigManager.console_print("Vosk not available")
            return ''
            
        # Import Vosk components only when needed
        from vosk import KaldiRecognizer
        
        # Convert audio data to WAV format for Vosk
        byte_io = io.BytesIO()
        sample_rate = ConfigManager.get_config_section('recording_options').get('sample_rate', 16000)
        ConfigManager.console_print(f"Converting audio for Vosk (sample rate: {sample_rate}Hz)")
        sf.write(byte_io, audio_data, sample_rate, format='wav')
        byte_io.seek(0)
        
        try:
            # Process with Vosk
            wf = wave.open(byte_io, "rb")
            recognizer = KaldiRecognizer(model, wf.getframerate())
            recognizer.SetWords(True)  # Enable word timing info
            
            transcription = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    if 'text' in result and result['text'].strip():
                        transcription.append(result['text'])
                    
            # Get final bits of audio
            final_result = json.loads(recognizer.FinalResult())
            if 'text' in final_result and final_result['text'].strip():
                transcription.append(final_result['text'])
                
            return ' '.join(transcription)
            
        except Exception as e:
            ConfigManager.console_print(f"Error transcribing with Vosk: {str(e)}")
            return ''
    else:
        # Existing Whisper transcription logic
        audio_data_float = audio_data.astype(np.float32) / 32768.0
        hints = get_transcription_hints()
        response = model.transcribe(
            audio=audio_data_float,
            language=hints['language'],
            initial_prompt=hints['prompt'],
            condition_on_previous_text=model_options['local']['condition_on_previous_text'],
            temperature=hints['temperature'],
            vad_filter=model_options['local']['vad_filter'],
        )
        return ''.join([segment.text for segment in list(response[0])])

def transcribe_api(audio_data):
    """Transcribe audio using an API service (OpenAI, Azure OpenAI, Deepgram, or Groq)."""
    api_options = ConfigManager.get_config_section('model_options')['api']
    provider = api_options['provider']
    model = api_options['model']
    
    ConfigManager.console_print(f"\n=== Using {provider.upper()} API Service ===")
    ConfigManager.console_print(f"Selected model: {model}")
    
    if provider == 'openai':
        return transcribe_with_openai(audio_data, api_options)
    elif provider == 'azure_openai':
        return transcribe_with_azure_openai(audio_data, api_options)
    elif provider == 'deepgram':
        return transcribe_with_deepgram(audio_data, api_options)
    elif provider == 'groq':
        return transcribe_with_groq(audio_data, api_options)
    else:
        ConfigManager.console_print(f"Unknown API provider: {provider}")
        return ''

def transcribe_with_openai(audio_data, api_options):
    """Transcribe audio through the OpenAI API (or an OpenAI-compatible endpoint)."""
    try:
        api_key = KeyringManager.get_api_key("openai_transcription")
        if not api_key:
            ConfigManager.console_print("OpenAI API key not found in keyring")
            return ''

        model = api_options['model']
        capabilities = resolve_transcription_capabilities(model)

        wav_bytes = encode_wav(audio_data)
        if not check_upload_size(wav_bytes, capabilities):
            return ''

        base_url = api_options.get('base_url') or None
        client = build_openai_client(api_key, base_url=base_url, timeout=TRANSCRIPTION_TIMEOUT)

        ConfigManager.console_print(f"Sending request to OpenAI API using {model}...")
        result = client.audio.transcriptions.create(
            model=model,
            file=('audio.wav', wav_bytes, 'audio/wav'),
            **build_transcription_request(model)
        )

        text = read_transcription_text(result)
        ConfigManager.console_print(f"Transcription: {text}")
        return text

    except Exception as e:
        ConfigManager.console_print(f"Error transcribing with OpenAI: {str(e)}")
        return ''

def transcribe_with_azure_openai(audio_data, api_options):
    """Transcribe audio through Azure OpenAI / Microsoft Foundry.

    Defaults to the classic `/openai/deployments/<name>/audio/transcriptions` path:
    the v1 API does not serve audio on Azure resources yet (it answers
    DeploymentNotFound even for deployments that work on the classic path).
    """
    try:
        api_key = KeyringManager.get_api_key("azure_openai_transcription")
        if not api_key:
            ConfigManager.console_print("Azure OpenAI API key not found in keyring")
            return ''

        endpoint = api_options.get('azure_openai_endpoint')
        api_version = api_options.get('azure_openai_api_version') or DEFAULT_AZURE_AUDIO_API_VERSION
        deployment_name = api_options.get('azure_openai_deployment_name')
        api_mode = resolve_azure_api_mode(
            api_options.get('azure_api_mode') or AZURE_MODE_LEGACY, api_version
        )

        if not deployment_name:
            ConfigManager.console_print("Azure OpenAI deployment name not configured")
            return ''

        # On Azure the deployment name identifies the model, so it also drives the
        # capability lookup unless the user pinned the family explicitly.
        family = api_options.get('azure_openai_model_family') or AZURE_MODEL_FAMILY_AUTO
        capabilities = resolve_transcription_capabilities(deployment_name, family)

        wav_bytes = encode_wav(audio_data)
        if not check_upload_size(wav_bytes, capabilities):
            return ''

        client = build_azure_client(
            api_key,
            endpoint,
            api_mode=api_mode,
            api_version=api_version,
            timeout=TRANSCRIPTION_TIMEOUT,
        )

        ConfigManager.console_print(
            f"Sending request to Azure OpenAI deployment {deployment_name} ({api_mode} API)..."
        )
        result = client.audio.transcriptions.create(
            model=deployment_name,
            file=('audio.wav', wav_bytes, 'audio/wav'),
            **build_transcription_request(deployment_name, family)
        )

        text = read_transcription_text(result)
        ConfigManager.console_print(f"Transcription: {text}")
        return text

    except Exception as e:
        ConfigManager.console_print(f"Error transcribing with Azure OpenAI: {str(e)}")
        return ''

def transcribe_with_deepgram(audio_data, api_options):
    """Transcribe audio using Deepgram's API."""
    try:
        api_key = KeyringManager.get_api_key("deepgram_transcription")
        if not api_key:
            ConfigManager.console_print("Deepgram API key not found in keyring")
            return ''
            
        # Convert audio to WAV format
        byte_io = io.BytesIO()
        sample_rate = get_recording_sample_rate()
        sf.write(byte_io, audio_data, sample_rate, format='wav')
        audio_data = byte_io.getvalue()
        
        headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "audio/wav"
        }
        
        # Set up Deepgram-specific parameters
        model = api_options['model']
        
        params = {
            "model": model,  # Use the full model name (nova-2 or nova-3)
            "smart_format": "true",
            "punctuate": "true",
        }
        
        # Always use Deepgram's API URL
        DEEPGRAM_BASE_URL = "https://api.deepgram.com/v1/listen"
        ConfigManager.console_print(f"Using Deepgram endpoint: {DEEPGRAM_BASE_URL}")
        ConfigManager.console_print(f"Model parameters: {params}")
        
        ConfigManager.console_print("Sending request to Deepgram API...")
        response = requests.post(
            DEEPGRAM_BASE_URL,
            headers=headers,
            params=params,
            data=audio_data,
            timeout=REQUEST_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            transcription = result['results']['channels'][0]['alternatives'][0]['transcript']
            ConfigManager.console_print("Deepgram API request successful")
            ConfigManager.console_print(f"Transcription: {transcription}")
            return transcription
        else:
            ConfigManager.console_print(f"Deepgram API error: {response.text}")
            return ''
            
    except Exception as e:
        ConfigManager.console_print(f"Error transcribing with Deepgram: {str(e)}")
        return ''

def transcribe_with_groq(audio_data, api_options):
    """Transcribe audio using Groq's Whisper API."""
    try:
        api_key = KeyringManager.get_api_key("groq_transcription")
        if not api_key:
            ConfigManager.console_print("Groq API key not found in keyring")
            return ''
            
        # Convert audio to WAV format
        byte_io = io.BytesIO()
        sample_rate = get_recording_sample_rate()
        sf.write(byte_io, audio_data, sample_rate, format='wav')
        byte_io.seek(0)
        
        if Groq is None:
            ConfigManager.console_print("Groq SDK not available. Please install 'groq' package or choose a different API.")
            return ''

        client = Groq(api_key=api_key)
        
        model = api_options['model']
        language = ConfigManager.get_config_value('model_options', 'common', 'language')
        initial_prompt = ConfigManager.get_config_value('model_options', 'common', 'initial_prompt')
        temperature = ConfigManager.get_config_value('model_options', 'common', 'temperature')
        
        ConfigManager.console_print("Sending request to Groq API...")
        response = client.audio.transcriptions.create(
            file=('audio.wav', byte_io.read()),
            model=model,
            prompt=initial_prompt,
            response_format="json",
            language=language or "en",
            temperature=temperature
        )
        
        if response and hasattr(response, 'text'):
            ConfigManager.console_print("Groq API request successful")
            ConfigManager.console_print(f"Transcription: {response.text}")
            return response.text
        else:
            ConfigManager.console_print("No transcription received from Groq API")
            return ''
            
    except Exception as e:
        ConfigManager.console_print(f"Error transcribing with Groq: {str(e)}")
        return ''

def post_process_transcription(transcription):
    """
    Apply post-processing to the transcription.
    """
    transcription = transcription.strip()
    post_processing = ConfigManager.get_config_section('post_processing')
    
    # Load and apply find/replace rules
    rules_file = ConfigManager.get_config_value('post_processing', 'find_replace_file')
    if rules_file:
        ConfigManager.console_print(f"Find/replace file path: {rules_file}")
        ConfigManager.console_print(f"Loading rules from: {rules_file}")
        if os.path.exists(rules_file):
            ConfigManager.console_print(f"File exists at: {rules_file}")
        else:
            ConfigManager.console_print(f"File not found at: {rules_file}")
        rules = TextProcessor.load_find_replace_rules(rules_file)
        ConfigManager.console_print(f"Loaded rules: {rules}")
        transcription = TextProcessor.apply_find_replace_rules(transcription, rules)
    
    # Apply other post-processing options
    if post_processing['remove_trailing_period'] and transcription.endswith('.'):
        transcription = transcription[:-1]
    if post_processing['add_trailing_space']:
        transcription += ' '
    if post_processing['remove_capitalization']:
        transcription = transcription.lower()

    return transcription

def transcribe(audio_data, local_model=None):
    """
    Transcribe audio using either local model or API based on availability
    """
    if HAS_FASTER_WHISPER:
        if audio_data is None:
            return ''

        if ConfigManager.get_config_value('model_options', 'use_api'):
            ConfigManager.console_print("Using OpenAI Whisper API for transcription")
            transcription = transcribe_api(audio_data)
        else:
            if not local_model:
                local_model = create_local_model()
            model_name = ConfigManager.get_config_value('model_options', 'local', 'model')
            device = ConfigManager.get_config_value('model_options', 'local', 'device')
            ConfigManager.console_print(f"Using local Whisper model: {model_name} on {device}")
            transcription = transcribe_local(audio_data, local_model)
    else:
        ConfigManager.console_print("Using OpenAI Whisper API for transcription (faster-whisper not available)")
        transcription = transcribe_api(audio_data)

    return post_process_transcription(transcription)

