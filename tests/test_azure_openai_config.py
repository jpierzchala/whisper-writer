import sys
import types
import pytest
from unittest.mock import MagicMock, patch

def test_azure_openai_provider_configuration():
    """Test that Azure OpenAI provider is properly configured in transcription module."""
    
    # Mock dependencies
    sys.modules['sounddevice'] = types.SimpleNamespace(InputStream=None)
    sys.modules['webrtcvad'] = types.SimpleNamespace(Vad=lambda mode: None)
    sys.modules['soundfile'] = MagicMock()
    
    class MockConfigManager:
        @staticmethod
        def console_print(msg, verbose=False):
            print(f"[TEST LOG] {msg}")
            
        @staticmethod 
        def get_config_section(section):
            if section == 'model_options':
                return {
                    'api': {
                        'provider': 'azure_openai',
                        'model': 'whisper-1',
                        'azure_openai_endpoint': 'https://test.openai.azure.com',
                        'azure_openai_deployment_name': 'whisper-deployment',
                        'azure_openai_api_version': '2024-02-01'
                    }
                }
            return {}
    
    class MockKeyringManager:
        @staticmethod
        def get_api_key(service_name):
            if service_name == "azure_openai_transcription":
                return "test-api-key"
            return ""
    
    sys.modules['utils'] = types.SimpleNamespace(ConfigManager=MockConfigManager)
    sys.modules['keyring_manager'] = types.SimpleNamespace(KeyringManager=MockKeyringManager)

    # Import after mocking
    sys.path.insert(0, 'src')
    if 'transcription' in sys.modules:
        del sys.modules['transcription']
    import transcription
    from transcription import transcribe_api

    with patch.object(transcription, 'build_azure_client') as mock_build_client, \
         patch.object(transcription, 'encode_wav', return_value=b'RIFFfake'):
        result_obj = MagicMock()
        result_obj.text = 'Test transcription'
        result_obj.languages = None
        mock_build_client.return_value.audio.transcriptions.create.return_value = result_obj

        import numpy as np
        audio_data = np.array([0.1, 0.2, 0.1], dtype=np.float32)

        result = transcribe_api(audio_data)

        assert result == 'Test transcription'

        # The client is built for the configured resource, in legacy mode because a
        # dated api-version is set (Azure has no v1 audio endpoint).
        mock_build_client.assert_called_once()
        args, kwargs = mock_build_client.call_args
        assert args[0] == 'test-api-key'
        assert args[1] == 'https://test.openai.azure.com'
        assert kwargs['api_mode'] == 'legacy'
        assert kwargs['api_version'] == '2024-02-01'

        # The deployment name is what goes on the wire as the model.
        create_kwargs = mock_build_client.return_value.audio.transcriptions.create.call_args.kwargs
        assert create_kwargs['model'] == 'whisper-deployment'

def test_azure_openai_missing_credentials():
    """Test Azure OpenAI provider handles missing credentials gracefully."""
    
    class MockConfigManager:
        @staticmethod
        def console_print(msg, verbose=False):
            print(f"[TEST LOG] {msg}")
            
        @staticmethod 
        def get_config_section(section):
            if section == 'model_options':
                return {
                    'api': {
                        'provider': 'azure_openai',
                        'model': 'whisper-1',
                    }
                }
            return {}
    
    class MockKeyringManager:
        @staticmethod
        def get_api_key(service_name):
            return ""  # No API key
    
    sys.modules['utils'] = types.SimpleNamespace(ConfigManager=MockConfigManager)
    sys.modules['keyring_manager'] = types.SimpleNamespace(KeyringManager=MockKeyringManager)
    
    # Import after mocking
    sys.path.insert(0, 'src')
    if 'transcription' in sys.modules:
        del sys.modules['transcription']
    from transcription import transcribe_api
    
    # Test with dummy audio data
    import numpy as np
    audio_data = np.array([0.1, 0.2, 0.1], dtype=np.float32)
    
    result = transcribe_api(audio_data)
    
    # Should return empty string when credentials are missing
    assert result == ''