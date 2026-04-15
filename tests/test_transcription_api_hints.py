import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np


def test_azure_transcription_request_includes_language_prompt_and_temperature():
    if 'transcription' in sys.modules:
        del sys.modules['transcription']
    sys.path.insert(0, 'src')

    import transcription

    with patch.object(transcription, 'ConfigManager') as mock_config, \
         patch.object(transcription, 'KeyringManager') as mock_keyring, \
         patch.object(transcription.requests, 'post') as mock_post:

        def mock_get_config_section(section):
            if section == 'model_options':
                return {
                    'common': {
                        'language': 'pl',
                        'initial_prompt': 'PBIX, MyHub, Fabric',
                        'temperature': 0.0
                    }
                }
            if section == 'recording_options':
                return {'sample_rate': 16000}
            return {}

        mock_config.get_config_section.side_effect = mock_get_config_section
        mock_config.console_print = lambda *args, **kwargs: None
        mock_keyring.get_api_key.return_value = 'test-azure-key'

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'text': 'ok'}
        mock_post.return_value = mock_response

        result = transcription.transcribe_with_azure_openai(
            np.zeros(16000, dtype=np.int16),
            {
                'azure_openai_endpoint': 'https://test.openai.azure.com',
                'azure_openai_api_version': '2025-03-01-preview',
                'azure_openai_deployment_name': 'whisper',
                'model': 'whisper-1'
            }
        )

        assert result == 'ok'

        request_data = mock_post.call_args[1]['data']
        assert request_data['model'] == 'whisper-1'
        assert request_data['language'] == 'pl'
        assert request_data['prompt'] == 'PBIX, MyHub, Fabric'
        assert request_data['temperature'] == 0.0

    sys.path.pop(0)


def test_apply_transcription_hints_normalizes_dropdown_language_labels():
    if 'transcription' in sys.modules:
        del sys.modules['transcription']
    sys.path.insert(0, 'src')

    import transcription

    with patch.object(transcription, 'ConfigManager') as mock_config:
        def mock_get_config_section(section):
            if section == 'model_options':
                return {
                    'common': {
                        'language': 'Polish (pl)',
                        'initial_prompt': None,
                        'temperature': None
                    }
                }
            return {}

        mock_config.get_config_section.side_effect = mock_get_config_section
        mock_config.console_print = lambda *args, **kwargs: None

        request_data = transcription.apply_transcription_hints({'model': 'whisper-1'})

        assert request_data['language'] == 'pl'

    sys.path.pop(0)


def test_local_transcription_treats_auto_language_as_detection():
    if 'transcription' in sys.modules:
        del sys.modules['transcription']
    sys.path.insert(0, 'src')

    import transcription

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([SimpleNamespace(text='hello')], None)

    with patch.object(transcription, 'ConfigManager') as mock_config:
        def mock_get_config_section(section):
            if section == 'model_options':
                return {
                    'common': {
                        'language': 'Auto',
                        'initial_prompt': None,
                        'temperature': 0.0
                    },
                    'local': {
                        'condition_on_previous_text': True,
                        'vad_filter': False
                    }
                }
            return {}

        mock_config.get_config_section.side_effect = mock_get_config_section
        mock_config.console_print = lambda *args, **kwargs: None

        result = transcription.transcribe_local(np.zeros(32, dtype=np.int16), ('whisper', mock_model))

        assert result == 'hello'
        assert mock_model.transcribe.call_args.kwargs['language'] is None

    sys.path.pop(0)