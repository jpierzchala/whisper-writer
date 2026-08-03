import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np


def _azure_transcription_call(deployment, config_section, api_options_extra=None):
    """Run transcribe_with_azure_openai against a mocked client, return its kwargs."""
    if 'transcription' in sys.modules:
        del sys.modules['transcription']
    sys.path.insert(0, 'src')

    import transcription
    import vocabulary

    try:
        with patch.object(transcription, 'ConfigManager') as mock_config, \
             patch.object(vocabulary, 'ConfigManager') as mock_vocab_config, \
             patch.object(transcription, 'KeyringManager') as mock_keyring, \
             patch.object(transcription, 'build_azure_client') as mock_build_client:

            def mock_get_config_section(section):
                if section == 'model_options':
                    return config_section
                if section == 'recording_options':
                    return {'sample_rate': 16000}
                if section == 'vocabulary':
                    return {
                        'terms': 'Nearshoring\nPBIX\nPower BI',
                        'terms_file': '',
                        'use_in_transcription': True,
                        'use_in_cleanup': True,
                    }
                return {}

            mock_config.get_config_section.side_effect = mock_get_config_section
            mock_config.console_print = lambda *args, **kwargs: None
            mock_vocab_config.get_config_section.side_effect = mock_get_config_section
            mock_vocab_config.console_print = lambda *args, **kwargs: None
            mock_keyring.get_api_key.return_value = 'test-azure-key'

            result_obj = MagicMock()
            result_obj.text = 'ok'
            result_obj.languages = None
            create = mock_build_client.return_value.audio.transcriptions.create
            create.return_value = result_obj

            api_options = {
                'azure_openai_endpoint': 'https://test.openai.azure.com',
                'azure_openai_api_version': '2025-03-01-preview',
                'azure_openai_deployment_name': deployment,
                'model': 'whisper-1',
            }
            api_options.update(api_options_extra or {})

            text = transcription.transcribe_with_azure_openai(
                np.zeros(16000, dtype=np.int16), api_options
            )
            assert text == 'ok'
            return create.call_args.kwargs
    finally:
        sys.path.pop(0)


def test_azure_whisper_request_includes_language_prompt_and_temperature():
    kwargs = _azure_transcription_call('whisper', {
        'common': {
            'language': 'pl',
            'initial_prompt': 'PBIX, MyHub, Fabric',
            'temperature': 0.0,
        }
    })

    assert kwargs['model'] == 'whisper'
    assert kwargs['language'] == 'pl'
    assert kwargs['prompt'] == 'PBIX, MyHub, Fabric'
    assert kwargs['temperature'] == 0.0
    # whisper takes neither the plural language field nor keyword hints.
    assert 'languages' not in kwargs
    assert 'keywords' not in kwargs


def test_gpt_transcribe_uses_plural_languages_and_keywords():
    """gpt-transcribe ignores the singular `language` field, so it must not be sent."""
    kwargs = _azure_transcription_call('gpt-transcribe-global', {
        'common': {
            'language': 'pl',
            'languages': '',
            'initial_prompt': None,
            'temperature': None,
        }
    })

    assert kwargs['model'] == 'gpt-transcribe-global'
    assert kwargs['languages'] == ['pl']
    assert 'language' not in kwargs
    assert kwargs['keywords'] == ['Nearshoring', 'PBIX', 'Power BI']


def test_gpt_transcribe_honours_a_multi_language_list():
    kwargs = _azure_transcription_call('gpt-transcribe-global', {
        'common': {
            'language': 'pl',
            'languages': 'pl, en',
            'initial_prompt': None,
            'temperature': None,
        }
    })

    assert kwargs['languages'] == ['pl', 'en']


def test_auto_language_sends_no_language_hint_at_all():
    kwargs = _azure_transcription_call('gpt-transcribe-global', {
        'common': {
            'language': 'auto',
            'languages': '',
            'initial_prompt': None,
            'temperature': None,
        }
    })

    assert 'languages' not in kwargs
    assert 'language' not in kwargs


def test_pinned_model_family_overrides_an_unrelated_deployment_name():
    kwargs = _azure_transcription_call(
        'prod-dictation',
        {'common': {'language': 'pl', 'languages': '', 'initial_prompt': None, 'temperature': None}},
        {'azure_openai_model_family': 'gpt-transcribe'},
    )

    assert kwargs['languages'] == ['pl']
    assert kwargs['keywords'] == ['Nearshoring', 'PBIX', 'Power BI']


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