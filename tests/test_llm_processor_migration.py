import sys

import pytest


@pytest.fixture
def patched_dependencies(monkeypatch):
    sys.path.insert(0, 'src')
    from llm_processor import ConfigManager, KeyringManager

    monkeypatch.setattr(ConfigManager, 'get_config_section', lambda section: {
        'api_type': 'chatgpt',
        'enabled': True,
        'temperature': 0.1,
        'endpoint': 'https://example.com'
    })
    monkeypatch.setattr(ConfigManager, 'console_print', lambda *args, **kwargs: None)
    monkeypatch.setattr(ConfigManager, 'get_schema', lambda: {'llm_post_processing': {}})
    monkeypatch.setattr(ConfigManager, 'set_config_value', lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(KeyringManager, 'get_api_key', lambda *_: 'dummy-key')
    return None


def test_chatgpt_api_type_is_migrated_to_openai(patched_dependencies):
    from llm_processor import LLMProcessor

    processor = LLMProcessor(api_type='chatgpt')
    assert processor.api_type == 'openai'

