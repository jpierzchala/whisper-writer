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


def test_legacy_chatgpt_api_type_still_resolves_to_openai(patched_dependencies):
    """A config predating the rename must keep working rather than silently no-op."""
    from llm_processor import LLMProcessor

    processor = LLMProcessor(api_type='chatgpt')
    assert processor.api_type == 'openai'


def test_legacy_api_type_from_config_is_also_aliased(patched_dependencies):
    from llm_processor import LLMProcessor

    processor = LLMProcessor()
    assert processor.api_type == 'openai'


def test_a_legacy_api_type_still_reaches_the_provider(patched_dependencies, monkeypatch):
    """The alias has to survive into process_text, which used to re-read the config.

    Asserting only on processor.api_type let a regression through where the text
    passed straight back out without any provider being called.
    """
    from llm_processor import ConfigManager, LLMProcessor

    monkeypatch.setattr(
        ConfigManager, 'get_config_value',
        lambda *keys: 'gpt-5.6-luna' if keys[-1].endswith('_model') else None
    )

    called = {}

    def fake_openai(self, text, system_message, model, mode):
        called['model'] = model
        return 'CLEANED'

    monkeypatch.setattr(LLMProcessor, '_process_openai', fake_openai)

    processor = LLMProcessor(api_type='chatgpt')
    result = processor.process_text('surowy transkrypt', 'system msg', mode='cleanup')

    assert called, "the openai provider should have been called for the 'chatgpt' alias"
    assert result == 'CLEANED'


def test_an_unknown_provider_is_reported_rather_than_silently_skipped(patched_dependencies, monkeypatch):
    from llm_processor import ConfigManager, LLMProcessor

    monkeypatch.setattr(ConfigManager, 'get_config_value', lambda *keys: None)
    messages = []
    monkeypatch.setattr(
        ConfigManager, 'console_print',
        lambda message, *args, **kwargs: messages.append(str(message))
    )

    processor = LLMProcessor(api_type='no-such-provider')
    result = processor.process_text('surowy transkrypt', 'system msg', mode='cleanup')

    assert result == 'surowy transkrypt'
    assert any('Unknown LLM provider' in message for message in messages), messages

