"""Tests for the model capability registry and the OpenAI/Azure client builders."""
import sys

import pytest

sys.path.insert(0, 'src')

from model_registry import (
    API_CHAT,
    API_RESPONSES,
    resolve_llm_capabilities,
    resolve_transcription_capabilities,
    is_reasoning_model,
)
from openai_clients import (
    AZURE_MODE_LEGACY,
    AZURE_MODE_V1,
    normalize_azure_v1_base_url,
    resolve_azure_api_mode,
)


@pytest.mark.parametrize("model", [
    'gpt-5.6-luna', 'gpt-5.6-terra', 'gpt-5.6-sol', 'gpt-5.4', 'gpt-5.2', 'gpt-5.1', 'o3',
])
def test_reasoning_models_use_the_responses_api_and_reject_temperature(model):
    caps = resolve_llm_capabilities(model)
    assert caps.api == API_RESPONSES
    assert caps.is_reasoning_model
    assert not caps.supports_temperature
    assert caps.max_tokens_param == 'max_output_tokens'


@pytest.mark.parametrize("model", ['gpt-4o', 'gpt-4o-mini', 'gpt-4.1', 'gpt-3.5-turbo'])
def test_classic_chat_models_take_a_temperature(model):
    caps = resolve_llm_capabilities(model)
    assert caps.api == API_CHAT
    assert not caps.is_reasoning_model
    assert caps.supports_temperature
    assert not is_reasoning_model(model)


def test_effort_max_is_only_available_on_gpt_5_6():
    assert 'max' in resolve_llm_capabilities('gpt-5.6-luna').reasoning_efforts
    for older in ('gpt-5.4', 'gpt-5.2', 'gpt-5.1'):
        assert 'max' not in resolve_llm_capabilities(older).reasoning_efforts


def test_effort_minimal_only_exists_on_the_original_gpt5_family():
    """'minimal' was replaced by 'none' from gpt-5.1 onwards and is rejected by it."""
    assert 'minimal' in resolve_llm_capabilities('gpt-5-2025-08-07').reasoning_efforts
    for newer in ('gpt-5.1', 'gpt-5.2', 'gpt-5.4', 'gpt-5.6-luna'):
        efforts = resolve_llm_capabilities(newer).reasoning_efforts
        assert 'minimal' not in efforts
        assert 'none' in efforts


def test_clamp_effort_falls_back_to_the_cheapest_supported_level():
    caps = resolve_llm_capabilities('gpt-5.2')
    assert caps.clamp_effort('low') == 'low'
    # gpt-5.2 has no xhigh/max, so latency must not regress: clamp down, not up.
    assert caps.clamp_effort('max') == 'none'
    assert caps.clamp_effort('xhigh') == 'none'
    # Non-reasoning models get nothing at all.
    assert resolve_llm_capabilities('gpt-4o').clamp_effort('low') is None


def test_azure_deployment_names_with_suffixes_resolve_to_their_model():
    """Deployments are usually named after the model with an environment suffix."""
    assert resolve_llm_capabilities('gpt-5.6-luna-global').api == API_RESPONSES
    assert resolve_llm_capabilities('gpt-5.4-global').api == API_RESPONSES
    assert resolve_llm_capabilities('gpt-4o-prod').api == API_CHAT


def test_unrelated_deployment_names_fall_back_to_plain_chat():
    """Guessing reasoning controls for an unknown model would be a hard 400."""
    caps = resolve_llm_capabilities('prod-cleanup')
    assert caps.api == API_CHAT
    assert not caps.is_reasoning_model


def test_pinned_family_overrides_the_deployment_name():
    assert resolve_llm_capabilities('prod-cleanup', family='gpt-5.6').api == API_RESPONSES
    assert resolve_llm_capabilities('gpt-5.6-luna-global', family='chat').api == API_CHAT


def test_gpt_transcribe_uses_the_plural_languages_field_and_accepts_keywords():
    caps = resolve_transcription_capabilities('gpt-transcribe')
    assert caps.language_param == 'languages'
    assert caps.supports_keywords


@pytest.mark.parametrize("model", ['whisper-1', 'gpt-4o-transcribe', 'gpt-4o-mini-transcribe'])
def test_older_transcription_models_use_the_singular_language_field(model):
    caps = resolve_transcription_capabilities(model)
    assert caps.language_param == 'language'
    assert not caps.supports_keywords


def test_azure_transcription_deployment_names_resolve_too():
    caps = resolve_transcription_capabilities('gpt-transcribe-global')
    assert caps.language_param == 'languages'
    assert caps.supports_keywords


@pytest.mark.parametrize("endpoint,expected", [
    # Classic Azure OpenAI, AI Foundry, and multi-service Cognitive Services hosts
    # all serve the same v1 API.
    ('https://res.openai.azure.com', 'https://res.openai.azure.com/openai/v1/'),
    ('https://res.services.ai.azure.com', 'https://res.services.ai.azure.com/openai/v1/'),
    ('https://res.cognitiveservices.azure.com', 'https://res.cognitiveservices.azure.com/openai/v1/'),
    ('https://res.openai.azure.com/', 'https://res.openai.azure.com/openai/v1/'),
    ('res.openai.azure.com', 'https://res.openai.azure.com/openai/v1/'),
    ('https://res.openai.azure.com/openai', 'https://res.openai.azure.com/openai/v1/'),
    ('https://res.openai.azure.com/openai/v1', 'https://res.openai.azure.com/openai/v1/'),
    ('https://res.openai.azure.com/openai/v1/', 'https://res.openai.azure.com/openai/v1/'),
])
def test_azure_endpoints_normalize_to_a_v1_base_url(endpoint, expected):
    assert normalize_azure_v1_base_url(endpoint) == expected


def test_normalizing_an_empty_endpoint_is_an_error():
    with pytest.raises(ValueError):
        normalize_azure_v1_base_url('')


def test_api_mode_defaults_to_v1_but_honours_a_dated_api_version():
    # Nothing configured at all: use the current API.
    assert resolve_azure_api_mode(None, None) == AZURE_MODE_V1
    assert resolve_azure_api_mode(None, 'v1') == AZURE_MODE_V1
    # A config carrying a dated api-version predates v1; do not switch it silently.
    assert resolve_azure_api_mode(None, '2024-02-01') == AZURE_MODE_LEGACY
    assert resolve_azure_api_mode(None, '2025-03-01-preview') == AZURE_MODE_LEGACY
    # An explicit mode always wins.
    assert resolve_azure_api_mode('v1', '2024-02-01') == AZURE_MODE_V1
    assert resolve_azure_api_mode('legacy', 'v1') == AZURE_MODE_LEGACY
