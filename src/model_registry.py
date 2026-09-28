"""What the models this app talks to actually support.

Every question of the form "does this model take a temperature?", "which API
surface does it use?" or "does it accept a keyword list?" is answered here, so
the answer cannot drift between the LLM processor, the transcription layer and
the settings UI.

Lookups are longest-prefix matches on the model ID, which also resolves Azure
deployment names that carry a suffix (``gpt-6-luna-dzs`` -> ``gpt-6-luna``).
Deployments named something unrelated to their model fall back to a conservative
default; pin them explicitly with the ``*_model_family`` setting instead.
"""
from dataclasses import dataclass

# Reasoning effort levels, cheapest/fastest first. 'none' skips thinking entirely
# and is the documented latency baseline; 'minimal' only ever existed on the
# original gpt-5 family and is rejected by gpt-5.1 and later.
EFFORT_NONE = 'none'
EFFORT_MINIMAL = 'minimal'
EFFORT_LOW = 'low'
EFFORT_MEDIUM = 'medium'
EFFORT_HIGH = 'high'
EFFORT_XHIGH = 'xhigh'
EFFORT_MAX = 'max'

# Cheapest to most expensive, used to clamp a requested effort to what a model takes.
EFFORT_ORDER = (
    EFFORT_NONE, EFFORT_MINIMAL, EFFORT_LOW, EFFORT_MEDIUM,
    EFFORT_HIGH, EFFORT_XHIGH, EFFORT_MAX,
)

API_RESPONSES = 'responses'
API_CHAT = 'chat'


@dataclass(frozen=True)
class LLMCapabilities:
    """What an OpenAI-shaped chat/reasoning model supports."""

    api: str = API_CHAT
    # Empty when the model has no reasoning control at all.
    reasoning_efforts: tuple = ()
    supports_temperature: bool = True
    supports_structured_outputs: bool = True
    supports_verbosity: bool = False
    # Chat Completions renamed max_tokens to max_completion_tokens for reasoning models.
    max_tokens_param: str = 'max_completion_tokens'

    @property
    def is_reasoning_model(self) -> bool:
        return bool(self.reasoning_efforts)

    def clamp_effort(self, effort):
        """Return `effort`, or the nearest supported level at or below it.

        Never above what was asked, so latency cannot regress past the request;
        but nearest rather than cheapest, because asking for 'max' on a model
        without it means "think hard", and answering with 'none' would inverse
        the request.
        """
        if not self.reasoning_efforts:
            return None
        if effort in self.reasoning_efforts:
            return effort

        supported = sorted(self.reasoning_efforts, key=EFFORT_ORDER.index)
        if effort not in EFFORT_ORDER:
            return supported[0]

        target = EFFORT_ORDER.index(effort)
        at_or_below = [level for level in supported if EFFORT_ORDER.index(level) < target]
        # Nothing lower exists (e.g. 'none' on a model whose minimum is 'low').
        return at_or_below[-1] if at_or_below else supported[0]


# Reasoning-model families share everything except which efforts they accept.
def _reasoning(efforts, **overrides):
    return LLMCapabilities(
        api=API_RESPONSES,
        reasoning_efforts=efforts,
        supports_temperature=False,
        supports_structured_outputs=True,
        supports_verbosity=True,
        max_tokens_param='max_output_tokens',
        **overrides
    )


_GPT6_EFFORTS = (EFFORT_NONE, EFFORT_LOW, EFFORT_MEDIUM, EFFORT_HIGH, EFFORT_XHIGH, EFFORT_MAX)
_GPT56_EFFORTS = (EFFORT_NONE, EFFORT_LOW, EFFORT_MEDIUM, EFFORT_HIGH, EFFORT_XHIGH, EFFORT_MAX)
_GPT54_EFFORTS = (EFFORT_NONE, EFFORT_LOW, EFFORT_MEDIUM, EFFORT_HIGH, EFFORT_XHIGH)
_GPT51_EFFORTS = (EFFORT_NONE, EFFORT_LOW, EFFORT_MEDIUM, EFFORT_HIGH)
_GPT5_EFFORTS = (EFFORT_MINIMAL, EFFORT_LOW, EFFORT_MEDIUM, EFFORT_HIGH)

_CHAT_MODEL = LLMCapabilities(
    api=API_CHAT,
    reasoning_efforts=(),
    supports_temperature=True,
    supports_structured_outputs=True,
    max_tokens_param='max_tokens',
)

# Longest matching prefix wins, so more specific IDs may appear in any order.
LLM_CAPABILITIES = {
    'gpt-6-sol': _reasoning(_GPT6_EFFORTS),
    'gpt-6-luna': _reasoning(_GPT6_EFFORTS),
    'gpt-5.6': _reasoning(_GPT56_EFFORTS),
    'gpt-5.5': _reasoning(_GPT54_EFFORTS),
    'gpt-5.4': _reasoning(_GPT54_EFFORTS),
    'gpt-5.2': _reasoning(_GPT51_EFFORTS),
    'gpt-5.1': _reasoning(_GPT51_EFFORTS),
    'gpt-5': _reasoning(_GPT5_EFFORTS),
    'o1': _reasoning((EFFORT_LOW, EFFORT_MEDIUM, EFFORT_HIGH)),
    'o3': _reasoning((EFFORT_LOW, EFFORT_MEDIUM, EFFORT_HIGH)),
    'o4-mini': _reasoning((EFFORT_LOW, EFFORT_MEDIUM, EFFORT_HIGH)),
    'gpt-4o': _CHAT_MODEL,
    'gpt-4.1': _CHAT_MODEL,
    'gpt-4-turbo': _CHAT_MODEL,
    'gpt-4': _CHAT_MODEL,
    'gpt-3.5': _CHAT_MODEL,
}

# Explicit family values the user can pin on an Azure deployment whose name does
# not identify its model. 'auto' means "infer from the deployment name".
AZURE_MODEL_FAMILY_AUTO = 'auto'
AZURE_LLM_FAMILIES = (AZURE_MODEL_FAMILY_AUTO, 'gpt-6', 'gpt-5.6', 'gpt-5.4', 'gpt-5.2', 'chat')

_AZURE_FAMILY_CAPABILITIES = {
    'gpt-6': _reasoning(_GPT6_EFFORTS),
    'chat': _CHAT_MODEL,
}


@dataclass(frozen=True)
class TranscriptionCapabilities:
    """What a speech-to-text model accepts on /audio/transcriptions."""

    # 'languages' models take a list and reject the singular field (and vice versa).
    language_param: str = 'language'
    supports_keywords: bool = False
    supports_prompt: bool = True
    supports_temperature: bool = True
    supports_timestamps: bool = False
    # Hard API limit shared by OpenAI and Azure.
    max_file_bytes: int = 25 * 1024 * 1024


TRANSCRIPTION_CAPABILITIES = {
    # OpenAI's recommended file-transcription model: plural `languages`, plus a
    # literal keyword list that biases recognition toward domain terms.
    'gpt-transcribe': TranscriptionCapabilities(
        language_param='languages',
        supports_keywords=True,
    ),
    'gpt-live-transcribe': TranscriptionCapabilities(
        language_param='languages',
        supports_keywords=True,
    ),
    'gpt-4o-transcribe': TranscriptionCapabilities(language_param='language'),
    'gpt-4o-mini-transcribe': TranscriptionCapabilities(language_param='language'),
    'whisper-1': TranscriptionCapabilities(language_param='language', supports_timestamps=True),
    'whisper': TranscriptionCapabilities(language_param='language', supports_timestamps=True),
}

_DEFAULT_TRANSCRIPTION = TranscriptionCapabilities()


def _longest_prefix_match(model, table):
    """Return the value whose key is the longest prefix of `model`."""
    if not model:
        return None
    lowered = str(model).strip().lower()
    best_key = None
    for key in table:
        if lowered.startswith(key) and (best_key is None or len(key) > len(best_key)):
            best_key = key
    return table[best_key] if best_key else None


def resolve_llm_capabilities(model, family=AZURE_MODEL_FAMILY_AUTO):
    """Return the capabilities of `model`.

    `family` pins the answer for Azure deployments whose name does not identify
    the underlying model; 'auto' (the default) infers it from the name.
    """
    if family and family != AZURE_MODEL_FAMILY_AUTO:
        pinned = _AZURE_FAMILY_CAPABILITIES.get(family) or LLM_CAPABILITIES.get(family)
        if pinned:
            return pinned

    matched = _longest_prefix_match(model, LLM_CAPABILITIES)
    if matched:
        return matched

    # Unknown model: assume a plain chat model. Sending reasoning controls to a
    # model that does not take them is a hard 400, whereas omitting them is not.
    return _CHAT_MODEL


def resolve_transcription_capabilities(model, family=AZURE_MODEL_FAMILY_AUTO):
    """Return the transcription capabilities of `model` (or of a pinned `family`)."""
    if family and family != AZURE_MODEL_FAMILY_AUTO:
        pinned = TRANSCRIPTION_CAPABILITIES.get(family)
        if pinned:
            return pinned

    return _longest_prefix_match(model, TRANSCRIPTION_CAPABILITIES) or _DEFAULT_TRANSCRIPTION


def is_reasoning_model(model, family=AZURE_MODEL_FAMILY_AUTO) -> bool:
    """True when the model takes reasoning controls and no temperature."""
    return resolve_llm_capabilities(model, family).is_reasoning_model
