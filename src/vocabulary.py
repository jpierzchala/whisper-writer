"""The user's domain vocabulary: one list of terms, used in two places.

Transcription sends the terms as `keywords`, which biases recognition toward
them. LLM cleanup receives them as a glossary section appended to the system
prompt, which normalises whatever still came out wrong.

Keeping the list in its own file (rather than inline in the cleanup prompt)
means the prompt stays about *behaviour* while the terms stay editable.
"""
import os

from utils import ConfigManager

# The API rejects the entire request if a keyword contains one of these, so terms
# carrying them are dropped rather than allowed to fail the whole transcription.
FORBIDDEN_KEYWORD_CHARS = ('<', '>', '\r', '\n')

GLOSSARY_HEADER = (
    "DOMAIN GLOSSARY\n"
    "Keep these terms exactly as written when they appear, including when they "
    "carry a Polish inflection (attach the suffix with a hyphen, e.g. \"PBIX-y\"). "
    "Correct near-misses and phonetic misspellings to the canonical form:"
)


def _read_vocabulary_file(path):
    if not path:
        return []
    if not os.path.exists(path):
        ConfigManager.console_print(f"Vocabulary file not found: {path}")
        return []
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            return handle.read().splitlines()
    except Exception as exc:
        ConfigManager.console_print(f"Error reading vocabulary file {path}: {exc}")
        return []


def _clean_terms(raw_lines):
    """Normalise raw lines into usable terms, preserving the author's order."""
    terms = []
    seen = set()
    for line in raw_lines:
        term = (line or '').strip()
        if not term or term.startswith('#'):
            continue
        rejected = [char for char in FORBIDDEN_KEYWORD_CHARS if char in term]
        if rejected:
            ConfigManager.console_print(
                f"Skipping vocabulary term with unsupported character(s): {term!r}"
            )
            continue
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        terms.append(term)
    return terms


def get_vocabulary_terms():
    """Return the configured domain terms, merging the inline list and the file."""
    section = ConfigManager.get_config_section('vocabulary') or {}

    inline = section.get('terms') or ''
    path = section.get('terms_file') or ''

    raw_lines = list(str(inline).splitlines())
    raw_lines.extend(_read_vocabulary_file(path))

    return _clean_terms(raw_lines)


def get_transcription_keywords():
    """Terms to pass as `keywords` to a transcription model that supports them."""
    section = ConfigManager.get_config_section('vocabulary') or {}
    if not section.get('use_in_transcription', True):
        return []
    return get_vocabulary_terms()


def build_glossary_section():
    """Glossary text to append to the cleanup prompt, or '' when disabled/empty."""
    section = ConfigManager.get_config_section('vocabulary') or {}
    if not section.get('use_in_cleanup', True):
        return ''

    terms = get_vocabulary_terms()
    if not terms:
        return ''

    listed = "\n".join(f"- {term}" for term in terms)
    return f"{GLOSSARY_HEADER}\n{listed}"


def append_glossary(system_message):
    """Append the glossary section to a cleanup system message."""
    glossary = build_glossary_section()
    if not glossary:
        return system_message
    if not system_message:
        return glossary
    return f"{system_message}\n\n{glossary}"
