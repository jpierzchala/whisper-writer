"""Tests for the shared domain-vocabulary list."""
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, 'src')

import vocabulary


@pytest.fixture
def vocab_config():
    """Patch the config section vocabulary reads, returning a mutable dict."""
    section = {
        'terms': '',
        'terms_file': '',
        'use_in_transcription': True,
        'use_in_cleanup': True,
    }
    with patch.object(vocabulary, 'ConfigManager') as mock_config:
        mock_config.get_config_section.side_effect = (
            lambda name: section if name == 'vocabulary' else {}
        )
        mock_config.console_print = lambda *args, **kwargs: None
        yield section


def test_inline_terms_are_parsed_in_order(vocab_config):
    vocab_config['terms'] = "Nearshoring\nPower BI\nPBIX"
    assert vocabulary.get_vocabulary_terms() == ['Nearshoring', 'Power BI', 'PBIX']


def test_comments_and_blank_lines_are_ignored(vocab_config):
    vocab_config['terms'] = "# projekty\nNearshoring\n\n   \n# BI\nPower BI"
    assert vocabulary.get_vocabulary_terms() == ['Nearshoring', 'Power BI']


def test_duplicates_are_dropped_case_insensitively(vocab_config):
    vocab_config['terms'] = "PBIX\npbix\nPBIX\nFabric"
    assert vocabulary.get_vocabulary_terms() == ['PBIX', 'Fabric']


@pytest.mark.parametrize("bad_term", ['a<b', 'a>b'])
def test_terms_with_forbidden_characters_are_skipped(vocab_config, bad_term):
    """The API rejects the whole request over one bad keyword, so drop it instead."""
    vocab_config['terms'] = f"Nearshoring\n{bad_term}\nPBIX"
    assert vocabulary.get_vocabulary_terms() == ['Nearshoring', 'PBIX']


def test_terms_from_a_file_are_merged_with_inline_terms(vocab_config, tmp_path):
    terms_file = tmp_path / "vocabulary.txt"
    terms_file.write_text("# z pliku\nFabric\nNeeds\nPBIX\n", encoding='utf-8')

    vocab_config['terms'] = "Nearshoring\nPBIX"
    vocab_config['terms_file'] = str(terms_file)

    # Inline terms come first, the file adds what is new, duplicates collapse.
    assert vocabulary.get_vocabulary_terms() == ['Nearshoring', 'PBIX', 'Fabric', 'Needs']


def test_a_missing_file_is_reported_but_not_fatal(vocab_config, tmp_path):
    vocab_config['terms'] = "Nearshoring"
    vocab_config['terms_file'] = str(tmp_path / "nope.txt")

    assert vocabulary.get_vocabulary_terms() == ['Nearshoring']


def test_transcription_keywords_can_be_switched_off(vocab_config):
    vocab_config['terms'] = "Nearshoring\nPBIX"

    assert vocabulary.get_transcription_keywords() == ['Nearshoring', 'PBIX']

    vocab_config['use_in_transcription'] = False
    assert vocabulary.get_transcription_keywords() == []
    # Turning it off for transcription must not affect the cleanup glossary.
    assert 'Nearshoring' in vocabulary.build_glossary_section()


def test_glossary_lists_the_terms_and_can_be_switched_off(vocab_config):
    vocab_config['terms'] = "Nearshoring\nPower BI"

    glossary = vocabulary.build_glossary_section()
    assert 'DOMAIN GLOSSARY' in glossary
    assert '- Nearshoring' in glossary
    assert '- Power BI' in glossary

    vocab_config['use_in_cleanup'] = False
    assert vocabulary.build_glossary_section() == ''


def test_glossary_is_empty_when_no_terms_are_configured(vocab_config):
    assert vocabulary.build_glossary_section() == ''
    assert vocabulary.append_glossary('Base prompt') == 'Base prompt'


def test_append_glossary_keeps_the_base_prompt_first(vocab_config):
    vocab_config['terms'] = "PBIX"

    combined = vocabulary.append_glossary('Base prompt')
    assert combined.startswith('Base prompt\n\n')
    assert '- PBIX' in combined

    # With no base prompt the glossary stands alone rather than starting with blanks.
    assert vocabulary.append_glossary('').startswith('DOMAIN GLOSSARY')
