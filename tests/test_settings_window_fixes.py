"""Regression tests for the settings-window defects fixed in the revamp."""
import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QCheckBox, QComboBox, QWidget

sys.path.insert(0, 'src')


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def settings_window(monkeypatch, qapp):
    from ui.settings_window import SettingsWindow, LLMProcessor

    monkeypatch.setattr(SettingsWindow, "get_available_sound_devices", lambda self: [])
    monkeypatch.setattr(LLMProcessor, "get_available_models", lambda self, api: [])

    window = SettingsWindow()
    yield window
    window.close()


def _select(combo: QComboBox, value: str):
    index = combo.findData(value)
    if index == -1:
        index = combo.findText(value)
    assert index != -1, f"{value} not found in {combo.objectName()}"
    combo.setCurrentIndex(index)


PROVIDER_FIELDS = {
    'openai': ['openai_transcription_api_key', 'base_url'],
    'azure_openai': [
        'azure_openai_api_key',
        'azure_openai_endpoint',
        'azure_openai_deployment_name',
        'azure_openai_api_version',
    ],
    'deepgram': ['deepgram_transcription_api_key'],
    'groq': ['groq_transcription_api_key'],
}


@pytest.mark.parametrize("provider", sorted(PROVIDER_FIELDS))
def test_transcription_provider_fields_become_visible_again(settings_window, qapp, provider):
    """Selecting a transcription provider must show exactly that provider's fields.

    Previously toggle_transcription_provider_options was truncated before its
    show-loop, so every provider field stayed hidden forever.
    """
    use_api = settings_window.findChild(QCheckBox, 'model_options_use_api_input')
    use_api.setChecked(True)
    settings_window.toggle_api_local_options(True)

    settings_window.toggle_transcription_provider_options(provider)
    qapp.processEvents()

    for field in PROVIDER_FIELDS[provider]:
        widget = settings_window.findChild(QWidget, f'model_options_api_{field}_input')
        assert widget is not None and not widget.isHidden(), f"{field} should be shown for {provider}"

    for other, fields in PROVIDER_FIELDS.items():
        if other == provider:
            continue
        for field in fields:
            widget = settings_window.findChild(QWidget, f'model_options_api_{field}_input')
            assert widget is not None and widget.isHidden(), f"{field} should be hidden for {provider}"


def test_provider_combo_is_wired_to_the_visibility_toggle(settings_window, qapp):
    """Changing the provider combo must re-apply field visibility."""
    use_api = settings_window.findChild(QCheckBox, 'model_options_use_api_input')
    use_api.setChecked(True)
    settings_window.toggle_api_local_options(True)

    provider_combo = settings_window.findChild(QComboBox, 'model_options_api_provider_input')
    assert provider_combo is not None

    target = next(p for p in PROVIDER_FIELDS if p != provider_combo.currentText())
    _select(provider_combo, target)
    qapp.processEvents()

    for field in PROVIDER_FIELDS[target]:
        widget = settings_window.findChild(QWidget, f'model_options_api_{field}_input')
        assert widget is not None and not widget.isHidden(), (
            f"selecting {target} in the combo should show {field}"
        )


def test_reset_to_saved_settings_does_not_raise(settings_window):
    """The reset button used to raise TypeError because iterate_settings needs a callback."""
    settings_window.reset_settings()


def test_iter_setting_widgets_yields_tuples(settings_window):
    rows = list(settings_window.iter_setting_widgets())
    assert rows, "expected at least one settings row"
    for widget, category, sub_category, key, meta in rows:
        assert widget is not None
        assert isinstance(category, str) and isinstance(key, str)
        assert isinstance(meta, dict)


@pytest.mark.parametrize("stored", [False, 0, 0.0, ''])
def test_falsy_config_values_are_not_replaced_by_schema_defaults(settings_window, monkeypatch, stored):
    """`False`/0/'' must round-trip instead of falling back to the schema default."""
    import ui.settings_window as settings_module

    monkeypatch.setattr(
        settings_module.ConfigManager, 'get_config_value',
        classmethod(lambda cls, *keys: stored)
    )
    meta = {'value': 'schema-default', 'type': 'str'}

    assert settings_window.get_config_value('misc', None, 'anything', meta) == stored
    assert settings_window.get_config_value('model_options', 'api', 'anything', meta) == stored


def test_missing_config_values_still_fall_back_to_schema_default(settings_window, monkeypatch):
    import ui.settings_window as settings_module

    monkeypatch.setattr(
        settings_module.ConfigManager, 'get_config_value',
        classmethod(lambda cls, *keys: None)
    )
    meta = {'value': 'schema-default', 'type': 'str'}

    assert settings_window.get_config_value('misc', None, 'anything', meta) == 'schema-default'


def test_browse_system_message_file_loads_file_contents(settings_window, monkeypatch, tmp_path):
    from PyQt5.QtWidgets import QFileDialog, QLineEdit, QTextEdit

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Nowy system prompt z pliku", encoding='utf-8')

    monkeypatch.setattr(
        QFileDialog, "getOpenFileName",
        staticmethod(lambda *args, **kwargs: (str(prompt_file), ''))
    )

    file_edit = QLineEdit()
    text_edit = QTextEdit()
    text_edit.setText("stara treść")

    settings_window.browse_system_message_file(file_edit, text_edit)

    assert file_edit.text() == str(prompt_file)
    assert text_edit.toPlainText() == "Nowy system prompt z pliku"


def _force_api_type(window, api_type):
    from ui.settings_window import LLMProcessor

    if window.llm_processor is None:
        window.llm_processor = LLMProcessor(api_type=api_type)
    window.llm_processor.api_type = api_type


def test_model_refresh_never_offers_error_text_as_a_model(settings_window):
    """Error states used to be inserted as selectable combo items."""
    _force_api_type(settings_window, 'ollama')
    settings_window._on_models_fetched([])

    for combo in settings_window._model_combos():
        items = [combo.itemText(i) for i in range(combo.count())]
        assert not any('No models' in item or 'Could not fetch' in item for item in items), items
        assert combo.isEnabled()


def test_model_refresh_falls_back_to_known_model_ids_for_openai(settings_window):
    from ui.settings_window import SettingsWindow

    _force_api_type(settings_window, 'openai')
    settings_window._on_models_fetched([])

    combo = settings_window.cleanup_model_combo
    items = [combo.itemText(i) for i in range(combo.count())]
    for model in SettingsWindow._default_llm_model_choices():
        assert model in items


def test_keyring_service_map_covers_every_api_key_setting():
    """Every *_api_key setting in the schema must map to a keyring entry."""
    import yaml

    from ui.settings_window import KEYRING_SERVICE_BY_CONFIG_KEY

    # Read the schema straight from disk: other tests in the suite replace the
    # ConfigManager module with a mock, which would make this assertion vacuous.
    with open(os.path.join('src', 'config_schema.yaml'), encoding='utf-8') as handle:
        schema = yaml.safe_load(handle)

    declared = set()
    for category, settings in schema.items():
        for sub_category, sub_settings in settings.items():
            if isinstance(sub_settings, dict) and 'value' in sub_settings:
                candidates = [(sub_category, sub_settings)]
            else:
                candidates = list(sub_settings.items())
            for key, _meta in candidates:
                if key.endswith('api_key'):
                    declared.add((category, key))

    assert declared, "schema should declare api-key settings"
    assert declared == set(KEYRING_SERVICE_BY_CONFIG_KEY), (
        f"missing: {declared - set(KEYRING_SERVICE_BY_CONFIG_KEY)}, "
        f"stale: {set(KEYRING_SERVICE_BY_CONFIG_KEY) - declared}"
    )
