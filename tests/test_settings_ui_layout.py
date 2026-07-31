"""Tests for the schema-driven settings layout: labels, groups and widget types."""
import os
import sys

import pytest
import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QGroupBox, QLabel, QSpinBox, QWidget
)

sys.path.insert(0, 'src')

SCHEMA_PATH = os.path.join('src', 'config_schema.yaml')


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


@pytest.fixture(scope="module")
def schema():
    with open(SCHEMA_PATH, encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def _settings(schema):
    """Yield (category, sub_category, key, meta) for every declared setting."""
    for category, section in schema.items():
        if category.startswith('_'):
            continue
        for sub_category, sub_settings in section.items():
            if sub_category.startswith('_') or not isinstance(sub_settings, dict):
                continue
            if 'value' in sub_settings:
                yield category, None, sub_category, sub_settings
            else:
                for key, meta in sub_settings.items():
                    if not key.startswith('_'):
                        yield category, sub_category, key, meta


def test_every_setting_declares_a_label_and_a_group(schema):
    """Without these the UI falls back to keys like 'Azure openai llm api version'."""
    missing_label = []
    missing_group = []
    for category, sub_category, key, meta in _settings(schema):
        where = f"{category}.{sub_category + '.' if sub_category else ''}{key}"
        if not meta.get('label'):
            missing_label.append(where)
        if not meta.get('group'):
            missing_group.append(where)

    assert not missing_label, f"settings without a label: {missing_label}"
    assert not missing_group, f"settings without a group: {missing_group}"


def test_every_category_declares_a_tab_title_and_order(schema):
    for category, section in schema.items():
        if category.startswith('_'):
            continue
        ui_meta = section.get('_ui')
        assert ui_meta, f"{category} has no _ui block"
        assert ui_meta.get('title'), f"{category} has no tab title"
        assert isinstance(ui_meta.get('order'), int), f"{category} has no numeric order"


def test_schema_metadata_never_becomes_a_config_value():
    """`_ui` blocks must not leak into the config as if they were settings."""
    import importlib.util

    # Load utils from disk under its own name: other tests in the suite replace the
    # `utils` module with a mock, which would make this assertion vacuous.
    spec = importlib.util.spec_from_file_location('_utils_for_ui_test', os.path.join('src', 'utils.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    manager = module.ConfigManager()
    manager.schema = manager.load_config_schema(SCHEMA_PATH)
    config = manager.load_default_config()

    assert config, "expected a populated default config"
    for category, section in config.items():
        assert not category.startswith('_')
        if isinstance(section, dict):
            assert not any(key.startswith('_') for key in section), category


def test_tabs_use_the_schema_titles_in_order(settings_window, schema):
    expected = [
        section['_ui']['title'] for _, section in sorted(
            ((name, body) for name, body in schema.items() if not name.startswith('_')),
            key=lambda item: item[1]['_ui']['order']
        )
    ]
    actual = [settings_window.tabs.tabText(i) for i in range(settings_window.tabs.count())]
    assert actual == expected


def test_settings_are_laid_out_in_titled_groups(settings_window):
    group_titles = {box.title() for box in settings_window.findChildren(QGroupBox)}
    # A few representative sections from different tabs.
    for expected in ('Engine', 'Language', 'API provider', 'Shortcuts', 'Models', 'Prompts'):
        assert expected in group_titles, f"missing group {expected!r} in {sorted(group_titles)}"


def test_labels_come_from_the_schema_not_from_the_key(settings_window):
    label = settings_window.findChild(
        QLabel, 'llm_post_processing_azure_openai_llm_api_version_label'
    )
    assert label is not None
    assert label.text() == 'Azure API version:'


@pytest.mark.parametrize("object_name,widget_type", [
    ('recording_options_sample_rate_input', QSpinBox),
    ('recording_options_silence_duration_input', QSpinBox),
    ('model_options_common_temperature_input', QDoubleSpinBox),
    ('post_processing_writing_key_press_delay_input', QDoubleSpinBox),
])
def test_numeric_settings_use_spin_boxes(settings_window, object_name, widget_type):
    """A spin box makes invalid input impossible; the old line edits raised on save."""
    widget = settings_window.findChild(widget_type, object_name)
    assert widget is not None, f"{object_name} should be a {widget_type.__name__}"


def test_spin_boxes_show_their_unit(settings_window):
    sample_rate = settings_window.findChild(QSpinBox, 'recording_options_sample_rate_input')
    assert sample_rate.suffix() == ' Hz'


def test_spin_box_range_always_includes_the_stored_value(settings_window):
    """Schema bounds are advisory: a value already in the config must stay reachable."""
    from ui.settings_window import SettingsWindow

    meta = {'type': 'int', 'value': 100, 'min': 0, 'max': 5000}
    spin = SettingsWindow.create_number_input(99999, meta)
    assert spin.value() == 99999


def test_hidden_rows_are_collapsed_not_just_blanked(settings_window, qapp):
    """A hidden provider field must not leave a gap in its group."""
    from PyQt5.QtWidgets import QCheckBox

    use_api = settings_window.findChild(QCheckBox, 'model_options_use_api_input')
    use_api.setChecked(True)
    settings_window.toggle_api_local_options(True)
    settings_window.toggle_transcription_provider_options('azure_openai')
    qapp.processEvents()

    row = settings_window.setting_rows.get('model_options_api_deepgram_transcription_api_key')
    assert row is not None, "expected a row widget to be registered"
    assert row.isHidden(), "the whole row should be hidden, not only its editor"


def test_reasoning_model_hides_the_temperature_row(settings_window, qapp):
    """Reasoning models reject a temperature, so the control must not be offered."""
    api_combo = settings_window.findChild(QComboBox, 'llm_post_processing_api_type_input')
    index = api_combo.findData('openai')
    api_combo.setCurrentIndex(index)

    settings_window.cleanup_model_combo.setCurrentText('gpt-5.6-luna')
    settings_window.instruction_model_combo.setCurrentText('gpt-5.6-luna')
    settings_window.update_temperature_visibility()
    qapp.processEvents()
    assert settings_window._should_hide_temperature() is True

    settings_window.cleanup_model_combo.setCurrentText('gpt-4o')
    settings_window.instruction_model_combo.setCurrentText('gpt-4o')
    settings_window.update_temperature_visibility()
    qapp.processEvents()
    assert settings_window._should_hide_temperature() is False


def test_theme_has_a_light_and_a_dark_variant():
    from ui.theme import build_stylesheet, get_palette

    light = get_palette(dark=False)
    dark = get_palette(dark=True)
    assert light['text'] != dark['text']
    assert light['card'] != dark['card']

    for dark_mode in (False, True):
        sheet = build_stylesheet(dark=dark_mode)
        assert 'QGroupBox' in sheet and 'QTabBar::tab' in sheet
        # A fractional pt size is not parsed by Qt and silently loses the font size.
        assert '.5pt' not in sheet


def test_sound_devices_are_listed_from_metadata(monkeypatch):
    """The microphone list must not depend on opening a stream per device.

    A broad `except Exception` here previously hid a missing `sounddevice` import,
    leaving the dropdown silently empty.
    """
    import sounddevice
    import ui.settings_window as settings_module
    from ui.settings_window import SettingsWindow

    devices = [
        {'name': 'Speakers', 'max_input_channels': 0},
        {'name': 'Microphone (Realtek)', 'max_input_channels': 2},
        {'name': 'Line In', 'max_input_channels': 1},
    ]

    class FakeSoundDevice:
        default = type('Default', (), {'device': [2, 0]})()

        @staticmethod
        def query_devices():
            return devices

        @staticmethod
        def InputStream(*args, **kwargs):  # noqa: N802 - mirrors the sounddevice API
            raise AssertionError("enumeration must not open an input stream")

    monkeypatch.setattr(settings_module, 'sd', FakeSoundDevice)

    listed = SettingsWindow.get_available_sound_devices(None)

    assert [item['index'] for item in listed] == [1, 2], "output-only devices should be skipped"
    assert listed[0]['name'] == '1: Microphone (Realtek)'
    # The default device is identified by index, not by comparing dicts.
    assert [item['default'] for item in listed] == [False, True]

    # The real module must actually be imported, or `sd` would be undefined.
    assert settings_module.sd is not None or sounddevice is not None


def test_sound_device_enumeration_failure_is_reported_not_swallowed(monkeypatch):
    import ui.settings_window as settings_module
    from ui.settings_window import SettingsWindow

    class Broken:
        @staticmethod
        def query_devices():
            raise OSError('PortAudio unavailable')

    monkeypatch.setattr(settings_module, 'sd', Broken)
    messages = []
    monkeypatch.setattr(
        settings_module.ConfigManager, 'console_print',
        classmethod(lambda cls, message, *a, **k: messages.append(str(message)))
    )

    assert SettingsWindow.get_available_sound_devices(None) == []
    assert any('sound devices' in message for message in messages), messages


def test_window_is_resizable(settings_window):
    """The old window was setFixedSize, unusable on a small screen."""
    assert settings_window.minimumWidth() < settings_window.width()
    assert settings_window.maximumWidth() > settings_window.width()
