import os
import sys
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QComboBox, QWidget


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _set_combobox_value(combo: QComboBox, value: str):
    index = combo.findData(value)
    if index == -1:
        index = combo.findText(value)
    assert index != -1, f"Value {value} not found in combo {combo.objectName()}"
    combo.setCurrentIndex(index)


@pytest.fixture
def settings_window(monkeypatch, qapp):
    sys.path.insert(0, 'src')
    from ui.settings_window import SettingsWindow, LLMProcessor

    monkeypatch.setattr(SettingsWindow, "get_available_sound_devices", lambda self: [])
    monkeypatch.setattr(LLMProcessor, "get_available_models", lambda self, api: ['gpt-5.1', 'gpt-4.1'])

    window = SettingsWindow()
    window.show()
    qapp.processEvents()
    yield window
    window.close()


def test_temperature_visibility_toggles(settings_window, qapp):
    api_combo = settings_window.findChild(QComboBox, 'llm_post_processing_api_type_input')
    _set_combobox_value(api_combo, 'openai')
    qapp.processEvents()

    cleanup_combo = settings_window.cleanup_model_combo
    instruction_combo = settings_window.instruction_model_combo
    assert cleanup_combo and instruction_combo

    cleanup_combo.setCurrentText('gpt-5.1')
    instruction_combo.setCurrentText('gpt-5.1')
    settings_window.update_temperature_visibility()
    qapp.processEvents()
    assert settings_window._should_hide_temperature() is True

    cleanup_combo.setCurrentText('gpt-4.1')
    instruction_combo.setCurrentText('gpt-4.1')
    settings_window.update_temperature_visibility()
    qapp.processEvents()
    assert settings_window._should_hide_temperature() is False


def test_azure_fields_visible(settings_window, qapp):
    api_combo = settings_window.findChild(QComboBox, 'llm_post_processing_api_type_input')
    _set_combobox_value(api_combo, 'azure_openai')
    qapp.processEvents()

    field_names = [
        'azure_openai_llm_cleanup_deployment_name',
        'azure_openai_llm_instruction_deployment_name',
        'azure_openai_llm_deployment_name',
        'azure_openai_llm_api_version'
    ]

    for name in field_names:
        widget = settings_window.findChild(QWidget, f'llm_post_processing_{name}_input')
        assert widget is not None and not widget.isHidden(), f"{name} should be visible"

