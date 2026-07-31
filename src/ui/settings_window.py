import os
import sys
from collections import OrderedDict

from dotenv import load_dotenv
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (
    QApplication, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel, QLineEdit,
    QPushButton, QComboBox, QCheckBox, QMessageBox, QTabWidget, QWidget, QSizePolicy,
    QSpacerItem, QToolButton, QStyle, QFileDialog, QTextEdit, QSpinBox, QDoubleSpinBox,
    QScrollArea
)
from PyQt5.QtCore import Qt, QCoreApplication, QProcess, pyqtSignal, QMetaObject, QThread, QTimer
from PyQt5.QtGui import QFont

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ui.base_window import BaseWindow, QT_WIDGETS_ARE_MOCKED
from ui.theme import apply_theme
from utils import ConfigManager
from keyring_manager import KeyringManager
from llm_processor import LLMProcessor
from model_registry import is_reasoning_model
from ui.model_refresh_worker import ModelRefreshWorker
from whisper_languages import WHISPER_LANGUAGE_CHOICES, normalize_whisper_language

QWIDGET_IS_TYPE = isinstance(QWidget, type)
# Models offered in the cleanup/instruction dropdowns, best default first. Kept to
# models that are current on both OpenAI and Azure as of July 2026; the combo is
# editable so anything else can still be typed in.
CURATED_LLM_MODELS = (
    'gpt-5.6-luna',
    'gpt-5.6-terra',
    'gpt-5.6-sol',
    'gpt-5.4',
    'gpt-5.4-mini',
    'gpt-5.2',
)

# Single source of truth mapping a secret config key to its keyring entry name.
# Keyed by (category, key) because the same key name can appear in several sections.
KEYRING_SERVICE_BY_CONFIG_KEY = {
    ('model_options', 'openai_transcription_api_key'): 'openai_transcription',
    ('model_options', 'deepgram_transcription_api_key'): 'deepgram_transcription',
    ('model_options', 'groq_transcription_api_key'): 'groq_transcription',
    ('model_options', 'azure_openai_api_key'): 'azure_openai_transcription',
    ('llm_post_processing', 'claude_api_key'): 'claude',
    ('llm_post_processing', 'openai_api_key'): 'openai_llm',
    ('llm_post_processing', 'azure_openai_llm_api_key'): 'azure_openai_llm',
    ('llm_post_processing', 'gemini_api_key'): 'gemini',
    ('llm_post_processing', 'groq_api_key'): 'groq',
}
KEYRING_SERVICE_BY_KEY = {key: name for (_, key), name in KEYRING_SERVICE_BY_CONFIG_KEY.items()}

API_TYPE_LABELS = {
    'openai': 'OpenAI API',
    'azure_openai': 'Azure OpenAI',
    'claude': 'Anthropic Claude',
    'gemini': 'Google Gemini',
    'groq': 'Groq',
    'ollama': 'Ollama (local)'
}

load_dotenv()

class SettingsWindow(BaseWindow):
    settings_closed = pyqtSignal()
    settings_saved = pyqtSignal()

    def __init__(self):
        """Initialize the settings window."""
        super().__init__('Settings', 860, 720)
        ConfigManager.initialize()
        self.schema = ConfigManager.get_schema()
        self.llm_processor = None  # Initialize to None
        self.cleanup_model_combo = None
        self.instruction_model_combo = None
        self.refresh_thread = None  # Active model-refresh thread, if any
        self.refresh_worker = None
        # {objectName prefix: (form layout, row index)} so rows can be hidden wholesale
        self.setting_rows = {}
        self.headless_mode = QT_WIDGETS_ARE_MOCKED
        if not self.headless_mode:
            self.init_settings_ui()
            
            # Check if we're in API mode (no GPU tools available)
            try:
                import faster_whisper
                has_faster_whisper = True
            except ImportError:
                has_faster_whisper = False
            
            try:
                import vosk
                has_vosk = True
            except ImportError:
                has_vosk = False
            
            # If neither is available, set API mode
            if not has_faster_whisper and not has_vosk:
                self.set_api_mode(True)

    def init_settings_ui(self):
        """Initialize the settings user interface."""
        apply_theme(self)
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        self.create_tabs()
        self.create_buttons()

        # Connect the use_api checkbox state change
        self.use_api_checkbox = self.findChild(QCheckBox, 'model_options_use_api_input')
        if self.use_api_checkbox:
            self.use_api_checkbox.stateChanged.connect(lambda: self.toggle_api_local_options(self.use_api_checkbox.isChecked()))
            self.toggle_api_local_options(self.use_api_checkbox.isChecked())
        
        # Initialize provider-specific option visibility
        self.toggle_llm_provider_options()
        self.toggle_transcription_provider_options()
        self.update_temperature_visibility()

    def _ordered_categories(self):
        """Schema categories in display order, skipping metadata keys."""
        categories = [
            (category, settings) for category, settings in self.schema.items()
            if not ConfigManager.is_schema_metadata(category)
        ]
        categories.sort(key=lambda item: (item[1].get('_ui') or {}).get('order', 999))
        return categories

    @staticmethod
    def _category_settings(settings):
        """Yield (key, meta, sub_category) for every setting in a category."""
        for sub_category, sub_settings in settings.items():
            if ConfigManager.is_schema_metadata(sub_category) or not isinstance(sub_settings, dict):
                continue
            if 'value' in sub_settings:
                yield sub_category, sub_settings, None
            else:
                for key, meta in sub_settings.items():
                    if not ConfigManager.is_schema_metadata(key):
                        yield key, meta, sub_category

    def create_tabs(self):
        """Create one tab per schema category, its settings grouped into sections."""
        for category, settings in self._ordered_categories():
            tab = QWidget()

            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

            scroll_content = QWidget()
            tab_layout = QVBoxLayout(scroll_content)
            tab_layout.setContentsMargins(4, 4, 4, 4)
            tab_layout.setSpacing(4)

            self.create_settings_widgets(tab_layout, category, settings)

            tab_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
            scroll_area.setWidget(scroll_content)

            main_tab_layout = QVBoxLayout(tab)
            main_tab_layout.setContentsMargins(4, 8, 4, 4)
            main_tab_layout.addWidget(scroll_area)

            ui_meta = settings.get('_ui') or {}
            self.tabs.addTab(tab, ui_meta.get('title') or category.replace('_', ' ').capitalize())

    def create_settings_widgets(self, layout, category, settings):
        """Lay a category out as titled sections, one per `group` in the schema."""
        grouped = OrderedDict()
        for key, meta, sub_category in self._category_settings(settings):
            grouped.setdefault(meta.get('group') or 'Options', []).append((key, meta, sub_category))

        for group_name, entries in grouped.items():
            group_box = QGroupBox(group_name)
            # A vertical stack of row widgets rather than a QFormLayout: hidden rows
            # then collapse completely, whereas a form layout keeps their spacing and
            # leaves visible gaps wherever provider-specific fields are hidden.
            group_layout = QVBoxLayout(group_box)
            group_layout.setContentsMargins(6, 6, 6, 2)
            group_layout.setSpacing(8)

            for key, meta, sub_category in entries:
                self.add_setting_widget(group_layout, key, meta, category, sub_category)

            layout.addWidget(group_box)

    def create_buttons(self):
        """Create reset and save buttons."""
        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 6, 0, 0)
        button_row.addStretch(1)

        reset_button = QPushButton('Reset to saved settings')
        reset_button.clicked.connect(self.reset_settings)
        button_row.addWidget(reset_button)

        save_button = QPushButton('Save')
        save_button.setProperty('role', 'primary')
        save_button.setDefault(True)
        save_button.clicked.connect(self.save_settings)
        button_row.addWidget(save_button)

        self.main_layout.addLayout(button_row)

    LABEL_COLUMN_WIDTH = 190

    def add_setting_widget(self, group_layout, key, meta, category, sub_category=None):
        """Add one setting as a labelled row inside a group."""
        widget = self.create_widget_for_type(key, meta, category, sub_category)
        if not widget:
            return

        prefix = f"{category}_{sub_category}_{key}" if sub_category else f"{category}_{key}"
        description = meta.get('description', '')

        label_text = meta.get('label') or key.replace('_', ' ').capitalize()
        label = QLabel(f"{label_text}:")
        label.setObjectName(f"{prefix}_label")
        label.setProperty('role', 'settingLabel')
        label.setWordWrap(True)
        # A fixed label column keeps the fields aligned down the whole tab.
        label.setFixedWidth(self.LABEL_COLUMN_WIDTH)
        label.setToolTip(description)

        help_button = self.create_help_button(description)
        help_button.setObjectName(f"{prefix}_help")

        row = QWidget()
        row.setObjectName(f"{prefix}_row")
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)
        row_layout.addWidget(label, 0, Qt.AlignTop)
        row_layout.addWidget(widget, 1)
        row_layout.addWidget(help_button, 0, Qt.AlignTop)

        is_widget_like = QWIDGET_IS_TYPE and isinstance(widget, QWidget)
        if not is_widget_like and hasattr(widget, 'setObjectName'):
            is_widget_like = True

        if is_widget_like:
            widget.setObjectName(f"{prefix}_input")
        else:
            inner = widget.layout() if hasattr(widget, 'layout') else None
            line_edit = inner.itemAt(0).widget() if inner and inner.count() else None
            if line_edit is not None and hasattr(line_edit, 'setObjectName'):
                line_edit.setObjectName(f"{prefix}_input")

        group_layout.addWidget(row)
        self.setting_rows[prefix] = row

    def create_widget_for_type(self, key, meta, category, sub_category):
        """Create a widget based on the meta type."""
        meta_type = meta.get('type')
        current_value = self.get_config_value(category, sub_category, key, meta)

        # Special handling for sound device selection
        if category == 'recording_options' and key == 'sound_device':
            combo = QComboBox()
            devices = self.get_available_sound_devices()
            default_index = None  # Initialize default_index
            
            for device in devices:
                combo.addItem(device['name'], device['index'])
                if device['default']:
                    default_index = device['index']
            
            # Set current value if it exists, otherwise use default device if available
            if current_value is not None:
                index = combo.findData(int(current_value))
                if index >= 0:
                    combo.setCurrentIndex(index)
            elif default_index is not None:  # Only try to set default if one was found
                index = combo.findData(default_index)
                if index >= 0:
                    combo.setCurrentIndex(index)
            elif combo.count() > 0:  # If no default, but we have devices, select the first one
                combo.setCurrentIndex(0)
                
            return combo

        if meta.get('widget') == 'prompt':
            return self.create_prompt_editor(key, current_value)

        if category == 'llm_post_processing':
            if key == 'api_type':
                combo = QComboBox()
                combo.setObjectName('llm_post_processing_api_type_input')
                for option in meta['options']:
                    label = API_TYPE_LABELS.get(option, option.replace('_', ' ').title())
                    combo.addItem(label, option)
                index = combo.findData(current_value)
                if index >= 0:
                    combo.setCurrentIndex(index)
                combo.currentIndexChanged.connect(self.refresh_model_choices)
                combo.currentIndexChanged.connect(self.toggle_llm_provider_options)
                combo.currentIndexChanged.connect(self.update_temperature_visibility)
                return combo
            elif key in ['cleanup_model', 'instruction_model']:
                return self._create_model_combobox(key, current_value)

        if category == 'model_options' and sub_category == 'common' and key == 'language':
            return self.create_combobox(current_value, WHISPER_LANGUAGE_CHOICES)

        if meta.get('widget') == 'file':
            return self.create_file_picker(current_value, meta)

        if meta.get('widget') == 'multiline':
            text_edit = QTextEdit()
            text_edit.setPlainText(current_value or '')
            text_edit.setMinimumHeight(120)
            text_edit.setAcceptRichText(False)
            return text_edit

        if meta_type == 'bool':
            return self.create_checkbox(current_value, key)
        elif meta_type == 'str' and 'options' in meta:
            widget = self.create_combobox(current_value, meta['options'])
            # Add special handling for provider combo boxes
            if key == 'provider':
                widget.currentTextChanged.connect(self.toggle_transcription_provider_options)
            return widget
        elif meta_type == 'str':
            is_api_key = key.endswith('api_key')
            return self.create_line_edit(current_value, key, is_api_key)
        elif meta_type in ['int', 'float']:
            return self.create_number_input(current_value, meta)
        return None

    @staticmethod
    def create_number_input(value, meta):
        """Spin box for a numeric setting, so bad input cannot be typed at all."""
        is_float = meta.get('type') == 'float'
        spin = QDoubleSpinBox() if is_float else QSpinBox()

        default = meta.get('value') or 0
        # Numeric bounds are advisory: keep any value already in the config reachable.
        minimum = meta.get('min', 0)
        maximum = meta.get('max', max(default * 100, 1000000))
        current = value if value is not None else default
        try:
            current = float(current) if is_float else int(current)
        except (TypeError, ValueError):
            current = default

        spin.setRange(min(minimum, current), max(maximum, current))
        if is_float:
            spin.setDecimals(meta.get('decimals', 2))
        spin.setSingleStep(meta.get('step', 0.1 if is_float else 1))
        if meta.get('suffix'):
            spin.setSuffix(meta['suffix'])
        spin.setValue(current)
        spin.setMinimumWidth(130)
        return spin

    def create_prompt_editor(self, key, current_value):
        """Editor for a system prompt: inline text plus an optional companion file.

        The file is appended to the inline text and re-read on every dictation, so
        prompts can be iterated on without restarting the app.
        """
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        text_edit = QTextEdit()
        text_edit.setPlaceholderText("System prompt…")
        text_edit.setPlainText(current_value or '')
        text_edit.setMinimumHeight(110)
        text_edit.setAcceptRichText(False)

        file_row = QHBoxLayout()
        file_row.setContentsMargins(0, 0, 0, 0)
        file_row.setSpacing(6)

        file_edit = QLineEdit()
        file_edit.setPlaceholderText("Optional: file appended to the prompt above")
        file_edit.setObjectName(f"{key}_file_path")
        saved_file_path = ConfigManager.get_config_value("llm_post_processing", f"{key}_file_path")
        if saved_file_path:
            file_edit.setText(saved_file_path)

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(lambda: self.browse_system_message_file(file_edit, text_edit))

        file_row.addWidget(file_edit, 1)
        file_row.addWidget(browse_button, 0)

        layout.addWidget(text_edit)
        layout.addLayout(file_row)
        return container

    def create_file_picker(self, current_value, meta):
        """Line edit plus a Browse button for a path setting."""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        file_edit = QLineEdit(current_value or '')
        file_edit.setPlaceholderText(meta.get('placeholder') or "Select a file…")

        browse_button = QPushButton('Browse')
        browse_button.clicked.connect(lambda: self.browse_into(file_edit))

        layout.addWidget(file_edit, 1)
        layout.addWidget(browse_button, 0)
        return container

    def browse_into(self, file_edit):
        """Pick a file and put its path into `file_edit`."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select File", "", "Text Files (*.txt *.json *.log);;All Files (*)"
        )
        if file_path:
            file_edit.setText(file_path)

    def create_checkbox(self, value, key):
        checkbox_class = getattr(QtWidgets, 'QCheckBox', QCheckBox)
        widget = checkbox_class()
        widget.setChecked(value)
        if key == 'use_api':
            widget.setObjectName('model_options_use_api_input')
        elif key == 'autostart_on_login':
            # Import AutostartManager for platform checking
            import platform
            if platform.system() != 'Windows':
                widget.setEnabled(False)
                widget.setToolTip("Autostart is only supported on Windows")
        return widget

    def _create_model_combobox(self, key, current_value):
        """Create a combo box for model selection with default and fetched options."""
        combo = QComboBox()
        combo.setEditable(True)
        combo.setInsertPolicy(QComboBox.NoInsert)
        combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        combo.setObjectName(f'llm_post_processing_{key}_input')

        placeholder = "Select cleanup model" if key == 'cleanup_model' else "Select instruction model"
        if combo.lineEdit():
            combo.lineEdit().setPlaceholderText(placeholder)

        for model_name in self._default_llm_model_choices():
            combo.addItem(model_name)

        if current_value:
            index = combo.findText(current_value)
            if index == -1:
                combo.insertItem(0, current_value)
                index = 0
            combo.setCurrentIndex(index)
        elif combo.count() > 0:
            combo.setCurrentIndex(0)

        if key == 'cleanup_model':
            self.cleanup_model_combo = combo
        else:
            self.instruction_model_combo = combo

        combo.currentTextChanged.connect(self.update_temperature_visibility)
        self.update_temperature_visibility()

        return combo

    @staticmethod
    def _default_llm_model_choices():
        """Currently supported OpenAI model IDs, best default first.

        Retired and deprecated models are deliberately absent: the combo stays
        editable, so anything omitted here can still be typed in by hand.
        """
        return list(CURATED_LLM_MODELS)

    def create_combobox(self, value, options):
        widget = QComboBox()
        for option in options:
            if isinstance(option, tuple):
                widget.addItem(option[0], option[1])
            else:
                widget.addItem(option)
        self._set_combobox_value(widget, value)
        return widget

    def _set_combobox_value(self, combo, value):
        index = combo.findData(value)
        if index == -1 and value is not None:
            index = combo.findText(str(value))

        if index == -1:
            normalized_value = normalize_whisper_language(value)
            for option_index in range(combo.count()):
                option_data = combo.itemData(option_index)
                option_text = combo.itemText(option_index)
                option_value = option_data if option_data is not None else option_text
                if normalize_whisper_language(option_value) == normalized_value:
                    index = option_index
                    break

        if index >= 0:
            combo.setCurrentIndex(index)

    def create_line_edit(self, value, key=None, password_mode=False):
        widget = QLineEdit(value)
        
        if password_mode:
            widget.setEchoMode(QLineEdit.Password)
            keyring_name = KEYRING_SERVICE_BY_KEY.get(key)
            if keyring_name:
                widget.setText(KeyringManager.get_api_key(keyring_name) or value)
        elif key == 'model_path':
            widget.setPlaceholderText("Optional: Path to local model file")
        
        return widget

    def create_help_button(self, description):
        help_button = QToolButton()
        if hasattr(self, 'style'):
            style_obj = self.style()
            if style_obj and hasattr(style_obj, 'standardIcon'):
                help_button.setIcon(style_obj.standardIcon(QStyle.SP_MessageBoxQuestion))
        help_button.setAutoRaise(True)
        help_button.setToolTip(description)
        help_button.setCursor(Qt.PointingHandCursor)
        help_button.setFocusPolicy(Qt.TabFocus)
        help_button.clicked.connect(lambda: self.show_description(description))
        return help_button

    def get_config_value(self, category, sub_category, key, meta):
        if sub_category:
            value = ConfigManager.get_config_value(category, sub_category, key)
        else:
            value = ConfigManager.get_config_value(category, key)
        return value if value is not None else meta['value']

    def browse_model_path(self, widget):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Whisper Model File", "", "Model Files (*.bin);;All Files (*)")
        if file_path:
            widget.setText(file_path)

    def show_description(self, description):
        """Show a description dialog."""
        QMessageBox.information(self, 'Description', description)

    def save_settings(self):
        """Save the settings to the config file and keyring."""
        ConfigManager.console_print("Saving settings...")
        self.iterate_settings(self.save_setting)

        # Move secrets from the config into the keyring, then blank them in the config
        for (category, key), keyring_name in KEYRING_SERVICE_BY_CONFIG_KEY.items():
            path = (category, 'api', key) if category == 'model_options' else (category, key)
            KeyringManager.save_api_key(keyring_name, ConfigManager.get_config_value(*path) or '')
            ConfigManager.set_config_value(None, *path)

        ConfigManager.save_config()
        
        # Handle autostart setting (Windows only)
        autostart_enabled = ConfigManager.get_config_value('misc', 'autostart_on_login')
        if autostart_enabled is not None:
            try:
                from autostart_manager import AutostartManager
                success, message = AutostartManager.set_autostart(autostart_enabled)
                if not success:
                    QMessageBox.warning(self, 'Autostart Warning', f'Could not configure autostart: {message}')
            except ImportError:
                QMessageBox.warning(self, 'Autostart Warning', 'Autostart functionality is not available')
        
        QMessageBox.information(self, 'Settings Saved', 'Settings have been saved. The application will now restart.')
        self.settings_saved.emit()
        self.close()

    def save_setting(self, widget, category, sub_category, key, meta):
        """Save a single setting to the config."""
        if isinstance(widget, QWidget) and widget.layout():
            text_edit, file_edit = self._find_composite_editors(widget.layout())

            # A prompt editor holds the prompt itself plus an optional companion
            # file; a plain file picker holds only a path, which IS the setting.
            if text_edit is not None:
                ConfigManager.set_config_value(text_edit.toPlainText(), category, key)
                if file_edit is not None:
                    ConfigManager.set_config_value(file_edit.text(), category, f"{key}_file_path")
            elif file_edit is not None:
                if sub_category:
                    ConfigManager.set_config_value(file_edit.text(), category, sub_category, key)
                else:
                    ConfigManager.set_config_value(file_edit.text(), category, key)
            return

        # Special handling for sound device combo box
        if category == 'recording_options' and key == 'sound_device' and isinstance(widget, QComboBox):
            value = widget.currentData()  # Get the device index
            ConfigManager.console_print(f"Saving sound device selection: {value} ({widget.currentText()})")
        else:
            # Handle regular widgets
            value = self.get_widget_value_typed(widget, meta.get('type'))
        
        if sub_category:
            ConfigManager.set_config_value(value, category, sub_category, key)
        else:
            ConfigManager.set_config_value(value, category, key)

    @staticmethod
    def _find_composite_editors(layout):
        """Find the text edit and/or line edit inside a composite setting widget."""
        text_edit = None
        file_edit = None
        for index in range(layout.count()):
            item = layout.itemAt(index)
            widget = item.widget()
            if isinstance(widget, QTextEdit):
                text_edit = widget
            elif isinstance(widget, QLineEdit):
                file_edit = file_edit or widget
            elif item.layout() is not None:
                nested_text, nested_file = SettingsWindow._find_composite_editors(item.layout())
                text_edit = text_edit or nested_text
                file_edit = file_edit or nested_file
        return text_edit, file_edit

    def reset_settings(self):
        """Reset the settings to the saved values."""
        ConfigManager.reload_config()
        self.update_widgets_from_config()

    def update_widgets_from_config(self):
        """Update all widgets with values from the current configuration."""
        ConfigManager.console_print("Updating widgets from config...")

        self.iterate_settings(self.update_widget_value)

        # Secret fields are not stored in the config file, so refill them from the keyring
        for widget, category, sub_category, key, _ in self.iter_setting_widgets():
            keyring_name = KEYRING_SERVICE_BY_CONFIG_KEY.get((category, key))
            if keyring_name and hasattr(widget, 'setText'):
                widget.setText(KeyringManager.get_api_key(keyring_name) or '')

    def update_widget_value(self, widget, category, sub_category, key, meta):
        """Update a single widget with its value from the config."""
        # Skip API key fields as they're refilled from the keyring separately
        if (category, key) in KEYRING_SERVICE_BY_CONFIG_KEY:
            return

        value = self.get_config_value(category, sub_category, key, meta)
        self.set_widget_value(widget, value, meta.get('type'))

    def set_widget_value(self, widget, value, value_type):
        """Set the value of the widget."""
        if isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, QComboBox):
            self._set_combobox_value(widget, value)
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            if value is not None:
                try:
                    widget.setValue(float(value) if isinstance(widget, QDoubleSpinBox) else int(value))
                except (TypeError, ValueError):
                    pass
        elif isinstance(widget, QLineEdit):
            widget.setText(str(value) if value is not None else '')
        elif isinstance(widget, QTextEdit):
            widget.setPlainText(str(value) if value is not None else '')
        elif isinstance(widget, QWidget) and widget.layout():
            # Composite rows (file pickers, model_path) keep the path in a line edit.
            line_edit = widget.layout().itemAt(0).widget()
            if isinstance(line_edit, QLineEdit):
                line_edit.setText(str(value) if value is not None else '')

    def get_widget_value_typed(self, widget, value_type):
        """Get the value of the widget with proper typing."""
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        elif isinstance(widget, QComboBox):
            return self._get_combobox_value(widget)
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return widget.value()
        elif isinstance(widget, QLineEdit):
            text = widget.text()
            if value_type == 'int':
                return int(text) if text else None
            elif value_type == 'float':
                return float(text) if text else None
            else:
                return text or None
        elif isinstance(widget, QTextEdit):
            return widget.toPlainText() or None
        elif isinstance(widget, QWidget) and widget.layout():
            line_edit = widget.layout().itemAt(0).widget()
            if isinstance(line_edit, QLineEdit):
                return line_edit.text() or None
        return None

    def update_temperature_visibility(self):
        temp_widget = self.findChild(QWidget, 'llm_post_processing_temperature_input')
        temp_label = self.findChild(QLabel, 'llm_post_processing_temperature_label')
        temp_help = self.findChild(QToolButton, 'llm_post_processing_temperature_help')
        should_hide = self._should_hide_temperature()
        for element in (temp_widget, temp_label, temp_help):
            if element:
                element.setVisible(not should_hide)

    def _should_hide_temperature(self) -> bool:
        provider_combo = self.findChild(QComboBox, 'llm_post_processing_api_type_input')
        provider = self._get_combobox_value(provider_combo)
        if provider not in ('openai', 'azure_openai'):
            return False
        selected_models = []
        for combo in (self.cleanup_model_combo, self.instruction_model_combo):
            if combo:
                selected_models.append(self._get_combobox_value(combo))
        return any(self._is_reasoning_model(value) for value in selected_models if value)

    @staticmethod
    def _is_reasoning_model(value: str) -> bool:
        """True when the model takes a reasoning effort instead of a temperature."""
        return bool(value) and is_reasoning_model(value)

    @staticmethod
    def _get_combobox_value(combo: QComboBox | None):
        if not combo:
            return None
        data = combo.currentData()
        if data is not None:
            return data
        return combo.currentText() or None

    def toggle_api_local_options(self, use_api):
        """Toggle visibility of API and local options."""
        self.iterate_settings(lambda w, c, s, k, m: self.toggle_widget_visibility(w, c, s, k, use_api))

    def toggle_widget_visibility(self, widget, category, sub_category, key, use_api):
        if sub_category in ['api', 'local']:
            widget.setVisible(use_api if sub_category == 'api' else not use_api)
            
            # Also toggle visibility of the corresponding label and help button
            label = self.findChild(QLabel, f"{category}_{sub_category}_{key}_label")
            help_button = self.findChild(QToolButton, f"{category}_{sub_category}_{key}_help")
            
            if label:
                label.setVisible(use_api if sub_category == 'api' else not use_api)
            if help_button:
                help_button.setVisible(use_api if sub_category == 'api' else not use_api)

    def iter_setting_widgets(self):
        """Yield (widget, category, sub_category, key, meta) for every settings widget."""
        for category, settings in self._ordered_categories():
            for key, meta, sub_category in self._category_settings(settings):
                prefix = f"{category}_{sub_category}_{key}" if sub_category else f"{category}_{key}"
                widget = self.findChild(QWidget, f"{prefix}_input")
                if widget:
                    yield widget, category, sub_category, key, meta

    def iterate_settings(self, func):
        """Iterate over all settings and apply a function to each."""
        for widget, category, sub_category, key, meta in self.iter_setting_widgets():
            func(widget, category, sub_category, key, meta)

    def handleCloseButton(self):
        """Override base window close button handler to hide instead of close."""
        ConfigManager.console_print("Settings window closing via X button")
        self.settings_closed.emit()
        self.hide()

    def reject(self):
        """Handle when the window is closed without saving (e.g., Escape key or X button)."""
        self.settings_closed.emit()
        self.hide()

    def closeEvent(self, event):
        """Make sure a running model refresh cannot outlive the window."""
        thread = self.refresh_thread
        if thread and thread.isRunning():
            thread.quit()
            thread.wait(2000)
        super().closeEvent(event)

    def refresh_model_choices(self, combo_box=None):
        """Fetch the available models for the selected API type on a worker thread."""
        api_type_combo = self.findChild(QComboBox, 'llm_post_processing_api_type_input')
        if not api_type_combo:
            ConfigManager.console_print("Error: API type combo box not found")
            return

        api_type = self._get_combobox_value(api_type_combo)
        ConfigManager.console_print(f"Refreshing model list for API type: {api_type}")

        if not self.llm_processor:
            self.llm_processor = LLMProcessor(api_type=api_type)
        else:
            self.llm_processor.api_type = api_type

        combos = self._model_combos()
        if not combos:
            return

        # A refresh is already running; it will pick up the latest api_type when it finishes.
        if self.refresh_thread is not None:
            return

        for combo in combos:
            combo.setEnabled(False)
            if combo.lineEdit():
                combo.lineEdit().setPlaceholderText("Loading models…")

        self.refresh_thread = QThread(self)
        self.refresh_worker = ModelRefreshWorker(self.llm_processor, api_type)
        self.refresh_worker.moveToThread(self.refresh_thread)
        self.refresh_thread.started.connect(self.refresh_worker.run)
        self.refresh_worker.finished.connect(self._on_models_fetched)
        self.refresh_worker.finished.connect(self.refresh_thread.quit)
        self.refresh_thread.finished.connect(self._on_refresh_thread_finished)
        self.refresh_thread.start()

    def _model_combos(self):
        """Return the cleanup/instruction model combo boxes that exist in the UI."""
        return [combo for combo in (self.cleanup_model_combo, self.instruction_model_combo) if combo]

    def _on_refresh_thread_finished(self):
        """Tear down the finished refresh thread so a later refresh can start."""
        thread, self.refresh_thread = self.refresh_thread, None
        self.refresh_worker = None
        if thread:
            thread.deleteLater()

    def _on_models_fetched(self, models):
        """Populate the model combos with the fetched list (runs on the UI thread)."""
        api_type = self.llm_processor.api_type if self.llm_processor else None
        options = list(models or [])
        status = ''

        if not options:
            if api_type in ('openai', 'azure_openai'):
                options = self._default_llm_model_choices()
                status = "Could not fetch models – showing known model IDs"
            elif api_type == 'ollama':
                status = "No models found – is Ollama running?"
            else:
                status = "No models available – check the API key"
            ConfigManager.console_print(f"Model refresh: {status or 'no models returned'}")

        self.update_model_combos(options, self._model_combos(), status)

    def update_model_combos(self, models, combos_to_update, status=''):
        """Repopulate the given combo boxes, preserving each current selection."""
        combo_options = [str(model) for model in (models or [])]

        for combo in combos_to_update:
            if not isinstance(combo, QComboBox):
                continue

            # Fall back to the configured value when nothing is typed in the combo yet.
            desired_text = combo.currentText()
            if not desired_text:
                config_key = 'cleanup_model' if combo is self.cleanup_model_combo else 'instruction_model'
                desired_text = ConfigManager.get_config_value('llm_post_processing', config_key) or ''

            combo.blockSignals(True)
            combo.clear()
            combo.addItems(combo_options)

            if desired_text:
                index = combo.findText(desired_text)
                if index == -1:
                    combo.insertItem(0, desired_text)
                    index = 0
                combo.setCurrentIndex(index)
            elif combo.count():
                combo.setCurrentIndex(0)
            combo.blockSignals(False)

            combo.setEnabled(True)
            if combo.lineEdit():
                combo.lineEdit().setPlaceholderText(status)

        self.update_temperature_visibility()

    def browse_system_message_file(self, file_edit, text_edit):
        """Browse for a system message file and load its contents."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select System Message File", "", "Text Files (*.txt);;All Files (*)")
        if file_path:
            file_edit.setText(file_path)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text_edit.setText(file.read())
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to read file: {str(e)}")

    def set_api_mode(self, use_api: bool):
        """Set the API mode checkbox state."""
        use_api_checkbox = self.findChild(QCheckBox, 'model_options_use_api_input')
        if use_api_checkbox:
            use_api_checkbox.setChecked(use_api)
            self.toggle_api_local_options(use_api)

    def get_available_sound_devices(self):
        """Get list of available sound devices that support recording."""
        try:
            devices = sd.query_devices()
            input_devices = []
            for i, device in enumerate(devices):
                try:
                    # Test if we can open an input stream with this device
                    with sd.InputStream(device=i, channels=1, samplerate=16000, blocksize=1024):
                        pass  # If we get here, the device works for recording
                    
                    if device['max_input_channels'] > 0:  # Only include input devices
                        name = f"{i}: {device['name']}"
                        input_devices.append({
                            'index': i,
                            'name': name,
                            'channels': device['max_input_channels'],
                            'default': device is sd.default.device[0]
                        })
                except sd.PortAudioError as e:
                    # ConfigManager.console_print(f"Device {i}: {device['name']} not suitable for recording: {str(e)}")
                    continue
                
            return input_devices
        except Exception as e:
            ConfigManager.console_print(f"Error getting sound devices: {str(e)}")
            return []

    def toggle_llm_provider_options(self, provider=None):
        """Toggle visibility of LLM provider-specific options based on selected provider."""
        if provider is None:
            api_type_combo = self.findChild(QComboBox, 'llm_post_processing_api_type_input')
            if api_type_combo:
                provider = self._get_combobox_value(api_type_combo)
            else:
                return
        
        ConfigManager.console_print(f"Toggling LLM provider options for: {provider}")
        
        # Map of provider to their specific API key fields
        provider_fields = {
            'openai': ['openai_api_key'],
            'azure_openai': ['azure_openai_llm_api_key', 'azure_openai_llm_endpoint',
                             'azure_openai_llm_cleanup_deployment_name',
                             'azure_openai_llm_cleanup_model_family',
                             'azure_openai_llm_instruction_deployment_name',
                             'azure_openai_llm_instruction_model_family',
                             'azure_openai_llm_deployment_name',
                             'azure_api_mode', 'azure_openai_llm_api_version'],
            'claude': ['claude_api_key'],
            'gemini': ['gemini_api_key'],
            'groq': ['groq_api_key'],
            'ollama': []  # No API key needed for Ollama
        }
        
        all_fields = {field for fields in provider_fields.values() for field in fields}
        selected_fields = set(provider_fields.get(provider, []))

        for field in all_fields:
            self._set_setting_row_visible(
                'llm_post_processing', None, field, field in selected_fields
            )

        ConfigManager.console_print(f"Finished toggling options for provider: {provider}")
        self.update_temperature_visibility()

    def toggle_transcription_provider_options(self, provider=None):
        """Toggle visibility of transcription provider-specific options."""
        if provider is None:
            provider_combo = self.findChild(QComboBox, 'model_options_api_provider_input')
            if provider_combo:
                provider = self._get_combobox_value(provider_combo)
            else:
                return

        ConfigManager.console_print(f"Toggling transcription provider options for: {provider}")

        # Map of provider to their specific fields
        provider_fields = {
            'openai': ['openai_transcription_api_key', 'base_url'],
            'azure_openai': ['azure_openai_api_key', 'azure_openai_endpoint',
                             'azure_openai_deployment_name', 'azure_openai_model_family',
                             'azure_api_mode', 'azure_openai_api_version'],
            'deepgram': ['deepgram_transcription_api_key'],
            'groq': ['groq_transcription_api_key']
        }

        # Only touch fields while the API section itself is visible; in local mode
        # toggle_api_local_options owns their visibility.
        use_api_checkbox = self.findChild(QCheckBox, 'model_options_use_api_input')
        if use_api_checkbox and not use_api_checkbox.isChecked():
            return

        all_fields = {field for fields in provider_fields.values() for field in fields}
        selected_fields = set(provider_fields.get(provider, []))

        for field in all_fields:
            self._set_setting_row_visible(
                'model_options', 'api', field, field in selected_fields
            )

    def _set_setting_row_visible(self, category, sub_category, key, visible):
        """Show or hide a whole settings row, label and help button included."""
        prefix = f'{category}_{sub_category}_{key}' if sub_category else f'{category}_{key}'

        row = self.setting_rows.get(prefix)
        if row is not None:
            # Hiding the whole row makes the layout reclaim its space and spacing.
            row.setVisible(visible)

        # Also set the flag on the editor itself: callers and tests ask about
        # visibility by object name, and a hidden parent does not set that flag.
        for widget_type, suffix in ((QWidget, 'input'), (QLabel, 'label'), (QToolButton, 'help')):
            element = self.findChild(widget_type, f'{prefix}_{suffix}')
            if element:
                element.setVisible(visible)
                