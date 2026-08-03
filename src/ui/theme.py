"""One stylesheet for the whole app, in a light and a dark variant.

Replaces per-widget setFont/setStyleSheet calls scattered across the UI files, so
spacing, colour and type are decided in one place and dark mode is possible at all.
"""
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QApplication

FONT_FAMILY = '"Segoe UI Variable", "Segoe UI", system-ui, sans-serif'

LIGHT = {
    'window': '#f5f6f8',
    'card': '#ffffff',
    'card_border': '#e2e5ea',
    'text': '#1b1f24',
    'text_muted': '#61686f',
    'field_bg': '#ffffff',
    'field_border': '#cfd4da',
    'field_border_focus': '#2f6fd0',
    'field_disabled_bg': '#f0f1f3',
    'accent': '#2f6fd0',
    'accent_text': '#ffffff',
    'accent_hover': '#2960b4',
    'button_bg': '#ffffff',
    'button_hover': '#eef1f6',
    'group_title': '#3b424a',
    'tab_bg': 'transparent',
    'tab_selected_bg': '#ffffff',
    'tab_hover_bg': '#e8ebf0',
    'shadow': 'rgba(16, 24, 40, 0.10)',
}

DARK = {
    'window': '#1c1f24',
    'card': '#24282e',
    'card_border': '#343941',
    'text': '#e8eaed',
    'text_muted': '#a2aab3',
    'field_bg': '#1a1d22',
    'field_border': '#3d434b',
    'field_border_focus': '#5b9bff',
    'field_disabled_bg': '#22252a',
    'accent': '#3d7dde',
    'accent_text': '#ffffff',
    'accent_hover': '#4c8bea',
    'button_bg': '#2b3037',
    'button_hover': '#343a42',
    'group_title': '#c3cad2',
    'tab_bg': 'transparent',
    'tab_selected_bg': '#24282e',
    'tab_hover_bg': '#2e333a',
    'shadow': 'rgba(0, 0, 0, 0.35)',
}


def prefers_dark_mode():
    """True when the OS palette is dark.

    Derived from the palette Qt already resolved rather than from a platform API,
    so it works the same on every platform and in offscreen tests.
    """
    app = QApplication.instance()
    if app is None:
        return False
    window_colour = app.palette().color(QPalette.Window)
    return window_colour.lightness() < 128


def get_palette(dark=None):
    return DARK if (prefers_dark_mode() if dark is None else dark) else LIGHT


def build_stylesheet(dark=None):
    """Return the application stylesheet for the light or dark variant."""
    colours = get_palette(dark)
    return f"""
    QWidget {{
        font-family: {FONT_FAMILY};
        font-size: 10pt;
        color: {colours['text']};
    }}

    QLabel[role="windowTitle"] {{
        font-size: 12pt;
        font-weight: 600;
        color: {colours['text']};
    }}
    QLabel[role="hint"] {{
        color: {colours['text_muted']};
        font-size: 9pt;
    }}
    QLabel[role="settingLabel"] {{
        color: {colours['text']};
        padding-top: 4px;
    }}

    QTabWidget::pane {{
        border: 1px solid {colours['card_border']};
        border-radius: 10px;
        background: {colours['card']};
        top: -1px;
    }}
    /* Qt sizes a styled tab from its text plus padding, but does not grow it for
       the bolder selected state, which clips the label. min-width covers that. */
    QTabBar::tab {{
        background: {colours['tab_bg']};
        color: {colours['text_muted']};
        padding: 8px 18px;
        margin-right: 2px;
        min-width: 92px;
        border: 1px solid transparent;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
    }}
    QTabBar::tab:hover {{
        background: {colours['tab_hover_bg']};
        color: {colours['text']};
    }}
    QTabBar::tab:selected {{
        background: {colours['tab_selected_bg']};
        color: {colours['text']};
        font-weight: 600;
        border: 1px solid {colours['card_border']};
        border-bottom-color: {colours['tab_selected_bg']};
    }}

    QGroupBox {{
        border: 1px solid {colours['card_border']};
        border-radius: 10px;
        margin-top: 16px;
        padding: 16px 14px 12px 14px;
        background: transparent;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 12px;
        padding: 0 6px;
        color: {colours['group_title']};
        font-weight: 600;
        font-size: 10pt;
    }}

    QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
        background: {colours['field_bg']};
        border: 1px solid {colours['field_border']};
        border-radius: 7px;
        padding: 6px 9px;
        selection-background-color: {colours['accent']};
        selection-color: {colours['accent_text']};
    }}
    QLineEdit:focus, QTextEdit:focus, QComboBox:focus,
    QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {colours['field_border_focus']};
    }}
    QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
        background: {colours['field_disabled_bg']};
        color: {colours['text_muted']};
    }}
    /* The drop-down subcontrol is deliberately left unstyled: overriding it drops
       the native arrow, and a combo box then looks like a plain text field. */
    QComboBox {{
        padding-right: 24px;
    }}
    QComboBox QAbstractItemView {{
        background: {colours['card']};
        border: 1px solid {colours['card_border']};
        selection-background-color: {colours['accent']};
        selection-color: {colours['accent_text']};
        padding: 4px;
    }}

    QPushButton {{
        background: {colours['button_bg']};
        border: 1px solid {colours['field_border']};
        border-radius: 7px;
        padding: 7px 16px;
        font-weight: 500;
    }}
    QPushButton:hover {{
        background: {colours['button_hover']};
    }}
    QPushButton[role="primary"] {{
        background: {colours['accent']};
        border-color: {colours['accent']};
        color: {colours['accent_text']};
    }}
    QPushButton[role="primary"]:hover {{
        background: {colours['accent_hover']};
        border-color: {colours['accent_hover']};
    }}
    QPushButton[role="titleBarClose"] {{
        background: transparent;
        border: none;
        color: {colours['text_muted']};
        font-size: 15pt;
        padding: 0;
    }}
    QPushButton[role="titleBarClose"]:hover {{
        color: {colours['text']};
    }}

    QToolButton[role="help"] {{
        background: transparent;
        border: none;
        color: {colours['text_muted']};
    }}

    QCheckBox::indicator {{
        width: 17px;
        height: 17px;
        border: 1px solid {colours['field_border']};
        border-radius: 4px;
        background: {colours['field_bg']};
    }}
    QCheckBox::indicator:checked {{
        background: {colours['accent']};
        border-color: {colours['accent']};
        image: none;
    }}

    QScrollArea {{
        border: none;
        background: transparent;
    }}
    QScrollBar:vertical {{
        background: transparent;
        width: 10px;
        margin: 2px;
    }}
    QScrollBar::handle:vertical {{
        background: {colours['field_border']};
        border-radius: 5px;
        min-height: 28px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {colours['text_muted']};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    """


def apply_theme(widget, dark=None):
    """Apply the stylesheet to a widget (usually a top-level window)."""
    widget.setStyleSheet(build_stylesheet(dark))


def card_colour(dark=None):
    """Background colour for the custom-painted window card."""
    return QColor(get_palette(dark)['card'])


def border_colour(dark=None):
    return QColor(get_palette(dark)['card_border'])
