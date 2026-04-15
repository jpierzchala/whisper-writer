import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def input_simulation_module(monkeypatch):
    if 'input_simulation' in sys.modules:
        del sys.modules['input_simulation']
    sys.path.insert(0, 'src')
    import input_simulation

    monkeypatch.setattr(
        input_simulation.ConfigManager,
        'get_config_value',
        lambda category, key: 'pynput' if (category, key) == ('post_processing', 'input_method') else None,
    )
    monkeypatch.setattr(input_simulation.ConfigManager, 'console_print', lambda *args, **kwargs: None)

    class DummyPressed:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyKeyboard:
        def pressed(self, _key):
            return DummyPressed()

        def press(self, _key):
            return None

        def release(self, _key):
            return None

    monkeypatch.setattr(input_simulation, 'PynputController', DummyKeyboard)

    yield input_simulation

    sys.path.pop(0)


def test_clipboard_restore_delay_is_longer_for_rich_content(input_simulation_module):
    module = input_simulation_module

    assert module.InputSimulator.get_clipboard_restore_delay({module.win32con.CF_UNICODETEXT: 'text'}) == 0.2
    assert module.InputSimulator.get_clipboard_restore_delay({module.win32con.CF_BITMAP: object()}) == 0.5


def test_paste_waits_longer_before_restoring_rich_clipboard(input_simulation_module, monkeypatch):
    module = input_simulation_module

    sleep_calls = []
    enum_order = [module.win32con.CF_BITMAP, module.win32con.CF_UNICODETEXT]

    def fake_enum_clipboard_formats(previous):
        if previous == 0:
            return enum_order[0]
        if previous == enum_order[0]:
            return enum_order[1]
        return 0

    def fake_get_clipboard_data(format_id):
        if format_id == module.win32con.CF_BITMAP:
            return SimpleNamespace(name='bitmap')
        if format_id == module.win32con.CF_UNICODETEXT:
            return 'pasted text'
        raise AssertionError(f'unexpected format {format_id}')

    monkeypatch.setattr(module.InputSimulator, 'safe_open_clipboard', staticmethod(lambda: True))
    monkeypatch.setattr(module.InputSimulator, 'safe_close_clipboard', staticmethod(lambda: True))
    monkeypatch.setattr(module.win32clipboard, 'EnumClipboardFormats', fake_enum_clipboard_formats)
    monkeypatch.setattr(module.win32clipboard, 'GetClipboardData', fake_get_clipboard_data)
    monkeypatch.setattr(module.win32clipboard, 'EmptyClipboard', lambda: None)
    monkeypatch.setattr(module.win32clipboard, 'SetClipboardText', lambda text, fmt: None)
    monkeypatch.setattr(module.win32clipboard, 'SetClipboardData', lambda fmt, data: None)
    monkeypatch.setattr(module.time, 'sleep', lambda seconds: sleep_calls.append(seconds))

    simulator = module.InputSimulator()
    simulator._paste_with_clipboard_preservation('pasted text')

    assert sleep_calls == [0.5]


def test_paste_skips_restoring_when_clipboard_changes(input_simulation_module, monkeypatch):
    module = input_simulation_module

    restored_formats = []

    monkeypatch.setattr(module.InputSimulator, 'safe_open_clipboard', staticmethod(lambda: True))
    monkeypatch.setattr(module.InputSimulator, 'safe_close_clipboard', staticmethod(lambda: True))
    monkeypatch.setattr(
        module.InputSimulator,
        'capture_open_clipboard_formats',
        classmethod(lambda cls: {module.win32con.CF_UNICODETEXT: 'original text'}),
    )
    monkeypatch.setattr(module.InputSimulator, 'get_open_clipboard_text', staticmethod(lambda: 'different text'))
    monkeypatch.setattr(module.win32clipboard, 'EmptyClipboard', lambda: None)
    monkeypatch.setattr(module.win32clipboard, 'SetClipboardText', lambda text, fmt: None)
    monkeypatch.setattr(
        module.InputSimulator,
        'restore_open_clipboard_formats',
        classmethod(lambda cls, saved_formats: restored_formats.append(saved_formats)),
    )
    monkeypatch.setattr(module.time, 'sleep', lambda seconds: None)

    simulator = module.InputSimulator()
    simulator._paste_with_clipboard_preservation('pasted text')

    assert restored_formats == []