import sys
import types


def test_pynput_backend_start_stop_are_idempotent(monkeypatch):
    sys.path.insert(0, 'src')
    from key_listener import PynputBackend, ConfigManager

    monkeypatch.setattr(ConfigManager, 'console_print', lambda *args, **kwargs: None)

    created_listeners = []

    class FakeListener:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.started = 0
            self.stopped = 0
            created_listeners.append(self)

        def start(self):
            self.started += 1

        def stop(self):
            self.stopped += 1

    backend = PynputBackend()
    backend.keyboard = types.SimpleNamespace(Listener=lambda **kwargs: FakeListener(**kwargs))
    backend.mouse = types.SimpleNamespace(Listener=lambda **kwargs: FakeListener(**kwargs))
    backend.key_map = {}

    assert backend.start() is True
    assert backend.start() is False
    assert backend.is_running is True
    assert len(created_listeners) == 2

    assert backend.stop() is True
    assert backend.stop() is False
    assert backend.is_running is False
    assert sum(listener.started for listener in created_listeners) == 2
    assert sum(listener.stopped for listener in created_listeners) == 2

    sys.path.pop(0)