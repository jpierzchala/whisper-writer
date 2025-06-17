import sys
import types
from unittest.mock import MagicMock
import numpy as np

import pytest

# Helper fixture to set up ResultThread with dependencies mocked
@pytest.fixture
def setup_thread(monkeypatch):
    # Stub modules required by ResultThread
    monkeypatch.setitem(sys.modules, 'sounddevice', types.SimpleNamespace(InputStream=None))
    monkeypatch.setitem(sys.modules, 'webrtcvad', types.SimpleNamespace(Vad=lambda mode: None))

    class DummyMediaController:
        def __init__(self):
            self.was_playing = False
            self.initial_state_playing = False
        def pause_media(self):
            pass
        def resume_media(self):
            pass

    monkeypatch.setitem(sys.modules, 'media_controller', types.SimpleNamespace(MediaController=DummyMediaController))

    transcribe_mock = MagicMock(return_value='')
    monkeypatch.setitem(sys.modules, 'transcription', types.SimpleNamespace(transcribe=transcribe_mock))

    # Ensure fresh import of ResultThread for each test
    if 'result_thread' in sys.modules:
        del sys.modules['result_thread']

    sys.path.insert(0, 'src')
    from result_thread import ResultThread, time as rt_time
    from utils import ConfigManager

    # Ensure fresh ConfigManager
    ConfigManager._instance = None
    ConfigManager.initialize('src/config_schema.yaml')

    thread = ResultThread()
    statuses = []
    results = []
    thread.statusSignal = types.SimpleNamespace(emit=lambda status, use_llm: statuses.append(status))
    thread.resultSignal = types.SimpleNamespace(emit=lambda result: results.append(result))

    monkeypatch.setattr(ResultThread, '_record_audio', lambda self: np.zeros(16000, dtype=np.int16))
    save_mock = MagicMock(return_value='dummy.flac')
    monkeypatch.setattr(ResultThread, '_save_failed_audio', save_mock)
    monkeypatch.setattr(rt_time, 'sleep', lambda x: None)

    yield thread, transcribe_mock, statuses, results, save_mock

    sys.path.pop(0)


def test_transcription_retries_fail(setup_thread):
    thread, transcribe_mock, statuses, results, save_mock = setup_thread
    # Always return empty string to trigger retries
    transcribe_mock.return_value = ''
    thread.run()
    assert transcribe_mock.call_count == 3
    assert save_mock.called
    assert statuses and statuses[-1] == 'transcription_failed'
    assert results == []


def test_transcription_succeeds_after_errors(setup_thread):
    thread, transcribe_mock, statuses, results, save_mock = setup_thread
    # First raise an exception, then empty result, then success
    transcribe_mock.side_effect = [Exception('api error'), '', 'ok']
    thread.run()
    assert transcribe_mock.call_count == 3
    assert not save_mock.called
    assert results == ['ok']
