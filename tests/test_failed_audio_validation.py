import sys
import types
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import tempfile
import os


@pytest.fixture
def setup_result_thread(monkeypatch):
    """Set up ResultThread with all dependencies mocked for testing failed audio saving."""
    # Mock all required modules
    monkeypatch.setitem(sys.modules, 'sounddevice', types.SimpleNamespace(InputStream=None))
    monkeypatch.setitem(sys.modules, 'webrtcvad', types.SimpleNamespace(Vad=lambda mode: None))
    monkeypatch.setitem(sys.modules, 'soundfile', MagicMock())

    class DummyMediaController:
        def __init__(self):
            self.was_playing = False
        def pause_media(self):
            pass
        def resume_media(self):
            pass

    monkeypatch.setitem(sys.modules, 'media_controller', types.SimpleNamespace(MediaController=DummyMediaController))
    monkeypatch.setitem(sys.modules, 'transcription', types.SimpleNamespace(transcribe=MagicMock(return_value='')))

    # Mock ConfigManager
    class MockConfigManager:
        @staticmethod
        def console_print(msg):
            pass  # Silent for tests
        
        @staticmethod
        def get_config_value(section, key, default=None):
            return False
        
        @staticmethod
        def get_config_section(section):
            return {'sample_rate': 16000}

    monkeypatch.setitem(sys.modules, 'utils', types.SimpleNamespace(ConfigManager=MockConfigManager))

    # Clean import for fresh state
    if 'result_thread' in sys.modules:
        del sys.modules['result_thread']

    sys.path.insert(0, 'src')
    from result_thread import ResultThread

    thread = ResultThread()
    yield thread
    
    sys.path.pop(0)


class TestFailedAudioSaving:
    """Test suite for the _save_failed_audio method validation and functionality."""

    def test_save_failed_audio_with_valid_data(self, setup_result_thread):
        """Test that valid audio data is saved successfully."""
        thread = setup_result_thread
        thread.sample_rate = 16000
        
        # Create valid audio data
        audio_data = np.random.randint(-32768, 32768, 16000, dtype=np.int16)
        
        with patch('soundfile.write') as mock_write, \
             patch('os.makedirs') as mock_makedirs, \
             patch('time.strftime', return_value='20240101-120000'):
            
            result = thread._save_failed_audio(audio_data)
            
            # Verify sf.write was called with correct parameters
            mock_write.assert_called_once()
            call_args = mock_write.call_args[0]
            assert call_args[1] is audio_data  # audio_data parameter
            assert call_args[2] == 16000  # sample_rate parameter
            
            # Verify return value is correct file path
            assert result.endswith('failed_20240101-120000.flac')
            assert 'failed_audio' in result
            
            # Verify directory creation was attempted
            mock_makedirs.assert_called_once()

    def test_save_failed_audio_with_none_data(self, setup_result_thread):
        """Test that None audio data is properly rejected."""
        thread = setup_result_thread
        thread.sample_rate = 16000
        
        with patch('soundfile.write') as mock_write:
            result = thread._save_failed_audio(None)
            
            # Should not attempt to write file
            mock_write.assert_not_called()
            # Should return empty string
            assert result == ''

    def test_save_failed_audio_with_empty_data(self, setup_result_thread):
        """Test that empty audio data is properly rejected."""
        thread = setup_result_thread
        thread.sample_rate = 16000
        
        empty_audio = np.array([], dtype=np.int16)
        
        with patch('soundfile.write') as mock_write:
            result = thread._save_failed_audio(empty_audio)
            
            # Should not attempt to write file
            mock_write.assert_not_called()
            # Should return empty string
            assert result == ''

    def test_save_failed_audio_with_no_sample_rate(self, setup_result_thread):
        """Test that missing sample rate is properly handled."""
        thread = setup_result_thread
        # Don't set sample_rate - it should be None by default
        
        audio_data = np.random.randint(-32768, 32768, 16000, dtype=np.int16)
        
        with patch('soundfile.write') as mock_write:
            result = thread._save_failed_audio(audio_data)
            
            # Should not attempt to write file
            mock_write.assert_not_called()
            # Should return empty string
            assert result == ''

    def test_save_failed_audio_with_zero_sample_rate(self, setup_result_thread):
        """Test that zero sample rate is properly handled."""
        thread = setup_result_thread
        thread.sample_rate = 0
        
        audio_data = np.random.randint(-32768, 32768, 16000, dtype=np.int16)
        
        with patch('soundfile.write') as mock_write:
            result = thread._save_failed_audio(audio_data)
            
            # Should not attempt to write file
            mock_write.assert_not_called()
            # Should return empty string
            assert result == ''

    def test_save_failed_audio_file_system_error(self, setup_result_thread):
        """Test that file system errors are properly handled."""
        thread = setup_result_thread
        thread.sample_rate = 16000
        
        audio_data = np.random.randint(-32768, 32768, 16000, dtype=np.int16)
        
        with patch('soundfile.write', side_effect=OSError("Permission denied")) as mock_write, \
             patch('os.makedirs') as mock_makedirs, \
             patch('time.strftime', return_value='20240101-120000'):
            
            result = thread._save_failed_audio(audio_data)
            
            # Should attempt to write file
            mock_write.assert_called_once()
            # Should return empty string due to error
            assert result == ''

    def test_save_failed_audio_soundfile_format_error(self, setup_result_thread):
        """Test that soundfile format errors are properly handled."""
        thread = setup_result_thread
        thread.sample_rate = 16000
        
        audio_data = np.random.randint(-32768, 32768, 16000, dtype=np.int16)
        
        with patch('soundfile.write', side_effect=ValueError("Invalid format")) as mock_write, \
             patch('os.makedirs') as mock_makedirs, \
             patch('time.strftime', return_value='20240101-120000'):
            
            result = thread._save_failed_audio(audio_data)
            
            # Should attempt to write file
            mock_write.assert_called_once()
            # Should return empty string due to error
            assert result == ''

    def test_save_failed_audio_directory_structure(self, setup_result_thread):
        """Test that the correct directory structure is created."""
        thread = setup_result_thread
        thread.sample_rate = 44100  # Different sample rate to test flexibility
        
        audio_data = np.random.randint(-32768, 32768, 44100, dtype=np.int16)
        
        with patch('soundfile.write') as mock_write, \
             patch('os.makedirs') as mock_makedirs, \
             patch('time.strftime', return_value='20240315-143022'), \
             patch('os.path.expanduser', return_value='/home/testuser'):
            
            result = thread._save_failed_audio(audio_data)
            
            # Verify correct directory creation
            expected_dir = '/home/testuser/.whisperwriter/failed_audio'
            mock_makedirs.assert_called_once_with(expected_dir, exist_ok=True)
            
            # Verify correct file path
            expected_path = f'{expected_dir}/failed_20240315-143022.flac'
            assert result == expected_path
            
            # Verify sf.write called with correct parameters
            mock_write.assert_called_once()
            call_args = mock_write.call_args[0]
            assert call_args[0] == expected_path
            assert call_args[1] is audio_data
            assert call_args[2] == 44100
            
            # Verify format parameter
            call_kwargs = mock_write.call_args[1]
            assert call_kwargs['format'] == 'FLAC'


class TestRunMethodFailedAudioIntegration:
    """Test integration of failed audio saving within the run method."""

    def test_run_method_saves_audio_on_transcription_failure(self, setup_result_thread):
        """Test that run method saves audio when all transcription attempts fail."""
        thread = setup_result_thread
        
        # Mock _record_audio to return valid audio data
        valid_audio = np.random.randint(-32768, 32768, 16000, dtype=np.int16)
        with patch.object(thread, '_record_audio', return_value=valid_audio), \
             patch.object(thread, '_save_failed_audio', return_value='/test/path/failed.flac') as mock_save, \
             patch('transcription.transcribe', return_value=''):  # Always fail transcription
            
            statuses = []
            thread.statusSignal = types.SimpleNamespace(emit=lambda status, use_llm: statuses.append(status))
            thread.resultSignal = types.SimpleNamespace(emit=lambda result: None)
            
            thread.run()
            
            # Should have attempted to save failed audio
            mock_save.assert_called_once_with(valid_audio)
            
            # Should emit transcription_failed status
            assert 'transcription_failed' in statuses

    def test_run_method_handles_failed_audio_save_failure(self, setup_result_thread):
        """Test that run method handles when audio saving also fails."""
        thread = setup_result_thread
        
        # Mock _record_audio to return valid audio data
        valid_audio = np.random.randint(-32768, 32768, 16000, dtype=np.int16)
        with patch.object(thread, '_record_audio', return_value=valid_audio), \
             patch.object(thread, '_save_failed_audio', return_value='') as mock_save, \
             patch('transcription.transcribe', return_value=''):  # Always fail transcription
            
            console_messages = []
            
            # Capture console messages
            def capture_console(msg):
                console_messages.append(msg)
            
            with patch('utils.ConfigManager.console_print', capture_console):
                thread.run()
            
            # Should have attempted to save failed audio
            mock_save.assert_called_once_with(valid_audio)
            
            # Should show appropriate failure message
            failure_messages = [msg for msg in console_messages if 'failed to save audio file' in msg.lower()]
            assert len(failure_messages) > 0

    def test_run_method_does_not_save_audio_on_successful_transcription(self, monkeypatch):
        """Test that run method does not save audio when transcription succeeds."""
        # Fresh setup without the fixture to avoid mock conflicts
        monkeypatch.setitem(sys.modules, 'sounddevice', types.SimpleNamespace(InputStream=None))
        monkeypatch.setitem(sys.modules, 'webrtcvad', types.SimpleNamespace(Vad=lambda mode: None))
        monkeypatch.setitem(sys.modules, 'soundfile', MagicMock())

        class DummyMediaController:
            def __init__(self):
                self.was_playing = False
            def pause_media(self):
                pass
            def resume_media(self):
                pass

        monkeypatch.setitem(sys.modules, 'media_controller', types.SimpleNamespace(MediaController=DummyMediaController))
        
        # Mock transcription to return success
        transcribe_mock = MagicMock(return_value='successful transcription')
        monkeypatch.setitem(sys.modules, 'transcription', types.SimpleNamespace(transcribe=transcribe_mock))

        # Mock ConfigManager
        class MockConfigManager:
            @staticmethod
            def console_print(msg):
                pass  # Silent for tests
            
            @staticmethod
            def get_config_value(section, key, default=None):
                return False
            
            @staticmethod
            def get_config_section(section):
                return {'sample_rate': 16000}

        monkeypatch.setitem(sys.modules, 'utils', types.SimpleNamespace(ConfigManager=MockConfigManager))

        # Clean import for fresh state
        if 'result_thread' in sys.modules:
            del sys.modules['result_thread']

        sys.path.insert(0, 'src')
        from result_thread import ResultThread

        thread = ResultThread()
        
        # Mock _record_audio to return valid audio data
        valid_audio = np.random.randint(-32768, 32768, 16000, dtype=np.int16)
        
        with patch.object(thread, '_record_audio', return_value=valid_audio), \
             patch.object(thread, '_save_failed_audio') as mock_save:
            
            results = []
            thread.statusSignal = types.SimpleNamespace(emit=lambda status, use_llm: None)
            thread.resultSignal = types.SimpleNamespace(emit=lambda result: results.append(result))
            
            thread.run()
            
            # Should NOT have attempted to save failed audio
            mock_save.assert_not_called()
            
            # Should have emitted successful result
            assert results == ['successful transcription']
            
        sys.path.pop(0)