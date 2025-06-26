import sys
import types
import os
import tempfile
from unittest.mock import MagicMock, patch
import numpy as np

# Simple test without pytest for now
def test_save_failed_audio_with_valid_data():
    """Test that valid audio data is saved successfully"""
    
    # Mock dependencies
    sys.modules['sounddevice'] = types.SimpleNamespace(InputStream=None)
    sys.modules['webrtcvad'] = types.SimpleNamespace(Vad=lambda mode: None)
    sys.modules['soundfile'] = MagicMock()
    
    class DummyMediaController:
        def __init__(self):
            self.was_playing = False
        def pause_media(self):
            pass
        def resume_media(self):
            pass
    
    sys.modules['media_controller'] = types.SimpleNamespace(MediaController=DummyMediaController)
    
    # Mock transcription
    sys.modules['transcription'] = types.SimpleNamespace(transcribe=MagicMock(return_value=''))
    
    # Mock ConfigManager
    class MockConfigManager:
        @staticmethod
        def console_print(msg):
            print(f"[TEST LOG] {msg}")
        
        @staticmethod
        def get_config_value(section, key, default=None):
            return False
        
        @staticmethod
        def get_config_section(section):
            return {'sample_rate': 16000}
    
    sys.modules['utils'] = types.SimpleNamespace(ConfigManager=MockConfigManager)
    
    # Import after mocking
    sys.path.insert(0, 'src')
    from result_thread import ResultThread
    
    # Create thread and set up
    thread = ResultThread()
    thread.sample_rate = 16000
    
    # Create valid audio data
    audio_data = np.random.randint(-32768, 32768, 16000, dtype=np.int16)
    
    # Mock sf.write to simulate successful save
    with patch('soundfile.write') as mock_write:
        with patch('os.makedirs') as mock_makedirs:
            with patch('time.strftime', return_value='20240101-120000'):
                result = thread._save_failed_audio(audio_data)
                
                # Check that sf.write was called
                assert mock_write.called
                # Check that a file path was returned
                assert result.endswith('failed_20240101-120000.flac')
                assert 'failed_audio' in result
    
    print("✓ test_save_failed_audio_with_valid_data passed")


def test_save_failed_audio_with_none_data():
    """Test that None audio data is handled properly"""
    
    # Mock dependencies (same as above)
    sys.modules['sounddevice'] = types.SimpleNamespace(InputStream=None)
    sys.modules['webrtcvad'] = types.SimpleNamespace(Vad=lambda mode: None)
    sys.modules['soundfile'] = MagicMock()
    
    class DummyMediaController:
        def __init__(self):
            self.was_playing = False
        def pause_media(self):
            pass
        def resume_media(self):
            pass
    
    sys.modules['media_controller'] = types.SimpleNamespace(MediaController=DummyMediaController)
    sys.modules['transcription'] = types.SimpleNamespace(transcribe=MagicMock(return_value=''))
    
    # Mock ConfigManager
    class MockConfigManager:
        @staticmethod
        def console_print(msg):
            print(f"[TEST LOG] {msg}")
        
        @staticmethod
        def get_config_value(section, key, default=None):
            return False
        
        @staticmethod
        def get_config_section(section):
            return {'sample_rate': 16000}
    
    sys.modules['utils'] = types.SimpleNamespace(ConfigManager=MockConfigManager)
    
    # Import after mocking
    if 'result_thread' in sys.modules:
        del sys.modules['result_thread']
    
    from result_thread import ResultThread
    
    # Create thread and set up
    thread = ResultThread()
    thread.sample_rate = 16000
    
    # Test with None audio data
    with patch('soundfile.write') as mock_write:
        result = thread._save_failed_audio(None)
        
        # Check that sf.write was NOT called
        assert not mock_write.called
        # Check that empty string was returned
        assert result == ''
    
    print("✓ test_save_failed_audio_with_none_data passed")


def test_save_failed_audio_with_empty_data():
    """Test that empty audio data is handled properly"""
    
    # Mock dependencies (same as above)
    sys.modules['sounddevice'] = types.SimpleNamespace(InputStream=None)
    sys.modules['webrtcvad'] = types.SimpleNamespace(Vad=lambda mode: None)
    sys.modules['soundfile'] = MagicMock()
    
    class DummyMediaController:
        def __init__(self):
            self.was_playing = False
        def pause_media(self):
            pass
        def resume_media(self):
            pass
    
    sys.modules['media_controller'] = types.SimpleNamespace(MediaController=DummyMediaController)
    sys.modules['transcription'] = types.SimpleNamespace(transcribe=MagicMock(return_value=''))
    
    # Mock ConfigManager
    class MockConfigManager:
        @staticmethod
        def console_print(msg):
            print(f"[TEST LOG] {msg}")
        
        @staticmethod
        def get_config_value(section, key, default=None):
            return False
        
        @staticmethod
        def get_config_section(section):
            return {'sample_rate': 16000}
    
    sys.modules['utils'] = types.SimpleNamespace(ConfigManager=MockConfigManager)
    
    # Import after mocking
    if 'result_thread' in sys.modules:
        del sys.modules['result_thread']
    
    from result_thread import ResultThread
    
    # Create thread and set up
    thread = ResultThread()
    thread.sample_rate = 16000
    
    # Test with empty audio data
    empty_audio = np.array([], dtype=np.int16)
    
    with patch('soundfile.write') as mock_write:
        result = thread._save_failed_audio(empty_audio)
        
        # Check that sf.write was NOT called
        assert not mock_write.called
        # Check that empty string was returned
        assert result == ''
    
    print("✓ test_save_failed_audio_with_empty_data passed")


def test_save_failed_audio_with_no_sample_rate():
    """Test that missing sample rate is handled properly"""
    
    # Mock dependencies (same as above)
    sys.modules['sounddevice'] = types.SimpleNamespace(InputStream=None)
    sys.modules['webrtcvad'] = types.SimpleNamespace(Vad=lambda mode: None)
    sys.modules['soundfile'] = MagicMock()
    
    class DummyMediaController:
        def __init__(self):
            self.was_playing = False
        def pause_media(self):
            pass
        def resume_media(self):
            pass
    
    sys.modules['media_controller'] = types.SimpleNamespace(MediaController=DummyMediaController)
    sys.modules['transcription'] = types.SimpleNamespace(transcribe=MagicMock(return_value=''))
    
    # Mock ConfigManager
    class MockConfigManager:
        @staticmethod
        def console_print(msg):
            print(f"[TEST LOG] {msg}")
        
        @staticmethod
        def get_config_value(section, key, default=None):
            return False
        
        @staticmethod
        def get_config_section(section):
            return {'sample_rate': 16000}
    
    sys.modules['utils'] = types.SimpleNamespace(ConfigManager=MockConfigManager)
    
    # Import after mocking
    if 'result_thread' in sys.modules:
        del sys.modules['result_thread']
    
    from result_thread import ResultThread
    
    # Create thread without setting sample_rate
    thread = ResultThread()
    # Don't set sample_rate
    
    # Create valid audio data
    audio_data = np.random.randint(-32768, 32768, 16000, dtype=np.int16)
    
    with patch('soundfile.write') as mock_write:
        result = thread._save_failed_audio(audio_data)
        
        # Check that sf.write was NOT called
        assert not mock_write.called
        # Check that empty string was returned
        assert result == ''
    
    print("✓ test_save_failed_audio_with_no_sample_rate passed")


if __name__ == '__main__':
    print("Running failed audio saving tests...")
    test_save_failed_audio_with_valid_data()
    test_save_failed_audio_with_none_data()
    test_save_failed_audio_with_empty_data()
    test_save_failed_audio_with_no_sample_rate()
    print("All tests passed!")