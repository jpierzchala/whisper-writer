import sys
import types
from unittest.mock import MagicMock, patch

def test_save_failed_audio_validation():
    """Test that audio data validation works correctly"""
    
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
    sys.modules['transcription'] = types.SimpleNamespace(transcribe=MagicMock(return_value=''))
    
    # Mock ConfigManager
    messages = []
    class MockConfigManager:
        @staticmethod
        def console_print(msg):
            messages.append(msg)
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
    
    # Test 1: None audio data
    print("Test 1: None audio data")
    messages.clear()
    thread = ResultThread()
    thread.sample_rate = 16000
    result = thread._save_failed_audio(None)
    assert result == '', f"Expected empty string, got: {result}"
    assert any('audio_data is None' in msg for msg in messages), f"Expected None message, got: {messages}"
    print("✓ None audio data test passed")
    
    # Test 2: Empty audio data
    print("Test 2: Empty audio data")
    messages.clear()
    result = thread._save_failed_audio([])
    assert result == '', f"Expected empty string, got: {result}"
    assert any('audio_data is empty' in msg for msg in messages), f"Expected empty message, got: {messages}"
    print("✓ Empty audio data test passed")
    
    # Test 3: No sample rate
    print("Test 3: No sample rate")
    messages.clear()
    thread_no_rate = ResultThread()
    # Don't set sample_rate
    result = thread_no_rate._save_failed_audio([1, 2, 3])
    assert result == '', f"Expected empty string, got: {result}"
    assert any('sample_rate is not set' in msg for msg in messages), f"Expected sample_rate message, got: {messages}"
    print("✓ No sample rate test passed")
    
    # Test 4: Valid data (mocked save)
    print("Test 4: Valid data")
    messages.clear()
    thread.sample_rate = 16000
    valid_audio = [1, 2, 3, 4, 5]  # Simple list representing audio data
    
    with patch('soundfile.write') as mock_write:
        with patch('os.makedirs') as mock_makedirs:
            with patch('time.strftime', return_value='20240101-120000'):
                result = thread._save_failed_audio(valid_audio)
                
                # Should return a file path
                assert result != '', f"Expected non-empty path, got: {result}"
                assert 'failed_20240101-120000.flac' in result, f"Expected filename in path: {result}"
                assert mock_write.called, "Expected sf.write to be called"
                
                # Check for success message
                assert any('Successfully saved' in msg for msg in messages), f"Expected success message, got: {messages}"
    
    print("✓ Valid data test passed")
    
    print("All validation tests passed!")

if __name__ == '__main__':
    print("Running simplified failed audio saving tests...")
    test_save_failed_audio_validation()
    print("All tests completed successfully!")