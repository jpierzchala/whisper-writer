import sys
import types
import pytest
from unittest.mock import MagicMock, patch

def test_azure_openai_provider_configuration():
    """Test that Azure OpenAI provider is properly configured in transcription module."""
    
    # Mock dependencies
    sys.modules['sounddevice'] = types.SimpleNamespace(InputStream=None)
    sys.modules['webrtcvad'] = types.SimpleNamespace(Vad=lambda mode: None)
    sys.modules['soundfile'] = MagicMock()
    
    class MockConfigManager:
        @staticmethod
        def console_print(msg):
            print(f"[TEST LOG] {msg}")
            
        @staticmethod 
        def get_config_section(section):
            if section == 'model_options':
                return {
                    'api': {
                        'provider': 'azure_openai',
                        'model': 'whisper-1',
                        'azure_openai_endpoint': 'https://test.openai.azure.com',
                        'azure_openai_deployment_name': 'whisper-deployment',
                        'azure_openai_api_version': '2024-02-01'
                    }
                }
            return {}
    
    class MockKeyringManager:
        @staticmethod
        def get_api_key(service_name):
            if service_name == "azure_openai_transcription":
                return "test-api-key"
            return ""
    
    sys.modules['utils'] = types.SimpleNamespace(ConfigManager=MockConfigManager)
    sys.modules['keyring_manager'] = types.SimpleNamespace(KeyringManager=MockKeyringManager)
    
    # Import after mocking
    sys.path.insert(0, 'src')
    from transcription import transcribe_api
    
    # Mock requests.post to simulate API response
    with patch('transcription.requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'text': 'Test transcription'}
        mock_post.return_value = mock_response
        
        # Test with dummy audio data
        import numpy as np
        audio_data = np.array([0.1, 0.2, 0.1], dtype=np.float32)
        
        result = transcribe_api(audio_data)
        
        # Verify the result
        assert result == 'Test transcription'
        
        # Verify the API was called with correct parameters
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        
        # Check that the URL contains the Azure OpenAI endpoint structure
        assert 'test.openai.azure.com' in call_args[0][0]
        assert 'whisper-deployment' in call_args[0][0]
        assert 'audio/transcriptions' in call_args[0][0]
        
        # Check headers contain api-key
        assert 'api-key' in call_args[1]['headers']
        assert call_args[1]['headers']['api-key'] == 'test-api-key'
        
        # Check params contain api-version
        assert 'api-version' in call_args[1]['params']
        assert call_args[1]['params']['api-version'] == '2024-02-01'

def test_azure_openai_missing_credentials():
    """Test Azure OpenAI provider handles missing credentials gracefully."""
    
    class MockConfigManager:
        @staticmethod
        def console_print(msg):
            print(f"[TEST LOG] {msg}")
            
        @staticmethod 
        def get_config_section(section):
            if section == 'model_options':
                return {
                    'api': {
                        'provider': 'azure_openai',
                        'model': 'whisper-1',
                    }
                }
            return {}
    
    class MockKeyringManager:
        @staticmethod
        def get_api_key(service_name):
            return ""  # No API key
    
    sys.modules['utils'] = types.SimpleNamespace(ConfigManager=MockConfigManager)
    sys.modules['keyring_manager'] = types.SimpleNamespace(KeyringManager=MockKeyringManager)
    
    # Import after mocking
    sys.path.insert(0, 'src')
    from transcription import transcribe_api
    
    # Test with dummy audio data
    import numpy as np
    audio_data = np.array([0.1, 0.2, 0.1], dtype=np.float32)
    
    result = transcribe_api(audio_data)
    
    # Should return empty string when credentials are missing
    assert result == ''