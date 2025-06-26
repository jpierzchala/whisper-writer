import sys
import types
import pytest
from unittest.mock import MagicMock, patch

@pytest.fixture(autouse=True)
def cleanup_modules():
    """Clean up sys.modules before and after each test to avoid conflicts."""
    # Store original modules
    original_modules = {}
    modules_to_clean = ['utils', 'keyring_manager', 'llm_processor']
    
    for module in modules_to_clean:
        if module in sys.modules:
            original_modules[module] = sys.modules[module]
    
    yield  # Run the test
    
    # Restore original modules and clean up mocked ones
    for module in modules_to_clean:
        if module in original_modules:
            sys.modules[module] = original_modules[module]
        elif module in sys.modules:
            del sys.modules[module]

def test_azure_openai_llm_processor_initialization():
    """Test that Azure OpenAI LLM processor initializes correctly."""
    
    sys.path.insert(0, 'src')
    
    with patch('llm_processor.ConfigManager') as mock_config, \
         patch('llm_processor.KeyringManager') as mock_keyring:
        
        # Mock configuration
        mock_config.get_config_section.return_value = {
            'api_type': 'azure_openai',
            'temperature': 0.3,
            'enabled': True
        }
        
        def mock_get_config_value(section, key):
            if section == 'llm_post_processing':
                if key == 'azure_openai_llm_endpoint':
                    return 'https://test.openai.azure.com'
                elif key == 'azure_openai_llm_api_version':
                    return '2024-02-01'
                elif key == 'azure_openai_llm_deployment_name':
                    return 'gpt-4o-deployment'
                elif key == 'cleanup_model':
                    return 'gpt-4o-mini'
                elif key == 'instruction_model':
                    return 'gpt-4o-mini'
            return None
        
        mock_config.get_config_value.side_effect = mock_get_config_value
        mock_config.console_print = lambda x: None
        mock_keyring.get_api_key.return_value = "test-azure-llm-key"
        
        from llm_processor import LLMProcessor
        
        processor = LLMProcessor(api_type='azure_openai')
        
        assert processor.api_type == 'azure_openai'
        assert processor.api_key == 'test-azure-llm-key'

def test_azure_openai_llm_text_processing():
    """Test Azure OpenAI LLM text processing functionality."""
    
    sys.path.insert(0, 'src')
    
    with patch('llm_processor.ConfigManager') as mock_config, \
         patch('llm_processor.KeyringManager') as mock_keyring, \
         patch('llm_processor.requests.post') as mock_post:
        
        # Mock configuration
        mock_config.get_config_section.return_value = {
            'api_type': 'azure_openai',
            'temperature': 0.3,
            'enabled': True
        }
        
        def mock_get_config_value(section, key):
            if section == 'llm_post_processing':
                if key == 'azure_openai_llm_endpoint':
                    return 'https://test.openai.azure.com'
                elif key == 'azure_openai_llm_api_version':
                    return '2024-02-01'
                elif key == 'azure_openai_llm_deployment_name':
                    return 'gpt-4o-deployment'
                elif key == 'cleanup_model':
                    return 'gpt-4o-mini'
                elif key == 'instruction_model':
                    return 'gpt-4o-mini'
            return None
        
        mock_config.get_config_value.side_effect = mock_get_config_value
        mock_config.console_print = lambda x: None
        mock_keyring.get_api_key.return_value = "test-azure-llm-key"
        
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': 'Cleaned up text from Azure OpenAI'
                }
            }]
        }
        mock_post.return_value = mock_response
        
        from llm_processor import LLMProcessor
        
        processor = LLMProcessor(api_type='azure_openai')
        
        result = processor.process_text(
            "test text to clean", 
            "You are a helpful assistant that cleans up text."
        )
        
        assert result == 'Cleaned up text from Azure OpenAI'
        
        # Verify the API was called with correct parameters
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        
        # Check Azure OpenAI URL structure
        assert 'test.openai.azure.com' in call_args[0][0]
        assert 'gpt-4o-deployment' in call_args[0][0]
        assert 'chat/completions' in call_args[0][0]
        assert 'api-version=2024-02-01' in call_args[0][0]
        
        # Check headers
        assert call_args[1]['headers']['api-key'] == 'test-azure-llm-key'
        assert call_args[1]['headers']['Content-Type'] == 'application/json'
        
        # Check request body
        request_data = call_args[1]['json']
        assert len(request_data['messages']) == 2
        assert request_data['messages'][0]['role'] == 'system'
        assert request_data['messages'][1]['role'] == 'user'
        assert request_data['messages'][1]['content'] == 'test text to clean'

def test_azure_openai_llm_missing_credentials():
    """Test Azure OpenAI LLM processor handles missing credentials gracefully."""
    
    sys.path.insert(0, 'src')
    
    with patch('llm_processor.ConfigManager') as mock_config, \
         patch('llm_processor.KeyringManager') as mock_keyring:
        
        mock_config.get_config_section.return_value = {
            'api_type': 'azure_openai',
            'temperature': 0.3,
            'enabled': True
        }
        
        mock_config.get_config_value.return_value = None  # No configuration values
        mock_config.console_print = lambda x: None
        mock_keyring.get_api_key.return_value = ""  # No API key
        
        from llm_processor import LLMProcessor
        
        processor = LLMProcessor(api_type='azure_openai')
        
        result = processor.process_text(
            "test text", 
            "system message"
        )
        
        # Should return original text when credentials are missing
        assert result == "test text"

def test_azure_openai_llm_missing_endpoint():
    """Test Azure OpenAI LLM processor handles missing endpoint gracefully."""
    
    sys.path.insert(0, 'src')
    
    with patch('llm_processor.ConfigManager') as mock_config, \
         patch('llm_processor.KeyringManager') as mock_keyring:
        
        mock_config.get_config_section.return_value = {
            'api_type': 'azure_openai',
            'temperature': 0.3,
            'enabled': True
        }
        
        def mock_get_config_value(section, key):
            if section == 'llm_post_processing':
                # Missing endpoint
                if key == 'azure_openai_llm_api_version':
                    return '2024-02-01'
                elif key == 'azure_openai_llm_deployment_name':
                    return 'gpt-4o-deployment'
            return None
        
        mock_config.get_config_value.side_effect = mock_get_config_value
        mock_config.console_print = lambda x: None
        mock_keyring.get_api_key.return_value = "test-key"
        
        from llm_processor import LLMProcessor
        
        processor = LLMProcessor(api_type='azure_openai')
        
        result = processor.process_text(
            "test text", 
            "system message"
        )
        
        # Should return original text when endpoint is missing
        assert result == "test text"

def test_azure_openai_llm_api_error():
    """Test Azure OpenAI LLM processor handles API errors gracefully."""
    
    sys.path.insert(0, 'src')
    
    with patch('llm_processor.ConfigManager') as mock_config, \
         patch('llm_processor.KeyringManager') as mock_keyring, \
         patch('llm_processor.requests.post') as mock_post:
        
        mock_config.get_config_section.return_value = {
            'api_type': 'azure_openai',
            'temperature': 0.3,
            'enabled': True
        }
        
        def mock_get_config_value(section, key):
            if section == 'llm_post_processing':
                if key == 'azure_openai_llm_endpoint':
                    return 'https://test.openai.azure.com'
                elif key == 'azure_openai_llm_api_version':
                    return '2024-02-01'
                elif key == 'azure_openai_llm_deployment_name':
                    return 'gpt-4o-deployment'
            return None
        
        mock_config.get_config_value.side_effect = mock_get_config_value
        mock_config.console_print = lambda x: None
        mock_keyring.get_api_key.return_value = "test-key"
        
        # Mock requests.post to return error
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = 'Unauthorized'
        mock_post.return_value = mock_response
        
        from llm_processor import LLMProcessor
        
        processor = LLMProcessor(api_type='azure_openai')
        
        result = processor.process_text(
            "test text", 
            "system message"
        )
        
        # Should return original text when API returns error
        assert result == "test text"
