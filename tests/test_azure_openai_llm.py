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
        mock_config.console_print = lambda *args, **kwargs: None
        mock_keyring.get_api_key.return_value = "test-azure-llm-key"
        
        from llm_processor import LLMProcessor
        
        processor = LLMProcessor(api_type='azure_openai')
        
        assert processor.api_type == 'azure_openai'
        assert processor.api_key == 'test-azure-llm-key'

def test_azure_openai_llm_text_processing():
    """A non-reasoning Azure deployment goes through Chat Completions."""

    sys.path.insert(0, 'src')

    with patch('llm_processor.ConfigManager') as mock_config, \
         patch('llm_processor.KeyringManager') as mock_keyring, \
         patch('llm_processor.build_azure_client') as mock_build_client:

        mock_config.get_config_section.return_value = {
            'api_type': 'azure_openai',
            'temperature': 0.3,
            'enabled': True
        }

        def mock_get_config_value(section, key):
            if section == 'llm_post_processing':
                return {
                    'azure_openai_llm_endpoint': 'https://test.openai.azure.com',
                    'azure_openai_llm_api_version': 'v1',
                    'azure_api_mode': 'v1',
                    'azure_openai_llm_cleanup_deployment_name': 'gpt-4o-deployment',
                    # The deployment runs a classic chat model, not a reasoning one.
                    'azure_openai_llm_cleanup_model_family': 'chat',
                    'cleanup_model': 'gpt-4o-mini',
                    'instruction_model': 'gpt-4o-mini',
                }.get(key)
            return None

        mock_config.get_config_value.side_effect = mock_get_config_value
        mock_config.console_print = lambda *args, **kwargs: None
        mock_config.should_log_cleanup_prompt.return_value = False
        mock_keyring.get_api_key.return_value = "test-azure-llm-key"

        completion = MagicMock()
        completion.output_text = None
        completion.output = None
        completion.choices = [MagicMock()]
        completion.choices[0].message.refusal = None
        completion.choices[0].message.content = 'Cleaned up text from Azure OpenAI'

        client = MagicMock()
        client.chat.completions.create.return_value = completion
        mock_build_client.return_value = client

        from llm_processor import LLMProcessor

        processor = LLMProcessor(api_type='azure_openai')

        result = processor.process_text(
            "test text to clean",
            "You are a helpful assistant that cleans up text."
        )

        assert result == 'Cleaned up text from Azure OpenAI'

        # The client is built from the endpoint, key and API mode...
        build_kwargs = mock_build_client.call_args
        assert build_kwargs.args[0] == 'test-azure-llm-key'
        assert build_kwargs.args[1] == 'https://test.openai.azure.com'
        assert build_kwargs.kwargs['api_mode'] == 'v1'

        # ...and the request carries the deployment name as the model.
        client.chat.completions.create.assert_called_once()
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs['model'] == 'gpt-4o-deployment'
        assert len(kwargs['messages']) == 2
        assert kwargs['messages'][0]['role'] == 'system'
        assert kwargs['messages'][1]['role'] == 'user'
        assert '<transcript>' in kwargs['messages'][1]['content']
        assert 'test text to clean' in kwargs['messages'][1]['content']
        # Cleanup is an edit, so sampling is off; chat models still accept it.
        assert kwargs['temperature'] == 0.0
        assert kwargs['max_tokens'] >= 1024
        assert 'reasoning_effort' not in kwargs

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
        mock_config.console_print = lambda *args, **kwargs: None
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
        mock_config.console_print = lambda *args, **kwargs: None
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
        mock_config.console_print = lambda *args, **kwargs: None
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
