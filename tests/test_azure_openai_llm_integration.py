import sys
import os
import pytest
from unittest.mock import patch, MagicMock

def test_azure_openai_llm_config_schema():
    """Test that Azure OpenAI LLM configuration is properly defined in schema."""
    
    sys.path.insert(0, 'src')
    
    with patch('utils.ConfigManager') as mock_config:
        # Mock the schema structure
        mock_schema = {
            'llm_post_processing': {
                'api_type': {
                    'options': ['openai', 'azure_openai', 'anthropic']
                },
                'azure_openai_llm_api_key': {'type': 'str'},
                'azure_openai_llm_endpoint': {'type': 'str'},
                'azure_openai_llm_deployment_name': {'type': 'str'},
                'azure_openai_llm_api_version': {'type': 'str'}
            }
        }
        mock_config.get_schema.return_value = mock_schema
        
        from utils import ConfigManager
        
        # Load the configuration schema
        schema = ConfigManager.get_schema()
        
        # Verify Azure OpenAI is in LLM post-processing options
        assert 'llm_post_processing' in schema
        llm_section = schema['llm_post_processing']
        
        # Check that azure_openai is in api_type options
        assert 'api_type' in llm_section
        api_type_options = llm_section['api_type']['options']
        assert 'azure_openai' in api_type_options
        
        # Check Azure OpenAI specific configuration fields exist
        required_fields = [
            'azure_openai_llm_api_key',
            'azure_openai_llm_endpoint', 
            'azure_openai_llm_deployment_name',
            'azure_openai_llm_api_version'
        ]
        
        for field in required_fields:
            assert field in llm_section, f"Missing required field: {field}"
            assert llm_section[field]['type'] == 'str'

def test_azure_openai_llm_processor_import():
    """Test that LLMProcessor can be imported and recognizes azure_openai."""
    
    sys.path.insert(0, 'src')
    
    # Mock dependencies to avoid import errors
    with patch('llm_processor.KeyringManager') as mock_keyring, \
         patch('llm_processor.ConfigManager') as mock_config:
        
        mock_config.get_config_section.return_value = {
            'api_type': 'azure_openai',
            'enabled': True,
            'temperature': 0.3
        }
        mock_keyring.get_api_key.return_value = "test-key"
        
        from llm_processor import LLMProcessor
        
        # Test that processor can be initialized with azure_openai
        processor = LLMProcessor(api_type='azure_openai')
        assert processor.api_type == 'azure_openai'

def test_azure_openai_llm_api_call_structure():
    """Test that Azure OpenAI LLM makes correctly structured API calls."""
    
    sys.path.insert(0, 'src')
    
    with patch('llm_processor.ConfigManager') as mock_config, \
         patch('llm_processor.KeyringManager') as mock_keyring, \
         patch('llm_processor.requests.post') as mock_requests:
        
        # Mock configuration
        mock_config.get_config_section.return_value = {
            'api_type': 'azure_openai',
            'enabled': True,
            'temperature': 0.3
        }
        
        def mock_get_config_value(section, key):
            config_map = {
                ('llm_post_processing', 'azure_openai_llm_endpoint'): 'https://test.openai.azure.com',
                ('llm_post_processing', 'azure_openai_llm_api_version'): '2024-02-01',
                ('llm_post_processing', 'azure_openai_llm_deployment_name'): 'gpt-4o-deployment',
                ('llm_post_processing', 'cleanup_model'): 'gpt-4o-mini'
            }
            return config_map.get((section, key))
        
        mock_config.get_config_value.side_effect = mock_get_config_value
        mock_config.console_print = lambda x: None
        
        # Mock keyring
        mock_keyring.get_api_key.return_value = "test-azure-llm-key"
        
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': 'Processed text'
                }
            }]
        }
        mock_requests.return_value = mock_response
        
        from llm_processor import LLMProcessor
        
        processor = LLMProcessor(api_type='azure_openai')
        
        # Test the process_text method
        result = processor.process_text("test text", "system message")
        
        # Verify API was called
        assert mock_requests.called
        call_args = mock_requests.call_args
        
        # Verify URL structure
        url = call_args[0][0]
        assert 'test.openai.azure.com' in url
        assert 'gpt-4o-deployment' in url
        assert 'chat/completions' in url
        assert 'api-version=2024-02-01' in url
        
        # Verify headers
        headers = call_args[1]['headers']
        assert headers['api-key'] == 'test-azure-llm-key'
        assert headers['Content-Type'] == 'application/json'
        
        # Verify request body structure
        request_data = call_args[1]['json']
        assert 'messages' in request_data
        assert len(request_data['messages']) == 2
        assert request_data['messages'][0]['role'] == 'system'
        assert request_data['messages'][1]['role'] == 'user'
        
        # Verify result
        assert result == 'Processed text'

def test_azure_openai_llm_missing_config_handling():
    """Test handling of missing Azure OpenAI LLM configuration."""
    
    sys.path.insert(0, 'src')
    
    with patch('llm_processor.KeyringManager') as mock_keyring, \
         patch('llm_processor.ConfigManager') as mock_config:
        
        mock_config.get_config_section.return_value = {
            'api_type': 'azure_openai',
            'enabled': True,
            'temperature': 0.3
        }
        
        # Mock missing configuration
        mock_config.get_config_value.return_value = None
        mock_config.console_print = lambda x: None
        mock_keyring.get_api_key.return_value = "test-key"
        
        from llm_processor import LLMProcessor
        
        processor = LLMProcessor(api_type='azure_openai')
        
        # Should return original text when config is missing
        result = processor.process_text("test text", "system message")
        assert result == "test text"

def test_keyring_azure_openai_llm_key_handling():
    """Test keyring handling of Azure OpenAI LLM keys."""
    
    # Clean up any existing module mocks
    for module in ['keyring_manager', 'utils']:
        if module in sys.modules:
            del sys.modules[module]
    
    sys.path.insert(0, 'src')
    
    with patch.dict('sys.modules', {}, clear=False):
        with patch('keyring_manager.keyring') as mock_keyring, \
             patch('keyring_manager.ConfigManager') as mock_config:
            
            mock_config.console_print = lambda x: None
            
            from keyring_manager import KeyringManager
            
            # Test saving key
            KeyringManager.save_api_key("azure_openai_llm", "test-azure-key")
            mock_keyring.set_password.assert_called_with(
                "whisperwriter", 
                "azure_openai_llm", 
                "test-azure-key"
            )
            
            # Test retrieving key
            mock_keyring.get_password.return_value = "test-azure-key"
            result = KeyringManager.get_api_key("azure_openai_llm")
            assert result == "test-azure-key"
            
            # Test empty key deletion
            mock_keyring.reset_mock()
            KeyringManager.save_api_key("azure_openai_llm", "")
            mock_keyring.delete_password.assert_called_with(
                "whisperwriter", 
                "azure_openai_llm"
            )
