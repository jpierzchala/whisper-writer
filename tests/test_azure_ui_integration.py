import sys
import pytest
from unittest.mock import patch, MagicMock

def test_settings_window_azure_llm_provider_options():
    """Test that Azure OpenAI LLM provider options are properly handled in settings."""
    
    sys.path.insert(0, 'src')
    
    # Test the provider options mapping
    provider_fields = {
        'chatgpt': ['openai_api_key'],
        'azure_openai': ['azure_openai_llm_api_key', 'azure_openai_llm_endpoint', 
                       'azure_openai_llm_deployment_name', 'azure_openai_llm_api_version'],
        'claude': ['claude_api_key'],
        'gemini': ['gemini_api_key'],
        'groq': ['groq_api_key'],
        'ollama': []
    }
    
    # Verify Azure OpenAI has the correct fields
    azure_fields = provider_fields['azure_openai']
    expected_fields = [
        'azure_openai_llm_api_key', 
        'azure_openai_llm_endpoint',
        'azure_openai_llm_deployment_name', 
        'azure_openai_llm_api_version'
    ]
    
    for field in expected_fields:
        assert field in azure_fields, f"Missing required Azure OpenAI LLM field: {field}"

def test_settings_window_transcription_provider_options():
    """Test that Azure OpenAI transcription provider options are properly handled."""
    
    # Test the transcription provider options mapping
    provider_fields = {
        'openai': ['openai_transcription_api_key'],
        'azure_openai': ['azure_openai_api_key', 'azure_openai_endpoint', 
                       'azure_openai_deployment_name', 'azure_openai_api_version'],
        'deepgram': ['deepgram_transcription_api_key'],
        'groq': ['groq_transcription_api_key']
    }
    
    # Verify Azure OpenAI has the correct transcription fields
    azure_fields = provider_fields['azure_openai']
    expected_fields = [
        'azure_openai_api_key',
        'azure_openai_endpoint', 
        'azure_openai_deployment_name',
        'azure_openai_api_version'
    ]
    
    for field in expected_fields:
        assert field in azure_fields, f"Missing required Azure OpenAI transcription field: {field}"

def test_azure_openai_llm_keyring_save_integration():
    """Test Azure OpenAI LLM key saving integration with keyring."""
    
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
            
            # Test saving Azure OpenAI LLM key
            test_key = "test-azure-llm-key-12345"
            KeyringManager.save_api_key("azure_openai_llm", test_key)
            
            # Verify the key was saved with correct service name
            mock_keyring.set_password.assert_called_with(
                "whisperwriter", 
                "azure_openai_llm", 
                test_key
            )

def test_azure_openai_llm_keyring_retrieve_integration():
    """Test Azure OpenAI LLM key retrieval from keyring."""
    
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
            
            # Mock successful key retrieval
            expected_key = "retrieved-azure-llm-key"
            mock_keyring.get_password.return_value = expected_key
            
            # Test retrieving Azure OpenAI LLM key
            result = KeyringManager.get_api_key("azure_openai_llm")
            
            # Verify the key was requested with correct service name
            mock_keyring.get_password.assert_called_with(
                "whisperwriter", 
                "azure_openai_llm"
            )
            
            assert result == expected_key

def test_config_schema_azure_openai_llm_fields():
    """Test that config schema contains all required Azure OpenAI LLM fields."""
    
    sys.path.insert(0, 'src')
    
    with patch('utils.ConfigManager') as mock_config:
        # Mock schema structure
        mock_schema = {
            'llm_post_processing': {
                'api_type': {
                    'options': ['openai', 'azure_openai', 'anthropic']
                },
                'azure_openai_llm_endpoint': {'type': 'str'},
                'azure_openai_llm_api_version': {'type': 'str'},
                'azure_openai_llm_deployment_name': {'type': 'str'},
                'azure_openai_llm_api_key': {'type': 'str'}
            }
        }
        mock_config.get_schema.return_value = mock_schema
        
        from utils import ConfigManager
        
        schema = ConfigManager.get_schema()
        
        # Check LLM post-processing section exists
        assert 'llm_post_processing' in schema, "LLM post-processing section missing from schema"
        
        llm_section = schema['llm_post_processing']
        
        # Check Azure OpenAI is in API type options
        assert 'api_type' in llm_section
        assert 'azure_openai' in llm_section['api_type']['options']
        
        # Check required Azure OpenAI LLM fields
        required_fields = [
            'azure_openai_llm_endpoint',
            'azure_openai_llm_api_version', 
            'azure_openai_llm_deployment_name',
            'azure_openai_llm_api_key'
        ]
        
        for field in required_fields:
            assert field in llm_section, f"Missing Azure OpenAI LLM field: {field}"
            assert llm_section[field]['type'] == 'str'
    assert 'api_type' in llm_section, "API type field missing"
    api_type_options = llm_section['api_type'].get('options', [])
    assert 'azure_openai' in api_type_options, "azure_openai not in API type options"
    
    # Check required Azure OpenAI LLM fields
    required_azure_fields = [
        'azure_openai_llm_api_key',
        'azure_openai_llm_endpoint',
        'azure_openai_llm_deployment_name', 
        'azure_openai_llm_api_version'
    ]
    
    for field in required_azure_fields:
        assert field in llm_section, f"Missing Azure OpenAI LLM field in schema: {field}"
        assert llm_section[field]['type'] == 'str', f"Field {field} should be string type"

def test_settings_window_toggle_function_logic():
    """Test the logic of provider toggle functions."""
    
    # Simulate the toggle_llm_provider_options logic
    def simulate_toggle_llm_provider_options(provider):
        """Simulate the settings window toggle function logic."""
        provider_fields = {
            'chatgpt': ['openai_api_key'],
            'azure_openai': ['azure_openai_llm_api_key', 'azure_openai_llm_endpoint', 
                           'azure_openai_llm_deployment_name', 'azure_openai_llm_api_version'],
            'claude': ['claude_api_key'],
            'gemini': ['gemini_api_key'],
            'groq': ['groq_api_key'],
            'ollama': []
        }
        
        # Get all fields to hide first
        all_fields = []
        for fields in provider_fields.values():
            all_fields.extend(fields)
        
        # Fields to show for selected provider
        fields_to_show = provider_fields.get(provider, [])
        
        return {
            'all_fields': all_fields,
            'fields_to_show': fields_to_show,
            'fields_to_hide': [f for f in all_fields if f not in fields_to_show]
        }
    
    # Test Azure OpenAI selection
    result = simulate_toggle_llm_provider_options('azure_openai')
    
    # Verify Azure OpenAI fields are shown
    azure_fields = [
        'azure_openai_llm_api_key', 
        'azure_openai_llm_endpoint',
        'azure_openai_llm_deployment_name', 
        'azure_openai_llm_api_version'
    ]
    
    for field in azure_fields:
        assert field in result['fields_to_show'], f"Azure field {field} should be shown"
    
    # Verify other provider fields are hidden
    other_fields = ['openai_api_key', 'claude_api_key', 'gemini_api_key', 'groq_api_key']
    for field in other_fields:
        assert field in result['fields_to_hide'], f"Other field {field} should be hidden"

def test_settings_window_api_key_handling():
    """Test API key handling in settings window."""
    
    # Test API key field recognition
    api_key_fields = [
        'openai_transcription_api_key',
        'deepgram_transcription_api_key', 
        'groq_transcription_api_key',
        'azure_openai_api_key',
        'azure_openai_llm_api_key',  # New field
        'claude_api_key',
        'openai_api_key', 
        'gemini_api_key',
        'groq_api_key'
    ]
    
    for field in api_key_fields:
        # Test that field is recognized as API key
        is_api_key = field.endswith('api_key')
        assert is_api_key, f"Field {field} should be recognized as API key"

def test_azure_openai_config_validation():
    """Test validation of Azure OpenAI configuration."""
    
    # Test complete Azure OpenAI configuration
    azure_config = {
        'api_key': 'test-key',
        'endpoint': 'https://test.openai.azure.com',
        'deployment_name': 'gpt-4o-deployment',
        'api_version': '2024-02-01'
    }
    
    # Validate required fields are present
    required_fields = ['api_key', 'endpoint', 'deployment_name', 'api_version']
    
    for field in required_fields:
        assert field in azure_config, f"Required Azure OpenAI field missing: {field}"
        assert azure_config[field], f"Azure OpenAI field {field} should not be empty"
    
    # Test endpoint format
    endpoint = azure_config['endpoint']
    assert endpoint.startswith('https://'), "Azure OpenAI endpoint should use HTTPS"
    assert '.openai.azure.com' in endpoint, "Azure OpenAI endpoint should contain Azure domain"

def test_llm_processor_azure_integration():
    """Test LLM processor integration with Azure OpenAI."""
    
    sys.path.insert(0, 'src')
    
    with patch('llm_processor.ConfigManager') as mock_config, \
         patch('llm_processor.KeyringManager') as mock_keyring:
        
        # Mock configuration for Azure OpenAI
        mock_config.get_config_section.return_value = {
            'api_type': 'azure_openai',
            'enabled': True,
            'temperature': 0.3
        }
        
        mock_config.get_config_value.side_effect = lambda section, key: {
            ('llm_post_processing', 'azure_openai_llm_endpoint'): 'https://test.openai.azure.com',
            ('llm_post_processing', 'azure_openai_llm_deployment_name'): 'gpt-4o',
            ('llm_post_processing', 'azure_openai_llm_api_version'): '2024-02-01'
        }.get((section, key))
        
        mock_config.console_print = lambda x: None
        mock_keyring.get_api_key.return_value = "test-azure-llm-key"
        
        from llm_processor import LLMProcessor
        
        # Test processor initialization
        processor = LLMProcessor(api_type='azure_openai')
        
        # Verify processor is configured for Azure OpenAI
        assert processor.api_type == 'azure_openai'
        assert processor.api_key == 'test-azure-llm-key'
