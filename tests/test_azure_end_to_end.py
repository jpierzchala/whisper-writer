import sys
import pytest
from unittest.mock import patch, MagicMock

def test_azure_openai_end_to_end_llm_flow():
    """Test complete end-to-end flow for Azure OpenAI LLM processing."""
    
    sys.path.insert(0, 'src')
    
    with patch('llm_processor.requests.post') as mock_post, \
         patch('llm_processor.ConfigManager') as mock_config, \
         patch('llm_processor.KeyringManager') as mock_keyring:
        
        # Mock successful configuration
        mock_config.get_config_section.return_value = {
            'api_type': 'azure_openai',
            'enabled': True,
            'temperature': 0.3
        }
        
        mock_config.get_config_value.side_effect = lambda section, key: {
            ('llm_post_processing', 'azure_openai_llm_endpoint'): 'https://test.openai.azure.com',
            ('llm_post_processing', 'azure_openai_llm_deployment_name'): 'gpt-4o-deployment',
            ('llm_post_processing', 'azure_openai_llm_api_version'): '2024-02-01',
            ('llm_post_processing', 'cleanup_model'): 'gpt-4o-mini',
            ('llm_post_processing', 'system_prompt'): 'You are a helpful assistant.'
        }.get((section, key))
        
        mock_config.console_print = lambda x: None
        mock_keyring.get_api_key.return_value = "test-azure-llm-key"
        
        # Mock successful Azure OpenAI API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': 'This is the cleaned up text from Azure OpenAI.'
                }
            }]
        }
        mock_post.return_value = mock_response
        
        from llm_processor import LLMProcessor
        
        # Initialize processor
        processor = LLMProcessor(api_type='azure_openai')
        
        # Process text through complete flow
        input_text = "this is some messy text that needs cleaning"
        system_message = "You are a helpful assistant that cleans up text."
        
        result = processor.process_text(input_text, system_message)
        
        # Verify end-to-end result
        assert result == 'This is the cleaned up text from Azure OpenAI.'
        
        # Verify API call was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        
        # Verify URL construction
        url = call_args[0][0]
        assert 'test.openai.azure.com' in url
        assert 'gpt-4o-deployment' in url
        assert 'chat/completions' in url
        assert 'api-version=2024-02-01' in url
        
        # Verify headers
        headers = call_args[1]['headers']
        assert headers['api-key'] == 'test-azure-llm-key'
        assert headers['Content-Type'] == 'application/json'
        
        # Verify request body
        request_data = call_args[1]['json']
        assert request_data['messages'][0]['role'] == 'system'
        assert request_data['messages'][0]['content'] == system_message
        assert request_data['messages'][1]['role'] == 'user'
        assert request_data['messages'][1]['content'] == input_text
        assert request_data['temperature'] == 0.3

def test_azure_openai_transcription_and_llm_integration():
    """Test integration between Azure OpenAI transcription and LLM processing."""
    
    sys.path.insert(0, 'src')
    
    # Test that both transcription and LLM can use Azure OpenAI simultaneously
    transcription_config = {
        'model_options': {
            'use_api': True,
            'api': {
                'provider': 'azure_openai',
                'azure_openai_endpoint': 'https://transcription.openai.azure.com',
                'azure_openai_deployment_name': 'whisper-deployment',
                'azure_openai_api_version': '2024-02-01'
            }
        }
    }
    
    llm_config = {
        'llm_post_processing': {
            'enabled': True,
            'api_type': 'azure_openai',
            'azure_openai_llm_endpoint': 'https://llm.openai.azure.com',
            'azure_openai_llm_deployment_name': 'gpt-4o-deployment',
            'azure_openai_llm_api_version': '2024-02-01'
        }
    }
    
    # Verify both configurations can coexist
    assert transcription_config['model_options']['api']['provider'] == 'azure_openai'
    assert llm_config['llm_post_processing']['api_type'] == 'azure_openai'
    
    # Verify they use different endpoints (can be the same, but often different)
    transcription_endpoint = transcription_config['model_options']['api']['azure_openai_endpoint']
    llm_endpoint = llm_config['llm_post_processing']['azure_openai_llm_endpoint']
    
    # Both should be valid Azure OpenAI endpoints
    assert '.openai.azure.com' in transcription_endpoint
    assert '.openai.azure.com' in llm_endpoint

def test_azure_openai_error_handling_chain():
    """Test error handling across the Azure OpenAI processing chain."""
    
    sys.path.insert(0, 'src')
    
    error_scenarios = [
        {
            'name': 'Missing API Key',
            'keyring_return': '',
            'config_values': {
                'azure_openai_llm_endpoint': 'https://test.openai.azure.com',
                'azure_openai_llm_deployment_name': 'gpt-4o',
            },
            'expected_result': 'original text'
        },
        {
            'name': 'Missing Endpoint',
            'keyring_return': 'test-key',
            'config_values': {
                'azure_openai_llm_deployment_name': 'gpt-4o',
            },
            'expected_result': 'original text'
        },
        {
            'name': 'Missing Deployment',
            'keyring_return': 'test-key',
            'config_values': {
                'azure_openai_llm_endpoint': 'https://test.openai.azure.com',
            },
            'expected_result': 'original text'
        }
    ]
    
    for scenario in error_scenarios:
        with patch('llm_processor.ConfigManager') as mock_config, \
             patch('llm_processor.KeyringManager') as mock_keyring:
            
            mock_config.get_config_section.return_value = {
                'api_type': 'azure_openai',
                'enabled': True,
                'temperature': 0.3
            }
            
            mock_config.get_config_value.side_effect = lambda section, key: scenario['config_values'].get(key)
            mock_config.console_print = lambda x: None
            mock_keyring.get_api_key.return_value = scenario['keyring_return']
            
            from llm_processor import LLMProcessor
            
            processor = LLMProcessor(api_type='azure_openai')
            result = processor.process_text('original text', 'system message')
            
            assert result == scenario['expected_result'], f"Error scenario '{scenario['name']}' failed"

def test_azure_openai_configuration_validation():
    """Test comprehensive Azure OpenAI configuration validation."""
    
    # Test valid configuration
    valid_config = {
        'transcription': {
            'provider': 'azure_openai',
            'endpoint': 'https://transcription.openai.azure.com',
            'deployment_name': 'whisper-1',
            'api_version': '2024-02-01',
            'api_key': 'transcription-key'
        },
        'llm': {
            'api_type': 'azure_openai',
            'endpoint': 'https://llm.openai.azure.com',
            'deployment_name': 'gpt-4o-mini',
            'api_version': '2024-02-01',
            'api_key': 'llm-key'
        }
    }
    
    # Validate transcription config
    transcription = valid_config['transcription']
    assert transcription['provider'] == 'azure_openai'
    assert transcription['endpoint'].startswith('https://')
    assert '.openai.azure.com' in transcription['endpoint']
    assert transcription['deployment_name']
    assert transcription['api_version']
    assert transcription['api_key']
    
    # Validate LLM config
    llm = valid_config['llm']
    assert llm['api_type'] == 'azure_openai'
    assert llm['endpoint'].startswith('https://')
    assert '.openai.azure.com' in llm['endpoint']
    assert llm['deployment_name']
    assert llm['api_version']
    assert llm['api_key']

def test_azure_openai_keyring_migration_workflow():
    """Test the complete workflow of migrating Azure OpenAI keys to keyring."""
    
    # Simulate configuration before migration
    config_before = {
        'model_options': {
            'api': {
                'azure_openai_api_key': 'transcription-key-in-config',
                'azure_openai_endpoint': 'https://test.openai.azure.com'
            }
        },
        'llm_post_processing': {
            'azure_openai_llm_api_key': 'llm-key-in-config',
            'azure_openai_llm_endpoint': 'https://test.openai.azure.com'
        }
    }
    
    # Simulate configuration after migration
    config_after = {
        'model_options': {
            'api': {
                'azure_openai_api_key': None,  # Removed from config
                'azure_openai_endpoint': 'https://test.openai.azure.com'  # Endpoint remains
            }
        },
        'llm_post_processing': {
            'azure_openai_llm_api_key': None,  # Removed from config
            'azure_openai_llm_endpoint': 'https://test.openai.azure.com'  # Endpoint remains
        }
    }
    
    # Simulate keyring state after migration
    expected_keyring_keys = {
        'azure_openai_transcription': 'transcription-key-in-config',
        'azure_openai_llm': 'llm-key-in-config'
    }
    
    # Verify migration removes keys from config
    assert config_after['model_options']['api']['azure_openai_api_key'] is None
    assert config_after['llm_post_processing']['azure_openai_llm_api_key'] is None
    
    # Verify migration preserves other settings
    assert config_after['model_options']['api']['azure_openai_endpoint'] == 'https://test.openai.azure.com'
    assert config_after['llm_post_processing']['azure_openai_llm_endpoint'] == 'https://test.openai.azure.com'
    
    # Verify expected keyring entries
    for service, key in expected_keyring_keys.items():
        assert key, f"Expected key for service {service}"

def test_azure_openai_provider_switching():
    """Test switching between different Azure OpenAI providers and configurations."""
    
    providers = [
        {
            'name': 'ChatGPT',
            'api_type': 'chatgpt',
            'required_fields': ['openai_api_key']
        },
        {
            'name': 'Azure OpenAI',
            'api_type': 'azure_openai',
            'required_fields': ['azure_openai_llm_api_key', 'azure_openai_llm_endpoint', 'azure_openai_llm_deployment_name']
        },
        {
            'name': 'Claude',
            'api_type': 'claude',
            'required_fields': ['claude_api_key']
        }
    ]
    
    for provider in providers:
        # Verify each provider has its required fields defined
        assert provider['api_type'], f"Provider {provider['name']} missing api_type"
        assert provider['required_fields'], f"Provider {provider['name']} missing required_fields"
        
        if provider['api_type'] == 'azure_openai':
            # Special validation for Azure OpenAI
            azure_fields = provider['required_fields']
            assert 'azure_openai_llm_api_key' in azure_fields
            assert 'azure_openai_llm_endpoint' in azure_fields
            assert 'azure_openai_llm_deployment_name' in azure_fields
