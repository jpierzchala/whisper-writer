#!/usr/bin/env python3
"""
Migration script to move Azure OpenAI API key from config.yaml to keyring.
Run this script to fix the "Azure OpenAI API key not found in keyring" issue.
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils import ConfigManager
from keyring_manager import KeyringManager

def migrate_azure_openai_key():
    """Migrate Azure OpenAI API key from config to keyring."""
    print("Checking for Azure OpenAI API key in config...")
    
    # Initialize ConfigManager
    ConfigManager.initialize()
    
    # Get Azure OpenAI key from config
    azure_key = ConfigManager.get_config_value('model_options', 'api', 'azure_openai_api_key')
    
    if azure_key and azure_key.strip():
        print(f"Found Azure OpenAI API key in config: {azure_key[:8]}...")
        
        # Save to keyring
        KeyringManager.save_api_key("azure_openai_transcription", azure_key)
        print("✅ Successfully saved Azure OpenAI API key to keyring")
        
        # Remove from config
        ConfigManager.set_config_value(None, 'model_options', 'api', 'azure_openai_api_key')
        ConfigManager.save_config()
        print("✅ Removed Azure OpenAI API key from config file")
        
        # Verify keyring storage
        retrieved_key = KeyringManager.get_api_key("azure_openai_transcription")
        if retrieved_key == azure_key:
            print("✅ Migration successful! Key verified in keyring")
        else:
            print("❌ Migration failed - key verification failed")
            return False
            
    else:
        print("No Azure OpenAI API key found in config.")
        
        # Check if it's already in keyring
        keyring_key = KeyringManager.get_api_key("azure_openai_transcription")
        if keyring_key and keyring_key.strip():
            print(f"✅ Azure OpenAI API key already exists in keyring: {keyring_key[:8]}...")
        else:
            print("❌ No Azure OpenAI API key found in either config or keyring.")
            print("Please set your Azure OpenAI API key in the application settings.")
            return False
    
    return True

if __name__ == "__main__":
    print("Azure OpenAI API Key Migration Tool")
    print("=" * 40)
    
    try:
        success = migrate_azure_openai_key()
        if success:
            print("\n🎉 Migration completed successfully!")
            print("You can now use Azure OpenAI transcription.")
        else:
            print("\n❌ Migration failed or no key found.")
            print("Please check your Azure OpenAI API key in settings.")
    except Exception as e:
        print(f"\n❌ Error during migration: {e}")
        sys.exit(1)
