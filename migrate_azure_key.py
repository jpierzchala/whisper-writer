import os
import yaml
import keyring


def get_nested(dct, keys, default=None):
    current = dct
    try:
        for k in keys:
            if current is None:
                return default
            current = current.get(k)
        return current if current is not None else default
    except Exception:
        return default


def migrate_azure_key(config_path: str = 'config.yaml') -> None:
    """Migrate Azure OpenAI transcription key from config.yaml to system keyring.

    - Reads azure_openai_api_key from model_options.api in config.yaml
    - Stores it in keyring under service 'whisperwriter' and name 'azure_openai_transcription'
    - Removes the key from config.yaml (sets to None) and saves the file
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}

    api = get_nested(config, ['model_options', 'api'], default={})
    azure_key = api.get('azure_openai_api_key')

    if not azure_key:
        print('No azure_openai_api_key found in config. Nothing to migrate.')
        return

    # Save to keyring
    keyring.set_password('whisperwriter', 'azure_openai_transcription', azure_key)
    print('Saved Azure key to keyring under service=whisperwriter, name=azure_openai_transcription')

    # Remove from config
    api['azure_openai_api_key'] = None
    if 'model_options' not in config:
        config['model_options'] = {}
    if 'api' not in config['model_options']:
        config['model_options']['api'] = {}
    config['model_options']['api'] = api

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)
    print('Removed key from config.yaml and saved changes.')


if __name__ == '__main__':
    migrate_azure_key()


