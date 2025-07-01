import yaml
import os
import logging
from datetime import datetime

class ConfigManager:
    _instance = None
    _schema = None
    _logger = None
    _file_handler = None

    def __init__(self):
        """Initialize the ConfigManager instance."""
        self.config = None
        self.schema = None

    @classmethod
    def initialize(cls, schema_path=None):
        """Initialize the ConfigManager with the given schema path."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.schema = cls._instance.load_config_schema(schema_path)
            cls._instance.config = cls._instance.load_default_config()
            cls._instance.load_user_config()
            cls.load_env_variables()
            cls._setup_logging()

    @classmethod
    def get_schema(cls):
        """Load and return the configuration schema."""
        if not cls._schema:
            schema_path = os.path.join(os.path.dirname(__file__), 'config_schema.yaml')
            try:
                with open(schema_path, 'r') as f:
                    cls._schema = yaml.safe_load(f)
                    # ConfigManager.console_print("Loaded schema:")
                    # ConfigManager.console_print(f"Model options in schema: {cls._schema['model_options']['local']['model']['options']}")
            except Exception as e:
                ConfigManager.console_print(f"Error loading schema: {str(e)}")
                cls._schema = {}
        return cls._schema

    @classmethod
    def get_config_section(cls, *keys):
        """Get a specific section of the configuration."""
        if cls._instance is None:
            raise RuntimeError("ConfigManager not initialized")

        section = cls._instance.config
        for key in keys:
            if isinstance(section, dict) and key in section:
                section = section[key]
            else:
                return {}
        return section

    @classmethod
    def get_config_value(cls, *keys):
        """Get a specific configuration value using nested keys."""
        if cls._instance is None:
            raise RuntimeError("ConfigManager not initialized")

        value = cls._instance.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

    @classmethod
    def set_config_value(cls, value, *keys):
        """Set a specific configuration value using nested keys."""
        if cls._instance is None:
            raise RuntimeError("ConfigManager not initialized")

        config = cls._instance.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            elif not isinstance(config[key], dict):
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value

    @staticmethod
    def load_config_schema(schema_path=None):
        """Load the configuration schema from a YAML file."""
        if schema_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            schema_path = os.path.join(base_dir, 'config_schema.yaml')

        with open(schema_path, 'r') as file:
            schema = yaml.safe_load(file)
        return schema

    def load_default_config(self):
        """Load default configuration values from the schema."""
        def extract_value(item):
            if isinstance(item, dict):
                if 'value' in item:
                    return item['value']
                else:
                    return {k: extract_value(v) for k, v in item.items()}
            return item

        config = {}
        for category, settings in self.schema.items():
            config[category] = extract_value(settings)
        return config

    def load_user_config(self, config_path=os.path.join('src', 'config.yaml')):
        """Load user configuration and merge with default config."""
        def deep_update(source, overrides):
            for key, value in overrides.items():
                if isinstance(value, dict) and key in source:
                    deep_update(source[key], value)
                else:
                    source[key] = value

        if config_path and os.path.isfile(config_path):
            try:
                with open(config_path, 'r') as file:
                    user_config = yaml.safe_load(file)
                    deep_update(self.config, user_config)
            except yaml.YAMLError:
                print("Error in configuration file. Using default configuration.")

    @classmethod
    def save_config(cls, config_path=os.path.join('src', 'config.yaml')):
        """Save the current configuration to a YAML file."""
        if cls._instance is None:
            raise RuntimeError("ConfigManager not initialized")
        with open(config_path, 'w') as file:
            yaml.dump(cls._instance.config, file, default_flow_style=False)
        # Reload logging configuration after saving config
        cls._setup_logging()

    @classmethod
    def reload_config(cls):
        """
        Reload the configuration from the file.
        """
        if cls._instance is None:
            raise RuntimeError("ConfigManager not initialized")
        cls._instance.config = cls._instance.load_default_config()
        cls._instance.load_user_config()

    @classmethod
    def config_file_exists(cls):
        """Check if a valid config file exists."""
        config_path = os.path.join('src', 'config.yaml')
        return os.path.isfile(config_path)

    @classmethod
    def console_print(cls, message, verbose=False):
        """Print a message to the console and/or log file based on configuration."""
        if cls._instance is None:
            print(message)  # Fallback if not initialized
            return
            
        config = cls._instance.config.get('misc', {})
        
        # Check if we should show this message
        show_message = True
        if verbose and not config.get('verbose_mode', False):
            show_message = False
            
        if not show_message:
            return
            
        # Print to console if enabled
        if config.get('print_to_terminal', True):
            print(message)
            
        # Log to file if enabled
        if config.get('log_to_file', False) and cls._logger:
            cls._logger.info(message)

    @classmethod
    def _setup_logging(cls):
        """Setup file logging based on configuration."""
        if cls._instance is None:
            return
            
        config = cls._instance.config.get('misc', {})
        
        if not config.get('log_to_file', False):
            return
            
        # Setup logger
        cls._logger = logging.getLogger('whisperwriter')
        cls._logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in cls._logger.handlers[:]:
            cls._logger.removeHandler(handler)
            
        # Determine log file path
        log_file_path = config.get('log_file_path')
        if not log_file_path:
            log_dir = os.path.join(os.path.expanduser('~'), '.whisperwriter', 'logs')
            os.makedirs(log_dir, exist_ok=True)
            log_file_path = os.path.join(log_dir, 'whisperwriter.log')
            
        # Create file handler
        cls._file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        cls._file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        cls._file_handler.setFormatter(formatter)
        
        # Add handler to logger
        cls._logger.addHandler(cls._file_handler)
        
        cls._logger.info("WhisperWriter logging started")

    @classmethod
    def set_verbose_mode(cls, verbose):
        """Set verbose mode for this session."""
        if cls._instance is None:
            return
        cls._instance.config['misc']['verbose_mode'] = verbose
        
    @classmethod 
    def get_verbose_mode(cls):
        """Get current verbose mode setting."""
        if cls._instance is None:
            return False
        return cls._instance.config.get('misc', {}).get('verbose_mode', False)

    @classmethod
    def reload_logging(cls):
        """Reload logging configuration after config changes."""
        cls._setup_logging()

    @classmethod
    def load_env_variables(cls):
        """Load environment variables from .env file"""
        if os.path.exists('.env'):
            try:
                with open('.env', 'r') as f:
                    for line in f:
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            os.environ[key] = value.strip('"').strip("'")
            except Exception as e:
                print(f"Error loading .env file: {e}")
