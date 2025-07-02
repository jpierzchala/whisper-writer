"""
Test cases for autostart functionality.
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add src to path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from autostart_manager import AutostartManager
from utils import ConfigManager


class TestAutostartManager(unittest.TestCase):
    """Test cases for AutostartManager functionality."""
    
    def test_platform_detection_non_windows(self):
        """Test that non-Windows platforms are detected correctly."""
        with patch('platform.system', return_value='Linux'):
            self.assertFalse(AutostartManager.is_windows())
    
    def test_platform_detection_windows(self):
        """Test that Windows platform is detected correctly."""
        with patch('platform.system', return_value='Windows'):
            self.assertTrue(AutostartManager.is_windows())
    
    def test_target_executable_detection(self):
        """Test that target executable is detected correctly."""
        target = AutostartManager.get_target_executable()
        self.assertIsNotNone(target)
        self.assertTrue(target.endswith(('run_project.bat', 'run.py')))
        self.assertTrue(os.path.exists(target))
    
    def test_non_windows_autostart_operations(self):
        """Test autostart operations on non-Windows platforms."""
        with patch('platform.system', return_value='Linux'):
            # Should return False for is_autostart_enabled
            self.assertFalse(AutostartManager.is_autostart_enabled())
            
            # Should return appropriate messages for enable/disable
            success, message = AutostartManager.create_autostart_shortcut()
            self.assertFalse(success)
            self.assertIn("Windows", message)
            
            success, message = AutostartManager.remove_autostart_shortcut()
            self.assertTrue(success)
            self.assertIn("non-Windows", message)
    
    @patch('platform.system', return_value='Windows')
    @patch('subprocess.run')
    def test_windows_startup_folder_detection(self, mock_subprocess, mock_platform):
        """Test Windows startup folder detection."""
        mock_subprocess.return_value = MagicMock(
            stdout='C:\\Users\\TestUser\\AppData\\Roaming\n',
            returncode=0
        )
        
        with patch('os.path.exists', return_value=True):
            startup_folder = AutostartManager.get_startup_folder()
            expected_path = 'C:\\Users\\TestUser\\AppData\\Roaming\\Microsoft\\Windows\\Start Menu\\Programs\\Startup'
            self.assertEqual(startup_folder, expected_path)
    
    @patch('platform.system', return_value='Windows')
    def test_windows_shortcut_path_generation(self, mock_platform):
        """Test shortcut path generation on Windows."""
        with patch.object(AutostartManager, 'get_startup_folder', return_value='C:\\StartupFolder'):
            shortcut_path = AutostartManager.get_shortcut_path()
            expected_path = 'C:\\StartupFolder\\WhisperWriter.lnk'
            self.assertEqual(shortcut_path, expected_path)
    
    @patch('platform.system', return_value='Windows')
    @patch('os.path.exists', return_value=True)
    def test_windows_autostart_enabled_detection(self, mock_exists, mock_platform):
        """Test detection of enabled autostart on Windows."""
        with patch.object(AutostartManager, 'get_shortcut_path', return_value='C:\\test.lnk'):
            self.assertTrue(AutostartManager.is_autostart_enabled())
    
    @patch('platform.system', return_value='Windows')
    @patch('os.path.exists', return_value=False)
    def test_windows_autostart_disabled_detection(self, mock_exists, mock_platform):
        """Test detection of disabled autostart on Windows."""
        with patch.object(AutostartManager, 'get_shortcut_path', return_value='C:\\test.lnk'):
            self.assertFalse(AutostartManager.is_autostart_enabled())


class TestAutostartConfiguration(unittest.TestCase):
    """Test cases for autostart configuration integration."""
    
    def setUp(self):
        """Set up test configuration."""
        ConfigManager.initialize()
    
    def test_autostart_config_option_exists(self):
        """Test that autostart configuration option exists."""
        schema = ConfigManager.get_schema()
        self.assertIn('misc', schema)
        self.assertIn('autostart_on_login', schema['misc'])
        
        autostart_config = schema['misc']['autostart_on_login']
        self.assertEqual(autostart_config['type'], 'bool')
        self.assertEqual(autostart_config['value'], False)
        self.assertIn('description', autostart_config)
    
    def test_autostart_config_default_value(self):
        """Test that autostart has correct default value."""
        default_value = ConfigManager.get_config_value('misc', 'autostart_on_login')
        self.assertEqual(default_value, False)
    
    def test_autostart_config_set_value(self):
        """Test setting autostart configuration value."""
        # Test setting to True
        ConfigManager.set_config_value(True, 'misc', 'autostart_on_login')
        value = ConfigManager.get_config_value('misc', 'autostart_on_login')
        self.assertEqual(value, True)
        
        # Test setting to False
        ConfigManager.set_config_value(False, 'misc', 'autostart_on_login')
        value = ConfigManager.get_config_value('misc', 'autostart_on_login')
        self.assertEqual(value, False)


if __name__ == '__main__':
    unittest.main()