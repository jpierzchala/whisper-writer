"""
Test checkbox creation for autostart functionality (without GUI dependencies).
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add src to path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestAutostartCheckbox(unittest.TestCase):
    """Test autostart checkbox creation and platform handling."""
    
    @patch('platform.system', return_value='Linux')
    def test_autostart_checkbox_disabled_on_non_windows(self, mock_platform):
        """Test that autostart checkbox is disabled on non-Windows platforms."""
        # Mock PyQt5 components
        with patch('PyQt5.QtWidgets.QCheckBox') as mock_checkbox:
            mock_checkbox_instance = MagicMock()
            mock_checkbox.return_value = mock_checkbox_instance
            
            # Import after mocking to avoid PyQt5 loading issues
            with patch.dict('sys.modules', {
                'PyQt5.QtWidgets': MagicMock(),
                'PyQt5.QtCore': MagicMock(),
                'PyQt5.QtGui': MagicMock(),
                'sounddevice': MagicMock(),
                'dotenv': MagicMock()
            }):
                from ui.settings_window import SettingsWindow
                
                # Create mock settings window
                settings_window = SettingsWindow()
                
                # Test checkbox creation
                checkbox = settings_window.create_checkbox(False, 'autostart_on_login')
                
                # Verify checkbox was created and configured correctly
                mock_checkbox.assert_called_once()
                mock_checkbox_instance.setChecked.assert_called_once_with(False)
                mock_checkbox_instance.setEnabled.assert_called_once_with(False)
                mock_checkbox_instance.setToolTip.assert_called_once_with("Autostart is only supported on Windows")
    
    @patch('platform.system', return_value='Windows')
    def test_autostart_checkbox_enabled_on_windows(self, mock_platform):
        """Test that autostart checkbox is enabled on Windows platforms."""
        # Mock PyQt5 components
        with patch('PyQt5.QtWidgets.QCheckBox') as mock_checkbox:
            mock_checkbox_instance = MagicMock()
            mock_checkbox.return_value = mock_checkbox_instance
            
            # Import after mocking to avoid PyQt5 loading issues
            with patch.dict('sys.modules', {
                'PyQt5.QtWidgets': MagicMock(),
                'PyQt5.QtCore': MagicMock(),
                'PyQt5.QtGui': MagicMock(),
                'sounddevice': MagicMock(),
                'dotenv': MagicMock()
            }):
                from ui.settings_window import SettingsWindow
                
                # Create mock settings window
                settings_window = SettingsWindow()
                
                # Test checkbox creation
                checkbox = settings_window.create_checkbox(True, 'autostart_on_login')
                
                # Verify checkbox was created and configured correctly
                mock_checkbox.assert_called_once()
                mock_checkbox_instance.setChecked.assert_called_once_with(True)
                # Should NOT call setEnabled(False) on Windows
                mock_checkbox_instance.setEnabled.assert_not_called()
                mock_checkbox_instance.setToolTip.assert_not_called()
    
    def test_regular_checkbox_creation(self):
        """Test that regular checkbox creation is not affected."""
        # Mock PyQt5 components
        with patch('PyQt5.QtWidgets.QCheckBox') as mock_checkbox:
            mock_checkbox_instance = MagicMock()
            mock_checkbox.return_value = mock_checkbox_instance
            
            # Import after mocking to avoid PyQt5 loading issues
            with patch.dict('sys.modules', {
                'PyQt5.QtWidgets': MagicMock(),
                'PyQt5.QtCore': MagicMock(),
                'PyQt5.QtGui': MagicMock(),
                'sounddevice': MagicMock(),
                'dotenv': MagicMock()
            }):
                from ui.settings_window import SettingsWindow
                
                # Create mock settings window
                settings_window = SettingsWindow()
                
                # Test regular checkbox creation
                checkbox = settings_window.create_checkbox(True, 'some_other_option')
                
                # Verify checkbox was created normally
                mock_checkbox.assert_called_once()
                mock_checkbox_instance.setChecked.assert_called_once_with(True)
                # Should NOT call setEnabled(False) for regular options
                mock_checkbox_instance.setEnabled.assert_not_called()
                mock_checkbox_instance.setToolTip.assert_not_called()


if __name__ == '__main__':
    unittest.main()