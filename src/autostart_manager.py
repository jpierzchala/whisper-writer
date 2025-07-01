"""
AutostartManager - Handles autostart functionality for Windows systems.
"""
import os
import sys
import platform
import subprocess
from pathlib import Path


class AutostartManager:
    """Manages autostart functionality for WhisperWriter on Windows."""
    
    APP_NAME = "WhisperWriter"
    
    @staticmethod
    def is_windows():
        """Check if the current platform is Windows."""
        return platform.system() == 'Windows'
    
    @staticmethod
    def get_startup_folder():
        """Get the Windows startup folder path for the current user."""
        if not AutostartManager.is_windows():
            return None
        
        try:
            # Use shell:startup to get the startup folder
            result = subprocess.run(
                ['cmd', '/c', 'echo %APPDATA%'],
                capture_output=True,
                text=True,
                check=True
            )
            appdata = result.stdout.strip()
            startup_folder = os.path.join(appdata, 'Microsoft', 'Windows', 'Start Menu', 'Programs', 'Startup')
            startup_folder = startup_folder.replace('/', '\\')  # Ensure Windows path separators
            return startup_folder if os.path.exists(startup_folder) else None
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    
    @staticmethod
    def get_shortcut_path():
        """Get the full path to the autostart shortcut."""
        startup_folder = AutostartManager.get_startup_folder()
        if not startup_folder:
            return None
        shortcut_path = os.path.join(startup_folder, f"{AutostartManager.APP_NAME}.lnk")
        return shortcut_path.replace('/', '\\') if AutostartManager.is_windows() else shortcut_path
    
    @staticmethod
    def get_target_executable():
        """Get the target executable/script to run on startup."""
        # Get the project root directory (where run_project.bat is located)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)  # Go up one level from src/
        
        # Prefer run_project.bat if it exists, otherwise use run.py
        bat_file = os.path.join(project_root, 'run_project.bat')
        py_file = os.path.join(project_root, 'run.py')
        
        if os.path.exists(bat_file):
            return bat_file
        elif os.path.exists(py_file):
            return py_file
        else:
            return None
    
    @staticmethod
    def create_autostart_shortcut():
        """Create an autostart shortcut in the Windows startup folder."""
        if not AutostartManager.is_windows():
            return False, "Autostart is only supported on Windows"
        
        startup_folder = AutostartManager.get_startup_folder()
        if not startup_folder:
            return False, "Could not access Windows startup folder"
        
        target_executable = AutostartManager.get_target_executable()
        if not target_executable:
            return False, "Could not find WhisperWriter executable"
        
        shortcut_path = AutostartManager.get_shortcut_path()
        if not shortcut_path:
            return False, "Could not determine shortcut path"
        
        try:
            # Use PowerShell to create the shortcut
            target_dir = os.path.dirname(target_executable)
            ps_script = f'''
$WshShell = New-Object -comObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("{shortcut_path}")
$Shortcut.TargetPath = "{target_executable}"
$Shortcut.WorkingDirectory = "{target_dir}"
$Shortcut.Description = "WhisperWriter - Voice Transcription Tool"
$Shortcut.Save()
'''
            
            result = subprocess.run(
                ['powershell', '-Command', ps_script],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Verify the shortcut was created
            if os.path.exists(shortcut_path):
                return True, "Autostart enabled successfully"
            else:
                return False, "Shortcut creation failed"
                
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            return False, f"Failed to create autostart shortcut: {str(e)}"
    
    @staticmethod
    def remove_autostart_shortcut():
        """Remove the autostart shortcut from the Windows startup folder."""
        if not AutostartManager.is_windows():
            return True, "Autostart not applicable on non-Windows systems"
        
        shortcut_path = AutostartManager.get_shortcut_path()
        if not shortcut_path:
            return True, "Shortcut path not found"
        
        try:
            if os.path.exists(shortcut_path):
                os.remove(shortcut_path)
                return True, "Autostart disabled successfully"
            else:
                return True, "Autostart shortcut was not present"
        except OSError as e:
            return False, f"Failed to remove autostart shortcut: {str(e)}"
    
    @staticmethod
    def is_autostart_enabled():
        """Check if autostart is currently enabled."""
        if not AutostartManager.is_windows():
            return False
        
        shortcut_path = AutostartManager.get_shortcut_path()
        return shortcut_path and os.path.exists(shortcut_path)
    
    @staticmethod
    def set_autostart(enabled):
        """Enable or disable autostart based on the enabled parameter."""
        if enabled:
            return AutostartManager.create_autostart_shortcut()
        else:
            return AutostartManager.remove_autostart_shortcut()