import subprocess
import os
import signal
import time
import win32clipboard
import win32con
from pynput.keyboard import Controller as PynputController, Key

from utils import ConfigManager


KNOWN_CLIPBOARD_FORMATS = {
    win32con.CF_TEXT: 'CF_TEXT',
    win32con.CF_BITMAP: 'CF_BITMAP',
    win32con.CF_METAFILEPICT: 'CF_METAFILEPICT',
    win32con.CF_SYLK: 'CF_SYLK',
    win32con.CF_DIF: 'CF_DIF',
    win32con.CF_TIFF: 'CF_TIFF',
    win32con.CF_OEMTEXT: 'CF_OEMTEXT',
    win32con.CF_DIB: 'CF_DIB',
    win32con.CF_PALETTE: 'CF_PALETTE',
    win32con.CF_PENDATA: 'CF_PENDATA',
    win32con.CF_RIFF: 'CF_RIFF',
    win32con.CF_WAVE: 'CF_WAVE',
    win32con.CF_UNICODETEXT: 'CF_UNICODETEXT',
    win32con.CF_ENHMETAFILE: 'CF_ENHMETAFILE',
    win32con.CF_HDROP: 'CF_HDROP',
    win32con.CF_LOCALE: 'CF_LOCALE',
    win32con.CF_DIBV5: 'CF_DIBV5',
}

TEXT_CLIPBOARD_FORMATS = {
    win32con.CF_TEXT,
    win32con.CF_OEMTEXT,
    win32con.CF_UNICODETEXT,
}

IMAGE_CLIPBOARD_FORMATS = {
    win32con.CF_BITMAP,
    win32con.CF_DIB,
    win32con.CF_DIBV5,
    win32con.CF_TIFF,
    win32con.CF_ENHMETAFILE,
    win32con.CF_METAFILEPICT,
}

IMAGE_CLIPBOARD_FORMAT_HINTS = (
    'bitmap',
    'dib',
    'image',
    'png',
    'jpg',
    'jpeg',
    'gif',
    'tiff',
)

TEXT_ONLY_CLIPBOARD_RESTORE_DELAY = 0.2
RICH_CONTENT_CLIPBOARD_RESTORE_DELAY = 0.5

def run_command_or_exit_on_failure(command):
    """
    Run a shell command and exit if it fails.

    Args:
        command (list): The command to run as a list of strings.
    """
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        exit(1)

class InputSimulator:
    """
    A class to simulate keyboard input using various methods.
    """

    def __init__(self):
        """
        Initialize the InputSimulator with the specified configuration.
        """
        self.input_method = ConfigManager.get_config_value('post_processing', 'input_method')
        self.dotool_process = None

        if self.input_method == 'pynput':
            self.keyboard = PynputController()
        elif self.input_method == 'dotool':
            self._initialize_dotool()

    def _initialize_dotool(self):
        """
        Initialize the dotool process for input simulation.
        """
        self.dotool_process = subprocess.Popen("dotool", stdin=subprocess.PIPE, text=True)
        assert self.dotool_process.stdin is not None

    def _terminate_dotool(self):
        """
        Terminate the dotool process if it's running.
        """
        if self.dotool_process:
            os.kill(self.dotool_process.pid, signal.SIGINT)
            self.dotool_process = None

    def typewrite(self, text):
        """
        Simulate typing the given text. Uses clipboard for long text and keystrokes for short text.

        Args:
            text (str): The text to type.
        """
        # Get the character threshold from config, default to 1000 if not set
        char_threshold = ConfigManager.get_config_value('post_processing', 'clipboard_threshold') or 1000
        
        # Use clipboard for long text
        if len(text) > char_threshold:
            if self._paste_with_clipboard_preservation(text):
                return

        self.typewrite_direct(text)

    def typewrite_direct(self, text):
        """Type text without using the clipboard."""
        interval = ConfigManager.get_config_value('post_processing', 'writing_key_press_delay')
        if self.input_method == 'pynput':
            self._typewrite_pynput(text, interval)
        elif self.input_method == 'ydotool':
            self._typewrite_ydotool(text, interval)
        elif self.input_method == 'dotool':
            self._typewrite_dotool(text, interval)
    
    @staticmethod
    def safe_open_clipboard(max_retries=5, delay=0.1):
        """
        Try to open the clipboard repeatedly.
        Returns True if successful, else False.
        """
        for _ in range(max_retries):
            try:
                win32clipboard.OpenClipboard()
                return True
            except Exception:
                time.sleep(delay)
        return False

    @staticmethod
    def safe_close_clipboard():
        """Close the clipboard if it is open."""
        try:
            win32clipboard.CloseClipboard()
            return True
        except Exception:
            return False

    @staticmethod
    def get_clipboard_format_name(format_id):
        """Return a readable clipboard format name for logs."""
        if format_id in KNOWN_CLIPBOARD_FORMATS:
            return KNOWN_CLIPBOARD_FORMATS[format_id]
        try:
            return win32clipboard.GetClipboardFormatName(format_id)
        except Exception:
            return f'FORMAT_{format_id}'

    @classmethod
    def describe_clipboard_formats(cls, formats):
        """Return a concise description of clipboard formats."""
        format_ids = formats.keys() if isinstance(formats, dict) else formats
        descriptions = [f"{cls.get_clipboard_format_name(format_id)}({format_id})" for format_id in format_ids]
        return ', '.join(descriptions) if descriptions else 'none'

    @classmethod
    def has_rich_clipboard_content(cls, formats):
        """Return True if clipboard contains non-text formats such as images."""
        format_ids = formats.keys() if isinstance(formats, dict) else formats
        return any(format_id not in TEXT_CLIPBOARD_FORMATS for format_id in format_ids)

    @classmethod
    def has_image_clipboard_content(cls, formats):
        """Return True if clipboard contains image-like formats that can override text paste."""
        format_ids = formats.keys() if isinstance(formats, dict) else formats
        for format_id in format_ids:
            if format_id in IMAGE_CLIPBOARD_FORMATS:
                return True
            format_name = cls.get_clipboard_format_name(format_id).lower()
            if any(hint in format_name for hint in IMAGE_CLIPBOARD_FORMAT_HINTS):
                return True
        return False

    @classmethod
    def get_clipboard_restore_delay(cls, formats):
        """Use a longer restore delay when the original clipboard had rich content."""
        if cls.has_rich_clipboard_content(formats):
            return RICH_CONTENT_CLIPBOARD_RESTORE_DELAY
        return TEXT_ONLY_CLIPBOARD_RESTORE_DELAY

    @classmethod
    def capture_open_clipboard_formats(cls):
        """Capture all readable clipboard formats while the clipboard is already open."""
        saved_formats = {}
        format_id = win32clipboard.EnumClipboardFormats(0)
        while format_id:
            try:
                saved_formats[format_id] = win32clipboard.GetClipboardData(format_id)
            except Exception as exc:
                ConfigManager.console_print(
                    f"Skipping clipboard format {cls.get_clipboard_format_name(format_id)}({format_id}): {exc}",
                    verbose=True,
                )
            format_id = win32clipboard.EnumClipboardFormats(format_id)
        return saved_formats

    @classmethod
    def restore_open_clipboard_formats(cls, saved_formats):
        """Restore clipboard formats while the clipboard is already open."""
        restored_formats = []
        for format_id, data in saved_formats.items():
            try:
                win32clipboard.SetClipboardData(format_id, data)
                restored_formats.append(format_id)
            except Exception as exc:
                ConfigManager.console_print(
                    f"Failed to restore clipboard format {cls.get_clipboard_format_name(format_id)}({format_id}): {exc}",
                    verbose=True,
                )
        return restored_formats

    @staticmethod
    def get_open_clipboard_text():
        """Return Unicode clipboard text while the clipboard is already open."""
        try:
            return win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
        except Exception:
            return None

    @staticmethod
    def should_restore_clipboard(current_text, pasted_text):
        """Only restore clipboard contents if the clipboard still contains our pasted text."""
        return current_text == pasted_text

    def _paste_with_clipboard_preservation(self, text):
        """
        Paste text using the clipboard while preserving original clipboard content,
        including images and other data formats.

        Args:
            text (str): The text to paste.
        """
        # Store all clipboard formats
        saved_formats = {}
        if not InputSimulator.safe_open_clipboard():
            ConfigManager.console_print(
                "Unable to open clipboard for preserving original content; falling back to direct typing."
            )
            return False
        
        try:
            saved_formats = InputSimulator.capture_open_clipboard_formats()
            ConfigManager.console_print(
                f"Clipboard paste captured formats: {InputSimulator.describe_clipboard_formats(saved_formats)}",
                verbose=True,
            )
            if InputSimulator.has_image_clipboard_content(saved_formats):
                ConfigManager.console_print(
                    "Clipboard contains image formats; using direct typing instead of clipboard paste to avoid pasting the image back into the target app."
                )
                return False
            if InputSimulator.has_rich_clipboard_content(saved_formats):
                ConfigManager.console_print(
                    "Clipboard contains non-text formats; delaying clipboard restore to avoid restoring image/rich content before paste completes.",
                    verbose=True,
                )
            
            # Clear clipboard and set our text
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardText(text, win32con.CF_UNICODETEXT)
        except Exception as exc:
            ConfigManager.console_print(
                f"Unable to prepare clipboard text for paste; falling back to direct typing ({exc})."
            )
            return False
        finally:
            InputSimulator.safe_close_clipboard()

        # Simulate Ctrl+V
        if self.input_method == 'pynput':
            with self.keyboard.pressed(Key.ctrl):
                self.keyboard.press('v')
                self.keyboard.release('v')
        elif self.input_method == 'ydotool':
            run_command_or_exit_on_failure([
                "ydotool", "key", "ctrl+v"
            ])
        elif self.input_method == 'dotool':
            assert self.dotool_process and self.dotool_process.stdin
            self.dotool_process.stdin.write("key ctrl+v\n")
            self.dotool_process.stdin.flush()
            
        # Wait for the target application to read clipboard data before restoring the original clipboard.
        restore_delay = InputSimulator.get_clipboard_restore_delay(saved_formats)
        ConfigManager.console_print(
            f"Waiting {restore_delay:.2f}s before restoring clipboard after paste.",
            verbose=True,
        )
        time.sleep(restore_delay)
        
        # Restore all original clipboard formats
        if not InputSimulator.safe_open_clipboard():
            ConfigManager.console_print("Unable to reopen clipboard for restoring original content.")
            return True
        try:
            current_text = InputSimulator.get_open_clipboard_text()
            if not InputSimulator.should_restore_clipboard(current_text, text):
                ConfigManager.console_print(
                    "Skipping clipboard restore because clipboard contents changed before restore.",
                    verbose=True,
                )
                return True

            win32clipboard.EmptyClipboard()
            restored_formats = InputSimulator.restore_open_clipboard_formats(saved_formats)
            ConfigManager.console_print(
                f"Restored clipboard formats: {InputSimulator.describe_clipboard_formats(restored_formats)}",
                verbose=True,
            )
        finally:
            InputSimulator.safe_close_clipboard()

        return True

    def _typewrite_pynput(self, text, interval):
        """
        Simulate typing using pynput.

        Args:
            text (str): The text to type.
            interval (float): The interval between keystrokes in seconds.
        """
        for char in text:
            self.keyboard.press(char)
            self.keyboard.release(char)
            time.sleep(interval)

    def _typewrite_ydotool(self, text, interval):
        """
        Simulate typing using ydotool.

        Args:
            text (str): The text to type.
            interval (float): The interval between keystrokes in seconds.
        """
        cmd = "ydotool"
        run_command_or_exit_on_failure([
            cmd,
            "type",
            "--key-delay",
            str(interval * 1000),
            "--",
            text,
        ])

    def _typewrite_dotool(self, text, interval):
        """
        Simulate typing using dotool.

        Args:
            text (str): The text to type.
            interval (float): The interval between keystrokes in seconds.
        """
        assert self.dotool_process and self.dotool_process.stdin
        self.dotool_process.stdin.write(f"typedelay {interval * 1000}\n")
        self.dotool_process.stdin.write(f"type {text}\n")
        self.dotool_process.stdin.flush()

    def cleanup(self):
        """
        Perform cleanup operations, such as terminating the dotool process.
        """
        if self.input_method == 'dotool':
            self._terminate_dotool()
