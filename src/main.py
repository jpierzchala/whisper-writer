import os
import sys
import time
import argparse
from audioplayer import AudioPlayer
from pynput.keyboard import Controller, Key
from PyQt5.QtCore import QObject, QProcess
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QAction, QMessageBox, QLineEdit
import win32clipboard
import win32con

from key_listener import KeyListener
from result_thread import ResultThread
from ui.main_window import MainWindow
from ui.settings_window import SettingsWindow
from ui.status_window import StatusWindow
from transcription import create_local_model
from input_simulation import InputSimulator
from utils import ConfigManager
from llm_processor import LLMProcessor


class WhisperWriterApp(QObject):
    def __init__(self, verbose_mode=False):
        """
        Initialize the application, opening settings window if no configuration file is found.
        
        :param verbose_mode: Enable verbose logging for this session
        """
        super().__init__()
        self.app = QApplication(sys.argv)
        self.app.setWindowIcon(QIcon(os.path.join('assets', 'ww-logo.png')))

        ConfigManager.initialize()
        
        # Set verbose mode if specified via command line
        if verbose_mode:
            ConfigManager.set_verbose_mode(True)

        self.settings_window = SettingsWindow()
        self.settings_window.settings_closed.connect(self.on_settings_closed)
        self.settings_window.settings_saved.connect(self.restart_app)

        if ConfigManager.config_file_exists():
            self.initialize_components()
            # Start listening immediately instead of showing main window
            self.key_listener.start()
        else:
            print('No valid configuration file found. Opening settings window...')
            self.settings_window.show()

    def initialize_components(self):
        """
        Initialize the components of the application.
        """
        self.input_simulator = InputSimulator()

        self.key_listener = KeyListener()
        self.key_listener.add_callback("on_activate", self.on_activation)
        self.key_listener.add_callback("on_activate_with_llm", self.on_activation_with_llm_cleanup)
        self.key_listener.add_callback("on_activate_with_llm_instruction", self.on_activation_with_llm_instruction)
        self.key_listener.add_callback("on_deactivate", self.on_deactivation)
        self.key_listener.add_callback("on_deactivate_with_llm", self.on_deactivation_with_llm)
        self.key_listener.add_callback("on_deactivate_with_llm_instruction", self.on_deactivation_with_llm_instruction)
        self.key_listener.add_callback("on_text_cleanup", self.handle_text_cleanup)

        model_options = ConfigManager.get_config_section('model_options')
        model_path = model_options.get('local', {}).get('model_path')
        self.local_model = create_local_model() if not model_options.get('use_api') else None

        self.result_thread = None
        self.llm_processor = LLMProcessor() if ConfigManager.get_config_value('llm_post_processing', 'enabled') else None

        if not ConfigManager.get_config_value('misc', 'hide_status_window'):
            self.status_window = StatusWindow()

        self.create_tray_icon()

    def create_tray_icon(self):
        """
        Create the system tray icon and its context menu.
        """
        self.tray_icon = QSystemTrayIcon(QIcon(os.path.join('assets', 'ww-logo.png')), self.app)

        tray_menu = QMenu()

        settings_action = QAction('Settings', self.app)
        settings_action.triggered.connect(self.settings_window.show)
        tray_menu.addAction(settings_action)

        exit_action = QAction('Exit', self.app)
        exit_action.triggered.connect(self.exit_app)
        tray_menu.addAction(exit_action)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

    def cleanup(self):
        if self.key_listener:
            self.key_listener.stop()
        if self.input_simulator:
            self.input_simulator.cleanup()

    def exit_app(self):
        """
        Exit the application.
        """
        self.cleanup()
        QApplication.quit()

    def restart_app(self):
        """Restart the application to apply the new settings."""
        self.cleanup()
        QApplication.quit()
        QProcess.startDetached(sys.executable, sys.argv)

    def on_settings_closed(self):
        """
        If settings is closed without saving on first run, initialize the components with default values.
        """
        if not os.path.exists(os.path.join('src', 'config.yaml')):
            QMessageBox.information(
                self.settings_window,
                'Using Default Values',
                'Settings closed without saving. Default values are being used.'
            )
            self.initialize_components()

    def on_activation(self, use_llm=False, is_instruction_mode=False):
        """
        Called when the activation key combination is pressed.
        Args:
            use_llm (bool): Whether to use LLM processing for this activation
            is_instruction_mode (bool): Whether to use instruction mode for LLM processing
        """
        if self.result_thread and self.result_thread.isRunning():
            recording_mode = ConfigManager.get_config_value('recording_options', 'recording_mode')
            if recording_mode == 'press_to_toggle':
                self.result_thread.stop_recording()
            elif recording_mode == 'continuous':
                self.stop_result_thread()
            return
        
        # If we get here, no thread is running, so start one
        self.use_llm = use_llm
        self.is_instruction_mode = is_instruction_mode
        self.start_result_thread()

    def on_activation_with_llm_cleanup(self):
        """Activation with LLM cleanup based on recording mode"""
        recording_mode = ConfigManager.get_config_value('recording_options', 'recording_mode')
        # Enable LLM for both press_to_toggle and hold_to_record modes
        # self.use_llm = recording_mode in ('press_to_toggle', 'hold_to_record')
        self.on_activation(use_llm=True, is_instruction_mode=False)

    def on_activation_with_llm_instruction(self):
        """Activation with LLM instruction processing based on recording mode"""
        recording_mode = ConfigManager.get_config_value('recording_options', 'recording_mode')
        # Enable LLM for both press_to_toggle and hold_to_record modes
        # self.use_llm = recording_mode in ('press_to_toggle', 'hold_to_record')
        """Activation with LLM instruction processing"""
        self.on_activation(use_llm=True, is_instruction_mode=True)

    def on_deactivation(self, use_llm=False, is_instruction_mode=False):
        """
        Called when the activation key combination is released.
        Args:
            use_llm (bool): Whether to use LLM processing for this deactivation
            is_instruction_mode (bool): Whether to use instruction mode for LLM processing
        """
        # Set the flags just like in on_activation
        self.use_llm = use_llm
        self.is_instruction_mode = is_instruction_mode
        
        ConfigManager.console_print(f"Deactivation called - use_llm: {use_llm}, is_instruction_mode: {is_instruction_mode}", verbose=True)
        ConfigManager.console_print(f"Recording mode: {ConfigManager.get_config_value('recording_options', 'recording_mode')}", verbose=True)
        ConfigManager.console_print(f"Result thread running: {self.result_thread and self.result_thread.isRunning()}", verbose=True)
        
        if ConfigManager.get_config_value('recording_options', 'recording_mode') == 'hold_to_record':
            if self.result_thread and self.result_thread.isRunning():
                ConfigManager.console_print("Stopping recording...", verbose=True)
                self.result_thread.stop_recording()

    def on_deactivation_with_llm(self):
        """Called when the LLM cleanup activation key combination is released."""
        ConfigManager.console_print("LLM cleanup deactivation triggered", verbose=True)
        self.on_deactivation(use_llm=True, is_instruction_mode=False)

    def on_deactivation_with_llm_instruction(self):
        """Called when the LLM instruction activation key combination is released."""
        ConfigManager.console_print("LLM instruction deactivation triggered", verbose=True)
        self.on_deactivation(use_llm=True, is_instruction_mode=True)

    def start_result_thread(self):
        """
        Start the result thread to record audio and transcribe it.
        """
        if self.result_thread and self.result_thread.isRunning():
            return

        self.result_thread = ResultThread(self.local_model, self.use_llm)
        if not ConfigManager.get_config_value('misc', 'hide_status_window'):
            self.result_thread.statusSignal.connect(self.status_window.updateStatus)
            self.status_window.closeSignal.connect(self.stop_result_thread)
        self.result_thread.resultSignal.connect(self.on_transcription_complete)
        self.result_thread.start()

    def stop_result_thread(self):
        """
        Stop the result thread.
        """
        if self.result_thread and self.result_thread.isRunning():
            self.result_thread.stop()

    def _pause_key_listener_for_processing(self):
        """Pause the key listener during typing/cleanup work and report whether it was running."""
        if not getattr(self, 'key_listener', None):
            return False
        return bool(self.key_listener.stop())

    def _resume_key_listener_after_processing(self, should_resume):
        """Resume the key listener only if this processing step paused it."""
        if should_resume and getattr(self, 'key_listener', None):
            self.key_listener.start()

    def on_transcription_complete(self, result):
        """Process transcription with or without LLM based on activation type."""
        listener_was_running = False
        try:
            # Temporarily disable key listener
            listener_was_running = self._pause_key_listener_for_processing()

            recording_mode = ConfigManager.get_config_value('recording_options', 'recording_mode')
            if self.use_llm and self.llm_processor and recording_mode in ('press_to_toggle', 'hold_to_record', 'continuous', 'voice_activity_detection'):
                try:
                    # Get the system message based on the mode
                    if self.is_instruction_mode:
                        base_message = ConfigManager.get_config_value("llm_post_processing", "instruction_system_message")
                        file_path = ConfigManager.get_config_value("llm_post_processing", "instruction_system_message_file_path")
                        mode_name = "instruction"
                    else:
                        base_message = ConfigManager.get_config_value("llm_post_processing", "system_prompt")
                        file_path = ConfigManager.get_config_value("llm_post_processing", "system_prompt_file_path")
                        mode_name = "cleanup"
                    
                    # Start with a clean system message
                    system_message = base_message.strip() if base_message else ""
                    log_cleanup_prompt = ConfigManager.should_log_cleanup_prompt()
                    
                    if mode_name == "cleanup" and not log_cleanup_prompt:
                        ConfigManager.console_print("Retrieved cleanup base message from settings", verbose=True)
                    else:
                        ConfigManager.console_print(f"Retrieved {mode_name} base message from settings: {system_message}", verbose=True)
                    
                    if not file_path:
                        ConfigManager.console_print("No file path set, using only the system message from settings")
                    elif not os.path.exists(file_path):
                        ConfigManager.console_print(f"File path set but file not found: {file_path}, using only the system message from settings")
                    
                    # Append file contents if file path exists and is not empty
                    if file_path and os.path.exists(file_path):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as file:
                                file_content = file.read().strip()
                                if file_content:  # Only append if file has content
                                    if system_message:
                                        system_message = f"{system_message}\n\n{file_content}"
                                    else:
                                        system_message = file_content
                                    ConfigManager.console_print(f"Added fresh file content from {file_path}", verbose=True)
                        except Exception as e:
                            ConfigManager.console_print(f"Error reading system message file: {str(e)}")
                    
                    if not system_message:
                        ConfigManager.console_print("Warning: No system message found, using original transcription")
                    else:
                        if mode_name == "cleanup" and not log_cleanup_prompt:
                            ConfigManager.console_print("Final cleanup system message prepared for LLM", verbose=True)
                        else:
                            ConfigManager.console_print(f"Final system message being sent to LLM: {system_message}", verbose=True)
                        original_result = result
                        processed_result = self.llm_processor.process_text(result, system_message, mode=mode_name)
                        if processed_result is not None:
                            ConfigManager.console_print(f"Cleanup raw output: {processed_result}", verbose=True)
                        if processed_result:
                            candidate_result = processed_result.strip()
                            if mode_name == "cleanup":
                                rejection_reason = LLMProcessor.get_cleanup_rejection_reason(original_result, candidate_result)
                                if rejection_reason:
                                    ConfigManager.console_print(
                                        f"Cleanup output rejected; falling back to original transcription ({rejection_reason})."
                                    )
                                    result = original_result
                                else:
                                    result = candidate_result
                            else:
                                result = candidate_result

                            if result == original_result:
                                ConfigManager.console_print("Cleanup output matches original transcription", verbose=True)
                            else:
                                ConfigManager.console_print("Cleanup output differs from original transcription", verbose=True)
                        else:
                            ConfigManager.console_print("LLM processing failed or returned empty output, using original transcription")
                            result = original_result
                    
                except Exception as e:
                    ConfigManager.console_print(f"Error processing text through LLM: {str(e)}")
                    ConfigManager.console_print("Falling back to original transcription after LLM processing error.")

            # Type the result
            typed_result = False
            try:
                ConfigManager.console_print(f"Typing transcription result length: {len(result)}", verbose=True)
                self.input_simulator.typewrite(result)
                typed_result = True
            except Exception as e:
                ConfigManager.console_print(f"Error typing transcription result: {str(e)}")
            finally:
                ConfigManager.console_print(f"Typed transcription result: {typed_result}", verbose=True)
            
            if ConfigManager.get_config_value('misc', 'noise_on_completion'):
                AudioPlayer(os.path.join('assets', 'beep.wav')).play(block=True)

            if ConfigManager.get_config_value('recording_options', 'recording_mode') == 'continuous':
                self.start_result_thread()

        finally:
            # Re-enable key listener
            self._resume_key_listener_after_processing(listener_was_running)

    def handle_text_cleanup(self):
        """Handle the text selection cleanup shortcut."""
        if not self.llm_processor:
            return

        listener_was_running = self._pause_key_listener_for_processing()
        
        # Store all clipboard formats
        saved_formats = {}
        if not InputSimulator.safe_open_clipboard():
            ConfigManager.console_print("Unable to open clipboard for text cleanup.")
            self._resume_key_listener_after_processing(listener_was_running)
            return
        
        try:
            saved_formats = InputSimulator.capture_open_clipboard_formats()
            ConfigManager.console_print(
                f"Clipboard cleanup captured formats: {InputSimulator.describe_clipboard_formats(saved_formats)}",
                verbose=True,
            )
            
            # Get the text content
            clipboard_text = saved_formats.get(win32con.CF_UNICODETEXT)
            InputSimulator.safe_close_clipboard()

            if not clipboard_text:
                ConfigManager.console_print("No text in clipboard")
                return

            if not isinstance(clipboard_text, str):
                ConfigManager.console_print(
                    f"Clipboard cleanup expected text, got {type(clipboard_text).__name__}; skipping cleanup."
                )
                return
            
            ConfigManager.console_print(f"Processing clipboard text: {clipboard_text[:100]}...", verbose=True)
            
            # Get the base system message
            base_message = ConfigManager.get_config_value("llm_post_processing", "text_cleanup_system_message")
            file_path = ConfigManager.get_config_value("llm_post_processing", "text_cleanup_system_message_file_path")
            
            # Start with a clean system message
            system_message = base_message.strip() if base_message else ""
            log_cleanup_prompt = ConfigManager.should_log_cleanup_prompt()
            
            if log_cleanup_prompt:
                ConfigManager.console_print(f"Base cleanup system message: {system_message}", verbose=True)
            else:
                ConfigManager.console_print("Base cleanup system message retrieved", verbose=True)
            
            # Append file contents if file path exists and is not empty
            if file_path and os.path.exists(file_path):
                try:
                    ConfigManager.console_print(f"Reading cleanup instructions from file: {file_path}", verbose=True)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        file_content = file.read().strip()
                        if file_content:  # Only append if file has content
                            if system_message:
                                system_message = f"{system_message}\n\n{file_content}"
                            else:
                                system_message = file_content
                            ConfigManager.console_print("Successfully added file content to cleanup instructions", verbose=True)
                except Exception as e:
                    ConfigManager.console_print(f"Error reading cleanup system message file: {str(e)}")
            
            if not system_message:
                ConfigManager.console_print("Warning: No cleanup system message found")
                return
            
            if log_cleanup_prompt:
                ConfigManager.console_print(f"Final cleanup system message: {system_message}", verbose=True)
            else:
                ConfigManager.console_print("Final cleanup system message prepared", verbose=True)
            
            # Run through LLM cleanup
            cleaned_text = self.llm_processor.process_text(clipboard_text, system_message, mode="cleanup")
            ConfigManager.console_print(f"Cleanup output (clipboard): {cleaned_text}", verbose=True)

            if cleaned_text:
                cleaned_text = cleaned_text.strip()
                rejection_reason = LLMProcessor.get_cleanup_rejection_reason(clipboard_text, cleaned_text)
                if rejection_reason:
                    ConfigManager.console_print(
                        f"Clipboard cleanup output rejected; leaving original text untouched ({rejection_reason})."
                    )
                    cleaned_text = clipboard_text
            
            if cleaned_text and cleaned_text != clipboard_text:
                paste_succeeded = False
                try:
                    # Simulate keyboard events with proper cleanup
                    keyboard = Controller()
                    use_direct_typing = InputSimulator.has_image_clipboard_content(saved_formats)
                    if use_direct_typing:
                        ConfigManager.console_print(
                            "Clipboard cleanup detected image formats; using direct typing instead of clipboard paste to avoid pasting the image back into the target app."
                        )
                    
                    if not use_direct_typing:
                        # First set the cleaned text to clipboard
                        if not InputSimulator.safe_open_clipboard():
                            raise RuntimeError("Unable to open clipboard for cleanup paste")
                        win32clipboard.EmptyClipboard()
                        win32clipboard.SetClipboardText(cleaned_text, win32con.CF_UNICODETEXT)
                        InputSimulator.safe_close_clipboard()
                    
                    try:
                        # Delete selected text and paste cleaned text
                        keyboard.press(Key.delete)
                        keyboard.release(Key.delete)
                        time.sleep(0.1)  # Small delay after delete

                        if use_direct_typing:
                            self.input_simulator.typewrite_direct(cleaned_text)
                        else:
                            with keyboard.pressed(Key.ctrl):
                                keyboard.press('v')
                                keyboard.release('v')

                        paste_succeeded = True
                        
                        if ConfigManager.get_config_value('misc', 'noise_on_completion'):
                            AudioPlayer(os.path.join('assets', 'beep.wav')).play(block=True)
                            
                    finally:
                        # Ensure all keys are released
                        keyboard.release(Key.ctrl)
                        keyboard.release('v')
                        keyboard.release(Key.delete)
                        
                        # Restore original clipboard content
                        if use_direct_typing:
                            ConfigManager.console_print(
                                "Skipped clipboard restore after cleanup because direct typing was used.",
                                verbose=True,
                            )
                        else:
                            restore_delay = InputSimulator.get_clipboard_restore_delay(saved_formats)
                            ConfigManager.console_print(
                                f"Waiting {restore_delay:.2f}s before restoring clipboard after cleanup paste.",
                                verbose=True,
                            )
                            time.sleep(restore_delay)

                            if InputSimulator.safe_open_clipboard():
                                try:
                                    current_text = InputSimulator.get_open_clipboard_text()
                                    if InputSimulator.should_restore_clipboard(current_text, cleaned_text):
                                        win32clipboard.EmptyClipboard()
                                        restored_formats = InputSimulator.restore_open_clipboard_formats(saved_formats)
                                        ConfigManager.console_print(
                                            f"Restored clipboard formats after cleanup paste: {InputSimulator.describe_clipboard_formats(restored_formats)}",
                                            verbose=True,
                                        )
                                    else:
                                        ConfigManager.console_print(
                                            "Skipping clipboard restore after cleanup paste because clipboard contents changed before restore.",
                                            verbose=True,
                                        )
                                finally:
                                    InputSimulator.safe_close_clipboard()
                            else:
                                ConfigManager.console_print("Unable to reopen clipboard for cleanup restore.")
                        
                    # Clear the key chord state
                    self.key_listener.text_cleanup_chord.pressed_keys.clear()
                    
                except Exception as e:
                    ConfigManager.console_print(f"Error simulating keyboard: {str(e)}")
                finally:
                    ConfigManager.console_print(f"Pasted cleaned clipboard text: {paste_succeeded}", verbose=True)
            else:
                ConfigManager.console_print("Text unchanged after LLM processing")
                ConfigManager.console_print("Cleanup output empty or unchanged; no paste performed", verbose=True)
            
        except Exception as e:
            ConfigManager.console_print(f"Error cleaning text: {str(e)}")
        finally:
            InputSimulator.safe_close_clipboard()
            # Ensure key listener is restarted
            self._resume_key_listener_after_processing(listener_was_running)

    def run(self):
        """
        Start the application.
        """
        sys.exit(self.app.exec_())


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='WhisperWriter - AI-powered speech-to-text application',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '-V', '--verbose',
        action='store_true',
        help='Enable verbose logging including full prompts, system messages, and API responses'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    app = WhisperWriterApp(verbose_mode=args.verbose)
    app.run()
