import os
import sys
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ui.base_window import BaseWindow
from ui.theme import apply_theme

class MainWindow(BaseWindow):
    openSettings = pyqtSignal()
    startListening = pyqtSignal()
    closeApp = pyqtSignal()

    def __init__(self):
        """
        Initialize the main window.
        """
        super().__init__('WhisperWriter', 360, 210)
        apply_theme(self)
        self.initMainUI()

    def initMainUI(self):
        """
        Initialize the main user interface.
        """
        hint = QLabel('Start listening, then use your shortcut to dictate.')
        hint.setProperty('role', 'hint')
        hint.setAlignment(Qt.AlignCenter)
        hint.setWordWrap(True)

        start_btn = QPushButton('Start listening')
        start_btn.setProperty('role', 'primary')
        start_btn.setMinimumHeight(38)
        start_btn.clicked.connect(self.startPressed)

        settings_btn = QPushButton('Settings')
        settings_btn.setMinimumHeight(38)
        settings_btn.clicked.connect(self.openSettings.emit)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        button_layout.addWidget(start_btn, 1)
        button_layout.addWidget(settings_btn, 1)

        self.main_layout.addStretch(1)
        self.main_layout.addWidget(hint)
        self.main_layout.addSpacing(10)
        self.main_layout.addLayout(button_layout)
        self.main_layout.addStretch(1)

    def closeEvent(self, event):
        """
        Close the application when the main window is closed.
        """
        self.closeApp.emit()

    def startPressed(self):
        """
        Emit the startListening signal when the start button is pressed.
        """
        self.startListening.emit()
        self.hide()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
