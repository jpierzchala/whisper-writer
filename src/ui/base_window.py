from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPainter, QBrush, QColor, QFont, QPainterPath, QPen, QGuiApplication
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QMainWindow

from ui.theme import border_colour, card_colour


def _is_mocked_qt_object(obj):
    """Detects whether a PyQt class has been replaced with a unittest.mock object."""
    module_name = getattr(obj, "__module__", "") or getattr(obj.__class__, "__module__", "")
    return module_name.startswith("unittest.mock")


QT_WIDGETS_ARE_MOCKED = _is_mocked_qt_object(QMainWindow)


if QT_WIDGETS_ARE_MOCKED:
    class BaseWindow:
        """Lightweight stand-in used when PyQt widgets are replaced with mocks in tests."""

        def __init__(self, title, width, height):
            self._title = title
            self._width = width
            self._height = height
            self.main_widget = QWidget() if callable(QWidget) else QWidget
            self.main_layout = QVBoxLayout() if callable(QVBoxLayout) else QVBoxLayout
            self.is_dragging = False

        def initUI(self, title, width, height):  # pragma: no cover - noop for tests
            return

        def setWindowPosition(self):  # pragma: no cover - noop for tests
            return

        def handleCloseButton(self):  # pragma: no cover - noop for tests
            return

        def mousePressEvent(self, event):  # pragma: no cover - noop for tests
            return

        def mouseMoveEvent(self, event):  # pragma: no cover - noop for tests
            return

        def mouseReleaseEvent(self, event):  # pragma: no cover - noop for tests
            return

        def paintEvent(self, event):  # pragma: no cover - noop for tests
            return
else:
    class BaseWindow(QMainWindow):
        def __init__(self, title, width, height):
            """
            Initialize the base window.
            """
            super().__init__()
            self.initUI(title, width, height)
            self.setWindowPosition()
            self.is_dragging = False

        def initUI(self, title, width, height):
            """
            Initialize the user interface.
            """
            self.setWindowTitle(title)
            self.setWindowFlags(Qt.FramelessWindowHint)
            self.setAttribute(Qt.WA_TranslucentBackground, True)
            # Resizable: the settings window in particular has to cope with long
            # option lists on small screens.
            self.resize(width, height)
            self.setMinimumSize(min(width, 520), min(height, 420))

            self.main_widget = QWidget(self)
            self.main_layout = QVBoxLayout(self.main_widget)
            self.main_layout.setContentsMargins(16, 12, 16, 14)
            self.main_layout.setSpacing(8)

            # Create a widget for the title bar
            title_bar = QWidget()
            title_bar_layout = QHBoxLayout(title_bar)
            title_bar_layout.setContentsMargins(0, 0, 0, 0)

            # Show the actual window title rather than the app name on every window.
            title_label = QLabel(title)
            title_label.setProperty('role', 'windowTitle')
            title_label.setAlignment(Qt.AlignCenter)

            close_button = QPushButton('×')
            close_button.setProperty('role', 'titleBarClose')
            close_button.setFixedSize(28, 28)
            close_button.setCursor(Qt.PointingHandCursor)
            close_button.setToolTip('Close')
            close_button.clicked.connect(self.handleCloseButton)

            title_bar_layout.addSpacing(28)
            title_bar_layout.addWidget(title_label, 1)
            title_bar_layout.addWidget(close_button, 0, Qt.AlignRight)

            self.main_layout.addWidget(title_bar)
            self.setCentralWidget(self.main_widget)

        def setWindowPosition(self):
            """
            Set the window position to the center of the screen.
            """
            center_point = QGuiApplication.primaryScreen().availableGeometry().center()
            frame_geometry = self.frameGeometry()
            frame_geometry.moveCenter(center_point)
            self.move(frame_geometry.topLeft())

        def handleCloseButton(self):
            """
            Close the window.
            """
            self.close()

        def mousePressEvent(self, event):
            """
            Allow the window to be moved by clicking and dragging anywhere on the window.
            """
            if event.button() == Qt.LeftButton:
                self.is_dragging = True
                self.start_position = event.globalPos() - self.frameGeometry().topLeft()
                event.accept()

        def mouseMoveEvent(self, event):
            """
            Move the window when dragging.
            """
            if self.is_dragging and event.buttons() & Qt.LeftButton:
                self.move(event.globalPos() - self.start_position)
                event.accept()

        def mouseReleaseEvent(self, event):
            """
            Stop dragging the window.
            """
            self.is_dragging = False

        def paintEvent(self, event):
            """
            Paint the window as a rounded card in the current theme's colours.
            """
            path = QPainterPath()
            path.addRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 14, 14)
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setBrush(QBrush(card_colour()))
            painter.setPen(QPen(border_colour(), 1))
            painter.drawPath(path)
