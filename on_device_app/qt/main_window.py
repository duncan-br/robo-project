from __future__ import annotations

from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget

from on_device_app.qt.live_widget import LiveInferenceWidget
from on_device_app.qt.review_widget import ReviewWidget
from on_device_app.api_client import ApiClient


class MainWindow(QMainWindow):
    def __init__(self, api: ApiClient) -> None:
        super().__init__()
        self.setWindowTitle("Open Robo Vision - On-device")
        self.resize(1320, 860)
        tabs = QTabWidget(self)
        self.setCentralWidget(tabs)

        self._review = ReviewWidget(api)
        self._live = LiveInferenceWidget(api, self._review.refresh)
        tabs.addTab(self._live, "Live Stream")
        tabs.addTab(self._review, "Review Queue")


def launch_qt(api: ApiClient) -> int:
    app = QApplication.instance() or QApplication([])
    win = MainWindow(api)
    win.show()
    return app.exec()

