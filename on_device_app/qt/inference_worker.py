from __future__ import annotations

import threading
from typing import Any

from PySide6.QtCore import QThread, Signal

from on_device_app.api_client import ApiClient
from on_device_app.dto import InferenceSettings


class InferenceWorker(QThread):
    result_ready = Signal(dict)
    failed = Signal(str)

    def __init__(self, api: ApiClient, parent=None) -> None:
        super().__init__(parent)
        self._api = api
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._running = True
        self._pending: tuple[bytes, InferenceSettings] | None = None

    def submit(self, frame_jpeg: bytes, settings: InferenceSettings) -> None:
        with self._lock:
            self._pending = (frame_jpeg, settings)
        self._wake.set()

    def stop(self) -> None:
        self._running = False
        self._wake.set()

    def run(self) -> None:
        while self._running:
            self._wake.wait(timeout=0.2)
            self._wake.clear()
            if not self._running:
                break
            payload: tuple[bytes, InferenceSettings] | None
            with self._lock:
                payload = self._pending
                self._pending = None
            if payload is None:
                continue
            frame_jpeg, settings = payload
            try:
                result: dict[str, Any] = self._api.ingest_ros2_frame_jpeg(frame_jpeg, settings)
            except Exception as exc:
                self.failed.emit(str(exc))
                continue
            self.result_ready.emit(dict(result))