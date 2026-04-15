from __future__ import annotations

import base64
import json
import logging

import cv2
import numpy as np
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtNetwork import QNetworkRequest
from PySide6.QtWebSockets import QWebSocket
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from on_device_app.api_client import ApiClient
from on_device_app.dto import InferenceSettings
from on_device_app.qt.inference_worker import InferenceWorker
from on_device_app.qt.shared import apply_roi_overlay, draw_detections_on_pixmap, load_scene_with_bbox
from on_device_app.ros2 import RosImageStreamHandler, _DEFAULT_TOPIC, detect_image_topics

log = logging.getLogger(__name__)
_INFER_EVERY_N = 1


class _SettingsDialog(QDialog):
    def __init__(self, parent: QWidget | None = None, preview_pixmap: QPixmap | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Stream Settings")
        self.setMinimumWidth(420)

        self._base_preview: QPixmap | None = None
        if preview_pixmap is not None and not preview_pixmap.isNull():
            self._base_preview = QPixmap(preview_pixmap)

        layout = QVBoxLayout(self)
        self._preview_label = QLabel()
        self._preview_label.setAlignment(Qt.AlignCenter)
        self._preview_label.setMinimumHeight(180)
        self._preview_label.setStyleSheet("background-color: #1a1a1a; color: #aaa;")
        if self._base_preview is not None:
            self._preview_label.setText("")
        else:
            self._preview_label.setText("No preview — upload a stream file (upload source) to see ROI on a frame.")
        layout.addWidget(self._preview_label)

        form = QFormLayout()

        self.conf = QLineEdit("0.2")
        self.high = QLineEdit("0.35")
        self.merge = QLineEdit("average")
        self.avg = QLineEdit("8")

        self.roi_x_slider = QSlider(Qt.Horizontal)
        self.roi_y_slider = QSlider(Qt.Horizontal)
        self.roi_w_slider = QSlider(Qt.Horizontal)
        self.roi_h_slider = QSlider(Qt.Horizontal)
        self.roi_x_value = QLabel("0.00")
        self.roi_y_value = QLabel("0.00")
        self.roi_w_value = QLabel("1.00")
        self.roi_h_value = QLabel("1.00")

        for sl in (self.roi_x_slider, self.roi_y_slider, self.roi_w_slider, self.roi_h_slider):
            sl.valueChanged.connect(self._refresh_preview_roi)

        form.addRow("Objectness threshold", self.conf)
        form.addRow("Auto-save threshold", self.high)
        form.addRow("Merging mode", self.merge)
        form.addRow("Avg count", self.avg)
        form.addRow("ROI x", self._build_slider_row(self.roi_x_slider, self.roi_x_value, 0))
        form.addRow("ROI y", self._build_slider_row(self.roi_y_slider, self.roi_y_value, 0))
        form.addRow("ROI w", self._build_slider_row(self.roi_w_slider, self.roi_w_value, 100))
        form.addRow("ROI h", self._build_slider_row(self.roi_h_slider, self.roi_h_value, 100))
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._refresh_preview_roi()

    def settings(self) -> InferenceSettings:
        roi_x, roi_y, roi_w, roi_h = self.roi_values()
        return InferenceSettings(
            conf_thresh=float(self.conf.text().strip() or "0.2"),
            high_conf_min=float(self.high.text().strip() or "0.35"),
            merging_mode=self.merge.text().strip() or "average",
            avg_count=int(self.avg.text().strip() or "8"),
            roi_x=roi_x,
            roi_y=roi_y,
            roi_w=roi_w,
            roi_h=roi_h,
        )

    def roi_values(self) -> tuple[float, float, float, float]:
        roi_x = self.roi_x_slider.value() / 100.0
        roi_y = self.roi_y_slider.value() / 100.0
        roi_w = self.roi_w_slider.value() / 100.0
        roi_h = self.roi_h_slider.value() / 100.0
        if roi_x + roi_w > 1.0:
            roi_w = max(0.0, 1.0 - roi_x)
        if roi_y + roi_h > 1.0:
            roi_h = max(0.0, 1.0 - roi_y)
        return roi_x, roi_y, roi_w, roi_h

    def refresh_preview_roi(self) -> None:
        self._refresh_preview_roi()

    def _refresh_preview_roi(self) -> None:
        if self._base_preview is None:
            return
        rx, ry, rw, rh = self.roi_values()
        pix = QPixmap(self._base_preview)
        apply_roi_overlay(pix, rx, ry, rw, rh)
        max_w = min(640, max(320, self.width() - 40))
        scaled = pix.scaledToWidth(max_w, Qt.SmoothTransformation)
        self._preview_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh_preview_roi()

    @staticmethod
    def _build_slider_row(slider: QSlider, value_label: QLabel, initial: int) -> QWidget:
        slider.setRange(0, 100)
        slider.setValue(initial)
        value_label.setFixedWidth(48)
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        slider.valueChanged.connect(lambda v, lbl=value_label: lbl.setText(f"{v / 100.0:.2f}"))
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(slider, stretch=1)
        row_layout.addWidget(value_label)
        return row


class _ReviewItemDialog(QDialog):
    def __init__(self, parent: QWidget | None, api: ApiClient, item: dict) -> None:
        super().__init__(parent)
        self.setWindowTitle("Review item")
        self.setMinimumSize(520, 480)
        self._api = api
        self._item = item
        self.action_taken = False

        layout = QVBoxLayout(self)
        self._scene = QGraphicsScene(self)
        self._view = QGraphicsView(self._scene)
        layout.addWidget(self._view, stretch=1)

        row = QHBoxLayout()
        row.addWidget(QLabel("Class"))
        self._combo = QComboBox()
        self._combo.setEditable(True)
        row.addWidget(self._combo, stretch=1)
        self._add_new = QCheckBox("Add as new class")
        row.addWidget(self._add_new)
        layout.addLayout(row)

        self._hint = QLineEdit()
        self._hint.setPlaceholderText("Class name override")
        self._hint.setText(str(item.get("class_name_suggested") or ""))
        layout.addWidget(self._hint)

        self._reload_class_list()
        path = str(item.get("image_path") or "")
        load_scene_with_bbox(self._scene, path, item)
        self._view.fitInView(self._scene.itemsBoundingRect(), Qt.KeepAspectRatio)

        actions = QHBoxLayout()
        confirm_btn = QPushButton("Confirm")
        skip_btn = QPushButton("Skip")
        cancel_btn = QPushButton("Cancel")
        actions.addWidget(confirm_btn)
        actions.addWidget(skip_btn)
        actions.addStretch(1)
        actions.addWidget(cancel_btn)
        layout.addLayout(actions)

        confirm_btn.clicked.connect(self._on_confirm)
        skip_btn.clicked.connect(self._on_skip)
        cancel_btn.clicked.connect(self.reject)

    def _reload_class_list(self) -> None:
        self._combo.clear()
        try:
            for name in self._api.classes():
                self._combo.addItem(name)
        except Exception:
            pass

    def _class_name(self) -> str:
        return self._hint.text().strip() or self._combo.currentText().strip()

    def _on_confirm(self) -> None:
        name = self._class_name()
        if not name:
            QMessageBox.information(self, "Class required", "Provide a class name before confirm.")
            return
        queue_id = str(self._item.get("queue_id", ""))
        if not queue_id:
            return
        try:
            self._api.confirm_review_item(queue_id, name, create_if_missing=self._add_new.isChecked())
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Confirm failed", str(exc))
            return
        self.action_taken = True
        self.accept()

    def _on_skip(self) -> None:
        queue_id = str(self._item.get("queue_id", ""))
        if not queue_id:
            return
        try:
            self._api.skip_review_item(queue_id)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Skip failed", str(exc))
            return
        self.action_taken = True
        self.accept()


class LiveInferenceWidget(QWidget):
    def __init__(self, api: ApiClient, refresh_review_cb) -> None:
        super().__init__()
        self._api = api
        self._refresh_review = refresh_review_cb
        self._uploaded_path = ""
        self._low_conf_by_id: dict[str, dict] = {}
        self._current_settings = InferenceSettings()
        self._stream_active = False
        self._mode = "ros2_live"
        self._latest_detections: list[dict] = []
        self._last_frame_bgr: np.ndarray | None = None
        self._frame_count = 0
        self._ros_handler: RosImageStreamHandler | None = None
        self._worker: InferenceWorker | None = None
        self._ws = QWebSocket()
        self._ws.connected.connect(self._on_ws_connected)
        self._ws.disconnected.connect(self._on_ws_disconnected)
        self._ws.textMessageReceived.connect(self._on_ws_message)

        layout = QVBoxLayout(self)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("ROS2 Live Stream", "ros2_live")
        self._mode_combo.addItem("Upload File", "upload")
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self._mode_combo)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)

        self._upload_row_widget = QWidget()
        top_row = QHBoxLayout(self._upload_row_widget)
        top_row.setContentsMargins(0, 0, 0, 0)
        self._picked = QLineEdit()
        self._picked.setPlaceholderText("Select .bag or video file for backend stream simulation")
        browse = QPushButton("Choose file")
        browse.clicked.connect(self._pick_stream_file)
        upload = QPushButton("Upload")
        upload.clicked.connect(self._upload_stream_file)
        top_row.addWidget(self._picked, stretch=1)
        top_row.addWidget(browse)
        top_row.addWidget(upload)
        layout.addWidget(self._upload_row_widget)

        source_row = QHBoxLayout()
        detected = detect_image_topics()
        self._ros_topic = QLineEdit(detected[0] if detected else "")
        self._ros_topic.setPlaceholderText(_DEFAULT_TOPIC)
        settings_btn = QPushButton("Settings...")
        settings_btn.clicked.connect(self._open_settings)
        start_btn = QPushButton("Start stream")
        stop_btn = QPushButton("Stop stream")
        start_btn.clicked.connect(self._start_stream)
        stop_btn.clicked.connect(self._stop_stream)
        source_row.addWidget(QLabel("ROS2 topic"))
        source_row.addWidget(self._ros_topic, stretch=1)
        source_row.addWidget(settings_btn)
        source_row.addWidget(start_btn)
        source_row.addWidget(stop_btn)
        layout.addLayout(source_row)

        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self._video = QLabel("No stream frame yet.")
        self._video.setAlignment(Qt.AlignCenter)
        self._video.setMinimumSize(840, 520)
        left_layout.addWidget(self._video, stretch=1)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("Low-confidence queue (live):"))
        self._low_list = QListWidget()
        self._low_list.itemClicked.connect(self._on_low_item_activated)
        right_layout.addWidget(self._low_list, stretch=1)
        clear_row = QHBoxLayout()
        clear_btn = QPushButton("Clear list")
        clear_btn.clicked.connect(self._clear_low_queue)
        clear_row.addWidget(clear_btn)
        clear_row.addStretch(1)
        right_layout.addLayout(clear_row)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, stretch=1)

        self._status = QLabel("Idle")
        self._status.setAlignment(Qt.AlignLeft)
        layout.addWidget(self._status)
        self._sync_mode_ui()

    def _pick_stream_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select stream file",
            "",
            "Streams (*.db3 *.bag *.mp4 *.avi *.mov *.mkv *.zip)",
        )
        if path:
            self._picked.setText(path)

    def _upload_stream_file(self) -> None:
        path = self._picked.text().strip()
        if not path:
            QMessageBox.information(self, "File required", "Choose a .db3/.bag or video file first.")
            return
        try:
            res = self._api.upload_stream_file(path)
            self._uploaded_path = str(res.get("path", ""))
            self._status.setText(f"Uploaded stream file: {self._uploaded_path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Upload failed", str(exc))

    def _open_settings(self) -> None:
        preview: QPixmap | None = None
        source = "upload" if self._mode == "upload" else "ros2"
        topic = self._ros_topic.text().strip()
        if self._mode == "upload":
            try:
                res = self._api.stream_preview(source=source, topic=topic)
                b64 = str(res.get("frame_jpeg_b64", ""))
                if b64:
                    raw = base64.b64decode(b64)
                    image = QImage.fromData(raw, "JPG")
                    p = QPixmap.fromImage(image)
                    if not p.isNull():
                        preview = p
            except Exception as exc:  # noqa: BLE001
                log.warning("Stream preview unavailable: %s", exc)
        elif self._last_frame_bgr is not None:
            preview = self._pixmap_from_bgr(self._last_frame_bgr)

        dlg = _SettingsDialog(self, preview_pixmap=preview)
        s = self._current_settings
        dlg.conf.setText(str(s.conf_thresh))
        dlg.high.setText(str(s.high_conf_min))
        dlg.merge.setText(s.merging_mode)
        dlg.avg.setText(str(s.avg_count))
        dlg.roi_x_slider.setValue(int(s.roi_x * 100))
        dlg.roi_y_slider.setValue(int(s.roi_y * 100))
        dlg.roi_w_slider.setValue(int(s.roi_w * 100))
        dlg.roi_h_slider.setValue(int(s.roi_h * 100))
        dlg.refresh_preview_roi()
        if dlg.exec() != QDialog.Accepted:
            return
        self._current_settings = dlg.settings()
        if self._stream_active:
            try:
                self._api.update_stream_settings(self._current_settings)
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to push settings update: %s", exc)

    def _clear_low_queue(self) -> None:
        self._low_conf_by_id.clear()
        self._render_low_list()

    def _start_stream(self) -> None:
        try:
            topic = self._ros_topic.text().strip()
            self._stop_stream()
            if self._mode == "upload":
                self._api.start_stream(source="upload", settings=self._current_settings, topic=topic)
                self._stream_active = True
                req = QNetworkRequest(QUrl(self._ws_url()))
                self._ws.open(req)
                self._status.setText("Starting upload stream...")
                return
            self._start_ros_live(topic=topic or _DEFAULT_TOPIC)
            self._status.setText(f"Starting ROS2 live stream from {topic or _DEFAULT_TOPIC}...")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Start failed", str(exc))

    def _stop_stream(self) -> None:
        self._stream_active = False
        self._latest_detections = []
        self._last_frame_bgr = None
        self._frame_count = 0
        self._clear_low_queue()
        self._stop_ros_live()
        if self._mode == "upload":
            try:
                self._api.stop_stream()
            except Exception:
                pass
            self._ws.abort()
        self._status.setText("Stream stopped.")

    def _on_ws_connected(self) -> None:
        self._status.setText("WebSocket connected. Receiving backend frames.")

    def _on_ws_disconnected(self) -> None:
        if self._mode == "upload":
            self._stream_active = False
            self._status.setText("WebSocket disconnected.")

    def _on_ws_message(self, message: str) -> None:
        payload = json.loads(message)
        frame_b64 = str(payload.get("frame_jpeg_b64", ""))
        detections = list(payload.get("detections", []))
        low_items = list(payload.get("low_confidence_items", []))
        for it in low_items:
            qid = str(it.get("queue_id") or "")
            if qid:
                self._low_conf_by_id[qid] = it
        self._render_low_list()

        if frame_b64:
            raw = base64.b64decode(frame_b64)
            image = QImage.fromData(raw, "JPG")
            pix = QPixmap.fromImage(image)
            if not pix.isNull():
                pix = draw_detections_on_pixmap(pix, detections)
                apply_roi_overlay(
                    pix,
                    self._current_settings.roi_x,
                    self._current_settings.roi_y,
                    self._current_settings.roi_w,
                    self._current_settings.roi_h,
                )
                self._video.setPixmap(
                    pix.scaled(self._video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
        n_low = len(self._low_conf_by_id)
        self._status.setText(
            f"Frame #{payload.get('frame_index', 0)} | det={len(detections)} | low={n_low} (accum.) | "
            f"fps={float(payload.get('stream_fps', 0.0)):.2f}"
        )
        if low_items:
            self._refresh_review()

    def _on_mode_changed(self) -> None:
        self._mode = str(self._mode_combo.currentData() or "upload")
        self._stop_stream()
        self._sync_mode_ui()

    def _sync_mode_ui(self) -> None:
        is_upload = self._mode == "upload"
        self._upload_row_widget.setVisible(is_upload)

    def _start_ros_live(self, topic: str) -> None:
        self._worker = InferenceWorker(self._api, self)
        self._worker.result_ready.connect(self._on_ros_inference_result)
        self._worker.failed.connect(self._on_ros_worker_failed)
        self._worker.start()

        self._ros_handler = RosImageStreamHandler(topic=topic, qos_depth=10, parent=self)
        self._ros_handler.frame_ready.connect(self._on_ros_frame_ready)
        self._ros_handler.failed.connect(self._on_ros_handler_failed)
        self._ros_handler.start()
        self._stream_active = True

    def _stop_ros_live(self) -> None:
        if self._ros_handler is not None:
            self._ros_handler.stop()
            self._ros_handler.wait(1500)
            self._ros_handler = None
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait(1500)
            self._worker = None

    def _on_ros_frame_ready(self, frame_bgr: np.ndarray) -> None:
        if not self._stream_active:
            return
        self._frame_count += 1
        self._last_frame_bgr = np.asarray(frame_bgr).copy()

        pix = self._pixmap_from_bgr(self._last_frame_bgr)
        if self._latest_detections:
            pix = draw_detections_on_pixmap(pix, self._latest_detections)
        apply_roi_overlay(
            pix,
            self._current_settings.roi_x,
            self._current_settings.roi_y,
            self._current_settings.roi_w,
            self._current_settings.roi_h,
        )
        self._video.setPixmap(pix.scaled(self._video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        if self._worker is not None and (self._frame_count % _INFER_EVERY_N == 0):
            ok, encoded = cv2.imencode(".jpg", self._last_frame_bgr)
            if ok:
                self._worker.submit(encoded.tobytes(), self._current_settings)

    def _on_ros_inference_result(self, payload: dict) -> None:
        self._latest_detections = list(payload.get("detections", []))
        low_items = list(payload.get("low_confidence_items", []))
        for it in low_items:
            qid = str(it.get("queue_id") or "")
            if qid:
                self._low_conf_by_id[qid] = it
        self._render_low_list()

        payload.pop("_frame_jpeg", None)

        if self._last_frame_bgr is not None:
            pix = self._pixmap_from_bgr(self._last_frame_bgr)
            if self._latest_detections:
                pix = draw_detections_on_pixmap(pix, self._latest_detections)
            apply_roi_overlay(
                pix,
                self._current_settings.roi_x,
                self._current_settings.roi_y,
                self._current_settings.roi_w,
                self._current_settings.roi_h,
            )
            self._video.setPixmap(pix.scaled(self._video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        n_low = len(self._low_conf_by_id)
        self._status.setText(
            f"ROS2 frame #{self._frame_count} | det={len(self._latest_detections)} | low={n_low} (accum.)"
        )
        if low_items:
            self._refresh_review()

    def _on_ros_handler_failed(self, err: str) -> None:
        QMessageBox.critical(self, "ROS2 stream failed", err)
        self._stop_stream()

    def _on_ros_worker_failed(self, err: str) -> None:
        log.warning("ROS2 inference worker failed: %s", err)

    @staticmethod
    def _pixmap_from_bgr(frame_bgr: np.ndarray) -> QPixmap:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        image = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(image)

    def _render_low_list(self) -> None:
        self._low_list.clear()
        for item in self._low_conf_by_id.values():
            qid = str(item.get("queue_id", ""))
            label = (
                f"{item.get('class_name_suggested', 'unknown')} "
                f"(score={float(item.get('score', 0.0)):.2f}) "
                f"id={qid[:8]}…"
            )
            lw_item = QListWidgetItem(label)
            lw_item.setData(Qt.ItemDataRole.UserRole, qid)
            self._low_list.addItem(lw_item)

    def _on_low_item_activated(self, lw_item: QListWidgetItem) -> None:
        qid = lw_item.data(Qt.ItemDataRole.UserRole)
        if not qid:
            return
        row = self._low_conf_by_id.get(str(qid))
        if not row:
            return
        dlg = _ReviewItemDialog(self, self._api, row)
        if dlg.exec() != QDialog.Accepted or not dlg.action_taken:
            return
        self._low_conf_by_id.pop(str(qid), None)
        self._render_low_list()
        self._refresh_review()
        self._status.setText(f"Queue item {qid[:8]}… updated.")

    def _ws_url(self) -> str:
        base = self._api.base_url.rstrip("/")
        if base.startswith("https://"):
            return base.replace("https://", "wss://", 1) + "/v1/stream/ws"
        if base.startswith("http://"):
            return base.replace("http://", "ws://", 1) + "/v1/stream/ws"
        return "ws://127.0.0.1:8000/v1/stream/ws"

    def closeEvent(self, event) -> None:  # noqa: N802
        self._stop_stream()
        super().closeEvent(event)
