from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from on_device_app.api_client import ApiClient
from on_device_app.qt.shared import load_scene_with_bbox


class ReviewWidget(QWidget):
    def __init__(self, api: ApiClient) -> None:
        super().__init__()
        self._api = api
        self._items = []
        self._idx = 0

        layout = QVBoxLayout(self)
        self._status = QLabel()
        layout.addWidget(self._status)

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
        self._hint.setPlaceholderText("Optional class name override")
        layout.addWidget(self._hint)

        actions = QHBoxLayout()
        self._prev = QPushButton("Previous")
        self._confirm = QPushButton("Confirm (Enter)")
        self._skip = QPushButton("Skip (S)")
        self._next = QPushButton("Next")
        for btn in (self._prev, self._confirm, self._skip, self._next):
            actions.addWidget(btn)
        layout.addLayout(actions)

        self._prev.clicked.connect(self._go_prev)
        self._next.clicked.connect(self._go_next)
        self._confirm.clicked.connect(self._confirm_current)
        self._skip.clicked.connect(self._skip_current)

        self.refresh()

    def refresh(self) -> None:
        try:
            payload = self._api.list_review_queue(limit=1000)
            self._items = payload.get("items", [])
            if self._idx >= len(self._items):
                self._idx = max(0, len(self._items) - 1)
            self._reload_class_list()
            self._render_current()
            self._set_actions_enabled(True)
        except Exception as exc:
            self._items = []
            self._scene.clear()
            self._status.setText(f"API offline: {exc}")
            self._set_actions_enabled(False)

    def _reload_class_list(self) -> None:
        self._combo.clear()
        try:
            for name in self._api.classes():
                self._combo.addItem(name)
        except Exception:
            return

    def _render_current(self) -> None:
        if not self._items:
            self._status.setText("Queue empty.")
            self._scene.clear()
            return
        item = self._items[self._idx]
        self._status.setText(
            f"Item {self._idx + 1} / {len(self._items)}  "
            f"Suggested: {item.get('class_name_suggested')}  "
            f"Score: {float(item.get('score', 0.0)):.3f}"
        )
        self._hint.setText(str(item.get("class_name_suggested") or ""))
        load_scene_with_bbox(self._scene, str(item.get("image_path") or ""), item)
        self._view.fitInView(self._scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def _class_name(self) -> str:
        name = self._hint.text().strip() or self._combo.currentText().strip()
        return name

    def _confirm_current(self) -> None:
        if not self._items:
            return
        name = self._class_name()
        if not name:
            QMessageBox.information(self, "Class required", "Provide a class name before confirm.")
            return
        item = self._items[self._idx]
        try:
            suggested = str(item.get("class_name_suggested") or "").strip().lower()
            auto_create = suggested == "unknown"
            self._api.confirm_review_item(
                str(item.get("queue_id")),
                name,
                create_if_missing=(self._add_new.isChecked() or auto_create),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Confirm failed", str(exc))
            return
        self.refresh()

    def _skip_current(self) -> None:
        if not self._items:
            return
        item = self._items[self._idx]
        try:
            self._api.skip_review_item(str(item.get("queue_id")))
        except Exception as exc:
            QMessageBox.critical(self, "Skip failed", str(exc))
            return
        self.refresh()

    def _set_actions_enabled(self, enabled: bool) -> None:
        self._prev.setEnabled(enabled)
        self._next.setEnabled(enabled)
        self._confirm.setEnabled(enabled)
        self._skip.setEnabled(enabled)

    def _go_prev(self) -> None:
        if self._idx > 0:
            self._idx -= 1
            self._render_current()

    def _go_next(self) -> None:
        if self._idx < len(self._items) - 1:
            self._idx += 1
            self._render_current()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._confirm_current()
            return
        if event.key() == Qt.Key_S:
            self._skip_current()
            return
        if event.key() == Qt.Key_Left:
            self._go_prev()
            return
        if event.key() == Qt.Key_Right:
            self._go_next()
            return
        super().keyPressEvent(event)