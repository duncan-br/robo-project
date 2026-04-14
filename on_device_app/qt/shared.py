from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsRectItem, QGraphicsScene, QGraphicsTextItem


def load_scene_with_bbox(scene: QGraphicsScene, image_path: str, item) -> None:
    scene.clear()
    pix = QPixmap(str(Path(image_path)))
    if pix.isNull():
        text = QGraphicsTextItem("Image not found")
        text.setDefaultTextColor(QColor(200, 200, 200))
        f = QFont()
        f.setPointSize(14)
        text.setFont(f)
        scene.addItem(text)
        return
    scene.addPixmap(pix)
    if item is None:
        return
    w = pix.width()
    h = pix.height()
    cx = float(item["cx"]) if isinstance(item, dict) else float(item.cx)
    cy = float(item["cy"]) if isinstance(item, dict) else float(item.cy)
    bw = float(item["w"]) if isinstance(item, dict) else float(item.w)
    bh = float(item["h"]) if isinstance(item, dict) else float(item.h)
    rect = QGraphicsRectItem(
        (cx - bw / 2.0) * w,
        (cy - bh / 2.0) * h,
        bw * w,
        bh * h,
    )
    rect.setPen(QPen(QColor(255, 220, 0), 2))
    scene.addItem(rect)


def apply_roi_overlay(
    pixmap: QPixmap,
    roi_x: float,
    roi_y: float,
    roi_w: float,
    roi_h: float,
) -> None:
    if roi_w <= 0.0 or roi_h <= 0.0:
        return
    if roi_x == 0.0 and roi_y == 0.0 and roi_w == 1.0 and roi_h == 1.0:
        return
    width = float(pixmap.width())
    height = float(pixmap.height())
    rect_x = int(roi_x * width)
    rect_y = int(roi_y * height)
    rect_w = max(1, int(roi_w * width))
    rect_h = max(1, int(roi_h * height))
    painter = QPainter(pixmap)
    painter.setPen(QPen(QColor(0, 255, 0), 2))
    painter.fillRect(rect_x, rect_y, rect_w, rect_h, QColor(0, 255, 0, 36))
    painter.drawRect(rect_x, rect_y, rect_w, rect_h)
    painter.end()


def pixmap_from_path(path: str) -> QPixmap:
    return QPixmap(path)


def draw_detections_on_pixmap(pixmap: QPixmap, detections: list[dict]) -> QPixmap:
    out = QPixmap(pixmap)
    painter = QPainter(out)
    painter.setRenderHint(QPainter.Antialiasing)
    width = max(1, out.width())
    height = max(1, out.height())
    for det in detections:
        cx = float(det.get("cx", 0.0))
        cy = float(det.get("cy", 0.0))
        bw = float(det.get("w", 0.0))
        bh = float(det.get("h", 0.0))
        x = (cx - bw / 2.0) * width
        y = (cy - bh / 2.0) * height
        rw = bw * width
        rh = bh * height
        score = float(det.get("score", 0.0))
        label = str(det.get("class_name", "unknown"))
        is_high = str(det.get("confidence_level", "low")) == "high"
        color = QColor(30, 230, 90) if is_high else QColor(255, 170, 0)
        painter.setPen(QPen(color, 2))
        painter.drawRect(int(x), int(y), int(rw), int(rh))
        painter.fillRect(int(x), int(max(0, y - 20)), int(max(100, rw)), 18, QColor(0, 0, 0, 140))
        painter.setPen(QPen(Qt.white))
        painter.drawText(int(x + 4), int(max(14, y - 6)), f"{label} {score:.2f}")
    painter.end()
    return out

