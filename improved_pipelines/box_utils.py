"""Convert OWL model boxes to normalized YOLO-style cx, cy, w, h (same space as GT)."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def model_box_to_yolo_cxcywh(
    box: np.ndarray | Tuple[float, float, float, float],
    orig_width: int,
    orig_height: int,
) -> Tuple[float, float, float, float]:
    """
    Model outputs center-x, center-y, width, height in internal normalized coords.
    Match ``matching.py`` / UI: apply ``h_ratio = W/H`` to cy and h.
    Returns YOLO-normalized cx, cy, w, h in [0, 1] image space.
    """
    h_ratio = orig_width / orig_height
    cx, cy, w, h = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    cx, cy, w, h = cx, cy * h_ratio, w, h * h_ratio
    return cx, cy, w, h


def model_boxes_to_yolo_lines(
    boxes: np.ndarray,
    class_ids: np.ndarray,
    orig_width: int,
    orig_height: int,
) -> List[Tuple[int, float, float, float, float]]:
    """Each line: class_id, cx, cy, w, h."""
    out: List[Tuple[int, float, float, float, float]] = []
    for box, cid in zip(boxes, class_ids):
        cx, cy, w, h = model_box_to_yolo_cxcywh(box, orig_width, orig_height)
        out.append((int(cid), cx, cy, w, h))
    return out
