"""Match OWL tokens to YOLO GT (same geometry as ui/main_ui.py From GT mode)."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from detection.OWL_VIT_v2.utils import calculate_iou


def match_tokens_to_gt(
    boxes: np.ndarray,
    objectnesses: np.ndarray,
    gt_data: Sequence[Sequence[float]],
    orig_width: int,
    orig_height: int,
    objectness_min: float = 0.01,
    iou_match: float = 0.7,
    dedupe_iou: float = 0.8,
) -> List[Tuple[int, List[float], float]]:
    """
    Returns list of ``(token_index, gt_row, objectness)`` where ``gt_row`` is
    ``[x1,y1,x2,y2,class_id]`` in normalized image coordinates (YOLO corner space).
    """
    if not gt_data:
        return []

    h_ratio = orig_width / orig_height
    prev_gt_corners: List[List[float]] = []
    matched: List[Tuple[int, List[float], float]] = []

    for idx, (box, objectness) in enumerate(zip(boxes, objectnesses)):
        if float(objectness) <= objectness_min:
            continue

        best_iou = 0.0
        best_gt: List[float] | None = None
        cx, cy, w, h = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        cx, cy, w, h = cx, cy * h_ratio, w, h * h_ratio
        x_min, y_min, x_max, y_max = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        pred = [x_min, y_min, x_max, y_max]

        for gt_bboox in gt_data:
            g = [float(x) for x in gt_bboox[:5]]
            iou = calculate_iou(pred, g[:4])
            if iou > best_iou and iou > iou_match:
                best_iou = iou
                best_gt = g

        if best_gt is None:
            continue

        gt_corners = best_gt[:4]
        for pb in prev_gt_corners:
            if calculate_iou(pb, gt_corners) > dedupe_iou:
                best_gt = None
                break
        if best_gt is None:
            continue

        prev_gt_corners.append(gt_corners)
        matched.append((idx, best_gt, float(objectness)))

    return matched
