from __future__ import annotations

from typing import Iterable

import numpy as np
from PIL import Image

from detection.OWL_VIT_v2.utils import calculate_iou
from improved_pipelines.matching import match_tokens_to_gt
from pathlib import Path

from improved_pipelines.review_queue import ReviewItem
from on_device_app.dto import ReviewItemDto
from on_device_app.services.context import ServiceContext


class ReviewService:
    def __init__(self, ctx: ServiceContext) -> None:
        self._ctx = ctx

    def list_pending(self) -> list[ReviewItemDto]:
        queue = self._ctx.review_queue()
        return [self._to_dto(item) for item in queue.iter_pending()]

    def skip_item(self, queue_id: str) -> None:
        queue = self._ctx.review_queue()
        item = self._find_item(queue_id)
        queue.mark_skipped(item)

    def confirm_item(self, queue_id: str, class_name: str, create_if_missing: bool = False) -> dict:
        queue = self._ctx.review_queue()
        store = self._ctx.object_store()
        embed_store = self._ctx.embedding_store()
        item = self._find_item(queue_id)
        class_id = store.class_id_for_name(class_name, create=create_if_missing)

        stem = f"rv_{item.queue_id.replace('-', '')[:12]}"
        image_path = store.ensure_image_copy(Path(item.image_path), stem)
        store.append_yolo_line(stem, (class_id, item.cx, item.cy, item.w, item.h))
        embedding_id = self._append_live_embedding(
            image_path=image_path,
            image_id=stem,
            class_id=class_id,
            class_name=class_name,
            item=item,
            embed_store=embed_store,
        )
        queue.mark_done(
            item,
            {"confirmed_class": class_name, "confirmed_class_id": class_id, "stem": stem},
        )
        return {
            "queue_id": queue_id,
            "confirmed_class_id": class_id,
            "stem": stem,
            "embedding_added": bool(embedding_id),
            "embedding_id": embedding_id,
        }

    def class_names(self) -> list[str]:
        return self._ctx.object_store().load_class_names()

    def _find_item(self, queue_id: str) -> ReviewItem:
        for item in self._ctx.review_queue().iter_pending():
            if item.queue_id == queue_id:
                return item
        raise KeyError(f"Queue item not found: {queue_id}")

    def _append_live_embedding(
        self,
        image_path: Path,
        image_id: str,
        class_id: int,
        class_name: str,
        item: ReviewItem,
        embed_store,
    ) -> str | None:
        """Extract and store one embedding for a newly confirmed live label."""
        raw_detector = self._ctx.raw_detector()
        boxes, objectnesses, class_embeddings = raw_detector.process(str(image_path.resolve()))
        boxes = np.asarray(boxes)
        objectnesses = np.asarray(objectnesses)
        class_embeddings = np.asarray(class_embeddings)
        if boxes.size == 0 or class_embeddings.size == 0:
            return None

        with Image.open(image_path) as im:
            width, height = im.size

        gt_box = self._review_item_gt_box(item, class_id)
        matched = match_tokens_to_gt(
            boxes,
            objectnesses,
            [gt_box],
            width,
            height,
            objectness_min=0.0,
            iou_match=0.3,
            dedupe_iou=1.0,
        )
        if matched:
            token_idx, _gt, objectness = matched[0]
        else:
            token_idx, objectness = self._best_token_for_box(boxes, objectnesses, gt_box, width, height)
            if token_idx < 0:
                return None

        emb = class_embeddings[int(token_idx)]
        vec = np.asarray(emb, dtype=np.float64).reshape(-1).tolist()
        embedding_id = f"{image_id}:live:{item.queue_id}"
        metadata = {
            "image_id": image_id,
            "class_id": int(class_id),
            "class_name": class_name,
            "patch_index": int(token_idx),
            "objectness": float(objectness),
            "gt_box_json": str([float(x) for x in gt_box[:4]]),
            "source_image": str(image_path.resolve()),
            "source": "live_confirm",
        }
        embed_store.add_embeddings([embedding_id], [vec], [metadata])
        return embedding_id

    @staticmethod
    def _review_item_gt_box(item: ReviewItem, class_id: int) -> list[float]:
        x1 = max(0.0, min(1.0, float(item.cx - (item.w / 2.0))))
        y1 = max(0.0, min(1.0, float(item.cy - (item.h / 2.0))))
        x2 = max(0.0, min(1.0, float(item.cx + (item.w / 2.0))))
        y2 = max(0.0, min(1.0, float(item.cy + (item.h / 2.0))))
        return [x1, y1, x2, y2, float(class_id)]

    @staticmethod
    def _best_token_for_box(
        boxes: np.ndarray,
        objectnesses: np.ndarray,
        gt_box: Iterable[float],
        orig_width: int,
        orig_height: int,
    ) -> tuple[int, float]:
        gt = [float(v) for v in gt_box]
        best_idx = -1
        best_score = -1.0
        best_objectness = 0.0
        h_ratio = float(orig_width) / float(orig_height)
        for idx, (box, objectness) in enumerate(zip(boxes, objectnesses)):
            cx, cy, w, h = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            cx, cy, w, h = cx, cy * h_ratio, w, h * h_ratio
            pred = [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]
            iou = calculate_iou(pred, gt[:4])
            score = iou * max(0.0, float(objectness))
            if score > best_score:
                best_score = score
                best_idx = int(idx)
                best_objectness = float(objectness)
        return best_idx, best_objectness

    @staticmethod
    def _to_dto(item: ReviewItem) -> ReviewItemDto:
        return ReviewItemDto(
            queue_id=item.queue_id,
            image_path=item.image_path,
            cx=item.cx,
            cy=item.cy,
            w=item.w,
            h=item.h,
            score=item.score,
            class_id_suggested=item.class_id_suggested,
            class_name_suggested=item.class_name_suggested,
        )

