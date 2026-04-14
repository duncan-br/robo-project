from __future__ import annotations

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
        item = self._find_item(queue_id)
        class_id = store.class_id_for_name(class_name, create=create_if_missing)

        stem = f"rv_{item.queue_id.replace('-', '')[:12]}"
        store.ensure_image_copy(Path(item.image_path), stem)
        store.append_yolo_line(stem, (class_id, item.cx, item.cy, item.w, item.h))
        queue.mark_done(
            item,
            {"confirmed_class": class_name, "confirmed_class_id": class_id, "stem": stem},
        )
        return {"queue_id": queue_id, "confirmed_class_id": class_id, "stem": stem}

    def class_names(self) -> list[str]:
        return self._ctx.object_store().load_class_names()

    def _find_item(self, queue_id: str) -> ReviewItem:
        for item in self._ctx.review_queue().iter_pending():
            if item.queue_id == queue_id:
                return item
        raise KeyError(f"Queue item not found: {queue_id}")

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

