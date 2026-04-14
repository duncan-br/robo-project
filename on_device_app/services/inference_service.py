from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import cv2
import numpy as np

from improved_pipelines.inference_image import DetectionDeduper
from on_device_app.dto import DetectionDto, InferenceResultDto, InferenceSettings, ReviewItemDto
from on_device_app.services.context import ServiceContext
from on_device_app.services.detector import Detection


class InferenceService:
    def __init__(self, ctx: ServiceContext) -> None:
        self._ctx = ctx

    def infer_image(self, image_path: str, settings: InferenceSettings) -> InferenceResultDto:
        path = Path(image_path).resolve()
        if not path.is_file():
            return InferenceResultDto(
                image_path=image_path,
                pred_count=0,
                n_high_saved=0,
                n_low_queued=0,
                status="missing_file",
            )

        result, _detections, _low_items, _skipped = self._infer_from_path_rich(path, settings)
        return result

    def infer_frame_bgr(self, frame: np.ndarray, settings: InferenceSettings) -> InferenceResultDto:
        return self.infer_frame_bgr_with_dedup(frame, settings, deduper=None)

    def infer_frame_bgr_with_dedup(
        self,
        frame: np.ndarray,
        settings: InferenceSettings,
        deduper: DetectionDeduper | None,
    ) -> InferenceResultDto:
        result, _skipped = self.infer_frame_bgr_with_dedup_count(frame, settings, deduper=deduper)
        return result

    def infer_frame_bgr_with_dedup_count(
        self,
        frame: np.ndarray,
        settings: InferenceSettings,
        deduper: DetectionDeduper | None,
    ) -> tuple[InferenceResultDto, int]:
        result, _detections, _low_items, skipped = self.infer_frame_bgr_rich(frame, settings, deduper=deduper)
        return result, skipped

    def infer_frame_bgr_rich(
        self,
        frame: np.ndarray,
        settings: InferenceSettings,
        deduper: DetectionDeduper | None,
    ) -> tuple[InferenceResultDto, list[DetectionDto], list[ReviewItemDto], int]:
        with tempfile.NamedTemporaryFile(prefix="orv_frame_", suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        ok = cv2.imwrite(str(tmp_path), frame)
        if not ok:
            return (
                InferenceResultDto(
                image_path=str(tmp_path),
                pred_count=0,
                n_high_saved=0,
                n_low_queued=0,
                status="failed_to_encode_frame",
                ),
                [],
                [],
                0,
            )
        try:
            return self._infer_from_path_rich(tmp_path, settings, deduper=deduper)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    def process_paths(self) -> dict[str, str]:
        return {
            "object_store_root": self._ctx.paths.object_store_root,
            "review_queue_root": self._ctx.paths.review_queue_root,
            "chroma_persist_dir": self._ctx.paths.chroma_persist_dir,
            "chroma_collection": self._ctx.paths.chroma_collection,
        }

    def _infer_from_path_rich(
        self,
        path: Path,
        settings: InferenceSettings,
        deduper: DetectionDeduper | None = None,
    ) -> tuple[InferenceResultDto, list[DetectionDto], list[ReviewItemDto], int]:
        detector = self._ctx.object_detector()
        object_store = self._ctx.object_store()
        review_queue = self._ctx.review_queue()
        detections = detector.detect(path, settings)
        total_candidates = int(len(detections))
        high_lines: list[tuple[int, float, float, float, float]] = []
        low_payloads: list[dict] = []
        detection_dtos: list[DetectionDto] = []
        low_items: list[ReviewItemDto] = []

        for det in detections:
            if deduper is not None and not deduper.allow(
                det.class_id, det.cx, det.cy, det.w, det.h, det.score
            ):
                continue
            confidence_level = "high" if det.score >= settings.high_conf_min else "low"
            detection_dtos.append(
                DetectionDto(
                    class_id=det.class_id,
                    class_name=det.class_name,
                    cx=det.cx,
                    cy=det.cy,
                    w=det.w,
                    h=det.h,
                    score=det.score,
                    confidence_level=confidence_level,
                )
            )
            if det.score >= settings.high_conf_min:
                high_lines.append((det.class_id, det.cx, det.cy, det.w, det.h))
            else:
                queue_id = str(uuid.uuid4())
                low_payloads.append(self._low_payload(path, det, queue_id=queue_id))
                low_items.append(
                    ReviewItemDto(
                        queue_id=queue_id,
                        image_path=str(path.resolve()),
                        cx=det.cx,
                        cy=det.cy,
                        w=det.w,
                        h=det.h,
                        score=det.score,
                        class_id_suggested=det.class_id,
                        class_name_suggested=det.class_name,
                    )
                )

        n_high = 0
        if high_lines:
            object_store.save_infer_result(path, high_lines)
            n_high = len(high_lines)

        if low_payloads:
            review_queue.append_items(low_payloads)

        n_low = len(low_payloads)
        kept = int(n_high + n_low)
        skipped = max(0, total_candidates - kept)
        return (
            InferenceResultDto(
                image_path=str(path),
                pred_count=total_candidates,
                n_high_saved=n_high,
                n_low_queued=n_low,
                status="ok",
            ),
            detection_dtos,
            low_items,
            skipped,
        )

    @staticmethod
    def _low_payload(path: Path, det: Detection, queue_id: str) -> dict:
        return {
            "queue_id": queue_id,
            "image_path": str(path.resolve()),
            "cx": det.cx,
            "cy": det.cy,
            "w": det.w,
            "h": det.h,
            "score": det.score,
            "class_id_suggested": det.class_id,
            "class_name_suggested": det.class_name,
        }

