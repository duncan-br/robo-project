from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path

import cv2
import numpy as np

from improved_pipelines.inference_image import DetectionDeduper
from on_device_app.dto import DetectionDto, InferenceResultDto, InferenceSettings, ReviewItemDto
from on_device_app.services.context import ServiceContext
from on_device_app.services.detector import Detection

log = logging.getLogger(__name__)

_REVIEW_IMAGES_DIR: Path | None = None


def _review_images_root(ctx: ServiceContext) -> Path:
    global _REVIEW_IMAGES_DIR  # noqa: PLW0603
    if _REVIEW_IMAGES_DIR is None:
        d = Path(ctx.paths.review_queue_root).resolve().parent / "review_images"
        d.mkdir(parents=True, exist_ok=True)
        _REVIEW_IMAGES_DIR = d
    return _REVIEW_IMAGES_DIR


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
        deduper: DetectionDeduper | None = None,
    ) -> tuple[InferenceResultDto, list[DetectionDto], list[ReviewItemDto], int]:
        """Run detection on a BGR frame *in memory* (no temp-file round-trip)."""
        detector = self._ctx.object_detector()
        object_store = self._ctx.object_store()
        review_queue = self._ctx.review_queue()

        crop_px, crop_norm = self._roi_crop(frame.shape[:2], settings)
        x0, y0, x1, y1 = crop_px
        roi_x, roi_y, roi_w, roi_h = crop_norm
        frame_for_model = frame[y0:y1, x0:x1]
        detections = detector.detect_frame(frame_for_model, settings)
        if crop_px != (0, 0, frame.shape[1], frame.shape[0]):
            detections = [
                self._remap_detection_from_roi(det, roi_x, roi_y, roi_w, roi_h)
                for det in detections
            ]
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
            has_known_class = det.class_id >= 0 and str(det.class_name).strip().lower() != "unknown"
            confidence_level = "high" if has_known_class and det.score >= settings.high_conf_min else "low"
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
            if has_known_class and det.score >= settings.high_conf_min:
                high_lines.append((det.class_id, det.cx, det.cy, det.w, det.h))
            else:
                queue_id = str(uuid.uuid4())
                low_payloads.append(self._low_payload_from_det(det, queue_id=queue_id))
                low_items.append(
                    ReviewItemDto(
                        queue_id=queue_id,
                        image_path="",  # filled in below after saving
                        cx=det.cx,
                        cy=det.cy,
                        w=det.w,
                        h=det.h,
                        score=det.score,
                        class_id_suggested=det.class_id,
                        class_name_suggested=det.class_name,
                    )
                )

        saved_image_path: Path | None = None

        n_high = 0
        if high_lines:
            with tempfile.NamedTemporaryFile(prefix="orv_hi_", suffix=".jpg", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            cv2.imwrite(str(tmp_path), frame)
            object_store.save_infer_result(tmp_path, high_lines)
            saved_image_path = tmp_path
            n_high = len(high_lines)

        if low_payloads:
            review_dir = _review_images_root(self._ctx)
            review_img = review_dir / f"review_{uuid.uuid4().hex}.jpg"
            cv2.imwrite(str(review_img), frame)
            for lp in low_payloads:
                lp["image_path"] = str(review_img)
            for li in low_items:
                li.image_path = str(review_img)
            review_queue.append_items(low_payloads)

        if saved_image_path is not None and not low_payloads:
            try:
                saved_image_path.unlink(missing_ok=True)
            except Exception:
                pass

        n_low = len(low_payloads)
        kept = int(n_high + n_low)
        skipped = max(0, total_candidates - kept)
        return (
            InferenceResultDto(
                image_path=str(saved_image_path or ""),
                pred_count=total_candidates,
                n_high_saved=n_high,
                n_low_queued=n_low,
                status="ok",
            ),
            detection_dtos,
            low_items,
            skipped,
        )

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
            has_known_class = det.class_id >= 0 and str(det.class_name).strip().lower() != "unknown"
            confidence_level = "high" if has_known_class and det.score >= settings.high_conf_min else "low"
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
            if has_known_class and det.score >= settings.high_conf_min:
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

    @staticmethod
    def _low_payload_from_det(det: Detection, queue_id: str) -> dict:
        return {
            "queue_id": queue_id,
            "image_path": "",
            "cx": det.cx,
            "cy": det.cy,
            "w": det.w,
            "h": det.h,
            "score": det.score,
            "class_id_suggested": det.class_id,
            "class_name_suggested": det.class_name,
        }

    @staticmethod
    def _roi_crop(shape_hw: tuple[int, int], settings: InferenceSettings) -> tuple[tuple[int, int, int, int], tuple[float, float, float, float]]:
        """Return ((x0,y0,x1,y1), (roi_x,roi_y,roi_w,roi_h)) with full-frame fallback."""
        h, w = shape_hw
        if h <= 0 or w <= 0:
            return (0, 0, 0, 0), (0.0, 0.0, 1.0, 1.0)

        roi_x = float(settings.roi_x)
        roi_y = float(settings.roi_y)
        roi_w = float(settings.roi_w)
        roi_h = float(settings.roi_h)

        x0 = int(round(roi_x * w))
        y0 = int(round(roi_y * h))
        x1 = int(round((roi_x + roi_w) * w))
        y1 = int(round((roi_y + roi_h) * h))

        x0 = max(0, min(x0, w - 1))
        y0 = max(0, min(y0, h - 1))
        x1 = max(x0 + 1, min(x1, w))
        y1 = max(y0 + 1, min(y1, h))

        cropped_w = x1 - x0
        cropped_h = y1 - y0
        if cropped_w <= 1 or cropped_h <= 1:
            return (0, 0, w, h), (0.0, 0.0, 1.0, 1.0)

        roi_x_eff = float(x0) / float(w)
        roi_y_eff = float(y0) / float(h)
        roi_w_eff = float(cropped_w) / float(w)
        roi_h_eff = float(cropped_h) / float(h)
        return (x0, y0, x1, y1), (roi_x_eff, roi_y_eff, roi_w_eff, roi_h_eff)

    @staticmethod
    def _remap_detection_from_roi(
        det: Detection,
        roi_x: float,
        roi_y: float,
        roi_w: float,
        roi_h: float,
    ) -> Detection:
        def _clamp01(v: float) -> float:
            return max(0.0, min(1.0, float(v)))

        return Detection(
            class_id=det.class_id,
            class_name=det.class_name,
            cx=_clamp01(roi_x + det.cx * roi_w),
            cy=_clamp01(roi_y + det.cy * roi_h),
            w=_clamp01(det.w * roi_w),
            h=_clamp01(det.h * roi_h),
            score=det.score,
        )

