from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from detection.OWL_VIT_v2.image_conditioned import ImageConditionedObjectDetector
from improved_pipelines.box_utils import model_boxes_to_yolo_lines
from improved_pipelines.embedding_store import ChromaEmbeddingStore
from improved_pipelines.inference_image import build_query_embeddings, run_inference_on_frame, run_inference_on_image
from improved_pipelines.object_store import ObjectStore
from on_device_app.dto import InferenceSettings


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    cx: float
    cy: float
    w: float
    h: float
    score: float


class ObjectDetector(ABC):
    @abstractmethod
    def detect(self, image_path: Path, settings: InferenceSettings) -> list[Detection]:
        raise NotImplementedError

    @abstractmethod
    def class_names(self) -> list[str]:
        raise NotImplementedError


class OwlVitV2Detector(ObjectDetector):
    def __init__(
        self,
        embedding_store_factory: Callable[[], ChromaEmbeddingStore],
        raw_detector_factory: Callable[[], ImageConditionedObjectDetector],
        object_store_factory: Callable[[], ObjectStore],
    ) -> None:
        self._embedding_store_factory = embedding_store_factory
        self._raw_detector_factory = raw_detector_factory
        self._object_store_factory = object_store_factory
        self._generic_query_embedding: dict[str, list[np.ndarray]] | None = None
        self._generic_class_names: list[str] = ["unknown"]

    def detect(self, image_path: Path, settings: InferenceSettings) -> list[Detection]:
        embed_store = self._embedding_store_factory()
        raw_detector = self._raw_detector_factory()
        query_embedding, class_names = build_query_embeddings(embed_store)
        if not class_names:
            query_embedding, class_names = self._generic_prompt_queries(raw_detector)
        class_ids, scores, boxes = run_inference_on_image(
            image_path,
            raw_detector,
            query_embedding,
            class_names,
            conf_thresh=settings.conf_thresh,
            merging_mode=settings.merging_mode,
            avg_count=settings.avg_count,
        )
        class_ids = np.asarray(class_ids)
        scores = np.asarray(scores)
        boxes = np.asarray(boxes)

        from PIL import Image

        with Image.open(image_path) as img:
            width, height = img.size
        yolo_lines = model_boxes_to_yolo_lines(boxes, class_ids, width, height)

        out: list[Detection] = []
        for line, score in zip(yolo_lines, scores):
            cid, cx, cy, w, h = line
            cname = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
            cid_out = -1 if cname == "unknown" else int(cid)
            out.append(
                Detection(
                    class_id=cid_out,
                    class_name=cname,
                    cx=float(cx),
                    cy=float(cy),
                    w=float(w),
                    h=float(h),
                    score=float(score),
                )
            )
        return out

    def detect_frame(self, frame_bgr: np.ndarray, settings: InferenceSettings) -> list[Detection]:
        """Run detection on an in-memory BGR frame (no disk I/O)."""
        embed_store = self._embedding_store_factory()
        raw_detector = self._raw_detector_factory()
        query_embedding, class_names = build_query_embeddings(embed_store)
        if not class_names:
            query_embedding, class_names = self._generic_prompt_queries(raw_detector)
        class_ids, scores, boxes = run_inference_on_frame(
            frame_bgr,
            raw_detector,
            query_embedding,
            class_names,
            conf_thresh=settings.conf_thresh,
            merging_mode=settings.merging_mode,
            avg_count=settings.avg_count,
        )
        class_ids = np.asarray(class_ids)
        scores = np.asarray(scores)
        boxes = np.asarray(boxes)

        h, w = frame_bgr.shape[:2]
        yolo_lines = model_boxes_to_yolo_lines(boxes, class_ids, w, h)

        out: list[Detection] = []
        for line, score in zip(yolo_lines, scores):
            cid, cx, cy, bw, bh = line
            cname = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
            cid_out = -1 if cname == "unknown" else int(cid)
            out.append(
                Detection(
                    class_id=cid_out,
                    class_name=cname,
                    cx=float(cx),
                    cy=float(cy),
                    w=float(bw),
                    h=float(bh),
                    score=float(score),
                )
            )
        return out

    def class_names(self) -> list[str]:
        return self._object_store_factory().load_class_names()

    def _generic_prompt_queries(
        self,
        raw_detector: ImageConditionedObjectDetector,
    ) -> tuple[dict[str, list[np.ndarray]], list[str]]:
        if self._generic_query_embedding is None:
            prompt_embeddings = raw_detector.tokenize_queries(["object"])
            vec = np.asarray(prompt_embeddings[0], dtype=np.float32)
            self._generic_query_embedding = {"unknown": [vec]}
        return self._generic_query_embedding, self._generic_class_names
