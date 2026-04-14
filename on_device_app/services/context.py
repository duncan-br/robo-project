from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

from detection.OWL_VIT_v2.image_conditioned import ImageConditionedObjectDetector
from improved_pipelines.embedding_store import ChromaEmbeddingStore
from improved_pipelines.object_store import ObjectStore
from improved_pipelines.review_queue import ReviewQueue
from on_device_app.config import AppPaths
from on_device_app.services.detector import ObjectDetector, OwlVitV2Detector
from on_device_app.services.stream_service import StreamProcessor


@dataclass
class ServiceContext:
    paths: AppPaths
    _raw_detector: ImageConditionedObjectDetector | None = field(default=None, init=False)
    _raw_detector_lock: Lock = field(default_factory=Lock, init=False)
    _object_detector: ObjectDetector | None = field(default=None, init=False)
    _object_detector_lock: Lock = field(default_factory=Lock, init=False)
    _stream_processor: StreamProcessor | None = field(default=None, init=False)
    _stream_processor_lock: Lock = field(default_factory=Lock, init=False)

    def object_store(self) -> ObjectStore:
        return ObjectStore(self.paths.object_store_root)

    def review_queue(self) -> ReviewQueue:
        return ReviewQueue(self.paths.review_queue_root)

    def embedding_store(self) -> ChromaEmbeddingStore:
        return ChromaEmbeddingStore(
            persist_directory=self.paths.chroma_persist_dir,
            collection_name=self.paths.chroma_collection,
        )

    def raw_detector(self) -> ImageConditionedObjectDetector:
        with self._raw_detector_lock:
            if self._raw_detector is None:
                self._raw_detector = ImageConditionedObjectDetector()
            return self._raw_detector

    def object_detector(self) -> ObjectDetector:
        with self._object_detector_lock:
            if self._object_detector is None:
                self._object_detector = OwlVitV2Detector(
                    embedding_store_factory=self.embedding_store,
                    raw_detector_factory=self.raw_detector,
                    object_store_factory=self.object_store,
                )
            return self._object_detector

    def stream_processor(self) -> StreamProcessor:
        with self._stream_processor_lock:
            if self._stream_processor is None:
                self._stream_processor = StreamProcessor(
                    detector=self.object_detector(),
                    review_queue_factory=self.review_queue,
                    object_store_factory=self.object_store,
                )
            return self._stream_processor

