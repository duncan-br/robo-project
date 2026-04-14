"""Shared test fixtures (fake detector, temp data dirs, pre-populated queues)."""

from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from improved_pipelines.object_store import ObjectStore
from improved_pipelines.review_queue import ReviewQueue
from on_device_app.api.app import create_app
from on_device_app.config import AppPaths
from on_device_app.dto import InferenceSettings, StreamFrameMessage, StreamStatusDto
from on_device_app.services.context import ServiceContext
from on_device_app.services.detector import Detection, ObjectDetector
from on_device_app.services.stream_service import StreamProcessor, StreamSource


class FakeDetector(ObjectDetector):

    def __init__(self, class_names_list: list[str] | None = None) -> None:
        self._class_names = class_names_list or ["bottle", "can", "box"]
        self._detections: list[Detection] = []

    def set_detections(self, detections: list[Detection]) -> None:
        self._detections = detections

    def detect(self, image_path: Path, settings: InferenceSettings) -> list[Detection]:
        return list(self._detections)

    def class_names(self) -> list[str]:
        return list(self._class_names)


class FakeStreamSource(StreamSource):

    def __init__(self, n_frames: int = 5) -> None:
        self._n = n_frames

    def name(self) -> str:
        return "fake:test"

    def frames(self) -> Iterator[tuple[np.ndarray, float]]:
        for i in range(self._n):
            frame = np.full((480, 640, 3), fill_value=(i * 40) % 256, dtype=np.uint8)
            yield frame, float(i)


@pytest.fixture()
def tmp_data_dir(tmp_path: Path) -> Path:
    obj_root = tmp_path / "object_store"
    obj_root.mkdir()
    (obj_root / "images").mkdir()
    (obj_root / "labels").mkdir()
    (obj_root / "classes.txt").write_text("bottle\ncan\nbox\n")

    review_root = tmp_path / "review_queue"
    review_root.mkdir()

    chroma_dir = tmp_path / "chroma_db"
    chroma_dir.mkdir()
    return tmp_path


@pytest.fixture()
def app_paths(tmp_data_dir: Path) -> AppPaths:
    return AppPaths(
        object_store_root=str(tmp_data_dir / "object_store"),
        review_queue_root=str(tmp_data_dir / "review_queue"),
        chroma_persist_dir=str(tmp_data_dir / "chroma_db"),
        chroma_collection="test_embeddings",
    )


@pytest.fixture()
def fake_detector() -> FakeDetector:
    return FakeDetector()


@pytest.fixture()
def service_context(app_paths: AppPaths, fake_detector: FakeDetector) -> ServiceContext:
    ctx = ServiceContext(paths=app_paths)
    ctx._object_detector = fake_detector
    return ctx


@pytest.fixture()
def client(service_context: ServiceContext, tmp_data_dir: Path) -> TestClient:
    app = create_app(service_context)
    app.state.upload_dir = tmp_data_dir / "stream_uploads"
    app.state.upload_dir.mkdir(parents=True, exist_ok=True)
    return TestClient(app)


@pytest.fixture()
def sample_jpeg_bytes() -> bytes:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(frame, (20, 20), (80, 80), (0, 255, 0), -1)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    return encoded.tobytes()


@pytest.fixture()
def sample_image_path(tmp_data_dir: Path) -> Path:
    img_path = tmp_data_dir / "sample_test_image.jpg"
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(frame, (40, 40), (160, 160), (255, 0, 0), -1)
    cv2.imwrite(str(img_path), frame)
    return img_path


@pytest.fixture()
def sample_video_path(tmp_data_dir: Path) -> Path:
    vid_path = tmp_data_dir / "test_stream.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (320, 240))
    for i in range(10):
        frame = np.full((240, 320, 3), fill_value=(i * 25) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return vid_path


@pytest.fixture()
def populated_review_queue(tmp_data_dir: Path) -> tuple[ReviewQueue, list[str]]:
    rq = ReviewQueue(str(tmp_data_dir / "review_queue"))
    ids = [str(uuid.uuid4()) for _ in range(3)]
    items = [
        {
            "queue_id": ids[i],
            "image_path": f"/fake/image_{i}.jpg",
            "cx": 0.5,
            "cy": 0.5,
            "w": 0.1,
            "h": 0.1,
            "score": 0.2 + i * 0.1,
            "class_id_suggested": 0,
            "class_name_suggested": "bottle",
        }
        for i in range(3)
    ]
    rq.append_items(items)
    return rq, ids
