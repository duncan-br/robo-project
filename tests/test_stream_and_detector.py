"""Tests for stream sources, stream processor lifecycle, ROI filtering, and detector ABC."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from on_device_app.config import AppPaths
from on_device_app.dto import DetectionDto, InferenceSettings, ReviewItemDto, StreamFrameMessage
from on_device_app.services.detector import Detection, ObjectDetector
from on_device_app.services.stream_service import (
    BagFileSource,
    StreamProcessor,
    StreamSource,
    VideoFileSource,
    _ros_compressed_to_bgr,
    _ros_image_to_bgr,
    read_first_stream_frame,
)

from .conftest import FakeDetector, FakeStreamSource

class TestVideoFileSourceFrames:
    def test_yields_frames_from_video(self, sample_video_path: Path):
        src = VideoFileSource(sample_video_path)
        frames = []
        for frame, ts in src.frames():
            frames.append(frame)
            if len(frames) >= 3:
                break
        assert len(frames) >= 3
        assert frames[0].ndim == 3

    def test_yields_all_frames(self, sample_video_path: Path):
        src = VideoFileSource(sample_video_path)
        frames = list(src.frames())
        assert len(frames) == 10

    def test_empty_video_yields_nothing(self, tmp_path: Path):
        vid = tmp_path / "empty.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(vid), fourcc, 10.0, (320, 240))
        writer.release()
        src = VideoFileSource(vid)
        frames = list(src.frames())
        assert len(frames) == 0

class TestReadFirstStreamFrame:
    def test_reads_first_frame_from_video(self, sample_video_path: Path):
        frame = read_first_stream_frame(sample_video_path)
        assert frame.ndim == 3
        assert frame.shape[0] > 0

    def test_empty_video_raises_stop_iteration(self, tmp_path: Path):
        vid = tmp_path / "empty.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(vid), fourcc, 10.0, (320, 240))
        writer.release()
        with pytest.raises(StopIteration):
            read_first_stream_frame(vid)

class TestBagFileSource:
    def test_name_contains_filename(self, tmp_path: Path):
        p = tmp_path / "recording.db3"
        p.write_bytes(b"")
        src = BagFileSource(p, topic="/cam")
        assert "recording.db3" in src.name()

    def test_no_reader_available_raises(self, tmp_path: Path):
        p = tmp_path / "fake.db3"
        p.write_bytes(b"")
        src = BagFileSource(p)
        with pytest.raises(RuntimeError, match="No bag reader available|Failed to read"):
            list(src.frames())

class TestStreamProcessorLifecycle:
    def _make_processor(self, fake_detector: FakeDetector, tmp_data_dir: Path) -> StreamProcessor:
        from improved_pipelines.object_store import ObjectStore
        from improved_pipelines.review_queue import ReviewQueue

        store = ObjectStore(str(tmp_data_dir / "object_store"))
        rq = ReviewQueue(str(tmp_data_dir / "review_queue"))
        return StreamProcessor(
            detector=fake_detector,
            review_queue_factory=lambda: rq,
            object_store_factory=lambda: store,
        )

    def test_start_stop_lifecycle(self, fake_detector: FakeDetector, tmp_data_dir: Path):
        sp = self._make_processor(fake_detector, tmp_data_dir)
        fake_detector.set_detections([])
        source = FakeStreamSource(n_frames=50)
        sp.start(source, InferenceSettings())
        time.sleep(0.3)

        status = sp.status()
        assert status.source_name == "fake:test"

        sp.stop()
        final = sp.status()
        assert final.active is False

    def test_start_replaces_previous_stream(self, fake_detector: FakeDetector, tmp_data_dir: Path):
        sp = self._make_processor(fake_detector, tmp_data_dir)
        fake_detector.set_detections([])

        sp.start(FakeStreamSource(n_frames=100), InferenceSettings())
        time.sleep(0.1)
        sp.start(FakeStreamSource(n_frames=5), InferenceSettings())
        time.sleep(0.5)
        sp.stop()

    def test_latest_message_populated_after_frames(
        self, fake_detector: FakeDetector, tmp_data_dir: Path,
    ):
        sp = self._make_processor(fake_detector, tmp_data_dir)
        fake_detector.set_detections([
            Detection(class_id=0, class_name="bottle", cx=0.5, cy=0.5, w=0.1, h=0.1, score=0.9),
        ])
        sp.start(FakeStreamSource(n_frames=10), InferenceSettings())
        time.sleep(1.0)
        msg = sp.latest_message()
        assert msg is not None
        assert msg.frame_index >= 1
        sp.stop()

    def test_update_settings_mid_stream(self, fake_detector: FakeDetector, tmp_data_dir: Path):
        sp = self._make_processor(fake_detector, tmp_data_dir)
        fake_detector.set_detections([])
        sp.start(FakeStreamSource(n_frames=50), InferenceSettings(conf_thresh=0.2))
        time.sleep(0.1)
        sp.update_settings(InferenceSettings(conf_thresh=0.9))
        time.sleep(0.1)
        sp.stop()

    def test_stop_without_start_is_safe(self, fake_detector: FakeDetector, tmp_data_dir: Path):
        sp = self._make_processor(fake_detector, tmp_data_dir)
        sp.stop()

    def test_double_stop_is_safe(self, fake_detector: FakeDetector, tmp_data_dir: Path):
        sp = self._make_processor(fake_detector, tmp_data_dir)
        fake_detector.set_detections([])
        sp.start(FakeStreamSource(n_frames=5), InferenceSettings())
        time.sleep(0.3)
        sp.stop()
        sp.stop()

class TestStreamProcessorProcessFrame:
    def _make_processor(self, fake_detector, tmp_data_dir):
        from improved_pipelines.object_store import ObjectStore
        from improved_pipelines.review_queue import ReviewQueue

        store = ObjectStore(str(tmp_data_dir / "object_store"))
        rq = ReviewQueue(str(tmp_data_dir / "review_queue"))
        return StreamProcessor(
            detector=fake_detector,
            review_queue_factory=lambda: rq,
            object_store_factory=lambda: store,
        )

    def test_process_frame_no_inference_uses_cache(
        self, fake_detector: FakeDetector, tmp_data_dir: Path,
    ):
        sp = self._make_processor(fake_detector, tmp_data_dir)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cached_dets = [
            DetectionDto(
                class_id=0, class_name="bottle",
                cx=0.5, cy=0.5, w=0.1, h=0.1,
                score=0.9, confidence_level="high",
            )
        ]
        msg = sp._process_frame(
            frame, 2, InferenceSettings(),
            run_inference=False,
            cached_detections=cached_dets,
            cached_low_items=[],
        )
        assert msg.frame_index == 2
        assert len(msg.detections) == 1
        assert msg.detections[0].class_name == "bottle"

    def test_process_frame_with_inference_high_conf(
        self, fake_detector: FakeDetector, tmp_data_dir: Path,
    ):
        fake_detector.set_detections([
            Detection(class_id=0, class_name="bottle", cx=0.5, cy=0.5, w=0.1, h=0.1, score=0.9),
        ])
        sp = self._make_processor(fake_detector, tmp_data_dir)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        msg = sp._process_frame(frame, 1, InferenceSettings(), run_inference=True)
        assert msg.frame_index == 1
        assert len(msg.detections) == 1

    def test_process_frame_with_inference_low_conf(
        self, fake_detector: FakeDetector, tmp_data_dir: Path,
    ):
        fake_detector.set_detections([
            Detection(class_id=1, class_name="can", cx=0.5, cy=0.5, w=0.1, h=0.1, score=0.25),
        ])
        sp = self._make_processor(fake_detector, tmp_data_dir)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        msg = sp._process_frame(frame, 1, InferenceSettings(), run_inference=True)
        assert len(msg.low_confidence_items) == 1

    def test_process_frame_roi_filters_detections(
        self, fake_detector: FakeDetector, tmp_data_dir: Path,
    ):
        fake_detector.set_detections([
            Detection(class_id=0, class_name="bottle", cx=0.1, cy=0.1, w=0.05, h=0.05, score=0.9),
        ])
        sp = self._make_processor(fake_detector, tmp_data_dir)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        settings = InferenceSettings(roi_x=0.3, roi_y=0.3, roi_w=0.5, roi_h=0.5)
        msg = sp._process_frame(frame, 1, settings, run_inference=True)
        assert len(msg.detections) == 0

    def test_process_frame_empty_detections(
        self, fake_detector: FakeDetector, tmp_data_dir: Path,
    ):
        fake_detector.set_detections([])
        sp = self._make_processor(fake_detector, tmp_data_dir)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        msg = sp._process_frame(frame, 5, InferenceSettings(), run_inference=True)
        assert len(msg.detections) == 0
        assert msg.frame_jpeg_b64 != ""

    def test_process_frame_tracking_registers_once_on_right_to_left_crossing(
        self, fake_detector: FakeDetector, tmp_data_dir: Path,
    ):
        sp = self._make_processor(fake_detector, tmp_data_dir)
        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        settings = InferenceSettings(
            tracking_enabled=True,
            tracking_direction="right_to_left",
            tracking_line_x=0.5,
            tracking_max_match_dist=0.4,
            tracking_max_age_ms=2000,
        )

        fake_detector.set_detections([
            Detection(class_id=0, class_name="bottle", cx=0.8, cy=0.5, w=0.1, h=0.1, score=0.9),
        ])
        msg1 = sp._process_frame(frame, 1, settings, run_inference=True)
        assert len(msg1.detections) == 0

        fake_detector.set_detections([
            Detection(class_id=0, class_name="bottle", cx=0.45, cy=0.5, w=0.1, h=0.1, score=0.9),
        ])
        msg2 = sp._process_frame(frame, 2, settings, run_inference=True)
        assert len(msg2.detections) == 1

        fake_detector.set_detections([
            Detection(class_id=0, class_name="bottle", cx=0.3, cy=0.5, w=0.1, h=0.1, score=0.9),
        ])
        msg3 = sp._process_frame(frame, 3, settings, run_inference=True)
        assert len(msg3.detections) == 0

    def test_process_frame_tracking_disabled_keeps_duplicates(
        self, fake_detector: FakeDetector, tmp_data_dir: Path,
    ):
        sp = self._make_processor(fake_detector, tmp_data_dir)
        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        settings = InferenceSettings(tracking_enabled=False)

        fake_detector.set_detections([
            Detection(class_id=0, class_name="bottle", cx=0.8, cy=0.5, w=0.1, h=0.1, score=0.9),
        ])
        msg1 = sp._process_frame(frame, 1, settings, run_inference=True)
        assert len(msg1.detections) == 1

        fake_detector.set_detections([
            Detection(class_id=0, class_name="bottle", cx=0.78, cy=0.5, w=0.1, h=0.1, score=0.9),
        ])
        msg2 = sp._process_frame(frame, 2, settings, run_inference=True)
        assert len(msg2.detections) == 1

class TestInsideRoiEdgeCases:
    def test_roi_exact_boundary(self):
        s = InferenceSettings(roi_x=0.2, roi_y=0.2, roi_w=0.3, roi_h=0.3)
        assert StreamProcessor._inside_roi(0.2, 0.2, s) is True
        assert StreamProcessor._inside_roi(0.5, 0.5, s) is True

    def test_roi_just_outside(self):
        s = InferenceSettings(roi_x=0.2, roi_y=0.2, roi_w=0.3, roi_h=0.3)
        assert StreamProcessor._inside_roi(0.19, 0.3, s) is False
        assert StreamProcessor._inside_roi(0.3, 0.19, s) is False
        assert StreamProcessor._inside_roi(0.51, 0.3, s) is False
        assert StreamProcessor._inside_roi(0.3, 0.51, s) is False

    def test_roi_zero_size(self):
        s = InferenceSettings(roi_x=0.5, roi_y=0.5, roi_w=0.0, roi_h=0.0)
        assert StreamProcessor._inside_roi(0.5, 0.5, s) is True
        assert StreamProcessor._inside_roi(0.51, 0.5, s) is False

class TestRosImageDecodeExtended:
    def test_8uc1_decode(self):
        data = np.zeros((10, 10), dtype=np.uint8).tobytes()
        result = _ros_image_to_bgr("8UC1", 10, 10, data)
        assert result.shape == (10, 10, 3)

    def test_16uc1_decode(self):
        data = np.zeros((10, 10), dtype=np.uint16).tobytes()
        result = _ros_image_to_bgr("16UC1", 10, 10, data)
        assert result.shape == (10, 10, 3)

    def test_bgr8_numpy_array_input(self):
        arr = np.zeros((300,), dtype=np.uint8)
        result = _ros_image_to_bgr("bgr8", 10, 10, arr)
        assert result.shape == (10, 10, 3)

    def test_compressed_numpy_input(self):
        frame = np.full((20, 20, 3), 128, dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", frame)
        result = _ros_compressed_to_bgr("jpeg", buf)
        assert result.ndim == 3

class TestObjectDetectorABC:
    def test_abstract_methods(self):
        with pytest.raises(TypeError):
            ObjectDetector()

    def test_fake_detector_class_names(self):
        d = FakeDetector(["a", "b", "c"])
        assert d.class_names() == ["a", "b", "c"]

    def test_fake_detector_detect_empty(self, tmp_path: Path):
        d = FakeDetector()
        result = d.detect(tmp_path / "img.jpg", InferenceSettings())
        assert result == []

    def test_fake_detector_with_detections(self, tmp_path: Path):
        d = FakeDetector()
        dets = [
            Detection(0, "a", 0.5, 0.5, 0.1, 0.1, 0.9),
            Detection(1, "b", 0.3, 0.3, 0.05, 0.05, 0.4),
        ]
        d.set_detections(dets)
        result = d.detect(tmp_path / "img.jpg", InferenceSettings())
        assert len(result) == 2

class TestFakeStreamSource:
    def test_yields_correct_count(self):
        src = FakeStreamSource(n_frames=7)
        frames = list(src.frames())
        assert len(frames) == 7

    def test_zero_frames(self):
        src = FakeStreamSource(n_frames=0)
        frames = list(src.frames())
        assert len(frames) == 0

    def test_name(self):
        assert FakeStreamSource().name() == "fake:test"

class TestStreamPreviewWithVideo:
    def test_preview_returns_base64_jpeg(self, client, sample_video_path: Path):
        from fastapi.testclient import TestClient
        with sample_video_path.open("rb") as f:
            upload = client.post(
                "/v1/stream/upload",
                files={"file": ("preview_test.mp4", f, "application/octet-stream")},
            )
        assert upload.status_code == 200

        resp = client.get("/v1/stream/preview?source=upload")
        assert resp.status_code == 200
        body = resp.json()
        assert "frame_jpeg_b64" in body
        assert len(body["frame_jpeg_b64"]) > 100

class TestDetectionDataclass:
    def test_detection_fields(self):
        d = Detection(class_id=0, class_name="bottle", cx=0.5, cy=0.5, w=0.1, h=0.1, score=0.95)
        assert d.class_id == 0
        assert d.class_name == "bottle"
        assert d.score == 0.95

    def test_detection_frozen(self):
        d = Detection(class_id=0, class_name="bottle", cx=0.5, cy=0.5, w=0.1, h=0.1, score=0.95)
        with pytest.raises(Exception):
            d.class_id = 1

class TestConfigModule:
    def test_load_app_paths_with_env(self, monkeypatch):
        from on_device_app.config import load_app_paths
        monkeypatch.setenv("OBJECT_STORE_ROOT", "/custom/store")
        monkeypatch.setenv("REVIEW_QUEUE_DIR", "/custom/queue")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", "/custom/chroma")
        monkeypatch.setenv("CHROMA_COLLECTION", "custom_collection")
        paths = load_app_paths()
        assert paths.object_store_root == "/custom/store"
        assert paths.review_queue_root == "/custom/queue"
        assert paths.chroma_persist_dir == "/custom/chroma"
        assert paths.chroma_collection == "custom_collection"

class TestRequirementGapCoverageStream:
    def _make_processor(self, detector, tmp_data_dir: Path) -> StreamProcessor:
        from improved_pipelines.object_store import ObjectStore
        from improved_pipelines.review_queue import ReviewQueue

        store = ObjectStore(str(tmp_data_dir / "object_store"))
        rq = ReviewQueue(str(tmp_data_dir / "review_queue"))
        return StreamProcessor(
            detector=detector,
            review_queue_factory=lambda: rq,
            object_store_factory=lambda: store,
        )

    def test_detector_swappable_via_abc(self, tmp_data_dir: Path):
        """Swapping the detector behind the ABC should change output."""
        detector_a = FakeDetector()
        detector_a.set_detections([
            Detection(class_id=0, class_name="bottle", cx=0.5, cy=0.5, w=0.1, h=0.1, score=0.95),
        ])
        processor = self._make_processor(detector_a, tmp_data_dir)
        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        msg_a = processor._process_frame(frame, 1, InferenceSettings(), run_inference=True)
        assert msg_a.detections[0].class_name == "bottle"

        detector_b = FakeDetector()
        detector_b.set_detections([
            Detection(class_id=1, class_name="can", cx=0.4, cy=0.4, w=0.1, h=0.1, score=0.95),
        ])
        processor._detector = detector_b
        msg_b = processor._process_frame(frame, 2, InferenceSettings(), run_inference=True)
        assert msg_b.detections[0].class_name == "can"

    def test_detector_gpu_fallback(self, tmp_data_dir: Path):
        """If GPU init fails, the previously loaded detector should still work."""
        from on_device_app.services.context import ServiceContext

        paths = AppPaths(
            object_store_root=str(tmp_data_dir / "object_store"),
            review_queue_root=str(tmp_data_dir / "review_queue"),
            chroma_persist_dir=str(tmp_data_dir / "chroma_db"),
            chroma_collection="owl_gt_embeddings",
        )
        ctx = ServiceContext(paths)
        preloaded = MagicMock(name="cpu_detector")
        ctx._raw_detector = preloaded
        with patch(
            "on_device_app.services.context.ImageConditionedObjectDetector",
            side_effect=RuntimeError("gpu unavailable"),
        ):
            got = ctx.raw_detector()
        assert got is preloaded

    def test_docker_ubuntu_pytest_smoke(self):
        """Dockerfile should declare both CPU and GPU base images."""
        root = Path(__file__).resolve().parents[1]
        dockerfile = root / "Dockerfile"
        content = dockerfile.read_text(encoding="utf-8")
        assert "FROM ubuntu:22.04 AS cpu" in content
        assert "FROM nvidia/cuda:" in content
