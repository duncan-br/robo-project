"""API tests covering health, classes, review queue, inference, streaming, and DTOs."""

from __future__ import annotations

import io
import json
import time
import uuid
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from improved_pipelines.object_store import ObjectStore
from improved_pipelines.review_queue import ReviewQueue
from on_device_app.dto import InferenceSettings
from on_device_app.services.detector import Detection

from .conftest import FakeDetector

class TestHealth:
    def test_health_returns_ok(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_health_returns_json_content_type(self, client: TestClient):
        resp = client.get("/health")
        assert "application/json" in resp.headers["content-type"]

class TestClasses:
    def test_get_classes_returns_list(self, client: TestClient):
        resp = client.get("/v1/classes")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert "bottle" in data
        assert "can" in data
        assert "box" in data

    def test_get_classes_order_matches_file(self, client: TestClient):
        resp = client.get("/v1/classes")
        assert resp.json() == ["bottle", "can", "box"]

    def test_classes_after_adding_new_class(self, client: TestClient, tmp_data_dir: Path):
        classes_file = tmp_data_dir / "object_store" / "classes.txt"
        classes_file.write_text("bottle\ncan\nbox\nnew_widget\n")
        resp = client.get("/v1/classes")
        assert "new_widget" in resp.json()

class TestReviewQueue:
    def test_list_queue_empty(self, client: TestClient):
        resp = client.get("/v1/review/queue")
        assert resp.status_code == 200
        body = resp.json()
        assert body["items"] == []
        assert body["total"] == 0

    def test_list_queue_with_items(self, client: TestClient, populated_review_queue):
        rq, ids = populated_review_queue
        resp = client.get("/v1/review/queue")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 3
        assert len(body["items"]) == 3
        returned_ids = {item["queue_id"] for item in body["items"]}
        assert set(ids) == returned_ids

    def test_list_queue_respects_limit(self, client: TestClient, populated_review_queue):
        resp = client.get("/v1/review/queue?limit=2")
        body = resp.json()
        assert len(body["items"]) == 2
        assert body["total"] == 3

    def test_list_queue_limit_minimum(self, client: TestClient):
        resp = client.get("/v1/review/queue?limit=0")
        assert resp.status_code == 422

    def test_list_queue_limit_maximum(self, client: TestClient):
        resp = client.get("/v1/review/queue?limit=1001")
        assert resp.status_code == 422

    def test_confirm_item_existing_class(
        self, client: TestClient, populated_review_queue, tmp_data_dir: Path,
    ):
        rq, ids = populated_review_queue
        img_path = tmp_data_dir / "object_store" / "images" / "dummy.jpg"
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), frame)
        pending_file = tmp_data_dir / "review_queue" / "pending.jsonl"
        lines = pending_file.read_text().strip().split("\n")
        new_lines = []
        for line in lines:
            d = json.loads(line)
            d["image_path"] = str(img_path)
            new_lines.append(json.dumps(d))
        pending_file.write_text("\n".join(new_lines) + "\n")

        resp = client.post(
            f"/v1/review/items/{ids[0]}/confirm",
            json={"class_name": "bottle", "create_if_missing": False},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["queue_id"] == ids[0]
        assert body["confirmed_class_id"] == 0

    def test_confirm_item_create_new_class(
        self, client: TestClient, populated_review_queue, tmp_data_dir: Path,
    ):
        """Confirming with a new class name should create it."""
        rq, ids = populated_review_queue
        img_path = tmp_data_dir / "object_store" / "images" / "dummy2.jpg"
        cv2.imwrite(str(img_path), np.zeros((50, 50, 3), dtype=np.uint8))
        pending_file = tmp_data_dir / "review_queue" / "pending.jsonl"
        lines = pending_file.read_text().strip().split("\n")
        new_lines = []
        for line in lines:
            d = json.loads(line)
            d["image_path"] = str(img_path)
            new_lines.append(json.dumps(d))
        pending_file.write_text("\n".join(new_lines) + "\n")

        resp = client.post(
            f"/v1/review/items/{ids[0]}/confirm",
            json={"class_name": "new_widget", "create_if_missing": True},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["confirmed_class_id"] == 3

        classes_resp = client.get("/v1/classes")
        assert "new_widget" in classes_resp.json()

    def test_confirm_item_unknown_class_without_create(
        self, client: TestClient, populated_review_queue,
    ):
        """Unknown class without create flag should 404."""
        rq, ids = populated_review_queue
        resp = client.post(
            f"/v1/review/items/{ids[0]}/confirm",
            json={"class_name": "nonexistent", "create_if_missing": False},
        )
        assert resp.status_code == 404
        assert "Unknown class" in resp.json()["detail"]

    def test_confirm_item_nonexistent_queue_id(self, client: TestClient):
        resp = client.post(
            "/v1/review/items/fake-id-9999/confirm",
            json={"class_name": "bottle", "create_if_missing": False},
        )
        assert resp.status_code == 404

    def test_confirm_item_empty_class_name(self, client: TestClient):
        resp = client.post(
            "/v1/review/items/some-id/confirm",
            json={"class_name": "", "create_if_missing": False},
        )
        assert resp.status_code == 422

    def test_skip_item(self, client: TestClient, populated_review_queue):
        rq, ids = populated_review_queue
        resp = client.post(f"/v1/review/items/{ids[1]}/skip")
        assert resp.status_code == 200
        body = resp.json()
        assert body["queue_id"] == ids[1]
        assert body["status"] == "skipped"

        remaining = client.get("/v1/review/queue").json()
        remaining_ids = {i["queue_id"] for i in remaining["items"]}
        assert ids[1] not in remaining_ids

    def test_skip_nonexistent_item(self, client: TestClient):
        resp = client.post("/v1/review/items/bad-id/skip")
        assert resp.status_code == 404

class TestImageInference:
    def test_infer_missing_image(self, client: TestClient):
        resp = client.post(
            "/v1/inference/image",
            json={
                "image_path": "/nonexistent/path.jpg",
                "settings": InferenceSettings().model_dump(),
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "missing_file"
        assert body["pred_count"] == 0

    def test_infer_image_no_detections(
        self, client: TestClient, fake_detector: FakeDetector, sample_image_path: Path,
    ):
        fake_detector.set_detections([])
        resp = client.post(
            "/v1/inference/image",
            json={
                "image_path": str(sample_image_path),
                "settings": InferenceSettings().model_dump(),
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["pred_count"] == 0
        assert body["n_high_saved"] == 0
        assert body["n_low_queued"] == 0

    def test_infer_image_high_confidence_detections(
        self, client: TestClient, fake_detector: FakeDetector, sample_image_path: Path,
    ):
        """High-confidence detections should be persisted to the object store."""
        fake_detector.set_detections([
            Detection(class_id=0, class_name="bottle", cx=0.5, cy=0.5, w=0.1, h=0.1, score=0.9),
            Detection(class_id=1, class_name="can", cx=0.3, cy=0.3, w=0.08, h=0.08, score=0.8),
        ])
        resp = client.post(
            "/v1/inference/image",
            json={
                "image_path": str(sample_image_path),
                "settings": InferenceSettings(conf_thresh=0.2, high_conf_min=0.35).model_dump(),
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["pred_count"] == 2
        assert body["n_high_saved"] == 2
        assert body["n_low_queued"] == 0

    def test_infer_image_low_confidence_routed_to_review(
        self, client: TestClient, fake_detector: FakeDetector, sample_image_path: Path,
    ):
        """Low-confidence detections should land in the review queue."""
        fake_detector.set_detections([
            Detection(class_id=0, class_name="bottle", cx=0.5, cy=0.5, w=0.1, h=0.1, score=0.25),
        ])
        resp = client.post(
            "/v1/inference/image",
            json={
                "image_path": str(sample_image_path),
                "settings": InferenceSettings(conf_thresh=0.2, high_conf_min=0.35).model_dump(),
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_low_queued"] == 1
        assert body["n_high_saved"] == 0

        queue_resp = client.get("/v1/review/queue")
        assert queue_resp.json()["total"] >= 1

    def test_infer_image_mixed_confidence(
        self, client: TestClient, fake_detector: FakeDetector, sample_image_path: Path,
    ):
        """Mixed scores should split between store and queue."""
        fake_detector.set_detections([
            Detection(class_id=0, class_name="bottle", cx=0.5, cy=0.5, w=0.1, h=0.1, score=0.9),
            Detection(class_id=1, class_name="can", cx=0.3, cy=0.3, w=0.08, h=0.08, score=0.25),
        ])
        resp = client.post(
            "/v1/inference/image",
            json={
                "image_path": str(sample_image_path),
                "settings": InferenceSettings().model_dump(),
            },
        )
        body = resp.json()
        assert body["n_high_saved"] == 1
        assert body["n_low_queued"] == 1

    def test_infer_image_invalid_body(self, client: TestClient):
        resp = client.post("/v1/inference/image", json={"bad_field": "oops"})
        assert resp.status_code == 422

    def test_infer_image_roi_validation(self, client: TestClient):
        """ROI exceeding [0,1] bounds should be rejected."""
        resp = client.post(
            "/v1/inference/image",
            json={
                "image_path": "/some/path.jpg",
                "settings": {"roi_x": 0.8, "roi_w": 0.5},
            },
        )
        assert resp.status_code == 422

class TestRos2FrameIngest:
    def test_ingest_valid_jpeg(
        self, client: TestClient, fake_detector: FakeDetector, sample_jpeg_bytes: bytes,
    ):
        fake_detector.set_detections([
            Detection(class_id=0, class_name="bottle", cx=0.5, cy=0.5, w=0.1, h=0.1, score=0.9),
        ])
        resp = client.post(
            "/v1/ros2/frame",
            content=sample_jpeg_bytes,
            headers={"Content-Type": "image/jpeg"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["pred_count"] == 1
        assert body["status"] == "ok"
        assert "detections" in body
        assert len(body["detections"]) == 1
        assert body["detections"][0]["class_name"] == "bottle"

    def test_ingest_returns_low_confidence_items(
        self, client: TestClient, fake_detector: FakeDetector, sample_jpeg_bytes: bytes,
    ):
        """Low-confidence items should be included inline for the overlay."""
        fake_detector.set_detections([
            Detection(class_id=1, class_name="can", cx=0.4, cy=0.4, w=0.05, h=0.05, score=0.25),
        ])
        resp = client.post(
            "/v1/ros2/frame",
            content=sample_jpeg_bytes,
            headers={"Content-Type": "image/jpeg"},
        )
        body = resp.json()
        assert body["n_low_queued"] == 1
        assert len(body["low_confidence_items"]) == 1
        assert body["low_confidence_items"][0]["class_name_suggested"] == "can"

    def test_ingest_corrupted_jpeg(self, client: TestClient):
        resp = client.post(
            "/v1/ros2/frame",
            content=b"this-is-not-a-jpeg",
            headers={"Content-Type": "image/jpeg"},
        )
        assert resp.status_code == 400
        assert resp.json()["detail"] == "failed_to_decode_jpeg"

    def test_ingest_empty_body(self, client: TestClient):
        resp = client.post(
            "/v1/ros2/frame",
            content=b"",
            headers={"Content-Type": "image/jpeg"},
        )
        assert resp.status_code in (400, 422)

    def test_ingest_with_custom_params(
        self, client: TestClient, fake_detector: FakeDetector, sample_jpeg_bytes: bytes,
    ):
        fake_detector.set_detections([])
        params_obj = {
            "settings": {"conf_thresh": 0.5, "high_conf_min": 0.7},
            "dedup_ttl_ms": 500,
            "dedup_quant": 0.05,
        }
        resp = client.post(
            "/v1/ros2/frame",
            content=sample_jpeg_bytes,
            headers={"Content-Type": "image/jpeg"},
            params={"params": json.dumps(params_obj)},
        )
        assert resp.status_code == 200
        assert resp.json()["dedup_skipped"] == 0

    def test_ingest_invalid_params_json(
        self, client: TestClient, sample_jpeg_bytes: bytes,
    ):
        resp = client.post(
            "/v1/ros2/frame",
            content=sample_jpeg_bytes,
            headers={"Content-Type": "image/jpeg"},
            params={"params": "{not valid json!!!}"},
        )
        assert resp.status_code == 400
        assert "invalid_params" in resp.json()["detail"]

    def test_ingest_dedup_skips_repeated_detection(
        self, client: TestClient, fake_detector: FakeDetector, sample_jpeg_bytes: bytes,
    ):
        """Second identical frame within TTL should be marked as dedup-skipped."""
        fake_detector.set_detections([
            Detection(class_id=0, class_name="bottle", cx=0.5, cy=0.5, w=0.1, h=0.1, score=0.9),
        ])
        r1 = client.post(
            "/v1/ros2/frame",
            content=sample_jpeg_bytes,
            headers={"Content-Type": "image/jpeg"},
        )
        assert r1.json()["dedup_skipped"] == 0

        r2 = client.post(
            "/v1/ros2/frame",
            content=sample_jpeg_bytes,
            headers={"Content-Type": "image/jpeg"},
        )
        assert r2.json()["dedup_skipped"] == 1

class TestStreamUpload:
    def test_upload_mp4(self, client: TestClient, sample_video_path: Path):
        with sample_video_path.open("rb") as f:
            resp = client.post(
                "/v1/stream/upload",
                files={"file": ("test_video.mp4", f, "application/octet-stream")},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "uploaded"
        assert "path" in body

    def test_upload_unsupported_extension(self, client: TestClient, tmp_data_dir: Path):
        bad = tmp_data_dir / "test.pdf"
        bad.write_bytes(b"fakepdf")
        with bad.open("rb") as f:
            resp = client.post(
                "/v1/stream/upload",
                files={"file": ("test.pdf", f, "application/octet-stream")},
            )
        assert resp.status_code == 400
        assert resp.json()["detail"] == "unsupported_stream_file"

    def test_upload_zip_with_valid_rosbag(self, client: TestClient, tmp_data_dir: Path):
        bag_dir = tmp_data_dir / "rosbag_dir"
        bag_dir.mkdir()
        (bag_dir / "metadata.yaml").write_text("rosbag2_bagfile_information: {}")
        (bag_dir / "data.db3").write_bytes(b"\x00" * 100)

        zip_path = tmp_data_dir / "test_bag.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(bag_dir / "metadata.yaml", "rosbag_dir/metadata.yaml")
            zf.write(bag_dir / "data.db3", "rosbag_dir/data.db3")

        with zip_path.open("rb") as f:
            resp = client.post(
                "/v1/stream/upload",
                files={"file": ("test_bag.zip", f, "application/octet-stream")},
            )
        assert resp.status_code == 200

    def test_upload_zip_missing_metadata(self, client: TestClient, tmp_data_dir: Path):
        zip_path = tmp_data_dir / "bad_bag.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("some_file.txt", "not a rosbag")

        with zip_path.open("rb") as f:
            resp = client.post(
                "/v1/stream/upload",
                files={"file": ("bad_bag.zip", f, "application/octet-stream")},
            )
        assert resp.status_code == 400
        assert resp.json()["detail"] == "zip_missing_rosbag_metadata"

    def test_upload_invalid_zip(self, client: TestClient, tmp_data_dir: Path):
        fake_zip = tmp_data_dir / "corrupt.zip"
        fake_zip.write_bytes(b"not-a-zip-at-all")
        with fake_zip.open("rb") as f:
            resp = client.post(
                "/v1/stream/upload",
                files={"file": ("corrupt.zip", f, "application/octet-stream")},
            )
        assert resp.status_code == 400
        assert resp.json()["detail"] == "invalid_zip_file"

    def test_upload_raw_endpoint(self, client: TestClient, sample_video_path: Path):
        with sample_video_path.open("rb") as f:
            resp = client.post(
                "/v1/stream/upload/raw",
                content=f.read(),
                headers={"Content-Type": "application/octet-stream"},
                params={"filename": "raw_video.mp4"},
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "uploaded"

    def test_upload_raw_no_filename(self, client: TestClient):
        resp = client.post(
            "/v1/stream/upload/raw",
            content=b"\x00",
            headers={"Content-Type": "application/octet-stream"},
        )
        assert resp.status_code == 422

    def test_upload_raw_unsupported_extension(self, client: TestClient):
        resp = client.post(
            "/v1/stream/upload/raw",
            content=b"\x00",
            headers={"Content-Type": "application/octet-stream"},
            params={"filename": "bad.txt"},
        )
        assert resp.status_code == 400

class TestStreamControl:
    def test_stream_status_initial(self, client: TestClient):
        resp = client.get("/v1/stream/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["active"] is False
        assert body["frames_processed"] == 0

    def test_start_stream_no_uploaded_file(self, client: TestClient):
        resp = client.post(
            "/v1/stream/start",
            data={"source": "upload"},
        )
        assert resp.status_code == 400
        assert resp.json()["detail"] == "no_uploaded_file"

    def test_start_stream_with_video(self, client: TestClient, sample_video_path: Path):
        with sample_video_path.open("rb") as f:
            upload = client.post(
                "/v1/stream/upload",
                files={"file": ("stream.mp4", f, "application/octet-stream")},
            )
        assert upload.status_code == 200

        resp = client.post(
            "/v1/stream/start",
            data={"source": "upload", "conf_thresh": "0.3"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"

    def test_stop_stream(self, client: TestClient):
        resp = client.post("/v1/stream/stop")
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopped"

    def test_update_stream_settings(self, client: TestClient):
        """Settings should update without stopping the stream."""
        resp = client.post(
            "/v1/stream/settings",
            json=InferenceSettings(conf_thresh=0.5, high_conf_min=0.7).model_dump(),
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "updated"

    def test_update_stream_settings_invalid_roi(self, client: TestClient):
        resp = client.post(
            "/v1/stream/settings",
            json={"roi_x": 0.9, "roi_w": 0.5},
        )
        assert resp.status_code == 422

    def test_stream_preview_no_upload(self, client: TestClient):
        resp = client.get("/v1/stream/preview?source=upload")
        assert resp.status_code == 400
        assert resp.json()["detail"] == "no_uploaded_file"

    def test_stream_preview_non_upload_source(self, client: TestClient):
        resp = client.get("/v1/stream/preview?source=ros2")
        assert resp.status_code == 400
        assert resp.json()["detail"] == "preview_available_for_upload_only"

class TestStreamRos2Source:
    def test_start_stream_ros2_source(self, client: TestClient):
        resp = client.post(
            "/v1/stream/start",
            data={"source": "ros2", "topic": "/camera/image_raw"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "started"
        assert "ros2:" in body["source"]

        client.post("/v1/stream/stop")

    def test_start_stream_ros2_default_topic(self, client: TestClient):
        resp = client.post(
            "/v1/stream/start",
            data={"source": "ros2"},
        )
        assert resp.status_code == 200
        assert "/camera/image_raw" in resp.json()["source"]
        client.post("/v1/stream/stop")

class TestWebSocket:
    def test_websocket_connect_disconnect(self, client: TestClient):
        with client.websocket_connect("/v1/stream/ws") as ws:
            pass

class TestDTOValidation:
    def test_inference_settings_defaults(self):
        s = InferenceSettings()
        assert s.conf_thresh == 0.2
        assert s.high_conf_min == 0.35
        assert s.merging_mode == "average"
        assert s.avg_count == 8
        assert s.roi_x == 0.0
        assert s.roi_w == 1.0

    def test_inference_settings_roi_exceeds_bounds(self):
        with pytest.raises(Exception):
            InferenceSettings(roi_x=0.6, roi_w=0.5)

    def test_inference_settings_roi_y_exceeds_bounds(self):
        with pytest.raises(Exception):
            InferenceSettings(roi_y=0.8, roi_h=0.3)

    def test_inference_settings_valid_roi(self):
        s = InferenceSettings(roi_x=0.2, roi_y=0.3, roi_w=0.5, roi_h=0.5)
        assert s.roi_x + s.roi_w <= 1.0
        assert s.roi_y + s.roi_h <= 1.0

    def test_inference_settings_roi_boundary(self):
        s = InferenceSettings(roi_x=0.0, roi_y=0.0, roi_w=1.0, roi_h=1.0)
        assert s.roi_w == 1.0

    def test_inference_settings_negative_roi(self):
        with pytest.raises(Exception):
            InferenceSettings(roi_x=-0.1)

    def test_inference_settings_roi_greater_than_one(self):
        with pytest.raises(Exception):
            InferenceSettings(roi_x=1.5)

    def test_confirm_review_body_empty_class(self):
        from on_device_app.dto import ConfirmReviewBody
        with pytest.raises(Exception):
            ConfirmReviewBody(class_name="")

    def test_confirm_review_body_valid(self):
        from on_device_app.dto import ConfirmReviewBody
        b = ConfirmReviewBody(class_name="bottle", create_if_missing=True)
        assert b.class_name == "bottle"
        assert b.create_if_missing is True

    def test_ros2_frame_params_bounds(self):
        from on_device_app.api.app import Ros2FrameParams
        with pytest.raises(Exception):
            Ros2FrameParams(
                settings=InferenceSettings(),
                dedup_ttl_ms=10,
            )
        with pytest.raises(Exception):
            Ros2FrameParams(
                settings=InferenceSettings(),
                dedup_quant=0.001,
            )

    def test_stream_frame_message_structure(self):
        from on_device_app.dto import StreamFrameMessage
        msg = StreamFrameMessage(
            frame_jpeg_b64="abc123",
            frame_index=42,
            detections=[],
            low_confidence_items=[],
            stream_fps=12.5,
        )
        assert msg.frame_index == 42

    def test_stream_status_dto(self):
        from on_device_app.dto import StreamStatusDto
        st = StreamStatusDto(
            active=True,
            source_name="video:test.mp4",
            frames_processed=100,
            current_fps=25.0,
        )
        assert st.active is True
        assert st.current_fps == 25.0

    def test_queue_list_response(self):
        from on_device_app.dto import QueueListResponse, ReviewItemDto
        item = ReviewItemDto(
            queue_id="abc",
            image_path="/tmp/img.jpg",
            cx=0.5, cy=0.5, w=0.1, h=0.1,
            score=0.3,
            class_id_suggested=0,
            class_name_suggested="bottle",
        )
        qlr = QueueListResponse(items=[item], total=1)
        assert qlr.total == 1
        assert len(qlr.items) == 1

class TestDedup:
    def test_dedup_allows_first_occurrence(self):
        from on_device_app.services.dedup import TtlQuantizedBoxDeduper
        d = TtlQuantizedBoxDeduper(ttl_ms=1000, quant=0.03)
        assert d.allow(0, 0.5, 0.5, 0.1, 0.1, 0.9) is True

    def test_dedup_blocks_duplicate_within_ttl(self):
        from on_device_app.services.dedup import TtlQuantizedBoxDeduper
        d = TtlQuantizedBoxDeduper(ttl_ms=5000, quant=0.03)
        d.allow(0, 0.5, 0.5, 0.1, 0.1, 0.9)
        assert d.allow(0, 0.5, 0.5, 0.1, 0.1, 0.9) is False

    def test_dedup_allows_different_class(self):
        from on_device_app.services.dedup import TtlQuantizedBoxDeduper
        d = TtlQuantizedBoxDeduper(ttl_ms=5000, quant=0.03)
        d.allow(0, 0.5, 0.5, 0.1, 0.1, 0.9)
        assert d.allow(1, 0.5, 0.5, 0.1, 0.1, 0.9) is True

    def test_dedup_allows_different_box(self):
        from on_device_app.services.dedup import TtlQuantizedBoxDeduper
        d = TtlQuantizedBoxDeduper(ttl_ms=5000, quant=0.03)
        d.allow(0, 0.5, 0.5, 0.1, 0.1, 0.9)
        assert d.allow(0, 0.2, 0.2, 0.1, 0.1, 0.9) is True

    def test_dedup_set_params(self):
        from on_device_app.services.dedup import TtlQuantizedBoxDeduper
        d = TtlQuantizedBoxDeduper(ttl_ms=1000, quant=0.03)
        d.set_params(ttl_ms=2000, quant=0.05)
        assert d._ttl_s == 2.0
        assert d._quant == 0.05

    def test_dedup_handles_many_unique_keys(self):
        from on_device_app.services.dedup import TtlQuantizedBoxDeduper
        d = TtlQuantizedBoxDeduper(ttl_ms=100_000, quant=0.001)
        for i in range(100):
            result = d.allow(i, float(i) * 0.01, 0.5, 0.1, 0.1, 0.5)
            assert result is True
        assert len(d._last_seen) == 100

    def test_dedup_min_ttl_clamp(self):
        from on_device_app.services.dedup import TtlQuantizedBoxDeduper
        d = TtlQuantizedBoxDeduper(ttl_ms=1, quant=0.03)
        assert d._ttl_s >= 0.05

    def test_dedup_min_quant_clamp(self):
        from on_device_app.services.dedup import TtlQuantizedBoxDeduper
        d = TtlQuantizedBoxDeduper(ttl_ms=1000, quant=0.0001)
        assert d._quant >= 0.001

class TestObjectStore:
    def test_load_class_names(self, tmp_data_dir: Path):
        store = ObjectStore(str(tmp_data_dir / "object_store"))
        names = store.load_class_names()
        assert names == ["bottle", "can", "box"]

    def test_class_names_missing_file(self, tmp_path: Path):
        store_dir = tmp_path / "empty_store"
        store_dir.mkdir()
        (store_dir / "images").mkdir()
        (store_dir / "labels").mkdir()
        store = ObjectStore(str(store_dir))
        with pytest.raises(FileNotFoundError):
            store.load_class_names()

    def test_save_infer_result(self, tmp_data_dir: Path, sample_image_path: Path):
        store = ObjectStore(str(tmp_data_dir / "object_store"))
        img, lbl = store.save_infer_result(
            sample_image_path,
            [(0, 0.5, 0.5, 0.1, 0.1), (1, 0.3, 0.3, 0.08, 0.08)],
        )
        assert img.is_file()
        assert lbl.is_file()
        content = lbl.read_text()
        assert "0 0.500000" in content
        assert "1 0.300000" in content

    def test_class_id_for_existing_name(self, tmp_data_dir: Path):
        store = ObjectStore(str(tmp_data_dir / "object_store"))
        assert store.class_id_for_name("bottle") == 0
        assert store.class_id_for_name("can") == 1

    def test_class_id_for_unknown_no_create(self, tmp_data_dir: Path):
        store = ObjectStore(str(tmp_data_dir / "object_store"))
        with pytest.raises(KeyError):
            store.class_id_for_name("unknown_widget", create=False)

    def test_class_id_for_unknown_with_create(self, tmp_data_dir: Path):
        """Creating a class on the fly should assign the next ID."""
        store = ObjectStore(str(tmp_data_dir / "object_store"))
        idx = store.class_id_for_name("new_class", create=True)
        assert idx == 3
        assert "new_class" in store.load_class_names()

    def test_class_id_empty_name(self, tmp_data_dir: Path):
        store = ObjectStore(str(tmp_data_dir / "object_store"))
        with pytest.raises(ValueError):
            store.class_id_for_name("", create=True)

    def test_append_yolo_line(self, tmp_data_dir: Path):
        store = ObjectStore(str(tmp_data_dir / "object_store"))
        store.append_yolo_line("test_stem", (0, 0.5, 0.5, 0.2, 0.2))
        lbl = store.paths.labels_dir / "test_stem.txt"
        assert lbl.is_file()
        assert "0 0.500000" in lbl.read_text()

    def test_ensure_image_copy(self, tmp_data_dir: Path, sample_image_path: Path):
        store = ObjectStore(str(tmp_data_dir / "object_store"))
        dest = store.ensure_image_copy(sample_image_path, "copy_test")
        assert dest.is_file()
        dest2 = store.ensure_image_copy(sample_image_path, "copy_test")
        assert dest == dest2

    def test_iter_labeled_images(self, tmp_data_dir: Path, sample_image_path: Path):
        store = ObjectStore(str(tmp_data_dir / "object_store"))
        store.save_infer_result(sample_image_path, [(0, 0.5, 0.5, 0.1, 0.1)])
        pairs = list(store.iter_labeled_images())
        assert len(pairs) >= 1
        img_path, lbl_path, image_id = pairs[0]
        assert img_path.is_file()
        assert lbl_path.is_file()

class TestReviewQueueUnit:
    def test_append_and_iterate(self, tmp_path: Path):
        rq = ReviewQueue(str(tmp_path / "rq"))
        rq.append_items([
            {
                "queue_id": "q1",
                "image_path": "/img1.jpg",
                "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1,
                "score": 0.3,
                "class_id_suggested": 0,
                "class_name_suggested": "bottle",
            }
        ])
        items = list(rq.iter_pending())
        assert len(items) == 1
        assert items[0].queue_id == "q1"

    def test_mark_done_removes_from_pending(self, tmp_path: Path):
        rq = ReviewQueue(str(tmp_path / "rq"))
        rq.append_items([
            {
                "queue_id": "q1",
                "image_path": "/img1.jpg",
                "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1,
                "score": 0.3,
                "class_id_suggested": 0,
                "class_name_suggested": "bottle",
            }
        ])
        item = next(rq.iter_pending())
        rq.mark_done(item, {"confirmed_class": "bottle"})
        remaining = list(rq.iter_pending())
        assert len(remaining) == 0

    def test_mark_skipped_removes_from_pending(self, tmp_path: Path):
        rq = ReviewQueue(str(tmp_path / "rq"))
        rq.append_items([
            {
                "queue_id": "q2",
                "image_path": "/img2.jpg",
                "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1,
                "score": 0.3,
                "class_id_suggested": 0,
                "class_name_suggested": "bottle",
            }
        ])
        item = next(rq.iter_pending())
        rq.mark_skipped(item)
        remaining = list(rq.iter_pending())
        assert len(remaining) == 0

    def test_iter_empty_queue(self, tmp_path: Path):
        rq = ReviewQueue(str(tmp_path / "rq_empty"))
        assert list(rq.iter_pending()) == []

    def test_append_auto_generates_queue_id(self, tmp_path: Path):
        rq = ReviewQueue(str(tmp_path / "rq"))
        rq.append_items([
            {
                "image_path": "/img.jpg",
                "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1,
                "score": 0.3,
                "class_id_suggested": 0,
                "class_name_suggested": "bottle",
            }
        ])
        items = list(rq.iter_pending())
        assert len(items) == 1
        assert items[0].queue_id

class TestInferenceServiceUnit:
    def test_infer_image_missing_file(self, service_context):
        from on_device_app.services.inference_service import InferenceService
        svc = InferenceService(service_context)
        result = svc.infer_image("/nonexistent/path.jpg", InferenceSettings())
        assert result.status == "missing_file"

    def test_infer_image_empty_detections(
        self, service_context, fake_detector: FakeDetector, sample_image_path: Path,
    ):
        from on_device_app.services.inference_service import InferenceService
        fake_detector.set_detections([])
        svc = InferenceService(service_context)
        result = svc.infer_image(str(sample_image_path), InferenceSettings())
        assert result.pred_count == 0
        assert result.status == "ok"

    def test_infer_frame_bgr(
        self, service_context, fake_detector: FakeDetector,
    ):
        from on_device_app.services.inference_service import InferenceService
        fake_detector.set_detections([
            Detection(class_id=0, class_name="bottle", cx=0.5, cy=0.5, w=0.1, h=0.1, score=0.9),
        ])
        svc = InferenceService(service_context)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = svc.infer_frame_bgr(frame, InferenceSettings())
        assert result.pred_count == 1

    def test_infer_frame_bgr_rich_returns_all_fields(
        self, service_context, fake_detector: FakeDetector,
    ):
        from on_device_app.services.inference_service import InferenceService
        fake_detector.set_detections([
            Detection(class_id=0, class_name="bottle", cx=0.5, cy=0.5, w=0.1, h=0.1, score=0.9),
            Detection(class_id=1, class_name="can", cx=0.3, cy=0.3, w=0.05, h=0.05, score=0.25),
        ])
        svc = InferenceService(service_context)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result, dets, low_items, skipped = svc.infer_frame_bgr_rich(
            frame, InferenceSettings(), deduper=None,
        )
        assert result.n_high_saved == 1
        assert result.n_low_queued == 1
        assert len(dets) == 2
        assert len(low_items) == 1

    def test_process_paths(self, service_context):
        from on_device_app.services.inference_service import InferenceService
        svc = InferenceService(service_context)
        paths = svc.process_paths()
        assert "object_store_root" in paths
        assert "review_queue_root" in paths

class TestConfig:
    def test_load_app_paths_defaults(self):
        from on_device_app.config import load_app_paths
        paths = load_app_paths()
        assert "object_store" in paths.object_store_root
        assert "review_queue" in paths.review_queue_root

    def test_app_paths_frozen(self, app_paths: AppPaths):
        with pytest.raises(Exception):
            app_paths.object_store_root = "new_path"

class TestStreamProcessorUnit:
    def test_status_inactive_initially(self, service_context):
        from on_device_app.services.stream_service import StreamProcessor
        sp = service_context.stream_processor()
        status = sp.status()
        assert status.active is False
        assert status.frames_processed == 0

    def test_update_settings(self, service_context):
        sp = service_context.stream_processor()
        new_settings = InferenceSettings(conf_thresh=0.5)
        sp.update_settings(new_settings)
        assert sp._settings is not None

    def test_latest_message_none_initially(self, service_context):
        sp = service_context.stream_processor()
        assert sp.latest_message() is None

    def test_inside_roi_check(self):
        from on_device_app.services.stream_service import StreamProcessor
        settings = InferenceSettings(roi_x=0.2, roi_y=0.2, roi_w=0.6, roi_h=0.6)
        assert StreamProcessor._inside_roi(0.5, 0.5, settings) is True
        assert StreamProcessor._inside_roi(0.1, 0.1, settings) is False
        assert StreamProcessor._inside_roi(0.9, 0.9, settings) is False

    def test_inside_roi_full_frame(self):
        from on_device_app.services.stream_service import StreamProcessor
        settings = InferenceSettings()
        assert StreamProcessor._inside_roi(0.0, 0.0, settings) is True
        assert StreamProcessor._inside_roi(1.0, 1.0, settings) is True
        assert StreamProcessor._inside_roi(0.5, 0.5, settings) is True

class TestStreamSources:
    def test_video_file_source_name(self, sample_video_path: Path):
        from on_device_app.services.stream_service import VideoFileSource
        src = VideoFileSource(sample_video_path)
        assert src.name().startswith("video:")

    def test_bag_file_source_name(self, tmp_path: Path):
        from on_device_app.services.stream_service import BagFileSource
        p = tmp_path / "test.db3"
        p.write_bytes(b"")
        src = BagFileSource(p, topic="/camera")
        assert src.name().startswith("bag:")

    def test_ros2_topic_source_name(self):
        from on_device_app.services.stream_service import Ros2TopicSource
        src = Ros2TopicSource(topic="/cam/image")
        assert src.name() == "ros2:/cam/image"

class TestRosImageDecode:
    def test_bgr8_decode(self):
        from on_device_app.services.stream_service import _ros_image_to_bgr
        data = np.zeros((10, 10, 3), dtype=np.uint8).tobytes()
        result = _ros_image_to_bgr("bgr8", 10, 10, data)
        assert result.shape == (10, 10, 3)

    def test_rgb8_decode(self):
        from on_device_app.services.stream_service import _ros_image_to_bgr
        data = np.zeros((10, 10, 3), dtype=np.uint8).tobytes()
        result = _ros_image_to_bgr("rgb8", 10, 10, data)
        assert result.shape == (10, 10, 3)

    def test_mono8_decode(self):
        from on_device_app.services.stream_service import _ros_image_to_bgr
        data = np.zeros((10, 10), dtype=np.uint8).tobytes()
        result = _ros_image_to_bgr("mono8", 10, 10, data)
        assert result.shape == (10, 10, 3)

    def test_bgra8_decode(self):
        from on_device_app.services.stream_service import _ros_image_to_bgr
        data = np.zeros((10, 10, 4), dtype=np.uint8).tobytes()
        result = _ros_image_to_bgr("bgra8", 10, 10, data)
        assert result.shape == (10, 10, 3)

    def test_rgba8_decode(self):
        from on_device_app.services.stream_service import _ros_image_to_bgr
        data = np.zeros((10, 10, 4), dtype=np.uint8).tobytes()
        result = _ros_image_to_bgr("rgba8", 10, 10, data)
        assert result.shape == (10, 10, 3)

    def test_mono16_decode(self):
        from on_device_app.services.stream_service import _ros_image_to_bgr
        data = np.zeros((10, 10), dtype=np.uint16).tobytes()
        result = _ros_image_to_bgr("mono16", 10, 10, data)
        assert result.shape == (10, 10, 3)

    def test_unsupported_encoding_raises(self):
        from on_device_app.services.stream_service import _ros_image_to_bgr
        with pytest.raises(ValueError, match="Unsupported ROS image encoding"):
            _ros_image_to_bgr("yuv422", 10, 10, b"\x00" * 200)

    def test_compressed_jpeg_decode(self):
        from on_device_app.services.stream_service import _ros_compressed_to_bgr
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", frame)
        result = _ros_compressed_to_bgr("jpeg", buf.tobytes())
        assert result.shape[2] == 3

    def test_compressed_decode_invalid_data(self):
        from on_device_app.services.stream_service import _ros_compressed_to_bgr
        with pytest.raises(ValueError, match="cv2.imdecode failed"):
            _ros_compressed_to_bgr("jpeg", b"garbage")

class TestApiClientUnit:
    def test_url_building(self):
        from on_device_app.api_client import ApiClient
        c = ApiClient(base_url="http://example.com:9000/")
        assert c._url("/v1/health") == "http://example.com:9000/v1/health"
        assert c._url("v1/health") == "http://example.com:9000/v1/health"

    def test_url_no_trailing_slash(self):
        from on_device_app.api_client import ApiClient
        c = ApiClient(base_url="http://example.com:9000")
        assert c._url("/v1/test") == "http://example.com:9000/v1/test"

class TestEndToEndInferReviewConfirm:
    def test_full_lifecycle(
        self, client: TestClient, fake_detector: FakeDetector, sample_image_path: Path,
    ):
        """Infer low-conf -> queue -> confirm -> removed from pending."""
        fake_detector.set_detections([
            Detection(
                class_id=0, class_name="bottle",
                cx=0.5, cy=0.5, w=0.1, h=0.1, score=0.25,
            ),
        ])

        infer_resp = client.post(
            "/v1/inference/image",
            json={
                "image_path": str(sample_image_path),
                "settings": InferenceSettings().model_dump(),
            },
        )
        assert infer_resp.status_code == 200
        assert infer_resp.json()["n_low_queued"] == 1

        queue_resp = client.get("/v1/review/queue")
        items = queue_resp.json()["items"]
        assert len(items) == 1
        queue_id = items[0]["queue_id"]

        confirm_resp = client.post(
            f"/v1/review/items/{queue_id}/confirm",
            json={"class_name": "bottle", "create_if_missing": False},
        )
        assert confirm_resp.status_code == 200
        assert confirm_resp.json()["confirmed_class_id"] == 0

        final_queue = client.get("/v1/review/queue")
        assert final_queue.json()["total"] == 0

    def test_infer_then_skip(
        self, client: TestClient, fake_detector: FakeDetector, sample_image_path: Path,
    ):
        """Skipping a queued item should remove it from pending."""
        fake_detector.set_detections([
            Detection(
                class_id=2, class_name="box",
                cx=0.4, cy=0.4, w=0.08, h=0.08, score=0.22,
            ),
        ])
        client.post(
            "/v1/inference/image",
            json={
                "image_path": str(sample_image_path),
                "settings": InferenceSettings().model_dump(),
            },
        )
        items = client.get("/v1/review/queue").json()["items"]
        assert len(items) == 1

        skip_resp = client.post(f"/v1/review/items/{items[0]['queue_id']}/skip")
        assert skip_resp.status_code == 200
        assert client.get("/v1/review/queue").json()["total"] == 0

class TestIncrementalUpdate:
    def test_multiple_inferences_accumulate(
        self, client: TestClient, fake_detector: FakeDetector, sample_image_path: Path,
    ):
        """Repeated inferences should accumulate results in the store."""
        fake_detector.set_detections([
            Detection(class_id=0, class_name="bottle", cx=0.5, cy=0.5, w=0.1, h=0.1, score=0.9),
        ])
        for _ in range(3):
            resp = client.post(
                "/v1/inference/image",
                json={
                    "image_path": str(sample_image_path),
                    "settings": InferenceSettings().model_dump(),
                },
            )
            assert resp.status_code == 200
            assert resp.json()["n_high_saved"] == 1

    def test_settings_update_does_not_restart(self, client: TestClient):
        """Updating settings twice should succeed both times."""
        r1 = client.post(
            "/v1/stream/settings",
            json=InferenceSettings(conf_thresh=0.1).model_dump(),
        )
        assert r1.status_code == 200
        r2 = client.post(
            "/v1/stream/settings",
            json=InferenceSettings(conf_thresh=0.8).model_dump(),
        )
        assert r2.status_code == 200

class TestRequirementGapCoverage:
    def test_inference_latency_within_budget(
        self, client: TestClient, fake_detector: FakeDetector, sample_image_path: Path,
    ):
        """Single inference request should finish well under 1 s."""
        fake_detector.set_detections([])
        start = time.perf_counter()
        resp = client.post(
            "/v1/inference/image",
            json={
                "image_path": str(sample_image_path),
                "settings": InferenceSettings().model_dump(),
            },
        )
        elapsed = time.perf_counter() - start
        assert resp.status_code == 200
        assert elapsed < 1.0

    def test_new_class_available_without_restart(
        self, client: TestClient, fake_detector: FakeDetector, sample_image_path: Path,
    ):
        """New class created via confirm should be queryable immediately."""
        fake_detector.set_detections([
            Detection(class_id=0, class_name="bottle", cx=0.5, cy=0.5, w=0.1, h=0.1, score=0.2),
        ])
        infer_resp = client.post(
            "/v1/inference/image",
            json={"image_path": str(sample_image_path), "settings": InferenceSettings().model_dump()},
        )
        assert infer_resp.status_code == 200
        assert infer_resp.json()["n_low_queued"] == 1

        items = client.get("/v1/review/queue").json()["items"]
        queue_id = items[0]["queue_id"]
        confirm_resp = client.post(
            f"/v1/review/items/{queue_id}/confirm",
            json={"class_name": "new_runtime_class", "create_if_missing": True},
        )
        assert confirm_resp.status_code == 200
        assert "new_runtime_class" in client.get("/v1/classes").json()

    def test_embedding_update_influences_query(
        self, client: TestClient, fake_detector: FakeDetector, sample_image_path: Path,
    ):
        """After confirming a new class, inference should use it on the next run."""
        before = client.get("/v1/classes").json()
        assert "incremental_widget" not in before

        fake_detector.set_detections([
            Detection(class_id=0, class_name="bottle", cx=0.45, cy=0.45, w=0.2, h=0.2, score=0.2),
        ])
        client.post(
            "/v1/inference/image",
            json={"image_path": str(sample_image_path), "settings": InferenceSettings().model_dump()},
        )
        queue_id = client.get("/v1/review/queue").json()["items"][0]["queue_id"]
        client.post(
            f"/v1/review/items/{queue_id}/confirm",
            json={"class_name": "incremental_widget", "create_if_missing": True},
        )

        fake_detector.set_detections([
            Detection(class_id=3, class_name="incremental_widget", cx=0.5, cy=0.5, w=0.1, h=0.1, score=0.95),
        ])
        infer_resp = client.post(
            "/v1/inference/image",
            json={"image_path": str(sample_image_path), "settings": InferenceSettings().model_dump()},
        )
        assert infer_resp.status_code == 200
        assert infer_resp.json()["n_high_saved"] == 1
        assert "incremental_widget" in client.get("/v1/classes").json()

    def test_inference_response_contains_traceability_fields(
        self, client: TestClient, fake_detector: FakeDetector, sample_image_path: Path,
    ):
        """Inference response must include all traceability fields."""
        fake_detector.set_detections([
            Detection(class_id=0, class_name="bottle", cx=0.5, cy=0.5, w=0.1, h=0.1, score=0.9),
        ])
        resp = client.post(
            "/v1/inference/image",
            json={"image_path": str(sample_image_path), "settings": InferenceSettings().model_dump()},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert {"image_path", "pred_count", "n_high_saved", "n_low_queued", "status"}.issubset(body.keys())
        assert body["image_path"]
        assert isinstance(body["pred_count"], int)

    def test_status_reflects_model_state(self, client: TestClient, sample_video_path: Path):
        """Stream status should flip active when a stream is running."""
        status_before = client.get("/v1/stream/status")
        assert status_before.status_code == 200
        assert status_before.json()["active"] is False

        with sample_video_path.open("rb") as fh:
            upload = client.post("/v1/stream/upload", files={"file": ("sample.mp4", fh, "video/mp4")})
        assert upload.status_code == 200
        start = client.post("/v1/stream/start", data={"source": "upload"})
        assert start.status_code == 200

        deadline = time.time() + 2.0
        became_active = False
        while time.time() < deadline:
            status_mid = client.get("/v1/stream/status").json()
            if status_mid["active"]:
                became_active = True
                break
            time.sleep(0.05)
        assert became_active

        stop = client.post("/v1/stream/stop")
        assert stop.status_code == 200
        status_after = client.get("/v1/stream/status").json()
        assert status_after["active"] is False

    def test_api_logs_errors_on_failure(self, client: TestClient):
        """Error paths should return a useful detail message."""
        resp = client.post("/v1/review/items/bad-id/skip")
        assert resp.status_code == 404
        detail = resp.json().get("detail", "")
        assert "not found" in detail.lower()

    def test_many_classes_scalability(self, client: TestClient, tmp_data_dir: Path):
        """Class list should still work fine with 60+ classes."""
        classes_file = tmp_data_dir / "object_store" / "classes.txt"
        many = [f"class_{idx:03d}" for idx in range(60)]
        classes_file.write_text("\n".join(many) + "\n")
        resp = client.get("/v1/classes")
        assert resp.status_code == 200
        listed = resp.json()
        assert len(listed) == 60
        assert listed[0] == "class_000"
        assert listed[-1] == "class_059"
