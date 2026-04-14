from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from on_device_app.dto import ConfirmReviewBody, InferenceSettings


@dataclass(frozen=True)
class ApiClient:
    base_url: str = "http://127.0.0.1:8000"
    timeout_s: float = 10.0

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def classes(self) -> list[str]:
        return list(self._get("/v1/classes"))

    def list_review_queue(self, limit: int = 100) -> dict[str, Any]:
        return self._get("/v1/review/queue", params={"limit": int(limit)})

    def confirm_review_item(self, queue_id: str, class_name: str, create_if_missing: bool) -> dict[str, Any]:
        body = ConfirmReviewBody(class_name=class_name, create_if_missing=create_if_missing)
        return self._post_json(f"/v1/review/items/{queue_id}/confirm", body.model_dump())

    def skip_review_item(self, queue_id: str) -> dict[str, Any]:
        return self._post_json(f"/v1/review/items/{queue_id}/skip", {})

    def infer_image(self, image_path: str, settings: InferenceSettings) -> dict[str, Any]:
        payload = {"image_path": image_path, "settings": settings.model_dump()}
        return self._post_json("/v1/inference/image", payload)

    def upload_stream_file(self, file_path: str) -> dict[str, Any]:
        path = Path(file_path)
        try:
            with path.open("rb") as fh:
                resp = requests.post(
                    self._url("/v1/stream/upload"),
                    files={"file": (path.name, fh, "application/octet-stream")},
                    timeout=max(self.timeout_s, 300.0),
                )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            if "OSError(22" not in str(exc) and "Errno 22" not in str(exc):
                raise
            with path.open("rb") as fh:
                fallback = requests.post(
                    self._url("/v1/stream/upload/raw"),
                    params={"filename": path.name},
                    data=fh,
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=max(self.timeout_s, 300.0),
                )
            fallback.raise_for_status()
            return fallback.json()

    def start_stream(self, source: str, settings: InferenceSettings, topic: str = "") -> dict[str, Any]:
        data = {
            "source": source,
            "topic": topic,
            "conf_thresh": settings.conf_thresh,
            "high_conf_min": settings.high_conf_min,
            "merging_mode": settings.merging_mode,
            "avg_count": settings.avg_count,
            "roi_x": settings.roi_x,
            "roi_y": settings.roi_y,
            "roi_w": settings.roi_w,
            "roi_h": settings.roi_h,
        }
        return self._post_form("/v1/stream/start", data)

    def update_stream_settings(self, settings: InferenceSettings) -> dict[str, Any]:
        return self._post_json("/v1/stream/settings", settings.model_dump())

    def stop_stream(self) -> dict[str, Any]:
        return self._post_json("/v1/stream/stop", {})

    def stream_status(self) -> dict[str, Any]:
        return self._get("/v1/stream/status")

    def stream_preview(self, source: str = "upload", topic: str = "") -> dict[str, Any]:
        return self._get("/v1/stream/preview", params={"source": source, "topic": topic})

    def ingest_ros2_frame_jpeg(
        self,
        frame_jpeg: bytes,
        settings: InferenceSettings,
        dedup_ttl_ms: int = 900,
        dedup_quant: float = 0.03,
    ) -> dict[str, Any]:
        params = {"settings": settings.model_dump(), "dedup_ttl_ms": int(dedup_ttl_ms), "dedup_quant": float(dedup_quant)}
        return self._post_bytes("/v1/ros2/frame", frame_jpeg, params=params)

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        resp = requests.get(
            self._url(path),
            params=params,
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        return resp.json()

    def _post_json(self, path: str, payload: dict[str, Any]) -> Any:
        resp = requests.post(
            self._url(path),
            json=payload,
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        return resp.json()

    def _post_bytes(self, path: str, data: bytes, params: dict[str, Any]) -> Any:
        resp = requests.post(
            self._url(path),
            params={"params": json.dumps(params)},
            data=data,
            headers={"Content-Type": "image/jpeg"},
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        return resp.json()

    def _post_form(self, path: str, data: dict[str, Any]) -> Any:
        resp = requests.post(
            self._url(path),
            data=data,
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        return resp.json()

    def _url(self, path: str) -> str:
        return self.base_url.rstrip("/") + "/" + path.lstrip("/")

