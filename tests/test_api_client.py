"""ApiClient integration tests using the FastAPI TestClient transport."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import requests

from on_device_app.api_client import ApiClient
from on_device_app.dto import InferenceSettings


@pytest.fixture()
def live_base_url(client) -> str:
    """Return the base URL for the ASGI transport used by TestClient."""
    return "http://testserver"


@pytest.fixture()
def api(client, live_base_url: str) -> ApiClient:
    """ApiClient wired to use the TestClient transport."""
    ac = ApiClient(base_url=live_base_url, timeout_s=5.0)

    original_session = requests.Session

    class PatchedSession(original_session):
        def __init__(self_inner, *a, **kw):
            super().__init__(*a, **kw)
            self_inner.mount("http://testserver", client._transport)

    _real_get = requests.get
    _real_post = requests.post

    def _patched_get(url, **kw):
        return client.get(url.replace(live_base_url, ""), **{k: v for k, v in kw.items() if k in ("params", "headers")})

    def _patched_post(url, **kw):
        clean = {}
        for k in ("json", "data", "files", "params", "headers", "content"):
            if k in kw:
                clean[k] = kw[k]
        return client.post(url.replace(live_base_url, ""), **clean)

    with patch.object(requests, "get", side_effect=_patched_get), \
         patch.object(requests, "post", side_effect=_patched_post):
        yield ac


class TestApiClientIntegration:
    def test_health(self, api: ApiClient):
        result = api.health()
        assert result == {"status": "ok"}

    def test_classes(self, api: ApiClient):
        result = api.classes()
        assert isinstance(result, list)
        assert "bottle" in result

    def test_list_review_queue_empty(self, api: ApiClient):
        result = api.list_review_queue(limit=50)
        assert result["total"] == 0
        assert result["items"] == []

    def test_stream_status(self, api: ApiClient):
        result = api.stream_status()
        assert result["active"] is False

    def test_stop_stream(self, api: ApiClient):
        result = api.stop_stream()
        assert result["status"] == "stopped"

    def test_update_stream_settings(self, api: ApiClient):
        result = api.update_stream_settings(InferenceSettings(conf_thresh=0.5))
        assert result["status"] == "updated"

    def test_infer_image_missing(self, api: ApiClient):
        result = api.infer_image("/nonexistent.jpg", InferenceSettings())
        assert result["status"] == "missing_file"

    def test_skip_nonexistent_raises(self, api: ApiClient):
        with pytest.raises(Exception):
            api.skip_review_item("fake-id")

    def test_confirm_nonexistent_raises(self, api: ApiClient):
        with pytest.raises(Exception):
            api.confirm_review_item("fake-id", "bottle", create_if_missing=False)
