from __future__ import annotations

import multiprocessing as mp
import queue
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from detection.OWL_VIT_v2.image_conditioned import ImageConditionedObjectDetector
from improved_pipelines.embedding_store import ChromaEmbeddingStore
from improved_pipelines.inference_image import (
    build_query_embeddings,
    route_and_persist,
    run_inference_on_image,
)
from improved_pipelines.object_store import ObjectStore
from improved_pipelines.review_queue import ReviewQueue
from on_device_app.dto import InferenceSettings


def _emit_result(result_q: mp.Queue, payload: dict[str, Any]) -> None:
    try:
        result_q.put_nowait(payload)
    except queue.Full:
        try:
            result_q.get_nowait()
        except queue.Empty:
            pass
        try:
            result_q.put_nowait(payload)
        except queue.Full:
            pass


def _settings_from_payload(payload: dict[str, Any]) -> InferenceSettings:
    return InferenceSettings(
        conf_thresh=float(payload.get("conf_thresh", 0.2)),
        high_conf_min=float(payload.get("high_conf_min", 0.35)),
        merging_mode=str(payload.get("merging_mode", "average")),
        avg_count=int(payload.get("avg_count", 8)),
    )


def _run_inference_for_path(
    path: Path,
    settings: InferenceSettings,
    detector: ImageConditionedObjectDetector,
    embed_store: ChromaEmbeddingStore,
    object_store: ObjectStore,
    review_queue: ReviewQueue,
) -> dict[str, Any]:
    if not path.is_file():
        return {
            "image_path": str(path),
            "pred_count": 0,
            "n_high_saved": 0,
            "n_low_queued": 0,
            "status": "missing_file",
        }
    query_embedding, class_names = build_query_embeddings(embed_store)
    with Image.open(path) as img:
        width, height = img.size
    class_ids, scores, boxes = run_inference_on_image(
        path,
        detector,
        query_embedding,
        class_names,
        conf_thresh=settings.conf_thresh,
        merging_mode=settings.merging_mode,
        avg_count=settings.avg_count,
    )
    n_high, n_low = route_and_persist(
        path,
        class_ids,
        scores,
        boxes,
        class_names,
        width,
        height,
        settings.high_conf_min,
        object_store,
        review_queue,
    )
    return {
        "image_path": str(path),
        "pred_count": int(len(scores)),
        "n_high_saved": int(n_high),
        "n_low_queued": int(n_low),
        "status": "ok",
    }


def inference_worker_main(job_q: mp.Queue, result_q: mp.Queue, paths: dict[str, str]) -> None:
    detector = ImageConditionedObjectDetector()
    embed_store = ChromaEmbeddingStore(
        persist_directory=paths["chroma_persist_dir"],
        collection_name=paths["chroma_collection"],
    )
    object_store = ObjectStore(paths["object_store_root"])
    review_queue = ReviewQueue(paths["review_queue_root"])

    while True:
        job = job_q.get()
        if job.get("cmd") == "shutdown":
            return
        try:
            settings = _settings_from_payload(job.get("settings", {}))
            kind = job.get("kind")
            if kind == "image_path":
                payload = _run_inference_for_path(
                    Path(str(job.get("image_path", ""))).resolve(),
                    settings,
                    detector,
                    embed_store,
                    object_store,
                    review_queue,
                )
                _emit_result(result_q, payload)
                continue
            if kind == "jpeg_frame":
                jpeg_bytes = job.get("jpeg")
                if not isinstance(jpeg_bytes, (bytes, bytearray)):
                    _emit_result(
                        result_q,
                        {
                            "image_path": "",
                            "pred_count": 0,
                            "n_high_saved": 0,
                            "n_low_queued": 0,
                            "status": "invalid_frame_payload",
                        },
                    )
                    continue
                arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    _emit_result(
                        result_q,
                        {
                            "image_path": "",
                            "pred_count": 0,
                            "n_high_saved": 0,
                            "n_low_queued": 0,
                            "status": "failed_to_decode_frame",
                        },
                    )
                    continue
                with tempfile.NamedTemporaryFile(prefix="orv_live_", suffix=".jpg", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                cv2.imwrite(str(tmp_path), frame)
                try:
                    payload = _run_inference_for_path(
                        tmp_path,
                        settings,
                        detector,
                        embed_store,
                        object_store,
                        review_queue,
                    )
                    _emit_result(result_q, payload)
                finally:
                    tmp_path.unlink(missing_ok=True)
                continue
            _emit_result(
                result_q,
                {
                    "image_path": "",
                    "pred_count": 0,
                    "n_high_saved": 0,
                    "n_low_queued": 0,
                    "status": f"unknown_job_kind:{kind}",
                },
            )
        except Exception as exc:  # noqa: BLE001
            _emit_result(
                result_q,
                {
                    "image_path": "",
                    "pred_count": 0,
                    "n_high_saved": 0,
                    "n_low_queued": 0,
                    "status": f"error:{exc}",
                },
            )


class InferenceProcessController:
    def __init__(self, paths: dict[str, str], max_jobs: int = 2, max_results: int = 64) -> None:
        self._paths = paths
        self._ctx = mp.get_context("spawn")
        self._job_q: mp.Queue = self._ctx.Queue(maxsize=max_jobs)
        self._result_q: mp.Queue = self._ctx.Queue(maxsize=max_results)
        self._process: mp.Process | None = None

    def start(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._process = self._ctx.Process(
            target=inference_worker_main,
            args=(self._job_q, self._result_q, self._paths),
            daemon=True,
        )
        self._process.start()

    def stop(self) -> None:
        if self._process is None:
            return
        if self._process.is_alive():
            try:
                self._job_q.put_nowait({"cmd": "shutdown"})
            except queue.Full:
                pass
            self._process.join(timeout=3)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2)
        self._process = None

    def submit_image_path(self, image_path: str, settings: InferenceSettings) -> None:
        self._submit_job(
            {
                "kind": "image_path",
                "image_path": image_path,
                "settings": settings.model_dump(),
            }
        )

    def submit_frame(self, frame: np.ndarray, settings: InferenceSettings) -> None:
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            return
        self._submit_job(
            {
                "kind": "jpeg_frame",
                "jpeg": encoded.tobytes(),
                "settings": settings.model_dump(),
            }
        )

    def _submit_job(self, job: dict[str, Any]) -> None:
        try:
            self._job_q.put_nowait(job)
        except queue.Full:
            try:
                self._job_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._job_q.put_nowait(job)
            except queue.Full:
                pass

    def drain_results(self, max_items: int = 16) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for _ in range(max_items):
            try:
                out.append(self._result_q.get_nowait())
            except queue.Empty:
                break
        return out

