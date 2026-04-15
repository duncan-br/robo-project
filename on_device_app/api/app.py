from __future__ import annotations

import asyncio
import base64
import cv2
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from on_device_app.dto import (
    ConfirmReviewBody,
    ImageInferenceRequest,
    InferenceSettings,
    QueueListResponse,
    Ros2FrameIngestResultDto,
    StreamStatusDto,
)
from on_device_app.services import (
    BagFileSource,
    InferenceService,
    ReviewService,
    Ros2TopicSource,
    ServiceContext,
    StreamProcessor,
    VideoFileSource,
    read_first_stream_frame,
)
from on_device_app.ros2.stream_handler import _DEFAULT_TOPIC
from on_device_app.services.dedup import TtlQuantizedBoxDeduper


class Ros2FrameParams(BaseModel):
    settings: InferenceSettings
    dedup_ttl_ms: int = Field(default=900, ge=50, le=30_000)
    dedup_quant: float = Field(default=0.03, ge=0.002, le=0.25)


def create_app(ctx: ServiceContext) -> FastAPI:
    review = ReviewService(ctx)
    inference = InferenceService(ctx)
    stream: StreamProcessor = ctx.stream_processor()
    app = FastAPI(title="Open Robo Vision On-device API", version="1.0.0")
    app.state.ros2_deduper = TtlQuantizedBoxDeduper(ttl_ms=900, quant=0.03)
    app.state.uploaded_stream_file = None
    app.state.upload_dir = Path("data/stream_uploads").resolve()
    app.state.upload_dir.mkdir(parents=True, exist_ok=True)

    def _validate_stream_suffix(filename: str) -> tuple[str, str]:
        suffix = Path(filename or "").suffix.lower()
        if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".bag", ".db3", ".zip"}:
            raise HTTPException(status_code=400, detail="unsupported_stream_file")
        stem = Path(filename or "upload").stem
        return stem, suffix

    def _resolve_uploaded_stream_path(dest: Path, stem: str, suffix: str) -> Path:
        if suffix != ".zip":
            return dest
        extract_dir = app.state.upload_dir / f"stream_{stem}_unzipped"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(dest) as zf:
                zf.extractall(extract_dir)
        except zipfile.BadZipFile as exc:
            raise HTTPException(status_code=400, detail="invalid_zip_file") from exc

        bag_dirs = sorted({p.parent for p in extract_dir.rglob("metadata.yaml") if p.parent.is_dir()})
        if not bag_dirs:
            raise HTTPException(status_code=400, detail="zip_missing_rosbag_metadata")
        return bag_dirs[0]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/classes")
    def get_classes() -> list[str]:
        return review.class_names()

    @app.delete("/v1/classes/{class_name}")
    def delete_class(class_name: str) -> dict:
        store = ctx.object_store()
        embed_store = ctx.embedding_store()
        try:
            class_update = store.remove_class(class_name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        deleted_embeddings = embed_store.delete_by_class_name(class_name)
        return {
            "status": "deleted",
            "class_name": class_name,
            "embeddings_deleted": int(deleted_embeddings),
            **class_update,
        }

    @app.get("/v1/embeddings/stats")
    def embedding_stats() -> dict:
        embed_store = ctx.embedding_store()
        return {
            "count": int(embed_store.count()),
            "class_counts": embed_store.class_counts(),
        }

    @app.delete("/v1/embeddings")
    def clear_embeddings() -> dict:
        embed_store = ctx.embedding_store()
        before = int(embed_store.count())
        embed_store.reset()
        return {
            "status": "cleared",
            "embeddings_before": before,
            "embeddings_after": int(embed_store.count()),
        }

    @app.delete("/v1/database")
    def clear_database(clear_object_store: bool = Query(default=True)) -> dict:
        embed_store = ctx.embedding_store()
        store = ctx.object_store()
        embeddings_before = int(embed_store.count())
        embed_store.reset()
        store_result = {"status": "skipped"}
        if clear_object_store:
            store_result = store.clear_all()

        rq_root = Path(ctx.paths.review_queue_root)
        if rq_root.exists():
            shutil.rmtree(rq_root)
        rq_root.mkdir(parents=True, exist_ok=True)

        return {
            "status": "cleared",
            "embeddings_before": embeddings_before,
            "embeddings_after": int(embed_store.count()),
            "object_store": store_result,
            "review_queue_reset": True,
        }

    @app.get("/v1/review/queue", response_model=QueueListResponse)
    def list_queue(limit: int = Query(default=100, ge=1, le=1000)) -> QueueListResponse:
        items = review.list_pending()
        return QueueListResponse(items=items[:limit], total=len(items))

    @app.post("/v1/review/items/{queue_id}/confirm")
    def confirm_queue_item(queue_id: str, body: ConfirmReviewBody) -> dict:
        try:
            return review.confirm_item(queue_id, body.class_name, create_if_missing=body.create_if_missing)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/review/items/{queue_id}/skip")
    def skip_queue_item(queue_id: str) -> dict:
        try:
            review.skip_item(queue_id)
            return {"queue_id": queue_id, "status": "skipped"}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/v1/inference/image")
    def run_image_inference(body: ImageInferenceRequest) -> dict:
        try:
            return inference.infer_image(body.image_path, body.settings).model_dump()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/ros2/frame", response_model=Ros2FrameIngestResultDto)
    def ingest_ros2_frame(
        request: Request,
        frame_jpeg: bytes = Body(..., media_type="image/jpeg"),
    ) -> Ros2FrameIngestResultDto:
        """Ingest a raw JPEG frame from the ROS2 subscriber, run inference with dedup."""
        params = None
        raw = request.query_params.get("params")
        if raw:
            try:
                params = Ros2FrameParams.model_validate(json.loads(raw))
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=400, detail=f"invalid_params:{exc}") from exc
        if params is None:
            params = Ros2FrameParams(settings=InferenceSettings())

        deduper: TtlQuantizedBoxDeduper = app.state.ros2_deduper
        deduper.set_params(ttl_ms=params.dedup_ttl_ms, quant=params.dedup_quant)

        arr = np.frombuffer(frame_jpeg, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="failed_to_decode_jpeg")
        try:
            result, detections, low_items, skipped = inference.infer_frame_bgr_rich(
                frame, params.settings, deduper=deduper
            )
            return Ros2FrameIngestResultDto(
                **result.model_dump(),
                dedup_skipped=int(skipped),
                detections=detections,
                low_confidence_items=low_items,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/stream/upload")
    async def upload_stream_file(file: UploadFile = File(...)) -> dict:
        stem, suffix = _validate_stream_suffix(file.filename or "upload")
        filename = f"stream_{stem}{suffix or '.bin'}"
        dest = app.state.upload_dir / filename
        with dest.open("wb") as fh:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
        resolved = _resolve_uploaded_stream_path(dest, stem, suffix)
        app.state.uploaded_stream_file = str(resolved)
        return {"status": "uploaded", "path": str(resolved)}

    @app.post("/v1/stream/upload/raw")
    async def upload_stream_file_raw(request: Request, filename: str = Query(..., min_length=1)) -> dict:
        stem, suffix = _validate_stream_suffix(filename)
        out_name = f"stream_{stem}{suffix or '.bin'}"
        dest = app.state.upload_dir / out_name
        with dest.open("wb") as fh:
            async for chunk in request.stream():
                if chunk:
                    fh.write(chunk)
        resolved = _resolve_uploaded_stream_path(dest, stem, suffix)
        app.state.uploaded_stream_file = str(resolved)
        return {"status": "uploaded", "path": str(resolved)}

    @app.post("/v1/stream/start")
    def start_stream(
        source: str = Form(default="upload"),
        topic: str = Form(default=""),
        conf_thresh: float = Form(default=0.2),
        high_conf_min: float = Form(default=0.35),
        merging_mode: str = Form(default="average"),
        avg_count: int = Form(default=8),
        roi_x: float = Form(default=0.0),
        roi_y: float = Form(default=0.0),
        roi_w: float = Form(default=1.0),
        roi_h: float = Form(default=1.0),
        tracking_enabled: bool = Form(default=True),
        tracking_direction: str = Form(default="right_to_left"),
        tracking_line_x: float = Form(default=0.5),
        tracking_max_match_dist: float = Form(default=0.08),
        tracking_max_age_ms: int = Form(default=1200),
        tracking_min_votes: int = Form(default=3),
    ) -> dict:
        settings = InferenceSettings(
            conf_thresh=conf_thresh,
            high_conf_min=high_conf_min,
            merging_mode=merging_mode,
            avg_count=avg_count,
            roi_x=roi_x,
            roi_y=roi_y,
            roi_w=roi_w,
            roi_h=roi_h,
            tracking_enabled=tracking_enabled,
            tracking_direction=tracking_direction,
            tracking_line_x=tracking_line_x,
            tracking_max_match_dist=tracking_max_match_dist,
            tracking_max_age_ms=tracking_max_age_ms,
            tracking_min_votes=tracking_min_votes,
        )
        if source == "ros2":
            stream_source = Ros2TopicSource(topic=topic or _DEFAULT_TOPIC)
        else:
            uploaded = app.state.uploaded_stream_file
            if not uploaded:
                raise HTTPException(status_code=400, detail="no_uploaded_file")
            path = Path(uploaded)
            bag_topic = topic.strip() or None
            if path.is_dir() or path.suffix.lower() in {".bag", ".db3"}:
                stream_source = BagFileSource(path, topic=bag_topic)
            else:
                stream_source = VideoFileSource(path)
        stream.start(stream_source, settings)
        return {"status": "started", "source": stream_source.name()}

    @app.post("/v1/stream/settings")
    def update_stream_settings(settings: InferenceSettings) -> dict:
        stream.update_settings(settings)
        return {"status": "updated"}

    @app.post("/v1/stream/stop")
    def stop_stream() -> dict:
        stream.stop()
        return {"status": "stopped"}

    @app.get("/v1/stream/status", response_model=StreamStatusDto)
    def stream_status() -> StreamStatusDto:
        return stream.status()

    @app.get("/v1/stream/preview")
    def stream_preview(
        source: str = Query(default="upload"),
        topic: str = Query(default=""),
    ) -> dict:
        if (source or "").strip().lower() != "upload":
            raise HTTPException(status_code=400, detail="preview_available_for_upload_only")
        uploaded = app.state.uploaded_stream_file
        if not uploaded:
            raise HTTPException(status_code=400, detail="no_uploaded_file")
        path = Path(uploaded)
        bag_topic = topic.strip() or None
        try:
            frame_bgr = read_first_stream_frame(path, topic=bag_topic)
        except StopIteration as exc:
            raise HTTPException(status_code=409, detail="no_frames_in_stream") from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        ok, encoded = cv2.imencode(".jpg", frame_bgr)
        if not ok:
            raise HTTPException(status_code=500, detail="encode_failed")
        b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
        return {"frame_jpeg_b64": b64}

    @app.websocket("/v1/stream/ws")
    async def stream_ws(websocket: WebSocket) -> None:
        await websocket.accept()
        last_sent_idx = -1
        try:
            while True:
                msg = stream.latest_message()
                if msg is not None and msg.frame_index != last_sent_idx:
                    last_sent_idx = msg.frame_index
                    await websocket.send_json(msg.model_dump())
                await asyncio.sleep(0.05)
        except WebSocketDisconnect:
            return

    return app

