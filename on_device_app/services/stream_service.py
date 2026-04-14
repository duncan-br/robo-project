from __future__ import annotations

import base64
import logging
import tempfile
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Iterator

import cv2
import numpy as np

from improved_pipelines.review_queue import ReviewQueue
from on_device_app.dto import DetectionDto, InferenceSettings, ReviewItemDto, StreamFrameMessage, StreamStatusDto
from on_device_app.services.detector import Detection, ObjectDetector

log = logging.getLogger(__name__)

_INFER_EVERY_N_FRAMES = 5


def _ros_image_to_bgr(encoding: str, height: int, width: int, data: bytes | np.ndarray) -> np.ndarray:
    """Decode a sensor_msgs/Image raw buffer to a BGR numpy array."""
    raw = np.frombuffer(data, dtype=np.uint8) if isinstance(data, (bytes, memoryview)) else np.asarray(data, dtype=np.uint8)
    enc = encoding.lower()
    if enc == "bgr8":
        return raw.reshape(height, width, 3)
    if enc == "rgb8":
        return cv2.cvtColor(raw.reshape(height, width, 3), cv2.COLOR_RGB2BGR)
    if enc == "bgra8":
        return cv2.cvtColor(raw.reshape(height, width, 4), cv2.COLOR_BGRA2BGR)
    if enc == "rgba8":
        return cv2.cvtColor(raw.reshape(height, width, 4), cv2.COLOR_RGBA2BGR)
    if enc in ("mono8", "8uc1"):
        return cv2.cvtColor(raw.reshape(height, width), cv2.COLOR_GRAY2BGR)
    if enc == "mono16" or enc == "16uc1":
        raw16 = np.frombuffer(data, dtype=np.uint16).reshape(height, width)
        scaled = (raw16 / 256).astype(np.uint8)
        return cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"Unsupported ROS image encoding: {encoding}")


def _ros_compressed_to_bgr(fmt: str, data: bytes | np.ndarray) -> np.ndarray:
    """Decode a sensor_msgs/CompressedImage to a BGR numpy array."""
    buf = np.frombuffer(data, dtype=np.uint8) if isinstance(data, (bytes, memoryview)) else np.asarray(data, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"cv2.imdecode failed for CompressedImage format={fmt}")
    return frame


class StreamSource(ABC):
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def frames(self) -> Iterator[tuple[np.ndarray, float]]:
        raise NotImplementedError


class VideoFileSource(StreamSource):
    def __init__(self, path: Path) -> None:
        self._path = path

    def name(self) -> str:
        return f"video:{self._path.name}"

    def frames(self) -> Iterator[tuple[np.ndarray, float]]:
        cap = cv2.VideoCapture(str(self._path))
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_sleep = 1.0 / fps if fps > 0 else 0.04
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame, time.time()
                if frame_sleep > 0:
                    time.sleep(min(frame_sleep, 0.1))
        finally:
            cap.release()


class BagFileSource(StreamSource):
    """Read .bag / .db3 files.  Primary: rosbag2_py (ROS2-native).  Fallback: rosbags pip."""

    _IMAGE_TYPES = {"sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"}

    def __init__(self, path: Path, topic: str | None = None) -> None:
        self._path = path
        self._topic = topic

    def name(self) -> str:
        return f"bag:{self._path.name}"

    def frames(self) -> Iterator[tuple[np.ndarray, float]]:
        try:
            yield from self._frames_rosbag2_py()
            return
        except ImportError:
            log.info("rosbag2_py not available, falling back to rosbags pip package")
        except Exception as exc:  # noqa: BLE001
            log.warning("rosbag2_py reader failed (%s), trying rosbags fallback", exc)

        try:
            yield from self._frames_rosbags()
            return
        except ImportError as exc:
            raise RuntimeError(
                "No bag reader available. Install ROS2 (rosbag2_py) or `pip install rosbags`."
            ) from exc

    def _frames_rosbag2_py(self) -> Iterator[tuple[np.ndarray, float]]:
        import rosbag2_py  # noqa: F811
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import CompressedImage as RosCompressed
        from sensor_msgs.msg import Image as RosImage

        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(
            uri=str(self._path), storage_id="sqlite3",
        )
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        )
        reader.open(storage_options, converter_options)

        topic_types: dict[str, str] = {
            t.name: t.type for t in reader.get_all_topics_and_types()
        }
        image_topics = {
            name: tp for name, tp in topic_types.items()
            if tp in self._IMAGE_TYPES
        }
        if self._topic:
            image_topics = {k: v for k, v in image_topics.items() if k == self._topic}
        if not image_topics:
            raise RuntimeError(f"No image topics in bag. Available: {list(topic_types)}")

        log.info("rosbag2_py reading topics: %s", list(image_topics))
        while reader.has_next():
            topic_name, data, timestamp_ns = reader.read_next()
            tp = image_topics.get(topic_name)
            if tp is None:
                continue
            if tp == "sensor_msgs/msg/CompressedImage":
                msg = deserialize_message(data, RosCompressed)
                frame = _ros_compressed_to_bgr(msg.format, bytes(msg.data))
            else:
                msg = deserialize_message(data, RosImage)
                frame = _ros_image_to_bgr(msg.encoding, msg.height, msg.width, bytes(msg.data))
            yield frame, float(timestamp_ns) / 1e9

    def _frames_rosbags(self) -> Iterator[tuple[np.ndarray, float]]:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore

        typestore = get_typestore(Stores.ROS2_HUMBLE)

        read_paths = [self._path]
        if self._path.is_dir():
            read_paths = [self._path]
        elif self._path.suffix.lower() == ".db3":
            parent = self._path.parent
            if parent.exists() and parent.is_dir():
                read_paths.append(parent)

        last_error: Exception | None = None
        for candidate in read_paths:
            try:
                with AnyReader([candidate], default_typestore=typestore) as reader:
                    connections = [
                        c for c in reader.connections
                        if c.msgtype in self._IMAGE_TYPES
                    ]
                    if self._topic:
                        connections = [c for c in connections if c.topic == self._topic]
                    if not connections:
                        raise RuntimeError(
                            f"No image topic found. Available: "
                            f"{[(c.topic, c.msgtype) for c in reader.connections]}"
                        )
                    log.info("rosbags reading %d connection(s) from %s", len(connections), candidate)
                    for connection, timestamp, rawdata in reader.messages(connections=connections):
                        msg = reader.deserialize(rawdata, connection.msgtype)
                        if "CompressedImage" in connection.msgtype:
                            frame = _ros_compressed_to_bgr(msg.format, bytes(msg.data))
                        else:
                            frame = _ros_image_to_bgr(msg.encoding, msg.height, msg.width, bytes(msg.data))
                        yield frame, float(timestamp) / 1e9
                    return
            except ImportError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                log.warning("rosbags candidate %s failed: %s", candidate, exc)
                continue

        raise RuntimeError(
            "Failed to read rosbag2 recording. "
            "If this is a ROS2 recording, try uploading the full rosbag2 directory."
        ) from last_error


def read_first_stream_frame(path: Path, topic: str | None = None) -> np.ndarray:
    """Return the first BGR frame from an uploaded bag/video path; closes underlying resources."""
    if path.is_dir() or path.suffix.lower() in {".bag", ".db3"}:
        src: StreamSource = BagFileSource(path, topic=topic)
    else:
        src = VideoFileSource(path)
    gen = src.frames()
    try:
        frame_bgr, _ts = next(gen)
        return np.asarray(frame_bgr).copy()
    finally:
        gen.close()


class Ros2TopicSource(StreamSource):
    def __init__(self, topic: str, qos_depth: int = 10) -> None:
        self._topic = topic
        self._qos_depth = qos_depth
        self._stop_event = threading.Event()

    def name(self) -> str:
        return f"ros2:{self._topic}"

    def stop(self) -> None:
        self._stop_event.set()

    def frames(self) -> Iterator[tuple[np.ndarray, float]]:
        try:
            import rclpy
            from cv_bridge import CvBridge
            from rclpy.node import Node
            from sensor_msgs.msg import Image as RosImage
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"ROS2 dependencies missing: {exc}") from exc

        q: Queue[np.ndarray] = Queue(maxsize=3)
        rclpy.init(args=None)
        node = Node("orv_backend_stream_source")
        bridge = CvBridge()

        def _cb(msg: RosImage) -> None:
            frame_bgr = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if q.full():
                try:
                    q.get_nowait()
                except Empty:
                    pass
            q.put_nowait(np.asarray(frame_bgr).copy())

        node.create_subscription(RosImage, self._topic, _cb, self._qos_depth)
        try:
            while not self._stop_event.is_set():
                rclpy.spin_once(node, timeout_sec=0.1)
                try:
                    frame = q.get_nowait()
                except Empty:
                    continue
                yield frame, time.time()
        finally:
            node.destroy_node()
            rclpy.shutdown()


@dataclass
class _Metrics:
    frames_processed: int = 0
    fps: float = 0.0


class StreamProcessor:
    def __init__(self, detector: ObjectDetector, review_queue_factory, object_store_factory) -> None:
        self._detector = detector
        self._review_queue_factory = review_queue_factory
        self._object_store_factory = object_store_factory
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._metrics = _Metrics()
        self._latest_message: StreamFrameMessage | None = None
        self._latest_lock = threading.Lock()
        self._settings: InferenceSettings | None = None
        self._settings_lock = threading.Lock()
        self._source_name = ""
        self._source: StreamSource | None = None

    def start(self, source: StreamSource, settings: InferenceSettings) -> None:
        self.stop()
        self._source = source
        self._source_name = source.name()
        with self._settings_lock:
            self._settings = settings
        self._stop_event.clear()
        self._metrics = _Metrics()
        self._thread = threading.Thread(target=self._run, args=(source,), daemon=True)
        self._thread.start()

    def update_settings(self, settings: InferenceSettings) -> None:
        with self._settings_lock:
            self._settings = settings

    def stop(self) -> None:
        self._stop_event.set()
        if isinstance(self._source, Ros2TopicSource):
            self._source.stop()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None
        self._source = None

    def status(self) -> StreamStatusDto:
        active = self._thread is not None and self._thread.is_alive()
        return StreamStatusDto(
            active=active,
            source_name=self._source_name,
            frames_processed=self._metrics.frames_processed,
            current_fps=self._metrics.fps,
        )

    def latest_message(self) -> StreamFrameMessage | None:
        with self._latest_lock:
            return self._latest_message

    def _run(self, source: StreamSource) -> None:
        last_tick = time.time()
        local_count = 0
        last_detections: list[DetectionDto] = []
        last_low_items: list[ReviewItemDto] = []
        for frame_idx, (frame_bgr, _ts) in enumerate(source.frames(), start=1):
            if self._stop_event.is_set():
                break
            with self._settings_lock:
                settings = self._settings
            if settings is None:
                continue
            run_inference = (frame_idx % _INFER_EVERY_N_FRAMES == 1)
            message = self._process_frame(
                frame_bgr, frame_idx, settings,
                run_inference=run_inference,
                cached_detections=last_detections,
                cached_low_items=last_low_items,
            )
            if run_inference:
                last_detections = message.detections
                last_low_items = message.low_confidence_items
            with self._latest_lock:
                self._latest_message = message
            self._metrics.frames_processed = frame_idx
            local_count += 1
            now = time.time()
            elapsed = now - last_tick
            if elapsed >= 1.0:
                self._metrics.fps = float(local_count) / elapsed
                last_tick = now
                local_count = 0

    def _process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        settings: InferenceSettings,
        *,
        run_inference: bool = True,
        cached_detections: list[DetectionDto] | None = None,
        cached_low_items: list[ReviewItemDto] | None = None,
    ) -> StreamFrameMessage:
        ok, encoded = cv2.imencode(".jpg", frame_bgr)
        b64 = base64.b64encode(encoded.tobytes()).decode("ascii") if ok else ""

        if not run_inference:
            return StreamFrameMessage(
                frame_jpeg_b64=b64,
                frame_index=frame_idx,
                detections=cached_detections or [],
                low_confidence_items=cached_low_items or [],
                stream_fps=self._metrics.fps,
            )

        with tempfile.NamedTemporaryFile(prefix="orv_stream_", suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        cv2.imwrite(str(tmp_path), frame_bgr)
        try:
            detections = self._detector.detect(tmp_path, settings)
        finally:
            tmp_path.unlink(missing_ok=True)

        object_store = self._object_store_factory()
        review_queue: ReviewQueue = self._review_queue_factory()

        high_yolo: list[tuple[int, float, float, float, float]] = []
        low_payloads: list[dict] = []
        detection_dtos: list[DetectionDto] = []
        low_items: list[ReviewItemDto] = []

        for det in detections:
            if not self._inside_roi(det.cx, det.cy, settings):
                continue
            confidence_level = "high" if det.score >= settings.high_conf_min else "low"
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
            if confidence_level == "high":
                high_yolo.append((det.class_id, det.cx, det.cy, det.w, det.h))
            else:
                queue_id = str(uuid.uuid4())
                payload = {
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
                low_payloads.append(payload)
                low_items.append(
                    ReviewItemDto(
                        queue_id=queue_id,
                        image_path="",
                        cx=det.cx,
                        cy=det.cy,
                        w=det.w,
                        h=det.h,
                        score=det.score,
                        class_id_suggested=det.class_id,
                        class_name_suggested=det.class_name,
                    )
                )

        with tempfile.NamedTemporaryFile(prefix="orv_stream_src_", suffix=".jpg", delete=False) as tmp:
            source_image = Path(tmp.name)
        cv2.imwrite(str(source_image), frame_bgr)
        try:
            if high_yolo:
                saved_img, _ = object_store.save_infer_result(source_image, high_yolo, stem_prefix="stream")
                saved_path = str(Path(saved_img).resolve())
            elif low_payloads:
                review_dir = Path("data/review_images")
                review_dir.mkdir(parents=True, exist_ok=True)
                persistent = review_dir / f"review_{uuid.uuid4().hex[:12]}.jpg"
                cv2.imwrite(str(persistent), frame_bgr)
                saved_path = str(persistent.resolve())
            else:
                saved_path = ""

            if low_payloads:
                for payload in low_payloads:
                    payload["image_path"] = saved_path
                review_queue.append_items(low_payloads)
                for item in low_items:
                    item.image_path = saved_path
        finally:
            source_image.unlink(missing_ok=True)

        return StreamFrameMessage(
            frame_jpeg_b64=b64,
            frame_index=frame_idx,
            detections=detection_dtos,
            low_confidence_items=low_items,
            stream_fps=self._metrics.fps,
        )

    @staticmethod
    def _inside_roi(cx: float, cy: float, settings: InferenceSettings) -> bool:
        max_x = settings.roi_x + settings.roi_w
        max_y = settings.roi_y + settings.roi_h
        return settings.roi_x <= cx <= max_x and settings.roi_y <= cy <= max_y
