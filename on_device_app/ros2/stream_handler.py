from __future__ import annotations

import queue
import threading

import cv2
import numpy as np
from PySide6.QtCore import QMutex, QThread, Signal

_DEFAULT_TOPIC = "/zed/zed_node/right/image_rect_color"
_IMAGE_TYPE = "sensor_msgs/msg/Image"

_ENCODING_CV = {
    "bgr8": (np.uint8, 3, None),
    "rgb8": (np.uint8, 3, cv2.COLOR_RGB2BGR),
    "bgra8": (np.uint8, 4, cv2.COLOR_BGRA2BGR),
    "rgba8": (np.uint8, 4, cv2.COLOR_RGBA2BGR),
    "mono8": (np.uint8, 1, cv2.COLOR_GRAY2BGR),
    "8UC3": (np.uint8, 3, None),
    "8UC1": (np.uint8, 1, cv2.COLOR_GRAY2BGR),
    "16UC1": (np.uint16, 1, None),
}


def _ros_image_to_bgr(msg: object) -> np.ndarray:
    """Convert a sensor_msgs/Image to a BGR numpy array without cv_bridge."""
    enc = getattr(msg, "encoding", "bgr8")
    dtype, channels, cvt = _ENCODING_CV.get(enc, (np.uint8, 3, None))
    buf = np.frombuffer(msg.data, dtype=dtype)  # type: ignore[arg-type]
    if channels == 1:
        img = buf.reshape(msg.height, msg.width)  # type: ignore[union-attr]
    else:
        img = buf.reshape(msg.height, msg.width, channels)  # type: ignore[union-attr]
    if cvt is not None:
        img = cv2.cvtColor(img, cvt)
    return img.copy()


def detect_image_topics(timeout_sec: float = 2.0) -> list[str]:
    """Return ROS2 topics publishing sensor_msgs/msg/Image.

    Falls back to ``[_DEFAULT_TOPIC]`` when rclpy is unavailable or no
    matching topics are discovered within *timeout_sec*.
    """
    try:
        import rclpy
        from rclpy.node import Node
    except Exception:  # noqa: BLE001
        return [_DEFAULT_TOPIC]

    did_init = False
    try:
        if not rclpy.ok():
            rclpy.init(args=None)
            did_init = True
        node = Node("_orv_topic_probe")
        topics: list[tuple[str, list[str]]] = node.get_topic_names_and_types()
        node.destroy_node()
        if did_init and rclpy.ok():
            rclpy.shutdown()
        matches = [name for name, types in topics if _IMAGE_TYPE in types]
        return matches or [_DEFAULT_TOPIC]
    except Exception:  # noqa: BLE001
        return [_DEFAULT_TOPIC]


class RosImageStreamHandler(QThread):
    frame_ready = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        topic: str,
        qos_depth: int = 10,
        max_buffer_size: int = 3,
        parent: QThread | None = None,
    ) -> None:
        super().__init__(parent)
        self._topic = topic
        self._qos_depth = qos_depth
        self._running = True
        self._paused = False
        self._pause_mutex = QMutex()
        self._frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max(1, max_buffer_size))
        self._queue_lock = threading.Lock()

    def pause(self) -> None:
        self._pause_mutex.lock()
        self._paused = True
        self._pause_mutex.unlock()

    def resume(self) -> None:
        self._pause_mutex.lock()
        self._paused = False
        self._pause_mutex.unlock()

    def stop(self) -> None:
        self._running = False

    def _push_frame(self, frame: np.ndarray) -> None:
        with self._queue_lock:
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                pass

    def _pop_latest_frame(self) -> np.ndarray | None:
        latest = None
        with self._queue_lock:
            while True:
                try:
                    latest = self._frame_queue.get_nowait()
                except queue.Empty:
                    break
        return latest

    def run(self) -> None:
        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import Image as RosImage
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"ROS2 dependencies missing or broken: {exc}")
            return

        try:
            did_init = False
            if not rclpy.ok():
                rclpy.init(args=None)
                did_init = True
            node = Node("orv_on_device_stream_handler")

            def _cb(msg: RosImage) -> None:
                self._pause_mutex.lock()
                paused = self._paused
                self._pause_mutex.unlock()
                if paused:
                    return
                try:
                    frame_bgr = _ros_image_to_bgr(msg)
                    self._push_frame(frame_bgr)
                except Exception:
                    pass

            node.create_subscription(RosImage, self._topic, _cb, self._qos_depth)
            while self._running:
                rclpy.spin_once(node, timeout_sec=0.1)
                latest = self._pop_latest_frame()
                if latest is not None:
                    self.frame_ready.emit(latest)
            node.destroy_node()
            if did_init and rclpy.ok():
                rclpy.shutdown()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))

