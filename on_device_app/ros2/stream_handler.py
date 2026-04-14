from __future__ import annotations

import queue
import threading

import numpy as np
from PySide6.QtCore import QMutex, QThread, Signal


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
            from cv_bridge import CvBridge
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
            bridge = CvBridge()

            def _cb(msg: RosImage) -> None:
                self._pause_mutex.lock()
                paused = self._paused
                self._pause_mutex.unlock()
                if paused:
                    return
                try:
                    frame_bgr = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                    self._push_frame(np.asarray(frame_bgr).copy())
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

