from __future__ import annotations

import time
from dataclasses import dataclass

from on_device_app.services.detector import Detection


@dataclass
class _Track:
    track_id: int
    class_id: int
    cx: float
    cy: float
    last_seen_s: float
    registered: bool = False


class ConveyorCrossingTracker:
    """Centroid tracker that emits detections once on line crossing."""

    def __init__(self) -> None:
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def filter_crossings(
        self,
        detections: list[Detection],
        *,
        line_x: float,
        max_match_dist: float,
        max_age_ms: int,
        direction: str = "right_to_left",
        now_s: float | None = None,
    ) -> list[Detection]:
        now = now_s if now_s is not None else time.time()
        self._expire_stale(now, max_age_ms=max_age_ms)
        emitted: list[Detection] = []

        for det in detections:
            track = self._match_track(det, max_match_dist=max_match_dist)
            if track is None:
                track = self._create_track(det, now_s=now)
                continue

            prev_cx = track.cx
            track.cx = float(det.cx)
            track.cy = float(det.cy)
            track.last_seen_s = now

            if track.registered:
                continue
            if self._crossed(prev_cx=prev_cx, curr_cx=det.cx, line_x=line_x, direction=direction):
                track.registered = True
                emitted.append(det)

        return emitted

    def _create_track(self, det: Detection, now_s: float) -> _Track:
        track = _Track(
            track_id=self._next_id,
            class_id=int(det.class_id),
            cx=float(det.cx),
            cy=float(det.cy),
            last_seen_s=now_s,
        )
        self._next_id += 1
        self._tracks[track.track_id] = track
        return track

    def _match_track(self, det: Detection, *, max_match_dist: float) -> _Track | None:
        best_track: _Track | None = None
        best_dist = float("inf")
        cx = float(det.cx)
        cy = float(det.cy)

        for track in self._tracks.values():
            if track.class_id != int(det.class_id):
                continue
            dx = track.cx - cx
            dy = track.cy - cy
            dist = (dx * dx + dy * dy) ** 0.5
            if dist > max_match_dist:
                continue
            if dist < best_dist:
                best_dist = dist
                best_track = track
        return best_track

    @staticmethod
    def _crossed(*, prev_cx: float, curr_cx: float, line_x: float, direction: str) -> bool:
        if direction == "right_to_left":
            return prev_cx > line_x >= curr_cx
        if direction == "left_to_right":
            return prev_cx < line_x <= curr_cx
        return False

    def _expire_stale(self, now_s: float, *, max_age_ms: int) -> None:
        ttl_s = max(0.05, float(max_age_ms) / 1000.0)
        stale_ids = [
            tid
            for tid, tr in self._tracks.items()
            if (now_s - tr.last_seen_s) > ttl_s
        ]
        for tid in stale_ids:
            self._tracks.pop(tid, None)
