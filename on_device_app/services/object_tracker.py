from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from on_device_app.services.detector import Detection


@dataclass
class _TrackState:
    track_id: int
    cx: float
    cy: float
    w: float
    h: float
    score: float
    vx: float = 0.0
    vy: float = 0.0
    lost_frames: int = 0
    total_seen: int = 0
    votes: Counter[tuple[int, str]] = field(default_factory=Counter)


@dataclass(frozen=True)
class TrackedObject:
    track_id: int
    class_id: int
    class_name: str
    cx: float
    cy: float
    w: float
    h: float
    score: float
    total_votes: int
    is_label_stable: bool


class ObjectTracker:
    """Track objects with ByteTrack + lightweight motion prediction between detections."""

    def __init__(self, *, min_votes: int = 3, max_lost_frames: int = 12) -> None:
        try:
            import supervision as sv
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "ByteTrack dependency missing. Install `supervision`."
            ) from exc

        # defaults are tuned for online stream processing.
        self._tracker = sv.ByteTrack()
        self._min_votes = max(1, int(min_votes))
        self._max_lost_frames = max(1, int(max_lost_frames))
        self._states: dict[int, _TrackState] = {}

    def reset(self, *, min_votes: int | None = None) -> None:
        if min_votes is not None:
            self._min_votes = max(1, int(min_votes))
        self._states.clear()
        # Supervision trackers expose reset() in recent versions; fallback to new instance.
        if hasattr(self._tracker, "reset"):
            self._tracker.reset()
            return
        tracker_type = type(self._tracker)
        self._tracker = tracker_type()

    def update(self, detections: list[Detection]) -> list[TrackedObject]:
        if not detections:
            return self.predict()

        sv_detections = self._to_sv_detections(detections)
        class_name_by_id: dict[int, str] = {}
        for det in detections:
            cid = int(det.class_id)
            if cid not in class_name_by_id:
                class_name_by_id[cid] = str(det.class_name)
        tracked = self._tracker.update_with_detections(sv_detections)

        xyxy = np.asarray(getattr(tracked, "xyxy", np.empty((0, 4))), dtype=np.float32)
        confidence = np.asarray(getattr(tracked, "confidence", np.empty((0,))), dtype=np.float32)
        class_ids = np.asarray(getattr(tracked, "class_id", np.empty((0,))), dtype=np.int32)
        tracker_ids = np.asarray(getattr(tracked, "tracker_id", np.empty((0,))), dtype=np.int64)
        n = min(len(xyxy), len(confidence), len(class_ids), len(tracker_ids))

        active_ids: set[int] = set()
        for idx in range(n):
            tid = int(tracker_ids[idx])
            if tid < 0:
                continue
            x1, y1, x2, y2 = [float(v) for v in xyxy[idx]]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            score = float(confidence[idx])
            cid = int(class_ids[idx])
            cname = class_name_by_id.get(cid, "unknown")

            state = self._states.get(tid)
            if state is None:
                state = _TrackState(
                    track_id=tid,
                    cx=cx,
                    cy=cy,
                    w=w,
                    h=h,
                    score=score,
                    total_seen=1,
                )
                self._states[tid] = state
            else:
                state.vx = cx - state.cx
                state.vy = cy - state.cy
                state.cx = cx
                state.cy = cy
                state.w = w
                state.h = h
                state.score = score
                state.total_seen += 1
            state.lost_frames = 0
            if cname != "unknown" and cid >= 0:
                state.votes[(cid, cname)] += 1
            active_ids.add(tid)

        self._age_missing(active_ids)
        return self._to_tracked_objects()

    def predict(self) -> list[TrackedObject]:
        stale: list[int] = []
        for tid, state in self._states.items():
            state.cx = self._clamp01(state.cx + state.vx)
            state.cy = self._clamp01(state.cy + state.vy)
            state.lost_frames += 1
            # Slightly decay velocity to avoid uncontrolled drift.
            state.vx *= 0.90
            state.vy *= 0.90
            if state.lost_frames > self._max_lost_frames:
                stale.append(tid)
        for tid in stale:
            self._states.pop(tid, None)
        return self._to_tracked_objects()

    def _to_tracked_objects(self) -> list[TrackedObject]:
        tracked: list[TrackedObject] = []
        for state in self._states.values():
            (class_id, class_name), vote_count = self._majority_label(state)
            total_votes = int(sum(state.votes.values()))
            is_stable = total_votes >= self._min_votes and vote_count >= 1
            tracked.append(
                TrackedObject(
                    track_id=state.track_id,
                    class_id=class_id if is_stable else -1,
                    class_name=class_name if is_stable else "unknown",
                    cx=self._clamp01(state.cx),
                    cy=self._clamp01(state.cy),
                    w=self._clamp01(state.w),
                    h=self._clamp01(state.h),
                    score=float(state.score),
                    total_votes=total_votes,
                    is_label_stable=is_stable,
                )
            )
        tracked.sort(key=lambda t: t.track_id)
        return tracked

    def _majority_label(self, state: _TrackState) -> tuple[tuple[int, str], int]:
        if not state.votes:
            return (-1, "unknown"), 0
        (cid, cname), cnt = state.votes.most_common(1)[0]
        return (int(cid), str(cname)), int(cnt)

    def _age_missing(self, active_ids: set[int]) -> None:
        stale: list[int] = []
        for tid, state in self._states.items():
            if tid in active_ids:
                continue
            state.lost_frames += 1
            state.cx = self._clamp01(state.cx + state.vx)
            state.cy = self._clamp01(state.cy + state.vy)
            state.vx *= 0.90
            state.vy *= 0.90
            if state.lost_frames > self._max_lost_frames:
                stale.append(tid)
        for tid in stale:
            self._states.pop(tid, None)

    @staticmethod
    def _to_sv_detections(detections: list[Detection]):
        import supervision as sv

        xyxy_rows: list[list[float]] = []
        confidence_rows: list[float] = []
        class_rows: list[int] = []
        for det in detections:
            half_w = float(det.w) / 2.0
            half_h = float(det.h) / 2.0
            x1 = float(det.cx) - half_w
            y1 = float(det.cy) - half_h
            x2 = float(det.cx) + half_w
            y2 = float(det.cy) + half_h
            xyxy_rows.append([x1, y1, x2, y2])
            confidence_rows.append(float(det.score))
            class_rows.append(int(det.class_id))
        return sv.Detections(
            xyxy=np.asarray(xyxy_rows, dtype=np.float32),
            confidence=np.asarray(confidence_rows, dtype=np.float32),
            class_id=np.asarray(class_rows, dtype=np.int32),
        )

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))
