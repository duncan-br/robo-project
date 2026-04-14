from __future__ import annotations

import time
from dataclasses import dataclass

from improved_pipelines.inference_image import DetectionDeduper


@dataclass(frozen=True)
class QuantizedBoxKey:
    class_id: int
    qcx: int
    qcy: int
    qw: int
    qh: int


class TtlQuantizedBoxDeduper(DetectionDeduper):
    """TTL-based filter that quantizes box coords to suppress duplicate detections
    across adjacent frames without collapsing distinct objects in the same frame."""

    def __init__(self, ttl_ms: int = 900, quant: float = 0.03, max_keys: int = 50_000) -> None:
        self._ttl_s = max(0.05, float(ttl_ms) / 1000.0)
        self._quant = max(0.001, float(quant))
        self._max_keys = max(1000, int(max_keys))
        self._last_seen: dict[QuantizedBoxKey, float] = {}

    def set_params(self, ttl_ms: int | None = None, quant: float | None = None) -> None:
        if ttl_ms is not None:
            self._ttl_s = max(0.05, float(ttl_ms) / 1000.0)
        if quant is not None:
            self._quant = max(0.001, float(quant))

    def allow(self, class_id: int, cx: float, cy: float, w: float, h: float, score: float) -> bool:  # noqa: ARG002
        now = time.monotonic()
        key = QuantizedBoxKey(
            class_id=int(class_id),
            qcx=int(round(float(cx) / self._quant)),
            qcy=int(round(float(cy) / self._quant)),
            qw=int(round(float(w) / self._quant)),
            qh=int(round(float(h) / self._quant)),
        )
        last = self._last_seen.get(key)
        if last is not None and (now - last) <= self._ttl_s:
            return False
        self._last_seen[key] = now

        if len(self._last_seen) > self._max_keys:
            self._prune(now)
        return True

    def _prune(self, now: float) -> None:
        ttl = self._ttl_s
        self._last_seen = {k: t for k, t in self._last_seen.items() if (now - t) <= ttl}
        if len(self._last_seen) <= self._max_keys:
            return
        drop_n = len(self._last_seen) - self._max_keys
        for k in list(self._last_seen.keys())[:drop_n]:
            self._last_seen.pop(k, None)

