"""JSONL review queue for low-confidence detections (operator UI)."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class ReviewItem:
    queue_id: str
    image_path: str
    cx: float
    cy: float
    w: float
    h: float
    score: float
    class_id_suggested: int
    class_name_suggested: str
    raw: Dict[str, Any]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ReviewItem:
        d = dict(d)
        return cls(
            queue_id=d["queue_id"],
            image_path=d["image_path"],
            cx=float(d["cx"]),
            cy=float(d["cy"]),
            w=float(d["w"]),
            h=float(d["h"]),
            score=float(d["score"]),
            class_id_suggested=int(d["class_id_suggested"]),
            class_name_suggested=str(d["class_name_suggested"]),
            raw=d,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "queue_id": self.queue_id,
            "image_path": self.image_path,
            "cx": self.cx,
            "cy": self.cy,
            "w": self.w,
            "h": self.h,
            "score": self.score,
            "class_id_suggested": self.class_id_suggested,
            "class_name_suggested": self.class_name_suggested,
        }


class ReviewQueue:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.pending_path = self.root / "pending.jsonl"
        self.done_path = self.root / "processed.jsonl"

    def append_items(self, items: List[Dict[str, Any]]) -> None:
        with self.pending_path.open("a", encoding="utf-8") as f:
            for raw in items:
                it = dict(raw)
                it.setdefault("queue_id", str(uuid.uuid4()))
                f.write(json.dumps(it) + "\n")

    def iter_pending(self) -> Iterator[ReviewItem]:
        if not self.pending_path.is_file():
            yield from ()
            return
        with self.pending_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield ReviewItem.from_dict(json.loads(line))

    def _rewrite_pending_without(self, remove_ids: set[str]) -> None:
        if not self.pending_path.is_file():
            return
        kept: List[str] = []
        with self.pending_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("queue_id") not in remove_ids:
                    kept.append(line)
        self.pending_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    def mark_done(self, item: ReviewItem, extra: Optional[Dict[str, Any]] = None) -> None:
        rec = {**item.raw, "resolved": True}
        if extra:
            rec.update(extra)
        with self.done_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        self._rewrite_pending_without({item.queue_id})

    def mark_skipped(self, item: ReviewItem) -> None:
        rec = {**item.raw, "skipped": True}
        with self.done_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        self._rewrite_pending_without({item.queue_id})
