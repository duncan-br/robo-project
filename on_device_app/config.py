from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppPaths:
    object_store_root: str
    review_queue_root: str
    chroma_persist_dir: str
    chroma_collection: str


def load_app_paths() -> AppPaths:
    return AppPaths(
        object_store_root=os.environ.get("OBJECT_STORE_ROOT", "data/object_store"),
        review_queue_root=os.environ.get("REVIEW_QUEUE_DIR", "data/review_queue"),
        chroma_persist_dir=os.environ.get("CHROMA_PERSIST_DIR", "data/chroma_db"),
        chroma_collection=os.environ.get("CHROMA_COLLECTION", "owl_gt_embeddings"),
    )

