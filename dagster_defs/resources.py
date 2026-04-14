"""Configurable resources for paths, Chroma, detector, review queue, MLflow."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import dagster as dg

_REPO_ROOT = Path(__file__).resolve().parents[1]


def default_repo_root() -> Path:
    return _REPO_ROOT


class RepoRootResource(dg.ConfigurableResource):
    """Repository root for resolving default artifact paths."""

    path: str = str(_REPO_ROOT)


class ObjectStoreResource(dg.ConfigurableResource):
    root: str = str(_REPO_ROOT / "data" / "object_store")

    def get_object_store(self):
        from improved_pipelines.object_store import ObjectStore

        return ObjectStore(self.root)


class ChromaStoreResource(dg.ConfigurableResource):
    persist_directory: str = str(_REPO_ROOT / "data" / "chroma_db")
    collection_name: str = "owl_gt_embeddings"

    def get_embedding_store(self):
        from improved_pipelines.embedding_store import ChromaEmbeddingStore

        return ChromaEmbeddingStore(self.persist_directory, collection_name=self.collection_name)


class ReviewQueueResource(dg.ConfigurableResource):
    root: str = str(_REPO_ROOT / "data" / "review_queue")

    def get_queue(self):
        from improved_pipelines.review_queue import ReviewQueue

        return ReviewQueue(self.root)


class OwlDetectorResource(dg.ConfigurableResource):
    """Lazy OWL-ViT detector (one process-wide instance)."""

    def get_detector(self):
        return _get_cached_detector()


@lru_cache(maxsize=1)
def _get_cached_detector():
    from detection.OWL_VIT_v2.image_conditioned import ImageConditionedObjectDetector

    return ImageConditionedObjectDetector()


class MlflowResource(dg.ConfigurableResource):
    """
    MLflow tracking URI; empty string disables logging at materialization time.
    If unset, validate_gold falls back to its default sqlite under data/mlflow.db
    (handled in improved_pipelines.validate_gold._mlflow_tracking_uri).
    """

    tracking_uri: Optional[str] = None

    def effective_tracking_uri(self) -> Optional[str]:
        if self.tracking_uri is not None:
            if str(self.tracking_uri).strip() == "":
                return None
            return str(self.tracking_uri)
        raw = os.environ.get("MLFLOW_TRACKING_URI")
        if raw is not None and raw.strip() == "":
            return None
        if raw:
            return raw
        return "sqlite:///" + (_REPO_ROOT / "data" / "mlflow.db").as_posix()
