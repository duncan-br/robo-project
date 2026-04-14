"""Embedding coherence validation (intra-class compactness, inter-class separation)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _mlflow_tracking_uri() -> str | None:
    raw = os.environ.get("MLFLOW_TRACKING_URI")
    if raw is not None and raw.strip() == "":
        return None
    if raw:
        return raw
    root = Path(__file__).resolve().parents[1]
    return "sqlite:///" + (root / "data" / "mlflow.db").as_posix()


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def compute_intra_class_similarity_df(grouped_embeddings: Dict[str, List[np.ndarray]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for class_name in sorted(grouped_embeddings.keys()):
        arr = np.vstack([np.asarray(v, dtype=np.float32) for v in grouped_embeddings[class_name]])
        count = int(arr.shape[0])
        if count < 2:
            rows.append(
                {
                    "class_name": class_name,
                    "count": count,
                    "mean_cosine_sim": 1.0 if count == 1 else 0.0,
                    "min_cosine_sim": 1.0 if count == 1 else 0.0,
                    "std_cosine_sim": 0.0,
                }
            )
            continue
        norm = _l2_normalize_rows(arr)
        sims = norm @ norm.T
        tri = sims[np.triu_indices(count, k=1)]
        rows.append(
            {
                "class_name": class_name,
                "count": count,
                "mean_cosine_sim": float(np.mean(tri)),
                "min_cosine_sim": float(np.min(tri)),
                "std_cosine_sim": float(np.std(tri)),
            }
        )
    return pd.DataFrame(rows)


def compute_inter_class_separation_df(grouped_embeddings: Dict[str, List[np.ndarray]]) -> pd.DataFrame:
    class_names = sorted(grouped_embeddings.keys())
    if len(class_names) < 2:
        return pd.DataFrame(columns=["class_a", "class_b", "centroid_cosine_sim"])

    centroids: Dict[str, np.ndarray] = {}
    for cname in class_names:
        arr = np.vstack([np.asarray(v, dtype=np.float32) for v in grouped_embeddings[cname]])
        centroids[cname] = np.mean(arr, axis=0)

    rows: List[Dict[str, Any]] = []
    for i, a in enumerate(class_names):
        ca = centroids[a].astype(np.float32)
        ca = ca / max(1e-12, float(np.linalg.norm(ca)))
        for b in class_names[i + 1 :]:
            cb = centroids[b].astype(np.float32)
            cb = cb / max(1e-12, float(np.linalg.norm(cb)))
            rows.append(
                {
                    "class_a": a,
                    "class_b": b,
                    "centroid_cosine_sim": float(np.dot(ca, cb)),
                }
            )
    return pd.DataFrame(rows)


def assess_class_health_df(
    intra_df: pd.DataFrame,
    inter_df: pd.DataFrame,
    *,
    min_exemplars: int,
    min_intra_sim: float,
    max_inter_sim: float,
) -> pd.DataFrame:
    if intra_df.empty:
        return pd.DataFrame(
            columns=[
                "class_name",
                "count",
                "mean_cosine_sim",
                "max_centroid_similarity_to_other",
                "is_sparse",
                "is_low_intra_similarity",
                "is_high_inter_overlap",
                "is_healthy",
            ]
        )

    max_inter_by_class: Dict[str, float] = {}
    for row in inter_df.itertuples(index=False):
        sim = float(row.centroid_cosine_sim)
        max_inter_by_class[row.class_a] = max(sim, max_inter_by_class.get(row.class_a, -1.0))
        max_inter_by_class[row.class_b] = max(sim, max_inter_by_class.get(row.class_b, -1.0))

    rows: List[Dict[str, Any]] = []
    for row in intra_df.itertuples(index=False):
        cname = str(row.class_name)
        count = int(row.count)
        mean_intra = float(row.mean_cosine_sim)
        max_inter = float(max_inter_by_class.get(cname, -1.0))

        sparse = count < int(min_exemplars)
        low_intra = mean_intra < float(min_intra_sim)
        high_inter = max_inter >= float(max_inter_sim)
        healthy = not (sparse or low_intra or high_inter)

        rows.append(
            {
                "class_name": cname,
                "count": count,
                "mean_cosine_sim": mean_intra,
                "max_centroid_similarity_to_other": max_inter,
                "is_sparse": sparse,
                "is_low_intra_similarity": low_intra,
                "is_high_inter_overlap": high_inter,
                "is_healthy": healthy,
            }
        )
    return pd.DataFrame(rows)


def build_coherence_report_dict(
    intra_df: pd.DataFrame,
    inter_df: pd.DataFrame,
    class_health_df: pd.DataFrame,
    *,
    min_exemplars: int,
    min_intra_sim: float,
    max_inter_sim: float,
    warning: Optional[str] = None,
) -> Dict[str, Any]:
    unhealthy = int((~class_health_df["is_healthy"]).sum()) if not class_health_df.empty else 0
    report: Dict[str, Any] = {
        "classes_total": int(len(class_health_df)),
        "classes_unhealthy": unhealthy,
        "min_exemplars": min_exemplars,
        "min_intra_sim": min_intra_sim,
        "max_inter_sim": max_inter_sim,
        "intra_class": intra_df.to_dict("records") if not intra_df.empty else [],
        "inter_class": inter_df.to_dict("records") if not inter_df.empty else [],
        "class_health": class_health_df.to_dict("records") if not class_health_df.empty else [],
    }
    if warning:
        report["warning"] = warning
    return report


def write_coherence_report_json(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def log_coherence_mlflow(
    report: Dict[str, Any],
    *,
    chroma_path: str,
    collection_name: str,
    report_path: Path,
) -> None:
    uri = _mlflow_tracking_uri()
    if not uri:
        return
    import mlflow

    mlflow.set_tracking_uri(uri)
    with mlflow.start_run(run_name=os.environ.get("MLFLOW_RUN_NAME", "validate_embedding_coherence")):
        mlflow.log_params(
            {
                "chroma_path": chroma_path,
                "collection_name": collection_name,
                "min_exemplars": report["min_exemplars"],
                "min_intra_sim": report["min_intra_sim"],
                "max_inter_sim": report["max_inter_sim"],
            }
        )
        mlflow.log_metrics(
            {
                "classes_total": float(report["classes_total"]),
                "classes_unhealthy": float(report["classes_unhealthy"]),
            }
        )
        if report_path.is_file():
            mlflow.log_artifact(str(report_path))
