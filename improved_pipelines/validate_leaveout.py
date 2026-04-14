"""K-fold embedding cross-validation using cosine KNN (no model forward pass)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from improved_pipelines.embedding_store import ChromaEmbeddingStore


def _mlflow_tracking_uri() -> str | None:
    raw = os.environ.get("MLFLOW_TRACKING_URI")
    if raw is not None and raw.strip() == "":
        return None
    if raw:
        return raw
    root = Path(__file__).resolve().parents[1]
    return "sqlite:///" + (root / "data" / "mlflow.db").as_posix()


def embeddings_dataframe(embed_store: ChromaEmbeddingStore) -> pd.DataFrame:
    rows = embed_store.get_all_embeddings_with_image_metadata()
    if not rows:
        return pd.DataFrame(columns=["chroma_id", "image_id", "class_name", "embedding"])
    return pd.DataFrame(rows)


def kfold_split_by_image_id(
    embeddings_df: pd.DataFrame,
    n_folds: int,
    seed: int,
) -> pd.DataFrame:
    """
    Assign each unique image_id to a fold index [0, n_folds-1].
    Returns DataFrame with columns: image_id, fold.
    """
    if embeddings_df.empty:
        return pd.DataFrame(columns=["image_id", "fold"])
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")

    image_ids = sorted(set(str(v) for v in embeddings_df["image_id"].tolist()))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(image_ids))
    shuffled = [image_ids[int(i)] for i in perm]
    rows = [{"image_id": iid, "fold": int(idx % n_folds)} for idx, iid in enumerate(shuffled)]
    return pd.DataFrame(rows)


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def _knn_predict_one(
    query_vec: np.ndarray,
    train_mat_norm: np.ndarray,
    train_labels: Sequence[str],
    k: int,
) -> Tuple[str, float]:
    sims = train_mat_norm @ query_vec
    k_eff = min(k, len(train_labels))
    top_idx = np.argpartition(sims, -k_eff)[-k_eff:]

    votes: Dict[str, int] = {}
    vote_sim: Dict[str, float] = {}
    for idx in top_idx:
        lbl = str(train_labels[int(idx)])
        votes[lbl] = votes.get(lbl, 0) + 1
        vote_sim[lbl] = vote_sim.get(lbl, 0.0) + float(sims[int(idx)])

    best_label = sorted(votes.keys(), key=lambda lbl: (-votes[lbl], -vote_sim[lbl], lbl))[0]
    confidence = float(np.mean(sims[top_idx])) if len(top_idx) else 0.0
    return best_label, confidence


def knn_classify_holdout_df(
    held_out_df: pd.DataFrame,
    remaining_df: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    cols = [
        "chroma_id",
        "image_id",
        "class_name",
        "predicted_class",
        "correct",
        "knn_confidence",
    ]
    if held_out_df.empty or remaining_df.empty:
        return pd.DataFrame(columns=cols)

    train_labels = [str(v) for v in remaining_df["class_name"].tolist()]
    train_embs = np.vstack([np.asarray(v, dtype=np.float32) for v in remaining_df["embedding"].tolist()])
    train_norm = _l2_normalize_rows(train_embs)

    records: List[Dict[str, Any]] = []
    for row in held_out_df.itertuples(index=False):
        q = np.asarray(row.embedding, dtype=np.float32).reshape(1, -1)
        q_norm = _l2_normalize_rows(q)[0]
        pred, conf = _knn_predict_one(q_norm, train_norm, train_labels, k=max(1, int(k)))
        truth = str(row.class_name)
        records.append(
            {
                "chroma_id": str(row.chroma_id),
                "image_id": str(row.image_id),
                "class_name": truth,
                "predicted_class": pred,
                "correct": pred == truth,
                "knn_confidence": conf,
            }
        )
    return pd.DataFrame(records)


def kfold_cross_validate_df(
    embeddings_df: pd.DataFrame,
    n_folds: int,
    k: int,
    seed: int,
) -> pd.DataFrame:
    """
    Run K-fold CV by image_id:
      - each image_id is assigned to one fold
      - each fold acts once as holdout; remaining folds are train
    """
    cols = [
        "fold",
        "chroma_id",
        "image_id",
        "class_name",
        "predicted_class",
        "correct",
        "knn_confidence",
    ]
    if embeddings_df.empty:
        return pd.DataFrame(columns=cols)

    fold_assignments = kfold_split_by_image_id(embeddings_df, n_folds=n_folds, seed=seed)
    if fold_assignments.empty:
        return pd.DataFrame(columns=cols)
    fold_by_image = {
        str(row.image_id): int(row.fold)
        for row in fold_assignments.itertuples(index=False)
    }

    all_rows: List[pd.DataFrame] = []
    for fold_idx in range(n_folds):
        holdout_mask = embeddings_df["image_id"].astype(str).map(fold_by_image.get) == fold_idx
        held_out_df = embeddings_df[holdout_mask].copy()
        remaining_df = embeddings_df[~holdout_mask].copy()
        fold_df = knn_classify_holdout_df(held_out_df, remaining_df, k=k)
        if not fold_df.empty:
            fold_df.insert(0, "fold", int(fold_idx))
            all_rows.append(fold_df)

    if not all_rows:
        return pd.DataFrame(columns=cols)
    return pd.concat(all_rows, axis=0, ignore_index=True)


def aggregate_leaveout_metrics_df(predictions_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["scope", "class_name", "accuracy", "precision", "recall", "f1", "support", "correct_count"]
    if predictions_df.empty:
        return pd.DataFrame(
            [
                {
                    "scope": "overall",
                    "class_name": "__all__",
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "support": 0,
                    "correct_count": 0,
                }
            ],
            columns=cols,
        )

    y_true = predictions_df["class_name"].astype(str)
    y_pred = predictions_df["predicted_class"].astype(str)
    classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    rows: List[Dict[str, Any]] = []
    for cname in classes:
        tp = int(((y_true == cname) & (y_pred == cname)).sum())
        fp = int(((y_true != cname) & (y_pred == cname)).sum())
        fn = int(((y_true == cname) & (y_pred != cname)).sum())
        support = int((y_true == cname).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        rows.append(
            {
                "scope": "class",
                "class_name": cname,
                "accuracy": float(tp / max(1, support)),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "support": support,
                "correct_count": tp,
            }
        )

    total = len(predictions_df)
    correct = int(predictions_df["correct"].astype(bool).sum())
    acc = correct / max(1, total)
    rows.insert(
        0,
        {
            "scope": "overall",
            "class_name": "__all__",
            "accuracy": float(acc),
            "precision": float(acc),
            "recall": float(acc),
            "f1": float(acc),
            "support": int(total),
            "correct_count": correct,
        },
    )
    return pd.DataFrame(rows, columns=cols)


def build_leaveout_report_dict(
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    *,
    n_folds: int,
    seed: int,
    k: int,
    total_image_ids_count: int,
    warning: Optional[str] = None,
) -> Dict[str, Any]:
    overall = metrics_df[metrics_df["scope"] == "overall"].iloc[0].to_dict()
    by_class = metrics_df[metrics_df["scope"] == "class"].to_dict("records")
    per_fold = []
    if not predictions_df.empty and "fold" in predictions_df.columns:
        for fold in sorted(set(int(v) for v in predictions_df["fold"].tolist())):
            fold_df = predictions_df[predictions_df["fold"] == fold]
            per_fold.append(
                {
                    "fold": int(fold),
                    "support": int(len(fold_df)),
                    "correct_count": int(fold_df["correct"].astype(bool).sum()),
                    "accuracy": float(fold_df["correct"].astype(bool).mean()),
                }
            )
    report: Dict[str, Any] = {
        "n_folds": n_folds,
        "seed": seed,
        "k": k,
        "total_image_ids_count": total_image_ids_count,
        "total_embeddings_evaluated": int(overall["support"]),
        "correct_embeddings": int(overall["correct_count"]),
        "accuracy": float(overall["accuracy"]),
        "precision": float(overall["precision"]),
        "recall": float(overall["recall"]),
        "f1": float(overall["f1"]),
        "per_fold": per_fold,
        "per_class": by_class,
        "predictions": predictions_df.to_dict("records") if not predictions_df.empty else [],
    }
    if warning:
        report["warning"] = warning
    return report


def write_leaveout_report_json(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def log_leaveout_mlflow(
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
    with mlflow.start_run(run_name=os.environ.get("MLFLOW_RUN_NAME", "validate_kfold_knn")):
        mlflow.log_params(
            {
                "chroma_path": chroma_path,
                "collection_name": collection_name,
                "n_folds": report["n_folds"],
                "seed": report["seed"],
                "k": report["k"],
                "total_image_ids_count": report["total_image_ids_count"],
            }
        )
        mlflow.log_metrics(
            {
                "accuracy": float(report["accuracy"]),
                "precision": float(report["precision"]),
                "recall": float(report["recall"]),
                "f1": float(report["f1"]),
                "total_embeddings_evaluated": float(report["total_embeddings_evaluated"]),
            }
        )
        if report_path.is_file():
            mlflow.log_artifact(str(report_path))
