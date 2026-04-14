"""Embedding robustness validation via small image augmentations."""

from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

from detection.OWL_VIT_v2.utils import read_yolo_label_file
from improved_pipelines.embedding_store import ChromaEmbeddingStore
from improved_pipelines.matching import match_tokens_to_gt


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


def augmentation_map() -> Dict[str, Callable[[Image.Image], Image.Image]]:
    return {
        "brightness_down": lambda im: ImageEnhance.Brightness(im).enhance(0.8),
        "brightness_up": lambda im: ImageEnhance.Brightness(im).enhance(1.2),
        "contrast_down": lambda im: ImageEnhance.Contrast(im).enhance(0.8),
        "contrast_up": lambda im: ImageEnhance.Contrast(im).enhance(1.2),
        "gaussian_noise": _apply_gaussian_noise,
        "gaussian_blur": lambda im: im.filter(ImageFilter.GaussianBlur(radius=1.0)),
    }


def _apply_gaussian_noise(im: Image.Image, sigma: float = 7.5) -> Image.Image:
    arr = np.asarray(im.convert("RGB"), dtype=np.float32)
    noise = np.random.normal(loc=0.0, scale=sigma, size=arr.shape).astype(np.float32)
    out = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def sample_images_for_robustness(
    labeled_pairs_df: pd.DataFrame,
    fraction: float,
    seed: int,
    max_images: Optional[int],
) -> pd.DataFrame:
    if labeled_pairs_df.empty:
        return labeled_pairs_df.copy()
    rows = labeled_pairs_df.to_dict("records")
    rng = random.Random(seed)
    rng.shuffle(rows)
    k = max(1, int(len(rows) * fraction))
    sampled = rows[:k]
    if max_images is not None:
        sampled = sampled[:max_images]
    return pd.DataFrame(sampled)


def extract_augmented_embeddings(
    sample_df: pd.DataFrame,
    *,
    class_names: List[str],
    detector: Any,
    iou_match: float,
    objectness_min: float,
    dedupe_iou: float,
) -> pd.DataFrame:
    cols = [
        "image_id",
        "class_id",
        "class_name",
        "aug_name",
        "embedding",
    ]
    if sample_df.empty:
        return pd.DataFrame(columns=cols)

    aug_map = augmentation_map()
    records: List[Dict[str, Any]] = []
    for row in sample_df.itertuples(index=False):
        image_path = Path(row.image_path)
        label_path = Path(row.label_path)
        image_id = str(row.image_id)
        gt_data = read_yolo_label_file(str(label_path))
        if not gt_data:
            continue
        with Image.open(image_path) as im:
            base = im.convert("RGB")
            orig_w, orig_h = base.size

            for aug_name, aug_fn in aug_map.items():
                aug_im = aug_fn(base.copy())
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
                    aug_im.save(tmp.name, format="JPEG", quality=95)
                    boxes, objectnesses, class_embeddings = detector.process(tmp.name)
                matched = match_tokens_to_gt(
                    boxes,
                    objectnesses,
                    gt_data,
                    orig_w,
                    orig_h,
                    objectness_min=objectness_min,
                    iou_match=iou_match,
                    dedupe_iou=dedupe_iou,
                )
                for tok_idx, gt_row, _objn in matched:
                    cid = int(gt_row[4])
                    if cid < 0 or cid >= len(class_names):
                        continue
                    records.append(
                        {
                            "image_id": image_id,
                            "class_id": cid,
                            "class_name": class_names[cid],
                            "aug_name": aug_name,
                            "embedding": np.asarray(class_embeddings[tok_idx], dtype=np.float32),
                        }
                    )
    return pd.DataFrame(records, columns=cols)


def compare_to_originals(
    augmented_df: pd.DataFrame,
    embed_store: ChromaEmbeddingStore,
) -> pd.DataFrame:
    cols = ["image_id", "class_name", "aug_name", "cosine_sim"]
    if augmented_df.empty:
        return pd.DataFrame(columns=cols)

    raw = embed_store.get_all_embeddings_with_image_metadata()
    originals: Dict[tuple[str, str], np.ndarray] = {}
    grouped: Dict[tuple[str, str], List[np.ndarray]] = {}
    for r in raw:
        key = (str(r["image_id"]), str(r["class_name"]))
        grouped.setdefault(key, []).append(np.asarray(r["embedding"], dtype=np.float32))
    for key, vecs in grouped.items():
        mat = np.vstack(vecs)
        originals[key] = _l2_normalize_rows(mat)

    rows: List[Dict[str, Any]] = []
    for row in augmented_df.itertuples(index=False):
        key = (str(row.image_id), str(row.class_name))
        orig = originals.get(key)
        if orig is None or orig.size == 0:
            continue
        q = np.asarray(row.embedding, dtype=np.float32).reshape(1, -1)
        qn = _l2_normalize_rows(q)[0]
        sims = orig @ qn
        rows.append(
            {
                "image_id": str(row.image_id),
                "class_name": str(row.class_name),
                "aug_name": str(row.aug_name),
                "cosine_sim": float(np.max(sims)),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def aggregate_robustness_metrics_df(comparison_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["scope", "class_name", "aug_name", "mean_cosine", "min_cosine", "std_cosine", "count"]
    if comparison_df.empty:
        return pd.DataFrame(
            [
                {
                    "scope": "overall",
                    "class_name": "__all__",
                    "aug_name": "__all__",
                    "mean_cosine": 0.0,
                    "min_cosine": 0.0,
                    "std_cosine": 0.0,
                    "count": 0,
                }
            ],
            columns=cols,
        )

    rows: List[Dict[str, Any]] = []
    for cname in sorted(set(comparison_df["class_name"].astype(str).tolist())):
        sub = comparison_df[comparison_df["class_name"].astype(str) == cname]
        vals = sub["cosine_sim"].astype(float).to_numpy()
        rows.append(
            {
                "scope": "class",
                "class_name": cname,
                "aug_name": "__all__",
                "mean_cosine": float(np.mean(vals)),
                "min_cosine": float(np.min(vals)),
                "std_cosine": float(np.std(vals)),
                "count": int(len(vals)),
            }
        )

    for aname in sorted(set(comparison_df["aug_name"].astype(str).tolist())):
        sub = comparison_df[comparison_df["aug_name"].astype(str) == aname]
        vals = sub["cosine_sim"].astype(float).to_numpy()
        rows.append(
            {
                "scope": "augmentation",
                "class_name": "__all__",
                "aug_name": aname,
                "mean_cosine": float(np.mean(vals)),
                "min_cosine": float(np.min(vals)),
                "std_cosine": float(np.std(vals)),
                "count": int(len(vals)),
            }
        )

    vals = comparison_df["cosine_sim"].astype(float).to_numpy()
    rows.insert(
        0,
        {
            "scope": "overall",
            "class_name": "__all__",
            "aug_name": "__all__",
            "mean_cosine": float(np.mean(vals)),
            "min_cosine": float(np.min(vals)),
            "std_cosine": float(np.std(vals)),
            "count": int(len(vals)),
        },
    )
    return pd.DataFrame(rows, columns=cols)


def build_robustness_report_dict(
    metrics_df: pd.DataFrame,
    comparisons_df: pd.DataFrame,
    *,
    sample_fraction: float,
    seed: int,
    max_images: Optional[int],
    min_expected_cosine: float,
    warning: Optional[str] = None,
) -> Dict[str, Any]:
    overall = metrics_df[metrics_df["scope"] == "overall"].iloc[0].to_dict()
    by_class = metrics_df[metrics_df["scope"] == "class"].to_dict("records")
    by_aug = metrics_df[metrics_df["scope"] == "augmentation"].to_dict("records")
    report: Dict[str, Any] = {
        "sample_fraction": sample_fraction,
        "seed": seed,
        "max_images": max_images,
        "min_expected_cosine": min_expected_cosine,
        "overall_mean_cosine": float(overall["mean_cosine"]),
        "overall_min_cosine": float(overall["min_cosine"]),
        "overall_std_cosine": float(overall["std_cosine"]),
        "comparisons_count": int(overall["count"]),
        "passes_threshold": float(overall["mean_cosine"]) >= float(min_expected_cosine),
        "per_class": by_class,
        "per_augmentation": by_aug,
        "comparisons": comparisons_df.to_dict("records") if not comparisons_df.empty else [],
    }
    if warning:
        report["warning"] = warning
    return report


def write_robustness_report_json(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def log_robustness_mlflow(
    report: Dict[str, Any],
    *,
    object_store_root: str,
    chroma_path: str,
    collection_name: str,
    report_path: Path,
) -> None:
    uri = _mlflow_tracking_uri()
    if not uri:
        return
    import mlflow

    mlflow.set_tracking_uri(uri)
    with mlflow.start_run(run_name=os.environ.get("MLFLOW_RUN_NAME", "validate_embedding_robustness")):
        mlflow.log_params(
            {
                "object_store_root": object_store_root,
                "chroma_path": chroma_path,
                "collection_name": collection_name,
                "sample_fraction": report["sample_fraction"],
                "seed": report["seed"],
                "max_images": report["max_images"] if report["max_images"] is not None else -1,
                "min_expected_cosine": report["min_expected_cosine"],
            }
        )
        mlflow.log_metrics(
            {
                "overall_mean_cosine": float(report["overall_mean_cosine"]),
                "overall_min_cosine": float(report["overall_min_cosine"]),
                "overall_std_cosine": float(report["overall_std_cosine"]),
                "comparisons_count": float(report["comparisons_count"]),
            }
        )
        if report_path.is_file():
            mlflow.log_artifact(str(report_path))
