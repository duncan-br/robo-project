"""Gold validation: sample labeled images, run inference, compare against GT."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from detection.OWL_VIT_v2.image_conditioned import ImageConditionedObjectDetector
from detection.OWL_VIT_v2.utils import calculate_iou, read_yolo_label_file

from improved_pipelines.box_utils import model_box_to_yolo_cxcywh
from improved_pipelines.embedding_store import ChromaEmbeddingStore
from improved_pipelines.inference_image import build_query_embeddings, run_inference_on_image
from improved_pipelines.object_store import ObjectStore, labeled_pairs_dataframe


def _mlflow_tracking_uri() -> str | None:
    """Resolve MLflow URI; empty string disables, unset defaults to local SQLite."""
    raw = os.environ.get("MLFLOW_TRACKING_URI")
    if raw is not None and raw.strip() == "":
        return None
    if raw:
        return raw
    root = Path(__file__).resolve().parents[1]
    return "sqlite:///" + (root / "data" / "mlflow.db").as_posix()


def _cxcywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> List[float]:
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def match_predictions_to_gt(
    pred_classes: np.ndarray,
    pred_boxes_model: np.ndarray,
    pred_scores: np.ndarray,
    gt_rows: Sequence[Sequence[float]],
    orig_w: int,
    orig_h: int,
    iou_thresh: float,
) -> Tuple[int, int, int]:
    """Greedy match: returns (tp, fp, fn)."""
    gt_unused: Set[int] = set(range(len(gt_rows)))
    tp = fp = 0

    order = np.argsort(-pred_scores)
    for j in order:
        cid = int(pred_classes[j])
        box = pred_boxes_model[j]
        cx, cy, w, h = model_box_to_yolo_cxcywh(box, orig_w, orig_h)
        pred_xyxy = _cxcywh_to_xyxy(cx, cy, w, h)
        best_iou = 0.0
        best_gi: int | None = None
        for gi in list(gt_unused):
            g = gt_rows[gi]
            g_cls = int(g[4])
            if g_cls != cid:
                continue
            iou = calculate_iou(pred_xyxy, [float(g[0]), float(g[1]), float(g[2]), float(g[3])])
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_gi is not None and best_iou >= iou_thresh:
            tp += 1
            gt_unused.remove(best_gi)
        else:
            fp += 1

    fn = len(gt_unused)
    return tp, fp, fn


def sample_gold_pairs_df(
    labeled_pairs: pd.DataFrame,
    gold_fraction: float,
    seed: int,
    max_images: Optional[int],
) -> pd.DataFrame:
    """Shuffle and take a gold_fraction slice of labeled pairs (same columns as input)."""
    if labeled_pairs.empty:
        return labeled_pairs.copy()
    df = labeled_pairs.copy()
    rows = df.to_dict("records")
    rng = random.Random(seed)
    rng.shuffle(rows)
    k = max(1, int(len(rows) * gold_fraction))
    sample = rows[:k]
    if max_images is not None:
        sample = sample[:max_images]
    return pd.DataFrame(sample)


def evaluate_gold_per_image_df(
    sample_df: pd.DataFrame,
    embed_store: ChromaEmbeddingStore,
    detector: ImageConditionedObjectDetector,
    conf_thresh: float,
    merging_mode: str,
    avg_count: int,
    iou_match: float,
) -> pd.DataFrame:
    """Run inference on each row; columns include tp, fp, fn, gt_count, pred_count."""
    if sample_df.empty:
        return pd.DataFrame(
            columns=["image_id", "tp", "fp", "fn", "gt_count", "pred_count", "image_path"]
        )

    query_embedding, class_names = build_query_embeddings(embed_store)
    records: List[Dict[str, Any]] = []

    for row in sample_df.itertuples(index=False):
        abs_img = Path(row.image_path)
        abs_lbl = Path(row.label_path)
        image_id = str(row.image_id)
        gt = read_yolo_label_file(str(abs_lbl))
        if not gt:
            continue
        with Image.open(abs_img) as im:
            ow, oh = im.size
        cids, scores, boxes = run_inference_on_image(
            abs_img,
            detector,
            query_embedding,
            class_names,
            conf_thresh=conf_thresh,
            merging_mode=merging_mode,
            avg_count=avg_count,
        )
        tp, fp, fn = match_predictions_to_gt(cids, boxes, scores, gt, ow, oh, iou_match)
        records.append(
            {
                "image_id": image_id,
                "image_path": str(abs_img),
                "label_path": str(abs_lbl),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "gt_count": len(gt),
                "pred_count": len(scores),
            }
        )

    return pd.DataFrame(records)


def aggregate_gold_metrics_df(per_image_df: pd.DataFrame) -> pd.DataFrame:
    """Single-row DataFrame: precision, recall, f1, tp, fp, fn, images_evaluated."""
    if per_image_df.empty:
        return pd.DataFrame(
            [
                {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "images_evaluated": 0,
                }
            ]
        )
    tp = int(per_image_df["tp"].sum())
    fp = int(per_image_df["fp"].sum())
    fn = int(per_image_df["fn"].sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return pd.DataFrame(
        [
            {
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "images_evaluated": len(per_image_df),
            }
        ]
    )


def build_validation_report_dict(
    per_image_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    *,
    gold_fraction: float,
    max_images: Optional[int],
    seed: int,
    sample_size: int,
    warning: Optional[str] = None,
) -> Dict[str, Any]:
    m = metrics_df.iloc[0].to_dict()
    per_image = per_image_df.to_dict("records") if not per_image_df.empty else []
    report: Dict[str, Any] = {
        "gold_fraction": gold_fraction,
        "max_images": max_images,
        "seed": seed,
        "sample_size": sample_size,
        "images_evaluated": int(m["images_evaluated"]),
        "tp": int(m["tp"]),
        "fp": int(m["fp"]),
        "fn": int(m["fn"]),
        "precision": float(m["precision"]),
        "recall": float(m["recall"]),
        "f1": float(m["f1"]),
        "per_image": per_image,
    }
    if warning:
        report["warning"] = warning
    return report


def write_validation_report_json(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def log_validation_mlflow(
    report: Dict[str, Any],
    *,
    object_store: str,
    chroma_path: str,
    merging_mode: str,
    conf_thresh: float,
    iou_match: float,
    report_path: Path,
    max_images: Optional[int] = None,
) -> None:
    uri = _mlflow_tracking_uri()
    if not uri:
        return
    import mlflow

    mlflow.set_tracking_uri(uri)
    with mlflow.start_run(run_name=os.environ.get("MLFLOW_RUN_NAME", "validate_gold")):
        params: Dict[str, Any] = {
            "object_store": object_store,
            "chroma_path": chroma_path,
            "gold_fraction": report["gold_fraction"],
            "seed": report["seed"],
            "merging_mode": merging_mode,
            "conf_thresh": conf_thresh,
            "iou_match": iou_match,
        }
        if max_images is not None:
            params["max_images"] = max_images
        mlflow.log_params(params)
        mlflow.log_metrics(
            {
                "precision": float(report["precision"]),
                "recall": float(report["recall"]),
                "f1": float(report["f1"]),
                "tp": float(report["tp"]),
                "fp": float(report["fp"]),
                "fn": float(report["fn"]),
            }
        )
        if report_path.is_file():
            mlflow.log_artifact(str(report_path))


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Gold validation with MLflow.")
    p.add_argument("--object-store", default=os.environ.get("OBJECT_STORE_ROOT", "data/object_store"))
    p.add_argument("--chroma-path", default=os.environ.get("CHROMA_PERSIST_DIR", "data/chroma_db"))
    p.add_argument("--collection", default="owl_gt_embeddings")
    p.add_argument("--gold-fraction", type=float, default=0.2)
    p.add_argument(
        "--max-images",
        type=int,
        default=None,
        metavar="N",
        help="After sampling by --gold-fraction, evaluate at most N images (quick dev). "
        "Production: omit and use e.g. --gold-fraction 0.2 only.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--merging-mode", default="average")
    p.add_argument("--avg-count", type=int, default=8)
    p.add_argument("--conf-thresh", type=float, default=0.2)
    p.add_argument("--iou-match", type=float, default=0.5)
    p.add_argument(
        "--report-out",
        default=os.environ.get("VALIDATION_REPORT_OUT", "data/validation_report.json"),
    )
    args = p.parse_args(argv)

    store = ObjectStore(args.object_store)
    labeled = labeled_pairs_dataframe(store)
    out_path = Path(args.report_out)

    if labeled.empty:
        report = build_validation_report_dict(
            pd.DataFrame(),
            aggregate_gold_metrics_df(pd.DataFrame()),
            gold_fraction=args.gold_fraction,
            max_images=args.max_images,
            seed=args.seed,
            sample_size=0,
            warning="no_labeled_pairs_in_object_store",
        )
        write_validation_report_json(out_path, report)
        print(json.dumps(report, indent=2))
        return 0

    if args.max_images is not None and args.max_images < 1:
        p.error("--max-images must be >= 1")

    sample = sample_gold_pairs_df(labeled, args.gold_fraction, args.seed, args.max_images)
    embed_store = ChromaEmbeddingStore(args.chroma_path, collection_name=args.collection)

    print("Loading OWL-ViT detector…")
    detector = ImageConditionedObjectDetector()

    per_image = evaluate_gold_per_image_df(
        sample,
        embed_store,
        detector,
        conf_thresh=args.conf_thresh,
        merging_mode=args.merging_mode,
        avg_count=args.avg_count,
        iou_match=args.iou_match,
    )
    metrics = aggregate_gold_metrics_df(per_image)
    report = build_validation_report_dict(
        per_image,
        metrics,
        gold_fraction=args.gold_fraction,
        max_images=args.max_images,
        seed=args.seed,
        sample_size=len(sample),
    )

    write_validation_report_json(out_path, report)
    print(json.dumps(report, indent=2))

    uri = _mlflow_tracking_uri()
    if uri:
        log_validation_mlflow(
            report,
            object_store=args.object_store,
            chroma_path=args.chroma_path,
            merging_mode=args.merging_mode,
            conf_thresh=args.conf_thresh,
            iou_match=args.iou_match,
            report_path=out_path,
            max_images=args.max_images,
        )
        print(
            "\nMLflow: the JSON report is attached to the run under Artifacts → validation_report.json.\n"
            "In the UI: open **Experiments** → **Default** → click the **validate_gold** run → **Artifacts**.\n"
            f"(The home page often shows empty models; runs live under Experiments. Local file: {out_path.resolve()})\n",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
