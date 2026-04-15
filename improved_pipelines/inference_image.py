"""OWL-ViT image inference with Chroma query merge and confidence routing."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from detection.OWL_VIT_v2.image_conditioned import ImageConditionedObjectDetector

from improved_pipelines.box_utils import model_boxes_to_yolo_lines
from improved_pipelines.embedding_store import ChromaEmbeddingStore
from improved_pipelines.inference_manifest import build_inference_manifest_df
from improved_pipelines.object_store import ObjectStore
from improved_pipelines.review_queue import ReviewQueue


class DetectionDeduper:
    """Base class for stateful duplicate-detection filters (see TtlQuantizedBoxDeduper)."""

    def allow(self, class_id: int, cx: float, cy: float, w: float, h: float, score: float) -> bool:  # noqa: ARG002
        return True


def build_query_embeddings(
    embed_store: ChromaEmbeddingStore,
) -> Tuple[Dict[str, List[np.ndarray]], List[str]]:
    grouped = embed_store.get_embeddings_grouped_by_class_name()
    if not grouped:
        raise ValueError(
            "Chroma collection is empty. Run preload_embeddings or check CHROMA_PERSIST_DIR."
        )
    class_names = sorted(grouped.keys())
    qd: Dict[str, List[np.ndarray]] = {
        cn: [np.asarray(e, dtype=np.float32) for e in grouped[cn]] for cn in class_names
    }
    return qd, class_names


def run_inference_on_image(
    image_path: Path,
    detector: ImageConditionedObjectDetector,
    query_embedding: Dict[str, List[np.ndarray]],
    class_names: List[str],
    conf_thresh: float,
    merging_mode: str,
    avg_count: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    abs_path = str(image_path.resolve())
    class_ids, scores, boxes, _ = detector.process_with_embeddings(
        abs_path,
        query_embedding,
        class_names,
        conf_thresh=conf_thresh,
        avg_count=avg_count,
        merging_mode=merging_mode,
    )
    return np.asarray(class_ids), np.asarray(scores), np.asarray(boxes)


def run_inference_on_frame(
    frame_bgr: np.ndarray,
    detector: ImageConditionedObjectDetector,
    query_embedding: Dict[str, List[np.ndarray]],
    class_names: List[str],
    conf_thresh: float,
    merging_mode: str,
    avg_count: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Like ``run_inference_on_image`` but accepts a BGR numpy array directly."""
    class_ids, scores, boxes, _ = detector.process_with_embeddings_bgr(
        frame_bgr,
        query_embedding,
        class_names,
        conf_thresh=conf_thresh,
        avg_count=avg_count,
        merging_mode=merging_mode,
    )
    return np.asarray(class_ids), np.asarray(scores), np.asarray(boxes)


def route_and_persist(
    image_path: Path,
    class_ids: np.ndarray,
    scores: np.ndarray,
    boxes: np.ndarray,
    class_names: List[str],
    orig_w: int,
    orig_h: int,
    high_conf_min: float,
    object_store: ObjectStore,
    review_queue: ReviewQueue,
    deduper: DetectionDeduper | None = None,
) -> Tuple[int, int]:
    """
    Returns (n_high_saved, n_low_queued).
    """
    yolo_lines = model_boxes_to_yolo_lines(boxes, class_ids, orig_w, orig_h)
    high: List[Tuple[int, float, float, float, float]] = []
    low_payloads: List[Dict[str, Any]] = []

    for line, score, box in zip(yolo_lines, scores, boxes):
        cid, cx, cy, w, h = line
        if deduper is not None and not deduper.allow(int(cid), float(cx), float(cy), float(w), float(h), float(score)):
            continue
        if float(score) >= high_conf_min:
            high.append(line)
        else:
            cname = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
            low_payloads.append(
                {
                    "image_path": str(image_path.resolve()),
                    "cx": cx,
                    "cy": cy,
                    "w": w,
                    "h": h,
                    "score": float(score),
                    "class_id_suggested": int(cid),
                    "class_name_suggested": cname,
                }
            )

    n_high = 0
    if high:
        object_store.save_infer_result(image_path, high)
        n_high = len(high)

    if low_payloads:
        review_queue.append_items(low_payloads)

    return n_high, len(low_payloads)


def run_inference_batch_df(
    manifest_df: pd.DataFrame,
    embed_store: ChromaEmbeddingStore,
    detector: ImageConditionedObjectDetector,
    object_store: ObjectStore,
    review_queue: ReviewQueue,
    conf_thresh: float,
    merging_mode: str,
    avg_count: int,
    high_conf_min: float,
) -> pd.DataFrame:
    """
    Run inference + confidence routing for each row in ``manifest_df`` (column ``image_path``).
    Returns one row per processed image with routing counts.
    """
    if manifest_df.empty:
        return pd.DataFrame(
            columns=[
                "image_path",
                "n_high_saved",
                "n_low_queued",
                "pred_count",
                "status",
            ]
        )

    query_embedding, class_names = build_query_embeddings(embed_store)
    rows: List[Dict[str, Any]] = []

    for path_str in manifest_df["image_path"].astype(str):
        img_path = Path(path_str).resolve()
        if not img_path.is_file():
            rows.append(
                {
                    "image_path": path_str,
                    "n_high_saved": 0,
                    "n_low_queued": 0,
                    "pred_count": 0,
                    "status": "missing_file",
                }
            )
            continue
        try:
            with Image.open(img_path) as im:
                ow, oh = im.size
            cids, scores, boxes = run_inference_on_image(
                img_path,
                detector,
                query_embedding,
                class_names,
                conf_thresh=conf_thresh,
                merging_mode=merging_mode,
                avg_count=avg_count,
            )
            nh, nl = route_and_persist(
                img_path,
                cids,
                scores,
                boxes,
                class_names,
                ow,
                oh,
                high_conf_min,
                object_store,
                review_queue,
            )
            rows.append(
                {
                    "image_path": str(img_path),
                    "n_high_saved": nh,
                    "n_low_queued": nl,
                    "pred_count": int(len(scores)),
                    "status": "ok",
                }
            )
        except Exception as ex:  # noqa: BLE001
            rows.append(
                {
                    "image_path": str(img_path),
                    "n_high_saved": 0,
                    "n_low_queued": 0,
                    "pred_count": 0,
                    "status": f"error:{ex}",
                }
            )

    return pd.DataFrame(rows)


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="OWL image inference + Chroma query merge.")
    p.add_argument("--image", type=str, default=None, help="Single image path")
    p.add_argument("--image-dir", type=str, default=None, help="Directory of images (no labels required)")
    p.add_argument(
        "--object-store",
        default=os.environ.get("OBJECT_STORE_ROOT", "data/object_store"),
    )
    p.add_argument(
        "--chroma-path",
        default=os.environ.get("CHROMA_PERSIST_DIR", "data/chroma_db"),
    )
    p.add_argument("--collection", default="owl_gt_embeddings")
    p.add_argument("--merging-mode", default="average", choices=["average", "median", "fine-grained", "knn_median"])
    p.add_argument("--avg-count", type=int, default=8)
    p.add_argument("--conf-thresh", type=float, default=0.2, help="Min objectness to keep detection")
    p.add_argument(
        "--high-conf-min",
        type=float,
        default=0.35,
        help="Objectness above this is auto-saved to object store; below goes to review queue",
    )
    p.add_argument(
        "--review-queue",
        default=os.environ.get("REVIEW_QUEUE_DIR", "data/review_queue"),
    )
    args = p.parse_args(argv)

    if not args.image and not args.image_dir:
        p.error("Provide --image or --image-dir")

    ipaths: List[str] = [args.image] if args.image else []
    manifest = build_inference_manifest_df(args.image_dir, ipaths if ipaths else None)
    if args.image and manifest.empty:
        manifest = pd.DataFrame({"image_path": [str(Path(args.image).resolve())]})

    embed_store = ChromaEmbeddingStore(args.chroma_path, collection_name=args.collection)

    print("Loading OWL-ViT detector…")
    detector = ImageConditionedObjectDetector()

    object_store = ObjectStore(args.object_store)
    review_queue = ReviewQueue(args.review_queue)

    batch_df = run_inference_batch_df(
        manifest,
        embed_store,
        detector,
        object_store,
        review_queue,
        conf_thresh=args.conf_thresh,
        merging_mode=args.merging_mode,
        avg_count=args.avg_count,
        high_conf_min=args.high_conf_min,
    )

    total_high = int(batch_df["n_high_saved"].sum()) if not batch_df.empty else 0
    total_low = int(batch_df["n_low_queued"].sum()) if not batch_df.empty else 0
    for _, row in batch_df.iterrows():
        if row["status"] == "missing_file":
            print(f"Skip missing: {row['image_path']}")
        elif str(row["status"]).startswith("error"):
            print(f"Error {row['image_path']}: {row['status']}")
        else:
            print(
                f"{Path(row['image_path']).name}: saved={row['n_high_saved']} "
                f"queued={row['n_low_queued']} (detections after conf_thresh={row['pred_count']})"
            )

    print(f"Done. total_saved={total_high} total_queued={total_low}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
