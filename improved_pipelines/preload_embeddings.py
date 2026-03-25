"""
Preload OWL-ViT class-head embeddings for GT boxes into Chroma.

Run from repo root: ``python -m improved_pipelines.preload_embeddings``
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from detection.OWL_VIT_v2.image_conditioned import ImageConditionedObjectDetector
from detection.OWL_VIT_v2.utils import read_yolo_label_file

from improved_pipelines.embedding_store import ChromaEmbeddingStore
from improved_pipelines.matching import match_tokens_to_gt
from improved_pipelines.object_store import ObjectStore


def _git_rev() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[1],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except Exception:
        return "unknown"


def _maybe_mlflow_start() -> Any:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        return None
    import mlflow

    mlflow.set_tracking_uri(uri)
    return mlflow.start_run(run_name=os.environ.get("MLFLOW_RUN_NAME", "preload_embeddings"))


def _maybe_mlflow_end(run, metrics: Dict[str, Any], summary_path: Optional[Path]) -> None:
    if run is None:
        return
    import mlflow

    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(k, float(v))
    if summary_path and summary_path.is_file():
        mlflow.log_artifact(str(summary_path))
    mlflow.end_run()


def run_preload(
    object_store_root: str,
    chroma_path: str,
    collection_name: str,
    reset_collection: bool,
    limit: Optional[int],
    summary_out: Path,
    iou_match: float,
    objectness_min: float,
    dedupe_iou: float,
) -> Dict[str, Any]:
    t0 = time.time()
    store = ObjectStore(object_store_root)
    class_names = store.load_class_names()

    checkpoint_hint = os.environ.get("OWL_CHECKPOINT_PATH", "")

    mlflow_run = _maybe_mlflow_start()
    if mlflow_run is not None:
        import mlflow

        mlflow.log_params(
            {
                "object_store_root": object_store_root,
                "chroma_path": chroma_path,
                "collection_name": collection_name,
                "iou_match": iou_match,
                "objectness_min": objectness_min,
                "dedupe_iou": dedupe_iou,
                "owl_checkpoint_path": checkpoint_hint or "(default)",
                "git_rev": _git_rev(),
            }
        )

    embed_store = ChromaEmbeddingStore(chroma_path, collection_name=collection_name)
    if reset_collection:
        embed_store.reset()

    images_processed = 0
    embeddings_written = 0
    images_skipped = 0
    errors: List[str] = []

    pairs = list(store.iter_labeled_images())
    if limit is not None:
        pairs = pairs[:limit]

    if pairs:
        print("Loading OWL-ViT detector…")
        detector = ImageConditionedObjectDetector()
        for abs_img, abs_lbl, image_id in pairs:
            abs_img = Path(abs_img)
            abs_lbl = Path(abs_lbl)
            try:
                gt_data = read_yolo_label_file(str(abs_lbl))
                if not gt_data:
                    images_skipped += 1
                    continue

                with Image.open(abs_img) as im:
                    orig_w, orig_h = im.size

                boxes, objectnesses, class_embeddings = detector.process(
                    str(abs_img.resolve())
                )
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

                ids: List[str] = []
                vecs: List[List[float]] = []
                metas: List[Dict[str, Any]] = []

                for gt_line_idx, (tok_idx, gt_row, objn) in enumerate(matched):
                    cid = int(gt_row[4])
                    if cid < 0 or cid >= len(class_names):
                        errors.append(f"{image_id}: class_id {cid} out of range")
                        continue
                    cname = class_names[cid]
                    emb = class_embeddings[tok_idx]
                    vec = np.asarray(emb, dtype=np.float64).reshape(-1).tolist()
                    eid = f"{image_id}:tok{tok_idx}:gtline{gt_line_idx}"
                    ids.append(eid)
                    vecs.append(vec)
                    metas.append(
                        {
                            "image_id": image_id,
                            "class_id": cid,
                            "class_name": cname,
                            "patch_index": int(tok_idx),
                            "objectness": float(objn),
                            "gt_box_json": json.dumps([float(x) for x in gt_row[:4]]),
                            "source_image": str(abs_img.resolve()),
                        }
                    )

                if ids:
                    embed_store.add_embeddings(ids, vecs, metas)
                    embeddings_written += len(ids)
                images_processed += 1
            except Exception as ex:  # noqa: BLE001
                errors.append(f"{image_id}: {ex}")
                images_skipped += 1

    elapsed = time.time() - t0
    summary = {
        "git_rev": _git_rev(),
        "object_store_root": str(Path(object_store_root).resolve()),
        "chroma_path": str(Path(chroma_path).resolve()),
        "collection_name": collection_name,
        "chroma_collection_count": embed_store.count(),
        "images_processed": images_processed,
        "embeddings_written": embeddings_written,
        "images_skipped": images_skipped,
        "elapsed_s": round(elapsed, 3),
        "errors": errors[:50],
        "error_count": len(errors),
    }

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    _maybe_mlflow_end(
        mlflow_run,
        {
            "images_processed": images_processed,
            "embeddings_written": embeddings_written,
            "images_skipped": images_skipped,
            "elapsed_s": elapsed,
            "chroma_collection_count": embed_store.count(),
            "error_count": len(errors),
        },
        summary_out,
    )

    return summary


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Preload GT-aligned OWL embeddings into Chroma.")
    p.add_argument(
        "--object-store",
        default=os.environ.get("OBJECT_STORE_ROOT", "data/object_store"),
        help="Root with images/, labels/, classes.txt",
    )
    p.add_argument(
        "--chroma-path",
        default=os.environ.get("CHROMA_PERSIST_DIR", "data/chroma_db"),
        help="Chroma persistence directory",
    )
    p.add_argument("--collection", default="owl_gt_embeddings", help="Chroma collection name")
    p.add_argument("--reset-collection", action="store_true", help="Delete and recreate collection")
    p.add_argument("--limit", type=int, default=None, help="Max number of image/label pairs")
    p.add_argument(
        "--summary-out",
        default=os.environ.get("PRELOAD_SUMMARY_OUT", "data/preload_run_summary.json"),
        help="JSON summary path (for DVC / MLflow)",
    )
    p.add_argument("--iou-match", type=float, default=0.7)
    p.add_argument("--objectness-min", type=float, default=0.01)
    p.add_argument("--dedupe-iou", type=float, default=0.8)
    p.add_argument(
        "--import-yolo",
        nargs=3,
        metavar=("IMAGES_DIR", "LABELS_DIR", "CLASSES_TXT"),
        help="Copy YOLO dataset into --object-store then exit",
    )
    args = p.parse_args(argv)

    if args.import_yolo:
        ObjectStore.import_yolo_dataset(
            args.object_store,
            args.import_yolo[0],
            args.import_yolo[1],
            args.import_yolo[2],
        )
        print(f"Imported YOLO data into {args.object_store}")
        return 0

    summary_path = Path(args.summary_out)
    run_preload(
        object_store_root=args.object_store,
        chroma_path=args.chroma_path,
        collection_name=args.collection,
        reset_collection=args.reset_collection,
        limit=args.limit,
        summary_out=summary_path,
        iou_match=args.iou_match,
        objectness_min=args.objectness_min,
        dedupe_iou=args.dedupe_iou,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
