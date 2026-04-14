"""Preload OWL-ViT class-head embeddings for GT boxes into Chroma."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s %(name)s - %(message)s"))
    log.addHandler(_handler)
    log.setLevel(logging.DEBUG)

import numpy as np
import pandas as pd
from PIL import Image

from detection.OWL_VIT_v2.image_conditioned import ImageConditionedObjectDetector
from detection.OWL_VIT_v2.utils import read_yolo_label_file

from improved_pipelines.embedding_store import ChromaEmbeddingStore
from improved_pipelines.matching import match_tokens_to_gt
from improved_pipelines.object_store import ObjectStore


def _preload_source_fingerprint(image_path: Path, label_path: Path) -> str:
    """Hash image + label file bytes so edits invalidate cached embeddings."""
    h = hashlib.sha256()
    h.update(b"img\x00")
    h.update(Path(image_path).read_bytes())
    h.update(b"lbl\x00")
    h.update(Path(label_path).read_bytes())
    return h.hexdigest()


def _select_preload_worklist(
    embed_store: ChromaEmbeddingStore,
    pairs: Sequence[Tuple[Path, Path, str]],
    reset_collection: bool,
) -> Tuple[List[Tuple[Path, Path, str, str]], Dict[str, int]]:
    """Diff manifest against Chroma to find which pairs actually need reprocessing."""
    stats = {
        "manifest_images": len(pairs),
        "skipped_unchanged": 0,
        "queued_new": 0,
        "queued_changed": 0,
        "queued_legacy_or_inconsistent": 0,
    }
    log.info(
        "Worklist selection: %d manifest pairs, reset_collection=%s",
        len(pairs), reset_collection,
    )

    if reset_collection:
        log.info("reset_collection=True → all %d pairs queued for reprocessing", len(pairs))
        out: List[Tuple[Path, Path, str, str]] = []
        for abs_img, abs_lbl, iid in pairs:
            abs_img = Path(abs_img)
            abs_lbl = Path(abs_lbl)
            iid = str(iid)
            fp = _preload_source_fingerprint(abs_img, abs_lbl)
            out.append((abs_img, abs_lbl, iid, fp))
        return out, stats

    seen, fp_by_id, bad_ids = embed_store.get_preload_incremental_index()
    log.info(
        "Chroma incremental index: %d image_ids seen, %d with fingerprint, %d flagged bad/legacy",
        len(seen), len(fp_by_id), len(bad_ids),
    )
    out = []
    for abs_img, abs_lbl, iid in pairs:
        abs_img = Path(abs_img)
        abs_lbl = Path(abs_lbl)
        iid = str(iid)
        cur_fp = _preload_source_fingerprint(abs_img, abs_lbl)
        if iid not in seen:
            log.debug("  %s → NEW (not in Chroma)", iid)
            stats["queued_new"] += 1
            out.append((abs_img, abs_lbl, iid, cur_fp))
            continue
        if iid in bad_ids:
            log.debug("  %s → LEGACY/INCONSISTENT (re-processing)", iid)
            stats["queued_legacy_or_inconsistent"] += 1
            embed_store.delete_embeddings_for_image_id(iid)
            out.append((abs_img, abs_lbl, iid, cur_fp))
            continue
        prev = fp_by_id.get(iid)
        if prev == cur_fp:
            log.debug("  %s → SKIP (fingerprint unchanged)", iid)
            stats["skipped_unchanged"] += 1
            continue
        log.debug("  %s → CHANGED (fp %s… → %s…)", iid, (prev or "")[:12], cur_fp[:12])
        stats["queued_changed"] += 1
        embed_store.delete_embeddings_for_image_id(iid)
        out.append((abs_img, abs_lbl, iid, cur_fp))

    log.info(
        "Worklist result: %d to process (new=%d, changed=%d, legacy=%d), %d skipped unchanged",
        len(out), stats["queued_new"], stats["queued_changed"],
        stats["queued_legacy_or_inconsistent"], stats["skipped_unchanged"],
    )
    return out, stats


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


def _run_preload_loop(
    pairs: Sequence[Tuple[Path, Path, str, str]],
    class_names: List[str],
    embed_store: ChromaEmbeddingStore,
    detector: ImageConditionedObjectDetector,
    iou_match: float,
    objectness_min: float,
    dedupe_iou: float,
) -> Tuple[int, int, int, List[str]]:
    images_processed = 0
    embeddings_written = 0
    images_skipped = 0
    errors: List[str] = []
    total = len(pairs)

    for i, (abs_img, abs_lbl, image_id, source_fp) in enumerate(pairs):
        abs_img = Path(abs_img)
        abs_lbl = Path(abs_lbl)
        try:
            gt_data = read_yolo_label_file(str(abs_lbl))
            if not gt_data:
                log.debug("[%d/%d] %s → skipped (empty label file)", i + 1, total, image_id)
                images_skipped += 1
                continue

            with Image.open(abs_img) as im:
                orig_w, orig_h = im.size

            boxes, objectnesses, class_embeddings = detector.process(str(abs_img.resolve()))
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
                        "preload_source_fingerprint": source_fp,
                    }
                )

            if ids:
                embed_store.add_embeddings(ids, vecs, metas)
                embeddings_written += len(ids)
            images_processed += 1
            log.info(
                "[%d/%d] %s → %d embeddings written (%d matched tokens)",
                i + 1, total, image_id, len(ids), len(matched),
            )
        except Exception as ex:  # noqa: BLE001
            log.warning("[%d/%d] %s → ERROR: %s", i + 1, total, image_id, ex)
            errors.append(f"{image_id}: {ex}")
            images_skipped += 1

    return images_processed, embeddings_written, images_skipped, errors


def run_preload_from_manifest(
    manifest_df: pd.DataFrame,
    object_store_root: str,
    chroma_path: str,
    collection_name: str,
    reset_collection: bool,
    embed_store: ChromaEmbeddingStore,
    detector: Optional[ImageConditionedObjectDetector] = None,
    iou_match: float = 0.7,
    objectness_min: float = 0.01,
    dedupe_iou: float = 0.8,
) -> pd.DataFrame:
    """Seed Chroma from a manifest DataFrame; returns a one-row summary.

    Lazily loads the detector only when the incremental worklist is non-empty.
    """
    t0 = time.time()
    store = ObjectStore(object_store_root)
    class_names = store.load_class_names()
    log.info(
        "run_preload_from_manifest: chroma=%s collection=%s reset=%s manifest_rows=%d classes=%d",
        chroma_path, collection_name, reset_collection, len(manifest_df), len(class_names),
    )
    log.info("Chroma collection count BEFORE worklist: %d", embed_store.count())

    if reset_collection:
        log.info("Resetting Chroma collection (reset_collection=True)")
        embed_store.reset()

    pair_rows: List[Tuple[Path, Path, str]] = []
    for row in manifest_df.itertuples(index=False):
        pair_rows.append((Path(row.image_path), Path(row.label_path), str(row.image_id)))

    worklist, inc_stats = _select_preload_worklist(embed_store, pair_rows, reset_collection)

    if worklist:
        log.info("Worklist has %d items → loading OWL-ViT detector", len(worklist))
        det = detector if detector is not None else ImageConditionedObjectDetector()
        images_processed, embeddings_written, images_skipped, errors = _run_preload_loop(
            worklist,
            class_names,
            embed_store,
            det,
            iou_match,
            objectness_min,
            dedupe_iou,
        )
        log.info(
            "Preload loop done: processed=%d written=%d skipped=%d errors=%d",
            images_processed, embeddings_written, images_skipped, len(errors),
        )
    else:
        log.info("Worklist is EMPTY → skipping detector load and embedding writes (all up to date)")
        images_processed = embeddings_written = images_skipped = 0
        errors = []

    elapsed = time.time() - t0
    count = embed_store.count()
    log.info("Chroma collection count AFTER: %d (elapsed=%.1fs)", count, elapsed)
    summary = {
        "git_rev": _git_rev(),
        "object_store_root": str(Path(object_store_root).resolve()),
        "chroma_path": str(Path(chroma_path).resolve()),
        "collection_name": collection_name,
        "chroma_collection_count": count,
        "images_processed": images_processed,
        "embeddings_written": embeddings_written,
        "images_skipped": images_skipped,
        "elapsed_s": round(elapsed, 3),
        "error_count": len(errors),
        "errors_json": json.dumps(errors[:50]),
        **inc_stats,
    }
    return pd.DataFrame([summary])


def summary_dataframe_to_json_dict(df: pd.DataFrame) -> Dict[str, Any]:
    """Convert one-row preload summary DataFrame to the legacy JSON shape."""
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    err_raw = row.pop("errors_json", "[]")
    try:
        errors = json.loads(err_raw) if isinstance(err_raw, str) else []
    except json.JSONDecodeError:
        errors = []
    out = {k: row[k] for k in row}
    out["errors"] = errors
    return out


def _maybe_mlflow_start() -> Any:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        return None
    import mlflow

    mlflow.set_tracking_uri(uri)
    return mlflow.start_run(run_name=os.environ.get("MLFLOW_RUN_NAME", "preload_embeddings"))


def _maybe_mlflow_end(run: Any, metrics: Dict[str, Any], summary_path: Optional[Path]) -> None:
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

    pairs = list(store.iter_labeled_images())
    if limit is not None:
        pairs = pairs[:limit]

    worklist, inc_stats = _select_preload_worklist(embed_store, pairs, reset_collection)

    images_processed = 0
    embeddings_written = 0
    images_skipped = 0
    errors: List[str] = []

    if worklist:
        print("Loading OWL-ViT detector…")
        detector = ImageConditionedObjectDetector()
        images_processed, embeddings_written, images_skipped, errors = _run_preload_loop(
            worklist,
            class_names,
            embed_store,
            detector,
            iou_match,
            objectness_min,
            dedupe_iou,
        )

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
        **inc_stats,
    }

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    mlflow_metrics = {
        "images_processed": images_processed,
        "embeddings_written": embeddings_written,
        "images_skipped": images_skipped,
        "elapsed_s": elapsed,
        "chroma_collection_count": embed_store.count(),
        "error_count": len(errors),
    }
    for k, v in inc_stats.items():
        if isinstance(v, (int, float)):
            mlflow_metrics[k] = float(v)
    _maybe_mlflow_end(mlflow_run, mlflow_metrics, summary_out)

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
