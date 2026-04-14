"""Stage 1: seed Chroma from ground-truth embeddings."""

import json
import time
from pathlib import Path

import dagster as dg
import pandas as pd

from dagster_defs.resources import (
    ChromaStoreResource,
    ObjectStoreResource,
    RepoRootResource,
)
class PreloadRunConfig(dg.Config):
    reset_collection: bool = False
    iou_match: float = 0.2
    objectness_min: float = 0.0
    dedupe_iou: float = 0.8
    summary_json_relative_path: str = "data/preload_run_summary.json"


@dg.asset(
    io_manager_key="pandas_parquet_io",
    group_name="stage1_seed_embedding_db",
    description="Compute OWL class-head embeddings for GT boxes and write to Chroma.",
)
def preload_embeddings_summary(
    labeled_image_pairs: pd.DataFrame,
    object_store: ObjectStoreResource,
    chroma: ChromaStoreResource,
    config: PreloadRunConfig,
) -> dg.MaterializeResult:
    from improved_pipelines.preload_embeddings import run_preload_from_manifest

    t0 = time.perf_counter()
    embed_store = chroma.get_embedding_store()
    df = run_preload_from_manifest(
        labeled_image_pairs,
        object_store.root,
        chroma.persist_directory,
        chroma.collection_name,
        config.reset_collection,
        embed_store,
        detector=None,
        iou_match=config.iou_match,
        objectness_min=config.objectness_min,
        dedupe_iou=config.dedupe_iou,
    )
    elapsed = time.perf_counter() - t0
    row = df.iloc[0]
    meta = {
        "elapsed_s": dg.MetadataValue.float(round(elapsed, 4)),
        "chroma_collection_count": dg.MetadataValue.int(int(row["chroma_collection_count"])),
        "images_processed": dg.MetadataValue.int(int(row["images_processed"])),
        "embeddings_written": dg.MetadataValue.int(int(row["embeddings_written"])),
        "images_skipped": dg.MetadataValue.int(int(row["images_skipped"])),
        "error_count": dg.MetadataValue.int(int(row["error_count"])),
        "git_rev": dg.MetadataValue.text(str(row["git_rev"])),
    }
    for key in (
        "manifest_images",
        "skipped_unchanged",
        "queued_new",
        "queued_changed",
        "queued_legacy_or_inconsistent",
    ):
        if key in row.index:
            meta[key] = dg.MetadataValue.int(int(row[key]))
    return dg.MaterializeResult(metadata=meta, value=df)


@dg.asset(
    io_manager_key="pandas_parquet_io",
    group_name="stage1_seed_embedding_db",
    description="Final JSON export of preload summary (DVC / humans).",
)
def preload_summary_json(
    preload_embeddings_summary: pd.DataFrame,
    repo: RepoRootResource,
    config: PreloadRunConfig,
) -> dg.MaterializeResult:
    from improved_pipelines.preload_embeddings import summary_dataframe_to_json_dict

    t0 = time.perf_counter()
    path = Path(repo.path) / config.summary_json_relative_path
    payload = summary_dataframe_to_json_dict(preload_embeddings_summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    elapsed = time.perf_counter() - t0
    out = pd.DataFrame(
        [
            {
                "json_path": str(path.resolve()),
                "bytes_written": path.stat().st_size,
            }
        ]
    )
    return dg.MaterializeResult(
        metadata={
            "json_path": dg.MetadataValue.path(str(path.resolve())),
            "elapsed_s": dg.MetadataValue.float(round(elapsed, 4)),
        },
        value=out,
    )
