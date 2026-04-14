"""Shared source assets (object store scan)."""

import time

import dagster as dg
import pandas as pd

from dagster_defs.resources import ObjectStoreResource
from improved_pipelines.object_store import labeled_pairs_dataframe


class LabeledPairsScanConfig(dg.Config):
    """``limit``: max pairs to scan; ``-1`` means no limit."""

    limit: int = -1


@dg.asset(
    io_manager_key="pandas_parquet_io",
    group_name="stage1_seed_embedding_db",
    description="Labeled image/label pairs from object storage (input to preload and validate gold).",
)
def labeled_image_pairs(
    object_store: ObjectStoreResource,
    config: LabeledPairsScanConfig,
) -> dg.MaterializeResult:
    t0 = time.perf_counter()
    store = object_store.get_object_store()
    lim = None if config.limit < 0 else config.limit
    df = labeled_pairs_dataframe(store, limit=lim)
    elapsed = time.perf_counter() - t0
    return dg.MaterializeResult(
        metadata={
            "row_count": dg.MetadataValue.int(len(df)),
            "elapsed_s": dg.MetadataValue.float(round(elapsed, 4)),
            "object_store_root": dg.MetadataValue.text(object_store.root),
        },
        value=df,
    )
