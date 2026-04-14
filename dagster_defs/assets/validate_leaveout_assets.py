"""Stage 2b: KNN leave-out validation on embeddings only (no model inference)."""

import json
import time
from pathlib import Path

import dagster as dg
import pandas as pd

from dagster_defs.resources import ChromaStoreResource, MlflowResource, RepoRootResource


class LeaveoutValidationConfig(dg.Config):
    n_folds: int = 5
    seed: int = 42
    k: int = 5
    report_relative_path: str = "data/validation_leaveout_report.json"


@dg.asset(
    io_manager_key="pandas_parquet_io",
    group_name="stage2_validate_leaveout",
    description="Split embeddings by image_id into held-out and remaining sets.",
)
def leaveout_embedding_split(
    preload_embeddings_summary: pd.DataFrame,
    chroma: ChromaStoreResource,
    config: LeaveoutValidationConfig,
) -> dg.MaterializeResult:
    from improved_pipelines.validate_leaveout import embeddings_dataframe, kfold_split_by_image_id

    _ = preload_embeddings_summary
    t0 = time.perf_counter()
    embed_store = chroma.get_embedding_store()
    df = embeddings_dataframe(embed_store)
    folds_df = kfold_split_by_image_id(df, n_folds=int(config.n_folds), seed=int(config.seed))
    total_image_ids = int(df["image_id"].astype(str).nunique()) if not df.empty else 0
    fold_sizes = (
        folds_df.groupby("fold")["image_id"].nunique().sort_index().to_dict() if not folds_df.empty else {}
    )
    out = pd.DataFrame(
        [
            {
                "n_folds": int(config.n_folds),
                "seed": int(config.seed),
                "k": int(config.k),
                "total_embeddings": int(len(df)),
                "total_image_ids": total_image_ids,
                "fold_sizes_json": json.dumps({str(int(k)): int(v) for k, v in fold_sizes.items()}),
            }
        ]
    )
    elapsed = time.perf_counter() - t0
    row = out.iloc[0]
    return dg.MaterializeResult(
        metadata={
            "total_embeddings": dg.MetadataValue.int(int(row["total_embeddings"])),
            "total_image_ids": dg.MetadataValue.int(int(row["total_image_ids"])),
            "n_folds": dg.MetadataValue.int(int(row["n_folds"])),
            "elapsed_s": dg.MetadataValue.float(round(elapsed, 4)),
        },
        value=out,
    )


@dg.asset(
    io_manager_key="pandas_parquet_io",
    group_name="stage2_validate_leaveout",
    description="KNN predictions for held-out embeddings against remaining embedding pool.",
)
def leaveout_knn_predictions(
    leaveout_embedding_split: pd.DataFrame,
    chroma: ChromaStoreResource,
) -> dg.MaterializeResult:
    from improved_pipelines.validate_leaveout import (
        embeddings_dataframe,
        kfold_cross_validate_df,
    )

    t0 = time.perf_counter()
    embed_store = chroma.get_embedding_store()
    all_embeddings = embeddings_dataframe(embed_store)
    split = leaveout_embedding_split.iloc[0]
    preds = kfold_cross_validate_df(
        all_embeddings,
        n_folds=int(split["n_folds"]),
        k=int(split["k"]),
        seed=int(split["seed"]),
    )
    elapsed = time.perf_counter() - t0
    correct = int(preds["correct"].astype(bool).sum()) if not preds.empty else 0
    return dg.MaterializeResult(
        metadata={
            "row_count": dg.MetadataValue.int(len(preds)),
            "correct": dg.MetadataValue.int(correct),
            "elapsed_s": dg.MetadataValue.float(round(elapsed, 4)),
        },
        value=preds,
    )


@dg.asset(
    io_manager_key="pandas_parquet_io",
    group_name="stage2_validate_leaveout",
    description="Aggregate leave-out KNN metrics (overall and per class).",
)
def leaveout_metrics(leaveout_knn_predictions: pd.DataFrame) -> dg.MaterializeResult:
    from improved_pipelines.validate_leaveout import aggregate_leaveout_metrics_df

    t0 = time.perf_counter()
    metrics = aggregate_leaveout_metrics_df(leaveout_knn_predictions)
    elapsed = time.perf_counter() - t0
    overall = metrics[metrics["scope"] == "overall"].iloc[0]
    return dg.MaterializeResult(
        metadata={
            "accuracy": dg.MetadataValue.float(float(overall["accuracy"])),
            "f1": dg.MetadataValue.float(float(overall["f1"])),
            "support": dg.MetadataValue.int(int(overall["support"])),
            "elapsed_s": dg.MetadataValue.float(round(elapsed, 4)),
        },
        value=metrics,
    )


@dg.asset(
    io_manager_key="pandas_parquet_io",
    group_name="stage2_validate_leaveout",
    description="Write KNN leave-out JSON report.",
)
def leaveout_report_json(
    leaveout_embedding_split: pd.DataFrame,
    leaveout_knn_predictions: pd.DataFrame,
    leaveout_metrics: pd.DataFrame,
    repo: RepoRootResource,
    config: LeaveoutValidationConfig,
) -> dg.MaterializeResult:
    from improved_pipelines.validate_leaveout import (
        build_leaveout_report_dict,
        write_leaveout_report_json,
    )

    t0 = time.perf_counter()
    split = leaveout_embedding_split.iloc[0]
    warning = None
    if int(split["total_embeddings"]) == 0:
        warning = "no_embeddings_in_chroma_collection"
    elif leaveout_knn_predictions.empty:
        warning = "no_predictions_generated"

    report = build_leaveout_report_dict(
        leaveout_metrics,
        leaveout_knn_predictions,
        n_folds=int(split["n_folds"]),
        seed=int(split["seed"]),
        k=int(split["k"]),
        total_image_ids_count=int(split["total_image_ids"]),
        warning=warning,
    )
    path = Path(repo.path) / config.report_relative_path
    write_leaveout_report_json(path, report)
    out = pd.DataFrame([{"json_path": str(path.resolve()), "bytes_written": path.stat().st_size}])
    elapsed = time.perf_counter() - t0
    return dg.MaterializeResult(
        metadata={
            "json_path": dg.MetadataValue.path(str(path.resolve())),
            "accuracy": dg.MetadataValue.float(float(report["accuracy"])),
            "f1": dg.MetadataValue.float(float(report["f1"])),
            "elapsed_s": dg.MetadataValue.float(round(elapsed, 4)),
        },
        value=out,
    )


@dg.asset(
    io_manager_key="pandas_parquet_io",
    group_name="stage2_validate_leaveout",
    description="Log leave-out KNN report and metrics to MLflow when enabled.",
)
def leaveout_mlflow_record(
    leaveout_report_json: pd.DataFrame,
    leaveout_metrics: pd.DataFrame,
    chroma: ChromaStoreResource,
    mlflow_resource: MlflowResource,
) -> dg.MaterializeResult:
    from improved_pipelines.validate_leaveout import log_leaveout_mlflow

    t0 = time.perf_counter()
    path = Path(leaveout_report_json.iloc[0]["json_path"])
    report = json.loads(path.read_text(encoding="utf-8"))
    uri = mlflow_resource.effective_tracking_uri()
    logged = False
    if uri:
        import os

        prev = os.environ.get("MLFLOW_TRACKING_URI")
        os.environ["MLFLOW_TRACKING_URI"] = uri
        try:
            log_leaveout_mlflow(
                report,
                chroma_path=chroma.persist_directory,
                collection_name=chroma.collection_name,
                report_path=path,
            )
            logged = True
        finally:
            if prev is None:
                os.environ.pop("MLFLOW_TRACKING_URI", None)
            else:
                os.environ["MLFLOW_TRACKING_URI"] = prev

    overall = leaveout_metrics[leaveout_metrics["scope"] == "overall"].iloc[0]
    out = pd.DataFrame(
        [
            {
                "mlflow_logged": logged,
                "tracking_uri_used": uri or "",
                "accuracy": float(overall["accuracy"]),
            }
        ]
    )
    elapsed = time.perf_counter() - t0
    return dg.MaterializeResult(
        metadata={
            "mlflow_logged": dg.MetadataValue.bool(logged),
            "accuracy": dg.MetadataValue.float(float(overall["accuracy"])),
            "elapsed_s": dg.MetadataValue.float(round(elapsed, 4)),
        },
        value=out,
    )
