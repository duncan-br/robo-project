"""Stage 2: gold QC — inference vs labels, metrics, MLflow."""

import json
import time
from pathlib import Path

import dagster as dg
import pandas as pd

from dagster_defs.resources import ChromaStoreResource, MlflowResource, ObjectStoreResource, RepoRootResource
from dagster_defs.resources import OwlDetectorResource


class GoldValidationConfig(dg.Config):
    gold_fraction: float = 0.2
    max_images: int = 0
    seed: int = 42
    merging_mode: str = "average"
    avg_count: int = 8
    conf_thresh: float = 0.2
    iou_match: float = 0.5
    report_relative_path: str = "data/validation_report.json"


@dg.asset(
    io_manager_key="pandas_parquet_io",
    group_name="stage2_validate_gold_qc",
    description="Random sample of labeled pairs for gold evaluation.",
)
def gold_validation_sample(
    labeled_image_pairs: pd.DataFrame,
    config: GoldValidationConfig,
) -> dg.MaterializeResult:
    from improved_pipelines.validate_gold import sample_gold_pairs_df

    t0 = time.perf_counter()
    cap = config.max_images if config.max_images > 0 else None
    df = sample_gold_pairs_df(
        labeled_image_pairs,
        config.gold_fraction,
        config.seed,
        cap,
    )
    elapsed = time.perf_counter() - t0
    return dg.MaterializeResult(
        metadata={
            "row_count": dg.MetadataValue.int(len(df)),
            "gold_fraction": dg.MetadataValue.float(config.gold_fraction),
            "elapsed_s": dg.MetadataValue.float(round(elapsed, 4)),
        },
        value=df,
    )


@dg.asset(
    io_manager_key="pandas_parquet_io",
    group_name="stage2_validate_gold_qc",
    description="Per-image TP/FP/FN on gold sample using Chroma query merge + OWL.",
)
def gold_validation_per_image(
    gold_validation_sample: pd.DataFrame,
    preload_embeddings_summary: pd.DataFrame,
    chroma: ChromaStoreResource,
    owl_detector: OwlDetectorResource,
    config: GoldValidationConfig,
) -> dg.MaterializeResult:
    from improved_pipelines.validate_gold import evaluate_gold_per_image_df

    _ = preload_embeddings_summary
    t0 = time.perf_counter()
    embed_store = chroma.get_embedding_store()
    detector = owl_detector.get_detector()
    df = evaluate_gold_per_image_df(
        gold_validation_sample,
        embed_store,
        detector,
        conf_thresh=config.conf_thresh,
        merging_mode=config.merging_mode,
        avg_count=config.avg_count,
        iou_match=config.iou_match,
    )
    elapsed = time.perf_counter() - t0
    return dg.MaterializeResult(
        metadata={
            "row_count": dg.MetadataValue.int(len(df)),
            "elapsed_s": dg.MetadataValue.float(round(elapsed, 4)),
        },
        value=df,
    )


@dg.asset(
    io_manager_key="pandas_parquet_io",
    group_name="stage2_validate_gold_qc",
    description="Aggregate precision, recall, F1 over evaluated images.",
)
def gold_validation_metrics(gold_validation_per_image: pd.DataFrame) -> dg.MaterializeResult:
    from improved_pipelines.validate_gold import aggregate_gold_metrics_df

    t0 = time.perf_counter()
    df = aggregate_gold_metrics_df(gold_validation_per_image)
    elapsed = time.perf_counter() - t0
    row = df.iloc[0]
    return dg.MaterializeResult(
        metadata={
            "precision": dg.MetadataValue.float(float(row["precision"])),
            "recall": dg.MetadataValue.float(float(row["recall"])),
            "f1": dg.MetadataValue.float(float(row["f1"])),
            "tp": dg.MetadataValue.int(int(row["tp"])),
            "fp": dg.MetadataValue.int(int(row["fp"])),
            "fn": dg.MetadataValue.int(int(row["fn"])),
            "images_evaluated": dg.MetadataValue.int(int(row["images_evaluated"])),
            "elapsed_s": dg.MetadataValue.float(round(elapsed, 4)),
        },
        value=df,
    )


@dg.asset(
    io_manager_key="pandas_parquet_io",
    group_name="stage2_validate_gold_qc",
    description="Write validation_report.json (final export).",
)
def validation_report_json(
    gold_validation_metrics: pd.DataFrame,
    gold_validation_per_image: pd.DataFrame,
    gold_validation_sample: pd.DataFrame,
    repo: RepoRootResource,
    config: GoldValidationConfig,
) -> dg.MaterializeResult:
    from improved_pipelines.validate_gold import build_validation_report_dict, write_validation_report_json

    t0 = time.perf_counter()
    warning = None
    if gold_validation_sample.empty and gold_validation_per_image.empty:
        warning = "no_labeled_pairs_in_object_store"
    cap = config.max_images if config.max_images > 0 else None
    report = build_validation_report_dict(
        gold_validation_per_image,
        gold_validation_metrics,
        gold_fraction=config.gold_fraction,
        max_images=cap,
        seed=config.seed,
        sample_size=len(gold_validation_sample),
        warning=warning,
    )
    path = Path(repo.path) / config.report_relative_path
    write_validation_report_json(path, report)
    elapsed = time.perf_counter() - t0
    out = pd.DataFrame([{"json_path": str(path.resolve()), "bytes_written": path.stat().st_size}])
    return dg.MaterializeResult(
        metadata={
            "json_path": dg.MetadataValue.path(str(path.resolve())),
            "precision": dg.MetadataValue.float(float(report["precision"])),
            "recall": dg.MetadataValue.float(float(report["recall"])),
            "f1": dg.MetadataValue.float(float(report["f1"])),
            "elapsed_s": dg.MetadataValue.float(round(elapsed, 4)),
        },
        value=out,
    )


@dg.asset(
    io_manager_key="pandas_parquet_io",
    group_name="stage2_validate_gold_qc",
    description="Log validation metrics and report artifact to MLflow when enabled.",
)
def validation_mlflow_record(
    validation_report_json: pd.DataFrame,
    gold_validation_metrics: pd.DataFrame,
    object_store: ObjectStoreResource,
    chroma: ChromaStoreResource,
    mlflow_resource: MlflowResource,
    config: GoldValidationConfig,
) -> dg.MaterializeResult:
    from improved_pipelines.validate_gold import log_validation_mlflow

    t0 = time.perf_counter()
    path = Path(validation_report_json.iloc[0]["json_path"])
    report = json.loads(path.read_text(encoding="utf-8"))
    uri = mlflow_resource.effective_tracking_uri()
    logged = False
    if uri:
        import os

        prev = os.environ.get("MLFLOW_TRACKING_URI")
        os.environ["MLFLOW_TRACKING_URI"] = uri
        try:
            cap = config.max_images if config.max_images > 0 else None
            log_validation_mlflow(
                report,
                object_store=object_store.root,
                chroma_path=chroma.persist_directory,
                merging_mode=config.merging_mode,
                conf_thresh=config.conf_thresh,
                iou_match=config.iou_match,
                report_path=path,
                max_images=cap,
            )
            logged = True
        finally:
            if prev is None:
                os.environ.pop("MLFLOW_TRACKING_URI", None)
            else:
                os.environ["MLFLOW_TRACKING_URI"] = prev
    elapsed = time.perf_counter() - t0
    row = gold_validation_metrics.iloc[0]
    out = pd.DataFrame(
        [
            {
                "mlflow_logged": logged,
                "tracking_uri_used": uri or "",
                "f1": float(row["f1"]),
            }
        ]
    )
    return dg.MaterializeResult(
        metadata={
            "mlflow_logged": dg.MetadataValue.bool(logged),
            "f1": dg.MetadataValue.float(float(row["f1"])),
            "elapsed_s": dg.MetadataValue.float(round(elapsed, 4)),
        },
        value=out,
    )
