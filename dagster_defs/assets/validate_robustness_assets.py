"""Stage 2d: embedding robustness validation under small image augmentations."""

import json
import time
from pathlib import Path

import dagster as dg
import pandas as pd

from dagster_defs.resources import (
    ChromaStoreResource,
    MlflowResource,
    ObjectStoreResource,
    OwlDetectorResource,
    RepoRootResource,
)


class RobustnessValidationConfig(dg.Config):
    sample_fraction: float = 0.2
    max_images: int = 10
    seed: int = 42
    iou_match: float = 0.2
    objectness_min: float = 0.0
    dedupe_iou: float = 0.8
    min_expected_cosine: float = 0.75
    report_relative_path: str = "data/validation_robustness_report.json"


@dg.asset(
    io_manager_key="pandas_parquet_io",
    group_name="stage2_validate_robustness",
    description="Sample labeled image pairs for robustness evaluation.",
)
def robustness_sample(
    labeled_image_pairs: pd.DataFrame,
    config: RobustnessValidationConfig,
) -> dg.MaterializeResult:
    from improved_pipelines.validate_robustness import sample_images_for_robustness

    t0 = time.perf_counter()
    cap = config.max_images if config.max_images > 0 else None
    df = sample_images_for_robustness(
        labeled_image_pairs,
        fraction=config.sample_fraction,
        seed=config.seed,
        max_images=cap,
    )
    elapsed = time.perf_counter() - t0
    return dg.MaterializeResult(
        metadata={
            "row_count": dg.MetadataValue.int(len(df)),
            "sample_fraction": dg.MetadataValue.float(float(config.sample_fraction)),
            "elapsed_s": dg.MetadataValue.float(round(elapsed, 4)),
        },
        value=df,
    )


@dg.asset(
    io_manager_key="pandas_parquet_io",
    group_name="stage2_validate_robustness",
    description="Extract augmented embeddings and compare cosine similarity to original Chroma embeddings.",
)
def robustness_augmented_embeddings(
    preload_embeddings_summary: pd.DataFrame,
    robustness_sample: pd.DataFrame,
    object_store: ObjectStoreResource,
    chroma: ChromaStoreResource,
    owl_detector: OwlDetectorResource,
    config: RobustnessValidationConfig,
) -> dg.MaterializeResult:
    from improved_pipelines.validate_robustness import compare_to_originals, extract_augmented_embeddings

    _ = preload_embeddings_summary
    t0 = time.perf_counter()
    store = object_store.get_object_store()
    class_names = store.load_class_names()
    detector = owl_detector.get_detector()

    aug_df = extract_augmented_embeddings(
        robustness_sample,
        class_names=class_names,
        detector=detector,
        iou_match=config.iou_match,
        objectness_min=config.objectness_min,
        dedupe_iou=config.dedupe_iou,
    )
    comparisons_df = compare_to_originals(aug_df, chroma.get_embedding_store())
    elapsed = time.perf_counter() - t0
    return dg.MaterializeResult(
        metadata={
            "augmented_embeddings": dg.MetadataValue.int(len(aug_df)),
            "comparisons": dg.MetadataValue.int(len(comparisons_df)),
            "elapsed_s": dg.MetadataValue.float(round(elapsed, 4)),
        },
        value=comparisons_df,
    )


@dg.asset(
    io_manager_key="pandas_parquet_io",
    group_name="stage2_validate_robustness",
    description="Aggregate robustness cosine metrics (overall, per class, per augmentation).",
)
def robustness_metrics(robustness_augmented_embeddings: pd.DataFrame) -> dg.MaterializeResult:
    from improved_pipelines.validate_robustness import aggregate_robustness_metrics_df

    t0 = time.perf_counter()
    df = aggregate_robustness_metrics_df(robustness_augmented_embeddings)
    elapsed = time.perf_counter() - t0
    overall = df[df["scope"] == "overall"].iloc[0]
    return dg.MaterializeResult(
        metadata={
            "overall_mean_cosine": dg.MetadataValue.float(float(overall["mean_cosine"])),
            "overall_min_cosine": dg.MetadataValue.float(float(overall["min_cosine"])),
            "count": dg.MetadataValue.int(int(overall["count"])),
            "elapsed_s": dg.MetadataValue.float(round(elapsed, 4)),
        },
        value=df,
    )


@dg.asset(
    io_manager_key="pandas_parquet_io",
    group_name="stage2_validate_robustness",
    description="Write robustness validation JSON report.",
)
def robustness_report_json(
    robustness_sample: pd.DataFrame,
    robustness_augmented_embeddings: pd.DataFrame,
    robustness_metrics: pd.DataFrame,
    repo: RepoRootResource,
    config: RobustnessValidationConfig,
) -> dg.MaterializeResult:
    from improved_pipelines.validate_robustness import (
        build_robustness_report_dict,
        write_robustness_report_json,
    )

    t0 = time.perf_counter()
    cap = config.max_images if config.max_images > 0 else None
    warning = None
    if robustness_sample.empty:
        warning = "no_sample_images"
    elif robustness_augmented_embeddings.empty:
        warning = "no_augmented_comparisons"
    report = build_robustness_report_dict(
        robustness_metrics,
        robustness_augmented_embeddings,
        sample_fraction=config.sample_fraction,
        seed=config.seed,
        max_images=cap,
        min_expected_cosine=config.min_expected_cosine,
        warning=warning,
    )
    path = Path(repo.path) / config.report_relative_path
    write_robustness_report_json(path, report)
    out = pd.DataFrame([{"json_path": str(path.resolve()), "bytes_written": path.stat().st_size}])
    elapsed = time.perf_counter() - t0
    return dg.MaterializeResult(
        metadata={
            "json_path": dg.MetadataValue.path(str(path.resolve())),
            "overall_mean_cosine": dg.MetadataValue.float(float(report["overall_mean_cosine"])),
            "passes_threshold": dg.MetadataValue.bool(bool(report["passes_threshold"])),
            "elapsed_s": dg.MetadataValue.float(round(elapsed, 4)),
        },
        value=out,
    )


@dg.asset(
    io_manager_key="pandas_parquet_io",
    group_name="stage2_validate_robustness",
    description="Log robustness report and summary metrics to MLflow when enabled.",
)
def robustness_mlflow_record(
    robustness_report_json: pd.DataFrame,
    object_store: ObjectStoreResource,
    chroma: ChromaStoreResource,
    mlflow_resource: MlflowResource,
) -> dg.MaterializeResult:
    from improved_pipelines.validate_robustness import log_robustness_mlflow

    t0 = time.perf_counter()
    path = Path(robustness_report_json.iloc[0]["json_path"])
    report = json.loads(path.read_text(encoding="utf-8"))
    uri = mlflow_resource.effective_tracking_uri()
    logged = False
    if uri:
        import os

        prev = os.environ.get("MLFLOW_TRACKING_URI")
        os.environ["MLFLOW_TRACKING_URI"] = uri
        try:
            log_robustness_mlflow(
                report,
                object_store_root=object_store.root,
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
    out = pd.DataFrame(
        [
            {
                "mlflow_logged": logged,
                "tracking_uri_used": uri or "",
                "overall_mean_cosine": float(report["overall_mean_cosine"]),
            }
        ]
    )
    elapsed = time.perf_counter() - t0
    return dg.MaterializeResult(
        metadata={
            "mlflow_logged": dg.MetadataValue.bool(logged),
            "overall_mean_cosine": dg.MetadataValue.float(float(report["overall_mean_cosine"])),
            "elapsed_s": dg.MetadataValue.float(round(elapsed, 4)),
        },
        value=out,
    )
