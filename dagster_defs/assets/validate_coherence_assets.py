"""Stage 2c: embedding coherence validation (compactness + class separation)."""

import json
import time
from pathlib import Path

import dagster as dg
import pandas as pd

from dagster_defs.resources import ChromaStoreResource, MlflowResource, RepoRootResource


class CoherenceValidationConfig(dg.Config):
    min_exemplars: int = 3
    min_intra_sim: float = 0.55
    max_inter_sim: float = 0.90
    report_relative_path: str = "data/validation_coherence_report.json"


@dg.asset(
    io_manager_key="pandas_parquet_io",
    group_name="stage2_validate_coherence",
    description="Per-class intra-class cosine similarity statistics.",
)
def coherence_intra_class(
    preload_embeddings_summary: pd.DataFrame,
    chroma: ChromaStoreResource,
) -> dg.MaterializeResult:
    from improved_pipelines.validate_coherence import compute_intra_class_similarity_df

    _ = preload_embeddings_summary
    t0 = time.perf_counter()
    grouped = chroma.get_embedding_store().get_embeddings_grouped_by_class_name()
    df = compute_intra_class_similarity_df(grouped)
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
    group_name="stage2_validate_coherence",
    description="Pairwise centroid cosine similarity between classes.",
)
def coherence_inter_class(
    preload_embeddings_summary: pd.DataFrame,
    chroma: ChromaStoreResource,
) -> dg.MaterializeResult:
    from improved_pipelines.validate_coherence import compute_inter_class_separation_df

    _ = preload_embeddings_summary
    t0 = time.perf_counter()
    grouped = chroma.get_embedding_store().get_embeddings_grouped_by_class_name()
    df = compute_inter_class_separation_df(grouped)
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
    group_name="stage2_validate_coherence",
    description="Per-class health flags using compactness and separation thresholds.",
)
def coherence_class_health(
    coherence_intra_class: pd.DataFrame,
    coherence_inter_class: pd.DataFrame,
    config: CoherenceValidationConfig,
) -> dg.MaterializeResult:
    from improved_pipelines.validate_coherence import assess_class_health_df

    t0 = time.perf_counter()
    df = assess_class_health_df(
        coherence_intra_class,
        coherence_inter_class,
        min_exemplars=config.min_exemplars,
        min_intra_sim=config.min_intra_sim,
        max_inter_sim=config.max_inter_sim,
    )
    elapsed = time.perf_counter() - t0
    unhealthy = int((~df["is_healthy"]).sum()) if not df.empty else 0
    return dg.MaterializeResult(
        metadata={
            "row_count": dg.MetadataValue.int(len(df)),
            "classes_unhealthy": dg.MetadataValue.int(unhealthy),
            "elapsed_s": dg.MetadataValue.float(round(elapsed, 4)),
        },
        value=df,
    )


@dg.asset(
    io_manager_key="pandas_parquet_io",
    group_name="stage2_validate_coherence",
    description="Write embedding coherence JSON report.",
)
def coherence_report_json(
    coherence_intra_class: pd.DataFrame,
    coherence_inter_class: pd.DataFrame,
    coherence_class_health: pd.DataFrame,
    repo: RepoRootResource,
    config: CoherenceValidationConfig,
) -> dg.MaterializeResult:
    from improved_pipelines.validate_coherence import (
        build_coherence_report_dict,
        write_coherence_report_json,
    )

    t0 = time.perf_counter()
    warning = None
    if coherence_intra_class.empty:
        warning = "no_embeddings_in_chroma_collection"
    report = build_coherence_report_dict(
        coherence_intra_class,
        coherence_inter_class,
        coherence_class_health,
        min_exemplars=config.min_exemplars,
        min_intra_sim=config.min_intra_sim,
        max_inter_sim=config.max_inter_sim,
        warning=warning,
    )
    path = Path(repo.path) / config.report_relative_path
    write_coherence_report_json(path, report)
    out = pd.DataFrame([{"json_path": str(path.resolve()), "bytes_written": path.stat().st_size}])
    elapsed = time.perf_counter() - t0
    return dg.MaterializeResult(
        metadata={
            "json_path": dg.MetadataValue.path(str(path.resolve())),
            "classes_total": dg.MetadataValue.int(int(report["classes_total"])),
            "classes_unhealthy": dg.MetadataValue.int(int(report["classes_unhealthy"])),
            "elapsed_s": dg.MetadataValue.float(round(elapsed, 4)),
        },
        value=out,
    )


@dg.asset(
    io_manager_key="pandas_parquet_io",
    group_name="stage2_validate_coherence",
    description="Log coherence report and summary metrics to MLflow when enabled.",
)
def coherence_mlflow_record(
    coherence_report_json: pd.DataFrame,
    coherence_class_health: pd.DataFrame,
    chroma: ChromaStoreResource,
    mlflow_resource: MlflowResource,
) -> dg.MaterializeResult:
    from improved_pipelines.validate_coherence import log_coherence_mlflow

    t0 = time.perf_counter()
    path = Path(coherence_report_json.iloc[0]["json_path"])
    report = json.loads(path.read_text(encoding="utf-8"))
    uri = mlflow_resource.effective_tracking_uri()
    logged = False
    if uri:
        import os

        prev = os.environ.get("MLFLOW_TRACKING_URI")
        os.environ["MLFLOW_TRACKING_URI"] = uri
        try:
            log_coherence_mlflow(
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

    out = pd.DataFrame(
        [
            {
                "mlflow_logged": logged,
                "tracking_uri_used": uri or "",
                "classes_unhealthy": int((~coherence_class_health["is_healthy"]).sum())
                if not coherence_class_health.empty
                else 0,
            }
        ]
    )
    elapsed = time.perf_counter() - t0
    return dg.MaterializeResult(
        metadata={
            "mlflow_logged": dg.MetadataValue.bool(logged),
            "classes_unhealthy": dg.MetadataValue.int(int(report["classes_unhealthy"])),
            "elapsed_s": dg.MetadataValue.float(round(elapsed, 4)),
        },
        value=out,
    )
