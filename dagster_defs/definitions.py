"""Dagster code location: assets, jobs, schedules, and resources for the validation pipelines."""

from __future__ import annotations

import os

import dagster as dg

from dagster_defs.assets import (
    coherence_class_health,
    coherence_inter_class,
    coherence_intra_class,
    coherence_mlflow_record,
    coherence_report_json,
    gold_validation_metrics,
    gold_validation_per_image,
    gold_validation_sample,
    labeled_image_pairs,
    leaveout_embedding_split,
    leaveout_knn_predictions,
    leaveout_metrics,
    leaveout_mlflow_record,
    leaveout_report_json,
    robustness_augmented_embeddings,
    robustness_metrics,
    robustness_mlflow_record,
    robustness_report_json,
    robustness_sample,
    preload_embeddings_summary,
    preload_summary_json,
    validation_mlflow_record,
    validation_report_json,
)
from dagster_defs.io.pandas_parquet_io import PandasParquetIOManager
from dagster_defs.orchestration import (
    build_daily_validation_schedule,
    build_seed_once_sensor,
)
from dagster_defs.resources import (
    ChromaStoreResource,
    MlflowResource,
    ObjectStoreResource,
    OwlDetectorResource,
    RepoRootResource,
    default_repo_root,
)

_REPO = default_repo_root()


def _env_path(var: str, default: str) -> str:
    v = os.environ.get(var)
    return v if v else default


_parquet_io = PandasParquetIOManager(
    base_path=_env_path("DAGSTER_PARQUET_STORAGE", str(_REPO / "dagster_storage" / "parquet"))
)

_all_assets = [
    labeled_image_pairs,
    preload_embeddings_summary,
    preload_summary_json,
    gold_validation_sample,
    gold_validation_per_image,
    gold_validation_metrics,
    validation_report_json,
    validation_mlflow_record,
    leaveout_embedding_split,
    leaveout_knn_predictions,
    leaveout_metrics,
    leaveout_report_json,
    leaveout_mlflow_record,
    coherence_intra_class,
    coherence_inter_class,
    coherence_class_health,
    coherence_report_json,
    coherence_mlflow_record,
    robustness_sample,
    robustness_augmented_embeddings,
    robustness_metrics,
    robustness_report_json,
    robustness_mlflow_record,
]

gold_validation_job = dg.define_asset_job(
    name="gold_validation_job",
    selection=[
        labeled_image_pairs,
        preload_embeddings_summary,
        preload_summary_json,
        gold_validation_sample,
        gold_validation_per_image,
        gold_validation_metrics,
        validation_report_json,
        validation_mlflow_record,
    ],
    description=(
        "Preload embeddings + gold QC validation pipeline. "
        "Live inference and review run in the on-device app."
    ),
)

seed_chroma_job = dg.define_asset_job(
    name="seed_chroma_job",
    selection=[
        labeled_image_pairs,
        preload_embeddings_summary,
        preload_summary_json,
    ],
    description="One-time Chroma seed from labeled pairs and preload assets.",
)

leaveout_validation_job = dg.define_asset_job(
    name="leaveout_validation_job",
    selection=[
        preload_embeddings_summary,
        leaveout_embedding_split,
        leaveout_knn_predictions,
        leaveout_metrics,
        leaveout_report_json,
        leaveout_mlflow_record,
    ],
    description="K-fold embedding cross-validation using cosine KNN.",
)

coherence_validation_job = dg.define_asset_job(
    name="coherence_validation_job",
    selection=[
        preload_embeddings_summary,
        coherence_intra_class,
        coherence_inter_class,
        coherence_class_health,
        coherence_report_json,
        coherence_mlflow_record,
    ],
    description="Embedding coherence validation (intra-class compactness and inter-class separation).",
)

robustness_validation_job = dg.define_asset_job(
    name="robustness_validation_job",
    selection=[
        labeled_image_pairs,
        preload_embeddings_summary,
        robustness_sample,
        robustness_augmented_embeddings,
        robustness_metrics,
        robustness_report_json,
        robustness_mlflow_record,
    ],
    description="Embedding robustness validation using small augmentations and cosine drift checks.",
)

daily_validation_schedule = build_daily_validation_schedule(gold_validation_job)
seed_once_sensor = build_seed_once_sensor(seed_chroma_job)

defs = dg.Definitions(
    assets=_all_assets,
    jobs=[
        gold_validation_job,
        seed_chroma_job,
        leaveout_validation_job,
        coherence_validation_job,
        robustness_validation_job,
    ],
    schedules=[daily_validation_schedule],
    sensors=[seed_once_sensor],
    resources={
        "repo": RepoRootResource(path=str(_REPO)),
        "object_store": ObjectStoreResource(
            root=_env_path("OBJECT_STORE_ROOT", str(_REPO / "data" / "object_store")),
        ),
        "chroma": ChromaStoreResource(
            persist_directory=_env_path("CHROMA_PERSIST_DIR", str(_REPO / "data" / "chroma_db")),
            collection_name=os.environ.get("CHROMA_COLLECTION", "owl_gt_embeddings"),
        ),
        "owl_detector": OwlDetectorResource(),
        "mlflow_resource": MlflowResource(
            tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
        ),
        "io_manager": _parquet_io,
        "pandas_parquet_io": _parquet_io,
    },
)
