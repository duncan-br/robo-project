"""
DAG: run DVC stage ``preload_embeddings`` in the Open Robo Vision repo.

- Set env ``OPEN_ROBO_VISION_ROOT`` to the repository root (default ``/app`` in Docker).
- Airflow should run in its **own** venv (see ``requirements-airflow.txt``); the stage
  invokes ``dvc`` + the project interpreter on the host/container where the repo lives.

Schedule is manual by default (``schedule=None``); change to a cron string if needed.
"""

from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="preload_owl_embeddings",
    default_args={
        "owner": "open_robo_vision",
        "depends_on_past": False,
        "retries": 1,
    },
    description="DVC repro: preload OWL GT embeddings → Chroma + summary JSON",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["open_robo_vision", "embeddings", "dvc"],
) as dag:
    BashOperator(
        task_id="dvc_repro_preload_embeddings",
        bash_command=r"""
set -euo pipefail
ROOT="${OPEN_ROBO_VISION_ROOT:-/app}"
cd "$ROOT"
dvc repro preload_embeddings
""",
    )
