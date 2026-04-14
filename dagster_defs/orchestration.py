"""Schedules and sensors for Dagster seed + validation jobs."""

from __future__ import annotations

import os
from pathlib import Path

import dagster as dg


def _default_chroma_dir() -> str:
    return os.environ.get("CHROMA_PERSIST_DIR", "")


def _is_seeded(chroma_dir: str) -> bool:
    if not chroma_dir:
        return False
    path = Path(chroma_dir)
    if not path.exists() or not path.is_dir():
        return False
    return any(path.iterdir())


def build_daily_validation_schedule(validation_job: dg.JobDefinition) -> dg.ScheduleDefinition:
    @dg.schedule(
        name="daily_validation_schedule",
        job=validation_job,
        cron_schedule="@daily",
        execution_timezone="UTC",
    )
    def _daily_validation_schedule(_context: dg.ScheduleEvaluationContext):
        return {}

    return _daily_validation_schedule


def build_seed_once_sensor(seed_job: dg.JobDefinition) -> dg.SensorDefinition:
    @dg.sensor(name="seed_once_sensor", job=seed_job, minimum_interval_seconds=30)
    def _seed_once_sensor(context: dg.SensorEvaluationContext):
        if (context.cursor or "").strip() == "seeded":
            return dg.SkipReason("Seed sensor already completed (cursor=seeded).")

        chroma_dir = _default_chroma_dir()
        if _is_seeded(chroma_dir):
            context.update_cursor("seeded")
            return dg.SkipReason(f"Chroma already has contents at {chroma_dir}; marking seed complete.")

        context.update_cursor("seeded")
        return dg.RunRequest(run_key="seed_once_v1")

    return _seed_once_sensor
