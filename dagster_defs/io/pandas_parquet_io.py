"""Parquet I/O manager for pandas DataFrame assets."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import dagster as dg
import pandas as pd


class PandasParquetIOManager(dg.ConfigurableIOManager):
    """Persist DataFrame assets as Parquet under ``base_path`` / asset_key path."""

    base_path: str = "dagster_storage/parquet"

    def _path_for_key(self, asset_key: dg.AssetKey) -> Path:
        safe = Path(*asset_key.path)
        return Path(self.base_path) / safe / "data.parquet"

    def handle_output(self, context: dg.OutputContext, obj: Any) -> None:
        if not isinstance(obj, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(obj)}")
        path = self._path_for_key(context.asset_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        obj.to_parquet(path, index=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        context.add_output_metadata(
            {
                "row_count": dg.MetadataValue.int(int(len(obj))),
                "num_columns": dg.MetadataValue.int(int(len(obj.columns))),
                "path": dg.MetadataValue.path(str(path.resolve())),
                "write_time_ms": dg.MetadataValue.float(round(elapsed_ms, 3)),
            }
        )

    def load_input(self, context: dg.InputContext) -> pd.DataFrame:
        if context.upstream_output is None:
            raise RuntimeError("Missing upstream output for DataFrame load")
        path = self._path_for_key(context.upstream_output.asset_key)
        if not path.is_file():
            raise FileNotFoundError(path)
        return pd.read_parquet(path)
