"""Build inference image manifests (no OWL/JAX imports)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd


def build_inference_manifest_df(
    image_dir: Optional[str],
    image_paths: Optional[List[str]],
) -> pd.DataFrame:
    """
    Build a manifest of images to run batch inference on.
    Provide ``image_dir`` and/or ``image_paths``; at least one must yield paths.
    """
    paths: List[Path] = []
    if image_paths:
        paths.extend(Path(p) for p in image_paths)
    if image_dir:
        root = Path(image_dir)
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            paths.extend(root.glob(f"*{ext}"))
            paths.extend(root.glob(f"*{ext.upper()}"))
    resolved = sorted({p.resolve() for p in paths if p.is_file()})
    return pd.DataFrame({"image_path": [str(p) for p in resolved]})
