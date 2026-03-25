"""On-device object storage layout: images/, labels/, classes.txt."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ObjectStorePaths:
    root: Path

    @property
    def images_dir(self) -> Path:
        return self.root / "images"

    @property
    def labels_dir(self) -> Path:
        return self.root / "labels"

    @property
    def classes_file(self) -> Path:
        return self.root / "classes.txt"


class ObjectStore:
    """Filesystem object store: ``images/<id>.<ext>`` + ``labels/<id>.txt`` + ``classes.txt``."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.paths = ObjectStorePaths(Path(root).resolve())
        self.paths.images_dir.mkdir(parents=True, exist_ok=True)
        self.paths.labels_dir.mkdir(parents=True, exist_ok=True)

    def load_class_names(self) -> List[str]:
        p = self.paths.classes_file
        if not p.is_file():
            raise FileNotFoundError(f"Missing classes file: {p}")
        lines = p.read_text(encoding="utf-8").splitlines()
        return [
            ln.strip()
            for ln in lines
            if ln.strip() and not ln.strip().startswith("#")
        ]

    def iter_labeled_images(self) -> Iterator[Tuple[Path, Path, str]]:
        """Yield ``(abs_image_path, abs_label_path, image_id)`` for pairs that exist."""
        if not self.paths.images_dir.is_dir():
            return
        for img in sorted(self.paths.images_dir.iterdir()):
            if not img.is_file() or img.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image_id = img.stem
            label = self.paths.labels_dir / f"{image_id}.txt"
            if label.is_file():
                yield (img.resolve(), label.resolve(), image_id)

    @staticmethod
    def import_yolo_dataset(
        dest_root: str | os.PathLike[str],
        images_dir: str | os.PathLike[str],
        labels_dir: str | os.PathLike[str],
        classes_file: str | os.PathLike[str],
    ) -> ObjectStore:
        """Copy a YOLO-style tree into the object store layout."""
        store = ObjectStore(dest_root)
        shutil.copy2(Path(classes_file), store.paths.classes_file)
        idest = store.paths.images_dir
        ldest = store.paths.labels_dir
        for img in Path(images_dir).iterdir():
            if img.is_file() and img.suffix.lower() in IMAGE_EXTENSIONS:
                shutil.copy2(img, idest / img.name)
                lab = Path(labels_dir) / f"{img.stem}.txt"
                if lab.is_file():
                    shutil.copy2(lab, ldest / lab.name)
        return store
