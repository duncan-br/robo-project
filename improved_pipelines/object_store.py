"""On-device object storage layout: images/, labels/, classes.txt."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd

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
            return []
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

    def save_infer_result(
        self,
        source_image: Path,
        yolo_lines: List[Tuple[int, float, float, float, float]],
        stem_prefix: str = "infer",
    ) -> Tuple[Path, Path]:
        """
        Copy ``source_image`` into ``images/`` with a new stem and write ``labels/<stem>.txt``.
        Each yolo line: ``class_id, cx, cy, w, h`` normalized.
        """
        import uuid

        sid = f"{stem_prefix}_{uuid.uuid4().hex[:10]}"
        ext = source_image.suffix.lower() or ".jpg"
        dest_img = self.paths.images_dir / f"{sid}{ext}"
        shutil.copy2(source_image, dest_img)
        dest_lbl = self.paths.labels_dir / f"{sid}.txt"
        with dest_lbl.open("w", encoding="utf-8") as f:
            for cid, cx, cy, w, h in yolo_lines:
                f.write(f"{int(cid)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        return dest_img, dest_lbl

    def append_yolo_line(self, image_stem: str, line: Tuple[int, float, float, float, float]) -> None:
        """Append one YOLO line to ``labels/<stem>.txt`` (create if missing)."""
        p = self.paths.labels_dir / f"{image_stem}.txt"
        cid, cx, cy, w, h = line
        with p.open("a", encoding="utf-8") as f:
            f.write(f"{int(cid)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    def ensure_image_copy(self, source_image: Path, stem: str) -> Path:
        """Copy image to ``images/<stem>.<ext>`` if not already there."""
        ext = source_image.suffix.lower() or ".jpg"
        dest = self.paths.images_dir / f"{stem}{ext}"
        if not dest.is_file():
            shutil.copy2(source_image, dest)
        return dest

    def class_id_for_name(self, name: str, create: bool = False) -> int:
        """Return YOLO class index for ``name``; optionally append to ``classes.txt``."""
        name = name.strip()
        if not name:
            raise ValueError("empty class name")
        names = self.load_class_names()
        if name in names:
            return names.index(name)
        if not create:
            raise KeyError(f"Unknown class {name!r}; enable create or pick existing.")
        with self.paths.classes_file.open("a", encoding="utf-8") as f:
            f.write(name + "\n")
        return len(names)

    def remove_class(self, class_name: str) -> dict[str, int]:
        """Remove class and reindex label ids; drop annotations of the removed class."""
        names = self.load_class_names()
        if class_name not in names:
            raise KeyError(f"Unknown class {class_name!r}")
        remove_idx = int(names.index(class_name))
        new_names = [name for i, name in enumerate(names) if i != remove_idx]
        self._write_classes(new_names)

        files_touched = 0
        boxes_removed = 0
        boxes_reindexed = 0
        for label_path in sorted(self.paths.labels_dir.glob("*.txt")):
            if not label_path.is_file():
                continue
            lines_out: list[str] = []
            changed = False
            for raw in label_path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    lines_out.append(line)
                    continue
                try:
                    cid = int(parts[0])
                except Exception:  # noqa: BLE001
                    lines_out.append(line)
                    continue
                if cid == remove_idx:
                    boxes_removed += 1
                    changed = True
                    continue
                if cid > remove_idx:
                    cid -= 1
                    boxes_reindexed += 1
                    changed = True
                parts[0] = str(cid)
                lines_out.append(" ".join(parts))
            if changed:
                label_path.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")
                files_touched += 1
        return {
            "files_touched": files_touched,
            "boxes_removed": boxes_removed,
            "boxes_reindexed": boxes_reindexed,
        }

    def clear_all(self) -> dict[str, int]:
        """Remove all images/labels/classes content and recreate empty layout."""
        images_deleted = self._clear_dir(self.paths.images_dir)
        labels_deleted = self._clear_dir(self.paths.labels_dir)
        classes_deleted = 0
        if self.paths.classes_file.exists():
            self.paths.classes_file.unlink()
            classes_deleted = 1
        self.paths.images_dir.mkdir(parents=True, exist_ok=True)
        self.paths.labels_dir.mkdir(parents=True, exist_ok=True)
        self.paths.classes_file.write_text("", encoding="utf-8")
        return {
            "images_deleted": images_deleted,
            "labels_deleted": labels_deleted,
            "classes_file_reset": classes_deleted,
        }

    def _write_classes(self, names: list[str]) -> None:
        payload = "\n".join(names)
        if payload:
            payload += "\n"
        self.paths.classes_file.write_text(payload, encoding="utf-8")

    @staticmethod
    def _clear_dir(path: Path) -> int:
        if not path.exists():
            return 0
        deleted = 0
        for child in path.iterdir():
            if child.is_file():
                child.unlink()
                deleted += 1
            elif child.is_dir():
                shutil.rmtree(child)
                deleted += 1
        return deleted


def labeled_pairs_dataframe(store: ObjectStore, limit: Optional[int] = None) -> pd.DataFrame:
    """Scan object store for image/label pairs; columns: image_path, label_path, image_id."""
    rows: List[Dict[str, str]] = []
    for abs_img, abs_lbl, image_id in store.iter_labeled_images():
        rows.append(
            {
                "image_path": str(abs_img),
                "label_path": str(abs_lbl),
                "image_id": image_id,
            }
        )
    if limit is not None:
        rows = rows[:limit]
    return pd.DataFrame(rows)
