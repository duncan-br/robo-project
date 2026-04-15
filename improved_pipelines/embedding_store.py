"""Chroma persistent vector store for OWL class-head embeddings."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import chromadb
from chromadb.api.models.Collection import Collection


class ChromaEmbeddingStore:
    def __init__(self, persist_directory: str, collection_name: str = "owl_gt_embeddings") -> None:
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection_name = collection_name
        self._collection: Collection = self._client.get_or_create_collection(name=collection_name)

    @property
    def collection(self) -> Collection:
        return self._collection

    def reset(self) -> None:
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(name=self._collection_name)

    def add_embeddings(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        emb_list = [list(map(float, e)) for e in embeddings]
        meta_list: Optional[List[Dict[str, Any]]] = None
        if metadatas is not None:
            meta_list = [_sanitize_metadata(m) for m in metadatas]
        self._collection.add(ids=list(ids), embeddings=emb_list, metadatas=meta_list)

    def count(self) -> int:
        return self._collection.count()

    def get_preload_incremental_index(
        self,
        batch_size: int = 1024,
    ) -> Tuple[Set[str], Dict[str, str], Set[str]]:
        """Scan metadata to determine which image_ids exist, their fingerprints,
        and which ones need re-processing (missing/inconsistent fingerprints)."""
        col = self._collection
        total = col.count()
        if total == 0:
            return set(), {}, set()

        image_ids_seen: Set[str] = set()
        fingerprint_by_image_id: Dict[str, str] = {}
        needs_reprocess: Set[str] = set()
        offset = 0
        while offset < total:
            batch = col.get(include=["metadatas"], limit=batch_size, offset=offset)
            ids = batch.get("ids") or []
            metas = batch.get("metadatas") or []
            for meta in metas:
                if not meta:
                    continue
                raw_id = meta.get("image_id")
                if raw_id is None:
                    continue
                iid = str(raw_id)
                image_ids_seen.add(iid)
                fp_raw = meta.get("preload_source_fingerprint")
                if fp_raw is None or fp_raw == "":
                    needs_reprocess.add(iid)
                    continue
                sfp = str(fp_raw)
                if iid not in fingerprint_by_image_id:
                    fingerprint_by_image_id[iid] = sfp
                elif fingerprint_by_image_id[iid] != sfp:
                    needs_reprocess.add(iid)
            offset += len(ids)
            if len(ids) == 0:
                break

        for iid in needs_reprocess:
            fingerprint_by_image_id.pop(iid, None)

        return image_ids_seen, fingerprint_by_image_id, needs_reprocess

    def delete_embeddings_for_image_id(self, image_id: str) -> int:
        """Remove all vectors whose metadata ``image_id`` matches. Returns number of rows deleted."""
        col = self._collection
        res = col.get(where={"image_id": {"$eq": image_id}}, include=[])
        ids = res.get("ids") or []
        if ids:
            col.delete(ids=list(ids))
        return len(ids)

    def delete_by_class_name(self, class_name: str) -> int:
        """Remove all vectors whose metadata ``class_name`` matches."""
        col = self._collection
        res = col.get(where={"class_name": {"$eq": str(class_name)}}, include=[])
        ids = res.get("ids") or []
        if ids:
            col.delete(ids=list(ids))
        return len(ids)

    def class_counts(self, batch_size: int = 1024) -> Dict[str, int]:
        """Return embedding counts per class_name."""
        total = self._collection.count()
        if total <= 0:
            return {}
        counts: Dict[str, int] = {}
        offset = 0
        while offset < total:
            batch = self._collection.get(include=["metadatas"], limit=batch_size, offset=offset)
            ids = batch.get("ids") or []
            metas = batch.get("metadatas") or []
            for meta in metas:
                if not meta:
                    continue
                cname = meta.get("class_name")
                if not cname:
                    continue
                key = str(cname)
                counts[key] = counts.get(key, 0) + 1
            offset += len(ids)
            if not ids:
                break
        return counts

    def get_embeddings_grouped_by_class_name(
        self,
        batch_size: int = 256,
    ) -> Dict[str, List[np.ndarray]]:
        """Fetch all vectors and group by metadata ``class_name`` (from preload)."""
        col = self._collection
        total = col.count()
        if total == 0:
            return {}

        grouped: DefaultDict[str, List[np.ndarray]] = defaultdict(list)
        offset = 0
        while offset < total:
            batch = col.get(
                include=["embeddings", "metadatas"],
                limit=batch_size,
                offset=offset,
            )
            ids = batch.get("ids")
            if ids is None:
                ids = []
            if len(ids) == 0:
                break
            embs = batch.get("embeddings")
            if embs is None:
                embs = []
            metas = batch.get("metadatas")
            if metas is None:
                metas = []
            if isinstance(embs, np.ndarray) and embs.ndim == 2:
                embs = [embs[i] for i in range(embs.shape[0])]
            for emb, meta in zip(embs, metas):
                if meta is None:
                    continue
                name = meta.get("class_name")
                if not name:
                    continue
                grouped[str(name)].append(np.asarray(emb, dtype=np.float32))
            offset += len(ids)

        return dict(grouped)

    def get_all_embeddings_with_image_metadata(
        self,
        batch_size: int = 256,
    ) -> List[Dict[str, Any]]:
        """Fetch all embeddings with image_id and class_name (skips incomplete rows)."""
        col = self._collection
        total = col.count()
        if total == 0:
            return []

        rows: List[Dict[str, Any]] = []
        offset = 0
        while offset < total:
            batch = col.get(
                include=["embeddings", "metadatas"],
                limit=batch_size,
                offset=offset,
            )
            ids = batch.get("ids")
            if ids is None:
                ids = []
            if len(ids) == 0:
                break
            embs = batch.get("embeddings")
            if embs is None:
                embs = []
            metas = batch.get("metadatas")
            if metas is None:
                metas = []
            if isinstance(embs, np.ndarray) and embs.ndim == 2:
                embs = [embs[i] for i in range(embs.shape[0])]

            for rid, emb, meta in zip(ids, embs, metas):
                if meta is None:
                    continue
                image_id = meta.get("image_id")
                class_name = meta.get("class_name")
                if image_id is None or not class_name:
                    continue
                rows.append(
                    {
                        "chroma_id": str(rid),
                        "image_id": str(image_id),
                        "class_name": str(class_name),
                        "embedding": np.asarray(emb, dtype=np.float32),
                    }
                )

            offset += len(ids)

        return rows


def _sanitize_metadata(m: Dict[str, Any]) -> Dict[str, Any]:
    """Chroma metadata: str | int | float | bool only."""
    out: Dict[str, Any] = {}
    for k, v in m.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (list, dict)):
            out[k] = json.dumps(v)
        else:
            out[k] = str(v)
    return out
