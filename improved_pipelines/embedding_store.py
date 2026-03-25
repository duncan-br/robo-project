"""Chroma persistent vector store for OWL class-head embeddings."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

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
