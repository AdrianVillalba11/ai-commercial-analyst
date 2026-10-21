"""Semantic search: embeddings stored in ChromaDB (cosine distance).

The embedder is injected, so the model can change without touching this module. A fingerprint
(hash of the model name and every chunk's text) is stored with the collection: if it matches
on the next start the stored vectors are reused, otherwise the index is rebuilt.
"""
import hashlib
from pathlib import Path
from typing import Protocol

import chromadb
from chromadb.config import Settings

from analyst.docs.chunker import Chunk

COLLECTION_NAME = "document_chunks"
_BATCH = 64


class Embedder(Protocol):
    @property
    def name(self) -> str: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


def fingerprint(chunks: list[Chunk], model_name: str) -> str:
    digest = hashlib.sha256(model_name.encode("utf-8"))
    for chunk in chunks:
        digest.update(f"\x1e{chunk.chunk_id}\x1f{chunk.embed_text}".encode("utf-8"))
    return digest.hexdigest()


class SemanticIndex:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, path: Path,
                 collection_name: str = COLLECTION_NAME):
        self._embedder = embedder
        # Telemetry off: this app handles company documents and must not phone home.
        self._client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
        wanted = fingerprint(chunks, embedder.name)
        collection = self._client.get_or_create_collection(collection_name)
        self.rebuilt = False
        if (collection.metadata or {}).get("fingerprint") != wanted or collection.count() != len(chunks):
            self._client.delete_collection(collection_name)
            collection = self._client.create_collection(
                collection_name, metadata={"hnsw:space": "cosine", "fingerprint": wanted})
            self._fill(collection, chunks)
            self.rebuilt = True
        self._collection = collection

    def _fill(self, collection, chunks: list[Chunk]) -> None:
        for start in range(0, len(chunks), _BATCH):
            batch = chunks[start:start + _BATCH]
            collection.add(
                ids=[c.chunk_id for c in batch],
                embeddings=self._embedder.embed_documents([c.embed_text for c in batch]),
                documents=[c.text for c in batch],
                metadatas=[{"doc": c.doc, "section": c.section, "kind": c.kind} for c in batch],
            )

    def search(self, query: str, k: int | None = None) -> list[str]:
        """Chunk ids from most to least similar. Without k, the full ranking (RRF needs it)."""
        total = self._collection.count()
        if total == 0:
            return []
        result = self._collection.query(
            query_embeddings=[self._embedder.embed_query(query)],
            n_results=min(k, total) if k else total,
            include=["distances"],
        )
        return list(result["ids"][0])
