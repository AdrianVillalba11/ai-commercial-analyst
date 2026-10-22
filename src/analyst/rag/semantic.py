"""
BÚSQUEDA SEMÁNTICA (ChromaDB)

Qué hace: guarda los vectores de los fragmentos en ChromaDB y devuelve los más parecidos a la pregunta
(distancia coseno).
Con qué se conecta: la usa rag/retriever.py; los vectores los genera rag/embedder.py.
Idea clave: el generador de vectores se inyecta, así se puede cambiar de modelo sin tocar este módulo.
Junto a la colección se guarda una "huella" (hash del nombre del modelo y del texto de cada fragmento):
si coincide en el siguiente arranque se reutilizan los vectores; si no, se reconstruye el índice.
"""
import hashlib
from pathlib import Path
from typing import Protocol

import chromadb
from chromadb.config import Settings

from analyst.docs.chunker import Chunk

# =====================================================================
# BLOQUE: Constantes
# =====================================================================
COLLECTION_NAME = "document_chunks"
_BATCH = 64


# =====================================================================
# BLOQUE: Contrato del generador de vectores
# =====================================================================
class Embedder(Protocol):
    """Lo único que se le exige a un generador de vectores (permite usar uno falso en las pruebas)."""

    @property
    def name(self) -> str: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


# =====================================================================
# BLOQUE: Índice persistente
# =====================================================================
def fingerprint(chunks: list[Chunk], model_name: str) -> str:
    """Huella del contenido y del modelo: si cambia cualquiera, hay que reconstruir los vectores."""
    digest = hashlib.sha256(model_name.encode("utf-8"))
    for chunk in chunks:
        digest.update(f"\x1e{chunk.chunk_id}\x1f{chunk.embed_text}".encode("utf-8"))
    return digest.hexdigest()


class SemanticIndex:
    """Colección de ChromaDB con los fragmentos; se reconstruye solo si la huella cambió."""

    def __init__(self, chunks: list[Chunk], embedder: Embedder, path: Path,
                 collection_name: str = COLLECTION_NAME):
        self._embedder = embedder
        # Telemetría desactivada: la app maneja documentos de la empresa y no debe enviar nada afuera.
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
        """Calcula y guarda los vectores de todos los fragmentos, por lotes."""
        for start in range(0, len(chunks), _BATCH):
            batch = chunks[start:start + _BATCH]
            collection.add(
                ids=[c.chunk_id for c in batch],
                embeddings=self._embedder.embed_documents([c.embed_text for c in batch]),
                documents=[c.text for c in batch],
                metadatas=[{"doc": c.doc, "section": c.section, "kind": c.kind} for c in batch],
            )

    def search(self, query: str, k: int | None = None) -> list[str]:
        """Ids de fragmentos del más al menos parecido. Sin k devuelve el ranking completo (RRF lo necesita)."""
        total = self._collection.count()
        if total == 0:
            return []
        result = self._collection.query(
            query_embeddings=[self._embedder.embed_query(query)],
            n_results=min(k, total) if k else total,
            include=["distances"],
        )
        return list(result["ids"][0])
