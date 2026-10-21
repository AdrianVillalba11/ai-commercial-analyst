"""Hybrid retriever: BM25 + semantic search fused with RRF.

Built only from chunks that passed the injection screening, so quarantined text is not in any
index. Results are re-checked at query time. If Ollama is unavailable it degrades to lexical
search and says so, instead of failing.
"""
from dataclasses import dataclass, field
from pathlib import Path

from analyst.config import settings
from analyst.docs.chunker import Chunk
from analyst.errors import OllamaError
from analyst.rag.embedder import OllamaEmbedder
from analyst.rag.fusion import reciprocal_rank_fusion
from analyst.rag.lexical import BM25Index
from analyst.rag.semantic import Embedder, SemanticIndex
from analyst.security.injection import ScreeningResult, filter_safe, load_safe_corpus

MODES = ("hybrid", "lexical", "semantic")


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    ranks: dict[str, int] = field(default_factory=dict)   # which ranker put it where


@dataclass
class RetrievalResult:
    items: list[RetrievedChunk]
    degraded: bool = False
    notes: list[str] = field(default_factory=list)


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None, store_path: Path | None = None,
                 candidates: int = 10, rrf_k: int = 60, weights: dict[str, float] | None = None):
        self._chunks = {c.chunk_id: c for c in chunks}
        self._lexical = BM25Index(chunks)
        self._embedder = embedder
        self._store_path = Path(store_path) if store_path else settings.chroma_path
        self._semantic: SemanticIndex | None = None
        self._candidates, self._rrf_k = candidates, rrf_k
        self._weights = weights

    def retrieve(self, query: str, top_k: int = 6, regions: list[str] | None = None,
                 mode: str = "hybrid") -> RetrievalResult:
        if top_k < 1:
            raise ValueError("top_k debe ser al menos 1.")
        if mode not in MODES:
            raise ValueError(f"mode no válido: {mode!r}. Opciones: {MODES}")
        if not query.strip():
            return RetrievalResult([], notes=["La consulta está vacía."])

        allowed = self._allowed_ids(regions)
        rankings: dict[str, list[str]] = {}
        notes: list[str] = []
        degraded = False

        if mode in ("hybrid", "lexical"):
            lexical = [cid for cid, _ in self._lexical.search(query) if cid in allowed]
            rankings["lexical"] = lexical[: self._candidates]
        if mode in ("hybrid", "semantic"):
            try:
                semantic = self._semantic_index().search(query)
                rankings["semantic"] = [cid for cid in semantic if cid in allowed][: self._candidates]
            except OllamaError as exc:
                if mode == "semantic":
                    raise
                degraded = True
                notes.append(f"Búsqueda semántica no disponible; se usó solo la léxica. {exc}")

        fused = reciprocal_rank_fusion(rankings, k=self._rrf_k, weights=self._weights)[:top_k]
        candidates = [self._chunks[hit.chunk_id] for hit in fused]
        safe_ids = {c.chunk_id for c in filter_safe(candidates)}   # second injection check, at query time
        items = [RetrievedChunk(self._chunks[h.chunk_id], h.score, h.ranks) for h in fused if h.chunk_id in safe_ids]
        return RetrievalResult(items, degraded, notes)

    def _semantic_index(self) -> SemanticIndex:
        if self._semantic is None:
            if self._embedder is None:
                raise OllamaError("No hay un modelo de embeddings configurado.")
            self._semantic = SemanticIndex(list(self._chunks.values()), self._embedder, self._store_path)
        return self._semantic

    def _allowed_ids(self, regions: list[str] | None) -> set[str]:
        """With a region filter, keep chunks about those regions plus general ones (no region mentioned)."""
        if not regions:
            return set(self._chunks)
        wanted = set(regions)
        return {cid for cid, c in self._chunks.items() if not c.regions or wanted & set(c.regions)}


def build_default_retriever() -> tuple[HybridRetriever, ScreeningResult]:
    """Loads, screens and indexes the project documents. Returns the screening too, for the alerts panel."""
    corpus = load_safe_corpus()
    return HybridRetriever(corpus.safe, OllamaEmbedder(), settings.chroma_path), corpus
