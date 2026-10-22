"""
BUSCADOR DE EVIDENCIA (léxico, semántico o híbrido)

Qué hace: dada una pregunta, devuelve los fragmentos de los documentos más relevantes.
Con qué se conecta: usa rag/lexical.py, rag/semantic.py y rag/fusion.py; lo llama el servicio.
Idea clave: se construye solo con fragmentos que pasaron la revisión de inyección, así que el texto
en cuarentena no está en ningún índice; además se revisa de nuevo al consultar. Si Ollama no está
disponible en modo híbrido, degrada a búsqueda léxica y lo avisa en vez de fallar.
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

# =====================================================================
# BLOQUE: Modos y resultados
# =====================================================================
MODES = ("hybrid", "lexical", "semantic")


@dataclass
class RetrievedChunk:
    """Un fragmento devuelto, con su puntaje y la posición que le dio cada buscador."""

    chunk: Chunk
    score: float
    ranks: dict[str, int] = field(default_factory=dict)   # qué buscador lo puso en qué posición


@dataclass
class RetrievalResult:
    """Resultado de una búsqueda: fragmentos, si hubo degradación y notas para mostrar."""

    items: list[RetrievedChunk]
    degraded: bool = False
    notes: list[str] = field(default_factory=list)


# =====================================================================
# BLOQUE: Buscador
# =====================================================================
class HybridRetriever:
    """Combina BM25 y búsqueda semántica con RRF; el índice semántico se crea la primera vez que se necesita."""

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
        """Busca evidencia. `regions` limita a esas regiones (más los fragmentos generales)."""
        if top_k < 1:
            raise ValueError("top_k debe ser al menos 1.")
        if mode not in MODES:
            raise ValueError(f"mode no válido: {mode!r}. Opciones: {MODES}")
        if not query.strip():
            return RetrievalResult([], notes=["La consulta está vacía."])

        # 1) Qué fragmentos son elegibles según el filtro de región.
        allowed = self._allowed_ids(regions)
        rankings: dict[str, list[str]] = {}
        notes: list[str] = []
        degraded = False

        # 2) Un ranking por cada buscador activo.
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

        # 3) Se fusionan los rankings y se revisa de nuevo que ningún fragmento sea inseguro.
        fused = reciprocal_rank_fusion(rankings, k=self._rrf_k, weights=self._weights)[:top_k]
        candidates = [self._chunks[hit.chunk_id] for hit in fused]
        safe_ids = {c.chunk_id for c in filter_safe(candidates)}   # segunda revisión de inyección, al consultar
        items = [RetrievedChunk(self._chunks[h.chunk_id], h.score, h.ranks) for h in fused if h.chunk_id in safe_ids]
        return RetrievalResult(items, degraded, notes)

    def _semantic_index(self) -> SemanticIndex:
        """Crea el índice semántico la primera vez (así el arranque no espera a Ollama si no se usa)."""
        if self._semantic is None:
            if self._embedder is None:
                raise OllamaError("No hay un modelo de embeddings configurado.")
            self._semantic = SemanticIndex(list(self._chunks.values()), self._embedder, self._store_path)
        return self._semantic

    def _allowed_ids(self, regions: list[str] | None) -> set[str]:
        """Con filtro de región: fragmentos de esas regiones más los generales (que no mencionan ninguna región)."""
        if not regions:
            return set(self._chunks)
        wanted = set(regions)
        return {cid for cid, c in self._chunks.items() if not c.regions or wanted & set(c.regions)}


# =====================================================================
# BLOQUE: Construcción por defecto
# =====================================================================
def build_default_retriever() -> tuple[HybridRetriever, ScreeningResult]:
    """Carga, revisa e indexa los documentos del proyecto. Devuelve también la revisión, para el panel de alertas."""
    corpus = load_safe_corpus()
    return HybridRetriever(corpus.safe, OllamaEmbedder(), settings.chroma_path), corpus
