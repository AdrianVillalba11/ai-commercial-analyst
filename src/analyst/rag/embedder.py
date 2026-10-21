"""Embeddings through Ollama (/api/embed). The default model is bge-m3, which needs no prefixes."""
import httpx

from analyst.config import settings
from analyst.errors import OllamaError
from analyst.ollama_http import post_json


class OllamaEmbedder:
    def __init__(self, model: str | None = None, base_url: str | None = None,
                 timeout: float | None = None, batch_size: int = 16,
                 document_prefix: str = "", query_prefix: str = "",
                 client: httpx.Client | None = None):
        self.model = model or settings.embedding_model
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._timeout = timeout or settings.llm_timeout_seconds
        self._batch_size = batch_size
        self._document_prefix, self._query_prefix = document_prefix, query_prefix
        self._client = client or httpx.Client()   # one connection reused across requests

    @property
    def name(self) -> str:
        return self.model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = [self._document_prefix + t for t in texts[start:start + self._batch_size]]
            vectors += self._embed(batch)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self._query_prefix + text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        body = post_json(self._client, self._base_url, "/api/embed", {"model": self.model, "input": texts},
                         timeout=self._timeout, model=self.model, purpose="generar embeddings")
        try:
            vectors = body["embeddings"]
        except (KeyError, TypeError) as exc:
            raise OllamaError("Respuesta inesperada de Ollama: no trae 'embeddings'.") from exc
        if len(vectors) != len(texts):
            raise OllamaError(f"Ollama devolvió {len(vectors)} embeddings para {len(texts)} textos.")
        return vectors
