"""Embeddings through Ollama (/api/embed). The default model is bge-m3, which needs no prefixes."""
import httpx

from analyst.config import settings
from analyst.errors import OllamaError


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
        url = f"{self._base_url}/api/embed"
        try:
            response = self._client.post(url, json={"model": self.model, "input": texts}, timeout=self._timeout)
        except httpx.ConnectError as exc:
            raise OllamaError(
                f"No se pudo conectar con Ollama en {self._base_url}. "
                "¿Está en ejecución? Inícialo con `ollama serve` (o abre la aplicación de Ollama)."
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaError(
                f"Ollama no respondió en {self._timeout:.0f} s al generar embeddings con '{self.model}'. "
                "La primera petición tarda más porque carga el modelo; vuelve a intentarlo."
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaError(f"Error de red al hablar con Ollama: {exc}") from exc
        if response.status_code == 404:
            raise OllamaError(f"El modelo de embeddings '{self.model}' no está descargado. Ejecuta: ollama pull {self.model}")
        if response.status_code != 200:
            raise OllamaError(f"Ollama respondió con error {response.status_code}: {response.text[:200]}")
        try:
            vectors = response.json()["embeddings"]
        except (ValueError, KeyError) as exc:
            raise OllamaError("Respuesta inesperada de Ollama: no trae 'embeddings'.") from exc
        if len(vectors) != len(texts):
            raise OllamaError(f"Ollama devolvió {len(vectors)} embeddings para {len(texts)} textos.")
        return vectors
