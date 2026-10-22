"""
EMBEDDINGS CON OLLAMA

Qué hace: convierte textos en vectores numéricos llamando a Ollama (/api/embed).
Con qué se conecta: lo usa rag/semantic.py para indexar fragmentos y para vectorizar la pregunta.
Idea clave: el modelo por defecto (bge-m3) no necesita prefijos de "documento"/"consulta"; los
parámetros existen por si se prueba otro modelo que sí los pida.
"""
import httpx

from analyst.config import settings
from analyst.errors import OllamaError
from analyst.ollama_http import post_json


class OllamaEmbedder:
    """Cliente de embeddings: procesa los documentos por lotes y la pregunta de una vez."""

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 timeout: float | None = None, batch_size: int = 16,
                 document_prefix: str = "", query_prefix: str = "",
                 client: httpx.Client | None = None):
        self.model = model or settings.embedding_model
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._timeout = timeout or settings.llm_timeout_seconds
        self._batch_size = batch_size
        self._document_prefix, self._query_prefix = document_prefix, query_prefix
        self._client = client or httpx.Client()   # una sola conexión reutilizada entre peticiones

    @property
    def name(self) -> str:
        """Nombre del modelo; sirve para saber si el índice guardado sigue siendo válido."""
        return self.model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Vectoriza muchos textos, en lotes para no saturar a Ollama."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = [self._document_prefix + t for t in texts[start:start + self._batch_size]]
            vectors += self._embed(batch)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Vectoriza la pregunta del usuario."""
        return self._embed([self._query_prefix + text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Una llamada a Ollama; comprueba que devuelva un vector por cada texto."""
        body = post_json(self._client, self._base_url, "/api/embed", {"model": self.model, "input": texts, "keep_alive": settings.llm_keep_alive},
                         timeout=self._timeout, model=self.model, purpose="generar embeddings")
        try:
            vectors = body["embeddings"]
        except (KeyError, TypeError) as exc:
            raise OllamaError("Respuesta inesperada de Ollama: no trae 'embeddings'.") from exc
        if len(vectors) != len(texts):
            raise OllamaError(f"Ollama devolvió {len(vectors)} embeddings para {len(texts)} textos.")
        return vectors
