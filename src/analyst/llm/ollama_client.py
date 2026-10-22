"""
CLIENTE DE GENERACIÓN CON OLLAMA

Qué hace: envía un prompt al modelo y devuelve una salida estructurada y validada.
Con qué se conecta: lo usan el planificador y el redactor (a través de la interfaz StructuredLLM);
la comunicación HTTP pasa por ollama_http.py.
Idea clave: Ollama obliga a cumplir el esquema JSON al generar, así que la forma es correcta. Aun así
pueden pasar dos cosas: el modelo se corta por el límite de tokens, o llena la forma con valores que no
validan. En cada caso hay exactamente un reintento.
"""
import time
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from analyst.config import settings
from analyst.errors import LLMOutputError, OllamaError
from analyst.llm.schemas import json_schema
from analyst.ollama_http import post_json

T = TypeVar("T", bound=BaseModel)

# Aviso que se agrega al prompt en el reintento, diciendo qué falló.
RETRY_REMINDER = ("\n\nIMPORTANTE: tu respuesta anterior no cumplió el formato requerido ({problem}). "
                  "Responde solo con JSON válido que cumpla el esquema.")
# Si la salida se cortó por longitud, el reintento permite 1,5 veces más tokens.
LENGTH_RETRY_FACTOR = 1.5


# =====================================================================
# BLOQUE: Resultado y métricas de una generación
# =====================================================================
@dataclass
class GenerationInfo:
    """Datos de la generación: tiempo, tokens, reintentos y avisos."""

    seconds: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0
    retries: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class Generated(Generic[T]):
    """Valor ya validado junto con los datos de su generación."""

    value: T
    info: GenerationInfo


# =====================================================================
# BLOQUE: Cliente de chat
# =====================================================================
class OllamaChat:
    """Habla con /api/chat de Ollama. Temperatura 0 y semilla fija para respuestas reproducibles."""

    def __init__(self, model: str | None = None, base_url: str | None = None, timeout: float | None = None,
                 num_ctx: int | None = None, keep_alive: str | None = None,
                 temperature: float = 0.0, seed: int = 7, client: httpx.Client | None = None):
        self.model = model or settings.llm_model
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._timeout = timeout or settings.llm_timeout_seconds
        self.num_ctx = num_ctx or settings.llm_num_ctx
        self._keep_alive = keep_alive or settings.llm_keep_alive
        self._temperature, self._seed = temperature, seed
        self._client = client or httpx.Client()   # una sola conexión reutilizada entre peticiones

    @property
    def name(self) -> str:
        """Nombre del modelo."""
        return self.model

    def generate_structured(self, model_cls: type[T], system: str, user: str, max_tokens: int) -> Generated[T]:
        """Genera una salida que cumple `model_cls`; reintenta una vez si se cortó o no validó."""
        schema = json_schema(model_cls)
        info = GenerationInfo()
        tokens, prompt, problem = max_tokens, user, ""
        # Hasta dos intentos: el segundo corrige el problema del primero (más tokens o aviso de formato).
        for attempt in range(2):
            started = time.perf_counter()
            body = post_json(self._client, self._base_url, "/api/chat", self._payload(system, prompt, schema, tokens),
                             timeout=self._timeout, model=self.model, purpose="generar la respuesta")
            info.seconds += time.perf_counter() - started
            info.prompt_tokens += body.get("prompt_eval_count", 0)
            info.output_tokens += body.get("eval_count", 0)
            try:
                content = body["message"]["content"]
            except (KeyError, TypeError) as exc:
                raise OllamaError("Respuesta inesperada de Ollama: no trae 'message'.") from exc

            if body.get("done_reason") == "length":
                problem = f"la salida se cortó al llegar al límite de {tokens} tokens"
                tokens = int(tokens * LENGTH_RETRY_FACTOR)
            else:
                try:
                    return Generated(model_cls.model_validate_json(content), _finish(info, attempt))
                except ValidationError as exc:
                    problem = "JSON inválido o fuera del esquema: " + str(exc.errors()[0]["msg"])
            info.notes.append(f"Intento {attempt + 1} descartado: {problem}.")
            prompt = user + RETRY_REMINDER.format(problem=problem)
        raise LLMOutputError(
            f"El modelo '{self.model}' no devolvió una respuesta válida tras reintentar ({problem}). "
            "Vuelve a intentarlo; si se repite, prueba otro modelo con LLM_MODEL.")

    def warm_up(self) -> None:
        """Carga el modelo con el mismo tamaño de contexto de las peticiones reales, para que no se recargue después."""
        post_json(self._client, self._base_url, "/api/chat",
                  {"model": self.model, "stream": False, "keep_alive": self._keep_alive,
                   "options": {"num_ctx": self.num_ctx, "num_predict": 1},
                   "messages": [{"role": "user", "content": "hola"}]},
                  timeout=self._timeout, model=self.model, purpose="cargar el modelo")

    def _payload(self, system: str, user: str, schema: dict, max_tokens: int) -> dict:
        """Cuerpo de la petición: esquema, opciones de determinismo y los dos mensajes."""
        return {
            "model": self.model, "stream": False, "format": schema, "keep_alive": self._keep_alive,
            "options": {"temperature": self._temperature, "seed": self._seed,
                        "num_ctx": self.num_ctx, "num_predict": max_tokens},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }


def _finish(info: GenerationInfo, attempt: int) -> GenerationInfo:
    """Anota cuántos reintentos hizo falta."""
    info.retries = attempt
    return info
