"""
COMUNICACIÓN HTTP CON OLLAMA

Qué hace: una única función para enviar peticiones a Ollama y traducir cualquier fallo a un
mensaje claro en español.
Con qué se conecta: la usan el cliente de generación (llm/ollama_client.py) y el de
embeddings (rag/embedder.py).
Idea clave: así ambos informan los errores exactamente igual.
"""
import httpx

from analyst.errors import OllamaError


def post_json(client: httpx.Client, base_url: str, path: str, payload: dict, *,
              timeout: float, model: str, purpose: str) -> dict:
    """Envía un POST a Ollama y devuelve el JSON. Todo fallo se convierte en un OllamaError que dice qué hacer."""
    try:
        response = client.post(f"{base_url}{path}", json=payload, timeout=timeout)
    except httpx.ConnectError as exc:
        raise OllamaError(
            f"No se pudo conectar con Ollama en {base_url}. "
            "¿Está en ejecución? Inícialo con `ollama serve` (o abre la aplicación de Ollama)."
        ) from exc
    except httpx.TimeoutException as exc:
        raise OllamaError(
            f"Ollama no respondió en {timeout:.0f} s al {purpose} con '{model}'. "
            "La primera petición tarda más porque carga el modelo; vuelve a intentarlo."
        ) from exc
    except httpx.HTTPError as exc:
        raise OllamaError(f"Error de red al hablar con Ollama: {exc}") from exc
    if response.status_code == 404:
        raise OllamaError(f"El modelo '{model}' no está descargado. Ejecuta: ollama pull {model}")
    if response.status_code != 200:
        raise OllamaError(f"Ollama respondió con error {response.status_code}: {response.text[:200]}")
    try:
        return response.json()
    except ValueError as exc:
        raise OllamaError("Ollama devolvió una respuesta que no es JSON.") from exc
