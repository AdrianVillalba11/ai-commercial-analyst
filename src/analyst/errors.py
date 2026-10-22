"""
ERRORES COMPARTIDOS

Qué hace: define las excepciones propias de la aplicación.
Con qué se conecta: las lanzan el cliente de Ollama, el planificador y el servicio; la API y la
línea de comandos las convierten en mensajes claros para quien usa la app.
"""


class OllamaError(Exception):
    """Ollama no responde, falta un modelo, o una petición falló o superó el tiempo.

    El mensaje está escrito para quien ejecuta la app e indica qué hacer.
    """


class LLMOutputError(OllamaError):
    """El modelo respondió, pero no en el formato exigido, ni siquiera tras el reintento."""
