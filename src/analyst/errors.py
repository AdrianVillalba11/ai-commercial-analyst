"""Errors shared across modules."""


class OllamaError(Exception):
    """Ollama is unreachable, a model is missing, or a request failed or timed out.

    The message is written for the person running the app and says what to do.
    """
