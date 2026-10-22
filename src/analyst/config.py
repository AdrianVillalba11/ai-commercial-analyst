"""
CONFIGURACIÓN CENTRAL

Qué hace: reúne todos los ajustes de la app en un único objeto (`settings`).
Con qué se conecta: lo importan casi todos los módulos. Los valores salen de variables de entorno,
que se pueden definir en el archivo .env (ver .env.example).
Idea clave: cada ajuste tiene un valor por defecto que funciona sin tocar nada.
"""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Carpeta raíz del proyecto: src/analyst/config.py -> dos niveles arriba.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Ajustes de la app. Es inmutable (frozen) para que nadie los cambie en ejecución."""
    # --- Conexión con Ollama y modelos ---
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    llm_model: str = os.getenv("LLM_MODEL", "llama3")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "bge-m3")
    # --- Límites del modelo de generación ---
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
    llm_num_ctx: int = int(os.getenv("LLM_NUM_CTX", "4096"))
    llm_keep_alive: str = os.getenv("LLM_KEEP_ALIVE", "30m")
    # --- Búsqueda de evidencia y rutas de datos ---
    retrieval_mode: str = os.getenv("RETRIEVAL_MODE", "semantic")   # semantic | hybrid | lexical
    chroma_path: Path = PROJECT_ROOT / os.getenv("CHROMA_PATH", ".cache/chroma")
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"


# Instancia única que importa el resto del código.
settings = Settings()
