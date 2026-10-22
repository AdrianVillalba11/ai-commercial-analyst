"""
API WEB Y PÁGINA

Qué hace: expone la aplicación con FastAPI: sirve la página (carpeta web/) y los endpoints que ella consume
(preguntar, estado de salud, calidad de datos, alertas de seguridad, evidencia).
Con qué se conecta: usa service.py; la lanza run.py. La página (web/app.js) consume estos endpoints.
Idea clave: se atiende UNA pregunta a la vez (Ollama genera de a una) y el servidor escucha solo en
127.0.0.1. Todo lo que muestra la página sale de aquí, y la página lo dibuja como texto plano.
"""
import json
import logging
import queue
import threading
from contextlib import asynccontextmanager
from typing import Callable, Iterator, Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from analyst.config import PROJECT_ROOT, settings
from analyst.errors import OllamaError
from analyst.service import AnalystService, build_default_service

# =====================================================================
# BLOQUE: Constantes y modelos de entrada
# =====================================================================
log = logging.getLogger("analyst")
WEB_DIR = PROJECT_ROOT / "web"
# Política de seguridad del navegador: solo recursos propios, sin scripts ni estilos externos.
CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; "
       "base-uri 'none'; frame-ancestors 'none'")


class AskRequest(BaseModel):
    """Cuerpo de una pregunta: el texto y, opcionalmente, el modo de búsqueda."""

    question: str
    mode: Literal["semantic", "hybrid", "lexical"] | None = None


# =====================================================================
# BLOQUE: Ayudantes
# =====================================================================
def check_ollama(base_url: str, chat_model: str, embedding_model: str, timeout: float = 2.0) -> dict:
    """¿Ollama está en ejecución y están descargados los dos modelos?"""
    def entry(name: str, available: bool) -> dict:
        return {"name": name, "available": available}

    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=timeout)
        installed = {m["name"].split(":")[0] for m in response.json()["models"]}
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return {"running": False, "chat_model": entry(chat_model, False), "embedding_model": entry(embedding_model, False)}
    return {"running": True,
            "chat_model": entry(chat_model, chat_model.split(":")[0] in installed),
            "embedding_model": entry(embedding_model, embedding_model.split(":")[0] in installed)}


def error_payload(exc: Exception) -> tuple[int, str]:
    """(código HTTP, mensaje para la persona). Los detalles internos se registran en consola y nunca se envían."""
    if isinstance(exc, ValueError):
        return 400, str(exc)
    if isinstance(exc, OllamaError):
        return 503, str(exc)
    log.exception("Error inesperado al responder una pregunta", exc_info=exc)
    return 500, "Ocurrió un error interno al responder. Revisa la consola donde se ejecuta la aplicación."


def sse(event: str, data: dict) -> str:
    """Da formato de evento SSE (Server-Sent Events) para enviar el progreso al navegador."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(service: AnalystService | None = None, *, warm_up: bool = True,
               service_factory: Callable[[], AnalystService] = build_default_service) -> FastAPI:
    """Crea la aplicación. Recibir el servicio por parámetro permite probarla con uno falso."""
    # Un solo candado: una pregunta a la vez.
    ask_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Al arrancar: construye el servicio y calienta los modelos en segundo plano (la página ya abre mientras tanto)."""
        if app.state.service is None:
            app.state.service = service_factory()
        app.state.warm = {"state": "pending", "detail": ""}
        if warm_up:
            threading.Thread(target=_warm, args=(app,), daemon=True).start()
        else:
            app.state.warm = {"state": "ready", "detail": ""}
        yield

    def _warm(app: FastAPI) -> None:
        """Calentamiento en segundo plano; el estado queda en app.state.warm para que la página lo muestre."""
        try:
            app.state.service.warm_up()
            app.state.warm = {"state": "ready", "detail": ""}
        except Exception as exc:       # noqa: BLE001 - la página muestra el motivo en vez de romper la app
            app.state.warm = {"state": "error", "detail": str(exc)}

    app = FastAPI(title="Analista comercial de Brío", lifespan=lifespan)
    app.state.service = service
    app.state.warm = {"state": "pending", "detail": ""}

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        """Agrega a cada respuesta las cabeceras de seguridad (y sin caché en /api/)."""
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def svc() -> AnalystService:
        """El servicio actual."""
        return app.state.service

    # =====================================================================
    # BLOQUE: Página
    # =====================================================================

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB_DIR / "index.html", media_type="text/html; charset=utf-8")

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    # =====================================================================
    # BLOQUE: Preguntar (respuesta completa y respuesta con progreso en vivo)
    # =====================================================================

    @app.post("/api/ask")
    def ask(body: AskRequest):
        try:
            with ask_lock:
                return svc().ask(body.question, mode=body.mode).to_dict()
        except Exception as exc:       # noqa: BLE001 - se traduce a un código HTTP y un mensaje para la persona
            status, detail = error_payload(exc)
            return JSONResponse({"detail": detail}, status_code=status)

    @app.post("/api/ask/stream")
    def ask_stream(body: AskRequest):
        """Igual que /api/ask, pero enviando el avance de cada paso mientras se trabaja (SSE)."""
        events: queue.Queue = queue.Queue()

        def work() -> None:
            """Hilo de trabajo: hace la pregunta y deja los eventos en la cola."""
            try:
                with ask_lock:
                    result = svc().ask(body.question, mode=body.mode, progress=lambda s: events.put(sse("step", {"step": s})))
                events.put(sse("result", result.to_dict()))
            except Exception as exc:   # noqa: BLE001
                status, detail = error_payload(exc)
                events.put(sse("error", {"status": status, "detail": detail}))
            events.put(None)

        threading.Thread(target=work, daemon=True).start()

        def stream() -> Iterator[str]:
            """Entrega los eventos al navegador a medida que aparecen, hasta recibir el final (None)."""
            while (item := events.get()) is not None:
                yield item

        return StreamingResponse(stream(), media_type="text/event-stream")

    # =====================================================================
    # BLOQUE: Información (estado, calidad de datos, seguridad, evidencia)
    # =====================================================================

    @app.get("/api/health")
    def health():
        """Estado para la cabecera de la página: Ollama, modelos, calentamiento, documentos y filas de ventas."""
        ollama = check_ollama(settings.ollama_base_url, settings.llm_model, settings.embedding_model)
        summary = svc().data_quality().get("summary", {}) if svc() else {}
        return {**ollama, "warm": app.state.warm, "retrieval_mode": settings.retrieval_mode,
                "documents": svc().document_count if svc() else 0, "sales_rows": summary.get("rows_out", 0)}

    @app.get("/api/quality")
    def quality():
        return svc().data_quality()

    @app.get("/api/security")
    def security():
        return {"alerts": svc().security_alerts()}

    @app.get("/api/evidence/{chunk_id}")
    def evidence(chunk_id: str):
        """Texto de un fragmento de evidencia citado en una respuesta."""
        chunk = svc().get_chunk(chunk_id)
        if chunk is None:
            raise HTTPException(status_code=404, detail="No existe un fragmento de evidencia con ese ID.")
        return {"chunk_id": chunk.chunk_id, "doc": chunk.doc, "doc_title": chunk.doc_title, "section": chunk.section,
                "page": chunk.page, "regions": chunk.regions, "kind": chunk.kind, "text": chunk.text}

    return app


# Instancia que usa run.py (uvicorn).
app = create_app()
