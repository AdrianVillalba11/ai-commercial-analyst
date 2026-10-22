"""
PRUEBAS DE LA API WEB

Qué verifica: api/main.py (página, preguntar, progreso paso a paso por SSE, estado, calidad, seguridad, evidencia, cabeceras
de seguridad) y vigilancias sobre el código de la propia página (sin HTML dinámico, solo tema claro, sin recursos externos,
y que la página solo lee campos que la API realmente devuelve).
Cómo: TestClient de FastAPI con un modelo falso, sin Ollama.
"""
import json
import re
import time

import pytest
from fastapi.testclient import TestClient

from analyst.analytics.metrics import SalesAnalyzer
from analyst.api import main as api
from analyst.api.main import WEB_DIR, create_app
from analyst.config import settings
from analyst.data.loader import load_clean_sales
from analyst.docs.parser import parse_reported_regional_variation
from analyst.errors import OllamaError
from analyst.llm.schemas import AnalysisSpec, Plan
from analyst.rag.retriever import HybridRetriever
from analyst.security.injection import load_safe_corpus
from analyst.service import AnalystService
from tests.helpers import FakeEmbedder, FakeLLM

CALC = AnalysisSpec(agrupar_por=["region"], metrica="venta_neta_cop", region="todas", canal="todos", presentacion="todas")
QUANTITATIVE = Plan(tipo="cuantitativa", analisis=[CALC], usar_documentos=False)
INTEGRATED = Plan(tipo="integrada", analisis=[CALC], usar_documentos=True)
CONTRAST = Plan(tipo="contraste_reporte", analisis=[CALC], usar_documentos=True)


@pytest.fixture(scope="module")
def cleaned():
    """Datos reales limpios."""
    return load_clean_sales()


@pytest.fixture(scope="module")
def corpus():
    """Corpus real de documentos ya revisado."""
    return load_safe_corpus()


@pytest.fixture(scope="module")
def reported():
    """Variaciones regionales que declara el reporte."""
    return parse_reported_regional_variation(settings.raw_data_dir / "05_Reporte_Comercial.pdf")


@pytest.fixture
# Fábrica de clientes de prueba con un modelo falso configurable.
def make_client(cleaned, corpus, reported, tmp_path):
    clients = []

    def build(llm, embedder=None, warm_up=False):
        retriever = HybridRetriever(corpus.safe, embedder or FakeEmbedder(), tmp_path)
        service = AnalystService(SalesAnalyzer(cleaned.data, cleaned.report), retriever, llm, cleaned.report, corpus,
                                 reported, corpus.safe)
        client = TestClient(create_app(service, warm_up=warm_up))
        client.__enter__()                      # ejecuta el arranque (y el hilo de calentamiento, si está activo)
        clients.append(client)
        return client

    yield build
    for client in clients:
        client.__exit__(None, None, None)


def frames(text: str) -> list[tuple[str, dict]]:
    """Convierte un cuerpo de eventos SSE en pares (evento, datos)."""
    result = []
    for block in text.strip().split("\n\n"):
        event, data = "message", ""
        for line in block.split("\n"):
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data += line[6:]
        result.append((event, json.loads(data)))
    return result


# =====================================================================
# BLOQUE: Página
# =====================================================================

def test_the_page_and_its_assets_are_served(make_client):
    client = make_client(FakeLLM(plan=QUANTITATIVE))
    page = client.get("/")
    assert page.status_code == 200 and "Analista comercial de Brío" in page.text
    assert '/static/app.js' in page.text and '/static/style.css' in page.text
    assert client.get("/static/app.js").status_code == 200 and client.get("/static/style.css").status_code == 200


def test_responses_carry_security_headers(make_client):
    headers = make_client(FakeLLM(plan=QUANTITATIVE)).get("/").headers
    assert "default-src 'self'" in headers["content-security-policy"] and "script-src 'self'" in headers["content-security-policy"]
    assert headers["x-content-type-options"] == "nosniff"
    assert make_client(FakeLLM(plan=QUANTITATIVE)).get("/api/quality").headers["cache-control"] == "no-store"


# =====================================================================
# BLOQUE: Preguntar
# =====================================================================

def test_ask_returns_the_checked_answer_with_its_trace(make_client):
    data = make_client(FakeLLM(plan=QUANTITATIVE)).post("/api/ask", json={"question": "¿Cuál región cayó más?"}).json()
    assert data["final"]["hechos"][0]["texto"].startswith("Mayor caída de ventas netas (COP) por región: Sur")
    assert {"plan", "calc", "evidence", "answer", "final", "timings", "warnings", "mode", "model"} <= set(data)
    assert data["calc"][0]["result"]["rows"][0]["keys"] == {"region": "Sur"}


def test_ask_accepts_a_search_mode(make_client):
    client = make_client(FakeLLM(plan=INTEGRATED))
    assert client.post("/api/ask", json={"question": "faltantes en el Sur", "mode": "lexical"}).json()["mode"] == "lexical"
    assert client.post("/api/ask", json={"question": "x", "mode": "inventado"}).status_code == 422


@pytest.mark.parametrize("question,message", [("   ", "vacía"), ("x" * 1001, "demasiado larga")])
def test_invalid_questions_get_a_clear_400(make_client, question, message):
    response = make_client(FakeLLM(plan=QUANTITATIVE)).post("/api/ask", json={"question": question})
    assert response.status_code == 400 and message in response.json()["detail"]


def test_ollama_down_is_a_503_that_says_what_to_do(make_client):
    llm = FakeLLM(plan_error=OllamaError("No se pudo conectar con Ollama. Inícialo con `ollama serve`."))
    response = make_client(llm).post("/api/ask", json={"question": "¿Cómo cayó Brío?"})
    assert response.status_code == 503 and "ollama serve" in response.json()["detail"]


def test_an_unexpected_error_is_a_500_without_internal_details(make_client):
    llm = FakeLLM(plan_error=RuntimeError("secret internal path C:/x/y"))
    response = make_client(llm).post("/api/ask", json={"question": "¿Cómo cayó Brío?"})
    assert response.status_code == 500 and "secret" not in response.text and "interno" in response.json()["detail"]


# =====================================================================
# BLOQUE: Progreso en vivo (SSE)
# =====================================================================

def test_stream_announces_each_step_then_the_result(make_client):
    response = make_client(FakeLLM(plan=QUANTITATIVE)).post("/api/ask/stream", json={"question": "¿Cuál región cayó más?"})
    assert response.headers["content-type"].startswith("text/event-stream")
    events = frames(response.text)
    assert [d["step"] for e, d in events if e == "step"] == ["planificacion", "calculo", "redaccion", "validacion"]
    assert events[-1][0] == "result" and events[-1][1]["final"]["hechos"]


def test_stream_includes_the_search_step_when_documents_are_used(make_client):
    events = frames(make_client(FakeLLM(plan=INTEGRATED)).post("/api/ask/stream", json={"question": "faltantes en el Sur"}).text)
    assert [d["step"] for e, d in events if e == "step"] == ["planificacion", "calculo", "recuperacion", "redaccion", "validacion"]


def test_stream_reports_errors_as_an_event(make_client):
    client = make_client(FakeLLM(plan_error=OllamaError("Ollama apagado")))
    events = frames(client.post("/api/ask/stream", json={"question": "¿Cómo cayó Brío?"}).text)
    assert events[-1] == ("error", {"status": 503, "detail": "Ollama apagado"})
    assert frames(client.post("/api/ask/stream", json={"question": " "}).text)[-1][1]["status"] == 400


def test_stream_returns_the_report_contrast(make_client):
    events = frames(make_client(FakeLLM(plan=CONTRAST)).post("/api/ask/stream", json={"question": "¿Es consistente el reporte?"}).text)
    result = events[-1][1]
    assert result["contrast"]["consistent"] is False and result["final"]["resumen"].startswith("El reporte NO es consistente")


# =====================================================================
# BLOQUE: Información (salud, calidad, seguridad, evidencia)
# =====================================================================

def test_health_reports_ollama_models_and_warm_up(make_client, monkeypatch):
    monkeypatch.setattr(api, "check_ollama", lambda *a, **k: {
        "running": True, "chat_model": {"name": "llama3", "available": True},
        "embedding_model": {"name": "bge-m3", "available": False}})
    health = make_client(FakeLLM(plan=QUANTITATIVE)).get("/api/health").json()
    assert health["running"] is True and health["embedding_model"]["available"] is False
    assert health["warm"]["state"] == "ready" and health["documents"] > 50 and health["sales_rows"] == 3994
    assert health["retrieval_mode"] == settings.retrieval_mode


def test_check_ollama_when_it_is_not_running():
    result = api.check_ollama("http://127.0.0.1:1", "llama3", "bge-m3", timeout=0.5)
    assert result["running"] is False and result["chat_model"]["available"] is False


def test_warm_up_loads_the_models_in_the_background(make_client):
    llm = FakeLLM(plan=QUANTITATIVE)
    llm.warmed = False
    llm.warm_up = lambda: setattr(llm, "warmed", True)
    client = make_client(llm, warm_up=True)
    deadline = time.time() + 5
    while client.get("/api/health").json()["warm"]["state"] == "pending" and time.time() < deadline:
        time.sleep(0.05)
    assert client.get("/api/health").json()["warm"] == {"state": "ready", "detail": ""} and llm.warmed


def test_a_failed_warm_up_is_reported_instead_of_breaking_the_app(make_client):
    client = make_client(FakeLLM(plan=QUANTITATIVE), embedder=FakeEmbedder(fail=True), warm_up=True)
    deadline = time.time() + 5
    while client.get("/api/health").json()["warm"]["state"] == "pending" and time.time() < deadline:
        time.sleep(0.05)
    warm = client.get("/api/health").json()["warm"]
    assert warm["state"] == "error" and "Ollama apagado" in warm["detail"]


def test_quality_and_security_endpoints(make_client):
    client = make_client(FakeLLM(plan=QUANTITATIVE))
    quality = client.get("/api/quality").json()
    assert quality["summary"]["rows_in"] == 4000 and quality["summary"]["rows_out"] == 3994
    alerts = client.get("/api/security").json()["alerts"]
    assert [a["chunk_id"] for a in alerts] == ["OC-EXT-06"] and alerts[0]["status"] == "cuarentena"
    assert "INSTRUCCIÓN" in alerts[0]["excerpt"] and alerts[0]["reasons"]


def test_evidence_endpoint_serves_citable_fragments_but_never_the_quarantined_one(make_client):
    client = make_client(FakeLLM(plan=QUANTITATIVE))
    ok = client.get("/api/evidence/INV-01").json()
    assert ok["doc_title"] == "Investigación de consumidor" and "Cinco de los ocho" in ok["text"] and ok["page"] == 1
    assert client.get("/api/evidence/OC-EXT-06").status_code == 404        # en cuarentena
    assert client.get("/api/evidence/INVENTADO-1").status_code == 404
    assert client.get("/api/evidence/CALC-1").status_code == 404


# =====================================================================
# BLOQUE: Añadidos al servicio
# =====================================================================

def test_service_reports_each_step_to_the_progress_callback(make_client):
    client = make_client(FakeLLM(plan=INTEGRATED))
    steps = []
    client.app.state.service.ask("faltantes en el Sur", progress=steps.append)
    assert steps == ["planificacion", "calculo", "recuperacion", "redaccion", "validacion"]


def test_service_serves_fragments_only_from_the_screened_corpus(make_client):
    service = make_client(FakeLLM(plan=QUANTITATIVE)).app.state.service
    assert service.get_chunk("INV-01").chunk_id == "INV-01" and service.get_chunk("OC-EXT-06") is None


# =====================================================================
# BLOQUE: Vigilancias sobre el código de la página
# =====================================================================

def read(name: str) -> str:
    """Contenido de un archivo de la carpeta web/."""
    return (WEB_DIR / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("forbidden", ["innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"])
def test_the_page_never_builds_html_from_strings(forbidden):
    assert forbidden not in read("app.js")


def test_the_page_is_light_only():
    assert "prefers-color-scheme" not in read("style.css") and "dark" not in read("style.css").lower()
    assert "color-scheme: light" in read("style.css")
    assert '<meta name="color-scheme" content="light">' in read("index.html")


@pytest.mark.parametrize("name", ["index.html", "app.js", "style.css"])
def test_the_page_loads_nothing_from_the_internet(name):
    assert not re.search(r"https?://", read(name))          # funciona sin internet


def test_the_page_reads_only_fields_the_health_endpoint_returns(make_client):
    """La página llegó a leer `ollama_running` mientras la API devolvía `running`: mostraba Ollama apagado estando encendido."""
    health = make_client(FakeLLM(plan=QUANTITATIVE)).get("/api/health").json()
    used = set(re.findall(r"\bh\.([a-z_]+)", read("app.js")))
    assert used and used <= set(health), used - set(health)
