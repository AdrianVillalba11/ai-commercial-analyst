"""End-to-end tests with the real models (skipped when Ollama or the models are not available).

They check structure, not wording: a language model never writes the same sentence twice. What
the model says is checked by the guardrails (task 8), not here. The printed table shows how many
cited sources are not real ids, as input for those guardrails.
"""
import pytest

from analyst.errors import OllamaError
from analyst.service import build_default_service

pytestmark = pytest.mark.ollama

QUESTIONS = {
    "cuantitativa": "¿Cuál fue la región con mayor caída porcentual en ventas netas de Brío entre T1 y T2 de 2026? "
                    "Muestra los valores y el cálculo.",
    "integrada": "¿Qué hallazgos de consumidor y observaciones comerciales podrían ayudar a explicar el deterioro de Brío?",
    "contraste": "¿Son consistentes las principales conclusiones del reporte comercial con los datos de ventas?",
    "causalidad": "¿Podemos afirmar que las promociones de la competencia causaron la caída de ventas? "
                  "¿Qué evidencia lo sustenta o limita?",
}
MAX_SECONDS = 90


@pytest.fixture(scope="module")
def service():
    try:
        svc = build_default_service()
        svc.ask("¿Cuál es la capital de Francia?")      # warm-up and availability check of the chat model
    except OllamaError as exc:
        pytest.skip(f"Ollama o los modelos no están disponibles: {exc}")
    return svc


@pytest.fixture(scope="module")
def results(service):
    return {name: service.ask(question) for name, question in QUESTIONS.items()}


def known_ids(result):
    return {b.calc_id for b in result.calc} | {e.chunk_id for e in result.evidence}


def claims(result):
    a = result.answer
    return a.hechos + a.hipotesis + a.contradicciones


def test_every_reference_question_gets_a_structured_answer_in_reasonable_time(results):
    for name, result in results.items():
        assert result.answer.resumen.strip(), name
        assert claims(result) or result.answer.insuficiente, name
        assert all(c.fuentes for c in claims(result)), name
        assert result.timings["total"] < MAX_SECONDS, name
        assert result.calc and result.calc[0].result["rows"], name


def test_the_quantitative_answer_names_the_region_with_the_largest_drop(results):
    result = results["cuantitativa"]
    text = result.answer.resumen + " ".join(c.texto for c in result.answer.hechos)
    assert "Sur" in text
    assert result.calc[0].result["rows"][0]["keys"] == {"region": "Sur"}
    assert result.evidence == []                     # pure number question: no documents


def test_document_questions_use_documents_and_never_the_quarantined_note(results):
    for name in ("integrada", "contraste", "causalidad"):
        ids = {e.chunk_id for e in results[name].evidence}
        assert ids and "OC-EXT-06" not in ids, name


def test_the_causality_answer_does_not_claim_causality(results):
    text = results["causalidad"].answer.resumen.lower()
    assert "no se puede" in text or "no está demostrad" in text or "no es posible" in text or "no se ha" in text


def test_out_of_scope_question_is_answered_without_documents_or_calculations(service):
    result = service.ask("¿Cuál es la capital de Francia?")
    assert result.plan.tipo == "fuera_de_alcance" and result.calc == [] and result.evidence == []


def test_report_table(results, capsys):
    with capsys.disabled():
        print(f"\n{'pregunta':13} {'plan':18} {'plan s':>6} {'recup s':>7} {'redac s':>7} {'total s':>7}  fuentes inventadas")
        for name, r in results.items():
            valid = known_ids(r)
            cited = [s for c in claims(r) for s in c.fuentes]
            fake = [s for s in cited if s not in valid]
            t = r.timings
            print(f"{name:13} {r.plan.tipo:18} {t['planificacion']:6.1f} {t.get('recuperacion', 0):7.1f} "
                  f"{t['redaccion']:7.1f} {t['total']:7.1f}  {len(fake)}/{len(cited)} {fake}")
