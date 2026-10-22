"""
PRUEBA DE CALIDAD DE LA BÚSQUEDA CON EL MODELO REAL

Qué verifica: que la búsqueda encuentra la evidencia esperada usando bge-m3 de verdad (se salta sola si Ollama no está).
Regla fijada ANTES de medir: para cada pregunta de referencia, la búsqueda híbrida debe devolver al menos uno de los ids
esperados entre los 6 primeros. No se ajusta a los resultados.
"""
import pytest

from analyst.errors import OllamaError
from analyst.rag.embedder import OllamaEmbedder
from analyst.rag.evaluation import RETRIEVAL_CASES, TOP_K, VALIDATION_CASES, evaluate
from analyst.rag.retriever import HybridRetriever
from analyst.security.injection import load_safe_corpus

pytestmark = pytest.mark.ollama


@pytest.fixture(scope="module")
def retriever(tmp_path_factory):
    embedder = OllamaEmbedder()
    try:
        embedder.embed_query("comprobación")
    except OllamaError as exc:
        pytest.skip(f"Ollama o el modelo de embeddings no están disponibles: {exc}")
    return HybridRetriever(load_safe_corpus().safe, embedder, tmp_path_factory.mktemp("chroma"))


@pytest.fixture(scope="module")
def summaries(retriever):
    return evaluate(retriever)


@pytest.fixture(scope="module")
def validation(retriever):
    return evaluate(retriever, cases=VALIDATION_CASES)


def test_report_quality_table(summaries, validation, capsys):
    with capsys.disabled():
        for title, result in (("preguntas de ajuste", summaries), ("preguntas de validación", validation)):
            print(f"\n{title}\n{'modo':10} acierto@{TOP_K} recall  prec.  nDCG   MRR   top1")
            for mode, s in result.items():
                print(f"{mode:10} {s.hits:>2}/{len(s.cases)}     {s.mean_recall:.2f}   {s.mean_precision:.2f}   "
                      f"{s.mean_ndcg:.2f}  {s.mrr:.2f}  {s.top1_rate:.2f}")
            for c in [c for c in result["hybrid"].cases if not c.hit]:
                print(f"  FALLA híbrido: {c.question!r} -> {c.retrieved}")


def test_hybrid_returns_an_expected_id_for_every_reference_question(summaries):
    missed = [(c.question, c.retrieved) for c in summaries["hybrid"].cases if not c.hit]
    assert not missed, f"Preguntas sin ningún ID esperado en el top {TOP_K}: {missed}"


def test_hybrid_holds_up_on_questions_it_was_not_tuned_on(validation):
    """Umbral fijado antes de medir: al menos 11 de las 12 preguntas de validación obtienen un id esperado."""
    assert validation["hybrid"].hits >= 11


def test_the_quarantined_note_never_shows_up(summaries, validation):
    for summary in [*summaries.values(), *validation.values()]:
        assert all("OC-EXT-06" not in c.retrieved for c in summary.cases)


def test_there_are_fourteen_reference_questions():
    assert len(RETRIEVAL_CASES) == 14
