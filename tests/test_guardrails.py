"""Tests for the guardrails: facts written by code, the report-vs-data contrast and the validator.

Many tests replay failures seen in real runs of the model (a hypothesis citing only a calculation,
"no contradictions" when the report is wrong, a claim attributed to the wrong region, an invented
source) and check that the final answer no longer contains them.
"""
import io

import pytest

from analyst.analytics.metrics import SalesAnalyzer
from analyst.cli import main
from analyst.config import settings
from analyst.data.loader import load_clean_sales
from analyst.docs.parser import parse_reported_regional_variation
from analyst.guardrails.facts import calc_facts
from analyst.guardrails.models import FinalAnswer
from analyst.guardrails.report_check import CONTRAST_ID, compare_report_to_data
from analyst.guardrails.validator import ValidationContext, figures_supported, validate_answer
from analyst.llm.prompts import CalcBlock, build_calc_blocks
from analyst.llm.schemas import Answer, AnalysisSpec, Plan
from analyst.rag.retriever import HybridRetriever
from analyst.security.injection import load_safe_corpus
from analyst.service import AnalystService
from tests.helpers import FakeEmbedder, FakeLLM


# ------------------------------------------------------------------ fixtures

@pytest.fixture(scope="module")
def cleaned():
    return load_clean_sales()


@pytest.fixture(scope="module")
def analyzer(cleaned):
    return SalesAnalyzer(cleaned.data, cleaned.report)


@pytest.fixture(scope="module")
def corpus():
    return load_safe_corpus()


@pytest.fixture(scope="module")
def chunks(corpus):
    return {c.chunk_id: c for c in corpus.safe}


@pytest.fixture(scope="module")
def reported():
    return parse_reported_regional_variation(settings.raw_data_dir / "05_Reporte_Comercial.pdf")


@pytest.fixture(scope="module")
def contrast(analyzer, reported, corpus):
    return compare_report_to_data(analyzer, reported, corpus.safe)


def spec(**kw) -> AnalysisSpec:
    base = {"agrupar_por": ["region"], "metrica": "venta_neta_cop", "region": "todas", "canal": "todos",
            "presentacion": "todas"}
    return AnalysisSpec(**{**base, **kw})


def region_blocks(analyzer):
    return build_calc_blocks(analyzer, Plan(tipo="cuantitativa", analisis=[spec()], usar_documentos=False))


def answer(resumen="Resumen.", hechos=(), hipotesis=(), contradicciones=(), insuficiente=()) -> Answer:
    claims = lambda items: [{"texto": t, "fuentes": s} for t, s in items]
    return Answer.model_validate({"resumen": resumen, "hechos": claims(hechos), "hipotesis": claims(hipotesis),
                                  "contradicciones": claims(contradicciones),
                                  "insuficiente": [{"texto": t} for t in insuficiente]})


def context(analyzer, chunks, ids=(), plan_type="cuantitativa", contrast=None) -> ValidationContext:
    blocks = region_blocks(analyzer)
    if contrast:
        blocks.append(CalcBlock(CONTRAST_ID, contrast.title, contrast.text, contrast.to_dict(), {}))
    return ValidationContext(blocks, [chunks[i] for i in ids], plan_type, contrast, tuple(analyzer.columns))


# ------------------------------------------------------------------- figures

@pytest.mark.parametrize("claim,source", [
    ("Sur cayó -30,27 %", "Sur: variación -110.027.000 (-30,27 %)"),
    ("Sur cayó 30.27 %", "(-30,27 %)"),                                       # decimal point instead of comma
    ("Pasó de 363.431.600 a 253.404.600", "T1 363.431.600 | T2 253.404.600"),
    ("Cayó cerca de un 30,3 %", "(-30,27 %)"),                                  # rounded by the model
    ("Cayó 30 %", "(-30,27 %)"),
    ("La cifra CALC-1 de 2026 abarca 5 regiones en T1 y T2", "nada"),           # ids, years, single digits
    ("Creció 4,65 %", "(-4,65 %)"),                                             # sign is ignored
])
def test_supported_figures_are_accepted(claim, source):
    assert figures_supported(claim, [source]) == []


def test_invented_figures_are_found():
    assert figures_supported("Sur cayó 35,1 % y 21 millones", ["(-30,27 %)"]) == ["35.1", "21"]


# --------------------------------------------------------------------- facts

def test_facts_are_complete_sentences_written_from_the_calculation(analyzer):
    facts = calc_facts(region_blocks(analyzer))
    assert facts[0].texto == ("Mayor caída de ventas netas (COP) por región: Sur, -30,27 % "
                              "(T1 363.431.600 → T2 253.404.600). Explica el 73,1 % de la caída total de Brío.")
    assert facts[1].texto == "Crecieron: Oriente +4,65 %."
    assert facts[2].texto == ("Total Brío (ventas netas (COP)): -10,50 % (T1 1.433.250.000 → T2 1.282.713.700).")
    assert all(f.fuentes == ["CALC-1"] and f.origen == "calculo" for f in facts)


def test_filtered_facts_do_not_pass_a_partial_total_off_as_the_brand_total(analyzer):
    blocks = build_calc_blocks(analyzer, Plan(tipo="cuantitativa", usar_documentos=False,
                                              analisis=[spec(region="Sur")]))
    texts = [f.texto for f in calc_facts(blocks)]
    assert not any("100,0 %" in t for t in texts)                       # a single row explains "100 %" trivially
    assert texts[-1].startswith("Total Brío con filtro region = Sur (ventas netas (COP)): -30,27 %")


def test_the_share_of_a_filtered_total_says_so(analyzer):
    blocks = build_calc_blocks(analyzer, Plan(tipo="cuantitativa", usar_documentos=False,
                                              analisis=[spec(canal="Tradicional")]))
    assert "de la caída del total filtrado (canal = Tradicional)" in calc_facts(blocks)[0].texto


def test_facts_carry_the_filters(analyzer):
    blocks = build_calc_blocks(analyzer, Plan(tipo="cuantitativa", usar_documentos=False,
                                              analisis=[spec(agrupar_por=["canal"], region="Sur")]))
    assert "por canal (region = Sur)" in calc_facts(blocks)[0].texto


# ------------------------------------------------------------------ contrast

def test_contrast_with_the_real_report(contrast):
    by_region = {r.region: r for r in contrast.rows}
    assert {r for r, v in by_region.items() if not v.coincide} == {"Centro", "Sur"}
    assert {r for r, v in by_region.items() if v.coincide} == {"Norte", "Oriente", "Occidente"}
    assert by_region["Centro"].diff_pp == pytest.approx(-13.97, abs=0.01)
    assert by_region["Sur"].diff_pp == pytest.approx(23.27, abs=0.01)
    assert (contrast.reported_top, contrast.computed_top) == ("Centro", "Sur")
    assert contrast.consistent is False


def test_contrast_checks_the_reports_own_claims(contrast):
    states = {c.chunk_id: c.estado for c in contrast.claims}
    assert states == {"REP-01": "contradicha", "REP-02": "no verificable"}


def test_contrast_verdict_and_findings_are_written_by_code(contrast):
    assert contrast.verdict == ("El reporte NO es consistente con los datos: 2 de 5 regiones difieren más de 1 pp "
                                "(Centro, Sur) y señala a Centro como la región de mayor caída, cuando los datos "
                                "indican Sur (-30,27 %).")
    assert len(contrast.contradictions) == 3
    assert contrast.contradictions[0].texto == ("Según el reporte, Centro varió -23 %, pero los datos dan -9,03 % "
                                                "(diferencia de -13,97 pp).")
    assert contrast.contradictions[0].fuentes == [CONTRAST_ID, "REP-REGION-CENTRO"]
    assert "REP-01" in contrast.contradictions[2].texto and "Sur (-30,27 %)" in contrast.contradictions[2].texto
    assert "no contiene variables de promoción ni de precio" in contrast.unverifiable[0].texto
    assert contrast.agreement.texto == "Coinciden con los datos (±1 pp): Norte, Oriente, Occidente."


def test_contrast_block_given_to_the_model_states_the_verdict(contrast):
    assert contrast.text.startswith("[CONTRASTE-1]") and "NO coincide" in contrast.text
    assert "Afirmación REP-01: CONTRADICHA" in contrast.text and "Veredicto: El reporte NO es consistente" in contrast.text


def test_contrast_tolerance_is_a_parameter(analyzer, reported, corpus):
    wide = compare_report_to_data(analyzer, reported, corpus.safe, tolerance_pp=15)
    assert {r.region for r in wide.rows if not r.coincide} == {"Sur"}


def test_a_report_that_matches_the_data_is_consistent(analyzer, corpus):
    computed = {r.keys["region"]: round(r.pct_change) for r in analyzer.compare_periods("Brio", group_by="region").rows}
    result = compare_report_to_data(analyzer, computed, [])
    assert result.consistent and result.verdict.startswith("El reporte es consistente") and result.contradictions == []


# ----------------------------------------------- validator: failures replayed

def test_replayed_run_a_hypothesis_citing_only_a_calculation_is_discarded(analyzer, chunks):
    bad = answer("La región con mayor caída fue el Sur, con una variación de -30,27 %.",
                 hechos=[("-110.027.000 (-30,27 %)", ["CALC-1"])],
                 hipotesis=[("La caída en el Sur puede estar relacionada con la competencia en el mercado local.", ["CALC-1"])])
    final = validate_answer(bad, context(analyzer, chunks))
    assert final.hipotesis == []
    assert [d.seccion for d in final.descartes] == ["hipotesis"]
    assert "no cita ningún documento" in final.descartes[0].motivo
    assert final.hechos[0].origen == "calculo" and "Sur" in final.hechos[0].texto      # code's fact, with its subject
    assert all(h.origen == "calculo" for h in final.hechos)                               # the model's version is redundant
    assert final.resumen.startswith("La región con mayor caída fue el Sur") and final.resumen_origen == "modelo"


def test_replayed_run_a_claim_attributed_to_the_wrong_region_is_discarded(analyzer, chunks):
    bad = answer(hipotesis=[("La región Sur podría ser la más afectada, según la supervisora.", ["OC-CEN-02"])])
    final = validate_answer(bad, context(analyzer, chunks, ["OC-CEN-02"]))
    assert final.hipotesis == []
    assert "menciona Sur, pero la fuente citada no lo respalda (OC-CEN-02 dice de Centro)" in final.descartes[0].motivo
    assert any("Se eliminó la fuente OC-CEN-02" in a for a in final.avisos)


def test_replayed_run_an_invented_source_is_removed(analyzer, chunks):
    ctx = context(analyzer, chunks, ["INV-01"])
    only_fake = validate_answer(answer(hipotesis=[("El precio influyó en Sur.", ["EVID-f6a465f8"])]), ctx)
    assert only_fake.hipotesis == [] and "no cita ninguna fuente válida" in only_fake.descartes[0].motivo
    mixed = validate_answer(answer(hipotesis=[("El precio podría influir en Sur.", ["EVID-f6a465f8", "INV-01"])]), ctx)
    assert mixed.hipotesis[0].fuentes == ["INV-01"]
    assert any("EVID-f6a465f8" in a for a in mixed.avisos) and mixed.descartes == []


def test_replayed_run_the_code_decides_the_contrast_whatever_the_model_says(analyzer, chunks, contrast):
    bad = answer("Sí, el reporte es consistente con los datos.",
                 contradicciones=[("No se encontraron contradicciones entre fuentes.", ["CALC-1"])])
    final = validate_answer(bad, context(analyzer, chunks, plan_type="contraste_reporte", contrast=contrast))
    assert final.resumen == contrast.verdict and final.resumen_origen == "contraste"
    assert [c.origen for c in final.contradicciones] == ["contraste"] * 3        # the model's own add nothing
    assert final.descartes == []                                                   # skipped as redundant, not an error


def test_saying_there_are_no_contradictions_is_discarded_when_the_contrast_found_them(analyzer, chunks, contrast):
    ctx = context(analyzer, chunks, plan_type="cuantitativa", contrast=contrast)
    final = validate_answer(answer(insuficiente=["No hay contradicciones entre el reporte y los datos."]), ctx)
    assert final.descartes[0].seccion == "insuficiente" and "no hay contradicciones" in final.descartes[0].motivo


def test_replayed_run_a_false_absence_of_data_is_discarded(analyzer, chunks):
    bad = answer(insuficiente=["No se puede determinar la causa en Occidente y Norte, ya que no hay datos disponibles "
                               "para estas regiones."])
    final = validate_answer(bad, context(analyzer, chunks))
    assert final.insuficiente == [] and "no hay datos de norte, occidente" in final.descartes[0].motivo.lower()


def test_an_absence_of_consumer_data_is_true_and_kept(analyzer, chunks):
    ok = answer(insuficiente=["No hay datos de consumidor para Oriente ni Occidente."])
    assert len(validate_answer(ok, context(analyzer, chunks)).insuficiente) == 1


# --------------------------------------------------- validator: the rules

def test_a_figure_not_in_the_cited_source_is_discarded(analyzer, chunks):
    ctx = context(analyzer, chunks, ["INV-01"])
    wrong = validate_answer(answer(hechos=[("El 45 % de los entrevistados en Sur evalúa alternativas.", ["INV-01"])]), ctx)
    assert wrong.descartes[0].motivo == "la cifra 45 no aparece en las fuentes que cita"
    right = validate_answer(answer(hechos=[("Cinco de los ocho entrevistados en Sur evalúan alternativas.", ["INV-01"])]), ctx)
    assert right.descartes == [] and right.hechos[-1].origen == "documento"


def test_causal_wording_needs_a_hedge_in_facts(analyzer, chunks):
    ctx = context(analyzer, chunks, ["INV-04"])
    bad = validate_answer(answer(hechos=[("Las promociones causaron la caída de ventas.", ["INV-04"])]), ctx)
    assert "afirma una causa" in bad.descartes[0].motivo
    good = validate_answer(answer(hechos=[("No se puede afirmar que las promociones causaron la caída.", ["INV-04"])]), ctx)
    assert good.descartes == []


def test_a_report_claim_is_not_a_fact(analyzer, chunks):
    final = validate_answer(answer(hechos=[("Las promociones explican la contracción de ventas.", ["REP-02"])]),
                            context(analyzer, chunks, ["REP-02"]))
    assert "afirmación del reporte" in final.descartes[0].motivo


def test_a_hypothesis_with_a_matching_document_is_kept(analyzer, chunks):
    final = validate_answer(answer(hipotesis=[("La sensibilidad al precio en Sur podría influir.", ["INV-01"])]),
                            context(analyzer, chunks, ["INV-01"]))
    assert final.hipotesis[0].fuentes == ["INV-01"] and final.hipotesis[0].origen == "modelo" and final.descartes == []


@pytest.mark.parametrize("summary", [
    "Centro tuvo la mayor caída de ventas.",                 # an injected conclusion
    "La región con mayor caída fue Oriente.",
    "El Sur cayó -35,1 %.",                                  # invented figure
    "Las promociones causaron la caída.",                    # causal, no hedge
    "La caída en el Sur se atribuye a la falta de producto, según los comerciantes.",
])
def test_a_summary_that_contradicts_the_calculations_is_replaced(analyzer, chunks, summary):
    final = validate_answer(answer(summary), context(analyzer, chunks))
    assert final.resumen_origen == "calculo" and final.resumen.startswith("Mayor caída de ventas netas (COP) por región: Sur")
    assert final.descartes[0].seccion == "resumen" and final.descartes[0].texto == summary


@pytest.mark.parametrize("summary", [
    "El Sur tuvo la mayor caída (-30,27 %), seguido de Centro.",
    "Según el reporte, Centro es la mayor caída.",           # about the report: checked by the contrast, not here
    "No se puede afirmar que las promociones causaron la caída.",
])
def test_a_faithful_summary_is_kept(analyzer, chunks, summary):
    final = validate_answer(answer(summary), context(analyzer, chunks))
    assert final.resumen == summary and final.resumen_origen == "modelo" and final.descartes == []


def test_an_empty_summary_is_replaced(analyzer, chunks):
    final = validate_answer(answer("  "), context(analyzer, chunks))
    assert final.resumen_origen == "calculo" and final.descartes[0].motivo == "el modelo no escribió un resumen"


def test_causality_questions_always_state_what_the_data_cannot_show(analyzer, chunks):
    final = validate_answer(answer("La caída se atribuye a la competencia, pero no se puede demostrar causalidad."),
                            context(analyzer, chunks, ["INV-04", "INV-LIM-02", "REP-02"], "causalidad"))
    assert final.resumen_origen == "calculo" and final.resumen.startswith("No se puede afirmar causalidad. La base de ventas")
    first, second = final.insuficiente[:2]
    assert "no contiene variables de promoción ni de precio" in first.texto and first.origen == "calculo"
    assert "no establece causalidad" in second.texto and {"INV-04", "INV-LIM-02"} <= set(second.fuentes)


def test_contrast_answers_start_from_the_codes_findings(analyzer, chunks, contrast):
    final = validate_answer(answer("lo que sea"), context(analyzer, chunks, plan_type="contraste_reporte", contrast=contrast))
    assert final.resumen == contrast.verdict and final.resumen_origen == "contraste"
    assert final.hechos[-1].texto.startswith("Coinciden con los datos") and "no es verificable" in final.insuficiente[0].texto


def test_unchecked_answer_for_out_of_scope_questions():
    unchecked = FinalAnswer.unchecked(answer("Fuera de alcance.", insuficiente=["No trata de Brío."]))
    assert unchecked.resumen == "Fuera de alcance." and unchecked.insuficiente[0].texto == "No trata de Brío."
    assert unchecked.descartes == []


# ------------------------------------------------------ the service end to end

@pytest.fixture
def make_service(cleaned, analyzer, corpus, reported, tmp_path):
    def build(llm):
        retriever = HybridRetriever(corpus.safe, FakeEmbedder(), tmp_path)
        return AnalystService(analyzer, retriever, llm, cleaned.report, corpus, reported, corpus.safe)
    return build


def test_service_a_model_that_obeys_an_injection_cannot_change_the_headline(make_service):
    plan = Plan(tipo="cuantitativa", analisis=[spec()], usar_documentos=False)
    llm = FakeLLM(plan=plan, answer=answer("Centro tuvo la mayor caída de ventas.",
                                           hechos=[("Centro cayó más que ninguna otra región.", ["CALC-1"])]))
    result = make_service(llm).ask("¿Cuál fue la región con mayor caída?")
    assert result.answer.resumen == "Centro tuvo la mayor caída de ventas."          # what the model wrote, kept for audit
    assert "Sur" in result.final.resumen and "Centro tuvo" not in result.final.resumen
    assert result.final.descartes[0].motivo.startswith("nombra a Centro como la región de mayor caída")


def test_service_contrast_question_gets_the_codes_verdict_whatever_the_model_says(make_service):
    plan = Plan(tipo="contraste_reporte", usar_documentos=True,
                analisis=[spec(agrupar_por=["region", "canal"])])             # the model asked for region x channel
    llm = FakeLLM(plan=plan, answer=answer("Sí, es consistente.", contradicciones=[("No se encontraron contradicciones.", ["CALC-1"])]))
    result = make_service(llm).ask("¿Son consistentes las conclusiones del reporte comercial con los datos de ventas?")
    assert result.plan.analisis[0].agrupar_por == ["region"]                      # forced to the report's granularity
    assert [b.calc_id for b in result.calc] == ["CALC-1", "CONTRASTE-1"]
    assert "[CONTRASTE-1]" in llm.calls[1]["user"] and "NO coincide" in llm.calls[1]["user"]
    assert result.final.resumen.startswith("El reporte NO es consistente con los datos")
    assert len(result.final.contradicciones) == 3 and result.contrast["consistent"] is False
    assert result.final.descartes == []          # the model's own contradictions only repeat the code's: skipped


def test_service_contrast_without_the_report_table_says_so(cleaned, analyzer, corpus, tmp_path):
    service = AnalystService(analyzer, HybridRetriever(corpus.safe, FakeEmbedder(), tmp_path),
                             FakeLLM(plan=Plan(tipo="contraste_reporte", analisis=[spec()], usar_documentos=True)),
                             cleaned.report, corpus)
    result = service.ask("¿Es consistente el reporte comercial?")
    assert any("contraste calculado no está disponible" in w for w in result.warnings) and result.contrast is None


def test_service_causality_question_always_brings_the_limitations(make_service):
    plan = Plan(tipo="causalidad", analisis=[spec()], usar_documentos=True)
    result = make_service(FakeLLM(plan=plan)).ask("¿Las promociones de la competencia causaron la caída?")
    ids = {e.chunk_id for e in result.evidence}
    assert ids & {"INV-04", "INV-LIM-02", "REP-02"}
    assert any("no contiene variables de promoción ni de precio" in c.texto for c in result.final.insuficiente)


def test_service_reports_what_was_discarded_in_the_cli(make_service):
    plan = Plan(tipo="cuantitativa", analisis=[spec()], usar_documentos=False)
    llm = FakeLLM(plan=plan, answer=answer("Resumen.", hipotesis=[("Puede ser la competencia local.", ["CALC-1"])]))
    out = io.StringIO()
    main(["¿Cuál fue la región con mayor caída?"], service_factory=lambda: make_service(llm), out=out)
    text = out.getvalue()
    assert "DESCARTADO POR EL VALIDADOR (1)" in text and "Motivo: es una hipótesis y no cita ningún documento" in text
    assert "Mayor caída de ventas netas (COP) por región: Sur" in text


def test_the_result_includes_the_model_answer_the_final_answer_and_the_contrast(make_service):
    plan = Plan(tipo="contraste_reporte", analisis=[spec()], usar_documentos=True)
    data = make_service(FakeLLM(plan=plan)).ask("¿Es consistente el reporte comercial?").to_dict()
    assert {"answer", "final", "contrast"} <= set(data)
    assert data["final"]["contradicciones"] and data["contrast"]["rows"]


# ---------------------------------------------- holes found in real runs, closed

def test_a_general_document_cannot_back_a_claim_about_a_region(analyzer, chunks):
    final = validate_answer(answer(hipotesis=[("La competencia puede haber influido, especialmente en la región Sur.", ["REP-02"])]),
                            context(analyzer, chunks, ["REP-02"]))
    assert final.hipotesis == [] and "REP-02 no menciona ninguna región" in final.descartes[0].motivo
    no_region = validate_answer(answer(hipotesis=[("La competencia puede haber influido en las ventas.", ["REP-02"])]),
                                context(analyzer, chunks, ["REP-02"]))
    assert len(no_region.hipotesis) == 1


def test_a_contradiction_must_be_between_two_sources(analyzer, chunks):
    ctx = context(analyzer, chunks, ["INV-04", "REP-02"])
    one = validate_answer(answer(contradicciones=[("La variación no se puede atribuir a la competencia.", ["CALC-1"])]), ctx)
    assert one.contradicciones == [] and "al menos dos" in one.descartes[0].motivo
    two = validate_answer(answer(contradicciones=[("El reporte atribuye el 70 %, pero el estudio no mide impacto.", ["INV-04", "REP-02"])]), ctx)
    assert len(two.contradicciones) == 1 and two.descartes == []


def test_a_false_absence_of_the_sales_base_is_discarded(analyzer, chunks):
    bad = answer(insuficiente=["No se puede afirmar nada, ya que no se dispone de información sobre la base de ventas."])
    final = validate_answer(bad, context(analyzer, chunks))
    assert final.insuficiente == [] and "los cálculos se hicieron con ella" in final.descartes[0].motivo


# ---------------------------------------- holes found in an open question, closed

def test_similar_drops_are_not_presented_as_a_largest_drop(analyzer):
    blocks = build_calc_blocks(analyzer, Plan(tipo="integrada", usar_documentos=True,
                                              analisis=[spec(agrupar_por=["canal"])]))
    assert "Caída similar en todos los grupos (entre -10,43 % y -10,59 %)" in blocks[0].text
    assert "Mayor caída" not in blocks[0].text
    fact = calc_facts(blocks)[0]
    assert fact.texto.startswith("Caída similar en todos los grupos por canal: entre -10,43 % y -10,59 %")


def test_a_summary_naming_a_largest_drop_is_discarded_when_the_drops_are_similar(analyzer, chunks):
    """Caso real: el resumen decía 'caídas similares' y a la vez 'la mayor caída fue en Moderno'."""
    blocks = build_calc_blocks(analyzer, Plan(tipo="cuantitativa", usar_documentos=False,
                                              analisis=[spec(agrupar_por=["canal"])]))
    ctx = ValidationContext(blocks, [], "cuantitativa", None, tuple(analyzer.columns))
    final = validate_answer(answer("Las caídas son similares, pero la mayor caída se dio en el canal Moderno."), ctx)
    assert final.resumen_origen == "calculo" and final.resumen.startswith("Caída similar")
    assert "caídas similares en todos los grupos" in final.descartes[0].motivo
    fine = validate_answer(answer("Todos los canales cayeron de forma parecida."), ctx)
    assert fine.resumen_origen == "modelo" and fine.descartes == []


def test_the_brand_total_is_stated_once_even_with_several_analyses(analyzer):
    blocks = build_calc_blocks(analyzer, Plan(tipo="integrada", usar_documentos=True,
                                              analisis=[spec(), spec(agrupar_por=["canal"])]))
    totals = [f for f in calc_facts(blocks) if f.texto.startswith("Total Brío")]
    assert len(totals) == 1 and totals[0].fuentes == ["CALC-1", "CALC-2"]


def test_a_claim_about_a_channel_needs_a_document_that_mentions_it(analyzer, chunks):
    ctx = context(analyzer, chunks, ["REP-02", "OC-SUR-01"])
    bad = validate_answer(answer(hipotesis=[("La caída en el canal Moderno puede deberse a la competitividad promocional.", ["REP-02"])]), ctx)
    assert bad.hipotesis == [] and "menciona Moderno" in bad.descartes[0].motivo and "no menciona ese canal" in bad.descartes[0].motivo
    ok = validate_answer(answer(hipotesis=[("En el canal tradicional se observaron faltantes en Sur.", ["OC-SUR-01"])]), ctx)
    assert len(ok.hipotesis) == 1


def test_saying_there_are_no_contradictions_is_never_a_contradiction(analyzer, chunks):
    ctx = context(analyzer, chunks, ["INV-04", "REP-02"])
    final = validate_answer(answer(contradicciones=[("No se encontraron contradicciones entre fuentes.", ["INV-04", "REP-02"])]), ctx)
    assert final.contradicciones == [] and "no es una contradicción" in final.descartes[0].motivo
