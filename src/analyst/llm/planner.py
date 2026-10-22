"""Turns a question into a validated plan. The model only picks from a closed catalog.

If the model cannot produce a valid plan (twice), a keyword-based default plan is used and the
answer says so. A broken Ollama (not running, model missing) is NOT hidden: that error propagates.
"""
import re
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from pydantic import BaseModel

from analyst.data.cleaning import REGIONS
from analyst.errors import LLMOutputError
from analyst.llm.ollama_client import Generated
from analyst.llm.prompts import spec_key
from analyst.llm.schemas import AnalysisSpec, Plan
from analyst.text import fold

T = TypeVar("T", bound=BaseModel)
PLAN_MAX_TOKENS = 300
DOCUMENT_TYPES = {"integrada", "contraste_reporte", "causalidad"}
# Words that show a question is about the commercial analysis; a model that calls it "out of scope" is wrong.
_DOMAIN = re.compile(r"brio|venta|region|\b(?:norte|sur|centro|oriente|occidente)\b|faltante|consumidor|promocion|"
                     r"reporte|canal|presentacion|caida|comerciante|visita|precio|disponibilidad|competencia|"
                     r"deterioro|entrevista|inventario|formato|\bt[12]\b")

PLAN_SYSTEM = (
    "Planificas el análisis de una pregunta sobre la marca Brío (ventas netas; T1 = ene-mar y T2 = abr-jun de 2026). "
    "Responde solo JSON.\n"
    "tipo: cuantitativa = pide cifras o cálculos de ventas; integrada = pide explicar o contextualizar con estudios de "
    "consumidor y visitas a puntos de venta; contraste_reporte = pide comparar el reporte comercial con los datos; "
    "causalidad = pregunta si algo causó la caída; fuera_de_alcance = no trata de Brío, de sus ventas ni de lo que las "
    "explica (consumidores, puntos de venta, reporte comercial); si no nombra la marca, se asume Brío.\n"
    "analisis: los cálculos a hacer (máximo 2, sin repetir; lista vacía si no hace falta). agrupar_por: por qué dimensión "
    "comparar. region, canal y presentacion: 'todas' o 'todos' salvo que la pregunta pida una en concreto.\n"
    "usar_documentos: true si la respuesta necesita los estudios, las visitas o el reporte comercial."
)


class StructuredLLM(Protocol):
    @property
    def name(self) -> str: ...
    def generate_structured(self, model_cls: type[T], system: str, user: str, max_tokens: int) -> Generated[T]: ...


@dataclass
class PlanResult:
    plan: Plan
    source: str                       # "llm" or "fallback"
    notes: list[str] = field(default_factory=list)
    seconds: float = 0.0


def plan_question(llm: StructuredLLM, question: str) -> PlanResult:
    try:
        generated = llm.generate_structured(Plan, PLAN_SYSTEM, f"Pregunta: {question}", PLAN_MAX_TOKENS)
    except LLMOutputError as exc:
        return PlanResult(fallback_plan(question), "fallback",
                          [f"El modelo no produjo un plan válido; se usó un plan por defecto. {exc}"])
    plan = normalize_plan(generated.value)
    if plan.tipo == "fuera_de_alcance" and _DOMAIN.search(fold(question)):
        return PlanResult(fallback_plan(question), "fallback", generated.info.notes + [
            "El modelo marcó la pregunta como fuera de alcance, pero trata temas del análisis comercial; "
            "se trató como una pregunta en alcance."], generated.info.seconds)
    return PlanResult(plan, "llm", generated.info.notes, generated.info.seconds)


def default_spec(dimension: str = "region", metric: str = "venta_neta_cop", region: str = "todas") -> AnalysisSpec:
    return AnalysisSpec(agrupar_por=[dimension], metrica=metric, region=region, canal="todos", presentacion="todas")


def normalize_plan(plan: Plan) -> Plan:
    """Drop repeated analyses and enforce the rules the code relies on, whatever the model said."""
    if plan.tipo == "fuera_de_alcance":
        return Plan(tipo=plan.tipo, analisis=[], usar_documentos=False)
    unique, seen = [], set()
    for spec in plan.analisis:
        if spec_key(spec) not in seen:
            seen.add(spec_key(spec))
            unique.append(spec)
    # Pure number questions do not use the documents: the commercial report quotes figures that
    # disagree with the data, and mixing it in only invites wrong attributions.
    use_documents = plan.tipo in DOCUMENT_TYPES or (plan.usar_documentos and plan.tipo != "cuantitativa")
    # The report speaks in regional totals, so the comparison must be by region whatever the model chose.
    if plan.tipo == "contraste_reporte":
        analyses = [default_spec()]
    elif plan.tipo == "causalidad":      # a region x channel table is noise for "did X cause the drop?"
        analyses = [s for s in unique if s.region != "todas"] or [default_spec()]
    else:
        analyses = unique or [default_spec()]
    return Plan(tipo=plan.tipo, analisis=analyses, usar_documentos=use_documents)


def fallback_plan(question: str) -> Plan:
    """Keyword-based plan: compare by region (or channel / presentation if named) and use documents."""
    q = fold(question)
    dimension = "region"
    if "canal" in q:
        dimension = "canal"
    elif re.search(r"presentacion|formato|familiar|individual", q):
        dimension = "presentacion"
    metric = "unidades" if re.search(r"\bunidades?\b", q) else "venta_neta_cop"
    named = [r for r in REGIONS if re.search(rf"\b{fold(r)}\b", q)]
    kind = "causalidad" if "caus" in q else "contraste_reporte" if "reporte" in q else "integrada"
    return Plan(tipo=kind, analisis=[default_spec(dimension, metric, named[0] if len(named) == 1 else "todas")],
                usar_documentos=True)
