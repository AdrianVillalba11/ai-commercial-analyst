"""
SERVICIO: EL FLUJO COMPLETO DE UNA PREGUNTA

Qué hace: orquesta todo el recorrido planificar -> calcular -> recuperar evidencia -> redactar -> validar.
Con qué se conecta: es el centro de la app. Lo usan la API web (api/main.py) y la línea de comandos
(cli.py); él usa todos los demás módulos.
Idea clave: `ask()` devuelve la respuesta ya verificada junto con la TRAZA completa (plan, cálculos con sus
fórmulas, evidencia usada, lo que descartó el validador, tiempos y avisos), de modo que cada cifra y cada
cita se pueden comprobar.
"""
import time
from dataclasses import dataclass, field
from typing import Callable

from analyst.analytics.metrics import SalesAnalyzer
from analyst.config import settings
from analyst.data.loader import load_clean_sales
from analyst.data.quality_report import QualityReport
from analyst.docs.chunker import Chunk
from analyst.docs.parser import parse_reported_regional_variation
from analyst.errors import OllamaError
from analyst.guardrails.models import FinalAnswer
from analyst.guardrails.report_check import CONTRAST_ID, ReportContrast, compare_report_to_data
from analyst.guardrails.validator import ValidationContext, question_notices, validate_answer
from analyst.llm.answerer import ANSWER_MAX_TOKENS, answer_question, out_of_scope_answer
from analyst.llm.ollama_client import OllamaChat
from analyst.llm.planner import StructuredLLM, plan_question
from analyst.llm.prompts import CalcBlock, build_answer_prompt, build_calc_blocks
from analyst.llm.schemas import Answer, Plan
from analyst.rag.retriever import HybridRetriever, RetrievedChunk, build_default_retriever
from analyst.security.injection import ScreeningResult

# =====================================================================
# BLOQUE: Constantes
# =====================================================================
MAX_QUESTION_CHARS = 1000
REPORT_FILENAME = "05_Reporte_Comercial.pdf"
# En preguntas de causalidad siempre se busca también la evidencia sobre lo que los estudios NO pueden mostrar.
CAUSALITY_QUERY = "causalidad grupo de control diseño experimental no medido promociones"
CAUSALITY_EXTRA_CHUNKS = 3


# =====================================================================
# BLOQUE: Estructuras de resultado
# =====================================================================
@dataclass
class EvidenceRef:
    """Fragmento de evidencia tal como se entrega a la interfaz (con el puesto que le dio cada buscador)."""

    chunk_id: str
    doc: str
    doc_title: str
    section: str
    page: int | None
    regions: list[str]
    kind: str
    ranks: dict[str, int]
    text: str

    @classmethod
    def from_retrieved(cls, item: RetrievedChunk) -> "EvidenceRef":
        """Crea la referencia a partir de un fragmento recuperado."""
        c = item.chunk
        return cls(c.chunk_id, c.doc, c.doc_title, c.section, c.page, c.regions, c.kind, item.ranks, c.text)


@dataclass
class AskResult:
    """Todo lo que devuelve una pregunta: respuesta final, lo que escribió el modelo y la traza para verificar."""

    question: str
    plan: Plan
    plan_source: str                       # "llm" o "fallback" (plan por defecto)
    calc: list[CalcBlock]
    evidence: list[EvidenceRef]
    answer: Answer                         # lo que escribió el modelo, tal cual
    final: FinalAnswer                     # lo que ve el analista: verificado, con lo descartado y por qué
    timings: dict[str, float]
    warnings: list[str]
    quality_notes: list[dict]
    mode: str
    model: str
    contrast: dict | None = None
    tokens: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Versión lista para JSON (la que consume la interfaz)."""
        return {
            "question": self.question, "plan": self.plan.model_dump(), "plan_source": self.plan_source,
            "calc": [{"calc_id": b.calc_id, "title": b.title, "text": b.text, "spec": b.spec, "result": b.result}
                     for b in self.calc],
            "evidence": [vars(e) for e in self.evidence], "answer": self.answer.model_dump(),
            "final": self.final.to_dict(), "contrast": self.contrast,
            "timings": self.timings, "warnings": self.warnings, "quality_notes": self.quality_notes,
            "mode": self.mode, "model": self.model, "tokens": self.tokens,
        }


# =====================================================================
# BLOQUE: Servicio
# =====================================================================
class AnalystService:
    """Une el analizador de ventas, el buscador de evidencia y el modelo en un solo flujo."""

    def __init__(self, analyzer: SalesAnalyzer, retriever: HybridRetriever, llm: StructuredLLM,
                 quality_report: QualityReport | None = None, screening: ScreeningResult | None = None,
                 reported_variation: dict[str, float] | None = None, corpus_chunks: list[Chunk] | None = None):
        self._analyzer, self._retriever, self._llm = analyzer, retriever, llm
        self._quality_report, self._screening = quality_report, screening
        self._reported = reported_variation
        self._corpus_chunks = corpus_chunks or (screening.safe if screening else [])
        self._chunks_by_id = {c.chunk_id: c for c in self._corpus_chunks}

    @property
    def document_count(self) -> int:
        """Cantidad de fragmentos de documentos disponibles."""
        return len(self._corpus_chunks)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Un fragmento del corpus ya revisado. Los de cuarentena no están, así que nunca se entregan."""
        return self._chunks_by_id.get(chunk_id)

    def warm_up(self) -> None:
        """Carga los modelos en memoria para que la primera pregunta real no sea lenta."""
        self._retriever.retrieve("calentamiento", top_k=1, mode=settings.retrieval_mode)
        warm = getattr(self._llm, "warm_up", None)
        if warm:
            warm()

    def data_quality(self) -> dict:
        """Reporte de calidad de datos para el panel."""
        return self._quality_report.to_dict() if self._quality_report else {}

    def security_alerts(self) -> list[dict]:
        """Fragmentos en cuarentena o con advertencia, para el panel de seguridad."""
        return [f.to_dict() for f in (self._screening.quarantined + self._screening.warnings)] if self._screening else []

    def ask(self, question: str, mode: str | None = None, top_k: int = 6,
            progress: Callable[[str], None] | None = None) -> AskResult:
        """Responde una pregunta. `progress`, si se da, se llama con el nombre de cada paso al comenzar."""
        mode = mode or settings.retrieval_mode
        notify = progress or (lambda step: None)
        question = question.strip()
        if not question:
            raise ValueError("La pregunta está vacía.")
        if len(question) > MAX_QUESTION_CHARS:
            raise ValueError(f"La pregunta es demasiado larga (máximo {MAX_QUESTION_CHARS} caracteres).")
        started = time.perf_counter()
        timings: dict[str, float] = {}
        warnings: list[str] = []

        # Paso 1 - Planificación: el modelo elige qué calcular (dentro de un catálogo cerrado).
        notify("planificacion")
        t = time.perf_counter()
        planned = plan_question(self._llm, question)
        timings["planificacion"] = time.perf_counter() - t
        warnings += [f"Planificador: {n}" for n in planned.notes]
        plan = planned.plan

        # Fuera de alcance: respuesta fija, sin cálculos ni más llamadas al modelo.
        if plan.tipo == "fuera_de_alcance":
            timings["total"] = time.perf_counter() - started
            answer = out_of_scope_answer()
            return AskResult(question, plan, planned.source, [], [], answer, FinalAnswer.unchecked(answer), timings,
                             warnings, [], mode, self._llm.name)

        # Paso 2 - Cálculo: pandas calcula las cifras (y el contraste con el reporte si corresponde).
        notify("calculo")
        t = time.perf_counter()
        calc = build_calc_blocks(self._analyzer, plan)
        contrast = self._contrast(plan, calc, warnings)
        timings["calculo"] = time.perf_counter() - t
        quality_notes = calc[0].result["quality_notes"] if calc else []

        # Paso 3 - Recuperación: solo si el plan necesita documentos.
        items: list[RetrievedChunk] = []
        if plan.usar_documentos:
            notify("recuperacion")
            t = time.perf_counter()
            regions = sorted({s.region for s in plan.analisis if s.region != "todas"}) or None
            retrieved = self._retriever.retrieve(question, top_k=top_k, regions=regions, mode=mode)
            items, warnings = retrieved.items, warnings + retrieved.notes
            if plan.tipo == "causalidad":
                items = self._with_limitations(items, mode)
            timings["recuperacion"] = time.perf_counter() - t

        # Armado del prompt ajustado a la ventana de contexto del modelo.
        num_ctx = getattr(self._llm, "num_ctx", 4096)
        bundle = build_answer_prompt(question, calc, items, quality_notes, use_documents=plan.usar_documentos,
                                     num_ctx=num_ctx, max_output_tokens=ANSWER_MAX_TOKENS)
        if bundle.evidence_dropped:
            warnings.append(f"Se recortaron {bundle.evidence_dropped} fragmentos de evidencia para caber en el "
                            f"contexto del modelo ({num_ctx} tokens).")

        # Paso 4 - Redacción: el modelo escribe la respuesta estructurada.
        notify("redaccion")
        t = time.perf_counter()
        generated = answer_question(self._llm, bundle)
        timings["redaccion"] = time.perf_counter() - t
        warnings += [f"Redactor: {n}" for n in generated.info.notes]

        # Paso 5 - Validación: el código verifica lo que escribió el modelo y arma la respuesta final.
        notify("validacion")
        t = time.perf_counter()
        context = ValidationContext(calc, [i.chunk for i in bundle.evidence_used], plan.tipo, contrast,
                                    tuple(self._analyzer.columns))
        final = validate_answer(generated.value, context)
        # Avisos sobre la pregunta (otra marca, promoción/precio): van primero en "Información insuficiente".
        final.insuficiente = question_notices(question, plan.tipo, context.columns) + final.insuficiente
        timings["validacion"] = time.perf_counter() - t
        timings["total"] = time.perf_counter() - started

        return AskResult(
            question, plan, planned.source, calc, [EvidenceRef.from_retrieved(i) for i in bundle.evidence_used],
            generated.value, final, timings, warnings, quality_notes, mode, self._llm.name,
            contrast.to_dict() if contrast else None,
            {"prompt": generated.info.prompt_tokens, "salida": generated.info.output_tokens})

    def _contrast(self, plan: Plan, calc: list[CalcBlock], warnings: list[str]) -> ReportContrast | None:
        """Para preguntas sobre el reporte comercial: lo compara con los datos en código y pasa el veredicto al modelo."""
        if plan.tipo != "contraste_reporte":
            return None
        if not self._reported:
            warnings.append("No se pudo leer la tabla del reporte comercial; el contraste calculado no está disponible.")
            return None
        contrast = compare_report_to_data(self._analyzer, self._reported, self._corpus_chunks)
        calc.append(CalcBlock(CONTRAST_ID, contrast.title, contrast.text, contrast.to_dict(), {}))
        return contrast

    def _with_limitations(self, items: list[RetrievedChunk], mode: str) -> list[RetrievedChunk]:
        """Las preguntas de causalidad siempre reciben la evidencia sobre lo que los estudios no pueden mostrar."""
        extra = self._retriever.retrieve(CAUSALITY_QUERY, top_k=CAUSALITY_EXTRA_CHUNKS, mode=mode).items
        known = {i.chunk.chunk_id for i in items}
        return items + [i for i in extra if i.chunk.chunk_id not in known]


# =====================================================================
# BLOQUE: Construcción por defecto
# =====================================================================
def build_default_service() -> AnalystService:
    """Carga y limpia los datos, revisa e indexa los documentos y conecta con Ollama."""
    cleaned = load_clean_sales()
    retriever, screening = build_default_retriever()
    try:
        reported = parse_reported_regional_variation(settings.raw_data_dir / REPORT_FILENAME)
    except (OSError, ValueError):
        reported = None     # el contraste se reporta entonces como no disponible, en vez de fallar toda la app
    return AnalystService(SalesAnalyzer(cleaned.data, cleaned.report), retriever, OllamaChat(),
                          cleaned.report, screening, reported, screening.safe)


__all__ = ["AnalystService", "AskResult", "EvidenceRef", "build_default_service", "OllamaError"]
