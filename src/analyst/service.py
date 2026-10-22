"""The question-answering pipeline: plan -> compute -> retrieve -> write -> validate.

`ask()` returns the checked answer together with a full trace (plan, calculations with their
formulas, evidence used, what the validator discarded, timings, warnings) so every figure and
every citation can be verified.
"""
import time
from dataclasses import dataclass, field

from analyst.analytics.metrics import SalesAnalyzer
from analyst.config import settings
from analyst.data.loader import load_clean_sales
from analyst.data.quality_report import QualityReport
from analyst.docs.chunker import Chunk
from analyst.docs.parser import parse_reported_regional_variation
from analyst.errors import OllamaError
from analyst.guardrails.models import FinalAnswer
from analyst.guardrails.report_check import CONTRAST_ID, ReportContrast, compare_report_to_data
from analyst.guardrails.validator import ValidationContext, validate_answer
from analyst.llm.answerer import ANSWER_MAX_TOKENS, answer_question, out_of_scope_answer
from analyst.llm.ollama_client import OllamaChat
from analyst.llm.planner import StructuredLLM, plan_question
from analyst.llm.prompts import CalcBlock, build_answer_prompt, build_calc_blocks
from analyst.llm.schemas import Answer, Plan
from analyst.rag.retriever import HybridRetriever, RetrievedChunk, build_default_retriever
from analyst.security.injection import ScreeningResult

MAX_QUESTION_CHARS = 1000
REPORT_FILENAME = "05_Reporte_Comercial.pdf"
CAUSALITY_QUERY = "causalidad grupo de control diseño experimental no medido promociones"
CAUSALITY_EXTRA_CHUNKS = 3


@dataclass
class EvidenceRef:
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
        c = item.chunk
        return cls(c.chunk_id, c.doc, c.doc_title, c.section, c.page, c.regions, c.kind, item.ranks, c.text)


@dataclass
class AskResult:
    question: str
    plan: Plan
    plan_source: str                       # "llm" or "fallback"
    calc: list[CalcBlock]
    evidence: list[EvidenceRef]
    answer: Answer                         # what the model wrote, as it came
    final: FinalAnswer                     # what the analyst sees: checked, with what was discarded and why
    timings: dict[str, float]
    warnings: list[str]
    quality_notes: list[dict]
    mode: str
    model: str
    contrast: dict | None = None
    tokens: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "question": self.question, "plan": self.plan.model_dump(), "plan_source": self.plan_source,
            "calc": [{"calc_id": b.calc_id, "title": b.title, "text": b.text, "spec": b.spec, "result": b.result}
                     for b in self.calc],
            "evidence": [vars(e) for e in self.evidence], "answer": self.answer.model_dump(),
            "final": self.final.to_dict(), "contrast": self.contrast,
            "timings": self.timings, "warnings": self.warnings, "quality_notes": self.quality_notes,
            "mode": self.mode, "model": self.model, "tokens": self.tokens,
        }


class AnalystService:
    def __init__(self, analyzer: SalesAnalyzer, retriever: HybridRetriever, llm: StructuredLLM,
                 quality_report: QualityReport | None = None, screening: ScreeningResult | None = None,
                 reported_variation: dict[str, float] | None = None, corpus_chunks: list[Chunk] | None = None):
        self._analyzer, self._retriever, self._llm = analyzer, retriever, llm
        self._quality_report, self._screening = quality_report, screening
        self._reported = reported_variation
        self._corpus_chunks = corpus_chunks or (screening.safe if screening else [])

    def data_quality(self) -> dict:
        return self._quality_report.to_dict() if self._quality_report else {}

    def security_alerts(self) -> list[dict]:
        return [f.to_dict() for f in (self._screening.quarantined + self._screening.warnings)] if self._screening else []

    def ask(self, question: str, mode: str | None = None, top_k: int = 6) -> AskResult:
        mode = mode or settings.retrieval_mode
        question = question.strip()
        if not question:
            raise ValueError("La pregunta está vacía.")
        if len(question) > MAX_QUESTION_CHARS:
            raise ValueError(f"La pregunta es demasiado larga (máximo {MAX_QUESTION_CHARS} caracteres).")
        started = time.perf_counter()
        timings: dict[str, float] = {}
        warnings: list[str] = []

        t = time.perf_counter()
        planned = plan_question(self._llm, question)
        timings["planificacion"] = time.perf_counter() - t
        warnings += [f"Planificador: {n}" for n in planned.notes]
        plan = planned.plan

        if plan.tipo == "fuera_de_alcance":
            timings["total"] = time.perf_counter() - started
            answer = out_of_scope_answer()
            return AskResult(question, plan, planned.source, [], [], answer, FinalAnswer.unchecked(answer), timings,
                             warnings, [], mode, self._llm.name)

        t = time.perf_counter()
        calc = build_calc_blocks(self._analyzer, plan)
        contrast = self._contrast(plan, calc, warnings)
        timings["calculo"] = time.perf_counter() - t
        quality_notes = calc[0].result["quality_notes"] if calc else []

        items: list[RetrievedChunk] = []
        if plan.usar_documentos:
            t = time.perf_counter()
            regions = sorted({s.region for s in plan.analisis if s.region != "todas"}) or None
            retrieved = self._retriever.retrieve(question, top_k=top_k, regions=regions, mode=mode)
            items, warnings = retrieved.items, warnings + retrieved.notes
            if plan.tipo == "causalidad":
                items = self._with_limitations(items, mode)
            timings["recuperacion"] = time.perf_counter() - t

        num_ctx = getattr(self._llm, "num_ctx", 4096)
        bundle = build_answer_prompt(question, calc, items, quality_notes, use_documents=plan.usar_documentos,
                                     num_ctx=num_ctx, max_output_tokens=ANSWER_MAX_TOKENS)
        if bundle.evidence_dropped:
            warnings.append(f"Se recortaron {bundle.evidence_dropped} fragmentos de evidencia para caber en el "
                            f"contexto del modelo ({num_ctx} tokens).")

        t = time.perf_counter()
        generated = answer_question(self._llm, bundle)
        timings["redaccion"] = time.perf_counter() - t
        warnings += [f"Redactor: {n}" for n in generated.info.notes]

        t = time.perf_counter()
        context = ValidationContext(calc, [i.chunk for i in bundle.evidence_used], plan.tipo, contrast,
                                    tuple(self._analyzer.columns))
        final = validate_answer(generated.value, context)
        timings["validacion"] = time.perf_counter() - t
        timings["total"] = time.perf_counter() - started

        return AskResult(
            question, plan, planned.source, calc, [EvidenceRef.from_retrieved(i) for i in bundle.evidence_used],
            generated.value, final, timings, warnings, quality_notes, mode, self._llm.name,
            contrast.to_dict() if contrast else None,
            {"prompt": generated.info.prompt_tokens, "salida": generated.info.output_tokens})

    def _contrast(self, plan: Plan, calc: list[CalcBlock], warnings: list[str]) -> ReportContrast | None:
        """For questions about the commercial report: compare it with the data in code, and hand the verdict to the model."""
        if plan.tipo != "contraste_reporte":
            return None
        if not self._reported:
            warnings.append("No se pudo leer la tabla del reporte comercial; el contraste calculado no está disponible.")
            return None
        contrast = compare_report_to_data(self._analyzer, self._reported, self._corpus_chunks)
        calc.append(CalcBlock(CONTRAST_ID, contrast.title, contrast.text, contrast.to_dict(), {}))
        return contrast

    def _with_limitations(self, items: list[RetrievedChunk], mode: str) -> list[RetrievedChunk]:
        """Causality questions always get the evidence about what the studies cannot show."""
        extra = self._retriever.retrieve(CAUSALITY_QUERY, top_k=CAUSALITY_EXTRA_CHUNKS, mode=mode).items
        known = {i.chunk.chunk_id for i in items}
        return items + [i for i in extra if i.chunk.chunk_id not in known]


def build_default_service() -> AnalystService:
    """Loads and cleans the data, screens and indexes the documents, and connects to Ollama."""
    cleaned = load_clean_sales()
    retriever, screening = build_default_retriever()
    try:
        reported = parse_reported_regional_variation(settings.raw_data_dir / REPORT_FILENAME)
    except (OSError, ValueError):
        reported = None     # the contrast is then reported as unavailable instead of failing the whole app
    return AnalystService(SalesAnalyzer(cleaned.data, cleaned.report), retriever, OllamaChat(),
                          cleaned.report, screening, reported, screening.safe)


__all__ = ["AnalystService", "AskResult", "EvidenceRef", "build_default_service", "OllamaError"]
