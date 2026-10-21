"""The question-answering pipeline: plan -> compute -> retrieve -> write.

`ask()` returns the answer together with a full trace (plan, calculations with their formulas,
evidence used, timings, warnings) so every figure and every citation can be checked.
"""
import time
from dataclasses import dataclass, field

from analyst.analytics.metrics import SalesAnalyzer
from analyst.data.loader import load_clean_sales
from analyst.data.quality_report import QualityReport
from analyst.errors import OllamaError
from analyst.llm.answerer import ANSWER_MAX_TOKENS, answer_question, out_of_scope_answer
from analyst.llm.ollama_client import OllamaChat
from analyst.llm.planner import StructuredLLM, plan_question
from analyst.llm.prompts import CalcBlock, build_answer_prompt, build_calc_blocks
from analyst.llm.schemas import Answer, Plan
from analyst.rag.retriever import HybridRetriever, RetrievedChunk, build_default_retriever
from analyst.security.injection import ScreeningResult

MAX_QUESTION_CHARS = 1000


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
    answer: Answer
    timings: dict[str, float]
    warnings: list[str]
    quality_notes: list[dict]
    mode: str
    model: str
    tokens: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "question": self.question, "plan": self.plan.model_dump(), "plan_source": self.plan_source,
            "calc": [{"calc_id": b.calc_id, "title": b.title, "text": b.text, "spec": b.spec, "result": b.result}
                     for b in self.calc],
            "evidence": [vars(e) for e in self.evidence], "answer": self.answer.model_dump(),
            "timings": self.timings, "warnings": self.warnings, "quality_notes": self.quality_notes,
            "mode": self.mode, "model": self.model, "tokens": self.tokens,
        }


class AnalystService:
    def __init__(self, analyzer: SalesAnalyzer, retriever: HybridRetriever, llm: StructuredLLM,
                 quality_report: QualityReport | None = None, screening: ScreeningResult | None = None):
        self._analyzer, self._retriever, self._llm = analyzer, retriever, llm
        self._quality_report, self._screening = quality_report, screening

    def data_quality(self) -> dict:
        return self._quality_report.to_dict() if self._quality_report else {}

    def security_alerts(self) -> list[dict]:
        return [f.to_dict() for f in (self._screening.quarantined + self._screening.warnings)] if self._screening else []

    def ask(self, question: str, mode: str = "hybrid", top_k: int = 6) -> AskResult:
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
            return AskResult(question, plan, planned.source, [], [], out_of_scope_answer(), timings, warnings, [],
                             mode, self._llm.name)

        t = time.perf_counter()
        calc = build_calc_blocks(self._analyzer, plan)
        timings["calculo"] = time.perf_counter() - t
        quality_notes = calc[0].result["quality_notes"] if calc else []

        items: list[RetrievedChunk] = []
        if plan.usar_documentos:
            t = time.perf_counter()
            regions = sorted({s.region for s in plan.analisis if s.region != "todas"}) or None
            retrieved = self._retriever.retrieve(question, top_k=top_k, regions=regions, mode=mode)
            timings["recuperacion"] = time.perf_counter() - t
            items = retrieved.items
            warnings += retrieved.notes

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
        timings["total"] = time.perf_counter() - started

        return AskResult(
            question, plan, planned.source, calc, [EvidenceRef.from_retrieved(i) for i in bundle.evidence_used],
            generated.value, timings, warnings, quality_notes, mode, self._llm.name,
            {"prompt": generated.info.prompt_tokens, "salida": generated.info.output_tokens})


def build_default_service() -> AnalystService:
    """Loads and cleans the data, screens and indexes the documents, and connects to Ollama."""
    cleaned = load_clean_sales()
    retriever, screening = build_default_retriever()
    return AnalystService(SalesAnalyzer(cleaned.data, cleaned.report), retriever, OllamaChat(),
                          cleaned.report, screening)


__all__ = ["AnalystService", "AskResult", "EvidenceRef", "build_default_service", "OllamaError"]
