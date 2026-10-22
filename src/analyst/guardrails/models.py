"""The answer the analyst actually sees: checked claims, what was discarded and why."""
from dataclasses import asdict, dataclass, field

from analyst.llm.schemas import Answer

ORIGIN_CALC = "calculo"          # written by code from the pandas results
ORIGIN_CONTRAST = "contraste"    # written by code from the report-vs-data comparison
ORIGIN_DOCUMENT = "documento"    # written by the model, cites documents, passed the checks
ORIGIN_MODEL = "modelo"          # written by the model (hypotheses, contradictions), passed the checks


@dataclass
class FinalClaim:
    texto: str
    fuentes: list[str] = field(default_factory=list)
    origen: str = ORIGIN_MODEL


@dataclass
class Discard:
    seccion: str
    texto: str
    motivo: str


@dataclass
class FinalAnswer:
    resumen: str
    resumen_origen: str = ORIGIN_MODEL
    hechos: list[FinalClaim] = field(default_factory=list)
    hipotesis: list[FinalClaim] = field(default_factory=list)
    contradicciones: list[FinalClaim] = field(default_factory=list)
    insuficiente: list[FinalClaim] = field(default_factory=list)
    descartes: list[Discard] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def unchecked(cls, answer: Answer) -> "FinalAnswer":
        """For answers that involve no model claims (e.g. out of scope): nothing to validate."""
        return cls(resumen=answer.resumen, resumen_origen=ORIGIN_CALC,
                   insuficiente=[FinalClaim(g.texto, [], ORIGIN_CALC) for g in answer.insuficiente])
