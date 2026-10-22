"""Short evaluation of the whole pipeline: hybrid vs semantic retrieval, with llama3, over 8 questions.

Run:  python scripts/evaluate_answers.py        (about 3 minutes; needs Ollama with the models)

Metrics and the decision rule were fixed BEFORE running it, and are not adjusted to the results:
  1. Main metric: share of the model's checkable claims that survive the validator
     (checkable = hypotheses + document-based facts + contradictions; facts that only repeat a
     calculation are not counted because the code already states them).
  2. Coverage: for questions with expected evidence, whether the final answer cites at least one of
     the expected ids.
  3. Tie-break: average time.
Rule: pick the mode with the higher share of surviving claims, provided its coverage is not lower.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analyst.service import AskResult, build_default_service  # noqa: E402

QUESTIONS: list[tuple[str, str, set[str] | None]] = [
    ("cuantitativa", "¿Cuál fue la región con mayor caída porcentual en ventas netas de Brío entre T1 y T2 de 2026? "
                     "Muestra los valores y el cálculo.", None),
    ("integrada", "¿Qué hallazgos de consumidor y observaciones comerciales podrían ayudar a explicar el deterioro de Brío?",
     {"INV-01", "INV-02", "INV-03", "OC-SUR-01"}),
    ("contraste", "¿Son consistentes las principales conclusiones del reporte comercial con los datos de ventas?", None),
    ("causalidad", "¿Podemos afirmar que las promociones de la competencia causaron la caída de ventas? "
                   "¿Qué evidencia lo sustenta o limita?", {"INV-04", "REP-02", "INV-LIM-02"}),
    ("canal en Sur", "¿Cómo cayó Brío por canal en el Sur?", None),
    ("faltantes Sur", "¿Qué dicen los comerciantes sobre faltantes de producto en el Sur?",
     {"OC-SUR-01", "INV-02", "CAM-101", "CAM-108", "CAM-112", "OC-REGION-SUR"}),
    ("limitaciones", "¿Qué limitaciones tienen los estudios de consumidor para explicar la caída?",
     {"INV-LIM-01", "INV-LIM-02", "INV-LIM-04", "INV-S5-1"}),
    ("disponibilidad", "¿Hay evidencia de que la disponibilidad afectó las ventas en Sur?",
     {"OC-SUR-01", "INV-02", "REP-03", "CAM-101"}),
]
MODES = ("hybrid", "semantic")


@dataclass
class QuestionScore:
    name: str
    checkable: int
    discarded: int
    covered: bool | None
    seconds: float


def score(name: str, expected: set[str] | None, result: AskResult) -> QuestionScore:
    raw = result.answer
    doc_ids = {e.chunk_id for e in result.evidence}
    checkable = (len(raw.hipotesis) + (0 if result.contrast else len(raw.contradicciones)) +
                 sum(any(s in doc_ids for s in c.fuentes) for c in raw.hechos))
    discarded = sum(d.seccion != "resumen" for d in result.final.descartes)
    cited = {s for section in (result.final.hechos, result.final.hipotesis, result.final.contradicciones)
             for claim in section for s in claim.fuentes}
    covered = None if expected is None else bool(cited & expected)
    return QuestionScore(name, checkable, discarded, covered, result.timings["total"])


def summarize(scores: list[QuestionScore]) -> dict:
    checkable = sum(s.checkable for s in scores)
    discarded = sum(s.discarded for s in scores)
    with_expected = [s for s in scores if s.covered is not None]
    return {
        "surviving": 1 - discarded / checkable if checkable else 1.0,
        "checkable": checkable, "discarded": discarded,
        "coverage": sum(s.covered for s in with_expected) / len(with_expected) if with_expected else 1.0,
        "covered": f"{sum(s.covered for s in with_expected)}/{len(with_expected)}",
        "seconds": sum(s.seconds for s in scores) / len(scores),
    }


def main() -> int:
    service = build_default_service()
    service.ask("¿Cuál es la capital de Francia?")          # warm-up: loads the models
    table: dict[str, list[QuestionScore]] = {}
    for mode in MODES:
        table[mode] = []
        for name, question, expected in QUESTIONS:
            result = service.ask(question, mode=mode)
            table[mode].append(score(name, expected, result))
            s = table[mode][-1]
            print(f"  {mode:9} {name:14} checkable={s.checkable} descartadas={s.discarded} "
                  f"cobertura={'-' if s.covered is None else int(s.covered)} {s.seconds:.1f}s", flush=True)
    print(f"\n{'modo':10} {'afirmaciones válidas':>21} {'cobertura':>10} {'tiempo medio':>13}")
    summaries = {mode: summarize(scores) for mode, scores in table.items()}
    for mode, m in summaries.items():
        print(f"{mode:10} {m['surviving']:>14.0%} ({m['checkable'] - m['discarded']}/{m['checkable']}) "
              f"{m['covered']:>7} {m['seconds']:>12.1f}s")
    a, b = (summaries[m] for m in MODES)
    if abs(a["surviving"] - b["surviving"]) < 1e-9:
        winner = MODES[0] if a["seconds"] <= b["seconds"] else MODES[1]
        print(f"\nEmpate en afirmaciones válidas; gana por tiempo: {winner}")
    else:
        best = MODES[0] if a["surviving"] > b["surviving"] else MODES[1]
        other = MODES[1] if best == MODES[0] else MODES[0]
        ok = summaries[best]["coverage"] >= summaries[other]["coverage"]
        winner = best if ok else other
        print(f"\nMás afirmaciones válidas: {best}; cobertura {'no es peor' if ok else 'ES PEOR'} -> gana: {winner}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
