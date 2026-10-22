"""
EVALUACIÓN CORTA DEL FLUJO COMPLETO

Qué hace: compara la búsqueda híbrida con la semántica sobre 8 preguntas, usando llama3 de extremo a extremo.
Con qué se conecta: usa service.py; sus resultados se resumen en docs/EVALUACIONES.md.

Ejecutar:  python scripts/evaluate_answers.py        (unos 3 minutos; necesita Ollama con los modelos)

Las métricas y la regla de decisión se fijaron ANTES de correrlo y no se ajustan a los resultados:
  1. Métrica principal: proporción de las afirmaciones verificables del modelo que sobreviven al validador
     (verificables = hipótesis + hechos basados en documentos + contradicciones; los hechos que solo repiten
     un cálculo no cuentan porque el código ya los enuncia).
  2. Cobertura: en las preguntas con evidencia esperada, si la respuesta final cita al menos uno de los
     ids esperados.
  3. Desempate: tiempo promedio.
Regla: se elige el modo con mayor proporción de afirmaciones que sobreviven, siempre que su cobertura no sea menor.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analyst.service import AskResult, build_default_service  # noqa: E402

# =====================================================================
# BLOQUE: Preguntas de la evaluación
# Cada una: (nombre, pregunta, ids de evidencia esperados o None si no aplica).
# =====================================================================
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


# =====================================================================
# BLOQUE: Puntaje por pregunta y resumen
# =====================================================================
@dataclass
class QuestionScore:
    """Resultado de una pregunta: afirmaciones verificables, descartadas, cobertura y tiempo."""

    name: str
    checkable: int
    discarded: int
    covered: bool | None
    seconds: float


def score(name: str, expected: set[str] | None, result: AskResult) -> QuestionScore:
    """Cuenta qué afirmaciones eran verificables, cuántas descartó el validador y si se citó evidencia esperada."""
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
    """Agrega los puntajes de un modo: proporción que sobrevive, cobertura y tiempo medio."""
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


# =====================================================================
# BLOQUE: Ejecución y decisión
# =====================================================================
def main() -> int:
    """Corre las preguntas en cada modo, imprime la tabla y aplica la regla de decisión."""
    service = build_default_service()
    service.ask("¿Cuál es la capital de Francia?")          # calentamiento: carga los modelos
    table: dict[str, list[QuestionScore]] = {}
    for mode in MODES:
        table[mode] = []
        for name, question, expected in QUESTIONS:
            result = service.ask(question, mode=mode)
            table[mode].append(score(name, expected, result))
            s = table[mode][-1]
            print(f"  {mode:9} {name:14} checkable={s.checkable} descartadas={s.discarded} "
                  f"cobertura={'-' if s.covered is None else int(s.covered)} {s.seconds:.1f}s", flush=True)
    # Tabla resumen por modo.
    print(f"\n{'modo':10} {'afirmaciones válidas':>21} {'cobertura':>10} {'tiempo medio':>13}")
    summaries = {mode: summarize(scores) for mode, scores in table.items()}
    for mode, m in summaries.items():
        print(f"{mode:10} {m['surviving']:>14.0%} ({m['checkable'] - m['discarded']}/{m['checkable']}) "
              f"{m['covered']:>7} {m['seconds']:>12.1f}s")
    # Regla de decisión fijada de antemano.
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
