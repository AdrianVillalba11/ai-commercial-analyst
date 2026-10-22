"""Generates the execution evidence in docs/evidencia/.

    python scripts/generate_evidence.py             # both files
    python scripts/generate_evidence.py --tests     # only the test run
    python scripts/generate_evidence.py --answers   # only the reference answers

Needs Ollama running with the configured models (see README.md). The answers file is the output of a
real run on the date it states; it is not a set of fixed answers.
"""
import argparse
import datetime
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from analyst.config import settings  # noqa: E402

EVIDENCE_DIR = ROOT / "docs" / "evidencia"
QUESTIONS = [
    ("1 · Cuantitativa", "¿Cuál fue la región con mayor caída porcentual en ventas netas de Brío entre T1 y T2 de 2026? "
                         "Muestra los valores y el cálculo."),
    ("2 · Integrada (cifras y documentos)", "¿Qué hallazgos de consumidor y observaciones comerciales podrían ayudar a "
                                            "explicar el deterioro de Brío?"),
    ("3 · Contraste con el reporte comercial", "¿Son consistentes las principales conclusiones del reporte comercial "
                                               "con los datos de ventas?"),
    ("4 · Causalidad", "¿Podemos afirmar que las promociones de la competencia causaron la caída de ventas? "
                       "¿Qué evidencia lo sustenta o limita?"),
]


def environment_lines() -> list[str]:
    return [
        f"- Fecha: {datetime.datetime.now():%Y-%m-%d %H:%M}",
        f"- Python: {platform.python_version()} · Sistema: {platform.system()} {platform.release()}",
        f"- Modelo de generación: {settings.llm_model} · Modelo de embeddings: {settings.embedding_model} · "
        f"Búsqueda por defecto: {settings.retrieval_mode}",
    ]


def write_tests() -> int:
    command = [sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider"]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    started = datetime.datetime.now()
    run = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
    seconds = (datetime.datetime.now() - started).total_seconds()
    header = ["# Evidencia de ejecución de las pruebas automatizadas", "", *environment_lines(),
              "Comando: python -m pytest -v", f"Código de salida: {run.returncode} (0 = todas pasan)",
              f"Duración total: {seconds:.0f} s", "",
              "Las pruebas marcadas `ollama` usan los modelos reales; sin Ollama se saltan solas.", "", "=" * 78, ""]
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    output = run.stdout.replace(str(ROOT), ".")          # do not leak the absolute path of the machine
    (EVIDENCE_DIR / "pytest_completo.txt").write_text("\n".join(header) + output, encoding="utf-8")
    summary = [line for line in run.stdout.splitlines() if " passed" in line or " failed" in line]
    print("Pruebas:", summary[-1] if summary else f"código {run.returncode}")
    return run.returncode


def claims(title: str, items) -> list[str]:
    if not items:
        return []
    lines = ["", f"**{title}**", ""]
    for item in items:
        sources = f" _[{', '.join(item.fuentes)}; {item.origen}]_" if item.fuentes else f" _[{item.origen}]_"
        lines.append(f"- {item.texto}{sources}")
    return lines


def format_question(title: str, question: str, result) -> list[str]:
    final = result.final
    origin = "" if final.resumen_origen == "modelo" else " _(redactado por el sistema a partir de los cálculos)_"
    lines = [f"## Pregunta {title}", "", f"> {question}", "",
             f"**Configuración:** modelo `{result.model}` · búsqueda `{result.mode}` · plan `{result.plan.tipo}` "
             f"({'modelo' if result.plan_source == 'llm' else 'plan por defecto'}) · {result.timings['total']:.1f} s", "",
             f"**Resumen:** {final.resumen}{origin}"]
    lines += claims("Hechos", final.hechos)
    if result.contrast:
        lines += ["", "**Reporte comercial frente a los datos** (variación T2 vs T1 por región)", "",
                  "| Región | Reporte | Datos | Diferencia | Resultado |", "|---|---:|---:|---:|---|"]
        for row in result.contrast["rows"]:
            computed = f"{row['computed']:.2f}".replace(".", ",")
            difference = f"{row['diff_pp']:+.2f}".replace(".", ",")
            lines.append(f"| {row['region']} | {row['reported']:+.0f} % | {computed} % | {difference} pp | "
                         f"{'Coincide' if row['coincide'] else 'No coincide'} |")
    lines += claims("Contradicciones entre fuentes", final.contradicciones)
    lines += claims("Hipótesis (no demostradas)", final.hipotesis)
    lines += claims("Información insuficiente", final.insuficiente)
    if final.descartes:
        lines += ["", "**Descartado por el validador** (afirmaciones del modelo que no pasaron las verificaciones)", ""]
        lines += [f"- «{d.texto}» → {d.motivo}" for d in final.descartes]
    if result.evidence:
        lines += ["", "**Evidencia consultada:** " + ", ".join(e.chunk_id for e in result.evidence)]
    notices = result.warnings + final.avisos
    if notices:
        lines += ["", "**Avisos:**"] + [f"- {n}" for n in notices]
    lines += ["", "**Cálculos**", ""]
    for block in result.calc:
        lines += ["```", block.text, "```", ""]
    return lines


def write_answers() -> int:
    from analyst.service import build_default_service
    service = build_default_service()
    service.warm_up()
    document = ["# Respuestas de referencia del sistema", "", *environment_lines(), "",
                "> **Qué es este archivo:** la salida de una ejecución real del sistema en la fecha indicada. "
                "**No son respuestas fijas.** Un modelo de lenguaje local puede redactar de forma distinta en cada "
                "ejecución. Lo que no cambia son las cifras, el veredicto del contraste con el reporte y las "
                "limitaciones de causalidad, porque salen de código y no del modelo. Para regenerarlo: "
                "`python scripts/generate_evidence.py --answers`.", ""]
    for title, question in QUESTIONS:
        print("Preguntando:", title, flush=True)
        document += format_question(title, question, service.ask(question))
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / "respuestas_referencia.md").write_text("\n".join(document) + "\n", encoding="utf-8")
    print("Respuestas guardadas en docs/evidencia/respuestas_referencia.md")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Genera la evidencia de ejecución")
    parser.add_argument("--tests", action="store_true", help="Solo la ejecución de pruebas")
    parser.add_argument("--answers", action="store_true", help="Solo las respuestas de referencia")
    args = parser.parse_args()
    both = not (args.tests or args.answers)
    status = 0
    if args.answers or both:
        status |= write_answers()
    if args.tests or both:
        status |= write_tests()
    return status


if __name__ == "__main__":
    sys.exit(main())
