"""
LÍNEA DE COMANDOS

Qué hace: permite hacer una pregunta desde la terminal y leer la respuesta estructurada.
Con qué se conecta: usa service.py (el mismo flujo que la web); se lanza con ask.py.

Ejemplos:
    python ask.py "¿Cuál fue la región con mayor caída de Brío?"
    python ask.py                    # interactivo: una pregunta por línea, línea vacía para salir
"""
import argparse
import json
import sys
from typing import Callable, TextIO

from analyst.errors import OllamaError
from analyst.service import AnalystService, AskResult, build_default_service

# =====================================================================
# BLOQUE: Formato de salida
# =====================================================================
SECTIONS = [("hechos", "HECHOS (respaldados por cálculos o documentos)"),
            ("hipotesis", "HIPÓTESIS (plausibles, no demostradas)"),
            ("contradicciones", "CONTRADICCIONES ENTRE FUENTES")]


def _claim_line(claim) -> str:
    """Una afirmación con sus fuentes entre corchetes."""
    sources = f"  [{', '.join(claim.fuentes)}]" if claim.fuentes else ""
    return f"  - {claim.texto}{sources}"


def format_result(result: AskResult) -> str:
    """Convierte el resultado en texto legible: resumen, secciones, descartes, cálculos, evidencia y avisos."""
    final = result.final
    written_by_code = "" if final.resumen_origen == "modelo" else "  (redactado por el sistema)"
    lines = ["", f"RESUMEN: {final.resumen}{written_by_code}"]
    for field_name, title in SECTIONS:
        claims = getattr(final, field_name)
        if claims:
            lines += ["", title + ":"] + [_claim_line(c) for c in claims]
    if final.insuficiente:
        lines += ["", "INFORMACIÓN INSUFICIENTE:"] + [_claim_line(c) for c in final.insuficiente]
    if final.descartes:
        lines += ["", f"DESCARTADO POR EL VALIDADOR ({len(final.descartes)}):"]
        lines += [f"  x {d.texto}\n      Motivo: {d.motivo}" for d in final.descartes]
    if result.calc:
        lines += ["", "CÁLCULOS:"] + [f"  {b.text}".replace("\n", "\n  ") for b in result.calc]
    if result.evidence:
        lines += ["", "EVIDENCIA USADA: " + ", ".join(e.chunk_id for e in result.evidence)]
    notices = result.warnings + final.avisos
    if notices:
        lines += ["", "AVISOS:"] + [f"  ! {w}" for w in notices]
    steps = "  ".join(f"{name} {secs:.1f}s" for name, secs in result.timings.items())
    lines += ["", f"[{result.model} | modo {result.mode} | plan: {result.plan_source} ({result.plan.tipo}) | {steps}]"]
    return "\n".join(lines)


# =====================================================================
# BLOQUE: Ejecución
# =====================================================================
def run_question(service: AnalystService, question: str, mode: str, as_json: bool, out: TextIO) -> int:
    """Hace una pregunta e imprime el resultado. Devuelve 0 si salió bien, 1 si la pregunta no es válida, 2 si falló Ollama."""
    try:
        result = service.ask(question, mode=mode)
    except ValueError as exc:
        print(f"Pregunta no válida: {exc}", file=out)
        return 1
    except OllamaError as exc:
        print(f"No se pudo responder: {exc}", file=out)
        return 2
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) if as_json else format_result(result), file=out)
    return 0


def main(argv: list[str] | None = None, service_factory: Callable[[], AnalystService] = build_default_service,
         out: TextIO | None = None) -> int:
    """Punto de entrada: con pregunta responde una vez; sin pregunta abre el modo interactivo."""
    parser = argparse.ArgumentParser(prog="analyst.cli", description="Asistente de análisis comercial de Brío")
    parser.add_argument("question", nargs="*", help="La pregunta (si se omite, modo interactivo)")
    parser.add_argument("--mode", choices=["hybrid", "semantic", "lexical"], default=None,
                        help="Búsqueda en documentos (por defecto, RETRIEVAL_MODE del .env)")
    parser.add_argument("--json", action="store_true", help="Imprime el resultado completo como JSON")
    args = parser.parse_args(argv)
    out = out or sys.stdout
    if hasattr(out, "reconfigure"):
        out.reconfigure(encoding="utf-8")
    try:
        service = service_factory()
    except OllamaError as exc:
        print(f"No se pudo iniciar: {exc}", file=out)
        return 2
    if args.question:
        return run_question(service, " ".join(args.question), args.mode, args.json, out)
    status = 0
    while True:
        try:
            question = input("\nPregunta (línea vacía para salir): ").strip()
        except EOFError:
            break
        if not question:
            break
        status = run_question(service, question, args.mode, args.json, out)
    return status


if __name__ == "__main__":
    sys.exit(main())
