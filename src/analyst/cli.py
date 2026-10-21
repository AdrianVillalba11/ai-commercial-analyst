"""Command line: ask a question and read the structured answer.

    python ask.py "¿Cuál fue la región con mayor caída de Brío?"
    python ask.py                    # interactive: one question per line, empty line to exit
"""
import argparse
import json
import sys
from typing import Callable, TextIO

from analyst.errors import OllamaError
from analyst.service import AnalystService, AskResult, build_default_service

SECTIONS = [("hechos", "HECHOS (respaldados por cálculos o documentos)"),
            ("hipotesis", "HIPÓTESIS (plausibles, no demostradas)"),
            ("contradicciones", "CONTRADICCIONES ENTRE FUENTES")]


def format_result(result: AskResult) -> str:
    answer = result.answer
    lines = ["", f"RESUMEN: {answer.resumen}"]
    for field_name, title in SECTIONS:
        claims = getattr(answer, field_name)
        if claims:
            lines += ["", title + ":"] + [f"  - {c.texto}  [{', '.join(c.fuentes)}]" for c in claims]
    if answer.insuficiente:
        lines += ["", "INFORMACIÓN INSUFICIENTE:"] + [f"  - {g.texto}" for g in answer.insuficiente]
    if result.calc:
        lines += ["", "CÁLCULOS:"] + [f"  {b.text}".replace("\n", "\n  ") for b in result.calc]
    if result.evidence:
        lines += ["", "EVIDENCIA USADA: " + ", ".join(e.chunk_id for e in result.evidence)]
    if result.warnings:
        lines += ["", "AVISOS:"] + [f"  ! {w}" for w in result.warnings]
    steps = "  ".join(f"{name} {secs:.1f}s" for name, secs in result.timings.items())
    lines += ["", f"[{result.model} | modo {result.mode} | plan: {result.plan_source} ({result.plan.tipo}) | {steps}]"]
    return "\n".join(lines)


def run_question(service: AnalystService, question: str, mode: str, as_json: bool, out: TextIO) -> int:
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
    parser = argparse.ArgumentParser(prog="analyst.cli", description="Asistente de análisis comercial de Brío")
    parser.add_argument("question", nargs="*", help="La pregunta (si se omite, modo interactivo)")
    parser.add_argument("--mode", choices=["hybrid", "semantic", "lexical"], default="hybrid")
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
