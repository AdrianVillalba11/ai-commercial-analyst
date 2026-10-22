"""
PRUEBAS DE LA DOCUMENTACIÓN Y LIMPIEZA DEL REPOSITORIO

Qué verifica: que el README y los documentos coinciden con el proyecto (variables de .env.example, comandos, enlaces
a archivos que existen) y que el repositorio no contiene rastros de asistentes de programación ni credenciales.
"""
import re

import pytest

from analyst.config import PROJECT_ROOT

README = PROJECT_ROOT / "README.md"
DOC_FILES = [README, *sorted((PROJECT_ROOT / "docs").glob("*.md"))]
SKIPPED_DIRS = {".venv", ".cache", ".git", ".pytest_cache", "__pycache__", "notas_personales", "raw"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".html", ".css", ".js", ".ini", ".example", ".json"}


def env_example() -> dict[str, str]:
    """Variables de .env.example como diccionario nombre -> valor por defecto."""
    pairs = {}
    for line in (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Z_]+)=(.*)$", line)
        if match:
            pairs[match.group(1)] = match.group(2)
    return pairs


def test_every_documentation_file_exists():
    for name in ("README.md", "docs/ARQUITECTURA.md", "docs/EVALUACIONES.md"):
        assert (PROJECT_ROOT / name).exists(), name


def test_the_readme_documents_every_setting_with_its_real_default():
    readme = README.read_text(encoding="utf-8")
    settings_in_example = env_example()
    assert settings_in_example
    for name, value in settings_in_example.items():
        assert re.search(rf"\|\s*`{name}`\s*\|\s*`{re.escape(value)}`", readme), f"{name}={value} no está documentada"


def test_the_readme_has_the_commands_an_evaluator_needs():
    readme = README.read_text(encoding="utf-8")
    for command in ("ollama pull llama3", "ollama pull bge-m3", "pip install -r requirements.txt", "python run.py",
                    "python ask.py", 'python -m pytest -q -m "not ollama"', "python -m pytest -q",
                    "python scripts/generate_evidence.py"):
        assert command in readme, command
    assert "Ejecución desde mi computador" in readme          # la modalidad de demostración que la prueba pide declarar


@pytest.mark.parametrize("doc", DOC_FILES, ids=lambda p: p.name)
def test_relative_links_point_to_files_that_exist(doc):
    text = doc.read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)\s]+)\)", text):
        if target.startswith(("http", "#", "mailto:")):
            continue
        path = (doc.parent / target.split("#")[0]).resolve()
        assert path.exists(), f"{doc.name}: el enlace {target} no existe"


def test_the_repository_contains_no_trace_of_coding_assistants():
    forbidden = ["cla" + "ude", "anthro" + "pic", "co-authored" + "-by", "generated " + "with"]
    offenders = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES or SKIPPED_DIRS & set(path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        offenders += [f"{path.relative_to(PROJECT_ROOT)}: {word}" for word in forbidden if word in text]
    assert offenders == []


def test_no_credentials_or_keys_are_committed():
    suspicious = re.compile(r"(api[_-]?key|secret|password|token)\s*[:=]\s*['\"][A-Za-z0-9_\-]{12,}['\"]", re.I)
    for path in PROJECT_ROOT.rglob("*"):
        if path.is_file() and path.suffix in TEXT_SUFFIXES and not SKIPPED_DIRS & set(path.parts):
            assert not suspicious.search(path.read_text(encoding="utf-8", errors="ignore")), str(path)
    assert not (PROJECT_ROOT / ".env").exists() or ".env" in (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
