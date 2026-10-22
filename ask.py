"""
LANZADOR DE LA LÍNEA DE COMANDOS

Qué hace: ejecuta src/analyst/cli.py sin necesidad de instalar nada.

Uso:
    python ask.py "¿Cuál fue la región con mayor caída de Brío?"
    python ask.py                      # modo interactivo
"""
import sys
from pathlib import Path

# Permite importar `analyst` sin instalarlo: se agrega la carpeta src al path.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from analyst.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
