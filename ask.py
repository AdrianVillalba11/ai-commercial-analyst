"""Launcher for the command line, so no installation step is needed.

    python ask.py "¿Cuál fue la región con mayor caída de Brío?"
    python ask.py                      # interactive mode
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from analyst.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
