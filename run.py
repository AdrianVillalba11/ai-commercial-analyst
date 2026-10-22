"""
ARRANQUE DE LA APLICACIÓN WEB

Qué hace: levanta el servidor y abre la página en el navegador.
Con qué se conecta: importa la aplicación de src/analyst/api/main.py. No requiere instalar el paquete.

Uso:
    python run.py                 # http://127.0.0.1:8000
    python run.py --port 8080 --no-browser
"""
import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Permite importar `analyst` sin instalarlo: se agrega la carpeta src al path.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import uvicorn  # noqa: E402

from analyst.api.main import app  # noqa: E402

HOST = "127.0.0.1"   # solo este computador puede acceder a la aplicación


def main() -> int:
    """Lee los argumentos, anuncia la dirección, abre el navegador a los 4 s y arranca el servidor."""
    parser = argparse.ArgumentParser(description="Analista comercial de Brío")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="No abrir el navegador automáticamente")
    args = parser.parse_args()
    url = f"http://{HOST}:{args.port}"
    print(f"Cargando datos, documentos e índice… la aplicación quedará disponible en {url}")
    print("Para detenerla: Ctrl + C")
    if not args.no_browser:
        threading.Thread(target=lambda: (time.sleep(4), webbrowser.open(url)), daemon=True).start()
    uvicorn.run(app, host=HOST, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
