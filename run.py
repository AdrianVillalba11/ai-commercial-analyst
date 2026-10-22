"""Start the web application and open it in the browser.

    python run.py                 # http://127.0.0.1:8000
    python run.py --port 8080 --no-browser
"""
import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import uvicorn  # noqa: E402

from analyst.api.main import app  # noqa: E402

HOST = "127.0.0.1"   # only this computer can reach the application


def main() -> int:
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
