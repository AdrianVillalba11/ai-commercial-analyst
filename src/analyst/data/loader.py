"""
CARGA DEL CSV DE VENTAS

Qué hace: lee el archivo de ventas y lo entrega limpio (usando cleaning.py).
Con qué se conecta: lo usa el servicio al arrancar y las pruebas.
Idea clave: todo se lee como TEXTO para que un valor raro (por ejemplo "N/D") no se convierta
en silencio en un número o en vacío; la limpieza decide qué hacer con él.
"""
from pathlib import Path

import pandas as pd

from analyst.config import settings
from analyst.data.cleaning import CleanResult, clean_sales

SALES_FILENAME = "02_Ventas_2026.csv"


def load_raw_sales(path: Path | None = None) -> pd.DataFrame:
    """Devuelve el CSV tal cual, todo como texto."""
    path = Path(path) if path else settings.raw_data_dir / SALES_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de ventas: {path}")
    # utf-8-sig quita el BOM del inicio; keep_default_na=False conserva 'N/D' o '' como texto
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def load_clean_sales(path: Path | None = None) -> CleanResult:
    """Carga el CSV y aplica las reglas de limpieza; devuelve la tabla limpia y el reporte."""
    return clean_sales(load_raw_sales(path))
