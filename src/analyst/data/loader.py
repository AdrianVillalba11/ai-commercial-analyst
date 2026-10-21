"""Load the sales CSV. Everything is read as text so bad values are not silently converted."""
from pathlib import Path

import pandas as pd

from analyst.config import settings
from analyst.data.cleaning import CleanResult, clean_sales

SALES_FILENAME = "02_Ventas_2026.csv"


def load_raw_sales(path: Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else settings.raw_data_dir / SALES_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de ventas: {path}")
    # utf-8-sig removes the BOM; keep_default_na=False keeps 'N/D' or '' as text
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def load_clean_sales(path: Path | None = None) -> CleanResult:
    return clean_sales(load_raw_sales(path))
