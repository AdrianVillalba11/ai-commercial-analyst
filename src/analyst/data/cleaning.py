"""Cleaning rules for the sales table.

Principles:
- Rules are generic (no hard-coded record ids), so they work on any similar file.
- Only unambiguous problems are corrected (e.g. stray spaces, letter case).
- If a value cannot be interpreted with certainty, the row is excluded, never guessed.
- Every correction or exclusion is recorded in the quality report.
"""
import re
from dataclasses import dataclass

import pandas as pd

from analyst.data.quality_report import CORRECTED, EXCLUDED, Issue, QualityReport

BRANDS = ["Brio", "Luma", "Nativa", "Rizo", "Senda"]
REGIONS = ["Norte", "Centro", "Sur", "Oriente", "Occidente"]
CHANNELS = ["Tradicional", "Moderno"]
PRESENTATIONS = ["Individual", "Familiar"]
VALID_MONTHS = [f"2026-{m:02d}" for m in range(1, 7)]

REGION_PREFIX = {"Norte": "NO", "Centro": "CE", "Sur": "SU", "Oriente": "OR", "Occidente": "OC"}
CHANNEL_LETTER = {"Tradicional": "T", "Moderno": "M"}
_PDV_PATTERN = re.compile(r"^([A-Z]{2})-([TM])-\d+$")

_CATEGORICAL = {
    "marca": BRANDS,
    "region": REGIONS,
    "canal": CHANNELS,
    "presentacion": PRESENTATIONS,
}
_TEXT_COLUMNS = ["registro_id", "mes", "punto_venta_id"]
_DEDUP_COLUMNS = [
    "registro_id", "mes", "marca", "region", "canal",
    "punto_venta_id", "presentacion", "unidades", "venta_neta_cop",
]


@dataclass
class CleanResult:
    data: pd.DataFrame
    report: QualityReport


def clean_sales(raw: pd.DataFrame) -> CleanResult:
    """Apply all cleaning rules to a raw sales table (all columns as text)."""
    df = raw.copy()
    report = QualityReport(rows_in=len(df))
    excluded: set = set()

    _rule_normalize_text(df, report)
    _rule_normalize_categories(df, report, excluded)
    _rule_exact_duplicates(df, report, excluded)
    _rule_conflicting_ids(df, report, excluded)
    _rule_valid_month(df, report, excluded)
    _rule_numeric_values(df, report, excluded)
    _rule_negative_values(df, report, excluded)
    _rule_point_of_sale_consistency(df, report, excluded)

    clean = df.drop(index=list(excluded)).reset_index(drop=True)
    clean["unidades"] = _to_numeric_column(clean["unidades"])
    clean["venta_neta_cop"] = _to_numeric_column(clean["venta_neta_cop"])
    report.rows_out = len(clean)
    return CleanResult(data=clean, report=report)


def _brand_of(df: pd.DataFrame, idx) -> str:
    value = str(df.at[idx, "marca"]).strip()
    lookup = {b.casefold(): b for b in BRANDS}
    return lookup.get(value.casefold(), value)


def _issue(df, idx, rule, field, original, action, detail) -> Issue:
    return Issue(
        registro_id=str(df.at[idx, "registro_id"]),
        rule=rule,
        field=field,
        original_value=repr(original),
        action=action,
        detail=detail,
        marca=_brand_of(df, idx),
    )


def _rule_normalize_text(df, report):
    """Rule 1a: strip surrounding whitespace in identifier-like columns."""
    for col in _TEXT_COLUMNS:
        stripped = df[col].str.strip()
        for idx in df.index[stripped != df[col]]:
            report.add(_issue(df, idx, "espacios_sobrantes", col, df.at[idx, col], CORRECTED,
                              "Se eliminaron espacios al inicio o al final."))
        df[col] = stripped


def _rule_normalize_categories(df, report, excluded):
    """Rule 1: map categorical values to the official spelling (case/space insensitive)."""
    for col, allowed in _CATEGORICAL.items():
        lookup = {v.casefold(): v for v in allowed}
        canonical = df[col].str.strip().str.casefold().map(lookup)
        for idx in df.index:
            original, canon = df.at[idx, col], canonical.at[idx]
            if pd.isna(canon):
                excluded.add(idx)
                report.add(_issue(df, idx, "valor_no_permitido", col, original, EXCLUDED,
                                  f"'{original}' no es un valor válido de {col}: {allowed}."))
            elif canon != original:
                report.add(_issue(df, idx, "normalizacion_texto", col, original, CORRECTED,
                                  f"Se unificó a '{canon}' (mayúsculas o espacios)."))
                df.at[idx, col] = canon


def _rule_exact_duplicates(df, report, excluded):
    """Rule 2: exact duplicate rows -> keep the first, exclude the repeats."""
    for idx in df.index[df.duplicated(subset=_DEDUP_COLUMNS, keep="first")]:
        excluded.add(idx)
        report.add(_issue(df, idx, "duplicado_exacto", "registro_id", df.at[idx, "registro_id"],
                          EXCLUDED, "Fila idéntica a una anterior; se conservó la primera."))


def _rule_conflicting_ids(df, report, excluded):
    """Rule 2b: same registro_id with different content -> exclude all (cannot know which is right)."""
    unique = df.drop(index=df.index[df.duplicated(subset=_DEDUP_COLUMNS, keep="first")])
    for idx in unique.index[unique.duplicated(subset=["registro_id"], keep=False)]:
        excluded.add(idx)
        report.add(_issue(df, idx, "id_conflictivo", "registro_id", df.at[idx, "registro_id"],
                          EXCLUDED, "Mismo registro_id con contenido distinto; no se sabe cuál es correcto."))


def _rule_valid_month(df, report, excluded):
    """Rule 3: month must be one of 2026-01 .. 2026-06."""
    for idx in df.index[~df["mes"].isin(VALID_MONTHS)]:
        excluded.add(idx)
        report.add(_issue(df, idx, "mes_invalido", "mes", df.at[idx, "mes"], EXCLUDED,
                          "Mes inexistente o fuera del periodo 2026-01 a 2026-06; no se puede asignar a T1 o T2."))


def _rule_numeric_values(df, report, excluded):
    """Rule 4: unidades and venta_neta_cop must be present and numeric."""
    for col in ["unidades", "venta_neta_cop"]:
        numeric = pd.to_numeric(df[col].str.strip(), errors="coerce")
        for idx in df.index[numeric.isna() | numeric.isin([float("inf"), float("-inf")])]:
            excluded.add(idx)
            report.add(_issue(df, idx, "valor_no_numerico", col, df.at[idx, col], EXCLUDED,
                              f"{col} vacío o no numérico; no se puede sumar."))


def _rule_negative_values(df, report, excluded):
    """Rule 5: negative units or sales are excluded (no policy to interpret returns)."""
    for col in ["unidades", "venta_neta_cop"]:
        numeric = pd.to_numeric(df[col].str.strip(), errors="coerce")
        for idx in df.index[numeric < 0]:
            excluded.add(idx)
            report.add(_issue(df, idx, "valor_negativo", col, df.at[idx, col], EXCLUDED,
                              f"{col} negativo; no hay política para interpretarlo."))


def _rule_point_of_sale_consistency(df, report, excluded):
    """Rule 6: punto_venta_id prefix must agree with region and channel (e.g. OC-T-001)."""
    for idx in df.index:
        region, canal = df.at[idx, "region"], df.at[idx, "canal"]
        if region not in REGION_PREFIX or canal not in CHANNEL_LETTER:
            continue  # already excluded by rule 1b
        pdv = df.at[idx, "punto_venta_id"]
        match = _PDV_PATTERN.match(pdv)
        if not match:
            problem = "formato inválido (se esperaba XX-T-000)"
        elif match.group(1) != REGION_PREFIX[region]:
            problem = f"el prefijo '{match.group(1)}' no corresponde a la región {region}"
        elif match.group(2) != CHANNEL_LETTER[canal]:
            problem = f"la letra '{match.group(2)}' no corresponde al canal {canal}"
        else:
            continue
        excluded.add(idx)
        report.add(_issue(df, idx, "pdv_inconsistente", "punto_venta_id", pdv, EXCLUDED,
                          f"punto_venta_id {problem}; la dimensión es dudosa."))


def _to_numeric_column(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series.str.strip())
    if (numeric % 1 == 0).all():
        return numeric.astype("int64")
    return numeric.astype("float64")
