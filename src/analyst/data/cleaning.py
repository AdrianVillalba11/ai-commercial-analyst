"""
LIMPIEZA DE LA TABLA DE VENTAS

Qué hace: aplica una serie de reglas al CSV crudo y devuelve una tabla confiable más el
reporte de cada corrección o exclusión.
Con qué se conecta: la llama data/loader.py; el reporte lo llena y lo consume quality_report.py.
Principios:
- Las reglas son genéricas (sin ids de registro escritos a mano): sirven para un archivo parecido.
- Solo se corrige lo que tiene UNA lectura posible (espacios sobrantes, mayúsculas/minúsculas).
- Si un valor no se puede interpretar con certeza, la fila se EXCLUYE; nunca se adivina.
- Toda corrección o exclusión queda registrada en el reporte de calidad.
"""
import re
from dataclasses import dataclass

import pandas as pd

from analyst.data.quality_report import CORRECTED, EXCLUDED, Issue, QualityReport

# =====================================================================
# BLOQUE: Catálogos oficiales y constantes
# Valores válidos de cada dimensión; sirven para normalizar y para validar.
# =====================================================================
BRANDS = ["Brio", "Luma", "Nativa", "Rizo", "Senda"]
REGIONS = ["Norte", "Centro", "Sur", "Oriente", "Occidente"]
CHANNELS = ["Tradicional", "Moderno"]
PRESENTATIONS = ["Individual", "Familiar"]
VALID_MONTHS = [f"2026-{m:02d}" for m in range(1, 7)]

# Prefijo y letra que debe tener el id de punto de venta según región y canal (ej. OC-T-001).
REGION_PREFIX = {"Norte": "NO", "Centro": "CE", "Sur": "SU", "Oriente": "OR", "Occidente": "OC"}
CHANNEL_LETTER = {"Tradicional": "T", "Moderno": "M"}
_PDV_PATTERN = re.compile(r"^([A-Z]{2})-([TM])-\d+$")

# Columnas categóricas y su lista de valores permitidos (regla 1).
_CATEGORICAL = {
    "marca": BRANDS,
    "region": REGIONS,
    "canal": CHANNELS,
    "presentacion": PRESENTATIONS,
}
# Columnas de texto a las que se les recortan espacios (regla 1a).
_TEXT_COLUMNS = ["registro_id", "mes", "punto_venta_id"]
# Columnas que definen que dos filas son "idénticas" (regla 2).
_DEDUP_COLUMNS = [
    "registro_id", "mes", "marca", "region", "canal",
    "punto_venta_id", "presentacion", "unidades", "venta_neta_cop",
]


# =====================================================================
# BLOQUE: Punto de entrada de la limpieza
# Aplica las reglas en orden. Cada regla marca filas a excluir en `excluded` y deja
# constancia en el reporte; al final se eliminan las filas excluidas de una sola vez.
# =====================================================================
@dataclass
class CleanResult:
    """Resultado de la limpieza: tabla limpia + reporte de calidad."""

    data: pd.DataFrame
    report: QualityReport


def clean_sales(raw: pd.DataFrame) -> CleanResult:
    """Aplica todas las reglas a la tabla cruda (todas las columnas como texto)."""
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

    # Se eliminan las filas excluidas y se convierten las columnas numéricas ya validadas.
    clean = df.drop(index=list(excluded)).reset_index(drop=True)
    clean["unidades"] = _to_numeric_column(clean["unidades"])
    clean["venta_neta_cop"] = _to_numeric_column(clean["venta_neta_cop"])
    report.rows_out = len(clean)
    return CleanResult(data=clean, report=report)


# =====================================================================
# BLOQUE: Ayudantes para registrar hallazgos
# =====================================================================
def _brand_of(df: pd.DataFrame, idx) -> str:
    """Marca de la fila con su ortografía oficial, para atribuir el hallazgo a la marca correcta."""
    value = str(df.at[idx, "marca"]).strip()
    lookup = {b.casefold(): b for b in BRANDS}
    return lookup.get(value.casefold(), value)


def _issue(df, idx, rule, field, original, action, detail) -> Issue:
    """Arma el registro de un hallazgo (qué fila, qué regla, qué valor, qué se hizo)."""
    return Issue(
        registro_id=str(df.at[idx, "registro_id"]),
        rule=rule,
        field=field,
        original_value=repr(original),
        action=action,
        detail=detail,
        marca=_brand_of(df, idx),
    )


# =====================================================================
# BLOQUE: Reglas de limpieza (una función por regla)
# =====================================================================
def _rule_normalize_text(df, report):
    """Regla 1a: quita espacios al inicio y al final en las columnas tipo identificador."""
    for col in _TEXT_COLUMNS:
        stripped = df[col].str.strip()
        for idx in df.index[stripped != df[col]]:
            report.add(_issue(df, idx, "espacios_sobrantes", col, df.at[idx, col], CORRECTED,
                              "Se eliminaron espacios al inicio o al final."))
        df[col] = stripped


def _rule_normalize_categories(df, report, excluded):
    """Regla 1: lleva los valores categóricos a su ortografía oficial (ignora mayúsculas y espacios)."""
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
    """Regla 2: filas exactamente duplicadas -> se conserva la primera y se excluyen las repetidas."""
    for idx in df.index[df.duplicated(subset=_DEDUP_COLUMNS, keep="first")]:
        excluded.add(idx)
        report.add(_issue(df, idx, "duplicado_exacto", "registro_id", df.at[idx, "registro_id"],
                          EXCLUDED, "Fila idéntica a una anterior; se conservó la primera."))


def _rule_conflicting_ids(df, report, excluded):
    """Regla 2b: mismo registro_id con contenido distinto -> se excluyen todas (no se sabe cuál es la correcta)."""
    unique = df.drop(index=df.index[df.duplicated(subset=_DEDUP_COLUMNS, keep="first")])
    for idx in unique.index[unique.duplicated(subset=["registro_id"], keep=False)]:
        excluded.add(idx)
        report.add(_issue(df, idx, "id_conflictivo", "registro_id", df.at[idx, "registro_id"],
                          EXCLUDED, "Mismo registro_id con contenido distinto; no se sabe cuál es correcto."))


def _rule_valid_month(df, report, excluded):
    """Regla 3: el mes debe ser uno de 2026-01 a 2026-06."""
    for idx in df.index[~df["mes"].isin(VALID_MONTHS)]:
        excluded.add(idx)
        report.add(_issue(df, idx, "mes_invalido", "mes", df.at[idx, "mes"], EXCLUDED,
                          "Mes inexistente o fuera del periodo 2026-01 a 2026-06; no se puede asignar a T1 o T2."))


def _rule_numeric_values(df, report, excluded):
    """Regla 4: unidades y venta_neta_cop deben existir y ser numéricos."""
    for col in ["unidades", "venta_neta_cop"]:
        numeric = pd.to_numeric(df[col].str.strip(), errors="coerce")
        for idx in df.index[numeric.isna() | numeric.isin([float("inf"), float("-inf")])]:
            excluded.add(idx)
            report.add(_issue(df, idx, "valor_no_numerico", col, df.at[idx, col], EXCLUDED,
                              f"{col} vacío o no numérico; no se puede sumar."))


def _rule_negative_values(df, report, excluded):
    """Regla 5: unidades o ventas negativas se excluyen (no hay política para interpretar devoluciones)."""
    for col in ["unidades", "venta_neta_cop"]:
        numeric = pd.to_numeric(df[col].str.strip(), errors="coerce")
        for idx in df.index[numeric < 0]:
            excluded.add(idx)
            report.add(_issue(df, idx, "valor_negativo", col, df.at[idx, col], EXCLUDED,
                              f"{col} negativo; no hay política para interpretarlo."))


def _rule_point_of_sale_consistency(df, report, excluded):
    """Regla 6: el prefijo del punto de venta debe coincidir con la región y el canal (ej. OC-T-001)."""
    for idx in df.index:
        region, canal = df.at[idx, "region"], df.at[idx, "canal"]
        if region not in REGION_PREFIX or canal not in CHANNEL_LETTER:
            continue  # ya excluida por la regla 1
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


# =====================================================================
# BLOQUE: Conversión final a número
# =====================================================================
def _to_numeric_column(series: pd.Series) -> pd.Series:
    """Convierte la columna validada a número: entero si todos los valores lo son, decimal si no."""
    numeric = pd.to_numeric(series.str.strip())
    if (numeric % 1 == 0).all():
        return numeric.astype("int64")
    return numeric.astype("float64")
