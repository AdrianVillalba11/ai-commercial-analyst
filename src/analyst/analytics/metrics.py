"""
MÉTRICAS COMERCIALES (T1 vs T2)

Qué hace: calcula con pandas las comparaciones entre trimestres (totales y por región, canal,
presentación o punto de venta).
Con qué se conecta: lo usa guardrails/facts.py y el servicio; el LLM nunca calcula.
Idea clave: toda cifra de una respuesta sale de aquí. El modelo solo elige QUÉ análisis pedir
(dentro del catálogo cerrado de abajo); la aritmética es siempre de código.
Convención (01_Contexto_Negocio.md): T1 = ene-mar 2026, T2 = abr-jun 2026,
cambio absoluto = T2 - T1, cambio porcentual = (T2 / T1 - 1) x 100.
"""
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from analyst.data.cleaning import BRANDS, CHANNELS, PRESENTATIONS, REGIONS, VALID_MONTHS
from analyst.data.quality_report import EXCLUDED, QualityReport
from analyst.text import fold

# =====================================================================
# BLOQUE: Catálogo cerrado de análisis posibles
# =====================================================================
T1_MONTHS = VALID_MONTHS[:3]
# Complemento de T1 (abr-jun). La clasificación usa "todo lo que no es T1", así que no se lee;
# se deja para que el par T1/T2 quede explícito.
T2_MONTHS = VALID_MONTHS[3:]

# Métricas, dimensiones de agrupación y filtros permitidos (lo único que el modelo puede pedir).
METRICS = ["venta_neta_cop", "unidades"]
GROUPABLE = ["region", "canal", "presentacion", "punto_venta_id"]
_FILTER_VALUES = {"region": REGIONS, "canal": CHANNELS, "presentacion": PRESENTATIONS}
FILTERABLE = list(_FILTER_VALUES) + ["punto_venta_id"]


# =====================================================================
# BLOQUE: Estructuras de resultado
# =====================================================================
@dataclass
class GroupRow:
    """Una fila del resultado: un grupo (o el total) con T1, T2, cambios y el cálculo en texto."""

    keys: dict
    t1: float
    t2: float
    abs_change: float
    pct_change: float | None
    n_t1: int
    n_t2: int
    share_of_total_change_pct: float | None
    calculation: str
    notes: list[str] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Resultado de comparar T1 vs T2 para una marca: filas por grupo, total y avisos de calidad."""

    marca: str
    metric: str
    group_by: list[str]
    filters: dict
    rows: list[GroupRow]
    total: GroupRow | None
    quality_notes: list[dict] = field(default_factory=list)
    message: str = ""

    def largest_drop(self) -> GroupRow | None:
        """Grupo con el menor cambio porcentual, solo si es una caída real (< 0)."""
        candidates = [r for r in self.rows if r.pct_change is not None]
        if not candidates:
            return None
        worst = min(candidates, key=lambda r: r.pct_change)
        return worst if worst.pct_change < 0 else None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MonthlySeries:
    """Serie mes a mes (ene-jun) de una métrica. Disponible para una vista mensual; la app aún no la expone."""

    marca: str
    metric: str
    filters: dict
    points: list[dict]
    quality_notes: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# =====================================================================
# BLOQUE: Validación y formato de números
# =====================================================================
def _canonical(value, allowed: list[str], label: str) -> str:
    """Devuelve la forma oficial del valor (ignora mayúsculas, tildes y espacios) o lanza ValueError."""
    lookup = {fold(a): a for a in allowed}
    found = lookup.get(fold(str(value).strip()))
    if found is None:
        raise ValueError(f"{label} no válido: {value!r}. Opciones: {allowed}")
    return found


def _fmt_number(value: float) -> str:
    """Formato colombiano: punto para miles y coma para decimales (1.234.567 / 1.234,50)."""
    if float(value).is_integer():
        return f"{int(value):,}".replace(",", ".")
    return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _fmt_pct(value: float) -> str:
    """Porcentaje con dos decimales y coma: 12,34 %."""
    return f"{value:.2f}".replace(".", ",") + " %"


format_number = _fmt_number   # nombres públicos para el armado del prompt y la interfaz
format_pct = _fmt_pct


def _python_number(value):
    """Convierte tipos de numpy a int/float de Python (para poder pasarlos a JSON)."""
    value = value.item() if hasattr(value, "item") else value
    return int(value) if float(value).is_integer() else float(value)


# =====================================================================
# BLOQUE: Analizador de ventas
# =====================================================================
class SalesAnalyzer:
    """Calcula comparaciones sobre la tabla limpia. Recibe también el reporte para avisar de exclusiones."""

    def __init__(self, data: pd.DataFrame, report: QualityReport | None = None):
        self._data = data
        self._report = report

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns)

    # ---- API pública ----

    def compare_periods(self, marca: str, group_by=None, metric: str = "venta_neta_cop",
                        filters: dict | None = None) -> ComparisonResult:
        """Compara T1 vs T2 de una marca, opcionalmente agrupando por dimensiones y filtrando."""
        marca = _canonical(marca, BRANDS, "Marca")
        metric = self._validate_metric(metric)
        keys = self._validate_group_by(group_by)
        filters = self._validate_filters(filters)

        df = self._select(marca, filters)
        if df.empty:
            return ComparisonResult(marca, metric, keys, filters, rows=[], total=None,
                                    quality_notes=self._quality_notes(marca),
                                    message="No hay registros para la marca y los filtros indicados.")

        # Cada fila se etiqueta T1 si su mes es ene-mar; si no, T2.
        df = df.assign(periodo=np.where(df["mes"].isin(T1_MONTHS), "T1", "T2"))
        total_values = df.groupby("periodo")[metric].sum().reindex(["T1", "T2"], fill_value=0)
        total_counts = df.groupby("periodo")[metric].count().reindex(["T1", "T2"], fill_value=0)
        total_change = float(total_values["T2"] - total_values["T1"])
        total = self._build_row({}, total_values["T1"], total_values["T2"],
                                total_counts["T1"], total_counts["T2"], total_change=None)

        # Una fila por grupo; se ordenan de la mayor caída a la mayor subida (los None al final).
        rows = []
        if keys:
            sums = df.pivot_table(index=keys, columns="periodo", values=metric,
                                  aggfunc="sum", fill_value=0).reindex(columns=["T1", "T2"], fill_value=0)
            counts = df.pivot_table(index=keys, columns="periodo", values=metric,
                                    aggfunc="count", fill_value=0).reindex(columns=["T1", "T2"], fill_value=0)
            for idx in sums.index:
                idx_tuple = idx if isinstance(idx, tuple) else (idx,)
                rows.append(self._build_row(dict(zip(keys, idx_tuple)),
                                            sums.at[idx, "T1"], sums.at[idx, "T2"],
                                            counts.at[idx, "T1"], counts.at[idx, "T2"],
                                            total_change=total_change))
            rows.sort(key=lambda r: (r.pct_change is None, r.pct_change if r.pct_change is not None else 0))
        return ComparisonResult(marca, metric, keys, filters, rows=rows, total=total,
                                quality_notes=self._quality_notes(marca))

    def monthly_series(self, marca: str, metric: str = "venta_neta_cop",
                       filters: dict | None = None) -> MonthlySeries:
        """Suma la métrica mes a mes (ene-jun), con cero en los meses sin registros."""
        marca = _canonical(marca, BRANDS, "Marca")
        metric = self._validate_metric(metric)
        filters = self._validate_filters(filters)
        df = self._select(marca, filters)
        sums = df.groupby("mes")[metric].sum().reindex(VALID_MONTHS, fill_value=0)
        counts = df.groupby("mes")[metric].count().reindex(VALID_MONTHS, fill_value=0)
        points = [{"mes": m, "value": _python_number(sums[m]), "n": int(counts[m])} for m in VALID_MONTHS]
        return MonthlySeries(marca, metric, filters, points, self._quality_notes(marca))

    # ---- Internos ----

    def _select(self, marca: str, filters: dict) -> pd.DataFrame:
        """Filas de la marca en los meses válidos que cumplen los filtros."""
        df = self._data[(self._data["marca"] == marca) & self._data["mes"].isin(VALID_MONTHS)]
        for col, values in filters.items():
            df = df[df[col].isin(values)]
        return df

    def _build_row(self, keys, t1, t2, n1, n2, total_change) -> GroupRow:
        """Arma la fila con cambios absoluto y porcentual, aporte al cambio total y el cálculo explicado."""
        t1, t2 = _python_number(t1), _python_number(t2)
        abs_change = _python_number(t2 - t1)
        notes = []
        if n1 == 0:
            notes.append("Sin observaciones en T1.")
        if n2 == 0:
            notes.append("Sin observaciones en T2.")
        pct = None
        if t1 == 0:
            notes.append("T1 es 0: la variación porcentual no está definida.")
            calc_pct = "variación porcentual indefinida (T1 = 0)"
        else:
            pct = (t2 / t1 - 1) * 100
            calc_pct = f"({_fmt_number(t2)} / {_fmt_number(t1)} - 1) × 100 = {_fmt_pct(pct)}"
        share = None
        if total_change:
            share = abs_change / total_change * 100
        calculation = f"T2 - T1 = {_fmt_number(t2)} - {_fmt_number(t1)} = {_fmt_number(abs_change)}; {calc_pct}"
        return GroupRow(keys=keys, t1=t1, t2=t2, abs_change=abs_change, pct_change=pct,
                        n_t1=int(n1), n_t2=int(n2), share_of_total_change_pct=share,
                        calculation=calculation, notes=notes)

    def _quality_notes(self, marca: str) -> list[dict]:
        """Filas excluidas de la marca, para avisar que las cifras no las incluyen."""
        if self._report is None:
            return []
        return [
            {"registro_id": i.registro_id, "rule": i.rule, "detail": i.detail}
            for i in self._report.for_brand(marca) if i.action == EXCLUDED
        ]

    @staticmethod
    def _validate_metric(metric: str) -> str:
        """Comprueba que la métrica esté en el catálogo."""
        if metric not in METRICS:
            raise ValueError(f"Métrica no válida: {metric!r}. Opciones: {METRICS}")
        return metric

    @staticmethod
    def _validate_group_by(group_by) -> list[str]:
        """Normaliza las dimensiones de agrupación (sin repetidas)."""
        if group_by is None:
            return []
        if isinstance(group_by, str):
            group_by = [group_by]
        keys = []
        for col in group_by:
            canon = _canonical(col, GROUPABLE, "Dimensión")
            if canon not in keys:
                keys.append(canon)
        return keys

    @staticmethod
    def _validate_filters(filters: dict | None) -> dict:
        """Normaliza los filtros a {columna: [valores oficiales]}."""
        clean = {}
        for col, value in (filters or {}).items():
            canon_col = _canonical(col, FILTERABLE, "Filtro")
            values = [value] if isinstance(value, str) else list(value)
            if canon_col in _FILTER_VALUES:
                values = [_canonical(v, _FILTER_VALUES[canon_col], canon_col) for v in values]
            clean[canon_col] = values
        return clean
