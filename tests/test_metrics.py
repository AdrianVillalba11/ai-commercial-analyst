"""
PRUEBAS DEL MOTOR DE MÉTRICAS (T1 vs T2)

Qué verifica: los cálculos de analytics/metrics.py: totales, agrupaciones, aporte al cambio, filtros, mayor caída,
validación de entradas y serie mensual.
Cómo: las tablas sintéticas se comparan con valores calculados a mano; con los datos reales se compara contra una
implementación independiente en Python puro (módulo csv, sin pandas), para que un error de pandas no pase inadvertido.
"""
import csv
from collections import defaultdict

import pandas as pd
import pytest

from analyst.analytics.metrics import SalesAnalyzer
from analyst.data.loader import SALES_FILENAME, load_clean_sales
from analyst.config import settings


def make_df(rows: list[tuple]) -> pd.DataFrame:
    """rows: (mes, region, canal, presentacion, unidades, venta)"""
    return pd.DataFrame(
        [{"registro_id": f"V{i}", "mes": mes, "marca": "Brio", "region": region, "canal": canal,
          "punto_venta_id": f"{region[:2].upper()}-{canal[0]}-001", "presentacion": pres,
          "unidades": u, "venta_neta_cop": v}
         for i, (mes, region, canal, pres, u, v) in enumerate(rows, start=1)]
    )


@pytest.fixture
def simple():
    return SalesAnalyzer(make_df([
        ("2026-01", "Norte", "Tradicional", "Individual", 10, 100),
        ("2026-04", "Norte", "Tradicional", "Individual", 5, 50),
        ("2026-02", "Sur", "Moderno", "Familiar", 20, 200),
        ("2026-05", "Sur", "Moderno", "Familiar", 30, 300),
    ]))


def test_total_variation_by_hand(simple):
    r = simple.compare_periods("Brio")
    assert (r.total.t1, r.total.t2, r.total.abs_change) == (300, 350, 50)
    assert r.total.pct_change == pytest.approx(16.6667, abs=1e-3)
    assert r.rows == []


def test_grouped_by_region_values_and_order(simple):
    r = simple.compare_periods("Brio", group_by="region")
    assert [row.keys["region"] for row in r.rows] == ["Norte", "Sur"]  # primero la peor
    norte, sur = r.rows
    assert (norte.t1, norte.t2, norte.abs_change, norte.pct_change) == (100, 50, -50, -50.0)
    assert (sur.t1, sur.t2, sur.abs_change, sur.pct_change) == (200, 300, 100, 50.0)
    assert (norte.n_t1, norte.n_t2) == (1, 1)


def test_share_of_total_change(simple):
    r = simple.compare_periods("Brio", group_by="region")
    norte, sur = r.rows
    # cambio total = +50; Norte -50 -> -100 %, Sur +100 -> +200 %
    assert norte.share_of_total_change_pct == pytest.approx(-100.0)
    assert sur.share_of_total_change_pct == pytest.approx(200.0)
    assert sum(row.share_of_total_change_pct for row in r.rows) == pytest.approx(100.0)


def test_calculation_text_shows_the_formula(simple):
    norte = simple.compare_periods("Brio", group_by="region").rows[0]
    assert "(50 / 100 - 1) × 100 = -50,00 %" in norte.calculation
    assert "T2 - T1 = 50 - 100 = -50" in norte.calculation


def test_largest_drop(simple):
    assert simple.compare_periods("Brio", group_by="region").largest_drop().keys == {"region": "Norte"}


def test_largest_drop_is_none_when_nothing_falls():
    a = SalesAnalyzer(make_df([("2026-01", "Norte", "Tradicional", "Individual", 1, 10),
                               ("2026-04", "Norte", "Tradicional", "Individual", 1, 20)]))
    assert a.compare_periods("Brio", group_by="region").largest_drop() is None


def test_units_metric(simple):
    r = simple.compare_periods("Brio", group_by="region", metric="unidades")
    norte = [x for x in r.rows if x.keys["region"] == "Norte"][0]
    assert (norte.t1, norte.t2) == (10, 5)


def test_filters(simple):
    r = simple.compare_periods("Brio", filters={"region": "Sur"})
    assert (r.total.t1, r.total.t2) == (200, 300)


def test_two_dimensions(simple):
    r = simple.compare_periods("Brio", group_by=["region", "canal"])
    assert {tuple(row.keys.items()) for row in r.rows} == {
        (("region", "Norte"), ("canal", "Tradicional")),
        (("region", "Sur"), ("canal", "Moderno")),
    }


def test_group_missing_in_t1_has_undefined_pct():
    a = SalesAnalyzer(make_df([("2026-04", "Sur", "Moderno", "Familiar", 1, 100)]))
    row = a.compare_periods("Brio", group_by="region").rows[0]
    assert row.pct_change is None and row.t1 == 0
    assert any("T1" in n for n in row.notes)


def test_accents_and_case_are_tolerated(simple):
    r = simple.compare_periods("BRÍO", group_by=" Región ".strip(), filters={"canal": "moderno"})
    assert r.marca == "Brio" and r.total.t1 == 200


def test_no_rows_returns_empty_result_with_message(simple):
    r = simple.compare_periods("Luma")
    assert r.rows == [] and r.total is None and r.message


@pytest.mark.parametrize("kwargs", [
    {"marca": "Inexistente"},
    {"marca": "Brio", "metric": "margen"},
    {"marca": "Brio", "group_by": "color"},
    {"marca": "Brio", "filters": {"region": "Atlantida"}},
    {"marca": "Brio", "filters": {"color": "rojo"}},
])
def test_invalid_inputs_raise_clear_errors(simple, kwargs):
    with pytest.raises(ValueError):
        simple.compare_periods(**kwargs)


def test_input_dataframe_is_not_modified():
    df = make_df([("2026-01", "Norte", "Tradicional", "Individual", 10, 100),
                  ("2026-04", "Norte", "Tradicional", "Individual", 5, 50)])
    before = df.copy(deep=True)
    SalesAnalyzer(df).compare_periods("Brio", group_by="region")
    pd.testing.assert_frame_equal(df, before)


def test_monthly_series_covers_all_months(simple):
    s = simple.monthly_series("Brio")
    assert [p["mes"] for p in s.points] == [f"2026-0{m}" for m in range(1, 7)]
    assert {p["mes"]: p["value"] for p in s.points}["2026-01"] == 100
    assert {p["mes"]: p["n"] for p in s.points}["2026-03"] == 0


# =====================================================================
# BLOQUE: Datos reales, verificados contra una implementación independiente
# =====================================================================

@pytest.fixture(scope="module")
def real():
    result = load_clean_sales()
    return SalesAnalyzer(result.data, result.report)


def independent_brio_region_totals() -> dict:
    """Recálculo en Python puro: módulo csv + diccionarios, con sus propias reglas mínimas de limpieza."""
    path = settings.raw_data_dir / SALES_FILENAME
    seen, totals = set(), defaultdict(lambda: {"T1": 0, "T2": 0})
    months = {f"2026-0{m}" for m in range(1, 7)}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = tuple(row.values())
            if key in seen:
                continue
            seen.add(key)
            if row["marca"].strip().casefold() != "brio" or row["mes"] not in months:
                continue
            try:
                units, sales = float(row["unidades"]), float(row["venta_neta_cop"])
            except ValueError:
                continue
            if units < 0 or sales < 0:
                continue
            period = "T1" if row["mes"] <= "2026-03" else "T2"
            totals[row["region"].strip()][period] += int(sales)
    return dict(totals)


def test_real_brio_by_region_matches_independent_calculation(real):
    expected = independent_brio_region_totals()
    result = real.compare_periods("Brio", group_by="region")
    assert {r.keys["region"] for r in result.rows} == set(expected)
    for row in result.rows:
        exp = expected[row.keys["region"]]
        assert (row.t1, row.t2) == (exp["T1"], exp["T2"])
        assert row.pct_change == pytest.approx((exp["T2"] / exp["T1"] - 1) * 100)


def test_real_largest_drop_is_sur(real):
    drop = real.compare_periods("Brio", group_by="region").largest_drop()
    assert drop.keys == {"region": "Sur"}
    assert (drop.t1, drop.t2) == (363_431_600, 253_404_600)
    assert drop.pct_change == pytest.approx((253_404_600 / 363_431_600 - 1) * 100)
    assert "-30,27 %" in drop.calculation


def test_real_groups_add_up_to_total_and_shares_to_100(real):
    result = real.compare_periods("Brio", group_by="region")
    assert sum(r.t1 for r in result.rows) == result.total.t1
    assert sum(r.t2 for r in result.rows) == result.total.t2
    assert sum(r.share_of_total_change_pct for r in result.rows) == pytest.approx(100.0)
    assert result.total.pct_change == pytest.approx(-10.5, abs=0.05)


def test_real_result_carries_quality_notes_for_brio(real):
    ids = {n["registro_id"] for n in real.compare_periods("Brio").quality_notes}
    assert {"V01999", "V03997", "V03999"} <= ids
