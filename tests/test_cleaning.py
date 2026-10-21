"""Tests for the sales cleaning rules.

Unit tests use tiny synthetic tables, which proves the rules are generic and not tied to the real CSV.
"""
import pandas as pd
import pytest

from analyst.data.cleaning import clean_sales
from analyst.data.loader import load_clean_sales, load_raw_sales
from analyst.data.quality_report import CORRECTED, EXCLUDED

BASE = {
    "registro_id": "V1", "mes": "2026-01", "marca": "Brio", "region": "Norte",
    "canal": "Tradicional", "punto_venta_id": "NO-T-001", "presentacion": "Familiar",
    "unidades": "100", "venta_neta_cop": "500000",
}


def make_df(*overrides: dict) -> pd.DataFrame:
    rows = []
    for i, o in enumerate(overrides, start=1):
        row = {**BASE, "registro_id": f"V{i}", **o}
        rows.append(row)
    return pd.DataFrame(rows, dtype=str)


def rules_of(result):
    return {(i.registro_id, i.rule, i.action) for i in result.report.issues}


def test_clean_row_passes_untouched():
    result = clean_sales(make_df({}))
    assert len(result.data) == 1
    assert result.report.issues == []
    assert result.data["venta_neta_cop"].dtype == "int64"


def test_case_and_trailing_space_are_corrected_not_excluded():
    result = clean_sales(make_df({"marca": "BRIO"}, {"region": "Norte "}))
    assert len(result.data) == 2
    assert set(result.data["marca"]) == {"Brio"}
    assert set(result.data["region"]) == {"Norte"}
    assert all(i.action == CORRECTED for i in result.report.issues)
    assert {i.field for i in result.report.issues} == {"marca", "region"}


def test_unknown_category_is_excluded():
    result = clean_sales(make_df({"region": "Atlantida"}))
    assert result.data.empty
    assert ("V1", "valor_no_permitido", EXCLUDED) in rules_of(result)


def test_exact_duplicate_keeps_first():
    df = make_df({}, {})
    df.loc[1, "registro_id"] = "V1"
    result = clean_sales(df)
    assert len(result.data) == 1
    assert ("V1", "duplicado_exacto", EXCLUDED) in rules_of(result)


def test_duplicate_after_normalization_is_detected():
    df = make_df({}, {"marca": "BRIO"})
    df.loc[1, "registro_id"] = "V1"
    result = clean_sales(df)
    assert len(result.data) == 1


def test_conflicting_id_excludes_all_rows_with_that_id():
    df = make_df({}, {"unidades": "999"}, {})
    df.loc[1, "registro_id"] = "V1"
    result = clean_sales(df)
    assert list(result.data["registro_id"]) == ["V3"]
    assert ("V1", "id_conflictivo", EXCLUDED) in rules_of(result)


@pytest.mark.parametrize("month", ["2026-13", "2026-07", "2025-12", "26-01", ""])
def test_invalid_month_is_excluded(month):
    result = clean_sales(make_df({"mes": month}))
    assert result.data.empty
    assert ("V1", "mes_invalido", EXCLUDED) in rules_of(result)


@pytest.mark.parametrize("col,value", [
    ("venta_neta_cop", "N/D"), ("venta_neta_cop", ""), ("unidades", ""), ("unidades", "abc"),
])
def test_non_numeric_values_are_excluded(col, value):
    result = clean_sales(make_df({col: value}))
    assert result.data.empty
    assert ("V1", "valor_no_numerico", EXCLUDED) in rules_of(result)


def test_negative_values_are_excluded():
    result = clean_sales(make_df({"unidades": "-5", "venta_neta_cop": "-1000"}))
    assert result.data.empty
    fields = {i.field for i in result.report.issues if i.rule == "valor_negativo"}
    assert fields == {"unidades", "venta_neta_cop"}


@pytest.mark.parametrize("override", [
    {"punto_venta_id": "OC-T-001"},            # prefix of another region
    {"canal": "Moderno"},                       # channel letter T but canal Moderno
    {"punto_venta_id": "PUNTO-1"},             # invalid format
])
def test_inconsistent_point_of_sale_is_excluded(override):
    result = clean_sales(make_df(override))
    assert result.data.empty
    assert ("V1", "pdv_inconsistente", EXCLUDED) in rules_of(result)


def test_report_summary_counts_rows():
    result = clean_sales(make_df({}, {"mes": "2026-13"}, {"marca": "BRIO"}))
    summary = result.report.summary()
    assert (summary["rows_in"], summary["rows_out"], summary["rows_excluded"]) == (3, 2, 1)


def test_report_can_filter_by_brand():
    result = clean_sales(make_df({"marca": "Luma", "mes": "2026-13"}, {"mes": "2026-13"}))
    assert [i.registro_id for i in result.report.for_brand("Brio")] == ["V2"]


def test_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_sales(tmp_path / "no_existe.csv")


# ---- integration with the real CSV ----

@pytest.fixture(scope="module")
def real():
    return load_clean_sales()


def test_real_csv_row_counts(real):
    assert real.report.rows_in == 4000
    assert real.report.rows_out == 3994


def test_real_csv_known_problems_are_reported(real):
    found = rules_of(real)
    assert ("V00667", "normalizacion_texto", CORRECTED) in found      # BRIO
    assert ("V02929", "normalizacion_texto", CORRECTED) in found      # "Sur "
    assert ("V01999", "duplicado_exacto", EXCLUDED) in found
    assert ("V03997", "valor_no_numerico", EXCLUDED) in found         # N/D
    assert ("V03999", "mes_invalido", EXCLUDED) in found              # 2026-13
    assert ("V03998", "valor_negativo", EXCLUDED) in found
    assert ("V03358", "valor_no_numerico", EXCLUDED) in found         # empty units
    assert ("V02612", "pdv_inconsistente", EXCLUDED) in found


def test_real_csv_clean_data_is_valid(real):
    df = real.data
    assert df["registro_id"].is_unique
    assert set(df["marca"]) == {"Brio", "Luma", "Nativa", "Rizo", "Senda"}
    assert set(df["region"]) == {"Norte", "Centro", "Sur", "Oriente", "Occidente"}
    assert (df["venta_neta_cop"] > 0).all() and (df["unidades"] > 0).all()
    assert df["mes"].min() == "2026-01" and df["mes"].max() == "2026-06"
