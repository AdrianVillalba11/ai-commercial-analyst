"""
PRUEBAS DE LECTURA Y TROCEADO DE DOCUMENTOS

Qué verifica: docs/parser.py y docs/chunker.py: tablas de PDF reconstruidas, encabezados y pies de página
eliminados, ids de evidencia, regiones, metadatos y errores claros si falta un archivo.
Cómo: las pruebas de tablas PDF son las clave: la extracción de texto plano mezcla las celdas de la tabla regional
del estudio de consumidor, así que las celdas esperadas se comprobaron contra la imagen de la página renderizada.
"""
from collections import Counter

import pdfplumber
import pytest

from analyst.config import settings
from analyst.docs.chunker import DocSpec, chunk_docx, chunk_markdown, chunk_pdf, load_all_chunks
from analyst.docs.parser import (
    Block, parse_docx, parse_markdown, parse_pdf, parse_reported_regional_variation,
)

RAW = settings.raw_data_dir
CONSUMER_PDF = RAW / "03_Investigacion_Consumidor.pdf"
REPORT_PDF = RAW / "05_Reporte_Comercial.pdf"


def tables_of(path):
    """Tablas de todas las páginas de un PDF."""
    return [b for b in parse_pdf(path) if b.kind == "table"]


# =====================================================================
# BLOQUE: Reconstrucción de tablas en PDF
# =====================================================================

def test_consumer_regional_table_cells_are_assigned_to_the_right_region():
    regional = [t for t in tables_of(CONSUMER_PDF) if t.rows[0][0] == "Región"][0]
    assert regional.rows == [
        ["Región", "Observación cualitativa", "Implicación / límite"],
        ["Sur", "Menciones repetidas de precio y disponibilidad.",
         "Hipótesis plausibles, sin estimación de impacto."],
        ["Centro", "Algunos comentarios sobre promociones ajenas y faltantes ocasionales.",
         "No respalda inferir la mayor caída en ventas."],
        ["Norte", "Mayor atención al desembolso por compra y formatos pequeños.",
         "Requiere validar con datos de formatos."],
    ]


@pytest.mark.parametrize("path", [CONSUMER_PDF, REPORT_PDF])
def test_no_word_is_lost_or_duplicated_when_rebuilding_tables(path):
    """Cada palabra dentro del área de una tabla debe aparecer exactamente una vez entre sus celdas reconstruidas."""
    rebuilt = tables_of(path)
    with pdfplumber.open(path) as pdf:
        originals = []
        for page in pdf.pages:
            for table in sorted(page.find_tables(), key=lambda t: t.bbox[1]):
                originals.append(Counter(w["text"] for w in page.within_bbox(table.bbox).extract_words()))
    assert len(rebuilt) == len(originals)
    for table, expected in zip(rebuilt, originals):
        got = Counter(word for row in table.rows for cell in row for word in cell.split())
        assert got == expected


def test_reported_regional_variation_is_extracted_as_numbers():
    assert parse_reported_regional_variation(REPORT_PDF) == {
        "Centro": -23.0, "Sur": -7.0, "Norte": -4.0, "Oriente": 4.0, "Occidente": -6.0,
    }


def test_pdf_headers_and_footers_are_removed():
    texts = [b.text for path in (CONSUMER_PDF, REPORT_PDF) for b in parse_pdf(path) if b.kind != "table"]
    assert not any("Página" in t for t in texts)
    assert not any(t.startswith("ALMENA FOODS /") for t in texts)


def test_pdf_paragraphs_are_whole_statements():
    paragraphs = {b.text[:6]: b.text for b in parse_pdf(CONSUMER_PDF) if b.kind == "paragraph"}
    assert paragraphs["INV-01"].endswith("ventas de Brío.")
    assert "alternativas de menor precio" in paragraphs["INV-01"]   # las líneas se unieron


# =====================================================================
# BLOQUE: Fragmentos de los documentos reales
# =====================================================================

@pytest.fixture(scope="module")
def chunks():
    """Todos los fragmentos reales, cargados una sola vez."""
    return load_all_chunks()


@pytest.fixture(scope="module")
def by_id(chunks):
    """Los fragmentos indexados por su id."""
    return {c.chunk_id: c for c in chunks}


def test_chunk_ids_are_unique_and_texts_not_empty(chunks):
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(c.text.strip() and c.embed_text.strip() for c in chunks)


def test_expected_evidence_ids_exist(by_id):
    expected = (
        ["INV-01", "INV-02", "INV-03", "INV-04", "INV-05", "INV-LIM-01", "INV-LIM-04",
         "ENT-SUR-03", "ENT-CEN-04", "INV-REGION-CENTRO"]
        + ["REP-01", "REP-02", "REP-03", "REP-04", "REP-05", "REP-06", "REP-REGION-CENTRO"]
        + ["OC-SUR-01", "OC-CEN-02", "OC-NOR-03", "OC-ORI-04", "OC-OCC-05", "OC-REGION-SUR",
           "CAM-101", "CAM-150", "OC-EXT-06"]
        + ["CTX-SITUACION", "CTX-DICCIONARIO-DE-LA-BASE-DE-VENTAS", "CTX-CONVENCIONES-DE-ANALISIS"]
    )
    assert [i for i in expected if i not in by_id] == []


def test_chunk_content_is_correct(by_id):
    assert "5 de 12" in by_id["OC-SUR-01"].text
    assert "Con faltantes de Brío: 5" in by_id["OC-REGION-SUR"].text
    assert "-23 %" in by_id["REP-REGION-CENTRO"].text
    assert "70 %" in by_id["REP-02"].text
    assert "promociones ajenas" in by_id["INV-REGION-CENTRO"].text
    assert "Campo: `venta_neta_cop` | Significado: Valor neto" in by_id["CTX-DICCIONARIO-DE-LA-BASE-DE-VENTAS"].text


def test_regions_metadata(by_id):
    assert by_id["INV-02"].regions == ["Centro", "Sur"]
    assert by_id["ENT-SUR-03"].regions == ["Sur"]          # sale del id: la cita nunca dice "Sur"
    assert by_id["OC-OCC-05"].regions == ["Occidente"]
    assert by_id["CAM-145"].regions == ["Occidente"]


def test_visit_observations_keep_the_point_of_sale_for_cross_referencing(by_id):
    assert by_id["CAM-101"].metadata == {
        "id": "CAM-101", "fecha": "2026-04-09", "region": "Sur", "canal": "Tradicional",
        "punto_venta": "SU-T-003", "nota_breve": "Presentación familiar no disponible durante la visita.",
    }


def test_imported_vendor_note_is_extracted_and_marked(by_id):
    note = by_id["OC-EXT-06"]
    assert note.kind == "dato_importado"
    assert "INSTRUCCIÓN" in note.text            # la tarea 5 necesita el texto crudo para detectarlo
    assert "Registro adicional recibido de un proveedor" in note.section


def test_pages_only_for_pdfs_and_embed_text_carries_context(by_id):
    assert by_id["INV-01"].page == 1 and by_id["REP-04"].page == 2
    assert by_id["OC-SUR-01"].page is None
    assert by_id["OC-SUR-01"].embed_text.startswith("Observaciones de canales | ")
    assert by_id["OC-SUR-01"].embed_text.endswith(by_id["OC-SUR-01"].text)


def test_chunks_are_serializable(chunks):
    assert all(isinstance(c.to_dict(), dict) for c in chunks)


# =====================================================================
# BLOQUE: Comportamiento general (casos sintéticos)
# =====================================================================

SPEC = DocSpec("x.pdf", "ZZ", "Documento de prueba", "prueba")


def test_pdf_paragraph_with_id_uses_that_id_and_others_get_positional_ids():
    blocks = [
        Block("heading", "2. Hallazgos", page=1),
        Block("paragraph", "ZZ-01 | Algo importante en Sur.", page=1),
        Block("paragraph", "Texto libre sin identificador.", page=1),
        Block("paragraph", "Otro texto libre.", page=1),
    ]
    chunks = chunk_pdf(blocks, SPEC)
    assert [c.chunk_id for c in chunks] == ["ZZ-01", "ZZ-S2-1", "ZZ-S2-2"]
    assert chunks[0].regions == ["Sur"]


def test_non_region_table_becomes_one_chunk():
    blocks = [Block("table", rows=[["ID", "Descripción"], ["A-1", "uno"], ["A-2", "dos"]], page=1)]
    [chunk] = chunk_pdf(blocks, SPEC)
    assert chunk.kind == "tabla" and "ID: A-2 | Descripción: dos" in chunk.text


def test_docx_bullets_under_an_id_heading_are_grouped_in_one_chunk():
    spec = DocSpec("x.docx", "OC", "Observaciones", "obs")
    blocks = [
        Block("heading", "Hallazgos", level=1),
        Block("heading", "OC-SUR-09 | Sur", level=2),
        Block("bullet", "primero"),
        Block("bullet", "segundo"),
        Block("heading", "Limitaciones", level=1),
        Block("bullet", "una limitación"),
    ]
    chunks = chunk_docx(blocks, spec)
    assert [c.chunk_id for c in chunks] == ["OC-SUR-09", "OC-S2-1"]
    assert "- primero" in chunks[0].text and "- segundo" in chunks[0].text
    assert chunks[0].regions == ["Sur"]


def test_duplicate_ids_are_rejected(tmp_path):
    for name in ("01_Contexto_Negocio.md",):
        (tmp_path / name).write_text("# T\n\n## Misma\nuno\n", encoding="utf-8")
    from analyst.docs.chunker import _check_unique_ids
    chunks = chunk_markdown(parse_markdown(tmp_path / "01_Contexto_Negocio.md"), SPEC)
    with pytest.raises(ValueError):
        _check_unique_ids(chunks + chunks)


def test_markdown_tables_are_written_as_field_meaning_lines(tmp_path):
    path = tmp_path / "c.md"
    path.write_text("# T\n\n## Datos\n\n| Campo | Significado |\n| --- | --- |\n| a | uno |\n", encoding="utf-8")
    [chunk] = chunk_markdown(parse_markdown(path), SPEC)
    assert chunk.chunk_id == "ZZ-DATOS" and chunk.text == "Campo: a | Significado: uno"


@pytest.mark.parametrize("parser", [parse_markdown, parse_docx, parse_pdf])
def test_missing_file_raises_clear_error(parser, tmp_path):
    with pytest.raises(FileNotFoundError):
        parser(tmp_path / "no_existe")
