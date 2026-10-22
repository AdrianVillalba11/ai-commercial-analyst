"""
TROCEADO DE DOCUMENTOS EN FRAGMENTOS CITABLES

Qué hace: convierte los bloques que entrega parser.py en fragmentos (chunks), uno por unidad de evidencia.
Con qué se conecta: sus fragmentos alimentan el buscador (rag/), el contraste con el reporte y el
validador de citas.
Idea clave: se corta por SIGNIFICADO (un hallazgo, una fila de tabla, una limitación) y no por número
de caracteres, así cada cita apunta a una afirmación completa. Si el documento define un id de
evidencia (INV-01, REP-02, OC-SUR-01, CAM-101...), ese id es el id del fragmento.
"""
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from analyst.config import settings
from analyst.data.cleaning import REGIONS
from analyst.docs.parser import Block, parse_docx, parse_markdown, parse_pdf
from analyst.text import REGION_EXACT, fold

# =====================================================================
# BLOQUE: Patrones de identificadores
# Reconocen ids como INV-01, OC-SUR-01 o CAM-101 al inicio de un texto.
# =====================================================================
_ID = r"[A-Z]{2,5}(?:-[A-Za-z0-9]+)+"
_ID_PARAGRAPH = re.compile(rf"^({_ID})\s*[|:]")
_ID_HEADING = re.compile(rf"^({_ID})\s*\|\s*(.+)$")
_ID_THEN_REST = re.compile(rf"^({_ID})\s+(.*)$")
# Códigos de región que aparecen dentro de los ids (OC-SUR-01 -> Sur).
_REGION_CODES = {"SUR": "Sur", "CEN": "Centro", "NOR": "Norte", "ORI": "Oriente", "OCC": "Occidente"}


# =====================================================================
# BLOQUE: Estructuras: fragmento y especificación de documento
# =====================================================================
@dataclass
class Chunk:
    """Un fragmento citable de un documento, con su id, su texto original y el texto que se vectoriza."""

    chunk_id: str
    doc: str                  # nombre del archivo
    doc_title: str
    doc_type: str
    section: str
    kind: str                 # hallazgo, limitacion, cita_entrevista, fila_tabla, tabla, texto, ...
    text: str                 # texto original, el que se cita
    embed_text: str           # texto más su línea de contexto; esto es lo que se vectoriza
    regions: list[str] = field(default_factory=list)
    page: int | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DocSpec:
    """Datos fijos de un documento: archivo, prefijo de ids, título y tipo."""

    filename: str
    prefix: str
    title: str
    doc_type: str


# Los cuatro documentos de apoyo (el CSV de ventas se maneja aparte).
DOCUMENTS = [
    DocSpec("01_Contexto_Negocio.md", "CTX", "Contexto de negocio", "contexto"),
    DocSpec("03_Investigacion_Consumidor.pdf", "INV", "Investigación de consumidor", "investigacion_consumidor"),
    DocSpec("04_Observaciones_Canales.docx", "OC", "Observaciones de canales", "observaciones_canales"),
    DocSpec("05_Reporte_Comercial.pdf", "REP", "Reporte comercial", "reporte_comercial"),
]


# =====================================================================
# BLOQUE: Carga de todos los documentos
# =====================================================================
def load_all_chunks(raw_dir: Path | None = None) -> list[Chunk]:
    """Lee los cuatro documentos y devuelve todos sus fragmentos; falla si hay ids repetidos."""
    raw_dir = Path(raw_dir) if raw_dir else settings.raw_data_dir
    chunks: list[Chunk] = []
    for spec in DOCUMENTS:
        path = raw_dir / spec.filename
        suffix = path.suffix.lower()
        if suffix == ".md":
            chunks += chunk_markdown(parse_markdown(path), spec)
        elif suffix == ".pdf":
            chunks += chunk_pdf(parse_pdf(path), spec)
        elif suffix == ".docx":
            chunks += chunk_docx(parse_docx(path), spec)
    _check_unique_ids(chunks)
    return chunks


# =====================================================================
# BLOQUE: Ayudantes
# =====================================================================
def _slug(text: str) -> str:
    """Texto -> identificador en mayúsculas sin tildes ('Región Sur' -> 'REGION-SUR')."""
    return re.sub(r"[^A-Z0-9]+", "-", fold(text).upper()).strip("-")


def _key(text: str) -> str:
    """Texto -> nombre de campo en minúsculas con guion bajo (para los metadatos)."""
    return _slug(text).lower().replace("-", "_")


def _check_unique_ids(chunks: list[Chunk]) -> None:
    """Un id repetido haría ambigua una cita, así que se detiene la carga."""
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id in seen:
            raise ValueError(f"chunk_id duplicado: {chunk.chunk_id}")
        seen.add(chunk.chunk_id)


def _regions_of(chunk_id: str, text: str) -> list[str]:
    """Regiones que menciona el texto (con mayúscula) o que codifica el id."""
    found = {m.group(1) for m in REGION_EXACT.finditer(text)}
    found |= {_REGION_CODES[t] for t in chunk_id.split("-")[1:] if t in _REGION_CODES}
    return [r for r in REGIONS if r in found]


def _kind_for_id(chunk_id: str) -> str:
    """Tipo de fragmento según la forma del id (limitación, entrevista, afirmación del reporte...)."""
    if "-LIM-" in chunk_id:
        return "limitacion"
    if chunk_id.startswith("ENT-"):
        return "cita_entrevista"
    if chunk_id.startswith("REP-"):
        return "afirmacion_reporte"
    if chunk_id.startswith("OC-EXT"):
        return "dato_importado"
    if chunk_id.startswith("OC-"):
        return "hallazgo_campo"
    return "hallazgo"


def _make_chunk(spec: DocSpec, chunk_id: str, section: str, kind: str, text: str,
                page: int | None = None, metadata: dict | None = None) -> Chunk:
    """Crea el fragmento y le antepone su contexto (documento y sección) al texto que se vectoriza."""
    context = " | ".join(part for part in (spec.title, section) if part)
    return Chunk(chunk_id=chunk_id, doc=spec.filename, doc_title=spec.title, doc_type=spec.doc_type,
                 section=section, kind=kind, text=text, embed_text=f"{context}\n{text}",
                 regions=_regions_of(chunk_id, text), page=page, metadata=metadata or {})


def _row_text(header: list[str], row: list[str]) -> str:
    """Fila de tabla -> 'Columna: valor | Columna: valor'."""
    return " | ".join(f"{h}: {c}" for h, c in zip(header, row))


def _first_column_is_regions(rows: list[list[str]]) -> bool:
    """¿La primera columna de la tabla son regiones? Entonces cada fila es un fragmento por región."""
    return len(rows) > 1 and all(row[0] in REGIONS for row in rows[1:])


# =====================================================================
# BLOQUE: Troceado de Markdown
# =====================================================================
def chunk_markdown(blocks: list[Block], spec: DocSpec) -> list[Chunk]:
    """Un fragmento por sección '##'; las tablas se escriben como líneas 'campo: significado'."""
    chunks, heading, lines = [], None, []

    def flush():
        if heading and lines:
            chunks.append(_make_chunk(spec, f"{spec.prefix}-{_slug(heading)}", heading, "contexto", "\n".join(lines)))

    for block in blocks:
        if block.kind == "heading":
            flush()
            heading, lines = block.text, []
        elif block.kind == "table":
            header, *rows = block.rows
            lines += [_row_text(header, row) for row in rows]
        elif block.kind in {"paragraph", "bullet"} and heading:
            lines.append(block.text)
    flush()
    return chunks


# =====================================================================
# BLOQUE: Troceado de PDF
# =====================================================================
def chunk_pdf(blocks: list[Block], spec: DocSpec) -> list[Chunk]:
    """Párrafos con id propio (INV-01...) -> un fragmento cada uno; el resto se numera por sección."""
    chunks: list[Chunk] = []
    section_number, section = 0, ""
    counters: dict[str, int] = defaultdict(int)

    def next_index(key: str) -> int:
        """Siguiente número consecutivo para ids generados dentro de una sección."""
        counters[key] += 1
        return counters[key]

    for block in blocks:
        if block.kind == "title":
            continue
        if block.kind == "heading":
            number = re.match(r"^(\d+)\.", block.text)
            section_number, section = (int(number.group(1)) if number else section_number), block.text
            continue
        if block.kind == "paragraph":
            match = _ID_PARAGRAPH.match(block.text)
            if match:
                chunk_id = match.group(1)
                chunks.append(_make_chunk(spec, chunk_id, section, _kind_for_id(chunk_id), block.text, block.page))
            else:
                chunk_id = f"{spec.prefix}-S{section_number}-{next_index(f'p{section_number}')}"
                chunks.append(_make_chunk(spec, chunk_id, section, "texto", block.text, block.page))
        elif block.kind == "table":
            chunks += _table_chunks(block, spec, section, section_number, next_index)
    return chunks


def _table_chunks(block: Block, spec: DocSpec, section: str, section_number: int, next_index) -> list[Chunk]:
    """Tabla por regiones -> un fragmento por fila; cualquier otra tabla -> un solo fragmento."""
    header, *rows = block.rows
    if _first_column_is_regions(block.rows):
        return [
            _make_chunk(spec, f"{spec.prefix}-{_slug(header[0])}-{_slug(row[0])}", section, "fila_tabla",
                        _row_text(header, row), block.page,
                        {_key(h): c for h, c in zip(header, row)})
            for row in rows
        ]
    text = "\n".join(_row_text(header, row) for row in rows)
    chunk_id = f"{spec.prefix}-TAB-S{section_number}-{next_index(f't{section_number}')}"
    return [_make_chunk(spec, chunk_id, section, "tabla", text, block.page)]


# =====================================================================
# BLOQUE: Troceado de Word
# =====================================================================
def chunk_docx(blocks: list[Block], spec: DocSpec) -> list[Chunk]:
    """Un hallazgo (encabezado con id + sus viñetas) o un párrafo -> un fragmento; las tablas van por fila."""
    chunks: list[Chunk] = []
    section_number, h1 = 0, ""
    pending: dict | None = None
    counters: dict[int, int] = defaultdict(int)

    def next_index() -> int:
        """Siguiente número consecutivo dentro de la sección actual."""
        counters[section_number] += 1
        return counters[section_number]

    def flush():
        """Cierra el hallazgo en curso y lo guarda como fragmento."""
        nonlocal pending
        if pending:
            text = f"{pending['title']}\n" + "\n".join(f"- {b}" for b in pending["bullets"])
            chunks.append(_make_chunk(spec, pending["id"], f"{h1} > {pending['title']}", "hallazgo_campo", text))
            pending = None

    for block in blocks:
        if block.kind == "title":
            continue
        if block.kind == "heading" and block.level == 1:
            flush()
            section_number, h1 = section_number + 1, block.text
        elif block.kind == "heading":
            flush()
            match = _ID_HEADING.match(block.text)
            if match:
                pending = {"id": match.group(1), "title": block.text, "bullets": []}
        elif block.kind == "bullet":
            if pending:
                pending["bullets"].append(block.text)
            else:
                chunks.append(_make_chunk(spec, f"{spec.prefix}-S{section_number}-{next_index()}", h1, "texto", block.text))
        elif block.kind in {"paragraph", "quote"}:
            flush()
            match = _ID_PARAGRAPH.match(block.text)
            if match:
                chunk_id = match.group(1)
                chunks.append(_make_chunk(spec, chunk_id, h1, _kind_for_id(chunk_id), block.text))
            else:
                chunks.append(_make_chunk(spec, f"{spec.prefix}-S{section_number}-{next_index()}", h1, "texto", block.text))
        elif block.kind == "table":
            flush()
            chunks += _docx_table_chunks(block, spec, h1)
    flush()
    return chunks


def _docx_table_chunks(block: Block, spec: DocSpec, section: str) -> list[Chunk]:
    """Cada fila de tabla es un fragmento: visita (CAM-xxx) o fila por región; otra cosa es un error."""
    header, *rows = block.rows
    chunks = []
    for row in rows:
        metadata = {_key(h): c for h, c in zip(header, row)}
        first = _ID_THEN_REST.match(row[0])
        if first:  # "CAM-101  2026-04-09" -> id + fecha
            chunk_id = first.group(1)
            row = [chunk_id, first.group(2)] + row[1:]
            fields = ["ID", "Fecha"] + header[1:]
            metadata = {_key(f): c for f, c in zip(fields, row)}
            text = _row_text(fields, row)
            kind = "observacion_visita"
        elif row[0] in REGIONS:
            chunk_id = f"{spec.prefix}-{_slug(header[0])}-{_slug(row[0])}"
            text, kind = _row_text(header, row), "fila_tabla"
        else:
            raise ValueError(f"Fila de tabla sin identificador reconocible: {row}")
        chunks.append(_make_chunk(spec, chunk_id, section, kind, text, metadata=metadata))
    return chunks
