"""Turn parsed blocks into citable chunks, one per unit of evidence.

Chunks are cut by meaning (a finding, a table row, a limitation), not by character count,
so every citation points to one complete statement. Where the document defines an evidence
id (INV-01, REP-02, OC-SUR-01, CAM-101...), that id is the chunk id.
"""
import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from analyst.config import settings
from analyst.data.cleaning import REGIONS
from analyst.docs.parser import Block, parse_docx, parse_markdown, parse_pdf

_ID = r"[A-Z]{2,5}(?:-[A-Za-z0-9]+)+"
_ID_PARAGRAPH = re.compile(rf"^({_ID})\s*[|:]")
_ID_HEADING = re.compile(rf"^({_ID})\s*\|\s*(.+)$")
_ID_THEN_REST = re.compile(rf"^({_ID})\s+(.*)$")
_REGION_WORD = re.compile(r"\b(" + "|".join(REGIONS) + r")\b")
_REGION_CODES = {"SUR": "Sur", "CEN": "Centro", "NOR": "Norte", "ORI": "Oriente", "OCC": "Occidente"}


@dataclass
class Chunk:
    chunk_id: str
    doc: str                  # file name
    doc_title: str
    doc_type: str
    section: str
    kind: str                 # hallazgo, limitacion, cita_entrevista, fila_tabla, tabla, texto, ...
    text: str                 # original, citable text
    embed_text: str           # text plus its context line; this is what gets embedded
    regions: list[str] = field(default_factory=list)
    page: int | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DocSpec:
    filename: str
    prefix: str
    title: str
    doc_type: str


DOCUMENTS = [
    DocSpec("01_Contexto_Negocio.md", "CTX", "Contexto de negocio", "contexto"),
    DocSpec("03_Investigacion_Consumidor.pdf", "INV", "Investigación de consumidor", "investigacion_consumidor"),
    DocSpec("04_Observaciones_Canales.docx", "OC", "Observaciones de canales", "observaciones_canales"),
    DocSpec("05_Reporte_Comercial.pdf", "REP", "Reporte comercial", "reporte_comercial"),
]


def load_all_chunks(raw_dir: Path | None = None) -> list[Chunk]:
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


# ------------------------------------------------------------------ helpers

def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _slug(text: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "-", _fold(text).upper()).strip("-")


def _key(text: str) -> str:
    return _slug(text).lower().replace("-", "_")


def _check_unique_ids(chunks: list[Chunk]) -> None:
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id in seen:
            raise ValueError(f"chunk_id duplicado: {chunk.chunk_id}")
        seen.add(chunk.chunk_id)


def _regions_of(chunk_id: str, text: str) -> list[str]:
    found = {m.group(1) for m in _REGION_WORD.finditer(text)}
    found |= {_REGION_CODES[t] for t in chunk_id.split("-")[1:] if t in _REGION_CODES}
    return [r for r in REGIONS if r in found]


def _kind_for_id(chunk_id: str) -> str:
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
    context = " | ".join(part for part in (spec.title, section) if part)
    return Chunk(chunk_id=chunk_id, doc=spec.filename, doc_title=spec.title, doc_type=spec.doc_type,
                 section=section, kind=kind, text=text, embed_text=f"{context}\n{text}",
                 regions=_regions_of(chunk_id, text), page=page, metadata=metadata or {})


def _row_text(header: list[str], row: list[str]) -> str:
    return " | ".join(f"{h}: {c}" for h, c in zip(header, row))


def _first_column_is_regions(rows: list[list[str]]) -> bool:
    return len(rows) > 1 and all(row[0] in REGIONS for row in rows[1:])


# ----------------------------------------------------------------- markdown

def chunk_markdown(blocks: list[Block], spec: DocSpec) -> list[Chunk]:
    """One chunk per '##' section; tables are written as 'field: meaning' lines."""
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


# ---------------------------------------------------------------------- pdf

def chunk_pdf(blocks: list[Block], spec: DocSpec) -> list[Chunk]:
    chunks: list[Chunk] = []
    section_number, section = 0, ""
    counters: dict[str, int] = defaultdict(int)

    def next_index(key: str) -> int:
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


# --------------------------------------------------------------------- docx

def chunk_docx(blocks: list[Block], spec: DocSpec) -> list[Chunk]:
    chunks: list[Chunk] = []
    section_number, h1 = 0, ""
    pending: dict | None = None
    counters: dict[int, int] = defaultdict(int)

    def next_index() -> int:
        counters[section_number] += 1
        return counters[section_number]

    def flush():
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
    header, *rows = block.rows
    chunks = []
    for row in rows:
        metadata = {_key(h): c for h, c in zip(header, row)}
        first = _ID_THEN_REST.match(row[0])
        if first:  # "CAM-101  2026-04-09" -> id + date
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
