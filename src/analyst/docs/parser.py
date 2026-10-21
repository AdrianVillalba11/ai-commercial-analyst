"""Extract structured blocks (titles, headings, paragraphs, bullets, tables) from each format.

Parsers only extract and clean; deciding what becomes a chunk is the chunker's job.
PDF tables lose their cell boundaries in plain text extraction, so tables are rebuilt from
word coordinates (see `_table_rows`).
"""
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import docx
import pdfplumber
from docx.table import Table
from docx.text.paragraph import Paragraph

_PARAGRAPH_GAP = 8.0      # points between lines; within a paragraph it is ~6, between paragraphs >= 10
_HEADING_SIZE = 11.0      # numbered section headings are 11.2 pt, body text 8.8 pt or less
_TITLE_SIZE = 16.0        # document title / page subtitle is 17 pt
_COLUMN_GAP = 12.0        # horizontal gap (points) that separates two table columns
_COLUMN_MARGIN = 6.0      # tolerance to the left of a column's start
_FOOTER_RE = re.compile(r"P[áa]gina\s+\d+\s*$")
_ID_START = re.compile(r"^[A-Z]{2,5}(?:-[A-Za-z0-9]+)+\s*[|:]")


@dataclass
class Block:
    kind: str                         # title | heading | paragraph | bullet | quote | table
    text: str = ""
    level: int = 0
    page: int | None = None
    rows: list[list[str]] | None = None   # tables: first row is the header


def _require(path: Path) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el documento: {path}")
    return path


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------- markdown

def parse_markdown(path: Path) -> list[Block]:
    lines = _require(path).read_text(encoding="utf-8").splitlines()
    blocks: list[Block] = []
    table_lines: list[str] = []
    paragraph: list[str] = []

    def flush_paragraph():
        if paragraph:
            blocks.append(Block("paragraph", _clean(" ".join(paragraph))))
            paragraph.clear()

    def flush_table():
        if table_lines:
            rows = []
            for line in table_lines:
                cells = [_clean(c).replace("**", "") for c in line.strip().strip("|").split("|")]
                if all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    continue
                rows.append(cells)
            blocks.append(Block("table", rows=rows))
            table_lines.clear()

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("|"):
            flush_paragraph()
            table_lines.append(line)
            continue
        flush_table()
        if not line.strip():
            flush_paragraph()
        elif line.startswith("# "):
            flush_paragraph()
            blocks.append(Block("title", line[2:].strip()))
        elif line.startswith("## "):
            flush_paragraph()
            blocks.append(Block("heading", line[3:].strip(), level=2))
        elif line.startswith("- "):
            flush_paragraph()
            blocks.append(Block("bullet", _clean(line[2:]).replace("**", "")))
        else:
            paragraph.append(line.replace("**", ""))
    flush_paragraph()
    flush_table()
    return blocks


# -------------------------------------------------------------------- docx

def parse_docx(path: Path) -> list[Block]:
    document = docx.Document(str(_require(path)))
    blocks: list[Block] = []
    for element in document.element.body.iterchildren():
        if element.tag.endswith("}tbl"):
            table = Table(element, document)
            rows = [[_clean(cell.text) for cell in row.cells] for row in table.rows]
            blocks.append(Block("table", rows=rows))
        elif element.tag.endswith("}p"):
            paragraph = Paragraph(element, document)
            text, style = _clean(paragraph.text), paragraph.style.name
            if not text:
                continue
            if style == "Title":
                blocks.append(Block("title", text))
            elif style == "Subtitle":
                continue  # a label such as "INSUMO DE NEGOCIO", not content
            elif style.startswith("Heading"):
                blocks.append(Block("heading", text, level=int(style.split()[-1])))
            elif style.startswith("List"):
                blocks.append(Block("bullet", text))
            elif style == "Quote":
                blocks.append(Block("quote", text))
            else:
                blocks.append(Block("paragraph", text))
    return blocks


# --------------------------------------------------------------------- pdf

def parse_pdf(path: Path) -> list[Block]:
    with pdfplumber.open(str(_require(path))) as pdf:
        page_lines = [_page_lines(page) for page in pdf.pages]
        repeated = _repeated_texts(page_lines)
        blocks: list[Block] = []
        for number, (page, lines) in enumerate(zip(pdf.pages, page_lines), start=1):
            items = [(table.bbox[1], Block("table", rows=_table_rows(page, table), page=number))
                     for table in page.find_tables()]
            lines = [l for l in lines if l["text"] not in repeated and not _FOOTER_RE.search(l["text"])]
            items += _assemble_text(lines, number)
            blocks += [block for _, block in sorted(items, key=lambda item: item[0])]
    return blocks


def _page_lines(page) -> list[dict]:
    """Text lines of the page, excluding whatever lies inside a table."""
    boxes = [t.bbox for t in page.find_tables()]

    def outside_tables(obj) -> bool:
        if obj.get("object_type") != "char":
            return True
        return not any(b[0] - 1 <= obj["x0"] and obj["x1"] <= b[2] + 1
                       and b[1] - 1 <= obj["top"] and obj["bottom"] <= b[3] + 1 for b in boxes)

    lines = []
    for line in page.filter(outside_tables).extract_text_lines(return_chars=True):
        size = Counter(round(c["size"], 1) for c in line["chars"]).most_common(1)[0][0]
        lines.append({"text": _clean(line["text"]), "top": line["top"], "bottom": line["bottom"], "size": size})
    return lines


def _repeated_texts(page_lines: list[list[dict]]) -> set[str]:
    """Running headers: identical lines that appear on every page of a multi-page document."""
    if len(page_lines) < 2:
        return set()
    per_page = [{l["text"] for l in lines} for lines in page_lines]
    return set.intersection(*per_page)


def _assemble_text(lines: list[dict], page_number: int) -> list[tuple[float, Block]]:
    items: list[tuple[float, Block]] = []
    current: dict | None = None
    previous: dict | None = None

    def flush():
        nonlocal current
        if current:
            items.append((current["top"], Block("paragraph", current["text"], page=page_number)))
            current = None

    for line in lines:
        if line["size"] >= _TITLE_SIZE:
            flush()
            if page_number == 1:   # later pages repeat a subtitle that is only layout
                items.append((line["top"], Block("title", line["text"], page=page_number)))
        elif line["size"] >= _HEADING_SIZE:
            flush()
            items.append((line["top"], Block("heading", line["text"], level=2, page=page_number)))
        else:
            continues = (
                current is not None and previous is not None
                and line["top"] - previous["bottom"] <= _PARAGRAPH_GAP
                and abs(line["size"] - previous["size"]) < 0.5
                and not _ID_START.match(line["text"])
            )
            if continues:
                current["text"] += " " + line["text"]
            else:
                flush()
                current = {"text": line["text"], "top": line["top"]}
        previous = line
    flush()
    return items


def _table_rows(page, table) -> list[list[str]]:
    """Rebuild table cells from word coordinates.

    The PDF tables come out of extraction as one string per row, and a cell that wraps over
    two lines can sit between the lines of its neighbours. Instead: rows come from the row
    bands detected by pdfplumber, columns from where each header cell starts (cells are
    left-aligned, so a column begins at the same x as its header), and each word is assigned
    to the column in which it starts.
    """
    words = page.within_bbox(table.bbox).extract_words()
    bands = [row.bbox for row in table.rows]

    def words_in(band):
        return [w for w in words if band[1] <= (w["top"] + w["bottom"]) / 2 <= band[3]]

    boundaries = _column_boundaries(words_in(bands[0]))
    rows = []
    for band in bands:
        cells: list[list[dict]] = [[] for _ in range(len(boundaries) + 1)]
        for word in words_in(band):
            cells[sum(word["x0"] > b for b in boundaries)].append(word)
        rows.append([_cell_text(cell) for cell in cells])
    return rows


def _column_boundaries(header_words: list[dict]) -> list[float]:
    intervals = sorted((w["x0"], w["x1"]) for w in header_words)
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start - merged[-1][1] < _COLUMN_GAP:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [start - _COLUMN_MARGIN for start, _ in merged[1:]]


def _cell_text(words: list[dict]) -> str:
    ordered = sorted(words, key=lambda w: (round(w["top"] / 3), w["x0"]))
    return _clean(" ".join(w["text"] for w in ordered))


# ---------------------------------------------- structured data from tables

_NUMBER_PCT = re.compile(r"([+\-−–]?)\s*(\d+(?:[.,]\d+)?)\s*%")


def parse_reported_regional_variation(path: Path) -> dict[str, float]:
    """Regional variations (% T2 vs T1) stated in the commercial report's table."""
    for block in parse_pdf(path):
        if block.kind != "table" or not any("Variación" in cell for cell in block.rows[0]):
            continue
        result = {}
        for row in block.rows[1:]:
            match = _NUMBER_PCT.search(row[1])
            if not match:
                continue
            value = float(match.group(2).replace(",", "."))
            result[row[0]] = -value if match.group(1) in {"-", "−", "–"} else value
        return result
    raise ValueError("No se encontró la tabla de variación regional en el reporte.")
