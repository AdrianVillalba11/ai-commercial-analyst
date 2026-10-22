"""Report vs data, resolved in code. The model only narrates this; it never decides the verdict.

Compares the regional variations stated in the commercial report with the ones computed from the
sales data, checks the report's "largest drop" claim, and flags attributions that the data cannot
verify (e.g. "promotions explain 70 %") with the reason computed from the data's columns.
"""
import re
from dataclasses import asdict, dataclass, field

from analyst.analytics.metrics import SalesAnalyzer, format_pct
from analyst.data.cleaning import REGIONS
from analyst.docs.chunker import Chunk
from analyst.guardrails.models import ORIGIN_CONTRAST, FinalClaim
from analyst.rag.lexical import fold

CONTRAST_ID = "CONTRASTE-1"
TOLERANCE_PP = 1.0     # the report states whole percentages, so +/- 0.5 is rounding; 1 pp leaves some margin
_LARGEST_DROP = re.compile(r"mayor ca[ií]da")
_ATTRIBUTION = re.compile(r"explican?\s+(?:aproximadamente\s+)?(?:el\s+)?(\d+)\s*%")
_REGION_WORD = re.compile(r"\b(" + "|".join(REGIONS) + r")\b")
_CAUSE_COLUMNS = re.compile(r"promo|precio|descuento")


@dataclass
class RegionContrast:
    region: str
    reported: float
    computed: float
    diff_pp: float
    coincide: bool


@dataclass
class ClaimCheck:
    chunk_id: str
    estado: str         # "contradicha" | "consistente" | "no verificable"
    texto: str


@dataclass
class ReportContrast:
    rows: list[RegionContrast]
    tolerance_pp: float
    reported_top: str | None
    computed_top: str | None
    claims: list[ClaimCheck]
    verdict: str
    consistent: bool
    title: str
    text: str                                   # block shown to the model (already verified)
    contradictions: list[FinalClaim] = field(default_factory=list)
    unverifiable: list[FinalClaim] = field(default_factory=list)
    agreement: FinalClaim | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _reported(value: float) -> str:
    return ("+" if value > 0 else "") + f"{value:.0f} %"


def _pp(value: float) -> str:
    return f"{value:+.2f}".replace(".", ",") + " pp"


def compare_report_to_data(analyzer: SalesAnalyzer, reported: dict[str, float], chunks: list[Chunk],
                           tolerance_pp: float = TOLERANCE_PP) -> ReportContrast:
    result = analyzer.compare_periods("Brio", group_by="region")
    computed = {r.keys["region"]: r.pct_change for r in result.rows if r.pct_change is not None}
    drop = result.largest_drop()
    computed_top = drop.keys["region"] if drop else None
    reported_top = min(reported, key=reported.get) if reported and min(reported.values()) < 0 else None

    rows = []
    for region in REGIONS:
        if region in reported and region in computed:
            diff = reported[region] - computed[region]
            rows.append(RegionContrast(region, reported[region], computed[region], diff, abs(diff) <= tolerance_pp))
    off = [r for r in rows if not r.coincide]
    ranking_ok = reported_top == computed_top
    top_pct = computed.get(computed_top) if computed_top else None

    claims = _check_claims(chunks, computed_top)
    contradicted = [c for c in claims if c.estado == "contradicha"]
    unverifiable_reason = _unverifiable_reason(analyzer.columns)

    contradictions = [
        FinalClaim(f"Según el reporte, {r.region} varió {_reported(r.reported)}, pero los datos dan "
                   f"{format_pct(r.computed)} (diferencia de {_pp(r.diff_pp)}).",
                   [CONTRAST_ID, f"REP-REGION-{fold(r.region).upper()}"], ORIGIN_CONTRAST)
        for r in off]
    for claim in contradicted:
        contradictions.append(FinalClaim(
            f"{claim.chunk_id} afirma algo que los datos contradicen: {claim.texto} Los datos indican "
            f"{computed_top} ({format_pct(top_pct)}) como la región de mayor caída.",
            [CONTRAST_ID, claim.chunk_id], ORIGIN_CONTRAST))
    unverifiable = [
        FinalClaim(f"{c.chunk_id} atribuye un porcentaje de la caída a una causa; no es verificable: {unverifiable_reason}.",
                   [CONTRAST_ID, c.chunk_id], ORIGIN_CONTRAST)
        for c in claims if c.estado == "no verificable"]
    agreeing = [r.region for r in rows if r.coincide]
    agreement = (FinalClaim(f"Coinciden con los datos (±{tolerance_pp:g} pp): {', '.join(agreeing)}.",
                            [CONTRAST_ID], ORIGIN_CONTRAST) if agreeing else None)

    consistent = bool(rows) and not off and ranking_ok and not contradicted
    verdict = _verdict(consistent, off, len(rows), tolerance_pp, ranking_ok, reported_top, computed_top, top_pct)
    title = "Reporte comercial frente a datos de ventas (variación T2 vs T1 por región)"
    return ReportContrast(rows, tolerance_pp, reported_top, computed_top, claims, verdict, consistent, title,
                          _render(title, rows, tolerance_pp, reported_top, computed_top, top_pct, claims,
                                  unverifiable_reason, verdict),
                          contradictions, unverifiable, agreement)


def _check_claims(chunks: list[Chunk], computed_top: str | None) -> list[ClaimCheck]:
    claims = []
    for chunk in chunks:
        if chunk.kind != "afirmacion_reporte":
            continue
        regions = sorted(set(_REGION_WORD.findall(chunk.text)))
        if _LARGEST_DROP.search(fold(chunk.text)) and len(regions) == 1 and computed_top:
            estado = "consistente" if regions[0] == computed_top else "contradicha"
            claims.append(ClaimCheck(chunk.chunk_id, estado,
                                     f"{regions[0]} es la región con mayor caída."))
        if _ATTRIBUTION.search(fold(chunk.text)):
            claims.append(ClaimCheck(chunk.chunk_id, "no verificable", "Atribuye un porcentaje de la caída a una causa."))
    return claims


def _unverifiable_reason(columns: list[str]) -> str:
    if any(_CAUSE_COLUMNS.search(fold(c)) for c in columns):
        return "no se evaluó con las variables de la base"
    return f"la base de ventas no contiene variables de promoción ni de precio (columnas: {', '.join(columns)})"


def _verdict(consistent, off, total, tolerance, ranking_ok, reported_top, computed_top, top_pct) -> str:
    if consistent:
        return f"El reporte es consistente con los datos: todas las regiones coinciden dentro de ±{tolerance:g} pp."
    text = "El reporte NO es consistente con los datos"
    if off:
        text += f": {len(off)} de {total} regiones difieren más de {tolerance:g} pp ({', '.join(r.region for r in off)})"
    if not ranking_ok and reported_top and computed_top:
        text += (f"{' y' if off else ':'} señala a {reported_top} como la región de mayor caída, "
                 f"cuando los datos indican {computed_top} ({format_pct(top_pct)})")
    return text + "."


def _render(title, rows, tolerance, reported_top, computed_top, top_pct, claims, reason, verdict) -> str:
    lines = [f"[{CONTRAST_ID}] {title}. Ya verificado con código: úsalo tal cual y no lo contradigas."]
    for r in rows:
        lines.append(f"- {r.region}: reporte {_reported(r.reported)} | datos {format_pct(r.computed)} | "
                     f"diferencia {_pp(r.diff_pp)} | {'coincide' if r.coincide else 'NO coincide'} (tolerancia ±{tolerance:g} pp)")
    if reported_top and computed_top:
        lines.append(f"Región con mayor caída: reporte {reported_top} | datos {computed_top} ({format_pct(top_pct)}) | "
                     f"{'coincide' if reported_top == computed_top else 'NO coincide'}")
    for c in claims:
        lines.append(f"Afirmación {c.chunk_id}: {c.estado.upper()}" + (f" ({reason})" if c.estado == "no verificable" else ""))
    lines.append(f"Veredicto: {verdict}")
    return "\n".join(lines)
