"""
CONTRASTE ENTRE EL REPORTE COMERCIAL Y LOS DATOS

Qué hace: compara las variaciones regionales que declara el reporte con las calculadas desde las ventas,
revisa la afirmación de "mayor caída" y marca las atribuciones que los datos no pueden verificar
(por ejemplo "las promociones explican el 70 %"), con el motivo sacado de las columnas de la base.
Con qué se conecta: usa analytics/metrics.py y los fragmentos del reporte; el validador y el servicio
usan su resultado (bloque CONTRASTE-1).
Idea clave: el veredicto lo decide el CÓDIGO. El modelo solo lo narra.
"""
import re
from dataclasses import asdict, dataclass, field

from analyst.analytics.metrics import SalesAnalyzer, format_pct
from analyst.data.cleaning import REGIONS
from analyst.docs.chunker import Chunk
from analyst.guardrails.models import ORIGIN_CONTRAST, FinalClaim
from analyst.text import REGION_EXACT, fold

# =====================================================================
# BLOQUE: Constantes
# =====================================================================
CONTRAST_ID = "CONTRASTE-1"
TOLERANCE_PP = 1.0     # el reporte da porcentajes enteros, así que +/- 0,5 es redondeo; 1 pp deja algo de margen
_LARGEST_DROP = re.compile(r"mayor ca[ií]da")
_ATTRIBUTION = re.compile(r"explican?\s+(?:aproximadamente\s+)?(?:el\s+)?(\d+)\s*%")
_CAUSE_COLUMNS = re.compile(r"promo|precio|descuento")


# =====================================================================
# BLOQUE: Estructuras de resultado
# =====================================================================
@dataclass
class RegionContrast:
    """Una región: variación del reporte, variación calculada, diferencia y si coinciden."""

    region: str
    reported: float
    computed: float
    diff_pp: float
    coincide: bool


@dataclass
class ClaimCheck:
    """Resultado de revisar una afirmación del reporte."""

    chunk_id: str
    estado: str         # "contradicha" | "consistente" | "no verificable"
    texto: str


@dataclass
class ReportContrast:
    """Contraste completo: filas por región, veredicto, afirmaciones revisadas y frases ya redactadas."""

    rows: list[RegionContrast]
    tolerance_pp: float
    reported_top: str | None
    computed_top: str | None
    claims: list[ClaimCheck]
    verdict: str
    consistent: bool
    title: str
    text: str                                   # bloque que ve el modelo (ya verificado)
    contradictions: list[FinalClaim] = field(default_factory=list)
    unverifiable: list[FinalClaim] = field(default_factory=list)
    agreement: FinalClaim | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _reported(value: float) -> str:
    """Variación del reporte como la escribe el reporte: +5 % / -12 %."""
    return ("+" if value > 0 else "") + f"{value:.0f} %"


def _pp(value: float) -> str:
    """Diferencia en puntos porcentuales con signo."""
    return f"{value:+.2f}".replace(".", ",") + " pp"


def compare_report_to_data(analyzer: SalesAnalyzer, reported: dict[str, float], chunks: list[Chunk],
                           tolerance_pp: float = TOLERANCE_PP) -> ReportContrast:
    """Compara el reporte con los datos y deja listas las frases de contradicciones, no verificables y coincidencias."""
    result = analyzer.compare_periods("Brio", group_by="region")
    computed = {r.keys["region"]: r.pct_change for r in result.rows if r.pct_change is not None}
    drop = result.largest_drop()
    computed_top = drop.keys["region"] if drop else None
    reported_top = min(reported, key=reported.get) if reported and min(reported.values()) < 0 else None

    # Diferencia por región (reporte - datos); coincide si cae dentro de la tolerancia.
    rows = []
    for region in REGIONS:
        if region in reported and region in computed:
            diff = reported[region] - computed[region]
            rows.append(RegionContrast(region, reported[region], computed[region], diff, abs(diff) <= tolerance_pp))
    off = [r for r in rows if not r.coincide]
    ranking_ok = reported_top == computed_top
    top_pct = computed.get(computed_top) if computed_top else None

    # Afirmaciones sueltas del reporte (mayor caída, atribución de causas).
    claims = _check_claims(chunks, computed_top)
    contradicted = [c for c in claims if c.estado == "contradicha"]
    unverifiable_reason = _unverifiable_reason(analyzer.columns)

    # Frases ya redactadas por código para cada tipo de hallazgo.
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
    """Revisa las afirmaciones del reporte: "mayor caída" se contrasta con los datos; una atribución de % a una causa no es verificable."""
    claims = []
    for chunk in chunks:
        if chunk.kind != "afirmacion_reporte":
            continue
        regions = sorted(set(REGION_EXACT.findall(chunk.text)))
        if _LARGEST_DROP.search(fold(chunk.text)) and len(regions) == 1 and computed_top:
            estado = "consistente" if regions[0] == computed_top else "contradicha"
            claims.append(ClaimCheck(chunk.chunk_id, estado,
                                     f"{regions[0]} es la región con mayor caída."))
        if _ATTRIBUTION.search(fold(chunk.text)):
            claims.append(ClaimCheck(chunk.chunk_id, "no verificable", "Atribuye un porcentaje de la caída a una causa."))
    return claims


def _unverifiable_reason(columns: list[str]) -> str:
    """Por qué no se puede verificar una causa: la base no trae variables de promoción ni de precio."""
    if any(_CAUSE_COLUMNS.search(fold(c)) for c in columns):
        return "no se evaluó con las variables de la base"
    return f"la base de ventas no contiene variables de promoción ni de precio (columnas: {', '.join(columns)})"


def _verdict(consistent, off, total, tolerance, ranking_ok, reported_top, computed_top, top_pct) -> str:
    """Frase de veredicto: consistente, o qué regiones difieren y si la región de mayor caída no coincide."""
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
    """Bloque de texto que ve el modelo, con el veredicto ya decidido por código."""
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
