"""
VALIDADOR DE LA RESPUESTA DEL MODELO

Qué hace: contrasta lo que escribió el modelo con lo que el código sabe y arma la respuesta final.
Con qué se conecta: recibe la respuesta de llm/answerer.py, usa facts.py y report_check.py, y entrega
una FinalAnswer (models.py) al servicio.
Idea clave: cada regla responde a un fallo visto en ejecuciones reales (fuentes inventadas, hipótesis que
citan solo un cálculo, afirmaciones atribuidas a la región equivocada, "no hay contradicciones" cuando el
reporte está mal...). Lo que se quita NO se oculta: va a `descartes` con su motivo.

Límite: son verificaciones de forma y consistencia (ids, cifras, regiones, lenguaje causal). No pueden
demostrar que una frase sea semánticamente fiel al fragmento que cita.
"""
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from analyst.docs.chunker import Chunk
from analyst.guardrails.facts import calc_facts
from analyst.guardrails.models import (
    ORIGIN_CALC, ORIGIN_CONTRAST, ORIGIN_DOCUMENT, ORIGIN_MODEL, Discard, FinalAnswer, FinalClaim,
)
from analyst.guardrails.report_check import CONTRAST_ID, ReportContrast
from analyst.llm.prompts import CalcBlock, similar_drops
from analyst.llm.schemas import Answer, Claim, Gap
from analyst.text import REGION_FOLDED, fold

# =====================================================================
# BLOQUE: Patrones de texto
# Se aplican sobre texto ya normalizado con fold() (minúsculas, sin tildes).
# =====================================================================
# Ids de evidencia (INV-01...), números, y palabras de canal.
_ID_PATTERN = re.compile(r"\b[A-Z]{1,5}(?:-[A-Za-z0-9]+)+\b")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_CHANNEL_WORD = re.compile(r"\b(tradicional(?:es)?|moderno)\b")
# Lenguaje causal ("causó", "se debe a", "explican el 70 %") y matices que lo suavizan.
_CAUSAL = re.compile(r"\b(causo|causaron|provoco|provocaron|se debe a|se debio a|debido a|a causa de|ocasiono|"
                     r"ocasionaron|origino|se atribuye a|se atribuyen a|atribuible a|se asocia a)\b|\bexplican?\s+(?:aproximadamente\s+)?(?:el\s+)?\d+\s*%")
# Frases que indican que un fragmento habla de limitaciones del estudio.
_LIMITATION_TEXT = re.compile(r"grupo de control|diseno experimental|elasticidad|cambio causal|antes/despues")
# Matices que vuelven aceptable una frase causal ("podría", "no está demostrado"...).
_HEDGE = re.compile(r"\b(no se puede|no se pudo|no es posible|no esta demostrad\w*|no demuestra|sin evidencia|"
                    r"no hay evidencia|no existe evidencia|podria|podrian|posible|sugiere|pudo|puede|pueden|"
                    r"hipotesis|no se ha)\b")
# Frases del tipo "no hay datos de..." y sus excepciones legítimas.
_NO_DATA = re.compile(r"\b(no hay|no existen|no se dispone de|no se cuenta con|sin|falta de|carece de|ausencia de)\s+"
                      r"(datos|informacion|registros|ventas)\b")
_SALES_BASE = re.compile(r"base de ventas|datos de ventas|informacion de ventas")
_NO_DATA_EXEMPT = re.compile(r"consumidor|entrevist|investigacion|estudio|visita|documento|cualitativ|muestra|"
                             r"inventario|precio|promocion|competen")
# "No hay contradicciones" / "es consistente": se compara con el contraste calculado por código.
_NO_CONTRADICTIONS = re.compile(
    r"\bno se (?:encontr\w+|hall\w+|identific\w+|observ\w+|detect\w+) (?:ninguna )?(?:contradiccion\w*|discrepancia\w*)|"
    r"\bsin (?:contradicciones|discrepancias)\b|\bno (?:hay|existen) (?:contradicciones|discrepancias)\b|"
    r"(?<!no )\b(?:son consistentes|es consistente|coinciden con los datos)\b")
# Titulares tipo "mayor caída" y referencias al reporte (para no confundir lo que dice el reporte con un hecho).
_HEADLINE = re.compile(r"\bmayor (?:caida|descenso|deterioro|disminucion|baja)\b|\bmas (?:cayo|cae|cayeron)\b|"
                       r"\bpeor desempeno\b")
_ABOUT_REPORT = re.compile(r"\breporte\b|\binforme\b|\brep-\d")


# =====================================================================
# BLOQUE: Contexto de validación
# =====================================================================
@dataclass
class ValidationContext:
    """Todo lo que el validador sabe: cálculos, evidencia, tipo de plan, contraste y columnas de la base."""

    calc_blocks: list[CalcBlock]          # incluye el bloque CONTRASTE-1 si existe
    evidence: list[Chunk]
    plan_type: str
    contrast: ReportContrast | None = None
    columns: tuple[str, ...] = ()


# =====================================================================
# BLOQUE: Verificación de cifras
# Una cifra de la respuesta debe aparecer (con tolerancia de redondeo) en las fuentes que cita.
# =====================================================================
def _canonical(token: str) -> Decimal | None:
    """'363.431.600' -> 363431600, '30,27' -> 30.27, '30.27' -> 30.27 (un punto seguido de 3 dígitos es separador de miles)."""
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", token):
        token = token.replace(".", "").replace(",", ".")
    elif "," in token:
        token = token.replace(".", "").replace(",", ".")
    try:
        return Decimal(token)
    except Exception:
        return None


def _decimals(token: str) -> int:
    """Cuántos decimales tiene la cifra tal como está escrita."""
    match = re.search(r"[.,](\d+)$", token)
    if not match or re.fullmatch(r"\d{1,3}(?:\.\d{3})+", token):
        return 0
    return len(match.group(1))


def _numbers(text: str) -> list[tuple[Decimal, int]]:
    """Cifras que vale la pena verificar en un texto: (valor, decimales). Se ignoran ids, fechas, años y dígitos sueltos."""
    text = _ID_PATTERN.sub(" ", text)
    text = re.sub(r"\b\d{4}-\d{2}(?:-\d{2})?\b", " ", text)
    text = re.sub(r"\bT[12]\b", " ", text)
    found = []
    for token in _NUMBER.findall(text):
        value = _canonical(token)
        if value is None or (value == value.to_integral() and (value < 10 or 2025 <= value <= 2027)):
            continue
        found.append((abs(value), _decimals(token)))
    return found


def figures_supported(text: str, source_texts: list[str]) -> list[str]:
    """Cifras de `text` que no aparecen en ninguna fuente (con tolerancia de redondeo). Lista vacía = todas respaldadas."""
    available = [v for src in source_texts for v, _ in _numbers(src)]
    missing = []
    for value, decimals in _numbers(text):
        quantum = Decimal(1).scaleb(-decimals)
        if not any(a.quantize(quantum, rounding=ROUND_HALF_UP) == value.quantize(quantum, rounding=ROUND_HALF_UP)
                   for a in available):
            missing.append(str(value.normalize()))
    return missing


# =====================================================================
# BLOQUE: Ayudantes de texto
# =====================================================================
def _regions(text: str) -> set[str]:
    """Regiones mencionadas en el texto (en minúsculas y sin tildes)."""
    return {m for m in REGION_FOLDED.findall(fold(text))}


def _channels(text: str) -> set[str]:
    """Canales mencionados (singular unificado: 'tradicionales' -> 'tradicional')."""
    return {c.rstrip("es") if c.startswith("tradicional") else c for c in _CHANNEL_WORD.findall(fold(text))}


def _sentences(text: str) -> list[str]:
    """Divide un texto en oraciones."""
    return [s for s in re.split(r"(?<=[.;!?])\s+", text) if s.strip()]


# =====================================================================
# BLOQUE: Reglas de verificación
# Cada regla devuelve el motivo (texto) si la afirmación se debe descartar, o None si pasa.
# =====================================================================
class _Checker:
    """Agrupa las reglas y los datos de apoyo (región de mayor caída, regiones con datos, ids válidos)."""

    def __init__(self, ctx: ValidationContext):
        self.ctx = ctx
        self.docs = {c.chunk_id: c for c in ctx.evidence}
        self.calc_text = {b.calc_id: b.text for b in ctx.calc_blocks}
        self.valid_ids = set(self.docs) | set(self.calc_text)
        self.all_text = list(self.calc_text.values()) + [c.text for c in ctx.evidence]
        self.top_region = self._top_region()
        self.data_regions = self._data_regions()
        self.only_similar_drops = self._only_similar_drops()
        self.contradicted = bool(ctx.contrast and not ctx.contrast.consistent)

    def _top_region(self) -> str | None:
        """Región de mayor caída según los cálculos (o según el contraste), para detectar titulares equivocados."""
        for block in self.ctx.calc_blocks:
            result = block.result
            if block.calc_id != CONTRAST_ID and result.get("group_by") == ["region"] and result.get("rows"):
                with_pct = [r for r in result["rows"] if r["pct_change"] is not None]
                worst = min(with_pct, key=lambda r: r["pct_change"], default=None)
                if worst and worst["pct_change"] < 0:
                    return fold(worst["keys"]["region"])
        if self.ctx.contrast and self.ctx.contrast.computed_top:
            return fold(self.ctx.contrast.computed_top)
        return None

    def _only_similar_drops(self) -> bool:
        """¿Todos los cálculos con grupos muestran caídas similares? Entonces ningún grupo es "la mayor caída"."""
        grouped = [b.result["rows"] for b in self.ctx.calc_blocks
                   if b.calc_id != CONTRAST_ID and len(b.result.get("rows") or []) > 1]
        return bool(grouped) and all(similar_drops(rows) for rows in grouped)

    def _data_regions(self) -> dict[str, int]:
        """Regiones con registros en los cálculos y cuántos (para desmentir un falso "no hay datos de X")."""
        found: dict[str, int] = {}
        for block in self.ctx.calc_blocks:
            if block.calc_id == CONTRAST_ID:
                continue
            for row in block.result.get("rows", []):
                region = row["keys"].get("region")
                if region:
                    found[fold(region)] = found.get(fold(region), 0) + row["n_t1"] + row["n_t2"]
        return found

    # -- reglas individuales
    def causal(self, text: str) -> str | None:
        folded = fold(text)
        if _CAUSAL.search(folded) and not _HEDGE.search(folded):
            return "afirma una causa como si estuviera demostrada; la evidencia disponible no mide causalidad"
        return None

    def headline(self, text: str) -> str | None:
        if not self.top_region and not self.only_similar_drops:
            return None
        for sentence in _sentences(text):
            folded = fold(sentence)
            if _HEADLINE.search(folded) and not _ABOUT_REPORT.search(folded):
                if self.only_similar_drops:
                    return ("señala una mayor caída, pero los cálculos muestran caídas similares en todos los grupos "
                            "(diferencias menores a 1 punto porcentual)")
                named = _regions(sentence)
                if named and self.top_region not in named:
                    return (f"nombra a {', '.join(sorted(r.capitalize() for r in named))} como la región de mayor caída, pero los cálculos "
                            f"indican {self.top_region.capitalize()}")
        return None

    def missing_data(self, text: str) -> str | None:
        folded = fold(text)
        if _NO_DATA.search(folded) and not _NO_DATA_EXEMPT.search(folded):
            with_data = [r for r in _regions(text) if self.data_regions.get(r)]
            if with_data:
                return (f"dice que no hay datos de {', '.join(sorted(r.capitalize() for r in with_data))}, pero los cálculos usan "
                        "registros de esas regiones")
        return None

    def missing_sales(self, text: str) -> str | None:
        folded = fold(text)
        if self.data_regions and _NO_DATA.search(folded) and _SALES_BASE.search(folded):
            return "dice que no hay información de la base de ventas, pero los cálculos se hicieron con ella"
        return None

    def no_contradictions(self, text: str) -> str | None:
        if self.contradicted and _NO_CONTRADICTIONS.search(fold(text)):
            return "afirma que no hay contradicciones, pero el contraste calculado con código encontró diferencias"
        return None

    def figures(self, text: str, sources: list[str]) -> str | None:
        texts = [self.calc_text.get(s) or self.docs[s].text for s in sources] or self.all_text
        missing = figures_supported(text, texts)
        if missing:
            return f"la cifra {', '.join(missing)} no aparece en las fuentes que cita"
        return None


# =====================================================================
# BLOQUE: Revisión de fuentes y afirmaciones
# =====================================================================
def _split_sources(checker: _Checker, claim_text: str, sources: list[str],
                   avisos: list[str]) -> tuple[list[str], list[str]]:
    """Quita ids inventados y fuentes de documentos que hablan de otra región (u otro canal) distinto al de la afirmación.

    Devuelve las fuentes que se quedan y una descripción de cada una descartada por hablar de otra región."""
    kept = []
    for source in dict.fromkeys(sources):
        if source not in checker.valid_ids:
            avisos.append(f"Se eliminó la fuente inexistente '{source}' de una afirmación.")
        else:
            kept.append(source)
    mentioned, channels = _regions(claim_text), _channels(claim_text)
    result, wrong_region = [], []
    for source in kept:
        doc = checker.docs.get(source)
        if doc and channels and not channels & _channels(doc.text):
            tail = "no menciona ese canal"
            wrong_region.append(f"{source} {tail}")
            avisos.append(f"Se eliminó la fuente {source} ({tail}): la afirmación habla del canal "
                          f"{', '.join(sorted(c.capitalize() for c in channels))}.")
        elif doc and mentioned and not mentioned & {fold(r) for r in doc.regions}:
            tail = f"dice de {', '.join(doc.regions)}" if doc.regions else "no menciona ninguna región"
            wrong_region.append(f"{source} {tail}")
            avisos.append(f"Se eliminó la fuente {source} ({tail}): la afirmación habla de "
                          f"{', '.join(sorted(r.capitalize() for r in mentioned))}.")
        else:
            result.append(source)
    return result, wrong_region


def _check_claim(checker: _Checker, claim: Claim, section: str, avisos: list[str]) -> tuple[FinalClaim | None, str | None]:
    """Devuelve (afirmación a mostrar, None) o (None, motivo del descarte). Un motivo '' significa 'redundante, no es un error'."""
    sources, wrong_region = _split_sources(checker, claim.texto, claim.fuentes, avisos)
    if not sources and not wrong_region:
        return None, "no cita ninguna fuente válida de esta consulta"
    documents = [s for s in sources if s in checker.docs]
    if section in ("hechos", "hipotesis") and not documents:
        if wrong_region:
            mentioned = ", ".join(sorted(r.capitalize() for r in _regions(claim.texto) | _channels(claim.texto)))
            return None, f"menciona {mentioned}, pero la fuente citada no lo respalda ({'; '.join(wrong_region)})"
        if section == "hechos":
            return None, ""       # un hecho que cita solo un cálculo: el código ya lo dice, mejor
        return None, "es una hipótesis y no cita ningún documento que la sugiera"
    if not sources:
        return None, "no cita ninguna fuente válida de esta consulta"
    if section == "hechos" and all(checker.docs[d].kind == "afirmacion_reporte" for d in documents):
        return None, "es una afirmación del reporte comercial, no un hecho verificado"
    rules = [checker.figures(claim.texto, sources), checker.headline(claim.texto), checker.missing_data(claim.texto),
             checker.no_contradictions(claim.texto)]
    if section == "hechos":
        rules.insert(0, checker.causal(claim.texto))
    reason = next((r for r in rules if r), None)
    if reason:
        return None, reason
    if section == "contradicciones" and _NO_CONTRADICTIONS.search(fold(claim.texto)):
        return None, "dice que no hay contradicciones: no es una contradicción"
    if section == "contradicciones" and len(sources) < 2:
        return None, "una contradicción es entre fuentes: debe citar al menos dos"
    origin = ORIGIN_DOCUMENT if section == "hechos" else ORIGIN_MODEL
    return FinalClaim(claim.texto, sources, origin), None


def _check_gap(checker: _Checker, gap: Gap) -> str | None:
    """Revisa un "información insuficiente" del modelo: no puede negar datos que sí existen."""
    return checker.missing_data(gap.texto) or checker.missing_sales(gap.texto) or checker.no_contradictions(gap.texto)


def _fallback_summary(ctx: ValidationContext, facts: list[FinalClaim]) -> str:
    """Resumen de respaldo (escrito por código) cuando el del modelo no pasa las reglas."""
    if ctx.contrast:
        return ctx.contrast.verdict
    if facts:
        return facts[0].texto
    return "Consulta los cálculos y la evidencia adjuntos."


def _causality_limits(ctx: ValidationContext) -> list[FinalClaim]:
    """Límites que el código sabe por sí mismo: la base no mide promociones/precio y los estudios cualitativos no miden impacto."""
    limits = []
    cause_columns = [c for c in ctx.columns if re.search(r"promo|precio|descuento", fold(c))]
    if ctx.columns and not cause_columns:
        limits.append(FinalClaim("La base de ventas no contiene variables de promoción ni de precio: los datos no "
                                 "permiten medir el efecto de las promociones de la competencia.", [], ORIGIN_CALC))
    cited = [c.chunk_id for c in ctx.evidence if c.kind == "limitacion" or _LIMITATION_TEXT.search(fold(c.text))]
    if cited:
        limits.append(FinalClaim("La evidencia cualitativa (entrevistas y visitas) no mide impacto: no hubo grupo de "
                                 "control ni mediciones antes y después, así que no establece causalidad.", cited,
                                 ORIGIN_CALC))
    return limits


# =====================================================================
# BLOQUE: Punto de entrada
# =====================================================================
def validate_answer(answer: Answer, ctx: ValidationContext) -> FinalAnswer:
    """Valida la respuesta del modelo y arma la final: hechos, hipótesis, contradicciones, insuficiente y resumen."""
    checker = _Checker(ctx)
    final = FinalAnswer(resumen="")
    facts = calc_facts(ctx.calc_blocks)

    # 1) Hechos: los escritos por código, más los del modelo basados en documentos que superen las reglas
    final.hechos = list(facts)
    if ctx.contrast and ctx.contrast.agreement:
        final.hechos.append(ctx.contrast.agreement)
    for claim in answer.hechos:
        kept, reason = _check_claim(checker, claim, "hechos", final.avisos)
        if kept:
            final.hechos.append(kept)
        elif reason:
            final.descartes.append(Discard("hechos", claim.texto, reason))

    # 2) Hipótesis: necesitan un documento que las sugiera
    for claim in answer.hipotesis:
        kept, reason = _check_claim(checker, claim, "hipotesis", final.avisos)
        if kept:
            final.hipotesis.append(kept)
        elif reason:
            final.descartes.append(Discard("hipotesis", claim.texto, reason))

    # 3) Contradicciones: primero las del código, luego las del modelo que sobrevivan
    if ctx.contrast:
        final.contradicciones = list(ctx.contrast.contradictions)
    for claim in ([] if ctx.contrast else answer.contradicciones):
        kept, reason = _check_claim(checker, claim, "contradicciones", final.avisos)
        if kept:
            final.contradicciones.append(kept)
        elif reason:
            final.descartes.append(Discard("contradicciones", claim.texto, reason))

    # 4) Lo que no se puede afirmar: los vacíos del modelo que se sostienen, más lo que el código puede establecer
    for gap in answer.insuficiente:
        reason = _check_gap(checker, gap)
        if reason:
            final.descartes.append(Discard("insuficiente", gap.texto, reason))
        else:
            final.insuficiente.append(FinalClaim(gap.texto, [], ORIGIN_MODEL))
    if ctx.contrast:
        final.insuficiente = list(ctx.contrast.unverifiable) + final.insuficiente
    if ctx.plan_type == "causalidad":
        final.insuficiente = _causality_limits(ctx) + final.insuficiente

    # 5) Resumen: en el contraste con el reporte es el veredicto del código; si no, el del modelo si se sostiene
    limits = _causality_limits(ctx) if ctx.plan_type == "causalidad" else []
    if ctx.plan_type == "contraste_reporte" and ctx.contrast:
        final.resumen, final.resumen_origen = ctx.contrast.verdict, ORIGIN_CONTRAST
    elif limits:
        final.resumen, final.resumen_origen = "No se puede afirmar causalidad. " + limits[0].texto, ORIGIN_CALC
    else:
        reason = (checker.figures(answer.resumen, []) or checker.causal(answer.resumen) or
                  checker.headline(answer.resumen) or checker.no_contradictions(answer.resumen)
                  if answer.resumen.strip() else "el modelo no escribió un resumen")
        if reason:
            final.descartes.append(Discard("resumen", answer.resumen, reason))
            final.resumen, final.resumen_origen = _fallback_summary(ctx, facts), ORIGIN_CALC
        else:
            final.resumen, final.resumen_origen = answer.resumen, ORIGIN_MODEL
    return final
