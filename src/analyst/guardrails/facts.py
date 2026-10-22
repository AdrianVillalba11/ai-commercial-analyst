"""
HECHOS NUMÉRICOS ESCRITOS POR CÓDIGO

Qué hace: redacta, a partir de los resultados de pandas, las frases con cifras que van en "Hechos".
Con qué se conecta: toma los bloques CALC-n de llm/prompts.py; el validador los pone primero en la respuesta.
Idea clave: las cifras nunca las escribe el modelo. Cada hecho es una frase completa con su sujeto,
para poder leerse sola.
"""
from analyst.analytics.metrics import format_number, format_pct
from analyst.guardrails.models import ORIGIN_CALC, FinalClaim
from analyst.llm.prompts import CalcBlock, similar_drops

# =====================================================================
# BLOQUE: Etiquetas y formato
# =====================================================================
_DIMENSIONS = {"region": "región", "canal": "canal", "presentacion": "presentación"}
_METRICS = {"venta_neta_cop": "ventas netas (COP)", "unidades": "unidades"}


def _label(keys: dict) -> str:
    """Nombre del grupo (ej. 'Sur / Moderno'), o 'Total' si no hay grupos."""
    return " / ".join(str(v) for v in keys.values()) or "Total"


def _signed(pct: float) -> str:
    """Porcentaje con signo + explícito cuando es positivo."""
    return ("+" if pct > 0 else "") + format_pct(pct)


def _filters(result: dict) -> str:
    """Filtros aplicados, en texto."""
    return ", ".join(f"{k} = {','.join(v)}" for k, v in result["filters"].items())


def _scope(result: dict) -> str:
    """Alcance del cálculo en texto: 'por región (canal = Moderno)'."""
    groups = " y ".join(_DIMENSIONS[g] for g in result["group_by"])
    filters = ", ".join(f"{k} = {','.join(v)}" for k, v in result["filters"].items())
    return " ".join(part for part in (f"por {groups}" if groups else "", f"({filters})" if filters else "") if part)


# =====================================================================
# BLOQUE: Generación de hechos
# =====================================================================
def calc_facts(blocks: list[CalcBlock]) -> list[FinalClaim]:
    """Por cada cálculo: mayor caída (o caída similar), grupos que crecieron y total, todo con su fuente CALC-n."""
    facts: list[FinalClaim] = []
    totals: dict[str, FinalClaim] = {}
    for block in blocks:
        result = block.result
        if block.calc_id.startswith("CONTRASTE") or not result.get("total"):
            continue
        metric = _METRICS[result["metric"]]
        total, rows = result["total"], result["rows"]
        scope = _scope(result)
        rows_with_pct = [r for r in rows if r["pct_change"] is not None]
        drop = min(rows_with_pct, key=lambda r: r["pct_change"], default=None)
        # Si todos cayeron casi igual, no se destaca ninguno como "el mayor".
        similar = similar_drops(rows)
        if similar:
            facts.append(FinalClaim(
                f"Caída similar en todos los grupos {scope}: entre {format_pct(similar[1])} y {format_pct(similar[0])} "
                "(diferencias menores a 1 punto porcentual).", [block.calc_id], ORIGIN_CALC))
        elif drop and drop["pct_change"] < 0:
            text = (f"Mayor caída de {metric} {scope}: {_label(drop['keys'])}, {format_pct(drop['pct_change'])} "
                    f"(T1 {format_number(drop['t1'])} → T2 {format_number(drop['t2'])})")
            if len(rows) > 1 and total["abs_change"] < 0 and drop["share_of_total_change_pct"] is not None:
                share = f"{drop['share_of_total_change_pct']:.1f}".replace(".", ",")
                whole = f"del total filtrado ({_filters(result)})" if result["filters"] else "total de Brío"
                text += f". Explica el {share} % de la caída {whole}."
            facts.append(FinalClaim(text, [block.calc_id], ORIGIN_CALC))
        grew = [r for r in rows_with_pct if r["pct_change"] > 0][:3]
        if grew:
            facts.append(FinalClaim("Crecieron: " + ", ".join(f"{_label(r['keys'])} {_signed(r['pct_change'])}" for r in grew)
                                    + ".", [block.calc_id], ORIGIN_CALC))
        if total["pct_change"] is not None:
            label = f"Total Brío con filtro {_filters(result)}" if result["filters"] else "Total Brío"
            text = (f"{label} ({metric}): {format_pct(total['pct_change'])} "
                    f"(T1 {format_number(total['t1'])} → T2 {format_number(total['t2'])}).")
            if text in totals:
                totals[text].fuentes.append(block.calc_id)          # misma cifra, otro cálculo que la muestra
            else:
                totals[text] = FinalClaim(text, [block.calc_id], ORIGIN_CALC)
                facts.append(totals[text])
    return facts
