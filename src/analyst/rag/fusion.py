"""
FUSIÓN DE RANKINGS (RRF)

Qué hace: une varios rankings (léxico y semántico) en uno solo usando únicamente las posiciones.
Con qué se conecta: lo usa rag/retriever.py en el modo híbrido.
Idea clave: puntaje(fragmento) = suma, por ranking, de peso / (k + posición). Se usan posiciones
y no puntajes porque los puntajes de BM25 y la similitud coseno no son comparables. Los pesos valen
1 por defecto (RRF clásico) y existen solo para que un ranking cuente más que otro.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FusedHit:
    """Un fragmento del ranking fusionado, con su puntaje y su posición en cada ranking original."""

    chunk_id: str
    score: float
    ranks: dict[str, int] = field(default_factory=dict)   # nombre del ranking -> posición (empieza en 1)


def reciprocal_rank_fusion(rankings: dict[str, list[str]], k: int = 60,
                           weights: dict[str, float] | None = None) -> list[FusedHit]:
    """Fusiona los rankings dados (nombre -> ids ordenados) y devuelve los fragmentos de mejor a peor."""
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for name, ranking in rankings.items():
        for position, chunk_id in enumerate(ranking, start=1):
            if name in ranks.get(chunk_id, {}):
                continue   # un id repetido en el mismo ranking cuenta una sola vez
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (weights or {}).get(name, 1.0) / (k + position)
            ranks.setdefault(chunk_id, {})[name] = position
    # Desempate estable: mejor mejor-posición, luego el id.
    ordered = sorted(scores, key=lambda cid: (-scores[cid], min(ranks[cid].values()), cid))
    return [FusedHit(cid, scores[cid], ranks[cid]) for cid in ordered]
