"""Reciprocal Rank Fusion: merge several rankings into one using only positions.

score(chunk) = sum over rankers of weight / (k + position). Positions are used instead of
scores because BM25 scores and cosine similarities are not comparable. Weights default to 1
(plain RRF) and only exist to let one ranker count more than another.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FusedHit:
    chunk_id: str
    score: float
    ranks: dict[str, int] = field(default_factory=dict)   # ranker name -> 1-based position


def reciprocal_rank_fusion(rankings: dict[str, list[str]], k: int = 60,
                           weights: dict[str, float] | None = None) -> list[FusedHit]:
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for name, ranking in rankings.items():
        for position, chunk_id in enumerate(ranking, start=1):
            if name in ranks.get(chunk_id, {}):
                continue   # a repeated id in the same ranking only counts once
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (weights or {}).get(name, 1.0) / (k + position)
            ranks.setdefault(chunk_id, {})[name] = position
    ordered = sorted(scores, key=lambda cid: (-scores[cid], min(ranks[cid].values()), cid))
    return [FusedHit(cid, scores[cid], ranks[cid]) for cid in ordered]
