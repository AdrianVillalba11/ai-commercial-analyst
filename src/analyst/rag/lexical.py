"""Lexical search: BM25 over the chunks, with Spanish-aware tokenization.

Good at exact terms (ids such as CAM-150 or SU-T-003, field names, region names); it cannot
match synonyms, which is what the semantic search is for.
"""
import math
import re
from collections import Counter

from analyst.docs.chunker import Chunk
from analyst.text import fold

STOPWORDS = frozenset("""
a al algo algun alguna algunas alguno algunos ante como con contra cual cuales cuando cuanto de del desde donde
e el ella ellas ello ellos en entre era eran es esa esas ese eso esos esta estan estas este esto estos fue fueron
ha han hasta hay la las le les lo los mas me mi muy ni no nos o para pero por porque que quien se segun ser si sin
sobre son su sus te tambien tu un una unas uno unos y ya
""".split())

_TOKEN = re.compile(r"[a-z0-9]+(?:[_-][a-z0-9]+)*")


def stem(word: str) -> str:
    """Light Spanish stemmer: collapses plurals, final vowels and a few common endings.

    Not linguistically complete; it only has to make 'faltante'/'faltantes' or
    'promoción'/'promociones' meet. Ids and compound names are left untouched.
    """
    if not word.isalpha():
        return word
    for suffix in ("ciones", "cion", "idades", "idad", "mente"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if word.endswith("es") and len(word) > 5:
        word = word[:-2]
    elif word.endswith("s") and len(word) > 3:
        word = word[:-1]
    if len(word) > 4 and word[-1] in "aeo":
        word = word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Words and ids. 'su-t-003' stays whole; 'venta_neta_cop' also yields 'venta', 'neta', 'cop'."""
    tokens = []
    for raw in _TOKEN.findall(fold(text)):
        if "-" in raw or "_" in raw:
            tokens.append(raw)
            tokens += [p for p in re.split(r"[_-]", raw) if len(p) >= 3 and not p.isdigit() and p not in STOPWORDS]
        elif raw not in STOPWORDS:
            tokens.append(stem(raw))
    return tokens


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self._ids = [c.chunk_id for c in chunks]
        self._k1, self._b = k1, b
        self._tf = [Counter(tokenize(c.embed_text)) for c in chunks]
        self._lengths = [sum(tf.values()) for tf in self._tf]
        self._avg_length = (sum(self._lengths) / len(chunks)) if chunks else 0.0
        document_frequency = Counter(term for tf in self._tf for term in tf)
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in document_frequency.items()}

    def search(self, query: str, k: int | None = None) -> list[tuple[str, float]]:
        """Chunks that share at least one term with the query, best first. Never returns zero scores."""
        terms = set(tokenize(query)) & set(self._idf)
        scored = []
        for position, tf in enumerate(self._tf):
            score = 0.0
            for term in terms:
                f = tf.get(term, 0)
                if f:
                    norm = 1 - self._b + self._b * self._lengths[position] / self._avg_length
                    score += self._idf[term] * f * (self._k1 + 1) / (f + self._k1 * norm)
            if score > 0:
                scored.append((self._ids[position], score, position))
        scored.sort(key=lambda item: (-item[1], item[2]))
        ranking = [(chunk_id, score) for chunk_id, score, _ in scored]
        return ranking[:k] if k else ranking
