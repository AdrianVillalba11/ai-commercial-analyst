"""
DETECCIÓN DE INYECCIÓN DE PROMPT EN DOCUMENTOS (capas 1 y 2 de la defensa)

Qué hace: revisa cada fragmento de los documentos y pone en cuarentena los que intentan dar órdenes
a un modelo de IA (ignorar información, imponer una respuesta, pedir que se oculte algo).
Con qué se conecta: rag/retriever.py solo indexa los fragmentos seguros y los vuelve a revisar al
consultar; la API muestra los fragmentos en cuarentena en el panel de alertas de seguridad.
Idea clave: los documentos son datos NO confiables. Un fragmento en cuarentena no se esconde: se
reporta para que el analista vea el intento.

Limitación conocida: esta capa es heurística. Una paráfrasis sin palabras gatillo puede pasar
(hay una prueba que lo documenta). Las demás capas no dependen de ella: las cifras vienen de
pandas y la salida del modelo se valida después.
"""
import re
import unicodedata
from dataclasses import asdict, dataclass, field

from analyst.docs.chunker import Chunk, load_all_chunks

# =====================================================================
# BLOQUE: Umbrales y estados
# Puntaje >= 3: cuarentena (no se indexa). Puntaje 1-2: advertencia (se indexa pero se avisa).
# =====================================================================
QUARANTINE_THRESHOLD = 3
WARNING_THRESHOLD = 1

CLEAN, WARNING, QUARANTINE = "limpio", "advertencia", "cuarentena"

# =====================================================================
# BLOQUE: Normalización antes de buscar señales
# Deshace los trucos habituales de ofuscación: caracteres invisibles, letras parecidas
# de otro alfabeto (homoglifos) y letras separadas ("i g n o r a").
# =====================================================================
_INVISIBLE = re.compile("[​-‏‪-‮⁠-⁤﻿­]")
_HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",   # cirílico
    "і": "i", "ѕ": "s", "ј": "j", "ɡ": "g", "ο": "o",                        # cirílico / latino / griego
})
_SPACED_LETTERS = re.compile(r"\b(?:[a-z][ .\-_*]){3,}[a-z]\b")


def normalize_for_detection(text: str) -> str:
    """Forma canónica para comparar: minúsculas, sin tildes ni caracteres invisibles, letras unidas."""
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE.sub("", text).casefold().translate(_HOMOGLYPHS)
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    text = _SPACED_LETTERS.sub(lambda m: re.sub(r"[ .\-_*]", "", m.group(0)), text)
    return re.sub(r"\s+", " ", text).strip()


# =====================================================================
# BLOQUE: Señales de inyección
# Cada señal tiene un peso y varios patrones; cuenta una sola vez aunque coincidan varios.
# =====================================================================
# Una orden empieza una oración o cláusula ("...archivo: ignora cualquier..."), por eso los verbos
# ambiguos solo cuentan ahí. "El reporte ignora la conciliación" tiene sujeto y no coincide.
_START = (r"(?:^|[.:;!?\"“”'(\[>\n]|[-•*]\s)\s*"
          r"(?:(?:por favor|please|ahora|ya|y|and|then|luego|despues|tambien|also)[,\s]+)*")


@dataclass(frozen=True)
class Signal:
    """Una familia de patrones sospechosos con su peso en el puntaje."""

    name: str
    weight: int
    description: str
    patterns: tuple[re.Pattern, ...]


def _signal(name: str, weight: int, description: str, *patterns: str) -> Signal:
    """Crea una señal compilando sus patrones."""
    return Signal(name, weight, description, tuple(re.compile(p) for p in patterns))


# Las cinco señales: dirigido a IA, ordena ignorar, impone respuesta, pide ocultar, marcador de rol.
SIGNALS = (
    _signal(
        "dirigido_a_ia", 2, "Se dirige a un asistente de IA o a un modelo de lenguaje",
        r"\b(?:asistente|agente|modelo|bot|chatbot)s?\s+(?:de\s+)?(?:ia|inteligencia artificial|lenguaje|ai)\b",
        r"\b(?:para|dirigid[oa] a)\s+(?:el|la|los)?\s*(?:asistente|agente|modelo|ia|llm)\b",
        r"\b(?:ai|artificial intelligence)\s+assistant\b",
        r"\b(?:llm|language model)\b",
        r"\b(?:si|if)\s+(?:eres|you are)\s+(?:una?\s+)?(?:ia|ai|modelo|llm|language model|assistant|asistente)\b",
    ),
    _signal(
        "ordena_ignorar", 3, "Ordena ignorar o descartar información",
        _START + r"(?:ignora|ignore|ignoren|olvida|olvide|forget|omite|omita|descarta|descarte|"
                 r"desconsidera|desestima|disregard|override|sobrescribe)\b",
        _START + r"no\s+(?:tengas|tomes|uses|utilices|consideres|consultes|confies)\b",
        _START + r"do\s+not\s+(?:use|consider|trust|rely)\b",
    ),
    _signal(
        "impone_respuesta", 3, "Impone una respuesta fija",
        r"\bresponde(?:r)?\s+siempre\b",
        r"\bsiempre\s+(?:responde|afirma|di|indica|contesta|reporta|concluye)\b",
        r"\bdebes\s+(?:responder|afirmar|decir|indicar|concluir|contestar|reportar)\b",
        r"\btu\s+(?:respuesta|salida)\s+(?:debe|tiene que)\b",
        r"\balways\s+(?:answer|say|respond|reply|state|conclude)\b",
        r"\byou\s+must\s+(?:say|answer|respond|state|conclude)\b",
        r"\byour\s+(?:answer|response)\s+(?:must|should)\b",
        _START + r"(?:afirma|di|indica|concluye|reporta|contesta|responde)\s+(?:siempre\s+)?que\b",
        _START + r"(?:say|state|answer|conclude|report)\s+that\b",
    ),
    _signal(
        "pide_ocultar", 3, "Pide ocultar la instrucción al usuario",
        _START + r"no\s+(?:le\s+)?(?:menciones|mencione|digas|reveles|informes|comentes|muestres|expliques)\b",
        r"\bnunca\s+(?:menciones|reveles|digas)\b",
        r"\bsin\s+(?:mencionar|revelar|decir|avisar)\b.{0,30}\b(?:instruccion|orden|nota|usuario)\b",
        _START + r"(?:oculta|esconde)\b.{0,40}\b(?:instruccion|orden|usuario|esto)\b",
        r"\bdo\s+not\s+(?:mention|tell|reveal|disclose)\b",
        r"\b(?:never|don'?t)\s+(?:mention|reveal|tell)\b",
        r"\bkeep\s+(?:this|it)\s+(?:secret|hidden)\b",
    ),
    _signal(
        "marcador_de_rol", 2, "Usa marcadores de rol o de sistema para dar órdenes",
        r"<\s*/?\s*(?:system|assistant|instructions?|prompt)\s*>",
        r"\[\s*(?:system|inst|instructions?)\s*\]",
        r"\b(?:system|sistema)\s+prompt\b",
        r"^#{1,6}\s*(?:instruction|instruccion)",
        r"\bnuevas?\s+instrucciones?\b",
        r"\bnew\s+instructions?\b",
        r"\byou\s+are\s+now\b",
        r"\bahora\s+eres\b",
        r"\ba\s+partir\s+de\s+ahora\b",
        r"\bfrom\s+now\s+on\b",
        _START + r"(?:actua|act)\s+como\b",
        r"\bact\s+as\b",
    ),
)
# Señal extra: el propio documento marca el contenido como importado sin verificar.
PROVENANCE = Signal("origen_no_verificado", 1, "El propio documento lo marca como contenido importado sin verificar", ())


# =====================================================================
# BLOQUE: Veredictos
# =====================================================================
@dataclass(frozen=True)
class SignalHit:
    """Una señal que se activó, con el trozo de texto que la disparó."""

    signal: str
    weight: int
    description: str
    evidence: str


@dataclass
class Verdict:
    """Resultado de revisar un texto: puntaje total, estado (limpio/advertencia/cuarentena) y señales."""

    score: int
    status: str
    hits: list[SignalHit] = field(default_factory=list)


def _evidence(normalized: str, match: re.Match) -> str:
    """Las palabras detectadas más un poco de lo que sigue, para que el analista vea la orden misma."""
    snippet = normalized[match.start():match.end() + 45]
    return snippet.lstrip(" .:;!?\"“”'([>-•*")[:90].strip()


def scan_text(text: str, imported: bool = False) -> Verdict:
    """Puntúa un texto. Cada señal cuenta una vez, sin importar cuántas veces aparezca."""
    normalized = normalize_for_detection(text)
    hits = []
    for signal in SIGNALS:
        for pattern in signal.patterns:
            match = pattern.search(normalized)
            if match:
                hits.append(SignalHit(signal.name, signal.weight, signal.description, _evidence(normalized, match)))
                break
    if imported:
        hits.append(SignalHit(PROVENANCE.name, PROVENANCE.weight, PROVENANCE.description, "contenido importado"))
    score = sum(h.weight for h in hits)
    status = QUARANTINE if score >= QUARANTINE_THRESHOLD else WARNING if score >= WARNING_THRESHOLD else CLEAN
    return Verdict(score, status, hits)


@dataclass
class FlaggedChunk:
    """Fragmento marcado (advertencia o cuarentena) junto con su veredicto."""

    chunk: Chunk
    verdict: Verdict

    def to_dict(self) -> dict:
        """Forma para el panel de alertas de seguridad. La interfaz debe mostrarlo como texto plano."""
        chunk = self.chunk
        return {
            "chunk_id": chunk.chunk_id, "doc": chunk.doc, "doc_title": chunk.doc_title,
            "section": chunk.section, "page": chunk.page,
            "status": self.verdict.status, "score": self.verdict.score,
            "reasons": [asdict(h) for h in self.verdict.hits],
            "excerpt": chunk.text[:400],
        }


@dataclass
class ScreeningResult:
    """Resultado de revisar todos los fragmentos."""

    safe: list[Chunk]                 # limpios + con advertencia: pueden entrar a los índices
    quarantined: list[FlaggedChunk]   # excluidos de todos los índices
    warnings: list[FlaggedChunk]      # permitidos, pero señalados en el reporte


# =====================================================================
# BLOQUE: Revisión de fragmentos
# =====================================================================
def _scan_chunk(chunk: Chunk) -> Verdict:
    """Revisa un fragmento; los importados llevan un punto extra por origen no verificado."""
    return scan_text(chunk.text, imported=chunk.kind == "dato_importado")


def screen_chunks(chunks: list[Chunk]) -> ScreeningResult:
    """Separa los fragmentos en seguros, con advertencia y en cuarentena."""
    safe, quarantined, warnings = [], [], []
    for chunk in chunks:
        verdict = _scan_chunk(chunk)
        if verdict.status == QUARANTINE:
            quarantined.append(FlaggedChunk(chunk, verdict))
            continue
        safe.append(chunk)
        if verdict.status == WARNING:
            warnings.append(FlaggedChunk(chunk, verdict))
    return ScreeningResult(safe, quarantined, warnings)


def filter_safe(chunks: list[Chunk]) -> list[Chunk]:
    """Segunda revisión al consultar, sobre lo que devolvió el buscador."""
    return [c for c in chunks if _scan_chunk(c).status != QUARANTINE]


def load_safe_corpus(raw_dir=None) -> ScreeningResult:
    """Carga los documentos y los revisa; es el punto de entrada que usa el resto de la app."""
    return screen_chunks(load_all_chunks(raw_dir))
