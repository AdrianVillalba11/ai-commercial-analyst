"""
EVALUACIÓN DE LA BÚSQUEDA

Qué hace: mide qué tan bien encuentra el buscador la evidencia correcta, con preguntas de referencia
y los ids de fragmentos que deberían volver.
Con qué se conecta: lo usan tests/test_retrieval_quality.py y los scripts de evaluación; no interviene
cuando la app responde preguntas.
Idea clave: dos conjuntos separados, fijados ANTES de medir y nunca ajustados a los resultados:
- RETRIEVAL_CASES: sirvió para elegir los pesos de la fusión.
- VALIDATION_CASES: escrito después y usado solo para confirmar la elección (sin ajustar nada con él).
"""
import math
from dataclasses import dataclass

from analyst.rag.retriever import HybridRetriever

# =====================================================================
# BLOQUE: Casos de referencia
# Cada caso: (pregunta, ids de evidencia que deberían aparecer entre los primeros TOP_K).
# =====================================================================
TOP_K = 6

RETRIEVAL_CASES: list[tuple[str, set[str]]] = [
    ("desabastecimiento en la región sur", {"OC-SUR-01", "INV-02", "OC-REGION-SUR"}),
    ("¿qué dijeron los consumidores sobre el precio en el Sur?", {"INV-01", "ENT-SUR-07", "INV-REGION-SUR"}),
    ("promociones de la competencia en Centro", {"OC-CEN-02", "INV-04", "ENT-CEN-04", "CAM-119"}),
    ("¿las promociones causaron la caída de ventas?", {"INV-04", "REP-02", "INV-LIM-02"}),
    ("descuentos de otras marcas", {"ENT-CEN-04", "INV-04", "CAM-103"}),
    ("formatos pequeños en Norte", {"INV-03", "ENT-NOR-02", "OC-NOR-03"}),
    ("limitaciones de la muestra del estudio", {"INV-LIM-01", "INV-LIM-04"}),
    ("¿cuál fue la variación de Centro según el reporte?", {"REP-REGION-CENTRO", "REP-01"}),
    ("visitas al punto de venta SU-T-003", {"CAM-101"}),
    ("CAM-150", {"CAM-150"}),
    ("¿cómo se define el primer trimestre?", {"CTX-CONVENCIONES-DE-ANALISIS"}),
    ("qué significa venta_neta_cop", {"CTX-DICCIONARIO-DE-LA-BASE-DE-VENTAS"}),
    ("reposición del formato familiar en Occidente", {"OC-OCC-05", "CAM-145"}),
    ("¿hay evidencia de consumidores en Oriente?", {"INV-LIM-03", "OC-ORI-04"}),
]

VALIDATION_CASES: list[tuple[str, set[str]]] = [
    ("¿cuántas visitas con faltantes hubo en Oriente?", {"OC-REGION-ORIENTE", "OC-ORI-04"}),
    ("¿se puede inferir qué región vendió menos a partir de las entrevistas?", {"INV-05", "INV-LIM-04"}),
    ("¿qué se sabe de la sensibilidad al precio entre los comerciantes?", {"OC-SUR-01", "INV-01"}),
    ("¿quién sugirió que Centro podría ser la región más afectada?", {"OC-CEN-02"}),
    ("SU-M-002 exhibición de competidor", {"CAM-103"}),
    ("¿qué recomienda el reporte comercial respecto a los inventarios?", {"REP-05", "REP-03"}),
    ("¿en qué fechas se hizo el trabajo de campo de las entrevistas?", {"INV-S0-1"}),
    ("¿el reporte comercial se hizo antes o después de conciliar la base de ventas?", {"REP-S5-1", "REP-S6-1"}),
    ("¿qué columnas tiene la base de ventas?", {"CTX-DICCIONARIO-DE-LA-BASE-DE-VENTAS", "CTX-INFORMACION-DISPONIBLE"}),
    ("dificultades de reposición del formato grande", {"OC-OCC-05", "CAM-145", "CAM-101"}),
    ("¿cuántas personas entrevistadas cambiaron de marca?", {"INV-01"}),
    ("¿qué observaron los visitadores en los puntos de venta de Norte?", {"OC-NOR-03", "OC-REGION-NORTE", "CAM-130"}),
]


# =====================================================================
# BLOQUE: Métricas por caso y por modo
# hit: ¿apareció alguno esperado? recall/precision: cobertura y exactitud. top1/MRR: qué tan arriba
# queda el primer acierto. nDCG: premia que los aciertos estén en las primeras posiciones.
# =====================================================================
@dataclass
class CaseResult:
    """Resultado de una pregunta de referencia: lo esperado frente a lo recuperado."""

    question: str
    expected: set[str]
    retrieved: list[str]

    @property
    def hit(self) -> bool:
        return bool(self.expected & set(self.retrieved))

    @property
    def recall(self) -> float:
        return len(self.expected & set(self.retrieved)) / len(self.expected)

    @property
    def precision(self) -> float:
        return len(self.expected & set(self.retrieved)) / max(len(self.retrieved), 1)

    @property
    def top1(self) -> bool:
        return bool(self.retrieved) and self.retrieved[0] in self.expected

    @property
    def first_rank(self) -> int | None:
        for position, chunk_id in enumerate(self.retrieved, start=1):
            if chunk_id in self.expected:
                return position
        return None

    @property
    def ndcg(self) -> float:
        gain = sum(1 / math.log2(i + 2) for i, cid in enumerate(self.retrieved) if cid in self.expected)
        ideal = sum(1 / math.log2(i + 2) for i in range(min(len(self.expected), TOP_K)))
        return gain / ideal


@dataclass
class ModeSummary:
    """Promedios de las métricas para un modo de búsqueda sobre todos los casos."""

    mode: str
    cases: list[CaseResult]

    def _mean(self, attr: str) -> float:
        """Promedio de un atributo numérico (o booleano) de los casos."""
        return sum(float(getattr(c, attr)) for c in self.cases) / len(self.cases)

    @property
    def hits(self) -> int:
        return sum(c.hit for c in self.cases)

    @property
    def mean_recall(self) -> float:
        return self._mean("recall")

    @property
    def mean_precision(self) -> float:
        return self._mean("precision")

    @property
    def mean_ndcg(self) -> float:
        return self._mean("ndcg")

    @property
    def top1_rate(self) -> float:
        return self._mean("top1")

    @property
    def mrr(self) -> float:
        return sum(1 / c.first_rank if c.first_rank else 0.0 for c in self.cases) / len(self.cases)


# =====================================================================
# BLOQUE: Ejecución
# =====================================================================
def evaluate(retriever: HybridRetriever, modes=("lexical", "semantic", "hybrid"), top_k: int = TOP_K,
             cases: list[tuple[str, set[str]]] | None = None) -> dict[str, ModeSummary]:
    """Corre los casos en cada modo y devuelve un resumen por modo."""
    cases = RETRIEVAL_CASES if cases is None else cases
    summaries = {}
    for mode in modes:
        results = []
        for question, expected in cases:
            retrieved = retriever.retrieve(question, top_k=top_k, mode=mode)
            results.append(CaseResult(question, expected, [i.chunk.chunk_id for i in retrieved.items]))
        summaries[mode] = ModeSummary(mode, results)
    return summaries
