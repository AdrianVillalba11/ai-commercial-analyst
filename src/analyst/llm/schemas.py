"""
ESTRUCTURAS QUE DEBE PRODUCIR EL MODELO

Qué hace: define, con Pydantic, la forma exacta del plan y de la respuesta. Esas mismas clases se
convierten en el esquema JSON que Ollama obliga a cumplir al generar.
Con qué se conecta: lo usan el cliente (ollama_client.py), el planificador, el redactor y el validador.
Idea clave: todo está acotado a propósito (opciones cerradas, listas con máximo): un esquema garantiza la
FORMA de la salida, no el buen comportamiento, y una lista sin tope puede meter a un modelo pequeño en un bucle.
"""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# =====================================================================
# BLOQUE: Opciones cerradas
# Lo único que el modelo puede elegir al planificar.
# =====================================================================
PlanType = Literal["cuantitativa", "integrada", "contraste_reporte", "causalidad", "fuera_de_alcance"]
Dimension = Literal["region", "canal", "presentacion"]
Metric = Literal["venta_neta_cop", "unidades"]
RegionFilter = Literal["todas", "Norte", "Centro", "Sur", "Oriente", "Occidente"]
ChannelFilter = Literal["todos", "Tradicional", "Moderno"]
PresentationFilter = Literal["todas", "Individual", "Familiar"]


# =====================================================================
# BLOQUE: Plan de análisis
# =====================================================================
class _Strict(BaseModel):
    """Base: rechaza campos que no estén en el esquema."""

    model_config = ConfigDict(extra="forbid")


class AnalysisSpec(_Strict):
    """Un cálculo pedido: cómo agrupar, qué métrica y qué filtros."""

    agrupar_por: list[Dimension] = Field(max_length=2)
    metrica: Metric
    region: RegionFilter
    canal: ChannelFilter
    presentacion: PresentationFilter


class Plan(_Strict):
    """Plan de la pregunta: tipo, cálculos a hacer y si se necesitan documentos."""

    tipo: PlanType
    analisis: list[AnalysisSpec] = Field(max_length=2)
    usar_documentos: bool


# =====================================================================
# BLOQUE: Respuesta estructurada
# =====================================================================
class Claim(_Strict):
    """Una afirmación con las fuentes (ids) que la respaldan."""

    texto: str
    fuentes: list[str] = Field(min_length=1, max_length=4)


class Gap(_Strict):
    """Algo que no se puede afirmar con la información disponible."""

    texto: str


class Answer(_Strict):
    """Respuesta separada en hechos, hipótesis, contradicciones e información insuficiente."""

    resumen: str
    hechos: list[Claim] = Field(max_length=5)
    hipotesis: list[Claim] = Field(max_length=4)
    contradicciones: list[Claim] = Field(max_length=3)
    insuficiente: list[Gap] = Field(max_length=3)


# =====================================================================
# BLOQUE: Esquema JSON para Ollama
# =====================================================================
def json_schema(model: type[BaseModel]) -> dict:
    """Esquema JSON del modelo con los $ref en línea y sin claves cosméticas (más pequeño y simple para Ollama)."""
    raw = model.model_json_schema()
    definitions = raw.pop("$defs", {})

    def clean(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return clean(definitions[node["$ref"].rsplit("/", 1)[-1]])
            return {k: clean(v) for k, v in node.items() if k not in {"title", "default"}}
        if isinstance(node, list):
            return [clean(v) for v in node]
        return node

    return clean(raw)
