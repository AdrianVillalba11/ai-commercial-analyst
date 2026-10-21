"""Structures the model must produce. They double as the JSON schema Ollama enforces.

Everything is bounded on purpose (closed enums, max list sizes): a schema guarantees the shape
of the output, not good behaviour, and an unbounded list can send a small model into a loop.
"""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

PlanType = Literal["cuantitativa", "integrada", "contraste_reporte", "causalidad", "fuera_de_alcance"]
Dimension = Literal["region", "canal", "presentacion"]
Metric = Literal["venta_neta_cop", "unidades"]
RegionFilter = Literal["todas", "Norte", "Centro", "Sur", "Oriente", "Occidente"]
ChannelFilter = Literal["todos", "Tradicional", "Moderno"]
PresentationFilter = Literal["todas", "Individual", "Familiar"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnalysisSpec(_Strict):
    agrupar_por: list[Dimension] = Field(max_length=2)
    metrica: Metric
    region: RegionFilter
    canal: ChannelFilter
    presentacion: PresentationFilter


class Plan(_Strict):
    tipo: PlanType
    analisis: list[AnalysisSpec] = Field(max_length=2)
    usar_documentos: bool


class Claim(_Strict):
    texto: str
    fuentes: list[str] = Field(min_length=1, max_length=4)


class Gap(_Strict):
    texto: str


class Answer(_Strict):
    resumen: str
    hechos: list[Claim] = Field(max_length=5)
    hipotesis: list[Claim] = Field(max_length=4)
    contradicciones: list[Claim] = Field(max_length=3)
    insuficiente: list[Gap] = Field(max_length=3)


def json_schema(model: type[BaseModel]) -> dict:
    """The model's JSON schema with $ref inlined and cosmetic keys removed (smaller and simpler for Ollama)."""
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
