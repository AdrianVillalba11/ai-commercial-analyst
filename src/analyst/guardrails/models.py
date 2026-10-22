"""
MODELOS DE LA RESPUESTA FINAL

Qué hace: define las estructuras de lo que el analista realmente ve: afirmaciones ya verificadas,
lo que se descartó y por qué.
Con qué se conecta: las llenan facts.py, report_check.py y validator.py; las muestran la API y la interfaz.
Idea clave: cada afirmación lleva su ORIGEN (calculada por código, contraste, documento o modelo),
para que se vea qué escribió el código y qué escribió el modelo.
"""
from dataclasses import asdict, dataclass, field

from analyst.llm.schemas import Answer

# =====================================================================
# BLOQUE: Origen de cada afirmación
# =====================================================================
ORIGIN_CALC = "calculo"          # escrito por código a partir de los resultados de pandas
ORIGIN_CONTRAST = "contraste"    # escrito por código a partir de la comparación reporte-datos
ORIGIN_DOCUMENT = "documento"    # escrito por el modelo, cita documentos y pasó las verificaciones
ORIGIN_MODEL = "modelo"          # escrito por el modelo (hipótesis, contradicciones) y pasó las verificaciones


# =====================================================================
# BLOQUE: Estructuras
# =====================================================================
@dataclass
class FinalClaim:
    """Una afirmación que se muestra: texto, fuentes que la respaldan y quién la escribió."""

    texto: str
    fuentes: list[str] = field(default_factory=list)
    origen: str = ORIGIN_MODEL


@dataclass
class Discard:
    """Una afirmación del modelo que se descartó, con la sección de origen y el motivo (no se oculta)."""

    seccion: str
    texto: str
    motivo: str


@dataclass
class FinalAnswer:
    """Respuesta final: resumen, las cuatro secciones, lo descartado y los avisos."""

    resumen: str
    resumen_origen: str = ORIGIN_MODEL
    hechos: list[FinalClaim] = field(default_factory=list)
    hipotesis: list[FinalClaim] = field(default_factory=list)
    contradicciones: list[FinalClaim] = field(default_factory=list)
    insuficiente: list[FinalClaim] = field(default_factory=list)
    descartes: list[Discard] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def unchecked(cls, answer: Answer) -> "FinalAnswer":
        """Para respuestas sin afirmaciones del modelo (por ejemplo, fuera de alcance): no hay nada que validar."""
        return cls(resumen=answer.resumen, resumen_origen=ORIGIN_CALC,
                   insuficiente=[FinalClaim(g.texto, [], ORIGIN_CALC) for g in answer.insuficiente])
