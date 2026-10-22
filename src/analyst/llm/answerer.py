"""
REDACCIÓN DE LA RESPUESTA

Qué hace: pide al modelo la respuesta estructurada (hechos, hipótesis, contradicciones,
información insuficiente) a partir de los cálculos y la evidencia ya aislada.
Con qué se conecta: lo llama service.py; el resultado pasa después por guardrails/validator.py.
Idea clave: el modelo redacta; las cifras vienen de pandas y se verifican después.
"""
from analyst.llm.ollama_client import Generated
from analyst.llm.planner import StructuredLLM
from analyst.llm.prompts import PromptBundle
from analyst.llm.schemas import Answer, Gap

# Tope de tokens de la respuesta (con un reintento sube x1.5 si se corta).
ANSWER_MAX_TOKENS = 900

OUT_OF_SCOPE_TEXT = ("Esa pregunta está fuera del alcance de este asistente, que analiza el desempeño de Brío "
                     "(ventas de enero a junio de 2026) con los datos y documentos entregados.")


def answer_question(llm: StructuredLLM, bundle: PromptBundle, max_tokens: int = ANSWER_MAX_TOKENS) -> Generated[Answer]:
    """Genera la respuesta estructurada con el prompt ya armado."""
    return llm.generate_structured(Answer, bundle.system, bundle.user, max_tokens)


def out_of_scope_answer() -> Answer:
    """Respuesta fija, sin llamar al modelo: no hay nada que calcular ni recuperar para esa pregunta."""
    return Answer(resumen=OUT_OF_SCOPE_TEXT, hechos=[], hipotesis=[], contradicciones=[],
                  insuficiente=[Gap(texto="La pregunta no se refiere a Brío ni a sus ventas.")])
