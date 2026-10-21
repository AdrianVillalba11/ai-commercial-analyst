"""Writes the structured answer from the computed results and the isolated evidence."""
from analyst.llm.ollama_client import Generated
from analyst.llm.planner import StructuredLLM
from analyst.llm.prompts import PromptBundle
from analyst.llm.schemas import Answer, Gap

ANSWER_MAX_TOKENS = 900

OUT_OF_SCOPE_TEXT = ("Esa pregunta está fuera del alcance de este asistente, que analiza el desempeño de Brío "
                     "(ventas de enero a junio de 2026) con los datos y documentos entregados.")


def answer_question(llm: StructuredLLM, bundle: PromptBundle, max_tokens: int = ANSWER_MAX_TOKENS) -> Generated[Answer]:
    return llm.generate_structured(Answer, bundle.system, bundle.user, max_tokens)


def out_of_scope_answer() -> Answer:
    """Fixed answer, no model call: there is nothing to compute or retrieve for such a question."""
    return Answer(resumen=OUT_OF_SCOPE_TEXT, hechos=[], hipotesis=[], contradicciones=[],
                  insuficiente=[Gap(texto="La pregunta no se refiere a Brío ni a sus ventas.")])
