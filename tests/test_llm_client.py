"""
PRUEBAS DEL CLIENTE DEL MODELO Y DE LOS ESQUEMAS

Qué verifica: llm/ollama_client.py (petición, reintentos, errores claros) y llm/schemas.py (esquemas cerrados y acotados).
Cómo: no necesita Ollama; un transporte HTTP simulado (MockTransport) entrega las respuestas en orden y registra las peticiones.
"""
import json

import httpx
import pytest

from analyst.errors import LLMOutputError, OllamaError
from analyst.llm.ollama_client import OllamaChat
from analyst.llm.schemas import Answer, Plan, json_schema

VALID_PLAN = {"tipo": "cuantitativa", "usar_documentos": False, "analisis": [
    {"agrupar_por": ["region"], "metrica": "venta_neta_cop", "region": "todas", "canal": "todos", "presentacion": "todas"}]}


def reply(content, done_reason="stop", status=200):
    body = content if isinstance(content, str) else json.dumps(content)
    return httpx.Response(status, json={"message": {"content": body}, "done_reason": done_reason,
                                        "prompt_eval_count": 100, "eval_count": 50})


def client_with(*responses):
    """Un OllamaChat cuya capa HTTP entrega las respuestas dadas en orden y registra las peticiones."""
    sent, queue = [], list(responses)

    def handler(request):
        sent.append(json.loads(request.content))
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    chat = OllamaChat(model="m", base_url="http://x", num_ctx=4096, keep_alive="30m",
                      client=httpx.Client(transport=httpx.MockTransport(handler)))
    return chat, sent


# =====================================================================
# BLOQUE: Esquemas
# =====================================================================

def test_schema_has_no_refs_and_everything_is_bounded():
    schema = json_schema(Plan)
    text = json.dumps(schema)
    assert "$ref" not in text and "$defs" not in text and '"title"' not in text
    assert schema["additionalProperties"] is False
    assert schema["properties"]["analisis"]["maxItems"] == 2
    spec = schema["properties"]["analisis"]["items"]
    assert spec["properties"]["agrupar_por"]["maxItems"] == 2
    assert "todas" in spec["properties"]["region"]["enum"] and "Sur" in spec["properties"]["region"]["enum"]
    assert set(schema["properties"]["tipo"]["enum"]) == {
        "cuantitativa", "integrada", "contraste_reporte", "causalidad", "fuera_de_alcance"}


def test_answer_schema_requires_sources_on_every_claim():
    schema = json_schema(Answer)
    for field in ("hechos", "hipotesis", "contradicciones"):
        assert schema["properties"][field]["items"]["properties"]["fuentes"]["minItems"] == 1
    assert schema["properties"]["hechos"]["maxItems"] == 5


def test_schema_models_reject_unknown_values_and_extra_fields():
    with pytest.raises(ValueError):
        Plan.model_validate({**VALID_PLAN, "tipo": "inventado"})
    with pytest.raises(ValueError):
        Plan.model_validate({**VALID_PLAN, "extra": 1})
    with pytest.raises(ValueError):
        Answer.model_validate({"resumen": "x", "hechos": [{"texto": "t", "fuentes": []}],
                               "hipotesis": [], "contradicciones": [], "insuficiente": []})


# =====================================================================
# BLOQUE: Peticiones
# =====================================================================

def test_request_carries_schema_and_deterministic_options():
    chat, sent = client_with(reply(VALID_PLAN))
    generated = chat.generate_structured(Plan, "sistema", "usuario", max_tokens=300)
    request = sent[0]
    assert request["model"] == "m" and request["stream"] is False and request["keep_alive"] == "30m"
    assert request["options"] == {"temperature": 0.0, "seed": 7, "num_ctx": 4096, "num_predict": 300}
    assert request["format"] == json_schema(Plan)
    assert [m["role"] for m in request["messages"]] == ["system", "user"]
    assert generated.value.tipo == "cuantitativa" and generated.info.retries == 0
    assert (generated.info.prompt_tokens, generated.info.output_tokens) == (100, 50)


# =====================================================================
# BLOQUE: Reintentos
# =====================================================================

def test_invalid_json_is_retried_once_with_a_reminder():
    chat, sent = client_with(reply("{no es json"), reply(VALID_PLAN))
    generated = chat.generate_structured(Plan, "s", "pregunta", max_tokens=300)
    assert generated.info.retries == 1 and len(sent) == 2
    assert "no cumplió el formato" in sent[1]["messages"][1]["content"]
    assert sent[1]["messages"][1]["content"].startswith("pregunta")
    assert generated.info.notes and "Intento 1 descartado" in generated.info.notes[0]
    assert generated.info.prompt_tokens == 200      # se cuentan los dos intentos


def test_valid_json_with_a_value_outside_the_schema_is_retried():
    chat, sent = client_with(reply({**VALID_PLAN, "tipo": "inventado"}), reply(VALID_PLAN))
    assert chat.generate_structured(Plan, "s", "u", 300).info.retries == 1


def test_output_cut_by_the_token_limit_is_retried_with_a_bigger_budget():
    chat, sent = client_with(reply('{"tipo": "cuant', done_reason="length"), reply(VALID_PLAN))
    generated = chat.generate_structured(Plan, "s", "u", max_tokens=300)
    assert [r["options"]["num_predict"] for r in sent] == [300, 450]
    assert "límite de 300 tokens" in generated.info.notes[0]


def test_two_bad_answers_raise_a_clear_error():
    chat, sent = client_with(reply("basura"), reply("más basura"))
    with pytest.raises(LLMOutputError) as error:
        chat.generate_structured(Plan, "s", "u", 300)
    assert len(sent) == 2 and "LLM_MODEL" in str(error.value)
    assert isinstance(error.value, OllamaError)


def test_two_cut_answers_raise_a_clear_error():
    chat, _ = client_with(reply("{", done_reason="length"), reply("{", done_reason="length"))
    with pytest.raises(LLMOutputError, match="se cortó"):
        chat.generate_structured(Plan, "s", "u", 300)


# =====================================================================
# BLOQUE: Infraestructura y errores
# =====================================================================

@pytest.mark.parametrize("failure,expected", [
    (httpx.ConnectError("refused"), "ollama serve"),
    (httpx.ReadTimeout("slow"), "no respondió"),
    (httpx.Response(404, json={"error": "model not found"}), "ollama pull m"),
    (httpx.Response(500, text="boom"), "500"),
    (httpx.Response(200, text="esto no es json"), "no es JSON"),
    (httpx.Response(200, json={"otra": 1}), "'message'"),
])
def test_infrastructure_failures_are_not_retried_and_say_what_to_do(failure, expected):
    chat, sent = client_with(failure)
    with pytest.raises(OllamaError) as error:
        chat.generate_structured(Plan, "s", "u", 300)
    assert expected in str(error.value) and len(sent) == 1
    assert not isinstance(error.value, LLMOutputError)


def test_the_http_client_is_reused():
    assert isinstance(OllamaChat(model="m", base_url="http://x")._client, httpx.Client)
