"""Shared test doubles: a fake embedder, a fake structured LLM and a chunk factory."""
import zlib

import numpy as np

from analyst.docs.chunker import Chunk
from analyst.errors import OllamaError
from analyst.llm.ollama_client import Generated, GenerationInfo
from analyst.llm.schemas import Answer, Plan
from analyst.rag.lexical import tokenize


def make_chunk(chunk_id, text, regions=None, kind="texto") -> Chunk:
    return Chunk(chunk_id=chunk_id, doc="d.md", doc_title="Doc", doc_type="t", section="S", kind=kind,
                 text=text, embed_text="Doc | S\n" + text, regions=regions or [])


class FakeEmbedder:
    """Deterministic bag-of-words vectors; counts calls so caching can be tested."""
    def __init__(self, name="fake-model", fail=False):
        self._name, self.fail = name, fail
        self.document_calls = self.query_calls = 0

    @property
    def name(self):
        return self._name

    def _vector(self, text):
        v = np.zeros(128)
        for token in tokenize(text):
            v[zlib.crc32(token.encode()) % 128] += 1
        return v.tolist()

    def embed_documents(self, texts):
        if self.fail:
            raise OllamaError("Ollama apagado (simulado)")
        self.document_calls += 1
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        if self.fail:
            raise OllamaError("Ollama apagado (simulado)")
        self.query_calls += 1
        return self._vector(text)


class FakeLLM:
    """Plays a fixed plan and answer; records every prompt it receives."""
    name = "fake-llm"

    def __init__(self, plan=None, answer=None, plan_error=None, answer_error=None, plan_notes=None,
                 answer_notes=None, num_ctx=4096):
        self.plan, self.answer = plan, answer
        self.plan_error, self.answer_error = plan_error, answer_error
        self.plan_notes, self.answer_notes = plan_notes or [], answer_notes or []
        self.num_ctx = num_ctx
        self.calls = []

    def generate_structured(self, model_cls, system, user, max_tokens):
        self.calls.append({"model": model_cls, "system": system, "user": user, "max_tokens": max_tokens})
        if model_cls is Plan:
            if self.plan_error:
                raise self.plan_error
            return Generated(self.plan, GenerationInfo(notes=self.plan_notes))
        if self.answer_error:
            raise self.answer_error
        return Generated(self.answer or SIMPLE_ANSWER, GenerationInfo(notes=self.answer_notes, prompt_tokens=10, output_tokens=5))


SIMPLE_ANSWER = Answer.model_validate({
    "resumen": "Resumen de prueba.", "hechos": [{"texto": "Hecho de prueba.", "fuentes": ["CALC-1"]}],
    "hipotesis": [], "contradicciones": [], "insuficiente": []})
