"""LLM layer: Claude or OpenAI via API key, with an honest extractive fallback.

The agent's DECISION logic (answer / clarify / don't know) never depends on
the LLM — it runs on retrieval scores alone (see agent/gate.py). The LLM is
only used to turn retrieved context into a fluent answer, so the whole agent
is testable with no API key and no network.

Without a key, answers come from the extractive fallback: the most relevant
retrieved chunk, quoted verbatim and clearly labeled as extractive. It never
pretends to be a generated answer.
"""
from collections.abc import Iterator
from typing import Protocol

from config import Settings

SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided context. "
    "If the context does not contain the answer, say so plainly. "
    "Do not add facts that are not in the context."
)


class LLM(Protocol):
    """Anything that can turn a question + retrieved context into an answer."""

    name: str

    def generate(self, question: str, context: list[str]) -> str: ...

    def generate_stream(self, question: str, context: list[str]) -> Iterator[str]: ...


def build_prompt(question: str, context: list[str]) -> str:
    joined = "\n\n---\n\n".join(context)
    return f"Context:\n{joined}\n\nQuestion: {question}\n\nAnswer:"


class ExtractiveFallback:
    """No-API-key fallback: quote the top retrieved chunk, labeled as such."""

    name = "extractive-fallback"

    def generate(self, question: str, context: list[str]) -> str:
        if not context:
            return "I don't have any retrieved context to answer from."
        return (
            "[Extractive answer — no LLM key configured, quoting the most "
            "relevant retrieved passage:]\n\n" + context[0]
        )

    def generate_stream(self, question: str, context: list[str]) -> Iterator[str]:
        # Yield word-by-word so the streaming endpoint behaves the same
        # with or without a real LLM behind it.
        answer = self.generate(question, context)
        for word in answer.split(" "):
            yield word + " "


class AnthropicLLM:
    """Claude via the Anthropic API. Imported lazily so tests never need it."""

    def __init__(self, api_key: str, model: str):
        import anthropic  # lazy: keeps `import llm` cheap and key-free

        self.name = f"anthropic:{model}"
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def generate(self, question: str, context: list[str]) -> str:
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_prompt(question, context)}],
        )
        return "".join(block.text for block in msg.content if block.type == "text")

    def generate_stream(self, question: str, context: list[str]) -> Iterator[str]:
        with self._client.messages.stream(
            model=self._model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_prompt(question, context)}],
        ) as stream:
            for text in stream.text_stream:
                yield text


class OpenAILLM:
    """OpenAI chat completions. Imported lazily so tests never need it."""

    def __init__(self, api_key: str, model: str):
        import openai  # lazy: keeps `import llm` cheap and key-free

        self.name = f"openai:{model}"
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model

    def generate(self, question: str, context: list[str]) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, context)},
            ],
        )
        return resp.choices[0].message.content or ""

    def generate_stream(self, question: str, context: list[str]) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self._model,
            stream=True,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, context)},
            ],
        )
        for chunk in stream:
            text = chunk.choices[0].delta.content
            if text:
                yield text


def build_llm(settings: Settings) -> LLM:
    """Pick the LLM from settings; fall back to extractive when no key exists."""
    if settings.llm_provider == "anthropic" and settings.anthropic_api_key:
        return AnthropicLLM(settings.anthropic_api_key, settings.anthropic_model)
    if settings.llm_provider == "openai" and settings.openai_api_key:
        return OpenAILLM(settings.openai_api_key, settings.openai_model)
    return ExtractiveFallback()
