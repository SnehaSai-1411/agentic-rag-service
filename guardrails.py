"""Output guardrail hooks.

`grounding_check` is a small, dependency-free check: does each sentence of
the answer actually overlap with the retrieved context? It flags unsupported
sentences instead of silently passing them through. It is a heuristic, not a
guarantee — a sentence can share words with the context and still be wrong,
and a correct paraphrase can score low. Treat it as a tripwire, not a proof.

Integration point for the sibling repo `llm-guardrails-pii` (Project 4):
that repo's middleware (PII redaction, prompt-injection checks, a stronger
grounding check) is designed to wrap this service's endpoints. Wire it in
`app/main.py` as FastAPI middleware / a dependency around the `/ask` handler
(see the README snippet) — this repo intentionally does not import it, so
each repo stays independently runnable.
"""
import re
from dataclasses import dataclass, field

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "for", "with", "by", "at", "from", "as", "and",
    "or", "but", "if", "then", "it", "its", "this", "that", "these", "those",
    "you", "your", "we", "our", "they", "their", "he", "she", "his", "her",
    "i", "me", "my", "do", "does", "did", "can", "could", "will", "would",
    "should", "so", "such", "than", "too", "very", "not", "no", "yes",
}


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def split_sentences(text: str) -> list[str]:
    """Naive sentence split; good enough for short generated answers."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


@dataclass
class GroundingResult:
    supported: bool
    coverage: float  # fraction of answer content words found in the context
    unsupported_sentences: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "supported": self.supported,
            "coverage": round(self.coverage, 3),
            "unsupported_sentences": self.unsupported_sentences,
        }


def grounding_check(
    answer: str,
    context: list[str],
    min_sentence_coverage: float = 0.5,
) -> GroundingResult:
    """Check each answer sentence against the retrieved context.

    A sentence is supported when at least `min_sentence_coverage` of its
    content words appear somewhere in the context. Returns the overall
    coverage plus the list of unsupported sentences.
    """
    context_words: set[str] = set()
    for chunk in context:
        context_words |= _content_words(chunk)

    sentences = split_sentences(answer)
    unsupported: list[str] = []
    all_answer_words: set[str] = set()
    covered_words: set[str] = set()

    for sentence in sentences:
        if sentence.startswith("["):
            continue  # bracketed label line (e.g. the extractive fallback
            # header) — meta-text, not a claim to verify
        words = _content_words(sentence)
        if not words:
            continue  # no checkable content (e.g. a bare label line)
        all_answer_words |= words
        covered = words & context_words
        covered_words |= covered
        if len(covered) / len(words) < min_sentence_coverage:
            unsupported.append(sentence)

    coverage = len(covered_words) / len(all_answer_words) if all_answer_words else 1.0
    return GroundingResult(
        supported=not unsupported,
        coverage=coverage,
        unsupported_sentences=unsupported,
    )


def apply_output_guardrails(answer: str, context: list[str]) -> GroundingResult:
    """The single hook the agent calls before returning an answer.

    Today this runs the local grounding check. To add the Project 4
    middleware (PII redaction, injection checks), wrap or extend this
    function — the agent only ever calls this one entry point.
    """
    return grounding_check(answer, context)
