"""Confidence gate: pure, unit-testable decision logic.

The agent decides ANSWER / CLARIFY / DON'T KNOW from retrieval scores and
the shape of the query alone — no LLM involved. That keeps the decision
deterministic, cheap, and testable without an API key.

Rules, in order:
1. Vague query (one word, or a known vague phrase like "tell me more")
   -> CLARIFY. There is nothing solid to retrieve on.
2. Best retrieved score below `min_top_score` -> DON'T KNOW. The knowledge
   base has nothing close enough; answering anyway would be guessing.
3. Top hits span different sources AND the top two scores are within
   `ambiguity_gap` of each other -> CLARIFY. The query plausibly means
   two different things and the retriever cannot tell them apart.
4. Otherwise -> ANSWER.

Scores are cosine similarities in [0, 1]; higher is better. (Chroma returns
cosine *distances*; the retriever converts them with `1 - distance`.)
"""
from dataclasses import dataclass

# Queries that carry no retrievable content on their own.
VAGUE_PHRASES = {
    "tell me more",
    "what about it",
    "what about that",
    "explain",
    "explain more",
    "more",
    "and then",
    "go on",
    "huh",
    "what",
    "why",
    "how",
}


@dataclass
class Decision:
    action: str  # "answer" | "clarify" | "dont_know"
    reason: str


def is_vague_query(query: str) -> bool:
    """True for single-word queries and known vague phrases."""
    normalized = query.strip().lower().rstrip("?.!").strip()
    if not normalized:
        return True
    if normalized in VAGUE_PHRASES:
        return True
    return len(normalized.split()) < 2


def decide(
    query: str,
    hits: list[dict],
    min_top_score: float,
    ambiguity_gap: float,
) -> Decision:
    """Decide what the agent should do. `hits` are dicts with `score` and
    `source`, sorted best-first. Pure function — no I/O, no LLM."""
    if is_vague_query(query):
        return Decision(
            action="clarify",
            reason="Query is too vague to retrieve on; asking for clarification.",
        )
    if not hits:
        return Decision(
            action="dont_know",
            reason="No chunks retrieved from the knowledge base.",
        )
    top_score = hits[0]["score"]
    if top_score < min_top_score:
        return Decision(
            action="dont_know",
            reason=f"Top retrieval score {top_score:.3f} is below the "
            f"confidence threshold {min_top_score:.3f}.",
        )
    if len(hits) >= 2:
        top_source = hits[0]["source"]
        second = hits[1]
        gap = top_score - second["score"]
        if second["source"] != top_source and gap < ambiguity_gap:
            return Decision(
                action="clarify",
                reason=f"Top hits span unrelated sources ({top_source} vs "
                f"{second['source']}) with a score gap of {gap:.3f}, below "
                f"the ambiguity gap {ambiguity_gap:.3f}.",
            )
    return Decision(
        action="answer",
        reason=f"Top score {top_score:.3f} clears the threshold "
        f"{min_top_score:.3f}; answering from the knowledge base.",
    )
