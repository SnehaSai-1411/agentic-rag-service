"""The agent loop: retrieve -> gate -> rerank -> answer / clarify / don't know.

Deliberately a plain function, not a framework: one pass through three
steps, with a structured trace recorded at each step. There is no multi-turn
planning here because the task doesn't need it — a single retrieval plus a
confidence gate covers "answer, clarify, or don't know" (see the README for
why this stays a simple loop instead of LangGraph).

The gate runs on the RETRIEVAL (bi-encoder cosine) scores, before reranking.
That is deliberate: the gate answers "is there anything relevant in the
knowledge base", and cosine similarities have a stable, comparable scale
for that threshold. The reranker answers a different question — "which of
the retrieved chunks is best" — and its scores (especially cross-encoder
logits, even after sigmoid normalization) compress at the top end, which
would make the gate's ambiguity gap trigger-happy. Reranking therefore
only orders the context for the answer; it never changes the decision.

Every response carries its trace: which tool was called, what scores came
back, and why the gate chose the decision it did. That trace is the
debugging story for the whole service.

The decision logic depends only on retrieval scores (agent/gate.py), so the
agent is fully testable with a fake retriever and no LLM.
"""
from collections.abc import Callable, Iterator

from config import Settings
from guardrails import apply_output_guardrails
from llm import LLM, ExtractiveFallback
from agent.gate import decide, is_vague_query
from agent.rerank import Reranker

# How many reranked chunks go into the LLM prompt / citations.
ANSWER_CONTEXT_K = 3

DONT_KNOW_MESSAGE = (
    "I don't know — I couldn't find anything in the knowledge base that "
    "closely matches your question, so I'd rather say so than guess. "
    "If you rephrase it, or ask about a topic covered in the docs, I can try again."
)

VAGUE_CLARIFY_MESSAGE = (
    "Could you say a bit more? A few more words about what you'd like to "
    "know would help me find the right passage — for example, are you asking "
    "for a definition, a how-to, or a comparison?"
)


def _citation(hit: dict) -> dict:
    return {
        "source": hit.get("source"),
        "page": hit.get("page"),
        "chunk_index": hit.get("chunk_index"),
        "score": round(hit.get("score", 0.0), 3),
    }


class Agent:
    """A tool-calling agent with one tool: search_knowledge_base.

    `search_fn(question, k)` returns retrieval hits as dicts with at least
    `text`, `source`, `page`, `score` (cosine similarity, higher is better),
    and optionally `embedding` (used by cosine reranking).
    `embed_fn(question)` optionally embeds the query for reranking.
    """

    def __init__(
        self,
        search_fn: Callable[[str, int], list[dict]],
        llm: LLM | None = None,
        reranker: Reranker | None = None,
        settings: Settings | None = None,
        embed_fn: Callable[[str], list[float]] | None = None,
    ):
        self.search_fn = search_fn
        self.llm = llm or ExtractiveFallback()
        self.reranker = reranker or Reranker(mode="cosine")
        self.settings = settings or Settings()
        self.embed_fn = embed_fn

    # -- shared retrieval + gating -------------------------------------

    def _retrieve_and_decide(self, question: str) -> tuple[dict, list[dict], list[dict]]:
        """Run retrieval and the gate. Returns (decision dict, retrieval
        hits in retrieval order, trace steps so far). Reranking happens
        later, only on the answer path (see module docstring)."""
        trace: list[dict] = []

        hits = self.search_fn(question, self.settings.top_k)
        trace.append({
            "step": "retrieve",
            "tool": "search_knowledge_base",
            "input": question,
            "scores": [round(h.get("score", 0.0), 3) for h in hits],
            "detail": f"Retrieved {len(hits)} chunk(s) from collection "
                      f"'{self.settings.collection}'.",
        })

        decision = decide(
            question,
            hits,
            min_top_score=self.settings.min_top_score,
            ambiguity_gap=self.settings.ambiguity_gap,
        )
        trace.append({
            "step": "gate",
            "tool": None,
            "input": None,
            "scores": None,
            "detail": f"Decision: {decision.action}. {decision.reason}",
        })
        return {"action": decision.action, "reason": decision.reason}, hits, trace

    def _rerank(self, question: str, hits: list[dict], trace: list[dict]) -> list[dict]:
        """Re-score hits before answering; appends the rerank trace step."""
        query_embedding = self.embed_fn(question) if self.embed_fn else None
        hits, rerank_mode = self.reranker.rerank(
            question, hits, query_embedding=query_embedding
        )
        trace.append({
            "step": "rerank",
            "tool": None,
            "input": None,
            "scores": [round(h.get("score", 0.0), 3) for h in hits],
            "detail": f"Reranked with mode '{rerank_mode}'.",
        })
        return hits

    def _clarify_message(self, question: str, hits: list[dict]) -> str:
        if is_vague_query(question):
            return VAGUE_CLARIFY_MESSAGE
        topics = []
        for hit in hits[:3]:
            if hit.get("source") not in topics:
                topics.append(hit.get("source"))
        if not topics:
            return VAGUE_CLARIFY_MESSAGE
        topic_list = ", ".join(topics)
        return (
            f"Your question could mean a few different things — the closest "
            f"matches come from {topic_list}. Could you narrow it down a "
            f"little? For example, which of those topics are you asking about, "
            f"and are you after a definition, a how-to, or a comparison?"
        )

    # -- non-streaming --------------------------------------------------

    def ask(self, question: str) -> dict:
        decision, hits, trace = self._retrieve_and_decide(question)
        action = decision["action"]

        if action == "clarify":
            return {
                "decision": "clarify",
                "answer": self._clarify_message(question, hits),
                "citations": [],
                "trace": trace,
                "grounding": None,
            }
        if action == "dont_know":
            return {
                "decision": "dont_know",
                "answer": DONT_KNOW_MESSAGE,
                "citations": [],
                "trace": trace,
                "grounding": None,
            }

        hits = self._rerank(question, hits, trace)
        context_hits = hits[:ANSWER_CONTEXT_K]
        context = [h["text"] for h in context_hits]
        answer = self.llm.generate(question, context)
        trace.append({
            "step": "generate",
            "tool": None,
            "input": None,
            "scores": None,
            "detail": f"Answer generated with LLM '{self.llm.name}' "
                      f"from {len(context_hits)} context chunk(s).",
        })

        grounding = apply_output_guardrails(answer, context)
        trace.append({
            "step": "grounding_check",
            "tool": None,
            "input": None,
            "scores": None,
            "detail": f"Grounding check: supported={grounding.supported}, "
                      f"coverage={grounding.coverage:.2f}.",
        })
        if not grounding.supported:
            answer += (
                "\n\nNote: part of this answer could not be verified against "
                "the retrieved context, so treat it with caution."
            )

        return {
            "decision": "answer",
            "answer": answer,
            "citations": [_citation(h) for h in context_hits],
            "trace": trace,
            "grounding": grounding.to_dict(),
        }

    # -- streaming ------------------------------------------------------

    def ask_stream(self, question: str) -> Iterator[dict]:
        """Yield events: trace steps, then answer tokens, then a final result.

        Event shapes:
        - {"type": "trace", "step": {...}}
        - {"type": "token", "text": "..."}
        - {"type": "result", "decision": ..., "citations": [...],
           "grounding": {...} | None}
        """
        decision, hits, trace = self._retrieve_and_decide(question)
        for step in trace:
            yield {"type": "trace", "step": step}

        action = decision["action"]
        if action == "clarify":
            yield {"type": "token", "text": self._clarify_message(question, hits)}
            yield {"type": "result", "decision": "clarify",
                   "citations": [], "grounding": None}
            return
        if action == "dont_know":
            yield {"type": "token", "text": DONT_KNOW_MESSAGE}
            yield {"type": "result", "decision": "dont_know",
                   "citations": [], "grounding": None}
            return

        hits = self._rerank(question, hits, trace)
        yield {"type": "trace", "step": trace[-1]}
        context_hits = hits[:ANSWER_CONTEXT_K]
        context = [h["text"] for h in context_hits]
        parts: list[str] = []
        for token in self.llm.generate_stream(question, context):
            parts.append(token)
            yield {"type": "token", "text": token}

        full_answer = "".join(parts)
        grounding = apply_output_guardrails(full_answer, context)
        if not grounding.supported:
            yield {"type": "token", "text": (
                "\n\nNote: part of this answer could not be verified against "
                "the retrieved context, so treat it with caution."
            )}
        yield {
            "type": "result",
            "decision": "answer",
            "citations": [_citation(h) for h in context_hits],
            "grounding": grounding.to_dict(),
        }
