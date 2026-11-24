"""Tests for the agent loop — fake retriever + fake LLM, no model, no key."""
from agent.core import Agent
from config import Settings
from tests.fakes import FakeLLM, fake_search, make_hits

CHUNK_TEXT = (
    "Chunking splits documents into pieces for retrieval. "
    "Small chunks are precise but can cut an argument in half."
)
EMBEDDING_TEXT = (
    "An embedding is a dense vector that represents the meaning of text. "
    "Similar meanings get vectors that point in similar directions."
)


def build_agent(hits, llm_answer="Chunking splits documents into pieces."):
    return Agent(
        search_fn=fake_search(hits),
        llm=FakeLLM(llm_answer),
        settings=Settings(min_top_score=0.35, ambiguity_gap=0.05),
    )


def test_agent_answers_on_high_confidence_retrieval():
    hits = make_hits([
        (CHUNK_TEXT, "chunking.md", 0.90),
        (CHUNK_TEXT, "chunking.md", 0.85),
    ])
    agent = build_agent(hits)
    resp = agent.ask("What is chunking?")

    assert resp["decision"] == "answer"
    assert "Chunking splits documents" in resp["answer"]
    assert resp["citations"], "answers carry citations"
    assert resp["citations"][0]["source"] == "chunking.md"
    assert resp["grounding"]["supported"] is True
    steps = [s["step"] for s in resp["trace"]]
    assert steps == ["retrieve", "gate", "rerank", "generate", "grounding_check"]
    assert resp["trace"][0]["tool"] == "search_knowledge_base"
    assert resp["trace"][0]["scores"] == [0.90, 0.85]


def test_agent_says_it_does_not_know_on_low_confidence():
    hits = make_hits([
        (CHUNK_TEXT, "chunking.md", 0.20),
        (EMBEDDING_TEXT, "embeddings.md", 0.15),
    ])
    agent = build_agent(hits)
    resp = agent.ask("What is the capital of France?")

    assert resp["decision"] == "dont_know"
    assert "don't know" in resp["answer"].lower()
    assert resp["citations"] == []
    assert resp["grounding"] is None
    # The LLM must not be called for a don't-know: nothing to generate from.
    assert agent.llm.calls == []


def test_agent_clarifies_on_vague_input():
    hits = make_hits([(CHUNK_TEXT, "chunking.md", 0.95)])
    agent = build_agent(hits)
    resp = agent.ask("chunking")

    assert resp["decision"] == "clarify"
    assert "say a bit more" in resp["answer"]
    assert agent.llm.calls == []


def test_agent_clarifies_when_top_hits_span_unrelated_topics():
    hits = make_hits([
        (CHUNK_TEXT, "chunking.md", 0.90),
        (EMBEDDING_TEXT, "embeddings.md", 0.88),
    ])
    agent = build_agent(hits)
    resp = agent.ask("How do chunks and embeddings relate?")

    assert resp["decision"] == "clarify"
    assert "chunking.md" in resp["answer"]
    assert "embeddings.md" in resp["answer"]


def test_agent_flags_unsupported_answer_via_grounding_hook():
    hits = make_hits([(CHUNK_TEXT, "chunking.md", 0.90)])
    agent = build_agent(
        hits,
        llm_answer="The moon is made of cheese and quantum flux.",
    )
    resp = agent.ask("What is chunking?")

    assert resp["decision"] == "answer"
    assert resp["grounding"]["supported"] is False
    assert resp["grounding"]["unsupported_sentences"]
    assert "could not be verified" in resp["answer"]


def test_gate_uses_retrieval_scores_not_reranked_scores():
    """The decision follows retrieval order even if reranking reverses it:
    reranking orders the answer context, it never changes the decision."""
    from agent.rerank import Reranker

    class ReversingReranker(Reranker):
        def rerank(self, query, hits, query_embedding=None, top_n=None):
            return list(reversed(hits)), "reversed-for-test"

    hits = make_hits([
        (CHUNK_TEXT, "chunking.md", 0.90),
        (EMBEDDING_TEXT, "embeddings.md", 0.60),
    ])
    agent = Agent(
        search_fn=fake_search(hits),
        llm=FakeLLM("Chunking splits documents into pieces."),
        reranker=ReversingReranker(),
        settings=Settings(min_top_score=0.35, ambiguity_gap=0.05),
    )
    resp = agent.ask("What is chunking?")
    # Retrieval scores say answer (0.90 clears the threshold, gap is clear);
    # the reversed rerank order must not flip that to clarify/don't-know.
    assert resp["decision"] == "answer"
    # ...but the reranked order does decide which chunk is cited first.
    assert resp["citations"][0]["source"] == "embeddings.md"


def test_agent_stream_yields_trace_tokens_and_result():
    hits = make_hits([(CHUNK_TEXT, "chunking.md", 0.90)])
    agent = build_agent(hits)
    events = list(agent.ask_stream("What is chunking?"))

    types = [e["type"] for e in events]
    assert types[0] == "trace"
    assert "token" in types
    assert types[-1] == "result"
    assert events[-1]["decision"] == "answer"
    assert events[-1]["citations"][0]["source"] == "chunking.md"
    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert "Chunking splits documents" in tokens


def test_agent_stream_dont_know_still_streams_a_result():
    hits = make_hits([(CHUNK_TEXT, "chunking.md", 0.10)])
    agent = build_agent(hits)
    events = list(agent.ask_stream("What is the capital of France?"))
    assert events[-1]["type"] == "result"
    assert events[-1]["decision"] == "dont_know"
