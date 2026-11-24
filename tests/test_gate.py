"""Tests for the confidence gate — pure functions, no I/O, no LLM."""
from agent.gate import decide, is_vague_query

MIN_TOP = 0.35
GAP = 0.05


def test_vague_queries():
    assert is_vague_query("chunking") is True
    assert is_vague_query("tell me more") is True
    assert is_vague_query("what about it?") is True
    assert is_vague_query("") is True
    assert is_vague_query("what is chunking?") is False
    assert is_vague_query("how does vector search work") is False


def test_decide_answer_on_high_confidence():
    hits = [
        {"score": 0.90, "source": "chunking.md"},
        {"score": 0.70, "source": "chunking.md"},
    ]
    d = decide("What is chunking?", hits, MIN_TOP, GAP)
    assert d.action == "answer"


def test_decide_dont_know_on_low_confidence():
    hits = [
        {"score": 0.20, "source": "chunking.md"},
        {"score": 0.15, "source": "embeddings.md"},
    ]
    d = decide("What is the capital of France?", hits, MIN_TOP, GAP)
    assert d.action == "dont_know"
    assert "0.200" in d.reason  # reason names the score and threshold


def test_decide_dont_know_when_nothing_retrieved():
    d = decide("What is chunking?", [], MIN_TOP, GAP)
    assert d.action == "dont_know"


def test_decide_clarify_on_vague_query_even_with_good_hits():
    hits = [{"score": 0.95, "source": "chunking.md"}]
    d = decide("chunking", hits, MIN_TOP, GAP)
    assert d.action == "clarify"


def test_decide_clarify_when_top_hits_span_sources_and_are_close():
    hits = [
        {"score": 0.90, "source": "chunking.md"},
        {"score": 0.88, "source": "embeddings.md"},
    ]
    d = decide("How do chunks and embeddings relate?", hits, MIN_TOP, GAP)
    assert d.action == "clarify"
    assert "chunking.md" in d.reason and "embeddings.md" in d.reason


def test_decide_answer_when_top_hits_span_sources_but_gap_is_clear():
    hits = [
        {"score": 0.90, "source": "chunking.md"},
        {"score": 0.60, "source": "embeddings.md"},
    ]
    d = decide("How do chunks and embeddings relate?", hits, MIN_TOP, GAP)
    assert d.action == "answer"


def test_decide_answer_when_top_hits_share_a_source():
    hits = [
        {"score": 0.90, "source": "chunking.md"},
        {"score": 0.89, "source": "chunking.md"},
    ]
    d = decide("What chunk size should I use?", hits, MIN_TOP, GAP)
    assert d.action == "answer"
