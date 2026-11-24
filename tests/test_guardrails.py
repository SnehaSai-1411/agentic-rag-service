"""Tests for the grounding hook in guardrails.py."""
from guardrails import grounding_check

CONTEXT = [
    "Chunking splits documents into pieces for retrieval. "
    "Small chunks are precise but can cut an argument in half.",
    "A common default is 800 characters per chunk with 100 characters of overlap.",
]


def test_supported_answer_passes():
    result = grounding_check(
        "Chunking splits documents into pieces for retrieval.",
        CONTEXT,
    )
    assert result.supported is True
    assert result.coverage == 1.0
    assert result.unsupported_sentences == []


def test_unsupported_answer_is_flagged():
    result = grounding_check(
        "The moon is made of cheese and quantum flux.",
        CONTEXT,
    )
    assert result.supported is False
    assert len(result.unsupported_sentences) == 1
    assert "moon" in result.unsupported_sentences[0]
    assert result.coverage == 0.0


def test_partly_supported_answer_flags_only_the_bad_sentence():
    result = grounding_check(
        "Chunking splits documents into pieces. "
        "The moon is made of cheese and quantum flux.",
        CONTEXT,
    )
    assert result.supported is False
    assert len(result.unsupported_sentences) == 1
    assert "moon" in result.unsupported_sentences[0]


def test_bracketed_label_lines_are_not_checked_as_claims():
    result = grounding_check(
        "[Extractive answer — no LLM key configured, quoting the most "
        "relevant retrieved passage:]\n\n"
        "Chunking splits documents into pieces for retrieval.",
        CONTEXT,
    )
    assert result.supported is True


def test_empty_context_flags_everything():
    result = grounding_check("Chunking splits documents.", [])
    assert result.supported is False
