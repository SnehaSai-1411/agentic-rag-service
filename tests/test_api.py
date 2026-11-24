"""API tests — TestClient with the real agent swapped for a fake one."""
import json

from fastapi.testclient import TestClient

from agent.core import Agent
from app.main import app, get_agent
from config import Settings
from tests.fakes import FakeLLM, fake_search, make_hits

CHUNK_TEXT = (
    "Chunking splits documents into pieces for retrieval. "
    "Small chunks are precise but can cut an argument in half."
)


def _fake_agent():
    hits = make_hits([(CHUNK_TEXT, "chunking.md", 0.90)])
    return Agent(
        search_fn=fake_search(hits),
        llm=FakeLLM("Chunking splits documents into pieces."),
        settings=Settings(min_top_score=0.35, ambiguity_gap=0.05),
    )


def make_client():
    app.dependency_overrides[get_agent] = _fake_agent
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


def test_health():
    with make_client() as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ask_returns_answer_trace_and_citations():
    with make_client() as client:
        resp = client.post("/ask", json={"question": "What is chunking?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] == "answer"
    assert "Chunking splits documents" in body["answer"]
    assert body["citations"][0]["source"] == "chunking.md"
    assert [s["step"] for s in body["trace"][:3]] == [
        "retrieve", "gate", "rerank"]


def _parse_sse(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if line.startswith("data:"):
            events.append(json.loads(line[len("data:"):].strip()))
    return events


def test_ask_stream_streams_trace_tokens_and_result():
    with make_client() as client:
        resp = client.post("/ask/stream", json={"question": "What is chunking?"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    events = _parse_sse(resp.text)
    types = [e["type"] for e in events]
    assert types[0] == "trace"
    assert "token" in types
    assert types[-1] == "result"
    assert events[-1]["decision"] == "answer"
    assert events[-1]["citations"][0]["source"] == "chunking.md"
    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert "Chunking splits documents" in tokens
    trace_steps = [e["step"]["step"] for e in events if e["type"] == "trace"]
    assert trace_steps == ["retrieve", "gate", "rerank"]
