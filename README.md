# agentic-rag-service

An agentic RAG service that decides per question whether to answer from its knowledge base, ask a clarifying question, or say it doesn't know — with a structured trace of every step.

## What it does
- **Decides, doesn't just answer.** A confidence gate (retrieval scores only, no LLM) routes each question to ANSWER, CLARIFY (vague query, or top hits span unrelated topics), or DON'T KNOW (top score below a configurable threshold).
- **Reranks before answering.** Cosine re-scoring by default; an optional cross-encoder (`RERANKER=cross-encoder`) that falls back to cosine if the model can't load.
- **Streams answers.** `POST /ask/stream` sends trace events, then answer tokens, then a result event over SSE.
- **Shows its work.** Every response carries a trace (tool calls + scores + the gate's reasoning), citations (source, page, chunk, score), and a grounding-check result.
- **Runs with no API key.** Without a Claude/OpenAI key, answers come from an extractive fallback that quotes the top retrieved passage and labels itself as such.

## Architecture

```
                ┌──────────────┐
  question ───▶ │  FastAPI     │  POST /ask · POST /ask/stream · GET /health
                │  app/main.py │
                └──────┬───────┘
                       │
                ┌──────▼───────┐
                │  Agent loop  │  agent/core.py — one pass, no framework:
                │  retrieve ──▶│  search_knowledge_base (Chroma + MiniLM)
                │  gate ──────▶│  ANSWER / CLARIFY / DON'T KNOW (gate.py)
                │  rerank ────▶│  cosine or cross-encoder (rerank.py)
                │  generate ──▶│  Claude / OpenAI / extractive fallback
                │  ground ────▶│  grounding check (guardrails.py)
                └──────┬───────┘
                       │  decision + answer + citations + trace + grounding
                       ▼
                   JSON / SSE
```

**How it composes with the sibling repos (conceptual — nothing is imported):**
Project 1 (`rag-data-ingestion-pipeline`) built the collection schema this repo reads and writes (chunk ids `{source}-p{page}-c{chunk_index}-{hash12}`, metadata `source`/`page`/`chunk_index`/`content_hash`, cosine space), so a collection built by either repo works in the other. Project 2 (`rag-chatbot-fastapi`) is the plain RAG service this agent wraps with decisions, reranking, and streaming; Project 3 (`rag-evaluation-harness`) is where the gate thresholds (`MIN_TOP_SCORE`, `AMBIGUITY_GAP`) should be tuned against a golden set instead of by feel. Project 4 (`llm-guardrails-pii`) plugs in as FastAPI middleware around `/ask` — the local grounding check in `guardrails.py` is the hook point:

```python
# app/main.py — wrap the handler with the Project 4 middleware when available:
from llm_guardrails_pii import GuardrailMiddleware  # sibling repo, not a dependency here
app.add_middleware(GuardrailMiddleware)  # PII redaction + injection checks + stronger grounding
```

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/ingest_sample.py
.venv/bin/uvicorn app.main:app
```

Then `POST http://127.0.0.1:8000/ask`. No API key needed (extractive fallback); to use Claude or OpenAI, copy `.env.example` to `.env` and add a key. On a CPU-only machine, install CPU-only torch first to avoid the ~500 MB CUDA wheel: `pip install torch --index-url https://download.pytorch.org/whl/cpu`. Tests: `.venv/bin/python -m pytest -q` (33 tests, all offline, no model download, no key).

## Example request/response

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is chunking and why does chunk size matter?"}'
```

```json
{
  "decision": "answer",
  "answer": "[Extractive answer — no LLM key configured, quoting the most relevant retrieved passage:]\n\nA common default is around 800 characters per chunk with about 100 characters of overlap between consecutive chunks. ...",
  "citations": [
    {"source": "chunking.md", "page": 1, "chunk_index": 1, "score": 0.649},
    {"source": "chunking.md", "page": 1, "chunk_index": 0, "score": 0.613},
    {"source": "embeddings.md", "page": 1, "chunk_index": 1, "score": 0.49}
  ],
  "trace": [
    {"step": "retrieve", "tool": "search_knowledge_base",
     "input": "What is chunking and why does chunk size matter?",
     "scores": [0.649, 0.613, 0.49, 0.235, 0.186],
     "detail": "Retrieved 5 chunk(s) from collection 'agentic-rag-docs'."},
    {"step": "gate",
     "detail": "Decision: answer. Top score 0.649 clears the threshold 0.350; answering from the knowledge base."},
    {"step": "rerank", "scores": [0.649, 0.613, 0.49, 0.235, 0.186],
     "detail": "Reranked with mode 'cosine'."},
    {"step": "generate",
     "detail": "Answer generated with LLM 'extractive-fallback' from 3 context chunk(s)."},
    {"step": "grounding_check",
     "detail": "Grounding check: supported=True, coverage=1.00."}
  ],
  "grounding": {"supported": true, "coverage": 1.0, "unsupported_sentences": []}
}
```

A question the knowledge base can't answer gets an honest don't-know instead of a guess:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}'
```

```json
{
  "decision": "dont_know",
  "answer": "I don't know — I couldn't find anything in the knowledge base that closely matches your question, so I'd rather say so than guess. ...",
  "citations": [],
  "trace": [
    {"step": "retrieve", "tool": "search_knowledge_base",
     "input": "What is the capital of France?",
     "scores": [0.058, 0.023, 0.018, 0.0, -0.001],
     "detail": "Retrieved 5 chunk(s) from collection 'agentic-rag-docs'."},
    {"step": "gate",
     "detail": "Decision: dont_know. Top retrieval score 0.058 is below the confidence threshold 0.350."}
  ],
  "grounding": null
}
```

## Design decisions & trade-offs
- **A simple loop instead of LangGraph:** the agent is one pass — retrieve, gate, rerank, generate — with no multi-turn planning, branching, or persistent state. A graph framework would add dependency weight and indirection without changing what the system does; it becomes worth it only when multi-hop retrieval or tool chaining arrives.
- **The gate runs on retrieval scores, not reranked scores:** cosine similarities have a stable scale for thresholding, while cross-encoder logits (even sigmoid-normalized) compress at the top end and would make the ambiguity gap trigger-happy. Reranking orders the answer context; it never changes the decision.
- **Cosine reranking by default, cross-encoder optional:** the cross-encoder reads query and chunk together, so it ranks better — but it costs one forward pass per pair plus a model download, and on this tiny sample set it disagreed with the bi-encoder about the top chunk. It falls back to cosine if the model can't load, so the service never fails on reranking.
- **The grounding check is a word-overlap heuristic, not a proof:** a correct paraphrase can score low, and shared words don't guarantee a true statement. It is a tripwire that flags unsupported sentences; the Project 4 middleware is the stronger check.
- **Thresholds are defaults, not measurements:** `MIN_TOP_SCORE=0.35` and `AMBIGUITY_GAP=0.05` were set by feel on four sample docs. Tuning them against a real golden set (Project 3) is the honest next step, not claiming these values generalize.

## What I'd add next
- Tune the gate thresholds with the Project 3 evaluation harness on a golden Q&A set, and report the resulting answer/clarify/don't-know rates.
- Conversation history (last N turns) so follow-up questions resolve against what was already asked.
- Hybrid retrieval (BM25 + vector) from Project 2, so keyword-heavy questions stop depending on embedding similarity alone.
- Multi-hop retrieval — and only then, revisit LangGraph for the orchestration.
- One deploy: GCP Cloud Run, with the Chroma directory on a persistent volume.
