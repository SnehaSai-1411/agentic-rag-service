"""FastAPI service for the agentic RAG agent.

Endpoints:
- POST /ask          full JSON answer + trace + citations
- POST /ask/stream   SSE: trace events, then answer tokens, then a result event
- GET  /health       liveness check (does not load any model)

The agent (embeddings + Chroma + LLM) is built lazily on first use, so
importing this module — and hitting /health — never loads torch or opens
the vector store. Tests override the `get_agent` dependency with a fake.
"""
from functools import lru_cache

from fastapi import Depends, FastAPI
from pydantic import BaseModel
from sse_starlette.event import JSONServerSentEvent
from sse_starlette.sse import EventSourceResponse

from agent import Agent, Reranker
from config import Settings
from llm import build_llm
from retrieval import ChromaRetriever, Embedder

app = FastAPI(title="agentic-rag-service")


class AskRequest(BaseModel):
    question: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()


def build_agent(settings: Settings | None = None) -> Agent:
    """Wire the real agent: Chroma retriever + reranker + LLM."""
    settings = settings or get_settings()
    embedder = Embedder(model_name=settings.embedding_model)
    retriever = ChromaRetriever(
        embedder,
        persist_dir=settings.chroma_dir,
        collection_name=settings.collection,
    )
    reranker = Reranker(
        mode=settings.reranker,
        cross_encoder_model=settings.cross_encoder_model,
    )
    return Agent(
        search_fn=retriever.search,
        llm=build_llm(settings),
        reranker=reranker,
        settings=settings,
        embed_fn=retriever.embed_query,
    )


@lru_cache(maxsize=1)
def get_agent() -> Agent:
    return build_agent()


@app.get("/health")
def health(settings: Settings = Depends(get_settings)):
    return {
        "status": "ok",
        "collection": settings.collection,
        "reranker": settings.reranker,
        "llm_provider": settings.llm_provider,
    }


@app.post("/ask")
def ask(req: AskRequest, agent: Agent = Depends(get_agent)):
    return agent.ask(req.question)


@app.post("/ask/stream")
def ask_stream(req: AskRequest, agent: Agent = Depends(get_agent)):
    def event_generator():
        for event in agent.ask_stream(req.question):
            # sse-starlette 3.x passes yielded dicts to ServerSentEvent as
            # kwargs, so build the event explicitly: the SSE event name is
            # the agent event type ("trace" / "token" / "result") and the
            # full event dict is the JSON data payload. (Plain
            # ServerSentEvent would str() the dict into a Python repr;
            # JSONServerSentEvent json-encodes it.)
            yield JSONServerSentEvent(data=event, event=event["type"])

    return EventSourceResponse(event_generator())
