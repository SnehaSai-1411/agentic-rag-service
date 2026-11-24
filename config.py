"""Configuration for the agentic RAG service.

All settings come from environment variables (or a local `.env` file).
No secrets live here — see `.env.example` for the placeholders.
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class Settings:
    # LLM
    llm_provider: str = "anthropic"  # "anthropic" | "openai" | "extractive"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-20250514"
    openai_model: str = "gpt-4o-mini"

    # Retrieval
    chroma_dir: str = "./chroma_db"
    collection: str = "agentic-rag-docs"
    top_k: int = 5
    embedding_model: str = "all-MiniLM-L6-v2"

    # Reranking: "cosine" (default) or "cross-encoder"
    reranker: str = "cosine"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Confidence gate (see agent/gate.py)
    min_top_score: float = 0.35   # below this, the agent says it doesn't know
    ambiguity_gap: float = 0.05   # top hits this close, from different sources -> clarify

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "anthropic").lower(),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            chroma_dir=os.getenv("CHROMA_DIR", "./chroma_db"),
            collection=os.getenv("COLLECTION", "agentic-rag-docs"),
            top_k=int(os.getenv("TOP_K", "5")),
            reranker=os.getenv("RERANKER", "cosine").lower(),
            min_top_score=float(os.getenv("MIN_TOP_SCORE", "0.35")),
            ambiguity_gap=float(os.getenv("AMBIGUITY_GAP", "0.05")),
        )
