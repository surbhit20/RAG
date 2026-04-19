"""Two-phase RAG engine: synchronous retrieval → streaming synthesis via Anthropic SDK."""
import sys
from pathlib import Path
from typing import Generator

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, MAX_TOKENS, HYBRID_TOP_K, RERANK_TOP_N, HYBRID_ALPHA
from rag.prompts import SYSTEM_PROMPT


def _build_context(hits: list[dict]) -> str:
    parts = []
    for i, hit in enumerate(hits, 1):
        m = hit["metadata"]
        header = (
            f"[Source {i}] {m.get('book', 'Unknown')} | "
            f"Chapter {m.get('chapter_number', '?')}: {m.get('chapter_title', '')} | "
            f"pp.{m.get('start_page', '?')}–{m.get('end_page', '?')}"
        )
        parts.append(f"{header}\n{hit['text']}")
    return "\n\n---\n\n".join(parts)


def retrieve_and_rerank(
    query: str,
    top_k: int = HYBRID_TOP_K,
    top_n: int = RERANK_TOP_N,
    alpha: float = HYBRID_ALPHA,
) -> list[dict]:
    """Phase 1: hybrid retrieval + cross-encoder reranking. Returns top_n hits."""
    from retrieval.hybrid_retriever import HybridRetriever
    from retrieval.reranker import rerank

    retriever = HybridRetriever(top_k=top_k, alpha=alpha)
    hits = retriever.retrieve(query, top_k=top_k, alpha=alpha)
    return rerank(query, hits, top_n=top_n)


def stream_answer(
    query: str,
    hits: list[dict],
    history: list[dict],
) -> Generator[str, None, None]:
    """Phase 2: stream answer from Claude given retrieved hits and chat history."""
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    context = _build_context(hits)

    user_message = (
        f"Context passages from the textbooks:\n\n{context}\n\n"
        f"Question: {query}"
    )

    messages = history + [{"role": "user", "content": user_message}]

    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text
