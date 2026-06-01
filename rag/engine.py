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


# ── MCP / Agentic tool-use ────────────────────────────────────────────────────

SEARCH_TOOL = {
    "name": "search_ml_books",
    "description": (
        "Search the 3 ML textbooks (Géron Hands-on ML, Goodfellow Deep Learning, "
        "Hands-On LLMs) using hybrid retrieval + cross-encoder reranking. "
        "Call this when the question asks about ML concepts, algorithms, or content "
        "from these books. Skip for conversational follow-ups already answered in context."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query — rephrase the user's question for best retrieval.",
            },
            "top_k": {"type": "integer", "default": 20},
            "top_n": {"type": "integer", "default": 5},
            "alpha": {"type": "number", "default": 0.75},
        },
        "required": ["query"],
    },
}


def agentic_stream_answer(
    query: str,
    history: list[dict],
    top_k: int = HYBRID_TOP_K,
    top_n: int = RERANK_TOP_N,
    alpha: float = HYBRID_ALPHA,
) -> Generator:
    """
    Agentic RAG: Claude decides whether to call the search tool.

    Yields:
        str  — text tokens for the streaming answer
        dict — {"type": "hits", "data": list[dict]} — emitted once if search ran,
               so app.py can capture hits for the sources panel
    """
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    messages = history + [{"role": "user", "content": query}]

    # ── Turn 1: let Claude decide whether to search ───────────────────────────
    tool_use_block = None
    text_tokens_turn1 = []

    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        tools=[SEARCH_TOOL],
        messages=messages,
    ) as stream:
        for event in stream:
            # Capture text tokens in case Claude answers directly (no tool call)
            if (
                hasattr(event, "type")
                and event.type == "content_block_delta"
                and hasattr(event, "delta")
                and hasattr(event.delta, "type")
                and event.delta.type == "text_delta"
            ):
                text_tokens_turn1.append(event.delta.text)
        final_msg = stream.get_final_message()

    # Check if Claude decided to call the search tool
    for block in final_msg.content:
        if block.type == "tool_use":
            tool_use_block = block
            break

    # ── Path A: Claude answered directly — no search needed ───────────────────
    if tool_use_block is None:
        yield from text_tokens_turn1
        return

    # ── Path B: Claude called the tool — run retrieval ────────────────────────
    tool_input = tool_use_block.input
    hits = retrieve_and_rerank(
        query=tool_input.get("query", query),
        top_k=tool_input.get("top_k", top_k),
        top_n=tool_input.get("top_n", top_n),
        alpha=tool_input.get("alpha", alpha),
    )

    # Emit hits so app.py can populate the sources panel
    yield {"type": "hits", "data": hits}

    # Append tool_use + tool_result to message history
    messages.append({"role": "assistant", "content": final_msg.content})
    messages.append({
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": tool_use_block.id,
            "content": _build_context(hits) if hits else "No relevant passages found.",
        }],
    })

    # ── Turn 2: stream the final grounded answer ──────────────────────────────
    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text
