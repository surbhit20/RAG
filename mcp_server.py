"""MCP server exposing ML Books RAG as tools for Claude Desktop / Claude Code."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mcp.server.fastmcp import FastMCP
from config import BOOK_DISPLAY_NAMES

mcp = FastMCP("ml-books-rag")


@mcp.tool()
def search_ml_books(
    query: str,
    top_k: int = 20,
    top_n: int = 5,
    alpha: float = 0.75,
) -> str:
    """
    Search 3 ML textbooks using hybrid semantic + BM25 retrieval
    with cross-encoder reranking. Returns the most relevant passages.

    Args:
        query: The question or topic to search for in the books.
        top_k: Candidates to retrieve before reranking (default 20).
        top_n: Final chunks after reranking (default 5).
        alpha: Dense/sparse balance — 1.0 = pure semantic, 0.0 = pure keyword (default 0.75).
    """
    # Deferred import — keeps server startup instant; loads models on first call
    from rag.engine import retrieve_and_rerank, _build_context

    hits = retrieve_and_rerank(query=query, top_k=top_k, top_n=top_n, alpha=alpha)
    if not hits:
        return "No relevant passages found for this query."
    return _build_context(hits)


@mcp.tool()
def list_books() -> str:
    """List the ML textbooks available in this RAG system."""
    lines = ["Available textbooks:"]
    for name in BOOK_DISPLAY_NAMES.values():
        lines.append(f"  - {name}")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
