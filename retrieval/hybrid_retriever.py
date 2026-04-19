"""Hybrid Pinecone retriever: combines dense semantic + BM25 sparse search."""
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PINECONE_API_KEY, PINECONE_INDEX_NAME,
    OPENAI_API_KEY, EMBED_MODEL,
    HYBRID_TOP_K, HYBRID_ALPHA,
    CACHE_DIR,
)


def _load_bm25():
    from pinecone_text.sparse import BM25Encoder
    cache = CACHE_DIR / "bm25_corpus.pkl"
    if not cache.exists():
        raise FileNotFoundError("Run ingest.embedder first to generate bm25_corpus.pkl")
    bm25 = BM25Encoder()
    bm25.load(str(cache))
    return bm25


def _get_pinecone_index():
    from pinecone import Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)


def _embed_query(text: str) -> list[float]:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding


class HybridRetriever:
    """Alpha-blended dense+sparse retriever backed by Pinecone."""

    def __init__(
        self,
        top_k: int = HYBRID_TOP_K,
        alpha: float = HYBRID_ALPHA,
    ):
        self._index = _get_pinecone_index()
        self._bm25 = _load_bm25()
        self._top_k = top_k
        self._alpha = alpha

    def retrieve(self, query: str, top_k: Optional[int] = None, alpha: Optional[float] = None) -> list[dict]:
        """
        Returns list of dicts with keys: id, score, text, metadata.
        """
        k = top_k or self._top_k
        a = alpha if alpha is not None else self._alpha

        dense = _embed_query(query)
        sparse = self._bm25.encode_queries([query])[0]

        scaled_dense = [v * a for v in dense]
        scaled_sparse = {
            "indices": sparse["indices"],
            "values": [v * (1 - a) for v in sparse["values"]],
        }

        results = self._index.query(
            vector=scaled_dense,
            sparse_vector=scaled_sparse,
            top_k=k,
            include_metadata=True,
        )

        hits = []
        for match in results.matches:
            hits.append({
                "id": match.id,
                "score": match.score,
                "text": match.metadata.get("text", ""),
                "metadata": match.metadata,
            })
        return hits
