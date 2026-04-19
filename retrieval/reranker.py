"""Cross-encoder reranker using ms-marco-MiniLM-L-6-v2 (runs locally, MPS on Apple Silicon)."""
import sys
from pathlib import Path
from functools import lru_cache

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CROSS_ENCODER_MODEL, RERANK_TOP_N


@lru_cache(maxsize=1)
def _load_cross_encoder():
    import transformers
    import logging
    transformers.logging.set_verbosity_error()
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

    from sentence_transformers import CrossEncoder
    model = CrossEncoder(CROSS_ENCODER_MODEL)
    return model


def rerank(query: str, hits: list[dict], top_n: int = RERANK_TOP_N) -> list[dict]:
    """
    Re-score hits using cross-encoder and return top_n sorted by score.
    Each hit must have a 'text' key.
    """
    if not hits:
        return hits

    model = _load_cross_encoder()
    pairs = [(query, h["text"]) for h in hits]
    scores = model.predict(pairs)

    ranked = sorted(
        zip(scores.tolist(), hits),
        key=lambda x: x[0],
        reverse=True,
    )
    results = []
    for score, hit in ranked[:top_n]:
        results.append({**hit, "rerank_score": score})
    return results
