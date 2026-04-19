"""Embed chunks with OpenAI and upsert to Pinecone with BM25 sparse vectors."""
import pickle
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CACHE_DIR, OPENAI_API_KEY, PINECONE_API_KEY,
    EMBED_MODEL, EMBED_DIMS,
    PINECONE_INDEX_NAME, PINECONE_CLOUD, PINECONE_REGION,
)

MAX_TOKENS_PER_BATCH = 250_000  # OpenAI limit is 300k; stay under with buffer
BM25_CACHE = CACHE_DIR / "bm25_corpus.pkl"


def _token_aware_batches(chunks, texts: list[str]) -> list[list]:
    """Group chunks into batches that stay under MAX_TOKENS_PER_BATCH."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    batches = []
    current_batch = []
    current_tokens = 0

    for chunk, text in zip(chunks, texts):
        n = len(enc.encode(text, disallowed_special=()))
        if current_batch and current_tokens + n > MAX_TOKENS_PER_BATCH:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(chunk)
        current_tokens += n

    if current_batch:
        batches.append(current_batch)
    return batches


def _get_index():
    from pinecone import Pinecone, ServerlessSpec
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if not pc.has_index(PINECONE_INDEX_NAME):
        print(f"[pinecone] Creating index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBED_DIMS,
            metric="dotproduct",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        # Wait for index to be ready
        while True:
            status = pc.describe_index(PINECONE_INDEX_NAME).status
            if isinstance(status, dict) and status.get("ready"):
                break
            elif hasattr(status, "ready") and status.ready:
                break
            print("  waiting for index to be ready...")
            time.sleep(3)
    return pc.Index(PINECONE_INDEX_NAME)


def _embed_batch(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def _fit_bm25(texts: list[str]):
    from pinecone_text.sparse import BM25Encoder
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if BM25_CACHE.exists():
        print("[bm25] Loading cached BM25 encoder...")
        bm25 = BM25Encoder()
        bm25.load(str(BM25_CACHE))
        return bm25
    print(f"[bm25] Fitting on {len(texts)} chunks...")
    bm25 = BM25Encoder()
    bm25.fit(texts)
    bm25.dump(str(BM25_CACHE))
    print(f"[bm25] Saved → {BM25_CACHE}")
    return bm25


def embed_and_upsert(chunks) -> None:
    from ingest.chunker import Chunk
    index = _get_index()
    texts = [c.text for c in chunks]
    bm25 = _fit_bm25(texts)

    total = len(chunks)
    batches = _token_aware_batches(chunks, texts)
    print(f"\n[embed] {total} chunks → {len(batches)} token-aware batches...")

    done = 0
    for batch in batches:
        batch_texts = [c.text for c in batch]

        dense_vecs = _embed_batch(batch_texts)
        sparse_vecs = bm25.encode_documents(batch_texts)

        vectors = []
        for chunk, dense, sparse in zip(batch, dense_vecs, sparse_vecs):
            vectors.append({
                "id": chunk.chunk_id,
                "values": dense,
                "sparse_values": sparse,
                "metadata": {
                    "book": chunk.book_name,
                    "book_key": chunk.book_key,
                    "chapter_number": chunk.chapter_number,
                    "chapter_title": chunk.chapter_title,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    "start_page": chunk.start_page,
                    "end_page": chunk.end_page,
                    "text": chunk.text[:4000],  # Pinecone metadata limit
                },
            })

        # Upsert in sub-batches of 50 to stay under Pinecone's 2MB request limit
        UPSERT_SIZE = 50
        for j in range(0, len(vectors), UPSERT_SIZE):
            index.upsert(vectors=vectors[j:j + UPSERT_SIZE])
        done += len(batch)
        print(f"  [{done}/{total}] upserted")

    print(f"\n[done] {total} vectors in Pinecone index '{PINECONE_INDEX_NAME}'")


if __name__ == "__main__":
    from ingest.chunker import load_chunks
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from cache")
    embed_and_upsert(chunks)
