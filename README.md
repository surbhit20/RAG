# ML Books RAG

A Streamlit chat app that lets you ask questions about three foundational ML/DL books using Retrieval-Augmented Generation (RAG).

**Books indexed:**
- Hands-on Machine Learning (Géron)
- Deep Learning (Goodfellow et al.)
- Hands-On Large Language Models

## Architecture

```
PDF files
   └─► LlamaParse (parser.py)
         └─► Chapter-aware chunker (chunker.py)
               └─► OpenAI embeddings + BM25 (embedder.py)
                     └─► Pinecone hybrid index (ml-rag-hybrid)
                           └─► Cross-encoder reranker
                                 └─► Claude claude-sonnet-4-6 (streaming)
```

- **Hybrid search** — dense (OpenAI `text-embedding-3-small`) + sparse (BM25) via Pinecone
- **Reranking** — `cross-encoder/ms-marco-MiniLM-L-6-v2` narrows candidates before LLM call
- **Streaming answers** — Claude claude-sonnet-4-6 with multi-turn chat memory
- **Source panel** — shows book, chapter, and page range for every retrieved chunk

## Setup

### 1. Clone & install

```bash
git clone https://github.com/surbhit20/RAG.git
cd RAG
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
PINECONE_API_KEY=...
LLAMA_CLOUD_API_KEY=...
HF_TOKEN=...
```

### 3. Add PDFs

Place the three PDF files in the project root (they are git-ignored):
- `Aurelien Geron - Hands-on Machine Learning...pdf`
- `Deep+Learning+Ian+Goodfellow.pdf`
- `Hands-On Large Language Models.pdf`

### 4. Run the ingest pipeline (first time only)

```bash
python -m ingest.parser     # parse PDFs via LlamaParse (~20–40 min)
python -m ingest.chunker    # detect chapters and split into chunks
python -m ingest.embedder   # embed and upload to Pinecone
```

### 5. Run the app

```bash
streamlit run app.py
```

## Retrieval settings (sidebar)

| Setting | Default | Description |
|---|---|---|
| Dense ↔ Sparse balance (α) | 0.75 | 1.0 = pure semantic, 0.0 = pure BM25 |
| Candidates to retrieve | 20 | Chunks fetched from Pinecone |
| Chunks after reranking | 5 | Chunks passed to the LLM |

## Cloud deployment

Deploy to [Streamlit Community Cloud](https://share.streamlit.io) (free):
1. Connect this repo
2. Set main file to `app.py`
3. Add API keys under **Advanced settings → Secrets**

The Pinecone index is already cloud-hosted — no PDFs needed on the server.
