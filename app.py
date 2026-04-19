"""ML Books RAG — Streamlit chat app with hybrid search, reranking, and streaming."""
import sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="ML Books Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 ML Books RAG")
    st.caption("Powered by Claude claude-sonnet-4-6 · Pinecone Hybrid Search · Cross-Encoder Reranking")
    st.divider()

    st.subheader("Retrieval Settings")
    alpha = st.slider(
        "Dense ↔ Sparse balance",
        min_value=0.0, max_value=1.0, value=0.75, step=0.05,
        help="1.0 = pure semantic (dense), 0.0 = pure keyword (BM25)",
    )
    top_k = st.slider("Candidates to retrieve", min_value=5, max_value=30, value=20)
    top_n = st.slider("Chunks after reranking", min_value=3, max_value=10, value=5)

    st.divider()
    st.subheader("Books indexed")
    st.markdown(
        "- 📘 Hands-on ML (Géron)\n"
        "- 📗 Deep Learning (Goodfellow)\n"
        "- 📙 Hands-On LLMs"
    )
    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.sources = []
        if "memory" in st.session_state:
            st.session_state.memory.clear()
        st.rerun()

# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources" not in st.session_state:
    st.session_state.sources = []
if "memory" not in st.session_state:
    from rag.memory import ChatMemory
    st.session_state.memory = ChatMemory()

# ── Layout: chat (left) + sources (right) ────────────────────────────────────
col_chat, col_sources = st.columns([2, 1], gap="large")

# ── Sources panel ─────────────────────────────────────────────────────────────
with col_sources:
    st.subheader("📎 Sources")
    if st.session_state.sources:
        for i, hit in enumerate(st.session_state.sources, 1):
            m = hit["metadata"]
            book = m.get("book", "Unknown")
            ch_num = m.get("chapter_number", "?")
            ch_title = m.get("chapter_title", "")
            p_start = m.get("start_page", "?")
            p_end = m.get("end_page", "?")
            label = f"{i}. {book[:28]}… | Ch.{ch_num} | pp.{p_start}–{p_end}"
            with st.expander(label):
                st.caption(f"**{book}**")
                st.caption(f"Chapter {ch_num}: {ch_title}")
                st.caption(f"Pages {p_start}–{p_end}")
                preview = hit.get("text", "")[:500]
                st.markdown(f"> {preview}…")
                if "rerank_score" in hit:
                    st.caption(f"Rerank score: {hit['rerank_score']:.3f}")
    else:
        st.caption("Sources will appear here after your first question.")

# ── Chat panel ────────────────────────────────────────────────────────────────
with col_chat:
    st.title("ML Books Assistant")
    st.caption("Ask anything about machine learning, deep learning, or large language models.")

    # Render conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("e.g. What is backpropagation? How does attention work?"):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Phase 1: retrieve + rerank (synchronous, fast ~0.5–1.5s)
        with st.spinner("Searching books…"):
            try:
                from rag.engine import retrieve_and_rerank, stream_answer
                hits = retrieve_and_rerank(
                    query=prompt,
                    top_k=top_k,
                    top_n=top_n,
                    alpha=alpha,
                )
                st.session_state.sources = hits
            except Exception as e:
                err_str = str(e)
                if "NOT_FOUND" in err_str or "not found" in err_str.lower():
                    st.error(
                        "**Pinecone index not found.**\n\n"
                        "You need to run the ingest pipeline first:\n"
                        "```\n"
                        "source venv/bin/activate\n"
                        "python -m ingest.parser    # parse PDFs (20–40 min)\n"
                        "python -m ingest.chunker   # detect chapters\n"
                        "python -m ingest.embedder  # embed + upload to Pinecone\n"
                        "```"
                    )
                else:
                    st.error(f"Retrieval error: {e}")
                st.stop()

        # Refresh sources panel immediately (rerun needed — use a placeholder instead)
        # Phase 2: stream LLM answer
        with st.chat_message("assistant"):
            history = st.session_state.memory.get_messages()
            try:
                full_response = st.write_stream(
                    stream_answer(prompt, hits, history)
                )
            except Exception as e:
                st.error(f"Generation error: {e}")
                st.stop()

        # Persist to session state
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.memory.add("user", prompt)
        st.session_state.memory.add("assistant", full_response)

        # Rerun to refresh sources panel with the new hits
        st.rerun()
