from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache"
PARSED_DIR = CACHE_DIR / "parsed"

PDF_FILES = {
    "geron": BASE_DIR / "Aurelien Geron - Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow-O\u2019reilly (2019).pdf",
    "goodfellow": BASE_DIR / "Deep+Learning+Ian+Goodfellow.pdf",
    "llm_book": BASE_DIR / "Hands-On Large Language Models.pdf",
}

BOOK_DISPLAY_NAMES = {
    "geron": "Hands-on Machine Learning (Géron)",
    "goodfellow": "Deep Learning (Goodfellow et al.)",
    "llm_book": "Hands-On Large Language Models",
}

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
LLAMA_CLOUD_API_KEY = os.environ.get("LLAMA_CLOUD_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# HuggingFace Hub reads this env var automatically for authenticated requests
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

CLAUDE_MODEL = "claude-sonnet-4-6"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 1536
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

PINECONE_INDEX_NAME = "ml-rag-hybrid"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

CHAPTER_TOKEN_LIMIT = 6000
CHUNK_OVERLAP_TOKENS = 200

HYBRID_TOP_K = 20
RERANK_TOP_N = 5
HYBRID_ALPHA = 0.75

MAX_TOKENS = 2048
CHAT_MEMORY_TURNS = 10
