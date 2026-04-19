"""Verify all API keys and dependencies are configured before running ingest."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

errors = []
warnings = []

print("Checking environment...\n")

# Check .env file
from config import (
    ANTHROPIC_API_KEY, OPENAI_API_KEY, PINECONE_API_KEY, LLAMA_CLOUD_API_KEY,
    PDF_FILES, BOOK_DISPLAY_NAMES,
)

keys = {
    "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "PINECONE_API_KEY": PINECONE_API_KEY,
    "LLAMA_CLOUD_API_KEY": LLAMA_CLOUD_API_KEY,
}
for name, val in keys.items():
    if not val:
        errors.append(f"Missing: {name}")
    else:
        print(f"  ✓ {name} set ({val[:8]}...)")

print()

# Check PDFs
for key, path in PDF_FILES.items():
    if path.exists():
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"  ✓ {BOOK_DISPLAY_NAMES[key]} ({size_mb:.1f} MB)")
    else:
        errors.append(f"PDF not found: {path.name}")

print()

# Check NLTK data
try:
    from nltk.corpus import stopwords
    _ = stopwords.words("english")
    print("  ✓ NLTK stopwords OK")
except LookupError:
    warnings.append("NLTK stopwords missing — run: python -c \"import nltk; nltk.download('stopwords')\"")

# Check imports
try:
    from pinecone_text.sparse import BM25Encoder
    print("  ✓ pinecone-text BM25 OK")
except Exception as e:
    errors.append(f"pinecone-text: {e}")

try:
    from sentence_transformers import CrossEncoder
    print("  ✓ sentence-transformers OK")
except Exception as e:
    errors.append(f"sentence-transformers: {e}")

print()

if warnings:
    for w in warnings:
        print(f"  ⚠ WARNING: {w}")

if errors:
    print("ERRORS — fix these before running ingest:")
    for e in errors:
        print(f"  ✗ {e}")
    sys.exit(1)
else:
    print("All checks passed! Ready to run:\n")
    print("  python -m ingest.parser")
    print("  python -m ingest.chunker")
    print("  python -m ingest.embedder")
    print("  streamlit run app.py")
