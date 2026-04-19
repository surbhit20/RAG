"""Parse PDFs using LlamaParse with aggressive disk caching."""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PARSED_DIR, LLAMA_CLOUD_API_KEY, PDF_FILES, BOOK_DISPLAY_NAMES


def _cache_key(pdf_path: Path) -> str:
    stat = pdf_path.stat()
    raw = f"{pdf_path.name}:{stat.st_size}:{stat.st_mtime}"
    return hashlib.md5(raw.encode()).hexdigest()


def parse_pdf(pdf_path: Path, book_key: str) -> str:
    """Return full markdown string for a PDF. Uses disk cache to avoid re-parsing."""
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(pdf_path)
    cache_file = PARSED_DIR / f"{book_key}_{key}.md"

    if cache_file.exists():
        print(f"[cache hit] {pdf_path.name}")
        return cache_file.read_text(encoding="utf-8")

    print(f"[parsing] {pdf_path.name} via LlamaParse (this may take 10–20 min)...")
    from llama_parse import LlamaParse

    parser = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="markdown",
        num_workers=4,
        verbose=True,
        language="en",
    )
    documents = parser.load_data(str(pdf_path))
    full_markdown = "\n\n".join(doc.text for doc in documents)
    cache_file.write_text(full_markdown, encoding="utf-8")
    print(f"[cached] → {cache_file}")
    return full_markdown


def parse_all() -> dict[str, str]:
    """Parse all 3 books. Returns {book_key: markdown_text}."""
    results = {}
    for book_key, pdf_path in PDF_FILES.items():
        if not pdf_path.exists():
            print(f"[skip] {pdf_path.name} not found")
            continue
        print(f"\n=== {BOOK_DISPLAY_NAMES[book_key]} ===")
        results[book_key] = parse_pdf(pdf_path, book_key)
    return results


if __name__ == "__main__":
    books = parse_all()
    for key, text in books.items():
        print(f"{key}: {len(text):,} chars")
