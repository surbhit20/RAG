"""Detect chapter boundaries from LlamaParse markdown and sub-chunk large chapters."""
import re
import bisect
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CACHE_DIR, BOOK_DISPLAY_NAMES,
    CHAPTER_TOKEN_LIMIT, CHUNK_OVERLAP_TOKENS,
)

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def _count_tokens(text: str) -> int:
        return len(_enc.encode(text, disallowed_special=()))
except Exception:
    def _count_tokens(text: str) -> int:
        return len(text) // 4


CHAPTER_PATTERNS = [
    re.compile(r'^#{1,2}\s+Chapter\s+\d+[^\n]*', re.IGNORECASE | re.MULTILINE),
    re.compile(r'^#\s+\d+\s+[^\n]+', re.MULTILINE),
    re.compile(r'^#{1,2}\s+Part\s+[IVXivx\d]+[^\n]*', re.IGNORECASE | re.MULTILINE),
]

# Minimum chars between two boundaries to treat them as distinct sections
_MIN_SECTION_GAP = 50

PAGE_MARKER = re.compile(r'<!--\s*page\s+(\d+)\s*-->', re.IGNORECASE)
HEADING_RE = re.compile(r'^(#{1,2})\s+(.+)', re.MULTILINE)


@dataclass
class Chunk:
    book_key: str
    book_name: str
    chapter_number: int
    chapter_title: str
    chunk_index: int
    total_chunks: int
    text: str
    start_page: int
    end_page: int
    chunk_id: str = field(init=False)

    def __post_init__(self):
        self.chunk_id = (
            f"{self.book_key}_ch{self.chapter_number:02d}_chunk{self.chunk_index:03d}"
        )


def _build_page_map(markdown: str) -> list[tuple[int, int]]:
    """Return list of (char_offset, page_number) sorted by offset."""
    return [(m.start(), int(m.group(1))) for m in PAGE_MARKER.finditer(markdown)]


def _page_at(page_map: list[tuple[int, int]], char_offset: int) -> int:
    if not page_map:
        return 1
    offsets = [p[0] for p in page_map]
    idx = bisect.bisect_right(offsets, char_offset) - 1
    return page_map[max(0, idx)][1]


def _find_chapter_boundaries(markdown: str) -> list[tuple[int, str]]:
    """Return list of (char_offset, heading_text) for each chapter/section."""
    matches = []
    for pattern in CHAPTER_PATTERNS:
        for m in pattern.finditer(markdown):
            matches.append((m.start(), m.group(0).strip()))

    if len(matches) < 3:
        for m in HEADING_RE.finditer(markdown):
            if len(m.group(1)) == 1:
                matches.append((m.start(), m.group(0).strip()))

    # Deduplicate: merge boundaries within _MIN_SECTION_GAP chars
    # When merging, combine chapter number from numbered heading with title from named heading
    sorted_matches = sorted(matches, key=lambda x: x[0])
    unique: list[tuple[int, str]] = []
    for offset, text in sorted_matches:
        if unique and abs(offset - unique[-1][0]) < _MIN_SECTION_GAP:
            prev_offset, prev_text = unique[-1]
            # If prev has a number but no title, and current has a title — combine
            prev_num = re.findall(r'\d+', prev_text)
            curr_num = re.findall(r'\d+', text)
            if prev_num and not curr_num:
                # e.g. "## Chapter 1" + "# Understanding LLMs" → "# Chapter 1: Understanding LLMs"
                title_text = re.sub(r'^#+\s*', '', text).strip()
                combined = f"{prev_text}: {title_text}"
                unique[-1] = (prev_offset, combined)
            elif len(text) > len(prev_text):
                unique[-1] = (offset, text)
            # else keep prev
        else:
            unique.append((offset, text))

    return unique


def _extract_chapter_number(heading: str) -> int:
    nums = re.findall(r'\d+', heading)
    return int(nums[0]) if nums else 0


def _clean_heading_title(heading: str) -> str:
    text = re.sub(r'^#{1,3}\s*', '', heading).strip()
    # Strip "Chapter N:" or "Chapter N " prefix
    text = re.sub(r'(?i)^chapter\s*\d+\s*[:\-–]?\s*', '', text).strip()
    # Strip leading bare number like "1 " or "12 "
    text = re.sub(r'^\d+\s+', '', text).strip()
    # Strip trailing asterisks or underscores (bold/italic artifacts)
    text = text.strip('*_').strip()
    return text or heading.strip('#').strip()


def _sub_chunk(text: str, limit: int, overlap: int) -> list[str]:
    """Split text on paragraph boundaries into chunks of ≤ limit tokens."""
    paragraphs = re.split(r'\n{2,}', text)
    chunks = []
    current_paras: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _count_tokens(para)
        if current_tokens + para_tokens > limit and current_paras:
            chunks.append("\n\n".join(current_paras))
            # keep overlap: drop paragraphs from front until under overlap limit
            while current_paras and current_tokens > overlap:
                dropped = current_paras.pop(0)
                current_tokens -= _count_tokens(dropped)
        current_paras.append(para)
        current_tokens += para_tokens

    if current_paras:
        chunks.append("\n\n".join(current_paras))

    return chunks if chunks else [text]


def chunk_book(book_key: str, markdown: str) -> list[Chunk]:
    """Detect chapters and return list of Chunk objects."""
    page_map = _build_page_map(markdown)
    boundaries = _find_chapter_boundaries(markdown)

    if not boundaries:
        boundaries = [(0, "# Full Text")]

    sections: list[tuple[int, str, str]] = []
    for i, (offset, heading) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(markdown)
        sections.append((offset, heading, markdown[offset:end]))

    book_name = BOOK_DISPLAY_NAMES.get(book_key, book_key)
    chunks: list[Chunk] = []

    for offset, heading, section_text in sections:
        chapter_number = _extract_chapter_number(heading)
        chapter_title = _clean_heading_title(heading)
        start_page = _page_at(page_map, offset)

        if _count_tokens(section_text) <= CHAPTER_TOKEN_LIMIT:
            end_page = _page_at(page_map, offset + len(section_text))
            chunks.append(Chunk(
                book_key=book_key,
                book_name=book_name,
                chapter_number=chapter_number,
                chapter_title=chapter_title,
                chunk_index=0,
                total_chunks=1,
                text=section_text.strip(),
                start_page=start_page,
                end_page=end_page,
            ))
        else:
            sub_texts = _sub_chunk(section_text, CHAPTER_TOKEN_LIMIT, CHUNK_OVERLAP_TOKENS)
            total = len(sub_texts)
            char_cursor = offset
            for idx, sub_text in enumerate(sub_texts):
                end_page = _page_at(page_map, char_cursor + len(sub_text))
                chunks.append(Chunk(
                    book_key=book_key,
                    book_name=book_name,
                    chapter_number=chapter_number,
                    chapter_title=chapter_title,
                    chunk_index=idx,
                    total_chunks=total,
                    text=sub_text.strip(),
                    start_page=_page_at(page_map, char_cursor),
                    end_page=end_page,
                ))
                char_cursor += len(sub_text)

    return chunks


def chunk_all(books: dict[str, str]) -> list[Chunk]:
    """Chunk all books and save to cache."""
    all_chunks: list[Chunk] = []
    for book_key, markdown in books.items():
        book_chunks = chunk_book(book_key, markdown)
        print(f"{BOOK_DISPLAY_NAMES.get(book_key, book_key)}: {len(book_chunks)} chunks")
        for i, c in enumerate(book_chunks[:3]):
            print(f"  chunk[{i}] ch{c.chapter_number} '{c.chapter_title}' "
                  f"pp.{c.start_page}-{c.end_page} ({_count_tokens(c.text)} tok)")
        all_chunks.extend(book_chunks)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / "chunks.pkl"
    with open(cache_file, "wb") as f:
        pickle.dump(all_chunks, f)
    print(f"\nSaved {len(all_chunks)} total chunks → {cache_file}")
    return all_chunks


class _ChunkUnpickler(pickle.Unpickler):
    """Remap __main__.Chunk → ingest.chunker.Chunk regardless of how chunks.pkl was created."""
    def find_class(self, module, name):
        if name == "Chunk":
            return Chunk
        return super().find_class(module, name)


def load_chunks() -> list[Chunk]:
    cache_file = CACHE_DIR / "chunks.pkl"
    if not cache_file.exists():
        raise FileNotFoundError("Run ingest.chunker first to generate chunks.pkl")
    with open(cache_file, "rb") as f:
        return _ChunkUnpickler(f).load()


if __name__ == "__main__":
    from ingest.parser import parse_all
    books = parse_all()
    chunks = chunk_all(books)
    print(f"\nTotal chunks: {len(chunks)}")
