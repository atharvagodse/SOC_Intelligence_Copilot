"""
src/ingestion/chunking.py
=========================
Text cleaning and document chunking for the SOC Intelligence Copilot.

Why this module exists:
  1. Raw text from PDFs often contains extraction artifacts (ligatures like
     'ﬁ', broken hyphenated words across lines, excessive whitespace, control
     characters).  ``clean_text()`` normalises these without altering meaning.
  2. LLMs and embedding models have context limits.  ``DocumentChunker``
     splits each ``LoadedPage`` into overlapping ``DocumentChunk`` objects
     using LangChain's ``RecursiveCharacterTextSplitter``.

Each ``DocumentChunk`` carries a deterministic ``chunk_id`` derived from the
source file's SHA-256 hash, page number, and chunk index, plus full metadata
for ChromaDB storage and source citation.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.ingestion.loaders import LoadedPage

logger = logging.getLogger("soc_copilot.chunking")


# ---------------------------------------------------------------------------
# Data model: a single chunk ready for embedding & vector storage
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    """
    A cleaned, size-bounded piece of text with full provenance metadata.

    Attributes
    ----------
    chunk_id      : Deterministic unique identifier for vector DB upsert/dedup.
    text          : Cleaned chunk text.
    doc_name      : Original file name (e.g. 'NIST_SP_800-61_Rev3.pdf').
    doc_type      : File type ('pdf', 'docx', 'txt', 'json').
    source        : Resolved file path.
    page_number   : 1-based page/section number from the loader.
    total_pages   : Total pages/sections in the source document.
    chunk_index   : 0-based index of this chunk within its page.
    upload_date   : ISO-8601 UTC timestamp of ingestion.
    file_hash     : SHA-256 hex digest of the source file.
    extra_metadata: Optional extra fields (kept flat for ChromaDB).
    """

    chunk_id: str
    text: str
    doc_name: str
    doc_type: str
    source: str
    page_number: int
    total_pages: int
    chunk_index: int
    upload_date: str
    file_hash: str
    extra_metadata: dict = field(default_factory=dict)

    def to_metadata_dict(self) -> dict[str, str | int | float | bool]:
        """
        Return flat metadata suitable for ChromaDB.

        ChromaDB requires metadata values to be str, int, float, or bool.
        """
        meta: dict[str, str | int | float | bool] = {
            "chunk_id": self.chunk_id,
            "doc_name": self.doc_name,
            "doc_type": self.doc_type,
            "source": self.source,
            "page_number": self.page_number,
            "total_pages": self.total_pages,
            "chunk_index": self.chunk_index,
            "upload_date": self.upload_date,
            "file_hash": self.file_hash,
        }
        for key, value in self.extra_metadata.items():
            if isinstance(value, (str, int, float, bool)):
                meta[key] = value
            else:
                meta[key] = str(value)
        return meta


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

# Pre-compiled regexes for speed on large documents
_RE_HYPHEN_LINEBREAK = re.compile(r"([A-Za-z])-\s*\n\s*([A-Za-z])")
_RE_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_RE_HORIZONTAL_WS = re.compile(r"[^\S\n]+")
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")


def clean_text(raw_text: str) -> str:
    """
    Normalise and clean extracted document text.

    Steps:
      1. Unicode NFKC normalisation (turns ligatures like 'ﬁ' into 'fi').
      2. Remove null bytes and non-printable control characters.
      3. Normalise CRLF/CR line endings to LF.
      4. Re-join words hyphenated across line breaks (e.g. 'cyber-\\nsecurity').
      5. Collapse repeated spaces/tabs on a line to a single space.
      6. Collapse 3+ consecutive newlines to 2 (preserving paragraph breaks).
      7. Strip leading/trailing whitespace on each line and overall.

    Security: treats input strictly as plain text; never interprets content.
    """
    if not raw_text:
        return ""

    text = unicodedata.normalize("NFKC", raw_text)
    text = _RE_CONTROL_CHARS.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _RE_HYPHEN_LINEBREAK.sub(r"\1\2", text)
    text = _RE_HORIZONTAL_WS.sub(" ", text)
    # Strip trailing/leading spaces on individual lines
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _RE_MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

def build_chunk_id(file_hash: str, page_number: int, chunk_index: int) -> str:
    """
    Create a deterministic chunk ID from file hash, page, and chunk index.

    Using the file hash ensures that re-ingesting the exact same file yields
    identical IDs so ChromaDB upserts instead of creating duplicates.
    """
    short_hash = file_hash[:12] if file_hash else "nohash"
    return f"{short_hash}_p{page_number}_c{chunk_index}"


class DocumentChunker:
    """
    Cleans and splits ``LoadedPage`` objects into ``DocumentChunk`` objects.

    Uses LangChain's ``RecursiveCharacterTextSplitter`` which tries to split
    on paragraph boundaries first, then newlines, then sentences, then words,
    keeping semantically related text together.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        min_chunk_chars: int = 20,
        clean: bool = True,
    ) -> None:
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self.chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        )
        self.min_chunk_chars = min_chunk_chars
        self.clean = clean

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be >= 0 and strictly less than chunk_size."
            )

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk_page(self, page: LoadedPage) -> list[DocumentChunk]:
        """Clean and split a single ``LoadedPage`` into ``DocumentChunk``s."""
        text = clean_text(page.text) if self.clean else page.text.strip()
        if not text or len(text) < self.min_chunk_chars:
            return []

        raw_chunks = self._splitter.split_text(text)
        chunks: list[DocumentChunk] = []

        for idx, chunk_text in enumerate(raw_chunks):
            stripped = chunk_text.strip()
            if len(stripped) < self.min_chunk_chars:
                continue

            chunk_id = build_chunk_id(page.file_hash, page.page_number, idx)
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=stripped,
                    doc_name=page.doc_name,
                    doc_type=page.doc_type,
                    source=page.source,
                    page_number=page.page_number,
                    total_pages=page.total_pages,
                    chunk_index=idx,
                    upload_date=page.upload_date,
                    file_hash=page.file_hash,
                    extra_metadata=dict(page.extra_metadata),
                )
            )

        return chunks

    def chunk_pages(self, pages: Iterable[LoadedPage]) -> list[DocumentChunk]:
        """Clean and split an iterable of ``LoadedPage`` objects."""
        all_chunks: list[DocumentChunk] = []
        page_count = 0

        for page in pages:
            page_count += 1
            all_chunks.extend(self.chunk_page(page))

        logger.info(
            "Chunked %d pages/sections → %d chunks (size=%d, overlap=%d).",
            page_count,
            len(all_chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return all_chunks
