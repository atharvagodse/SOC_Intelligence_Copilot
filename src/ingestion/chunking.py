"""
src/ingestion/chunking.py
=========================
Text cleaning and chunking for the SOC Intelligence Copilot.

This module sits between the loaders (Phase 2) and the embeddings (Phase 4):

    LoadedPage  →  TextCleaner  →  DocumentChunker  →  List[DocumentChunk]

Two responsibilities are kept separate on purpose:

  TextCleaner     : Pure text transformation — no splitting.
                    Removes PDF artefacts, normalises whitespace, strips
                    running headers/footers, and handles Unicode ligatures.
                    Every cleaning step is a separate method so it can be
                    tested individually and switched on/off in config.

  DocumentChunker : Splits clean text into overlapping chunks using
                    LangChain's RecursiveCharacterTextSplitter.
                    Preserves full metadata from the source LoadedPage and
                    adds chunk-specific fields (chunk_id, chunk_index).

Design note — why RecursiveCharacterTextSplitter?
  It tries to split on paragraph breaks first (\\n\\n), then on single
  newlines, then on sentence boundaries, and finally on individual words.
  This keeps semantically related sentences together as long as possible,
  which improves retrieval quality significantly over naive fixed-size splits.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.config import settings
from src.ingestion.loaders import DocumentLoaderFactory, LoadedPage

logger = logging.getLogger("soc_copilot.chunking")


# ---------------------------------------------------------------------------
# Data model: a single chunk ready for embedding
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    """
    A cleaned, size-bounded piece of a source document.

    This is the fundamental unit that gets embedded and stored in ChromaDB.

    Attributes
    ----------
    text         : Cleaned chunk text — ready to embed.
    chunk_id     : Globally unique identifier: ``<file_hash>-<chunk_index>``.
    chunk_index  : 0-based position of this chunk within its source document.
    total_chunks : Total number of chunks produced from the source document.
    char_count   : Character count of the chunk text (useful for diagnostics).
    source       : Absolute path to the original file.
    doc_name     : Human-readable file name.
    doc_type     : 'pdf', 'docx', 'txt', or 'json'.
    page_number  : Source page this chunk came from (best-effort for PDFs).
    total_pages  : Total pages in the source document.
    upload_date  : ISO-8601 timestamp from the loader.
    file_hash    : SHA-256 of the source file (for duplicate detection).
    extra        : Any loader-specific extra metadata.
    """
    text: str
    chunk_id: str
    chunk_index: int
    total_chunks: int
    char_count: int
    source: str
    doc_name: str
    doc_type: str
    page_number: int
    total_pages: int
    upload_date: str
    file_hash: str
    extra: dict = field(default_factory=dict)

    def to_metadata_dict(self) -> dict:
        """
        Return a flat dict suitable for ChromaDB metadata storage.

        ChromaDB only accepts str, int, float, and bool metadata values —
        no nested dicts.  All fields here comply with that constraint.
        """
        return {
            "chunk_id":    self.chunk_id,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "char_count":  self.char_count,
            "source":      self.source,
            "doc_name":    self.doc_name,
            "doc_type":    self.doc_type,
            "page_number": self.page_number,
            "total_pages": self.total_pages,
            "upload_date": self.upload_date,
            "file_hash":   self.file_hash,
            **{f"extra_{k}": str(v) for k, v in self.extra.items()},
        }


# ---------------------------------------------------------------------------
# Text Cleaner
# ---------------------------------------------------------------------------

class TextCleaner:
    """
    Cleans raw extracted text before chunking.

    Each method handles one specific artefact type so the pipeline is easy
    to audit, test, and extend.

    Call ``clean(text)`` to run the full pipeline in the correct order.

    Cleaning steps (in order):
      1. Unicode normalisation  — convert ligatures (ﬁ → fi) and smart quotes
      2. Remove PDF control chars  — null bytes, soft hyphens, form feeds
      3. Fix soft hyphenation  — "identi-\\nfication" → "identification"
      4. Collapse excessive newlines  — 3+ blank lines → 2 (a paragraph break)
      5. Strip running headers/footers  — repeating short lines like
         "NIST SP 800-61r3" and date stamps ("April 2025")
      6. Normalise whitespace  — tabs → space, multiple spaces → single
      7. Strip leading/trailing whitespace per line
    """

    # Patterns that look like PDF running headers / footers
    # These short repeated lines appear on every page and add noise.
    _HEADER_FOOTER_PATTERNS: list[re.Pattern] = [
        # "NIST SP 800-61r3" style document codes
        re.compile(r'(?m)^NIST\s+SP\s+\d{3}[-–]\d+\w*\s*$'),
        # Month + year on its own line e.g. "April 2025", "January 2024"
        re.compile(
            r'(?m)^(January|February|March|April|May|June|July|August|'
            r'September|October|November|December)\s+\d{4}\s*$'
        ),
        # Page number lines e.g. "Page 1 of 48", "1 | P a g e", bare "i", "ii"
        re.compile(r'(?m)^\s*(Page\s+\d+\s+of\s+\d+|\d+\s*\|\s*P\s*a\s*g\s*e|[ivxlcdm]+)\s*$'),
        # "This publication is available free of charge from:" boilerplate
        re.compile(
            r'(?m)^This publication is available free of charge from:\s*\nhttps?://\S+\s*$',
            re.MULTILINE
        ),
    ]

    # PDF Unicode ligatures that text extractors sometimes produce
    _LIGATURE_MAP: dict[str, str] = {
        '\ufb00': 'ff', '\ufb01': 'fi', '\ufb02': 'fl',
        '\ufb03': 'ffi', '\ufb04': 'ffl', '\ufb05': 'st', '\ufb06': 'st',
        '\u2019': "'",  '\u2018': "'",   # curly apostrophes
        '\u201c': '"',  '\u201d': '"',   # curly double quotes
        '\u2013': '-',  '\u2014': '-',   # en/em dash → hyphen
        '\u00a0': ' ',                    # non-breaking space → regular space
        '\u00ad': '',                     # soft hyphen → remove
    }

    # Minimum meaningful chunk length (characters)
    MIN_MEANINGFUL_LENGTH: int = 20

    def clean(self, text: str) -> str:
        """Run all cleaning steps in order and return the cleaned text."""
        text = self._normalise_unicode(text)
        text = self._remove_control_characters(text)
        text = self._fix_soft_hyphenation(text)
        text = self._strip_header_footer_lines(text)
        text = self._collapse_newlines(text)
        text = self._normalise_whitespace(text)
        return text.strip()

    def _normalise_unicode(self, text: str) -> str:
        """
        NFC-normalise text and replace known ligature characters.

        NFC (Canonical Decomposition, followed by Canonical Composition)
        ensures that accented characters like 'é' are stored as one code
        point, not as 'e' + combining accent, which would confuse tokenisers.
        """
        text = unicodedata.normalize("NFC", text)
        for ligature, replacement in self._LIGATURE_MAP.items():
            text = text.replace(ligature, replacement)
        return text

    def _remove_control_characters(self, text: str) -> str:
        """
        Remove non-printable control characters.

        Keeps: normal ASCII, tabs (\\t), newlines (\\n), carriage returns (\\r).
        Removes: null bytes, form feeds, vertical tabs, DEL, etc.
        """
        # Allow printable chars + whitespace
        return re.sub(r'[^\x09\x0a\x0d\x20-\x7e\x80-\xff\u0100-\ufffd]', '', text)

    def _fix_soft_hyphenation(self, text: str) -> str:
        """
        Re-join words broken across lines by PDF hyphenation.

        Transforms:
            "identi-\\nfication"  →  "identification"
            "recom-\\n  mend"     →  "recommend"
        """
        # Pattern: letter, hyphen at end of line, optional whitespace, letter
        return re.sub(r'(\w)-\n\s*(\w)', r'\1\2', text)

    def _strip_header_footer_lines(self, text: str) -> str:
        """Remove lines that match known header/footer patterns."""
        for pattern in self._HEADER_FOOTER_PATTERNS:
            text = pattern.sub('', text)
        return text

    def _collapse_newlines(self, text: str) -> str:
        """
        Collapse 3 or more consecutive blank lines into 2 (a paragraph break).

        This preserves paragraph structure (double newline) while removing
        the visual spacing artefacts that PDFs introduce.
        """
        # Three or more newlines → double newline (paragraph separator)
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Lines that are only whitespace → empty lines
        text = re.sub(r'(?m)^[ \t]+$', '', text)
        return text

    def _normalise_whitespace(self, text: str) -> str:
        """
        Normalise whitespace within lines.

        - Tabs → single space
        - Multiple consecutive spaces → single space
        - Strips leading/trailing whitespace from each line
        """
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.replace('\t', ' ')
            line = re.sub(r' {2,}', ' ', line)
            line = line.strip()
            cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)

    def is_meaningful(self, text: str) -> bool:
        """
        Return True if the text is long enough to be worth embedding.

        Very short fragments (page numbers, stray characters) should be
        dropped rather than stored as low-quality chunks.
        """
        return len(text.strip()) >= self.MIN_MEANINGFUL_LENGTH


# ---------------------------------------------------------------------------
# Document Chunker
# ---------------------------------------------------------------------------

class DocumentChunker:
    """
    Splits a list of LoadedPages into DocumentChunks using LangChain's
    RecursiveCharacterTextSplitter.

    Why RecursiveCharacterTextSplitter?
    ------------------------------------
    It tries a hierarchy of separators:
        ["\\n\\n", "\\n", ". ", " ", ""]

    This means it will:
      1. First try to split on paragraph breaks (\\n\\n)
      2. Then on single newlines
      3. Then on sentence boundaries (". ")
      4. Then on word boundaries (" ")
      5. Only split mid-word as a last resort

    Result: chunks respect natural language boundaries, which produces
    better semantic embeddings compared to fixed-size character splits.

    Parameters
    ----------
    chunk_size    : Max characters per chunk (default from settings).
    chunk_overlap : Characters to repeat between adjacent chunks (default from settings).
    cleaner       : TextCleaner instance (creates a default one if not given).
    """

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
        cleaner: Optional[TextCleaner] = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.cleaner = cleaner or TextCleaner()

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            # Prefer natural language boundaries
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
            is_separator_regex=False,
        )
        logger.debug(
            "DocumentChunker ready: chunk_size=%d, overlap=%d",
            chunk_size, chunk_overlap,
        )

    def chunk_pages(self, pages: list[LoadedPage]) -> list[DocumentChunk]:
        """
        Clean and chunk a list of LoadedPages.

        Workflow per page:
          1. Clean the raw text with TextCleaner.
          2. Skip pages whose cleaned text is too short to be meaningful.
          3. Split with RecursiveCharacterTextSplitter.
          4. Wrap each split into a DocumentChunk with full metadata.

        Parameters
        ----------
        pages : Output of a DocumentLoader — one item per PDF page / paragraph.

        Returns
        -------
        List of DocumentChunk objects, indexed sequentially across all pages.
        """
        if not pages:
            logger.warning("chunk_pages() received an empty page list.")
            return []

        all_chunks: list[DocumentChunk] = []
        global_chunk_index = 0

        for page in pages:
            cleaned = self.cleaner.clean(page.text)

            if not self.cleaner.is_meaningful(cleaned):
                logger.debug(
                    "Page %d of '%s' too short after cleaning (%d chars) — skipped.",
                    page.page_number, page.doc_name, len(cleaned),
                )
                continue

            # Split the cleaned page text into sub-chunks
            raw_splits = self._splitter.split_text(cleaned)

            # Filter out any splits that are still too short after splitting
            splits = [s.strip() for s in raw_splits if self.cleaner.is_meaningful(s.strip())]

            for split_text in splits:
                chunk = DocumentChunk(
                    text=split_text,
                    chunk_id=f"{page.file_hash}-{global_chunk_index:05d}",
                    chunk_index=global_chunk_index,
                    total_chunks=0,       # will be back-filled below
                    char_count=len(split_text),
                    source=page.source,
                    doc_name=page.doc_name,
                    doc_type=page.doc_type,
                    page_number=page.page_number,
                    total_pages=page.total_pages,
                    upload_date=page.upload_date,
                    file_hash=page.file_hash,
                    extra=page.extra_metadata,
                )
                all_chunks.append(chunk)
                global_chunk_index += 1

        # Back-fill total_chunks now that we know the final count
        total = len(all_chunks)
        for chunk in all_chunks:
            chunk.total_chunks = total

        logger.info(
            "'%s' → %d pages → %d chunks (size≤%d, overlap=%d).",
            pages[0].doc_name if pages else "?",
            len(pages),
            total,
            self.chunk_size,
            self.chunk_overlap,
        )
        return all_chunks

    def chunk_file(self, path: Path) -> list[DocumentChunk]:
        """
        Convenience method: load a file then chunk it in one call.

        Parameters
        ----------
        path : Path to any supported document file.

        Returns
        -------
        List of DocumentChunk objects ready for embedding.
        """
        pages = DocumentLoaderFactory.load(Path(path))
        return self.chunk_pages(pages)

    def chunk_directory(
        self,
        directory: Path,
        recursive: bool = True,
    ) -> list[DocumentChunk]:
        """
        Load every supported document in a directory and chunk them all.

        Parameters
        ----------
        directory : Root folder to scan.
        recursive : Descend into sub-folders when True (default).

        Returns
        -------
        Combined and sequentially-indexed list of DocumentChunks.
        """
        pages = DocumentLoaderFactory.load_directory(
            Path(directory), recursive=recursive
        )
        return self.chunk_pages(pages)


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def chunk_document(
    path: Path,
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> list[DocumentChunk]:
    """
    One-liner: load, clean, chunk — return DocumentChunks.

    Suitable for quick scripts and Streamlit callbacks.

    Example
    -------
    ::
        from src.ingestion.chunking import chunk_document
        chunks = chunk_document(Path("data/raw/NIST_SP_800-61.pdf"))
        print(f"Generated {len(chunks)} chunks")
        print(chunks[0].text[:200])
    """
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_file(path)
