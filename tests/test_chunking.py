"""
tests/test_chunking.py
======================
Phase 3 — Tests for text cleaning and document chunking.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.chunking import (
    DocumentChunk,
    DocumentChunker,
    build_chunk_id,
    clean_text,
)
from src.ingestion.loaders import DocumentLoaderFactory, LoadedPage


def _make_page(
    text: str,
    page_number: int = 1,
    file_hash: str = "a" * 64,
) -> LoadedPage:
    return LoadedPage(
        text=text,
        source="/tmp/sample.pdf",
        doc_name="sample.pdf",
        doc_type="pdf",
        page_number=page_number,
        total_pages=1,
        upload_date="2026-11-27T00:00:00+00:00",
        file_hash=file_hash,
    )


class TestCleanText:
    def test_empty_string(self) -> None:
        assert clean_text("") == ""

    def test_normalises_ligatures(self) -> None:
        assert clean_text("ﬁrewall ﬂow") == "firewall flow"

    def test_rejoins_hyphenated_linebreaks(self) -> None:
        raw = "cyber-\nsecurity incident re-\nsponse"
        assert clean_text(raw) == "cybersecurity incident response"

    def test_collapses_whitespace_and_preserves_paragraphs(self) -> None:
        raw = "Line   one\t\there.\r\n\r\n\r\n\r\nLine   two."
        assert clean_text(raw) == "Line one here.\n\nLine two."

    def test_strips_control_characters(self) -> None:
        raw = "Alert\x00 detected\x07 on host."
        assert clean_text(raw) == "Alert detected on host."


class TestDocumentChunker:
    def test_invalid_chunk_params_raise(self) -> None:
        with pytest.raises(ValueError):
            DocumentChunker(chunk_size=0)
        with pytest.raises(ValueError):
            DocumentChunker(chunk_size=100, chunk_overlap=100)
        with pytest.raises(ValueError):
            DocumentChunker(chunk_size=100, chunk_overlap=-1)

    def test_chunk_id_is_deterministic(self) -> None:
        id1 = build_chunk_id("abcdef1234567890", page_number=3, chunk_index=2)
        id2 = build_chunk_id("abcdef1234567890", page_number=3, chunk_index=2)
        assert id1 == id2 == "abcdef123456_p3_c2"

    def test_splits_long_page_with_metadata(self) -> None:
        long_text = (
            "Incident response containment isolates affected hosts. " * 25
        )
        page = _make_page(long_text, page_number=4)
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=40)
        chunks = chunker.chunk_page(page)

        assert len(chunks) > 1
        assert all(isinstance(c, DocumentChunk) for c in chunks)
        assert all(len(c.text) <= 200 for c in chunks)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

        meta = chunks[0].to_metadata_dict()
        assert meta["doc_name"] == "sample.pdf"
        assert meta["page_number"] == 4
        assert meta["chunk_id"] == chunks[0].chunk_id
        assert meta["file_hash"] == page.file_hash

    def test_skips_tiny_fragments(self) -> None:
        page = _make_page("Page 1")
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=20, min_chunk_chars=20)
        assert chunker.chunk_page(page) == []

    def test_chunk_real_nist_pdf_sample(self) -> None:
        nist_pdf = (
            PROJECT_ROOT
            / "data"
            / "raw"
            / "incident_response"
            / "NIST_SP_800-61_Rev3.pdf"
        )
        if not nist_pdf.exists():
            pytest.skip("NIST PDF not found")

        pages = DocumentLoaderFactory.load(nist_pdf)[:3]
        chunker = DocumentChunker()
        chunks = chunker.chunk_pages(pages)
        assert len(chunks) >= len(pages)
        assert all(c.doc_name == "NIST_SP_800-61_Rev3.pdf" for c in chunks)
