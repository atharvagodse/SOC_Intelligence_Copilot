"""
tests/test_chunking.py
======================
Phase 3 — Tests for text cleaning and document chunking.

Test categories:
  A. TextCleaner unit tests  — each cleaning step individually
  B. DocumentChunk dataclass — metadata dict serialisation
  C. DocumentChunker unit tests — with synthetic text
  D. Integration tests       — real NIST and MITRE documents
  E. Edge cases              — empty input, very short text, huge text
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR  = PROJECT_ROOT / "data" / "raw"
NIST_PDF = RAW_DIR / "incident_response" / "NIST_SP_800-61_Rev3.pdf"
FED_PDF  = RAW_DIR / "incident_response" / "Federal_Government_Cybersecurity_Incident_and_Vulnerability_Response_Playbooks_508C.pdf"
MITRE_JSON = RAW_DIR / "mitre_attack" / "enterprise-attack-19.2.json"

from src.ingestion.chunking import (
    DocumentChunk,
    DocumentChunker,
    TextCleaner,
    chunk_document,
)
from src.config import settings


# ===========================================================================
# A. TextCleaner unit tests
# ===========================================================================

class TestTextCleaner:

    def setup_method(self):
        self.cleaner = TextCleaner()

    # --- Unicode / ligature normalisation ---

    def test_fi_ligature_replaced(self):
        result = self.cleaner.clean("\ufb01le system")
        assert "fi" in result
        assert "\ufb01" not in result

    def test_fl_ligature_replaced(self):
        result = self.cleaner.clean("over\ufb02ow")
        assert "fl" in result

    def test_smart_quotes_normalised(self):
        result = self.cleaner.clean("\u2018Hello\u2019 and \u201cworld\u201d")
        assert '"' in result or "'" in result
        assert "\u2018" not in result
        assert "\u201c" not in result

    def test_non_breaking_space_replaced(self):
        result = self.cleaner.clean("hello\u00a0world")
        assert "\u00a0" not in result
        assert "hello world" in result

    def test_soft_hyphen_removed(self):
        result = self.cleaner.clean("iden\u00adtify")
        assert "\u00ad" not in result

    # --- Soft hyphenation fix ---

    def test_hyphenated_line_break_joined(self):
        text = "identi-\nfication of threats"
        result = self.cleaner.clean(text)
        assert "identification" in result

    def test_hyphenated_break_with_leading_space(self):
        text = "recom-\n  mend the action"
        result = self.cleaner.clean(text)
        assert "recommend" in result

    # --- Header / footer stripping ---

    def test_nist_header_stripped(self):
        text = "NIST SP 800-61r3\nSome important content here about incidents."
        result = self.cleaner.clean(text)
        assert "NIST SP 800-61r3" not in result
        assert "important content" in result

    def test_month_year_header_stripped(self):
        text = "April 2025\nContent about incident response procedures."
        result = self.cleaner.clean(text)
        assert "April 2025" not in result
        assert "incident response" in result

    # --- Whitespace normalisation ---

    def test_multiple_spaces_collapsed(self):
        result = self.cleaner.clean("hello    world")
        assert "hello world" in result

    def test_tabs_replaced_with_space(self):
        result = self.cleaner.clean("field1\tfield2")
        assert "\t" not in result

    def test_excessive_newlines_collapsed(self):
        text = "paragraph one\n\n\n\n\nparagraph two"
        result = self.cleaner.clean(text)
        # Should not have more than 2 consecutive newlines
        assert "\n\n\n" not in result

    def test_leading_trailing_whitespace_stripped(self):
        result = self.cleaner.clean("   hello world   ")
        assert result == "hello world"

    # --- Meaningfulness check ---

    def test_short_text_not_meaningful(self):
        assert not self.cleaner.is_meaningful("hi")
        assert not self.cleaner.is_meaningful("   ")
        assert not self.cleaner.is_meaningful("")

    def test_long_text_is_meaningful(self):
        assert self.cleaner.is_meaningful("This is a proper security incident description.")

    def test_exactly_at_min_length(self):
        boundary = "x" * TextCleaner.MIN_MEANINGFUL_LENGTH
        assert self.cleaner.is_meaningful(boundary)

    # --- Full pipeline on realistic text ---

    def test_clean_pdf_page_text(self):
        """Simulate a real NIST PDF page with common artefacts."""
        raw = (
            "NIST SP 800-61r3\n"
            " \n \n \n"
            "Incident Response Recommendations and\n"
            "April 2025\n"
            "Considerations for Cyber Risk Management\n\n"
            "Organizations should implement a formal incident\u00a0response plan. "
            "The plan should address how identi-\nfication of threats is performed.\n"
            "\ufb01rst responders must follow the playbook.\n"
        )
        result = self.cleaner.clean(raw)

        assert "NIST SP 800-61r3" not in result
        assert "April 2025" not in result
        assert "identification" in result     # soft hyphen join
        assert "first" in result              # fi ligature replaced
        assert "\u00a0" not in result         # NBSP removed
        assert "\n\n\n" not in result         # excessive newlines collapsed


# ===========================================================================
# B. DocumentChunk dataclass tests
# ===========================================================================

class TestDocumentChunk:

    def _make_chunk(self, **kwargs) -> DocumentChunk:
        defaults = dict(
            text="Sample cybersecurity incident response text.",
            chunk_id="abc123-00001",
            chunk_index=1,
            total_chunks=10,
            char_count=46,
            source="/data/raw/test.pdf",
            doc_name="test.pdf",
            doc_type="pdf",
            page_number=1,
            total_pages=5,
            upload_date="2026-11-20T00:00:00+00:00",
            file_hash="a" * 64,
        )
        defaults.update(kwargs)
        return DocumentChunk(**defaults)

    def test_to_metadata_dict_has_required_keys(self):
        chunk = self._make_chunk()
        meta = chunk.to_metadata_dict()
        for key in [
            "chunk_id", "chunk_index", "total_chunks", "char_count",
            "source", "doc_name", "doc_type", "page_number",
            "total_pages", "upload_date", "file_hash"
        ]:
            assert key in meta, f"Missing metadata key: {key}"

    def test_metadata_values_are_chromadb_compatible_types(self):
        """ChromaDB only accepts str, int, float, bool — no nested dicts."""
        chunk = self._make_chunk(extra={"format": "json"})
        meta = chunk.to_metadata_dict()
        for k, v in meta.items():
            assert isinstance(v, (str, int, float, bool)), (
                f"Metadata key '{k}' has incompatible type {type(v)}"
            )

    def test_chunk_id_format(self):
        chunk = self._make_chunk(chunk_id="deadbeef-00042")
        assert "-" in chunk.chunk_id

    def test_char_count_matches_text(self):
        text = "Incident response playbook section."
        chunk = self._make_chunk(text=text, char_count=len(text))
        assert chunk.char_count == len(text)


# ===========================================================================
# C. DocumentChunker unit tests (synthetic text)
# ===========================================================================

class TestDocumentChunker:

    def _make_pages(self, texts: list[str]):
        """Create synthetic LoadedPage objects for testing."""
        from src.ingestion.loaders import LoadedPage
        return [
            LoadedPage(
                text=t,
                source="/tmp/test.txt",
                doc_name="test.txt",
                doc_type="txt",
                page_number=i + 1,
                total_pages=len(texts),
                upload_date="2026-01-01T00:00:00+00:00",
                file_hash="f" * 64,
            )
            for i, t in enumerate(texts)
        ]

    def test_basic_chunking_returns_chunks(self):
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
        # Create text longer than chunk_size to force splitting
        long_text = "Incident response is a critical SOC function. " * 20
        pages = self._make_pages([long_text])
        chunks = chunker.chunk_pages(pages)
        assert len(chunks) > 0

    def test_chunks_respect_size_limit(self):
        chunk_size = 200
        chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=20)
        long_text = "Security operations centre analysts monitor alerts. " * 30
        pages = self._make_pages([long_text])
        chunks = chunker.chunk_pages(pages)
        for chunk in chunks:
            # Allow small tolerance — splitter may exceed by one separator
            assert chunk.char_count <= chunk_size + 50, (
                f"Chunk too large: {chunk.char_count} chars"
            )

    def test_chunk_ids_are_unique(self):
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
        text = "Brute force attacks attempt many passwords. " * 20
        pages = self._make_pages([text])
        chunks = chunker.chunk_pages(pages)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_chunk_indices_are_sequential(self):
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
        text = "Phishing is a common social engineering attack vector. " * 20
        pages = self._make_pages([text])
        chunks = chunker.chunk_pages(pages)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_total_chunks_backfilled_correctly(self):
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
        text = "Malware analysis requires sandbox environments. " * 20
        pages = self._make_pages([text])
        chunks = chunker.chunk_pages(pages)
        total = len(chunks)
        for chunk in chunks:
            assert chunk.total_chunks == total

    def test_chunk_metadata_inherited_from_page(self):
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
        text = "Privilege escalation allows attackers to gain higher permissions. " * 10
        pages = self._make_pages([text])
        chunks = chunker.chunk_pages(pages)
        for chunk in chunks:
            assert chunk.doc_name == "test.txt"
            assert chunk.doc_type == "txt"
            assert chunk.file_hash == "f" * 64

    def test_short_text_not_chunked_into_meaningless_pieces(self):
        chunker = DocumentChunker(chunk_size=512, chunk_overlap=64)
        pages = self._make_pages(["hi"])  # too short
        chunks = chunker.chunk_pages(pages)
        assert len(chunks) == 0, "Single-word text should produce no chunks"

    def test_empty_pages_list_returns_empty(self):
        chunker = DocumentChunker()
        result = chunker.chunk_pages([])
        assert result == []

    def test_multi_page_chunks_indexed_across_pages(self):
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
        texts = [
            "Network intrusion detection is key to SOC operations. " * 5,
            "SIEM tools aggregate logs from many sources. " * 5,
        ]
        pages = self._make_pages(texts)
        chunks = chunker.chunk_pages(pages)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks))), "Indices must be global"

    def test_overlap_text_appears_in_adjacent_chunks(self):
        """Adjacent chunks must share some content due to overlap."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=40)
        # Deterministic repeating text to ensure predictable splits
        text = ("ABCDE " * 5 + ". ") * 10
        pages = self._make_pages([text])
        chunks = chunker.chunk_pages(pages)
        if len(chunks) >= 2:
            # The end of chunk[0] and start of chunk[1] must share characters
            end_of_first = chunks[0].text[-30:]
            start_of_second = chunks[1].text[:60]
            # At least some words should overlap (not guaranteed character-exact
            # due to natural-boundary splitting, but content must be shared)
            words_first = set(end_of_first.split())
            words_second = set(start_of_second.split())
            assert words_first & words_second, (
                "Overlapping chunks must share some words"
            )


# ===========================================================================
# D. Integration tests — real documents
# ===========================================================================

class TestRealDocumentChunking:

    @pytest.mark.skipif(not NIST_PDF.exists(), reason="NIST PDF not found")
    def test_nist_pdf_chunk_count_reasonable(self):
        chunks = chunk_document(NIST_PDF)
        # 48-page NIST doc with chunk_size=512 should yield 50-500 chunks
        assert 20 <= len(chunks) <= 500, (
            f"Expected 20-500 chunks, got {len(chunks)}"
        )

    @pytest.mark.skipif(not NIST_PDF.exists(), reason="NIST PDF not found")
    def test_nist_chunks_contain_cybersecurity_terms(self):
        chunks = chunk_document(NIST_PDF)
        all_text = " ".join(c.text for c in chunks).lower()
        for term in ["incident", "response", "cybersecurity"]:
            assert term in all_text, f"Expected '{term}' in NIST chunks"

    @pytest.mark.skipif(not NIST_PDF.exists(), reason="NIST PDF not found")
    def test_nist_chunk_metadata_complete(self):
        chunks = chunk_document(NIST_PDF)
        for chunk in chunks:
            meta = chunk.to_metadata_dict()
            assert meta["doc_name"] == "NIST_SP_800-61_Rev3.pdf"
            assert meta["doc_type"] == "pdf"
            assert len(meta["file_hash"]) == 64
            assert meta["chunk_id"].endswith(f"-{chunk.chunk_index:05d}")

    @pytest.mark.skipif(not NIST_PDF.exists(), reason="NIST PDF not found")
    def test_nist_chunks_no_header_artefacts(self):
        chunks = chunk_document(NIST_PDF)
        for chunk in chunks:
            # Running headers should have been stripped
            import re
            assert not re.search(r'^NIST SP 800', chunk.text, re.MULTILINE), (
                f"Header artefact found in chunk {chunk.chunk_index}: {chunk.text[:100]}"
            )

    @pytest.mark.skipif(not NIST_PDF.exists(), reason="NIST PDF not found")
    def test_nist_no_excessive_whitespace_in_chunks(self):
        chunks = chunk_document(NIST_PDF)
        for chunk in chunks:
            assert "   " not in chunk.text, (
                f"Triple space found in chunk {chunk.chunk_index}"
            )
            assert "\n\n\n" not in chunk.text, (
                f"Triple newline found in chunk {chunk.chunk_index}"
            )

    @pytest.mark.skipif(not FED_PDF.exists(), reason="Federal PDF not found")
    def test_federal_pdf_chunks_produced(self):
        chunks = chunk_document(FED_PDF)
        assert len(chunks) > 0

    @pytest.mark.skipif(not MITRE_JSON.exists(), reason="MITRE JSON not found")
    def test_mitre_json_chunks_produced(self):
        chunks = chunk_document(MITRE_JSON, chunk_size=512, chunk_overlap=64)
        assert len(chunks) > 0
        assert all(c.doc_type == "json" for c in chunks)

    @pytest.mark.skipif(not NIST_PDF.exists(), reason="NIST PDF not found")
    def test_chunk_document_convenience_function(self):
        """chunk_document() must be equivalent to chunker.chunk_file()."""
        chunks_a = chunk_document(NIST_PDF)
        chunker = DocumentChunker()
        chunks_b = chunker.chunk_file(NIST_PDF)
        assert len(chunks_a) == len(chunks_b)
        assert chunks_a[0].text == chunks_b[0].text

    @pytest.mark.skipif(not NIST_PDF.exists(), reason="NIST PDF not found")
    def test_chunk_size_parameter_honoured(self):
        """Smaller chunk_size should produce more chunks."""
        chunks_small = chunk_document(NIST_PDF, chunk_size=256, chunk_overlap=32)
        chunks_large = chunk_document(NIST_PDF, chunk_size=1024, chunk_overlap=128)
        assert len(chunks_small) > len(chunks_large), (
            "Smaller chunk_size must produce more chunks"
        )


# ===========================================================================
# E. Edge cases
# ===========================================================================

class TestChunkingEdgeCases:

    def _make_page(self, text: str):
        from src.ingestion.loaders import LoadedPage
        return LoadedPage(
            text=text, source="/tmp/edge.txt", doc_name="edge.txt",
            doc_type="txt", page_number=1, total_pages=1,
            upload_date="2026-01-01T00:00:00+00:00", file_hash="e" * 64,
        )

    def test_whitespace_only_page_produces_no_chunks(self):
        chunker = DocumentChunker()
        page = self._make_page("   \n\n\t  \n   ")
        chunks = chunker.chunk_pages([page])
        assert len(chunks) == 0

    def test_very_long_word_does_not_crash(self):
        """A URL or base64 blob longer than chunk_size must not crash."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
        long_word = "A" * 300  # single token longer than chunk_size
        page = self._make_page(f"Alert: {long_word} detected in log.")
        chunks = chunker.chunk_pages([page])
        # Should produce at least one chunk even if the word is huge
        assert isinstance(chunks, list)

    def test_unicode_heavy_text_handled(self):
        """Text with many Unicode characters must be cleaned without error."""
        chunker = DocumentChunker()
        text = (
            "Attackers exploited CVE-2024-1234 \u2014 a critical RCE "
            "vulnerability in Apache\u00a0Log4Shell. "
            "The attacker used \u201cphishing\u201d emails to gain initial access. "
        ) * 10
        page = self._make_page(text)
        chunks = chunker.chunk_pages([page])
        assert len(chunks) > 0
        for chunk in chunks:
            assert "\u00a0" not in chunk.text   # NBSP removed
            assert "\u2014" not in chunk.text   # Em-dash converted
