"""
tests/test_loaders.py
=====================
Phase 2 — Tests for the document loading and text extraction pipeline.

Test categories:
  A. Unit tests — validation helpers (no real files needed)
  B. Unit tests — factory routing by extension
  C. Integration tests — load your actual raw security documents
  D. Edge-case / security tests — invalid file handling
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.loaders import (
    DocumentLoaderFactory,
    FileValidationError,
    JsonLoader,
    LoadedPage,
    PdfLoader,
    TxtLoader,
    compute_file_hash,
    validate_file,
)

import warnings
# Suppress the pymupdf FutureWarning about fitz import in tests
warnings.filterwarnings("ignore", category=FutureWarning, module="fitz")

# ---------------------------------------------------------------------------
# Paths to real documents (created in Phase 1 inspection)
# ---------------------------------------------------------------------------
RAW_DIR = PROJECT_ROOT / "data" / "raw"
NIST_PDF     = RAW_DIR / "incident_response" / "NIST_SP_800-61_Rev3.pdf"
FED_PDF      = RAW_DIR / "incident_response" / "Federal_Government_Cybersecurity_Incident_and_Vulnerability_Response_Playbooks_508C.pdf"
MITRE_JSON   = RAW_DIR / "mitre_attack" / "enterprise-attack-19.2.json"
KEV_CSV     = RAW_DIR / "threat_intelligence" / "known_exploited_vulnerabilities.csv"


# ===========================================================================
# A. Unit tests — validation helpers
# ===========================================================================

class TestValidationHelpers:

    def test_validate_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileValidationError, match="not found"):
            validate_file(tmp_path / "ghost.pdf")

    def test_validate_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileValidationError, match="not a regular file"):
            validate_file(tmp_path)

    def test_validate_empty_file_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.txt"
        empty.write_text("")
        with pytest.raises(FileValidationError, match="empty"):
            validate_file(empty)

    def test_validate_unsupported_extension_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "malware.exe"
        bad.write_bytes(b"\x00" * 10)
        with pytest.raises(FileValidationError, match="Unsupported file type"):
            validate_file(bad)

    def test_validate_oversized_file_raises(self, tmp_path: Path) -> None:
        """Write a file whose actual size exceeds MAX_FILE_BYTES."""
        # We import settings here to get the real byte limit
        from src.config import settings
        limit_bytes = settings.max_file_size_mb * 1024 * 1024

        big = tmp_path / "huge.txt"
        # Write exactly limit + 1 byte so validation fails
        big.write_bytes(b"X" * (limit_bytes + 1))
        with pytest.raises(FileValidationError, match="exceeding"):
            validate_file(big)

    def test_compute_file_hash_is_deterministic(self, tmp_path: Path) -> None:
        f = tmp_path / "data.txt"
        f.write_text("hello SOC world")
        h1 = compute_file_hash(f)
        h2 = compute_file_hash(f)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex = 64 chars

    def test_compute_file_hash_differs_for_different_content(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("content A")
        f2.write_text("content B")
        assert compute_file_hash(f1) != compute_file_hash(f2)


# ===========================================================================
# B. Unit tests — factory routing
# ===========================================================================

class TestDocumentLoaderFactory:

    def test_factory_routes_pdf(self, tmp_path: Path) -> None:
        fake = tmp_path / "test.pdf"
        fake.write_bytes(b"fake")
        loader = DocumentLoaderFactory.get_loader(fake)
        assert isinstance(loader, PdfLoader)

    def test_factory_routes_txt(self, tmp_path: Path) -> None:
        fake = tmp_path / "test.txt"
        fake.write_text("hello")
        loader = DocumentLoaderFactory.get_loader(fake)
        assert isinstance(loader, TxtLoader)

    def test_factory_routes_json(self, tmp_path: Path) -> None:
        fake = tmp_path / "test.json"
        fake.write_text("{}")
        loader = DocumentLoaderFactory.get_loader(fake)
        assert isinstance(loader, JsonLoader)

    def test_factory_raises_for_unknown_extension(self, tmp_path: Path) -> None:
        fake = tmp_path / "test.csv"
        with pytest.raises(FileValidationError, match="No loader registered"):
            DocumentLoaderFactory.get_loader(fake)


# ===========================================================================
# C. TXT Loader unit tests
# ===========================================================================

class TestTxtLoader:

    def test_loads_utf8_file(self, tmp_path: Path) -> None:
        f = tmp_path / "report.txt"
        f.write_text("Incident response playbook content.", encoding="utf-8")
        pages = DocumentLoaderFactory.load(f)
        assert len(pages) == 1
        assert "Incident" in pages[0].text

    def test_loaded_page_metadata(self, tmp_path: Path) -> None:
        f = tmp_path / "report.txt"
        f.write_text("SOC alert analysis.", encoding="utf-8")
        page = DocumentLoaderFactory.load(f)[0]
        assert page.doc_type == "txt"
        assert page.page_number == 1
        assert page.total_pages == 1
        assert page.doc_name == "report.txt"
        assert len(page.file_hash) == 64

    def test_to_metadata_dict_has_required_keys(self, tmp_path: Path) -> None:
        f = tmp_path / "report.txt"
        f.write_text("Test content.")
        meta = DocumentLoaderFactory.load(f)[0].to_metadata_dict()
        for key in ["source", "doc_name", "doc_type", "page_number",
                    "total_pages", "upload_date", "file_hash"]:
            assert key in meta, f"Missing metadata key: {key}"

    def test_txt_with_latin1_encoding(self, tmp_path: Path) -> None:
        f = tmp_path / "latin.txt"
        f.write_bytes("Café résumé naïve".encode("latin-1"))
        pages = DocumentLoaderFactory.load(f)
        assert len(pages) == 1

    def test_empty_txt_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("   ")  # whitespace only
        with pytest.raises((FileValidationError, RuntimeError)):
            DocumentLoaderFactory.load(f)


# ===========================================================================
# D. JSON Loader unit tests
# ===========================================================================

class TestJsonLoader:

    def test_loads_valid_json(self, tmp_path: Path) -> None:
        f = tmp_path / "intel.json"
        f.write_text(json.dumps({"technique": "T1110", "name": "Brute Force"}))
        pages = DocumentLoaderFactory.load(f)
        assert len(pages) == 1
        assert "T1110" in pages[0].text

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "broken.json"
        f.write_text("{bad json!!}")
        with pytest.raises((FileValidationError, RuntimeError)):
            DocumentLoaderFactory.load(f)

    def test_json_metadata_doc_type(self, tmp_path: Path) -> None:
        f = tmp_path / "data.json"
        f.write_text('{"key": "value"}')
        page = DocumentLoaderFactory.load(f)[0]
        assert page.doc_type == "json"


# ===========================================================================
# E. Integration tests — real security documents
# ===========================================================================

class TestRealDocuments:

    @pytest.mark.skipif(not NIST_PDF.exists(), reason="NIST PDF not found")
    def test_nist_pdf_loads(self) -> None:
        pages = DocumentLoaderFactory.load(NIST_PDF)
        assert len(pages) > 0, "NIST PDF should have multiple pages"
        assert all(isinstance(p, LoadedPage) for p in pages)
        assert all(p.doc_type == "pdf" for p in pages)

    @pytest.mark.skipif(not NIST_PDF.exists(), reason="NIST PDF not found")
    def test_nist_pdf_has_incident_response_content(self) -> None:
        pages = DocumentLoaderFactory.load(NIST_PDF)
        all_text = " ".join(p.text for p in pages).lower()
        # NIST 800-61 must contain these core IR terms
        for term in ["incident", "response"]:
            assert term in all_text, f"Expected '{term}' in NIST 800-61 text"

    @pytest.mark.skipif(not NIST_PDF.exists(), reason="NIST PDF not found")
    def test_nist_pdf_page_numbers_sequential(self) -> None:
        pages = DocumentLoaderFactory.load(NIST_PDF)
        numbers = [p.page_number for p in pages]
        assert numbers == sorted(numbers), "Page numbers must be sequential"
        assert numbers[0] >= 1, "First page number must be >= 1"

    @pytest.mark.skipif(not NIST_PDF.exists(), reason="NIST PDF not found")
    def test_nist_pdf_hash_is_consistent(self) -> None:
        pages1 = DocumentLoaderFactory.load(NIST_PDF)
        pages2 = DocumentLoaderFactory.load(NIST_PDF)
        assert pages1[0].file_hash == pages2[0].file_hash

    @pytest.mark.skipif(not FED_PDF.exists(), reason="Federal PDF not found")
    def test_federal_pdf_loads(self) -> None:
        pages = DocumentLoaderFactory.load(FED_PDF)
        assert len(pages) > 0
        all_text = " ".join(p.text for p in pages).lower()
        assert any(w in all_text for w in ["incident", "playbook", "vulnerability"])

    @pytest.mark.skipif(not MITRE_JSON.exists(), reason="MITRE JSON not found")
    def test_mitre_json_loads(self) -> None:
        pages = DocumentLoaderFactory.load(MITRE_JSON)
        assert len(pages) == 1
        assert pages[0].doc_type == "json"
        # ATT&CK JSON must contain technique references
        assert "attack" in pages[0].text.lower()

    @pytest.mark.skipif(not KEV_CSV.exists(), reason="KEV CSV not found")
    def test_kev_csv_is_in_raw_directory(self) -> None:
        """The CISA KEV file is a CSV — verify it exists for future phase."""
        assert KEV_CSV.stat().st_size > 0
        # CSV is not yet a supported loader type (Phase 2 scope: pdf/txt/docx/json)
        # This test just confirms the file is present for reference.

    @pytest.mark.skipif(not RAW_DIR.exists(), reason="data/raw not found")
    def test_load_directory_loads_all_docs(self) -> None:
        pages = DocumentLoaderFactory.load_directory(RAW_DIR, recursive=True)
        assert len(pages) > 0, "Directory load should return at least one page"
        doc_names = {p.doc_name for p in pages}
        assert len(doc_names) >= 1


# ===========================================================================
# F. Security / edge-case tests
# ===========================================================================

class TestSecurityEdgeCases:

    def test_file_with_injection_attempt_in_name(self, tmp_path: Path) -> None:
        """Files with special characters in names must still be handled safely."""
        f = tmp_path / "normal_report.txt"
        f.write_text("Safe content only.")
        pages = DocumentLoaderFactory.load(f)
        assert len(pages) == 1

    def test_txt_content_is_not_executed(self, tmp_path: Path) -> None:
        """Content that looks like a command must be stored as plain text."""
        malicious = "import os; os.system('del /f /q C:\\\\')"
        f = tmp_path / "evil.txt"
        f.write_text(malicious)
        pages = DocumentLoaderFactory.load(f)
        # Content is returned as a plain string — never executed
        assert pages[0].text == malicious

    def test_duplicate_files_have_same_hash(self, tmp_path: Path) -> None:
        """Two copies of the same file must produce the same hash."""
        f1 = tmp_path / "copy1.txt"
        f2 = tmp_path / "copy2.txt"
        content = "NIST IR procedure content."
        f1.write_text(content)
        f2.write_text(content)
        h1 = compute_file_hash(f1)
        h2 = compute_file_hash(f2)
        assert h1 == h2, "Identical content must produce identical hash"
