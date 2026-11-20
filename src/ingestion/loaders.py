"""
src/ingestion/loaders.py
========================
Document loading and text extraction for the SOC Intelligence Copilot.

Supports three file types:
  - PDF  → extracted page-by-page using PyMuPDF (fitz)
  - DOCX → extracted paragraph-by-paragraph using python-docx
  - TXT  → read with automatic encoding detection
  - JSON → loaded as plain text (for MITRE ATT&CK / CISA KEV feeds)

Every loader returns a list of ``LoadedPage`` objects — a simple dataclass
that carries both the raw text and its metadata.  The rest of the pipeline
(chunking, embedding) only ever sees ``LoadedPage`` objects, so adding a new
file type later only requires a new loader class here.

Design pattern used: **Factory Method**
  ``DocumentLoaderFactory.get_loader(path)`` inspects the file extension and
  returns the right loader without the caller needing to know the type.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.config import settings

logger = logging.getLogger("soc_copilot.loaders")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_FILE_BYTES = settings.max_file_size_mb * 1024 * 1024
# JSON threat-intel feeds (e.g. MITRE ATT&CK at ~51 MB) can exceed the default
# upload limit. We allow up to 2× the configured limit for JSON files.
MAX_JSON_FILE_BYTES = max(MAX_FILE_BYTES, 100 * 1024 * 1024)  # at least 100 MB
ALLOWED_EXTENSIONS = settings.allowed_extensions_list   # ['pdf','txt','docx']


# ---------------------------------------------------------------------------
# Data model: a single page / section of a document
# ---------------------------------------------------------------------------

@dataclass
class LoadedPage:
    """
    Represents one logical unit of extracted text from a document.

    For PDFs  → one LoadedPage = one PDF page.
    For DOCX  → one LoadedPage = one non-empty paragraph.
    For TXT   → one LoadedPage = the whole file (page_number = 1).
    For JSON  → one LoadedPage = the whole file as a string.

    Attributes
    ----------
    text          : The extracted, raw text content (not yet cleaned/chunked).
    source        : Absolute path to the original file.
    doc_name      : Human-readable file name without path.
    doc_type      : 'pdf', 'docx', 'txt', or 'json'.
    page_number   : 1-based page index.
    total_pages   : Total pages/paragraphs in the document.
    upload_date   : ISO-8601 timestamp of when this run was started.
    file_hash     : SHA-256 of the source file (for duplicate detection).
    extra_metadata: Optional dict for any loader-specific extras.
    """
    text: str
    source: str
    doc_name: str
    doc_type: str
    page_number: int
    total_pages: int
    upload_date: str
    file_hash: str
    extra_metadata: dict = field(default_factory=dict)

    def to_metadata_dict(self) -> dict:
        """Return a flat dict suitable for ChromaDB metadata storage."""
        return {
            "source":      self.source,
            "doc_name":    self.doc_name,
            "doc_type":    self.doc_type,
            "page_number": self.page_number,
            "total_pages": self.total_pages,
            "upload_date": self.upload_date,
            "file_hash":   self.file_hash,
            **self.extra_metadata,
        }


# ---------------------------------------------------------------------------
# File validation helpers
# ---------------------------------------------------------------------------

class FileValidationError(ValueError):
    """Raised when an uploaded file fails security or format validation."""


def validate_file(path: Path) -> None:
    """
    Validate a file before loading.

    Checks performed:
    1. File exists and is a regular file (not a symlink, directory, etc.).
    2. Extension is in the allowed list (pdf, txt, docx, or json).
    3. File size is within the configured limit.

    Raises FileValidationError on any failure.
    """
    if not path.exists():
        raise FileValidationError(f"File not found: {path}")

    if not path.is_file():
        raise FileValidationError(f"Path is not a regular file: {path}")

    resolved = path.resolve()
    ext = resolved.suffix.lstrip(".").lower()

    # Allow json in addition to configured extensions (for threat intel feeds)
    allowed = set(ALLOWED_EXTENSIONS) | {"json"}
    if ext not in allowed:
        raise FileValidationError(
            f"Unsupported file type '.{ext}'. Allowed: {sorted(allowed)}"
        )

    size = resolved.stat().st_size
    if size == 0:
        raise FileValidationError(f"File is empty: {path.name}")

    # JSON files (threat-intel feeds) may be larger — use an expanded limit
    effective_limit = MAX_JSON_FILE_BYTES if ext == "json" else MAX_FILE_BYTES
    if size > effective_limit:
        size_mb = size / (1024 * 1024)
        limit_mb = effective_limit // (1024 * 1024)
        raise FileValidationError(
            f"File '{path.name}' is {size_mb:.1f} MB, "
            f"exceeding the {limit_mb} MB limit for '.{ext}' files."
        )


def compute_file_hash(path: Path) -> str:
    """Return the SHA-256 hex digest of a file (for duplicate detection)."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Abstract base loader
# ---------------------------------------------------------------------------

class BaseDocumentLoader(ABC):
    """
    Abstract base class for all document loaders.

    Subclasses must implement ``_load(path, file_hash, upload_date)``.
    The public ``load()`` method handles validation and error wrapping.
    """

    def load(self, path: Path) -> list[LoadedPage]:
        """
        Validate then load a document.

        Returns
        -------
        List of LoadedPage objects — one per page / paragraph / section.

        Raises
        ------
        FileValidationError  : if validation fails.
        RuntimeError         : if the loader itself encounters an error.
        """
        validate_file(path)
        file_hash = compute_file_hash(path)
        upload_date = utc_now_iso()
        logger.info("Loading '%s' (hash=%s…)", path.name, file_hash[:8])

        try:
            pages = self._load(path, file_hash, upload_date)
        except FileValidationError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Failed to load '{path.name}': {exc}") from exc

        logger.info(
            "Loaded '%s' → %d pages/sections extracted.", path.name, len(pages)
        )
        return pages

    @abstractmethod
    def _load(
        self, path: Path, file_hash: str, upload_date: str
    ) -> list[LoadedPage]:
        ...


# ---------------------------------------------------------------------------
# PDF Loader
# ---------------------------------------------------------------------------

class PdfLoader(BaseDocumentLoader):
    """
    Extracts text from PDF files, one LoadedPage per PDF page.

    Uses PyMuPDF (``import fitz``) which is:
      - Fast (C-based, no JVM required)
      - Handles complex layouts (multi-column, tables)
      - Preserves page number information precisely
      - CPU-only, no GPU required
    """

    def _load(
        self, path: Path, file_hash: str, upload_date: str
    ) -> list[LoadedPage]:
        try:
            import pymupdf as fitz  # preferred import (avoids FutureWarning)
        except ImportError:
            try:
                import fitz  # fallback for older pymupdf installs
            except ImportError as exc:
                raise RuntimeError(
                    "PyMuPDF not installed. Run: pip install pymupdf"
                ) from exc

        pages: list[LoadedPage] = []

        with fitz.open(str(path)) as doc:
            total_pages = len(doc)
            if total_pages == 0:
                raise FileValidationError(f"PDF has no pages: {path.name}")

            for page_index, page in enumerate(doc, start=1):
                raw_text = page.get_text("text")

                # Skip pages with no extractable text (e.g. scanned images)
                if not raw_text or not raw_text.strip():
                    logger.debug(
                        "'%s' page %d has no extractable text — skipping.",
                        path.name, page_index,
                    )
                    continue

                pages.append(LoadedPage(
                    text=raw_text,
                    source=str(path.resolve()),
                    doc_name=path.name,
                    doc_type="pdf",
                    page_number=page_index,
                    total_pages=total_pages,
                    upload_date=upload_date,
                    file_hash=file_hash,
                ))

        if not pages:
            raise FileValidationError(
                f"No text could be extracted from '{path.name}'. "
                "It may be a scanned PDF with no OCR text layer."
            )
        return pages


# ---------------------------------------------------------------------------
# DOCX Loader
# ---------------------------------------------------------------------------

class DocxLoader(BaseDocumentLoader):
    """
    Extracts text from DOCX files, one LoadedPage per non-empty paragraph.

    Uses python-docx which reads the Open XML format directly.
    Tables and headers/footers are not extracted in Phase 2.
    """

    def _load(
        self, path: Path, file_hash: str, upload_date: str
    ) -> list[LoadedPage]:
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError(
                "python-docx not installed. Run: pip install python-docx"
            ) from exc

        doc = Document(str(path))
        all_paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

        if not all_paragraphs:
            raise FileValidationError(
                f"No text paragraphs found in '{path.name}'."
            )

        total = len(all_paragraphs)
        pages: list[LoadedPage] = []

        for idx, para_text in enumerate(all_paragraphs, start=1):
            pages.append(LoadedPage(
                text=para_text,
                source=str(path.resolve()),
                doc_name=path.name,
                doc_type="docx",
                page_number=idx,
                total_pages=total,
                upload_date=upload_date,
                file_hash=file_hash,
            ))

        return pages


# ---------------------------------------------------------------------------
# TXT Loader
# ---------------------------------------------------------------------------

class TxtLoader(BaseDocumentLoader):
    """
    Loads a plain-text file as a single LoadedPage.

    Tries UTF-8 first, then falls back to latin-1 to handle most real-world
    text files without crashing on unusual characters.
    """

    _ENCODINGS: list[str] = ["utf-8", "utf-8-sig", "latin-1"]

    def _load(
        self, path: Path, file_hash: str, upload_date: str
    ) -> list[LoadedPage]:
        text: Optional[str] = None

        for enc in self._ENCODINGS:
            try:
                text = path.read_text(encoding=enc)
                logger.debug("'%s' read with encoding '%s'.", path.name, enc)
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            raise FileValidationError(
                f"Could not decode '{path.name}' with any supported encoding."
            )

        if not text.strip():
            raise FileValidationError(f"File '{path.name}' contains no text.")

        return [LoadedPage(
            text=text,
            source=str(path.resolve()),
            doc_name=path.name,
            doc_type="txt",
            page_number=1,
            total_pages=1,
            upload_date=upload_date,
            file_hash=file_hash,
        )]


# ---------------------------------------------------------------------------
# JSON Loader  (bonus: handles MITRE ATT&CK & CISA KEV feeds)
# ---------------------------------------------------------------------------

class JsonLoader(BaseDocumentLoader):
    """
    Loads a JSON file and treats it as plain text for the pipeline.

    Security note: JSON content is NEVER executed or eval'd — it is
    parsed only for validation, then stored as a plain string.
    """

    def _load(
        self, path: Path, file_hash: str, upload_date: str
    ) -> list[LoadedPage]:
        import json

        try:
            raw = path.read_text(encoding="utf-8")
            json.loads(raw)  # validate structure; keep string form
        except json.JSONDecodeError as exc:
            raise FileValidationError(
                f"'{path.name}' is not valid JSON: {exc}"
            ) from exc
        except UnicodeDecodeError as exc:
            raise FileValidationError(
                f"Could not decode '{path.name}' as UTF-8: {exc}"
            ) from exc

        if not raw.strip():
            raise FileValidationError(f"File '{path.name}' is empty.")

        return [LoadedPage(
            text=raw,
            source=str(path.resolve()),
            doc_name=path.name,
            doc_type="json",
            page_number=1,
            total_pages=1,
            upload_date=upload_date,
            file_hash=file_hash,
            extra_metadata={"format": "json"},
        )]


# ---------------------------------------------------------------------------
# Factory — the single entry point for the rest of the pipeline
# ---------------------------------------------------------------------------

class DocumentLoaderFactory:
    """
    Returns the correct loader for a given file path.

    Usage
    -----
    ::
        from src.ingestion.loaders import DocumentLoaderFactory
        from pathlib import Path

        pages = DocumentLoaderFactory.load(Path("data/raw/NIST_SP_800-61.pdf"))
        for page in pages:
            print(f"Page {page.page_number}: {page.text[:80]}")

    Supported extensions: pdf, docx, txt, json
    """

    _REGISTRY: dict[str, type[BaseDocumentLoader]] = {
        "pdf":  PdfLoader,
        "docx": DocxLoader,
        "txt":  TxtLoader,
        "json": JsonLoader,
    }

    @classmethod
    def get_loader(cls, path: Path) -> BaseDocumentLoader:
        """Return the appropriate loader instance for a file."""
        ext = Path(path).suffix.lstrip(".").lower()
        loader_class = cls._REGISTRY.get(ext)
        if loader_class is None:
            raise FileValidationError(
                f"No loader registered for extension '.{ext}'. "
                f"Supported: {list(cls._REGISTRY.keys())}"
            )
        return loader_class()

    @classmethod
    def load(cls, path: Path) -> list[LoadedPage]:
        """
        One-liner convenience method: validate → load → return pages.

        Parameters
        ----------
        path : Path or str pointing to the document.

        Returns
        -------
        List of LoadedPage objects ready for the chunking pipeline.
        """
        path = Path(path)
        loader = cls.get_loader(path)
        return loader.load(path)

    @classmethod
    def load_directory(
        cls,
        directory: Path,
        recursive: bool = True,
    ) -> list[LoadedPage]:
        """
        Load every supported document in a directory.

        Parameters
        ----------
        directory : Root folder to scan.
        recursive : If True (default), descend into sub-folders.

        Returns
        -------
        Combined list of all LoadedPage objects from all files.
        Files that fail to load are logged and skipped (never crash the batch).
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise FileValidationError(f"Not a directory: {directory}")

        pattern = "**/*" if recursive else "*"
        all_pages: list[LoadedPage] = []
        supported_exts = set(cls._REGISTRY.keys())

        for file_path in sorted(directory.glob(pattern)):
            if not file_path.is_file():
                continue
            ext = file_path.suffix.lstrip(".").lower()
            if ext not in supported_exts:
                continue
            try:
                pages = cls.load(file_path)
                all_pages.extend(pages)
                logger.info("✓ %-55s → %d sections", file_path.name, len(pages))
            except (FileValidationError, RuntimeError) as exc:
                logger.warning("✗ Skipping '%s': %s", file_path.name, exc)

        logger.info(
            "Directory scan complete: %d total sections from '%s'",
            len(all_pages), directory,
        )
        return all_pages
