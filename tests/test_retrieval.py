"""
tests/test_retrieval.py
=======================
Phase 1 — Smoke tests for project structure and configuration.

These tests do NOT require any ML models or external services.
They verify that:
  1. All required source files and directories exist.
  2. The Settings object loads and validates correctly.
  3. Derived properties return the right types.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path regardless of where pytest runs
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# 1. Project structure tests
# ---------------------------------------------------------------------------

REQUIRED_DIRS = [
    "app",
    "data/raw",
    "data/processed",
    "vector_db",
    "src",
    "src/ingestion",
    "evaluation",
    "tests",
]

REQUIRED_FILES = [
    "requirements.txt",
    ".gitignore",
    ".env.example",
    "src/__init__.py",
    "src/config.py",
    "src/ingestion/__init__.py",
    "evaluation/test_questions.json",
]


@pytest.mark.parametrize("directory", REQUIRED_DIRS)
def test_required_directory_exists(directory: str) -> None:
    """Every required directory must exist in the project root."""
    target = PROJECT_ROOT / directory
    assert target.is_dir(), f"Missing directory: {directory}"


@pytest.mark.parametrize("filepath", REQUIRED_FILES)
def test_required_file_exists(filepath: str) -> None:
    """Every required file must exist in the project root."""
    target = PROJECT_ROOT / filepath
    assert target.is_file(), f"Missing file: {filepath}"


# ---------------------------------------------------------------------------
# 2. Configuration tests
# ---------------------------------------------------------------------------

def test_settings_loads_without_error() -> None:
    """Settings must instantiate without raising an exception."""
    from src.config import settings
    assert settings is not None


def test_settings_default_values() -> None:
    """Core defaults must match the documented spec."""
    from src.config import settings
    assert settings.ollama_model == "llama3.2:1b"
    assert settings.embedding_model == "all-MiniLM-L6-v2"
    assert settings.chunk_size == 512
    assert settings.chunk_overlap == 64
    assert settings.top_k_results == 5


def test_allowed_extensions_list() -> None:
    """The derived list property must parse the comma-separated string."""
    from src.config import settings
    exts = settings.allowed_extensions_list
    assert isinstance(exts, list)
    assert "pdf" in exts
    assert "txt" in exts
    assert "docx" in exts


def test_path_properties_return_path_objects() -> None:
    """Path helper properties must return pathlib.Path instances."""
    from src.config import settings
    assert isinstance(settings.raw_docs_path, Path)
    assert isinstance(settings.processed_docs_path, Path)
    assert isinstance(settings.chroma_persist_path, Path)


def test_requirements_txt_contains_key_packages() -> None:
    """requirements.txt must list the core dependencies."""
    req_file = PROJECT_ROOT / "requirements.txt"
    content = req_file.read_text(encoding="utf-8").lower()
    for pkg in ["streamlit", "langchain", "chromadb", "sentence-transformers", "pymupdf", "ollama"]:
        assert pkg in content, f"Missing package in requirements.txt: {pkg}"


def test_env_example_has_required_keys() -> None:
    """.env.example must define every critical environment variable."""
    env_file = PROJECT_ROOT / ".env.example"
    content = env_file.read_text(encoding="utf-8")
    for key in ["OLLAMA_MODEL", "EMBEDDING_MODEL", "CHROMA_PERSIST_DIR", "CHUNK_SIZE"]:
        assert key in content, f"Missing key in .env.example: {key}"
