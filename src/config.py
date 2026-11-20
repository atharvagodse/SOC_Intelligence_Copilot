"""
src/config.py
=============
Centralised configuration for SOC Intelligence Copilot.

All runtime settings are loaded from environment variables (via .env).
Every other module imports from here — no hard-coded values anywhere else.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    Application-wide settings loaded from the .env file.

    Pydantic-Settings reads each field from the matching environment variable
    (case-insensitive).  Defaults are used when the variable is absent.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",          # ignore unknown keys in .env
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_name: str = Field(default="SOC Intelligence Copilot")
    app_version: str = Field(default="0.1.0")
    log_level: str = Field(default="INFO")

    # ── Ollama ───────────────────────────────────────────────────────────────
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.2:1b")

    # ── Embedding ────────────────────────────────────────────────────────────
    embedding_model: str = Field(default="all-MiniLM-L6-v2")

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    chroma_persist_dir: str = Field(default="./vector_db")
    chroma_collection_name: str = Field(default="soc_knowledge_base")

    # ── Documents ────────────────────────────────────────────────────────────
    raw_docs_dir: str = Field(default="./data/raw")
    processed_docs_dir: str = Field(default="./data/processed")
    max_file_size_mb: int = Field(default=50)
    allowed_extensions: str = Field(default="pdf,txt,docx")

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)

    # ── Retrieval ────────────────────────────────────────────────────────────
    top_k_results: int = Field(default=5)
    similarity_threshold: float = Field(default=0.3)

    # ── Derived helpers (not env vars) ───────────────────────────────────────
    @property
    def allowed_extensions_list(self) -> list[str]:
        """Return allowed extensions as a clean list e.g. ['pdf', 'txt', 'docx']."""
        return [ext.strip().lower() for ext in self.allowed_extensions.split(",")]

    @property
    def raw_docs_path(self) -> Path:
        return Path(self.raw_docs_dir)

    @property
    def processed_docs_path(self) -> Path:
        return Path(self.processed_docs_dir)

    @property
    def chroma_persist_path(self) -> Path:
        return Path(self.chroma_persist_dir)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return upper


# ---------------------------------------------------------------------------
# Singleton instance — import this everywhere
# ---------------------------------------------------------------------------

settings = Settings()


# ---------------------------------------------------------------------------
# Logging setup — call once at application startup
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    """
    Configure the root logger based on settings.log_level.

    Returns the application-level logger named 'soc_copilot'.
    """
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("soc_copilot")
    logger.setLevel(getattr(logging, settings.log_level))
    return logger


# Module-level logger for config itself
logger = setup_logging()
logger.debug("Configuration loaded: %s v%s", settings.app_name, settings.app_version)
