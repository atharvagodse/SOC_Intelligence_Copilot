"""
src/embeddings.py
=================
Embedding generation for the SOC Intelligence Copilot.

Uses Sentence Transformers (``all-MiniLM-L6-v2`` by default):
  - Lightweight (~90 MB), 384-dimensional dense vectors
  - Runs comfortably on CPU within tight RAM budgets
  - Completely free and offline after the initial model download

Design notes:
  - **Lazy loading**: the model is only loaded into memory when first needed,
    keeping application startup fast and avoiding unnecessary RAM usage.
  - **Module-level cache**: ``get_embedding_service()`` reuses a single
    instance so we never load multiple copies of the model into RAM.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Sequence

from src.config import settings

logger = logging.getLogger("soc_copilot.embeddings")


class EmbeddingService:
    """
    Wrapper around ``SentenceTransformer`` for generating normalised embeddings.

    Parameters
    ----------
    model_name : HuggingFace / Sentence-Transformers model identifier.
                 Defaults to ``settings.embedding_model`` ('all-MiniLM-L6-v2').
    device     : Target compute device ('cpu' by default for compatibility).
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name or settings.embedding_model
        self.device = device
        self._model = None

    def _load_model(self):
        """Lazily load the SentenceTransformer model on first use."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed. "
                    "Run: pip install sentence-transformers"
                ) from exc

            logger.info(
                "Loading embedding model '%s' on device='%s'...",
                self.model_name,
                self.device,
            )
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(
                "Embedding model '%s' ready (dim=%d).",
                self.model_name,
                self._model.get_sentence_embedding_dimension(),
            )
        return self._model

    @property
    def dimension(self) -> int:
        """Return the vector dimensionality of the loaded model (e.g. 384)."""
        model = self._load_model()
        dim = model.get_sentence_embedding_dimension()
        if dim is None:
            raise RuntimeError("Could not determine embedding dimension.")
        return int(dim)

    def embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        """
        Generate L2-normalised embeddings for a sequence of texts.

        Parameters
        ----------
        texts      : Non-empty strings to embed.
        batch_size : Number of texts encoded per batch (keeps RAM usage bounded).

        Returns
        -------
        List of float vectors (one vector per input text).
        """
        if not texts:
            return []

        model = self._load_model()
        vectors = model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Generate an L2-normalised embedding for a single search query.

        Raises ValueError if the query is empty or whitespace-only.
        """
        if not query or not query.strip():
            raise ValueError("Query text must not be empty.")
        return self.embed_texts([query.strip()], batch_size=1)[0]

    # LangChain Embeddings interface compatibility
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Alias for ``embed_texts`` (LangChain ``Embeddings`` compatibility)."""
        return self.embed_texts(texts)


@lru_cache(maxsize=1)
def get_embedding_service(
    model_name: str | None = None,
    device: str = "cpu",
) -> EmbeddingService:
    """
    Return a cached ``EmbeddingService`` singleton.

    Using a singleton avoids loading the model multiple times into RAM.
    """
    return EmbeddingService(model_name=model_name, device=device)
