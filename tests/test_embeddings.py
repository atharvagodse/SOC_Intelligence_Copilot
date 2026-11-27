"""
tests/test_embeddings.py
========================
Phase 4 — Tests for SentenceTransformer embedding generation.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings import EmbeddingService, get_embedding_service


class TestEmbeddingServiceUnit:
    def test_empty_texts_returns_empty_list(self) -> None:
        svc = EmbeddingService()
        assert svc.embed_texts([]) == []

    @pytest.mark.parametrize("bad_query", ["", "   ", "\n\t"])
    def test_empty_query_raises(self, bad_query: str) -> None:
        svc = EmbeddingService()
        with pytest.raises(ValueError, match="must not be empty"):
            svc.embed_query(bad_query)

    def test_singleton_returns_same_instance(self) -> None:
        s1 = get_embedding_service()
        s2 = get_embedding_service()
        assert s1 is s2


class TestEmbeddingServiceIntegration:
    def test_real_model_embeds_and_normalises(self) -> None:
        svc = get_embedding_service()
        vec = svc.embed_query("How should a SOC analyst contain ransomware?")

        assert svc.dimension == 384
        assert len(vec) == 384
        # L2 norm of a normalised vector must be ~1.0
        norm = math.sqrt(sum(x * x for x in vec))
        assert norm == pytest.approx(1.0, abs=1e-4)

    def test_semantic_similarity_ordering(self) -> None:
        svc = get_embedding_service()
        q = svc.embed_query("brute force SSH login attempts")
        docs = svc.embed_texts([
            "Repeated failed SSH authentication attempts indicate a brute-force attack.",
            "Baking chocolate chip cookies requires flour, sugar, and butter.",
        ])

        sim_security = sum(a * b for a, b in zip(q, docs[0]))
        sim_unrelated = sum(a * b for a, b in zip(q, docs[1]))
        assert sim_security > sim_unrelated
