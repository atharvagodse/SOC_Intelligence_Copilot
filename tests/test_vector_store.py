"""
tests/test_vector_store.py
==========================
Phase 5 — Tests for ChromaDB VectorStore integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings import get_embedding_service
from src.ingestion.chunking import DocumentChunk, DocumentChunker
from src.ingestion.loaders import DocumentLoaderFactory
from src.vector_store import SearchResult, VectorStore


def _make_chunk(
    text: str,
    doc_name: str = "playbook.txt",
    file_hash: str = "hash_abc_123456",
    page_number: int = 1,
    chunk_index: int = 0,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{file_hash[:12]}_p{page_number}_c{chunk_index}",
        text=text,
        doc_name=doc_name,
        doc_type="txt",
        source=f"/tmp/{doc_name}",
        page_number=page_number,
        total_pages=1,
        chunk_index=chunk_index,
        upload_date="2026-11-27T00:00:00+00:00",
        file_hash=file_hash,
    )


class TestVectorStore:
    def test_add_search_list_and_delete(self, tmp_path: Path) -> None:
        store = VectorStore(
            persist_dir=tmp_path / "chroma",
            collection_name="test_soc_kb",
            embedding_service=get_embedding_service(),
        )
        assert store.count() == 0

        chunks = [
            _make_chunk(
                "Containment isolates infected endpoints from the corporate network.",
                doc_name="ir_playbook.txt",
                file_hash="hash_ir_000001",
                chunk_index=0,
            ),
            _make_chunk(
                "Phishing emails often contain spoofed sender domains and malicious links.",
                doc_name="phishing_guide.txt",
                file_hash="hash_phish_002",
                chunk_index=0,
            ),
        ]

        added = store.add_chunks(chunks)
        assert added == 2
        assert store.count() == 2
        assert store.has_document("hash_ir_000001") is True
        assert store.has_document("nonexistent_hash") is False

        # Duplicate ingestion should be skipped
        added_again = store.add_chunks(chunks, skip_duplicates=True)
        assert added_again == 0
        assert store.count() == 2

        # Semantic search
        results = store.similarity_search("How do we isolate an infected host?", top_k=2)
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].metadata["doc_name"] == "ir_playbook.txt"
        assert 0.0 <= results[0].score <= 1.0

        # List documents
        docs = store.list_documents()
        assert len(docs) == 2
        names = {d["doc_name"] for d in docs}
        assert names == {"ir_playbook.txt", "phishing_guide.txt"}

        # Delete one document
        deleted = store.delete_document(file_hash="hash_ir_000001")
        assert deleted == 1
        assert store.count() == 1
        assert store.has_document("hash_ir_000001") is False

        # Reset collection
        store.reset()
        assert store.count() == 0

    def test_persistence_across_restarts(self, tmp_path: Path) -> None:
        db_dir = tmp_path / "persistent_chroma"
        svc = get_embedding_service()

        store1 = VectorStore(
            persist_dir=db_dir,
            collection_name="persist_test",
            embedding_service=svc,
        )
        store1.add_chunks([
            _make_chunk("MITRE ATT&CK T1110 covers brute force credential access.")
        ])
        assert store1.count() == 1

        # Re-open from the same directory
        store2 = VectorStore(
            persist_dir=db_dir,
            collection_name="persist_test",
            embedding_service=svc,
        )
        assert store2.count() == 1
        hits = store2.similarity_search("brute force T1110", top_k=1)
        assert len(hits) == 1
        assert "T1110" in hits[0].text

    def test_end_to_end_nist_sample_ingestion(self, tmp_path: Path) -> None:
        nist_pdf = (
            PROJECT_ROOT
            / "data"
            / "raw"
            / "incident_response"
            / "NIST_SP_800-61_Rev3.pdf"
        )
        if not nist_pdf.exists():
            pytest.skip("NIST PDF not found")

        pages = DocumentLoaderFactory.load(nist_pdf)[:5]
        chunks = DocumentChunker().chunk_pages(pages)
        store = VectorStore(
            persist_dir=tmp_path / "nist_chroma",
            collection_name="nist_sample",
            embedding_service=get_embedding_service(),
        )
        added = store.add_chunks(chunks)
        assert added == len(chunks)
        results = store.similarity_search("incident response cybersecurity", top_k=3)
        assert len(results) > 0
        assert results[0].metadata["doc_name"] == "NIST_SP_800-61_Rev3.pdf"
