"""
src/vector_store.py
===================
ChromaDB vector database integration for the SOC Intelligence Copilot.

Responsibilities:
  - Persist document chunk embeddings and metadata on disk (``./vector_db``)
  - Prevent duplicate document ingestion via ``file_hash`` checks and
    deterministic chunk IDs
  - Provide semantic similarity search (cosine distance)
  - Support listing indexed documents, deleting a document, and resetting
    the collection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from src.config import settings
from src.embeddings import EmbeddingService, get_embedding_service
from src.ingestion.chunking import DocumentChunk

logger = logging.getLogger("soc_copilot.vector_store")


@dataclass
class SearchResult:
    """
    A single retrieved chunk from ChromaDB with its similarity score.

    Attributes
    ----------
    chunk_id : Unique chunk identifier.
    text     : Retrieved chunk text.
    score    : Cosine similarity in [0, 1] (higher = more relevant).
    metadata : Flat metadata dict (doc_name, page_number, file_hash, etc.).
    """

    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]


class VectorStore:
    """
    Persistent ChromaDB wrapper for storing and querying SOC knowledge chunks.

    Parameters
    ----------
    persist_dir       : Directory where ChromaDB stores SQLite & HNSW index files.
    collection_name   : Name of the ChromaDB collection.
    embedding_service : Optional ``EmbeddingService`` (defaults to shared singleton).
    """

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection_name: str | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self.persist_dir = Path(persist_dir or settings.chroma_persist_dir)
        self.collection_name = collection_name or settings.chroma_collection_name
        self.embedding_service = embedding_service or get_embedding_service()

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = self._create_client()
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorStore ready: dir='%s', collection='%s', chunks=%d",
            self.persist_dir,
            self.collection_name,
            self._collection.count(),
        )

    def _create_client(self):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "chromadb is not installed. Run: pip install chromadb"
            ) from exc

        return chromadb.PersistentClient(path=str(self.persist_dir))

    # ── Read / Inspect ───────────────────────────────────────────────────────

    def count(self) -> int:
        """Return the total number of chunks stored in the collection."""
        return self._collection.count()

    def has_document(self, file_hash: str) -> bool:
        """Return True if any chunk with ``file_hash`` is already indexed."""
        if not file_hash:
            return False
        result = self._collection.get(
            where={"file_hash": file_hash},
            limit=1,
            include=[],
        )
        return len(result.get("ids", [])) > 0

    def list_documents(self) -> list[dict[str, Any]]:
        """
        Return a summary list of all indexed documents.

        Each entry contains: ``doc_name``, ``doc_type``, ``file_hash``,
        ``upload_date``, and ``chunk_count``.
        """
        total = self.count()
        if total == 0:
            return []

        records = self._collection.get(include=["metadatas"])
        docs_by_hash: dict[str, dict[str, Any]] = {}

        for meta in records.get("metadatas") or []:
            if not meta:
                continue
            file_hash = str(meta.get("file_hash", "unknown"))
            if file_hash not in docs_by_hash:
                docs_by_hash[file_hash] = {
                    "doc_name": meta.get("doc_name", "unknown"),
                    "doc_type": meta.get("doc_type", "unknown"),
                    "file_hash": file_hash,
                    "upload_date": meta.get("upload_date", ""),
                    "chunk_count": 0,
                }
            docs_by_hash[file_hash]["chunk_count"] += 1

        return sorted(docs_by_hash.values(), key=lambda d: str(d["doc_name"]))

    # ── Write / Delete ───────────────────────────────────────────────────────

    def add_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        batch_size: int = 64,
        skip_duplicates: bool = True,
    ) -> int:
        """
        Embed and upsert ``DocumentChunk`` objects into ChromaDB.

        Parameters
        ----------
        chunks          : Sequence of ``DocumentChunk`` objects to store.
        batch_size      : Number of chunks embedded and upserted per batch.
        skip_duplicates : If True, chunks whose ``file_hash`` is already in the
                          collection are skipped.

        Returns
        -------
        Number of chunks actually added/upserted.
        """
        if not chunks:
            return 0

        filtered_chunks: list[DocumentChunk] = list(chunks)
        if skip_duplicates:
            seen_hashes: dict[str, bool] = {}
            kept: list[DocumentChunk] = []
            for chunk in chunks:
                if chunk.file_hash not in seen_hashes:
                    seen_hashes[chunk.file_hash] = self.has_document(chunk.file_hash)
                if not seen_hashes[chunk.file_hash]:
                    kept.append(chunk)
            skipped = len(chunks) - len(kept)
            if skipped > 0:
                logger.info(
                    "Skipped %d chunks from already-indexed document(s).",
                    skipped,
                )
            filtered_chunks = kept

        if not filtered_chunks:
            return 0

        added = 0
        for start in range(0, len(filtered_chunks), batch_size):
            batch = filtered_chunks[start : start + batch_size]
            ids = [c.chunk_id for c in batch]
            texts = [c.text for c in batch]
            metadatas = [c.to_metadata_dict() for c in batch]
            embeddings = self.embedding_service.embed_texts(
                texts, batch_size=min(32, batch_size)
            )

            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            added += len(batch)

        logger.info("Upserted %d chunks into '%s'.", added, self.collection_name)
        return added

    def delete_document(
        self,
        file_hash: str | None = None,
        doc_name: str | None = None,
    ) -> int:
        """
        Delete all chunks belonging to a document by ``file_hash`` or ``doc_name``.

        Returns the number of deleted chunks.
        """
        if not file_hash and not doc_name:
            raise ValueError("Provide either 'file_hash' or 'doc_name' to delete.")

        where_filter: dict[str, str] = (
            {"file_hash": file_hash} if file_hash else {"doc_name": doc_name}  # type: ignore[dict-item]
        )
        existing = self._collection.get(where=where_filter, include=[])
        ids_to_delete = existing.get("ids", [])
        if not ids_to_delete:
            return 0

        self._collection.delete(ids=ids_to_delete)
        logger.info(
            "Deleted %d chunks matching %s.", len(ids_to_delete), where_filter
        )
        return len(ids_to_delete)

    def reset(self) -> None:
        """Delete and recreate the collection (removes all stored chunks)."""
        self._client.delete_collection(name=self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("Collection '%s' has been reset.", self.collection_name)

    # ── Query ────────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Retrieve the ``top_k`` most semantically similar chunks for ``query``.

        Parameters
        ----------
        query : Natural-language question or search phrase.
        top_k : Maximum number of chunks to return (defaults to ``settings.top_k_results``).
        where : Optional ChromaDB metadata filter dict.

        Returns
        -------
        List of ``SearchResult`` objects sorted by descending similarity score.
        """
        if self.count() == 0:
            return []

        k = min(top_k or settings.top_k_results, self.count())
        query_embedding = self.embedding_service.embed_query(query)

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        raw = self._collection.query(**query_kwargs)

        ids = (raw.get("ids") or [[]])[0]
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]

        results: list[SearchResult] = []
        for chunk_id, text, meta, dist in zip(ids, docs, metas, dists):
            # Chroma cosine distance = 1 - cosine_similarity
            similarity = max(0.0, min(1.0, 1.0 - float(dist)))
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    text=text,
                    score=round(similarity, 4),
                    metadata=meta or {},
                )
            )

        return results
