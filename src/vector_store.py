"""ChromaDB vector storage and semantic search operations."""

from typing import Any

import chromadb
import structlog

from src.config import settings
from src.text_chunker import Chunk

logger = structlog.get_logger(__name__)


class VectorStore:
    """Manages ChromaDB for storing and searching document embeddings.

    All data is persisted to disk — survives application restarts.
    """

    def __init__(self, persist_dir: str | None = None, collection_name: str | None = None):
        self.persist_dir = persist_dir or str(settings.chromadb_dir)
        self.collection_name = collection_name or settings.collection_name
        self._client: chromadb.PersistentClient | None = None
        self._collection: chromadb.Collection | None = None

    @property
    def client(self) -> chromadb.PersistentClient:
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            logger.info("chromadb_initialized", path=self.persist_dir)
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "collection_ready",
                name=self.collection_name,
                count=self._collection.count(),
            )
        return self._collection

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Store chunks with their embeddings in ChromaDB.

        Args:
            chunks: Text chunks with metadata from TextChunker.
            embeddings: Corresponding embedding vectors.
        """
        if not chunks:
            return

        ids = [f"{chunks[0].source_file}_{c.chunk_index}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "source": c.source_file,
                "page": c.page_number,
                "chunk_index": c.chunk_index,
                "total_chunks": c.total_chunks,
            }
            for c in chunks
        ]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(
            "chunks_stored",
            source=chunks[0].source_file,
            count=len(chunks),
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        source_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Perform semantic search for the most relevant chunks.

        Args:
            query_embedding: Vector embedding of the query.
            top_k: Number of results to return.
            source_filter: Optional filename to restrict search to one document.

        Returns:
            List of result dicts with keys: text, source, page, score
        """
        top_k = top_k or settings.top_k
        where = {"source": source_filter} if source_filter else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # ChromaDB cosine distance: 0 = identical, 2 = opposite
                # Convert to similarity: 1 - (distance / 2)
                similarity = 1 - (distance / 2)
                search_results.append(
                    {
                        "text": doc,
                        "source": meta["source"],
                        "page": meta["page"],
                        "chunk_index": meta["chunk_index"],
                        "score": round(similarity, 4),
                    }
                )

        logger.info(
            "search_complete",
            results=len(search_results),
            top_score=search_results[0]["score"] if search_results else 0,
        )
        return search_results

    def delete_by_source(self, source_file: str) -> int:
        """Delete all chunks belonging to a specific document.

        Returns the number of chunks deleted.
        """
        # Get IDs for the source first
        existing = self.collection.get(where={"source": source_file})
        count = len(existing["ids"])

        if count > 0:
            self.collection.delete(where={"source": source_file})
            logger.info("chunks_deleted", source=source_file, count=count)

        return count

    def get_all_sources(self) -> list[str]:
        """Get a list of all unique source filenames in the store."""
        results = self.collection.get(include=["metadatas"])
        sources = set()
        if results["metadatas"]:
            for meta in results["metadatas"]:
                sources.add(meta["source"])
        return sorted(sources)

    def get_stats(self) -> dict:
        """Get vector store statistics."""
        total = self.collection.count()
        sources = self.get_all_sources()
        return {
            "total_chunks": total,
            "total_documents": len(sources),
            "sources": sources,
            "collection_name": self.collection_name,
        }

    def reset(self) -> None:
        """Delete the entire collection and recreate it."""
        self.client.delete_collection(self.collection_name)
        self._collection = None
        logger.warning("collection_reset", name=self.collection_name)
