"""Tests for vector_store.py — ChromaDB operations."""

import pytest

from src.text_chunker import Chunk
from src.vector_store import VectorStore


@pytest.fixture
def vector_store(tmp_data_dir):
    return VectorStore(
        persist_dir=str(tmp_data_dir / "chromadb"),
        collection_name="test_collection",
    )


@pytest.fixture
def sample_chunks():
    return [
        Chunk(text="Revenue was $4.2 billion in Q3.", source_file="report.pdf", page_number=1, chunk_index=0, total_chunks=3),
        Chunk(text="The company expanded to 5 new markets.", source_file="report.pdf", page_number=2, chunk_index=1, total_chunks=3),
        Chunk(text="Employee count reached 10,000.", source_file="report.pdf", page_number=3, chunk_index=2, total_chunks=3),
    ]


@pytest.fixture
def sample_embeddings():
    # Simple fake embeddings (3 chunks, 4 dimensions each)
    return [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 0.1, 0.2, 0.3],
    ]


class TestAddAndSearch:
    def test_add_chunks(self, vector_store, sample_chunks, sample_embeddings):
        vector_store.add_chunks(sample_chunks, sample_embeddings)
        stats = vector_store.get_stats()
        assert stats["total_chunks"] == 3

    def test_search(self, vector_store, sample_chunks, sample_embeddings):
        vector_store.add_chunks(sample_chunks, sample_embeddings)
        results = vector_store.search([0.1, 0.2, 0.3, 0.4], top_k=2)
        assert len(results) == 2
        assert "text" in results[0]
        assert "source" in results[0]
        assert "page" in results[0]
        assert "score" in results[0]

    def test_search_with_source_filter(self, vector_store, sample_chunks, sample_embeddings):
        vector_store.add_chunks(sample_chunks, sample_embeddings)

        # Add chunks from another source
        other_chunks = [
            Chunk(text="Weather is sunny.", source_file="weather.txt", page_number=1, chunk_index=0, total_chunks=1),
        ]
        vector_store.add_chunks(other_chunks, [[0.2, 0.3, 0.4, 0.5]])

        results = vector_store.search([0.1, 0.2, 0.3, 0.4], top_k=10, source_filter="report.pdf")
        assert all(r["source"] == "report.pdf" for r in results)

    def test_search_empty_store(self, vector_store):
        results = vector_store.search([0.1, 0.2, 0.3, 0.4], top_k=5)
        assert len(results) == 0

    def test_add_empty_chunks(self, vector_store):
        vector_store.add_chunks([], [])
        assert vector_store.get_stats()["total_chunks"] == 0


class TestDeleteAndManage:
    def test_delete_by_source(self, vector_store, sample_chunks, sample_embeddings):
        vector_store.add_chunks(sample_chunks, sample_embeddings)
        count = vector_store.delete_by_source("report.pdf")
        assert count == 3
        assert vector_store.get_stats()["total_chunks"] == 0

    def test_delete_nonexistent_source(self, vector_store):
        count = vector_store.delete_by_source("nonexistent.pdf")
        assert count == 0

    def test_get_all_sources(self, vector_store, sample_chunks, sample_embeddings):
        vector_store.add_chunks(sample_chunks, sample_embeddings)
        other_chunks = [
            Chunk(text="Other doc.", source_file="other.txt", page_number=1, chunk_index=0, total_chunks=1),
        ]
        vector_store.add_chunks(other_chunks, [[0.5, 0.5, 0.5, 0.5]])

        sources = vector_store.get_all_sources()
        assert "report.pdf" in sources
        assert "other.txt" in sources

    def test_get_stats(self, vector_store, sample_chunks, sample_embeddings):
        vector_store.add_chunks(sample_chunks, sample_embeddings)
        stats = vector_store.get_stats()
        assert stats["total_chunks"] == 3
        assert stats["total_documents"] == 1
        assert "report.pdf" in stats["sources"]

    def test_reset(self, vector_store, sample_chunks, sample_embeddings):
        vector_store.add_chunks(sample_chunks, sample_embeddings)
        vector_store.reset()
        stats = vector_store.get_stats()
        assert stats["total_chunks"] == 0
