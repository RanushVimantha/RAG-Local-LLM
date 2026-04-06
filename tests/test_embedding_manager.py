"""Tests for embedding_manager.py — local embedding generation.

Note: These tests load the actual sentence-transformers model (~80MB).
They are marked as integration tests and skipped in CI by default.
"""

import pytest

from src.embedding_manager import EmbeddingManager


@pytest.fixture
def embedder():
    return EmbeddingManager(model_name="all-MiniLM-L6-v2")


@pytest.mark.integration
class TestEmbeddings:
    def test_embed_texts(self, embedder):
        texts = ["Hello world", "Goodbye world"]
        embeddings = embedder.embed_texts(texts)
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 384  # MiniLM dimensions

    def test_embed_query(self, embedder):
        embedding = embedder.embed_query("What is AI?")
        assert len(embedding) == 384

    def test_consistent_embeddings(self, embedder):
        e1 = embedder.embed_query("test sentence")
        e2 = embedder.embed_query("test sentence")
        assert e1 == e2

    def test_different_texts_different_embeddings(self, embedder):
        e1 = embedder.embed_query("cats are pets")
        e2 = embedder.embed_query("quantum physics equations")
        assert e1 != e2

    def test_empty_list(self, embedder):
        assert embedder.embed_texts([]) == []

    def test_dimensions_property(self, embedder):
        assert embedder.dimensions == 384

    def test_batch_embedding(self, embedder):
        texts = [f"Document number {i}" for i in range(10)]
        embeddings = embedder.embed_texts(texts, batch_size=4)
        assert len(embeddings) == 10
        assert all(len(e) == 384 for e in embeddings)
