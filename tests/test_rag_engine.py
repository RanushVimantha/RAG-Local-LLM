"""Tests for rag_engine.py — RAG pipeline with mocked LLM."""

from unittest.mock import MagicMock, patch

import pytest

from src.embedding_manager import EmbeddingManager
from src.rag_engine import RAGEngine, RAGResponse
from src.text_chunker import Chunk
from src.vector_store import VectorStore


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate.return_value = "The revenue was $4.2 billion according to the report (page 12)."
    llm.generate_stream.return_value = iter(["The ", "revenue ", "was ", "$4.2B."])
    return llm


@pytest.fixture
def populated_store(tmp_data_dir):
    """Vector store with pre-loaded test data."""
    store = VectorStore(
        persist_dir=str(tmp_data_dir / "chromadb"),
        collection_name="test_rag",
    )
    # Use real-ish fake embeddings
    chunks = [
        Chunk(text="Q3 revenue reached $4.2 billion, a 12% increase.", source_file="report.pdf", page_number=12, chunk_index=0, total_chunks=3),
        Chunk(text="The company opened offices in 5 new countries.", source_file="report.pdf", page_number=5, chunk_index=1, total_chunks=3),
        Chunk(text="Python is a popular programming language.", source_file="guide.txt", page_number=1, chunk_index=2, total_chunks=3),
    ]
    embeddings = [
        [0.9, 0.1, 0.05, 0.02],
        [0.1, 0.8, 0.05, 0.02],
        [0.05, 0.05, 0.9, 0.02],
    ]
    store.add_chunks(chunks, embeddings)
    return store


@pytest.fixture
def mock_embedder():
    embedder = MagicMock(spec=EmbeddingManager)
    embedder.embed_query.return_value = [0.85, 0.1, 0.05, 0.02]
    embedder.embed_texts.return_value = [[0.85, 0.1, 0.05, 0.02]]
    embedder.model_name = "test-model"
    return embedder


@pytest.fixture
def rag_engine(populated_store, mock_embedder, mock_llm):
    return RAGEngine(
        vector_store=populated_store,
        embedding_manager=mock_embedder,
        llm_manager=mock_llm,
        use_reranker=False,  # Skip reranker in tests
    )


class TestRetrieval:
    def test_retrieve_returns_results(self, rag_engine):
        results = rag_engine.retrieve("What was the revenue?", top_k=2)
        assert len(results) > 0
        assert "text" in results[0]
        assert "source" in results[0]
        assert "score" in results[0]

    def test_retrieve_respects_top_k(self, rag_engine):
        results = rag_engine.retrieve("revenue", top_k=1)
        assert len(results) <= 1

    def test_retrieve_with_source_filter(self, rag_engine):
        results = rag_engine.retrieve("revenue", source_filter="report.pdf")
        assert all(r["source"] == "report.pdf" for r in results)


class TestQuery:
    def test_query_returns_response(self, rag_engine):
        response = rag_engine.query("What was Q3 revenue?")
        assert isinstance(response, RAGResponse)
        assert response.answer != ""
        assert response.query == "What was Q3 revenue?"
        assert isinstance(response.sources, list)
        assert response.generation_time_ms >= 0

    def test_query_calls_llm(self, rag_engine, mock_llm):
        rag_engine.query("What was Q3 revenue?")
        mock_llm.generate.assert_called()

    def test_query_with_chat_history(self, rag_engine, mock_llm):
        # When chat history is provided, the engine should rephrase
        mock_llm.generate.side_effect = [
            "What was the company revenue in Q3?",  # rephrase
            "The revenue was $4.2B.",  # answer
        ]
        response = rag_engine.query(
            "How about Q3?",
            chat_history="Human: Tell me about the company.\nAssistant: It's a large corporation.",
        )
        assert response.answer != ""
        assert mock_llm.generate.call_count == 2


class TestQueryStream:
    def test_stream_returns_generator_and_sources(self, rag_engine):
        gen, sources = rag_engine.query_stream("What is revenue?")
        assert hasattr(gen, '__next__')
        assert isinstance(sources, list)

    def test_stream_yields_tokens(self, rag_engine, mock_llm):
        mock_llm.generate_stream.return_value = iter(["Hello", " world"])
        gen, _ = rag_engine.query_stream("Test?")
        tokens = list(gen)
        assert len(tokens) > 0


class TestHybridSearch:
    def test_hybrid_search_combines_results(self, rag_engine):
        # Hybrid search should still return results
        results = rag_engine.retrieve("revenue Q3", use_hybrid=True, top_k=2)
        assert len(results) > 0

    def test_non_hybrid_search(self, rag_engine):
        results = rag_engine.retrieve("revenue", use_hybrid=False, top_k=2)
        assert len(results) > 0
