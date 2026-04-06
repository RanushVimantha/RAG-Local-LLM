"""Tests for text_chunker.py — text splitting strategies."""

import pytest

from src.text_chunker import Chunk, TextChunker


@pytest.fixture
def sample_pages():
    return [
        {"page": 1, "text": "This is the first page with some content. " * 20},
        {"page": 2, "text": "Second page has different content here. " * 20},
    ]


@pytest.fixture
def chunker():
    return TextChunker(chunk_size=200, chunk_overlap=20, strategy="recursive")


class TestChunking:
    def test_basic_chunking(self, chunker, sample_pages):
        chunks = chunker.chunk_pages(sample_pages, "test.txt")
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_metadata(self, chunker, sample_pages):
        chunks = chunker.chunk_pages(sample_pages, "test.txt")
        for chunk in chunks:
            assert chunk.source_file == "test.txt"
            assert chunk.page_number in [1, 2]
            assert chunk.chunk_index >= 0
            assert chunk.total_chunks == len(chunks)

    def test_chunk_size_respected(self, sample_pages):
        chunker = TextChunker(chunk_size=100, chunk_overlap=10, strategy="fixed")
        chunks = chunker.chunk_pages(sample_pages, "test.txt")
        for chunk in chunks:
            # Allow some tolerance for splitting
            assert len(chunk.text) <= 150

    def test_chunk_indices_sequential(self, chunker, sample_pages):
        chunks = chunker.chunk_pages(sample_pages, "test.txt")
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_empty_pages_handled(self, chunker):
        pages = [{"page": 1, "text": ""}, {"page": 2, "text": "   "}]
        chunks = chunker.chunk_pages(pages, "empty.txt")
        assert len(chunks) == 0

    def test_single_short_text(self):
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        pages = [{"page": 1, "text": "Short text."}]
        chunks = chunker.chunk_pages(pages, "short.txt")
        assert len(chunks) == 1
        assert chunks[0].text == "Short text."

    def test_total_chunks_set(self, chunker, sample_pages):
        chunks = chunker.chunk_pages(sample_pages, "test.txt")
        total = len(chunks)
        for chunk in chunks:
            assert chunk.total_chunks == total


class TestStrategies:
    def test_recursive_strategy(self, sample_pages):
        chunker = TextChunker(chunk_size=200, strategy="recursive")
        chunks = chunker.chunk_pages(sample_pages, "test.txt")
        assert len(chunks) > 0

    def test_fixed_strategy(self, sample_pages):
        chunker = TextChunker(chunk_size=200, strategy="fixed")
        chunks = chunker.chunk_pages(sample_pages, "test.txt")
        assert len(chunks) > 0

    def test_sentence_strategy(self, sample_pages):
        chunker = TextChunker(chunk_size=200, strategy="sentence")
        chunks = chunker.chunk_pages(sample_pages, "test.txt")
        assert len(chunks) > 0

    def test_paragraph_strategy(self, sample_pages):
        chunker = TextChunker(chunk_size=200, strategy="paragraph")
        chunks = chunker.chunk_pages(sample_pages, "test.txt")
        assert len(chunks) > 0

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            TextChunker(strategy="nonexistent")


class TestPreview:
    def test_preview_chunks(self, chunker, sample_pages):
        preview = chunker.preview_chunks(sample_pages, "test.txt", max_preview=3)
        assert len(preview) <= 3
        for p in preview:
            assert "index" in p
            assert "page" in p
            assert "length" in p
            assert "preview" in p

    def test_preview_truncation(self):
        chunker = TextChunker(chunk_size=500)
        long_text = "A" * 1000
        pages = [{"page": 1, "text": long_text}]
        preview = chunker.preview_chunks(pages, "long.txt")
        for p in preview:
            assert len(p["preview"]) <= 203  # 200 + "..."
