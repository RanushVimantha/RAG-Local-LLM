"""Tests for document_registry.py — SQLite document metadata tracking."""

import pytest

from src.document_registry import DocumentRegistry


@pytest.fixture
def registry(tmp_data_dir):
    return DocumentRegistry(db_path=tmp_data_dir / "db" / "test.db")


class TestRegisterAndRetrieve:
    def test_register_document(self, registry):
        doc_id = registry.register("test.pdf", "/path/test.pdf", 1024, "pdf")
        assert doc_id is not None
        assert len(doc_id) == 32  # UUID hex

    def test_get_by_id(self, registry):
        doc_id = registry.register("test.pdf", "/path/test.pdf", 1024, "pdf")
        doc = registry.get_by_id(doc_id)
        assert doc is not None
        assert doc["filename"] == "test.pdf"
        assert doc["file_size_bytes"] == 1024
        assert doc["status"] == "processing"

    def test_get_by_filename(self, registry):
        registry.register("report.pdf", "/path/report.pdf", 2048, "pdf")
        doc = registry.get_by_filename("report.pdf")
        assert doc is not None
        assert doc["filename"] == "report.pdf"

    def test_get_nonexistent(self, registry):
        assert registry.get_by_id("nonexistent") is None
        assert registry.get_by_filename("nonexistent.pdf") is None

    def test_get_all(self, registry):
        registry.register("a.pdf", "/a.pdf", 100, "pdf")
        registry.register("b.txt", "/b.txt", 200, "txt")
        docs = registry.get_all()
        assert len(docs) == 2


class TestUpdateAndDelete:
    def test_update_status(self, registry):
        doc_id = registry.register("test.pdf", "/path/test.pdf", 1024, "pdf")
        registry.update_status(
            doc_id, status="ready", page_count=5, chunk_count=20,
            character_count=5000, chunk_size_used=500, chunk_overlap_used=50,
        )
        doc = registry.get_by_id(doc_id)
        assert doc["status"] == "ready"
        assert doc["page_count"] == 5
        assert doc["chunk_count"] == 20
        assert doc["character_count"] == 5000

    def test_update_error_status(self, registry):
        doc_id = registry.register("bad.pdf", "/bad.pdf", 100, "pdf")
        registry.update_status(doc_id, status="error", error_message="Corrupt file")
        doc = registry.get_by_id(doc_id)
        assert doc["status"] == "error"
        assert doc["error_message"] == "Corrupt file"

    def test_delete(self, registry):
        doc_id = registry.register("test.pdf", "/test.pdf", 100, "pdf")
        assert registry.delete(doc_id) is True
        assert registry.get_by_id(doc_id) is None

    def test_delete_nonexistent(self, registry):
        assert registry.delete("nonexistent") is False


class TestStats:
    def test_stats_empty(self, registry):
        stats = registry.get_stats()
        assert stats["total_documents"] == 0
        assert stats["total_chunks"] == 0

    def test_stats_with_docs(self, registry):
        doc_id = registry.register("test.pdf", "/test.pdf", 1024, "pdf")
        registry.update_status(doc_id, status="ready", chunk_count=10, character_count=5000)
        stats = registry.get_stats()
        assert stats["total_documents"] == 1
        assert stats["total_chunks"] == 10
        assert stats["total_characters"] == 5000
