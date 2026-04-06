"""Tests for document_processor.py — file loading and text extraction."""

import tempfile
from pathlib import Path

import pytest

from src.document_processor import DocumentProcessor, SUPPORTED_EXTENSIONS


@pytest.fixture
def processor():
    return DocumentProcessor()


class TestExtractText:
    def test_extract_txt(self, processor, sample_txt):
        pages = processor.extract_text(sample_txt)
        assert len(pages) == 1
        assert pages[0]["page"] == 1
        assert "Artificial Intelligence" in pages[0]["text"]
        assert len(pages[0]["text"]) > 100

    def test_extract_markdown(self, processor, sample_md):
        pages = processor.extract_text(sample_md)
        assert len(pages) == 1
        assert "Weather Monitoring API" in pages[0]["text"]

    def test_unsupported_file_type(self, processor, tmp_path):
        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("content")
        with pytest.raises(ValueError, match="Unsupported file type"):
            processor.extract_text(bad_file)

    def test_file_not_found(self, processor):
        with pytest.raises(FileNotFoundError):
            processor.extract_text(Path("/nonexistent/file.txt"))

    def test_empty_file(self, processor, tmp_path):
        empty = tmp_path / "empty.txt"
        empty.write_text("")
        with pytest.raises(ValueError, match="empty"):
            processor.extract_text(empty)

    def test_whitespace_only_file(self, processor, tmp_path):
        ws_file = tmp_path / "whitespace.txt"
        ws_file.write_text("   \n\n   \t  ")
        with pytest.raises(ValueError, match="No extractable text"):
            processor.extract_text(ws_file)

    def test_extract_preserves_content(self, processor, tmp_path):
        test_file = tmp_path / "test.txt"
        content = "Line one.\nLine two.\nLine three."
        test_file.write_text(content)
        pages = processor.extract_text(test_file)
        assert pages[0]["text"] == content


class TestHelpers:
    def test_is_supported(self, processor):
        assert processor.is_supported("doc.pdf") is True
        assert processor.is_supported("doc.txt") is True
        assert processor.is_supported("doc.md") is True
        assert processor.is_supported("doc.docx") is True
        assert processor.is_supported("doc.xlsx") is False
        assert processor.is_supported("doc.jpg") is False

    def test_get_file_type(self, processor):
        assert processor.get_file_type("report.pdf") == "pdf"
        assert processor.get_file_type("notes.txt") == "txt"
        assert processor.get_file_type("readme.MD") == "md"
        assert processor.get_file_type("doc.DOCX") == "docx"

    def test_supported_extensions_set(self):
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".txt" in SUPPORTED_EXTENSIONS
        assert ".md" in SUPPORTED_EXTENSIONS
        assert ".docx" in SUPPORTED_EXTENSIONS
