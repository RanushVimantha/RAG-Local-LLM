"""Document loading and text extraction for PDF, TXT, Markdown, and DOCX files."""

import logging
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


class DocumentProcessor:
    """Extracts raw text from various document formats."""

    def extract_text(self, file_path: Path) -> list[dict]:
        """Extract text from a document file.

        Returns a list of page dicts: [{"page": 1, "text": "..."}, ...]
        For non-paginated formats, returns a single page.
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
            )

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.stat().st_size == 0:
            raise ValueError(f"File is empty: {file_path.name}")

        extractor = {
            ".pdf": self._extract_pdf,
            ".txt": self._extract_text,
            ".md": self._extract_text,
            ".docx": self._extract_docx,
        }[ext]

        logger.info("extracting_text", file=file_path.name, format=ext)
        pages = extractor(file_path)

        # Filter out empty pages
        pages = [p for p in pages if p["text"].strip()]
        if not pages:
            raise ValueError(f"No extractable text found in: {file_path.name}")

        total_chars = sum(len(p["text"]) for p in pages)
        logger.info(
            "extraction_complete",
            file=file_path.name,
            pages=len(pages),
            characters=total_chars,
        )
        return pages

    def _extract_pdf(self, file_path: Path) -> list[dict]:
        """Extract text from PDF, page by page."""
        from PyPDF2 import PdfReader

        reader = PdfReader(str(file_path))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append({"page": i + 1, "text": text})
        return pages

    def _extract_text(self, file_path: Path) -> list[dict]:
        """Extract text from plain text or markdown files."""
        text = file_path.read_text(encoding="utf-8")
        return [{"page": 1, "text": text}]

    def _extract_docx(self, file_path: Path) -> list[dict]:
        """Extract text from DOCX files."""
        from docx import Document

        doc = Document(str(file_path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        full_text = "\n\n".join(paragraphs)
        return [{"page": 1, "text": full_text}]

    @staticmethod
    def is_supported(filename: str) -> bool:
        """Check if a file extension is supported."""
        return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS

    @staticmethod
    def get_file_type(filename: str) -> str:
        """Get the file type from a filename."""
        return Path(filename).suffix.lower().lstrip(".")
