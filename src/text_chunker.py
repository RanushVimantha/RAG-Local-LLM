"""Text chunking strategies for splitting documents into optimal-sized pieces."""

from dataclasses import dataclass

import structlog
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.config import settings

logger = structlog.get_logger(__name__)


@dataclass
class Chunk:
    """A text chunk with metadata about its source."""

    text: str
    source_file: str
    page_number: int
    chunk_index: int
    total_chunks: int = 0


class TextChunker:
    """Splits document text into chunks using configurable strategies."""

    STRATEGIES = {
        "recursive": "Recursive Character (Default — splits by paragraphs, sentences, characters)",
        "fixed": "Fixed Size (splits at exact character count)",
        "sentence": "Sentence-based (splits at sentence boundaries)",
        "paragraph": "Paragraph-based (splits at paragraph boundaries)",
    }

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        strategy: str = "recursive",
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.strategy = strategy
        self._splitter = self._create_splitter()

    def _create_splitter(self) -> CharacterTextSplitter | RecursiveCharacterTextSplitter:
        """Create the appropriate text splitter based on strategy."""
        if self.strategy == "recursive":
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len,
            )
        elif self.strategy == "fixed":
            return CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separator="",
                length_function=len,
            )
        elif self.strategy == "sentence":
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=[". ", "! ", "? ", "\n", " ", ""],
                length_function=len,
            )
        elif self.strategy == "paragraph":
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n\n", "\n\n", "\n", " ", ""],
                length_function=len,
            )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}. Use: {list(self.STRATEGIES)}")

    def chunk_pages(self, pages: list[dict], source_file: str) -> list[Chunk]:
        """Split extracted pages into chunks with metadata.

        Args:
            pages: List of page dicts from DocumentProcessor [{"page": 1, "text": "..."}]
            source_file: Original filename for metadata tracking
        """
        all_chunks: list[Chunk] = []

        for page_data in pages:
            page_num = page_data["page"]
            text = page_data["text"].strip()

            if not text:
                continue

            text_pieces = self._splitter.split_text(text)

            for piece in text_pieces:
                chunk = Chunk(
                    text=piece,
                    source_file=source_file,
                    page_number=page_num,
                    chunk_index=len(all_chunks),
                )
                all_chunks.append(chunk)

        # Set total_chunks on all chunks
        for chunk in all_chunks:
            chunk.total_chunks = len(all_chunks)

        logger.info(
            "chunking_complete",
            source=source_file,
            strategy=self.strategy,
            total_chunks=len(all_chunks),
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )
        return all_chunks

    def preview_chunks(self, pages: list[dict], source_file: str, max_preview: int = 5) -> list[dict]:
        """Generate a preview of how the document would be chunked."""
        chunks = self.chunk_pages(pages, source_file)
        return [
            {
                "index": c.chunk_index,
                "page": c.page_number,
                "length": len(c.text),
                "preview": c.text[:200] + ("..." if len(c.text) > 200 else ""),
            }
            for c in chunks[:max_preview]
        ]
