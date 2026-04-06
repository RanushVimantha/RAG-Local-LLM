"""SQLite-based document metadata registry for tracking uploaded documents."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import structlog

from src.config import settings

logger = structlog.get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    file_type TEXT NOT NULL,
    page_count INTEGER DEFAULT 0,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    character_count INTEGER NOT NULL DEFAULT 0,
    chunk_size_used INTEGER NOT NULL DEFAULT 500,
    chunk_overlap_used INTEGER NOT NULL DEFAULT 50,
    embedding_model TEXT NOT NULL DEFAULT 'all-MiniLM-L6-v2',
    status TEXT NOT NULL DEFAULT 'processing' CHECK (status IN ('processing', 'ready', 'error')),
    error_message TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
"""


class DocumentRegistry:
    """Tracks uploaded document metadata in SQLite."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or settings.db_path
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_conn() as conn:
            conn.executescript(SCHEMA)

    def register(
        self,
        filename: str,
        file_path: str,
        file_size_bytes: int,
        file_type: str,
    ) -> str:
        """Register a new document. Returns the document ID."""
        doc_id = uuid4().hex
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO documents (id, filename, file_path, file_size_bytes, file_type)
                   VALUES (?, ?, ?, ?, ?)""",
                (doc_id, filename, file_path, file_size_bytes, file_type),
            )
        logger.info("document_registered", id=doc_id, filename=filename)
        return doc_id

    def update_status(
        self,
        doc_id: str,
        status: str,
        page_count: int = 0,
        chunk_count: int = 0,
        character_count: int = 0,
        chunk_size_used: int = 500,
        chunk_overlap_used: int = 50,
        embedding_model: str = "all-MiniLM-L6-v2",
        error_message: str | None = None,
    ) -> None:
        """Update document processing status and metadata."""
        with self._get_conn() as conn:
            conn.execute(
                """UPDATE documents SET
                    status = ?, page_count = ?, chunk_count = ?,
                    character_count = ?, chunk_size_used = ?,
                    chunk_overlap_used = ?, embedding_model = ?,
                    error_message = ?
                   WHERE id = ?""",
                (
                    status,
                    page_count,
                    chunk_count,
                    character_count,
                    chunk_size_used,
                    chunk_overlap_used,
                    embedding_model,
                    error_message,
                    doc_id,
                ),
            )
        logger.info("document_status_updated", id=doc_id, status=status)

    def get_all(self) -> list[dict[str, Any]]:
        """Get all registered documents."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_by_id(self, doc_id: str) -> dict[str, Any] | None:
        """Get a single document by ID."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id = ?", (doc_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_by_filename(self, filename: str) -> dict[str, Any] | None:
        """Get a document by filename."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE filename = ? ORDER BY created_at DESC LIMIT 1",
                (filename,),
            ).fetchone()
        return dict(row) if row else None

    def delete(self, doc_id: str) -> bool:
        """Delete a document record. Returns True if found and deleted."""
        with self._get_conn() as conn:
            cursor = conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("document_deleted", id=doc_id)
        return deleted

    def get_stats(self) -> dict[str, Any]:
        """Get summary statistics across all documents."""
        with self._get_conn() as conn:
            row = conn.execute(
                """SELECT
                    COUNT(*) as total_documents,
                    COALESCE(SUM(chunk_count), 0) as total_chunks,
                    COALESCE(SUM(character_count), 0) as total_characters,
                    COALESCE(SUM(file_size_bytes), 0) as total_size_bytes
                   FROM documents WHERE status = 'ready'"""
            ).fetchone()
        return dict(row)
