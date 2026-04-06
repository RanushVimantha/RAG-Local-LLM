"""SQLite-based conversation and message persistence."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import structlog

from src.config import settings

logger = structlog.get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT 'New Conversation',
    model_name TEXT NOT NULL DEFAULT 'mistral',
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    sources TEXT,
    tokens_used INTEGER,
    generation_time_ms INTEGER,
    retrieval_time_ms INTEGER,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
"""


class ConversationManager:
    """Manages conversation persistence in SQLite.

    Stores conversations, messages, and source citations so users
    can revisit past Q&A sessions.
    """

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

    # --- Conversations ---

    def create_conversation(self, title: str = "New Conversation", model_name: str = "mistral") -> str:
        """Create a new conversation. Returns the conversation ID."""
        conv_id = uuid4().hex
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, model_name) VALUES (?, ?, ?)",
                (conv_id, title, model_name),
            )
        logger.info("conversation_created", id=conv_id, title=title)
        return conv_id

    def get_conversations(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get all conversations, most recent first."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT c.*, COUNT(m.id) as message_count
                   FROM conversations c
                   LEFT JOIN messages m ON m.conversation_id = c.id
                   GROUP BY c.id
                   ORDER BY c.updated_at DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_conversation(self, conv_id: str) -> dict[str, Any] | None:
        """Get a single conversation by ID."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?", (conv_id,)
            ).fetchone()
        return dict(row) if row else None

    def update_title(self, conv_id: str, title: str) -> None:
        """Update a conversation's title."""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = datetime('now') WHERE id = ?",
                (title, conv_id),
            )

    def delete_conversation(self, conv_id: str) -> bool:
        """Delete a conversation and all its messages (CASCADE)."""
        with self._get_conn() as conn:
            cursor = conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("conversation_deleted", id=conv_id)
        return deleted

    # --- Messages ---

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: list[dict] | None = None,
        tokens_used: int | None = None,
        generation_time_ms: int | None = None,
        retrieval_time_ms: int | None = None,
    ) -> int:
        """Add a message to a conversation. Returns the message ID."""
        sources_json = json.dumps(sources) if sources else None
        with self._get_conn() as conn:
            cursor = conn.execute(
                """INSERT INTO messages
                   (conversation_id, role, content, sources, tokens_used, generation_time_ms, retrieval_time_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (conversation_id, role, content, sources_json, tokens_used, generation_time_ms, retrieval_time_ms),
            )
            # Touch the conversation's updated_at
            conn.execute(
                "UPDATE conversations SET updated_at = datetime('now') WHERE id = ?",
                (conversation_id,),
            )
        return cursor.lastrowid

    def get_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        """Get all messages for a conversation, in chronological order."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
                (conversation_id,),
            ).fetchall()

        messages = []
        for row in rows:
            msg = dict(row)
            if msg["sources"]:
                msg["sources"] = json.loads(msg["sources"])
            messages.append(msg)
        return messages

    def get_chat_history_text(self, conversation_id: str, window: int = 5) -> str:
        """Get recent messages formatted as a text string for the LLM.

        Args:
            conversation_id: The conversation to get history from.
            window: Number of recent Q&A pairs to include.
        """
        messages = self.get_messages(conversation_id)
        recent = messages[-(window * 2):]
        if not recent:
            return ""
        lines = []
        for msg in recent:
            role = "Human" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    # --- Auto-title ---

    def auto_title(self, conv_id: str) -> None:
        """Set the conversation title based on the first user message."""
        messages = self.get_messages(conv_id)
        user_msgs = [m for m in messages if m["role"] == "user"]
        if user_msgs:
            first_q = user_msgs[0]["content"]
            title = first_q[:60] + ("..." if len(first_q) > 60 else "")
            self.update_title(conv_id, title)

    # --- Export ---

    def export_markdown(self, conversation_id: str) -> str:
        """Export a conversation as Markdown."""
        conv = self.get_conversation(conversation_id)
        if not conv:
            return ""

        messages = self.get_messages(conversation_id)
        lines = [
            f"# {conv['title']}",
            f"*Created: {conv['created_at']} | Model: {conv['model_name']}*",
            "",
        ]

        for msg in messages:
            if msg["role"] == "user":
                lines.append(f"## Question")
                lines.append(msg["content"])
            else:
                lines.append(f"## Answer")
                lines.append(msg["content"])
                if msg.get("sources"):
                    lines.append("")
                    lines.append("**Sources:**")
                    for src in msg["sources"]:
                        lines.append(f"- {src['source']} (Page {src['page']}, Score: {src['score']})")
            lines.append("")

        return "\n".join(lines)

    def export_json(self, conversation_id: str) -> dict[str, Any]:
        """Export a conversation as a JSON-serializable dict."""
        conv = self.get_conversation(conversation_id)
        if not conv:
            return {}

        messages = self.get_messages(conversation_id)
        return {
            "conversation": conv,
            "messages": messages,
        }
