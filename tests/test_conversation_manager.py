"""Tests for conversation_manager.py — conversation persistence."""

import json

import pytest

from src.conversation_manager import ConversationManager


@pytest.fixture
def conv_mgr(tmp_data_dir):
    return ConversationManager(db_path=tmp_data_dir / "db" / "test.db")


class TestConversations:
    def test_create_conversation(self, conv_mgr):
        conv_id = conv_mgr.create_conversation("Test Chat")
        assert conv_id is not None
        assert len(conv_id) == 32

    def test_get_conversation(self, conv_mgr):
        conv_id = conv_mgr.create_conversation("My Chat", model_name="mistral")
        conv = conv_mgr.get_conversation(conv_id)
        assert conv["title"] == "My Chat"
        assert conv["model_name"] == "mistral"

    def test_get_conversations_list(self, conv_mgr):
        conv_mgr.create_conversation("Chat 1")
        conv_mgr.create_conversation("Chat 2")
        convs = conv_mgr.get_conversations()
        assert len(convs) == 2

    def test_update_title(self, conv_mgr):
        conv_id = conv_mgr.create_conversation("Old Title")
        conv_mgr.update_title(conv_id, "New Title")
        conv = conv_mgr.get_conversation(conv_id)
        assert conv["title"] == "New Title"

    def test_delete_conversation(self, conv_mgr):
        conv_id = conv_mgr.create_conversation("To Delete")
        assert conv_mgr.delete_conversation(conv_id) is True
        assert conv_mgr.get_conversation(conv_id) is None

    def test_delete_nonexistent(self, conv_mgr):
        assert conv_mgr.delete_conversation("nonexistent") is False


class TestMessages:
    def test_add_and_get_messages(self, conv_mgr):
        conv_id = conv_mgr.create_conversation("Test")
        conv_mgr.add_message(conv_id, "user", "Hello?")
        conv_mgr.add_message(conv_id, "assistant", "Hi there!")
        messages = conv_mgr.get_messages(conv_id)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello?"
        assert messages[1]["role"] == "assistant"

    def test_message_with_sources(self, conv_mgr):
        conv_id = conv_mgr.create_conversation("Test")
        sources = [{"source": "doc.pdf", "page": 1, "text": "data", "score": 0.95}]
        conv_mgr.add_message(conv_id, "assistant", "Answer", sources=sources)
        messages = conv_mgr.get_messages(conv_id)
        assert messages[0]["sources"] == sources

    def test_cascade_delete(self, conv_mgr):
        conv_id = conv_mgr.create_conversation("Test")
        conv_mgr.add_message(conv_id, "user", "Question")
        conv_mgr.add_message(conv_id, "assistant", "Answer")
        conv_mgr.delete_conversation(conv_id)
        assert conv_mgr.get_messages(conv_id) == []


class TestChatHistory:
    def test_get_chat_history_text(self, conv_mgr):
        conv_id = conv_mgr.create_conversation("Test")
        conv_mgr.add_message(conv_id, "user", "What is AI?")
        conv_mgr.add_message(conv_id, "assistant", "AI is artificial intelligence.")
        history = conv_mgr.get_chat_history_text(conv_id)
        assert "Human: What is AI?" in history
        assert "Assistant: AI is artificial intelligence." in history

    def test_empty_history(self, conv_mgr):
        conv_id = conv_mgr.create_conversation("Empty")
        history = conv_mgr.get_chat_history_text(conv_id)
        assert history == ""

    def test_history_window(self, conv_mgr):
        conv_id = conv_mgr.create_conversation("Test")
        for i in range(10):
            conv_mgr.add_message(conv_id, "user", f"Q{i}")
            conv_mgr.add_message(conv_id, "assistant", f"A{i}")
        history = conv_mgr.get_chat_history_text(conv_id, window=2)
        # Should only contain last 4 messages (2 Q&A pairs)
        assert "Q8" in history
        assert "Q9" in history
        assert "Q0" not in history


class TestAutoTitle:
    def test_auto_title(self, conv_mgr):
        conv_id = conv_mgr.create_conversation()
        conv_mgr.add_message(conv_id, "user", "What is the revenue in Q3 2025?")
        conv_mgr.auto_title(conv_id)
        conv = conv_mgr.get_conversation(conv_id)
        assert "revenue" in conv["title"].lower()

    def test_auto_title_truncation(self, conv_mgr):
        conv_id = conv_mgr.create_conversation()
        long_q = "A" * 100
        conv_mgr.add_message(conv_id, "user", long_q)
        conv_mgr.auto_title(conv_id)
        conv = conv_mgr.get_conversation(conv_id)
        assert len(conv["title"]) <= 63  # 60 + "..."


class TestExport:
    def test_export_markdown(self, conv_mgr):
        conv_id = conv_mgr.create_conversation("Export Test")
        conv_mgr.add_message(conv_id, "user", "Question?")
        conv_mgr.add_message(conv_id, "assistant", "Answer.")
        md = conv_mgr.export_markdown(conv_id)
        assert "# Export Test" in md
        assert "Question?" in md
        assert "Answer." in md

    def test_export_json(self, conv_mgr):
        conv_id = conv_mgr.create_conversation("JSON Test")
        conv_mgr.add_message(conv_id, "user", "Q")
        data = conv_mgr.export_json(conv_id)
        assert "conversation" in data
        assert "messages" in data
        assert len(data["messages"]) == 1

    def test_export_nonexistent(self, conv_mgr):
        assert conv_mgr.export_markdown("nonexistent") == ""
        assert conv_mgr.export_json("nonexistent") == {}
