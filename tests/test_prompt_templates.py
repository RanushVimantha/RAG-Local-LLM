"""Tests for prompt_templates.py — prompt formatting and context building."""

from src.prompt_templates import (
    build_context,
    build_rag_prompt,
    build_rephrase_prompt,
)


class TestBuildContext:
    def test_basic_context(self):
        results = [
            {"source": "doc.pdf", "page": 1, "text": "Revenue was $1M."},
            {"source": "doc.pdf", "page": 3, "text": "Growth was 12%."},
        ]
        context = build_context(results)
        assert "[Source: doc.pdf, Page: 1]" in context
        assert "Revenue was $1M." in context
        assert "[Source: doc.pdf, Page: 3]" in context
        assert "Growth was 12%." in context

    def test_empty_results(self):
        context = build_context([])
        assert "No relevant context found" in context

    def test_multiple_sources(self):
        results = [
            {"source": "a.pdf", "page": 1, "text": "Text A."},
            {"source": "b.txt", "page": 1, "text": "Text B."},
        ]
        context = build_context(results)
        assert "a.pdf" in context
        assert "b.txt" in context


class TestBuildRagPrompt:
    def test_full_prompt(self):
        results = [{"source": "doc.pdf", "page": 1, "text": "The answer is 42."}]
        prompt = build_rag_prompt("What is the answer?", results)
        assert "What is the answer?" in prompt
        assert "The answer is 42." in prompt
        assert "RULES:" in prompt
        assert "CONTEXT:" in prompt

    def test_prompt_with_history(self):
        results = [{"source": "doc.pdf", "page": 1, "text": "Data."}]
        prompt = build_rag_prompt("Follow up?", results, chat_history="Human: First question")
        assert "Human: First question" in prompt

    def test_prompt_without_history(self):
        results = [{"source": "doc.pdf", "page": 1, "text": "Data."}]
        prompt = build_rag_prompt("Question?", results)
        assert "No previous conversation" in prompt

    def test_prompt_no_results(self):
        prompt = build_rag_prompt("Question?", [])
        assert "No relevant context found" in prompt


class TestBuildRephrasePrompt:
    def test_rephrase_prompt(self):
        prompt = build_rephrase_prompt(
            "How does that compare?",
            "Human: What was Q3 revenue?\nAssistant: Q3 revenue was $4.2B.",
        )
        assert "How does that compare?" in prompt
        assert "Q3 revenue" in prompt
        assert "STANDALONE QUESTION:" in prompt
