"""Core RAG pipeline — orchestrates retrieval, re-ranking, and generation."""

import time
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

import structlog
from rank_bm25 import BM25Okapi

from src.config import settings
from src.embedding_manager import EmbeddingManager
from src.llm_manager import LLMManager
from src.prompt_templates import build_rag_prompt, build_rephrase_prompt
from src.vector_store import VectorStore

logger = structlog.get_logger(__name__)


@dataclass
class RAGResponse:
    """Structured response from the RAG pipeline."""

    answer: str
    sources: list[dict[str, Any]]
    query: str
    rephrased_query: str | None = None
    generation_time_ms: int = 0
    retrieval_time_ms: int = 0


class RAGEngine:
    """Orchestrates the full RAG pipeline: retrieve → re-rank → generate.

    Supports:
    - Semantic search (vector similarity via ChromaDB)
    - BM25 keyword search (exact term matching)
    - Hybrid search (combines both with Reciprocal Rank Fusion)
    - Cross-encoder re-ranking (optional, improves precision)
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedding_manager: EmbeddingManager | None = None,
        llm_manager: LLMManager | None = None,
        use_reranker: bool = True,
    ):
        self.store = vector_store or VectorStore()
        self.embedder = embedding_manager or EmbeddingManager()
        self.llm = llm_manager or LLMManager()
        self.use_reranker = use_reranker
        self._reranker = None

    @property
    def reranker(self):
        """Lazy-load the cross-encoder re-ranker."""
        if self._reranker is None and self.use_reranker:
            try:
                from sentence_transformers import CrossEncoder

                logger.info("loading_reranker", model=settings.reranker_model)
                self._reranker = CrossEncoder(settings.reranker_model)
                logger.info("reranker_loaded")
            except Exception as e:
                logger.warning("reranker_load_failed", error=str(e))
                self.use_reranker = False
        return self._reranker

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        source_filter: str | None = None,
        use_hybrid: bool = True,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant chunks using hybrid search (semantic + BM25).

        Args:
            query: The user's question.
            top_k: Number of final results to return.
            source_filter: Restrict search to a specific document.
            use_hybrid: Whether to use BM25 alongside semantic search.

        Returns:
            List of search result dicts sorted by relevance.
        """
        top_k = top_k or settings.top_k
        start = time.perf_counter()

        # 1. Semantic search
        query_embedding = self.embedder.embed_query(query)
        # Fetch more candidates for re-ranking
        fetch_k = top_k * 3 if self.use_reranker else top_k * 2
        semantic_results = self.store.search(
            query_embedding, top_k=fetch_k, source_filter=source_filter
        )

        if not semantic_results:
            return []

        # 2. BM25 keyword search (if hybrid enabled and we have documents)
        if use_hybrid and semantic_results:
            results = self._hybrid_search(query, semantic_results, top_k=fetch_k)
        else:
            results = semantic_results

        # 3. Filter by similarity threshold
        results = [r for r in results if r["score"] >= settings.similarity_threshold]

        # 4. Re-rank with cross-encoder
        if self.use_reranker and self.reranker and len(results) > 1:
            results = self._rerank(query, results)

        # 5. Trim to top_k
        results = results[:top_k]

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "retrieval_complete",
            query=query[:80],
            results=len(results),
            time_ms=elapsed_ms,
            hybrid=use_hybrid,
            reranked=self.use_reranker,
        )
        return results

    def _hybrid_search(
        self,
        query: str,
        semantic_results: list[dict],
        top_k: int = 10,
    ) -> list[dict]:
        """Combine semantic and BM25 results using Reciprocal Rank Fusion (RRF).

        RRF score = sum(1 / (k + rank)) across both result lists.
        This is the industry-standard approach for hybrid search.
        """
        # Build BM25 index from semantic results (re-rank within candidates)
        corpus = [r["text"] for r in semantic_results]
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)

        # Score query against corpus
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)

        # Create BM25 ranking
        bm25_ranked = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )

        # Semantic ranking (already sorted by score)
        semantic_ranked = list(range(len(semantic_results)))

        # RRF fusion (k=60 is standard)
        rrf_k = 60
        rrf_scores: dict[int, float] = {}

        for rank, idx in enumerate(semantic_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (rrf_k + rank + 1)

        for rank, idx in enumerate(bm25_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (rrf_k + rank + 1)

        # Sort by RRF score
        sorted_indices = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)

        results = []
        for idx in sorted_indices[:top_k]:
            result = semantic_results[idx].copy()
            result["rrf_score"] = round(rrf_scores[idx], 6)
            results.append(result)

        return results

    def _rerank(self, query: str, results: list[dict]) -> list[dict]:
        """Re-rank results using a cross-encoder model for better precision."""
        if not results:
            return results

        pairs = [[query, r["text"]] for r in results]
        scores = self.reranker.predict(pairs)

        for result, score in zip(results, scores):
            result["rerank_score"] = round(float(score), 4)

        results.sort(key=lambda r: r["rerank_score"], reverse=True)

        logger.info(
            "reranking_complete",
            candidates=len(results),
            top_score=results[0]["rerank_score"] if results else 0,
        )
        return results

    def query(
        self,
        question: str,
        chat_history: str = "",
        top_k: int | None = None,
        source_filter: str | None = None,
        use_hybrid: bool = True,
    ) -> RAGResponse:
        """Full RAG pipeline: rephrase → retrieve → generate.

        Args:
            question: The user's question.
            chat_history: Formatted conversation history for context.
            top_k: Number of chunks to retrieve.
            source_filter: Restrict to a specific document.
            use_hybrid: Enable hybrid search.

        Returns:
            RAGResponse with answer, sources, and timing metrics.
        """
        # Rephrase follow-up questions if there's conversation history
        rephrased = None
        search_query = question
        if chat_history and chat_history != "No previous conversation.":
            rephrase_prompt = build_rephrase_prompt(question, chat_history)
            rephrased = self.llm.generate(rephrase_prompt).strip()
            if rephrased:
                search_query = rephrased
                logger.info("query_rephrased", original=question[:60], rephrased=rephrased[:60])

        # Retrieve relevant chunks
        retrieval_start = time.perf_counter()
        results = self.retrieve(
            search_query, top_k=top_k, source_filter=source_filter, use_hybrid=use_hybrid
        )
        retrieval_ms = int((time.perf_counter() - retrieval_start) * 1000)

        # Generate answer
        gen_start = time.perf_counter()
        prompt = build_rag_prompt(question, results, chat_history)
        answer = self.llm.generate(prompt)
        gen_ms = int((time.perf_counter() - gen_start) * 1000)

        return RAGResponse(
            answer=answer.strip(),
            sources=results,
            query=question,
            rephrased_query=rephrased,
            generation_time_ms=gen_ms,
            retrieval_time_ms=retrieval_ms,
        )

    def query_stream(
        self,
        question: str,
        chat_history: str = "",
        top_k: int | None = None,
        source_filter: str | None = None,
        use_hybrid: bool = True,
    ) -> tuple[Generator[str, None, None], list[dict]]:
        """Streaming version of query — returns token generator and sources.

        Returns:
            Tuple of (token_generator, search_results) so the UI can
            display sources alongside the streaming answer.
        """
        # Rephrase if needed
        search_query = question
        if chat_history and chat_history != "No previous conversation.":
            rephrase_prompt = build_rephrase_prompt(question, chat_history)
            rephrased = self.llm.generate(rephrase_prompt).strip()
            if rephrased:
                search_query = rephrased

        # Retrieve
        results = self.retrieve(
            search_query, top_k=top_k, source_filter=source_filter, use_hybrid=use_hybrid
        )

        # Build prompt and stream
        prompt = build_rag_prompt(question, results, chat_history)
        token_stream = self.llm.generate_stream(prompt)

        return token_stream, results
