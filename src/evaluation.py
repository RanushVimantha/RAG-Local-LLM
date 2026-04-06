"""RAG evaluation framework — measures retrieval precision, answer faithfulness, and latency."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from src.config import settings
from src.embedding_manager import EmbeddingManager
from src.llm_manager import LLMManager
from src.rag_engine import RAGEngine
from src.vector_store import VectorStore

logger = structlog.get_logger(__name__)


@dataclass
class EvalResult:
    """Result of evaluating a single question."""

    question: str
    expected_sources: list[str]
    retrieved_sources: list[str]
    answer: str
    precision_at_k: float
    recall_at_k: float
    retrieval_time_ms: int
    generation_time_ms: int
    total_time_ms: int
    top_score: float
    faithfulness_score: float | None = None


@dataclass
class EvalSummary:
    """Aggregated evaluation results across all questions."""

    total_questions: int
    avg_precision: float
    avg_recall: float
    avg_retrieval_ms: float
    avg_generation_ms: float
    avg_total_ms: float
    avg_top_score: float
    avg_faithfulness: float | None = None
    results: list[EvalResult] = field(default_factory=list)


FAITHFULNESS_PROMPT = """You are an impartial judge evaluating whether an AI answer is faithful to the provided context.

CONTEXT:
{context}

QUESTION: {question}

ANSWER: {answer}

Rate the faithfulness of the answer on a scale of 0.0 to 1.0:
- 1.0 = Answer is fully supported by the context, no hallucinated facts
- 0.5 = Answer is partially supported, some claims lack context backing
- 0.0 = Answer is not supported by context or contains fabricated information

Respond with ONLY a single decimal number (e.g., 0.8). Nothing else."""


class RAGEvaluator:
    """Evaluates RAG pipeline quality using test datasets.

    Metrics:
    - Precision@K: What fraction of retrieved chunks are actually relevant?
    - Recall@K: What fraction of relevant chunks were retrieved?
    - Faithfulness: Does the answer stick to what's in the context? (LLM-judged)
    - Latency: Retrieval time, generation time, total time
    """

    def __init__(self, rag_engine: RAGEngine | None = None):
        self.rag = rag_engine or RAGEngine()

    def load_eval_dataset(self, path: Path) -> list[dict]:
        """Load an evaluation dataset from a JSON file.

        Expected format:
        [
            {
                "question": "What was Q3 revenue?",
                "expected_sources": ["annual_report.pdf"],
                "expected_answer_contains": ["4.2 billion"]  // optional
            }
        ]
        """
        with open(path) as f:
            return json.load(f)

    def evaluate_question(
        self,
        question: str,
        expected_sources: list[str],
        check_faithfulness: bool = False,
    ) -> EvalResult:
        """Evaluate a single question against the RAG pipeline."""
        start = time.perf_counter()

        response = self.rag.query(question)

        total_ms = int((time.perf_counter() - start) * 1000)

        # Extract retrieved source filenames
        retrieved_sources = list({r["source"] for r in response.sources})

        # Precision@K: fraction of retrieved that are relevant
        relevant_retrieved = [s for s in retrieved_sources if s in expected_sources]
        precision = len(relevant_retrieved) / len(retrieved_sources) if retrieved_sources else 0.0

        # Recall@K: fraction of expected that were retrieved
        recall = len(relevant_retrieved) / len(expected_sources) if expected_sources else 0.0

        top_score = response.sources[0]["score"] if response.sources else 0.0

        # Faithfulness check (optional, requires LLM call)
        faithfulness = None
        if check_faithfulness and response.sources:
            faithfulness = self._check_faithfulness(
                question, response.answer, response.sources
            )

        result = EvalResult(
            question=question,
            expected_sources=expected_sources,
            retrieved_sources=retrieved_sources,
            answer=response.answer,
            precision_at_k=round(precision, 4),
            recall_at_k=round(recall, 4),
            retrieval_time_ms=response.retrieval_time_ms,
            generation_time_ms=response.generation_time_ms,
            total_time_ms=total_ms,
            top_score=round(top_score, 4),
            faithfulness_score=faithfulness,
        )

        logger.info(
            "eval_question",
            question=question[:60],
            precision=result.precision_at_k,
            recall=result.recall_at_k,
            faithfulness=faithfulness,
            time_ms=total_ms,
        )
        return result

    def evaluate_dataset(
        self,
        dataset: list[dict],
        check_faithfulness: bool = False,
    ) -> EvalSummary:
        """Evaluate an entire dataset and return aggregated metrics."""
        results = []

        for item in dataset:
            result = self.evaluate_question(
                question=item["question"],
                expected_sources=item.get("expected_sources", []),
                check_faithfulness=check_faithfulness,
            )
            results.append(result)

        n = len(results)
        if n == 0:
            return EvalSummary(total_questions=0, avg_precision=0, avg_recall=0,
                               avg_retrieval_ms=0, avg_generation_ms=0,
                               avg_total_ms=0, avg_top_score=0)

        faithfulness_scores = [r.faithfulness_score for r in results if r.faithfulness_score is not None]

        summary = EvalSummary(
            total_questions=n,
            avg_precision=round(sum(r.precision_at_k for r in results) / n, 4),
            avg_recall=round(sum(r.recall_at_k for r in results) / n, 4),
            avg_retrieval_ms=round(sum(r.retrieval_time_ms for r in results) / n, 1),
            avg_generation_ms=round(sum(r.generation_time_ms for r in results) / n, 1),
            avg_total_ms=round(sum(r.total_time_ms for r in results) / n, 1),
            avg_top_score=round(sum(r.top_score for r in results) / n, 4),
            avg_faithfulness=round(sum(faithfulness_scores) / len(faithfulness_scores), 4) if faithfulness_scores else None,
            results=results,
        )

        logger.info(
            "eval_complete",
            questions=n,
            avg_precision=summary.avg_precision,
            avg_recall=summary.avg_recall,
            avg_total_ms=summary.avg_total_ms,
            avg_faithfulness=summary.avg_faithfulness,
        )
        return summary

    def _check_faithfulness(
        self,
        question: str,
        answer: str,
        sources: list[dict],
    ) -> float | None:
        """Use the LLM to judge answer faithfulness against the context."""
        context = "\n---\n".join(
            f"[{s['source']}, Page {s['page']}]: {s['text']}" for s in sources
        )

        prompt = FAITHFULNESS_PROMPT.format(
            context=context, question=question, answer=answer
        )

        try:
            response = self.rag.llm.generate(prompt).strip()
            score = float(response)
            return round(min(max(score, 0.0), 1.0), 2)
        except (ValueError, Exception) as e:
            logger.warning("faithfulness_parse_error", response=response[:50], error=str(e))
            return None

    def export_results(self, summary: EvalSummary, output_path: Path) -> None:
        """Export evaluation results to a JSON file."""
        data = {
            "summary": {
                "total_questions": summary.total_questions,
                "avg_precision": summary.avg_precision,
                "avg_recall": summary.avg_recall,
                "avg_retrieval_ms": summary.avg_retrieval_ms,
                "avg_generation_ms": summary.avg_generation_ms,
                "avg_total_ms": summary.avg_total_ms,
                "avg_top_score": summary.avg_top_score,
                "avg_faithfulness": summary.avg_faithfulness,
            },
            "results": [
                {
                    "question": r.question,
                    "expected_sources": r.expected_sources,
                    "retrieved_sources": r.retrieved_sources,
                    "precision": r.precision_at_k,
                    "recall": r.recall_at_k,
                    "top_score": r.top_score,
                    "faithfulness": r.faithfulness_score,
                    "retrieval_ms": r.retrieval_time_ms,
                    "generation_ms": r.generation_time_ms,
                    "total_ms": r.total_time_ms,
                }
                for r in summary.results
            ],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("eval_results_exported", path=str(output_path))
