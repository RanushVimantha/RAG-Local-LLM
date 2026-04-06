"""FastAPI REST API for the RAG Local LLM system.

Provides programmatic access to document ingestion, querying, and management.
Swagger docs available at /docs when running.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.config import settings
from src.conversation_manager import ConversationManager
from src.document_processor import DocumentProcessor
from src.document_registry import DocumentRegistry
from src.embedding_manager import EmbeddingManager
from src.evaluation import RAGEvaluator
from src.llm_manager import LLMManager
from src.rag_engine import RAGEngine
from src.text_chunker import TextChunker
from src.utils import configure_logging
from src.vector_store import VectorStore

configure_logging()
logger = structlog.get_logger(__name__)

app = FastAPI(
    title="RAG Local LLM",
    description="Privacy-first RAG system — 100% local, zero data leakage",
    version="0.1.0",
)

# --- Shared instances ---
settings.ensure_dirs()
processor = DocumentProcessor()
embedder = EmbeddingManager()
store = VectorStore()
registry = DocumentRegistry()
llm = LLMManager()
conv_mgr = ConversationManager()
rag = RAGEngine(vector_store=store, embedding_manager=embedder, llm_manager=llm)


# --- Request/Response Models ---


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="The question to ask")
    chat_history: str = Field(default="", description="Formatted conversation history")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    source_filter: str | None = Field(default=None, description="Restrict to a specific document")
    use_hybrid: bool = Field(default=True, description="Enable hybrid search (BM25 + semantic)")


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    query: str
    rephrased_query: str | None = None
    generation_time_ms: int = 0
    retrieval_time_ms: int = 0


class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size_bytes: int
    page_count: int
    chunk_count: int
    character_count: int
    status: str
    created_at: str


class HealthResponse(BaseModel):
    status: str
    ollama_running: bool
    model_available: bool
    available_models: list[str]
    vector_store_chunks: int
    documents_count: int


class IngestResponse(BaseModel):
    id: str
    filename: str
    pages: int
    chunks: int
    characters: int
    status: str


# --- Endpoints ---


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check system health — Ollama, ChromaDB, and embedding model status."""
    ollama_health = llm.health_check()
    store_stats = store.get_stats()
    doc_stats = registry.get_stats()

    return HealthResponse(
        status="healthy" if ollama_health["ollama_running"] else "degraded",
        ollama_running=ollama_health["ollama_running"],
        model_available=ollama_health["model_available"],
        available_models=ollama_health["available_models"],
        vector_store_chunks=store_stats["total_chunks"],
        documents_count=doc_stats["total_documents"],
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Documents"])
async def ingest_document(
    file: UploadFile = File(...),
    chunk_size: int = Query(default=500, ge=200, le=2000),
    chunk_overlap: int = Query(default=50, ge=0, le=200),
    strategy: str = Query(default="recursive"),
):
    """Upload and process a document into the RAG pipeline."""
    if not processor.is_supported(file.filename):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")

    # Save to uploads directory
    save_path = settings.uploads_dir / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    file_size = save_path.stat().st_size
    if file_size > settings.max_file_size_mb * 1024 * 1024:
        save_path.unlink()
        raise HTTPException(
            status_code=413, detail=f"File exceeds {settings.max_file_size_mb}MB limit"
        )

    # Register document
    doc_id = registry.register(
        filename=file.filename,
        file_path=str(save_path),
        file_size_bytes=file_size,
        file_type=processor.get_file_type(file.filename),
    )

    try:
        # Extract → Chunk → Embed → Store
        pages = processor.extract_text(save_path)
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap, strategy=strategy)
        chunks = chunker.chunk_pages(pages, file.filename)
        embeddings = embedder.embed_texts([c.text for c in chunks])
        store.add_chunks(chunks, embeddings)

        total_chars = sum(len(p["text"]) for p in pages)
        registry.update_status(
            doc_id=doc_id,
            status="ready",
            page_count=len(pages),
            chunk_count=len(chunks),
            character_count=total_chars,
            chunk_size_used=chunk_size,
            chunk_overlap_used=chunk_overlap,
            embedding_model=embedder.model_name,
        )

        return IngestResponse(
            id=doc_id,
            filename=file.filename,
            pages=len(pages),
            chunks=len(chunks),
            characters=total_chars,
            status="ready",
        )

    except Exception as e:
        registry.update_status(doc_id=doc_id, status="error", error_message=str(e))
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_documents(request: QueryRequest):
    """Ask a question about your uploaded documents."""
    if store.get_stats()["total_chunks"] == 0:
        raise HTTPException(status_code=400, detail="No documents uploaded yet. Ingest first.")

    response = rag.query(
        question=request.question,
        chat_history=request.chat_history,
        top_k=request.top_k,
        source_filter=request.source_filter,
        use_hybrid=request.use_hybrid,
    )

    return QueryResponse(
        answer=response.answer,
        sources=response.sources,
        query=response.query,
        rephrased_query=response.rephrased_query,
        generation_time_ms=response.generation_time_ms,
        retrieval_time_ms=response.retrieval_time_ms,
    )


@app.post("/query/stream", tags=["RAG"])
async def query_documents_stream(request: QueryRequest):
    """Ask a question and receive a streaming response."""
    if store.get_stats()["total_chunks"] == 0:
        raise HTTPException(status_code=400, detail="No documents uploaded yet. Ingest first.")

    token_stream, sources = rag.query_stream(
        question=request.question,
        chat_history=request.chat_history,
        top_k=request.top_k,
        source_filter=request.source_filter,
        use_hybrid=request.use_hybrid,
    )

    def generate():
        for token in token_stream:
            yield token

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/documents", response_model=list[DocumentResponse], tags=["Documents"])
async def list_documents():
    """List all uploaded documents."""
    docs = registry.get_all()
    return [DocumentResponse(**doc) for doc in docs]


@app.get("/documents/{doc_id}", response_model=DocumentResponse, tags=["Documents"])
async def get_document(doc_id: str):
    """Get details for a specific document."""
    doc = registry.get_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentResponse(**doc)


@app.delete("/documents/{doc_id}", tags=["Documents"])
async def delete_document(doc_id: str):
    """Delete a document and its vectors."""
    doc = registry.get_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove vectors
    store.delete_by_source(doc["filename"])

    # Remove file
    file_path = Path(doc["file_path"])
    if file_path.exists():
        file_path.unlink()

    # Remove registry entry
    registry.delete(doc_id)

    return {"message": f"Deleted {doc['filename']}", "id": doc_id}


@app.get("/stats", tags=["System"])
async def get_stats():
    """Get system-wide statistics."""
    doc_stats = registry.get_stats()
    store_stats = store.get_stats()
    return {
        "documents": doc_stats,
        "vector_store": store_stats,
    }


# --- Conversation Endpoints ---


class ConversationCreate(BaseModel):
    title: str = Field(default="New Conversation", max_length=200)
    model_name: str = Field(default="mistral")


class MessageCreate(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1)
    sources: list[dict[str, Any]] | None = None


class ConversationResponse(BaseModel):
    id: str
    title: str
    model_name: str
    message_count: int = 0
    created_at: str
    updated_at: str


@app.post("/conversations", tags=["Conversations"])
async def create_conversation(request: ConversationCreate):
    """Create a new conversation."""
    conv_id = conv_mgr.create_conversation(title=request.title, model_name=request.model_name)
    return {"id": conv_id, "title": request.title}


@app.get("/conversations", response_model=list[ConversationResponse], tags=["Conversations"])
async def list_conversations(limit: int = Query(default=50, ge=1, le=200)):
    """List all conversations."""
    convs = conv_mgr.get_conversations(limit=limit)
    return [ConversationResponse(**c) for c in convs]


@app.get("/conversations/{conv_id}", tags=["Conversations"])
async def get_conversation(conv_id: str):
    """Get a conversation with all its messages."""
    conv = conv_mgr.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages = conv_mgr.get_messages(conv_id)
    return {"conversation": conv, "messages": messages}


@app.delete("/conversations/{conv_id}", tags=["Conversations"])
async def delete_conversation(conv_id: str):
    """Delete a conversation and all its messages."""
    if not conv_mgr.delete_conversation(conv_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"message": "Conversation deleted", "id": conv_id}


@app.post("/conversations/{conv_id}/messages", tags=["Conversations"])
async def add_message(conv_id: str, request: MessageCreate):
    """Add a message to a conversation."""
    conv = conv_mgr.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    msg_id = conv_mgr.add_message(
        conv_id, request.role, request.content, sources=request.sources
    )
    return {"id": msg_id, "conversation_id": conv_id}


@app.get("/conversations/{conv_id}/export/markdown", tags=["Conversations"])
async def export_conversation_markdown(conv_id: str):
    """Export a conversation as Markdown."""
    md = conv_mgr.export_markdown(conv_id)
    if not md:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return StreamingResponse(
        iter([md]),
        media_type="text/markdown",
        headers={"Content-Disposition": f"attachment; filename=conversation_{conv_id}.md"},
    )


@app.get("/conversations/{conv_id}/export/json", tags=["Conversations"])
async def export_conversation_json(conv_id: str):
    """Export a conversation as JSON."""
    data = conv_mgr.export_json(conv_id)
    if not data:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return data


# --- Evaluation Endpoints ---


class EvalQuestion(BaseModel):
    question: str
    expected_sources: list[str] = Field(default_factory=list)


class EvalRequest(BaseModel):
    questions: list[EvalQuestion]
    check_faithfulness: bool = Field(default=False, description="Use LLM to judge answer faithfulness")


@app.post("/evaluate", tags=["Evaluation"])
async def evaluate_rag(request: EvalRequest):
    """Run evaluation against the RAG pipeline with test questions.

    Returns precision, recall, faithfulness, and latency metrics.
    """
    if store.get_stats()["total_chunks"] == 0:
        raise HTTPException(status_code=400, detail="No documents uploaded. Ingest first.")

    evaluator = RAGEvaluator(rag_engine=rag)
    dataset = [{"question": q.question, "expected_sources": q.expected_sources} for q in request.questions]

    summary = evaluator.evaluate_dataset(dataset, check_faithfulness=request.check_faithfulness)

    return {
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
                "answer": r.answer[:500],
                "expected_sources": r.expected_sources,
                "retrieved_sources": r.retrieved_sources,
                "precision": r.precision_at_k,
                "recall": r.recall_at_k,
                "top_score": r.top_score,
                "faithfulness": r.faithfulness_score,
                "retrieval_ms": r.retrieval_time_ms,
                "generation_ms": r.generation_time_ms,
            }
            for r in summary.results
        ],
    }
