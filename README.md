# RAG Local LLM

[![CI](https://github.com/RanushVimantha/rag-local-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/RanushVimantha/rag-local-llm/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**A fully local, privacy-first Retrieval-Augmented Generation system powered by open-source models.**

Zero external API calls. Zero data leakage. Everything runs on your machine.

<!-- ![Demo](assets/demo.gif) -->

---

## What It Does

Upload documents (PDF, TXT, Markdown, DOCX) and ask natural language questions. The system retrieves relevant content from your documents and generates accurate answers with source citations — all running 100% locally.

**Key Features:**
- **Document Intelligence** — Upload and query across multiple documents
- **Hybrid Search** — Combines semantic (vector) + keyword (BM25) search with Reciprocal Rank Fusion
- **Cross-Encoder Re-ranking** — Precision-boosted retrieval using a second-pass re-ranker
- **Streaming Responses** — Token-by-token answer generation with real-time display
- **Source Citations** — Every answer shows which document and page it came from
- **Conversation Memory** — Follow-up questions understand context from previous exchanges
- **REST API** — Full FastAPI backend with auto-generated Swagger docs
- **Evaluation Framework** — Built-in metrics for retrieval precision, recall, faithfulness, and latency
- **Docker Ready** — One command to run the entire stack

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Streamlit Frontend                       │
│   Upload Documents  │  Chat Interface  │  Dashboard      │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                   FastAPI REST API                        │
│   /ingest  │  /query  │  /documents  │  /evaluate        │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                    RAG Engine                             │
│                                                          │
│  ┌──────────┐  ┌────────────┐  ┌──────────────────────┐ │
│  │ Document  │  │ Embedding  │  │ Hybrid Search        │ │
│  │ Processor │  │ Manager    │  │ (Semantic + BM25)    │ │
│  └─────┬────┘  └─────┬──────┘  │ + RRF Fusion         │ │
│        │             │         │ + Cross-Encoder Rerank│ │
│  ┌─────▼────┐  ┌─────▼──────┐ └──────────┬───────────┘ │
│  │ Text     │  │ ChromaDB   │            │              │
│  │ Chunker  │  │ (Vectors)  │  ┌─────────▼───────────┐ │
│  └──────────┘  └────────────┘  │ Ollama LLM          │ │
│                                │ (Local Generation)   │ │
│  ┌──────────────────────────┐  └─────────────────────┘ │
│  │ SQLite                    │                          │
│  │ (Documents + Conversations│                          │
│  │  + Settings)              │                          │
│  └──────────────────────────┘                          │
└─────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                    Ollama Server                          │
│          Mistral 7B │ Llama 3.1 │ Phi-3 │ ...           │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| LLM Runtime | [Ollama](https://ollama.ai) | Serve open-source LLMs locally |
| Embeddings | [sentence-transformers](https://sbert.net) | Local vector embeddings (all-MiniLM-L6-v2) |
| Vector DB | [ChromaDB](https://www.trychroma.com) | Store and search document vectors |
| Keyword Search | [rank-bm25](https://github.com/dorianbrown/rank_bm25) | BM25 for hybrid search |
| Re-ranker | [cross-encoder](https://www.sbert.net/docs/cross_encoder/pretrained_models.html) | ms-marco-MiniLM for precision |
| Backend API | [FastAPI](https://fastapi.tiangolo.com) | REST API with auto Swagger docs |
| Frontend | [Streamlit](https://streamlit.io) | Web UI for upload, chat, dashboard |
| Database | SQLite | Document metadata + conversations |
| Orchestration | [LangChain](https://python.langchain.com) | Text splitting |
| Config | [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) | Type-safe configuration |
| Logging | [structlog](https://www.structlog.org) | Structured JSON logging |
| Testing | pytest | Unit + integration tests |
| CI/CD | GitHub Actions | Lint, test, Docker build |
| Container | Docker + Compose | Reproducible deployment |

---

## Quick Start

### Prerequisites

- **Python 3.11+**
- **[Ollama](https://ollama.ai/download)** installed and running
- 8 GB RAM minimum (16 GB recommended)

### 1. Clone and Install

```bash
git clone https://github.com/RanushVimantha/rag-local-llm.git
cd rag-local-llm

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Start Ollama and Pull a Model

```bash
ollama serve                # Start Ollama (if not already running)
ollama pull mistral         # Download Mistral 7B (~4.1 GB)
```

### 3. Run the App

```bash
# Streamlit UI
streamlit run src/app.py

# Or use the FastAPI server
uvicorn src.api:app --reload --port 8000
```

Open **http://localhost:8501** for the UI or **http://localhost:8000/docs** for the API.

### Docker (Alternative)

```bash
docker-compose up --build
```

This starts Ollama + the Streamlit app + the FastAPI server automatically.

---

## Usage

### Upload Documents
1. Go to the **Upload Documents** tab
2. Drag and drop your PDF, TXT, MD, or DOCX files
3. Configure chunk size and strategy
4. Click **Process** — watch the pipeline: extract → chunk → embed → store

### Ask Questions
1. Switch to the **Ask Questions** tab
2. Type a natural language question
3. Get a streamed answer with source citations
4. Ask follow-up questions — the system remembers context

### API Access

```bash
# Upload a document
curl -X POST http://localhost:8000/ingest \
  -F "file=@document.pdf"

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the revenue in Q3?"}'

# Check system health
curl http://localhost:8000/health
```

Full API docs at **http://localhost:8000/docs** (Swagger UI).

---

## Supported Models

### LLM Models (via Ollama)

| Model | Size | Best For | RAM |
|-------|------|----------|-----|
| Mistral 7B | 4.1 GB | Best balance for RAG | 8 GB |
| Llama 3.1 8B | 4.7 GB | Strong reasoning | 8 GB |
| Phi-3 Mini | 2.3 GB | Low-RAM systems | 4 GB |
| Gemma 2 2B | 1.6 GB | Ultra-fast Q&A | 4 GB |
| Qwen 2.5 7B | 4.4 GB | Structured output | 8 GB |

### Embedding Models

| Model | Dimensions | Size | Quality |
|-------|-----------|------|---------|
| all-MiniLM-L6-v2 (default) | 384 | 80 MB | Good |
| all-mpnet-base-v2 | 768 | 420 MB | Better |
| nomic-embed-text (Ollama) | 768 | 274 MB | Better |

---

## Configuration

All settings are configurable via the Settings tab, environment variables (`RAG_` prefix), or `.env` file:

| Setting | Default | Description |
|---------|---------|-------------|
| `RAG_LLM_MODEL` | mistral | Ollama model name |
| `RAG_LLM_TEMPERATURE` | 0.3 | Lower = more factual |
| `RAG_CHUNK_SIZE` | 500 | Characters per chunk |
| `RAG_CHUNK_OVERLAP` | 50 | Overlap between chunks |
| `RAG_TOP_K` | 5 | Chunks retrieved per query |
| `RAG_SIMILARITY_THRESHOLD` | 0.3 | Minimum relevance score |
| `RAG_OLLAMA_HOST` | http://localhost:11434 | Ollama server URL |

---

## Project Structure

```
rag-local-llm/
├── src/
│   ├── app.py                  # Streamlit frontend
│   ├── api.py                  # FastAPI REST API
│   ├── rag_engine.py           # Core RAG pipeline (hybrid search + re-ranking)
│   ├── document_processor.py   # PDF/TXT/MD/DOCX text extraction
│   ├── text_chunker.py         # 4 chunking strategies
│   ├── embedding_manager.py    # Local HuggingFace embeddings
│   ├── vector_store.py         # ChromaDB operations
│   ├── llm_manager.py          # Ollama integration + streaming
│   ├── prompt_templates.py     # RAG prompts with citation rules
│   ├── conversation_manager.py # Chat persistence (SQLite)
│   ├── document_registry.py    # Document metadata tracking
│   ├── evaluation.py           # RAG quality metrics
│   ├── config.py               # Pydantic settings
│   └── utils.py                # Logging setup + helpers
├── tests/                      # Comprehensive test suite
├── .github/workflows/ci.yml    # CI pipeline
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

---

## Development

```bash
# Run tests
make test

# Run all tests (including integration)
make test-all

# Lint and format
make lint
make format

# Install pre-commit hooks
make setup
```

---

## How It Works

### RAG Pipeline

1. **Ingest** — Documents are loaded, split into chunks, embedded locally, and stored in ChromaDB
2. **Retrieve** — Questions are embedded and searched against the vector store using hybrid search (semantic + BM25 with Reciprocal Rank Fusion)
3. **Re-rank** — Top candidates are re-scored with a cross-encoder for precision
4. **Generate** — Retrieved context is injected into a grounded prompt and sent to the local LLM
5. **Cite** — Sources are displayed alongside the answer

### Hybrid Search

Unlike pure semantic search which can miss exact keywords, this system combines:
- **Vector similarity** (captures meaning) via ChromaDB
- **BM25 keyword matching** (captures exact terms) via rank-bm25
- **Reciprocal Rank Fusion** (merges both rankings with balanced weighting)

This approach is the industry standard for production RAG systems.

---

## Evaluation

Built-in evaluation framework measures:

- **Precision@K** — Are the retrieved chunks relevant?
- **Recall@K** — Were all relevant chunks found?
- **Faithfulness** — Does the answer stick to the context? (LLM-judged)
- **Latency** — Retrieval time, generation time, total time

```bash
# Run via API
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"questions": [{"question": "What is...", "expected_sources": ["doc.pdf"]}]}'
```

---

## License

[MIT](LICENSE)

---

## Author

**Ranush Vimantha** — [GitHub](https://github.com/RanushVimantha)
