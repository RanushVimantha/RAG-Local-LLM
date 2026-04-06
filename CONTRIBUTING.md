# Contributing to RAG Local LLM

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/RanushVimantha/rag-local-llm.git
cd rag-local-llm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Install and start Ollama
# Visit https://ollama.ai/download
ollama serve
ollama pull mistral
```

## Running Tests

```bash
# Unit tests only (fast, no external dependencies)
make test

# All tests including integration (requires Ollama + embedding model)
make test-all

# With coverage report
pytest tests/ -v --cov=src --cov-report=html
open htmlcov/index.html
```

## Code Style

This project uses [ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Check for issues
make lint

# Auto-fix and format
make format
```

Key conventions:
- Python 3.11+ with type hints
- 100 character line length
- Structured logging with `structlog` (no print statements)
- One responsibility per module
- Tests mirror source: `src/foo.py` → `tests/test_foo.py`

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear, focused commits
3. Ensure all tests pass: `make test`
4. Ensure code is formatted: `make format`
5. Update documentation if needed
6. Submit a PR with a clear description

## Commit Messages

Use conventional commit style:

```
feat: add CSV file support
fix: handle empty PDF pages gracefully
docs: update API endpoint documentation
test: add edge case tests for text chunker
chore: update dependencies
```

## Project Architecture

```
src/
├── app.py              # Streamlit frontend (UI layer)
├── api.py              # FastAPI REST API (API layer)
├── rag_engine.py       # Core pipeline (business logic)
├── document_processor.py  # File I/O
├── text_chunker.py     # Text processing
├── embedding_manager.py   # ML embeddings
├── vector_store.py     # Database (ChromaDB)
├── llm_manager.py      # External service (Ollama)
├── conversation_manager.py  # Database (SQLite)
├── document_registry.py    # Database (SQLite)
├── evaluation.py       # Quality metrics
├── config.py           # Configuration
└── utils.py            # Shared utilities
```

## Areas for Contribution

- **New file formats** — CSV, Excel, HTML support
- **OCR** — Tesseract integration for scanned PDFs
- **Web URL ingestion** — Scrape and index web pages
- **Improved UI** — Better theming, mobile responsiveness
- **Performance** — Caching, async processing, batch operations
- **Evaluation** — More metrics, automated benchmarking

## Questions?

Open an issue on GitHub if you have questions or want to discuss a feature before implementing it.
