from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"


class Settings(BaseSettings):
    """Application configuration with sensible defaults for local RAG."""

    # Paths
    data_dir: Path = DATA_DIR
    chromadb_dir: Path = DATA_DIR / "chromadb"
    uploads_dir: Path = DATA_DIR / "uploads"
    db_path: Path = DATA_DIR / "db" / "rag.db"

    # Ollama
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "mistral"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1024

    # Chunking
    chunk_size: int = Field(default=500, ge=200, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=200)

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    embedding_batch_size: int = 32

    # Retrieval
    top_k: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

    # Re-ranking
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 3

    # Upload limits
    max_file_size_mb: int = 50

    # ChromaDB
    collection_name: str = "documents"

    # Conversation
    conversation_window: int = 5

    model_config = {"env_prefix": "RAG_", "env_file": ".env", "extra": "ignore"}

    def ensure_dirs(self) -> None:
        """Create required data directories if they don't exist."""
        self.chromadb_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
