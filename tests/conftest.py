"""Pytest fixtures for RAG Local LLM tests."""

import tempfile
from pathlib import Path

import pytest

from src.config import Settings

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def sample_txt(fixtures_dir) -> Path:
    return fixtures_dir / "sample.txt"


@pytest.fixture
def sample_md(fixtures_dir) -> Path:
    return fixtures_dir / "sample.md"


@pytest.fixture
def tmp_data_dir(tmp_path) -> Path:
    """Create a temporary data directory for tests."""
    chromadb_dir = tmp_path / "chromadb"
    uploads_dir = tmp_path / "uploads"
    db_dir = tmp_path / "db"
    chromadb_dir.mkdir()
    uploads_dir.mkdir()
    db_dir.mkdir()
    return tmp_path


@pytest.fixture
def test_settings(tmp_data_dir) -> Settings:
    """Settings pointing to temporary directories for isolated tests."""
    return Settings(
        data_dir=tmp_data_dir,
        chromadb_dir=tmp_data_dir / "chromadb",
        uploads_dir=tmp_data_dir / "uploads",
        db_path=tmp_data_dir / "db" / "test.db",
    )
