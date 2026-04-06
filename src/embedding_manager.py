"""Local embedding generation using HuggingFace sentence-transformers."""

import structlog
from sentence_transformers import SentenceTransformer

from src.config import settings

logger = structlog.get_logger(__name__)


class EmbeddingManager:
    """Generates vector embeddings locally using sentence-transformers.

    No data leaves the machine — the model runs entirely on local hardware.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the embedding model on first use."""
        if self._model is None:
            logger.info("loading_embedding_model", model=self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info(
                "embedding_model_loaded",
                model=self.model_name,
                dimensions=self._model.get_sentence_embedding_dimension(),
            )
        return self._model

    @property
    def dimensions(self) -> int:
        """Get the embedding dimensions for the loaded model."""
        return self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.
            batch_size: Number of texts to process at once.

        Returns:
            List of embedding vectors (each is a list of floats).
        """
        if not texts:
            return []

        batch_size = batch_size or settings.embedding_batch_size

        logger.info("generating_embeddings", count=len(texts), batch_size=batch_size)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        result = embeddings.tolist()

        logger.info(
            "embeddings_generated",
            count=len(result),
            dimensions=len(result[0]) if result else 0,
        )
        return result

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a single query string."""
        result = self.embed_texts([query])
        return result[0]
