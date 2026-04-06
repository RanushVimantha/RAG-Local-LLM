"""Ollama LLM integration with streaming support."""

from collections.abc import Generator
from typing import Any

import httpx
import structlog

from src.config import settings

logger = structlog.get_logger(__name__)


class OllamaError(Exception):
    """Raised when Ollama is unreachable or returns an error."""


class LLMManager:
    """Manages interaction with the Ollama local LLM server.

    All inference runs locally — no data leaves the machine.
    """

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        self.host = (host or settings.ollama_host).rstrip("/")
        self.model = model or settings.llm_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens

    def health_check(self) -> dict[str, Any]:
        """Check if Ollama is running and the model is available."""
        try:
            resp = httpx.get(f"{self.host}/api/tags", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            model_available = any(self.model in m for m in models)
            return {
                "ollama_running": True,
                "model_available": model_available,
                "available_models": models,
                "configured_model": self.model,
            }
        except (httpx.ConnectError, httpx.TimeoutException):
            return {
                "ollama_running": False,
                "model_available": False,
                "available_models": [],
                "configured_model": self.model,
            }
        except Exception as e:
            return {
                "ollama_running": False,
                "model_available": False,
                "error": str(e),
                "configured_model": self.model,
            }

    def generate(self, prompt: str) -> str:
        """Generate a complete response (non-streaming).

        Args:
            prompt: The full prompt string to send to the LLM.

        Returns:
            The generated response text.
        """
        try:
            resp = httpx.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()

            logger.info(
                "llm_response",
                model=self.model,
                eval_count=data.get("eval_count", 0),
                eval_duration_ms=data.get("eval_duration", 0) // 1_000_000,
            )
            return data.get("response", "")

        except httpx.ConnectError:
            raise OllamaError(
                f"Cannot connect to Ollama at {self.host}. "
                "Make sure Ollama is running: `ollama serve`"
            )
        except httpx.TimeoutException:
            raise OllamaError("Ollama request timed out. The model may be loading.")
        except httpx.HTTPStatusError as e:
            raise OllamaError(f"Ollama returned error {e.response.status_code}: {e.response.text}")

    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """Generate a streaming response, yielding tokens as they arrive.

        Args:
            prompt: The full prompt string to send to the LLM.

        Yields:
            Individual tokens/chunks of the response.
        """
        try:
            with httpx.stream(
                "POST",
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                },
                timeout=120,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        import json

                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            yield token
                        if data.get("done", False):
                            break

        except httpx.ConnectError:
            raise OllamaError(
                f"Cannot connect to Ollama at {self.host}. "
                "Make sure Ollama is running: `ollama serve`"
            )
        except httpx.TimeoutException:
            raise OllamaError("Ollama request timed out. The model may be loading.")

    def list_models(self) -> list[str]:
        """List all models available in Ollama."""
        try:
            resp = httpx.get(f"{self.host}/api/tags", timeout=5)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []
