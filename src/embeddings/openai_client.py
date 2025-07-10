"""OpenAI API client for generating embeddings."""

import asyncio
from typing import Any

import httpx
import openai
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.mcp_server.config import get_settings
from src.utils.exceptions import EmbeddingError, OpenAIError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OpenAIClient:
    """Client for OpenAI API operations."""

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize OpenAI client.

        Args:
            api_key: OpenAI API key. If not provided, will use from settings.
        """
        self.settings = get_settings()
        self.api_key = api_key or self.settings.openai_api_key.get_secret_value()
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.embedding_model = self.settings.embeddings.model
        self.max_tokens = self.settings.embeddings.max_tokens
        self.batch_size = self.settings.embeddings.batch_size

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, openai.APIError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to generate embedding for

        Returns:
            Embedding vector

        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            # Truncate text if too long
            truncated_text = self._truncate_text(text)

            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=truncated_text,
            )

            embedding = response.data[0].embedding
            logger.debug(
                "Generated embedding for text of length %d, dimension: %d",
                len(text),
                len(embedding),
            )

            return embedding

        except openai.APIError as e:
            logger.exception("OpenAI API error: %s")
            msg = "API error"
            raise OpenAIError(msg) from e
        except Exception as e:
            logger.exception("Unexpected error generating embedding: %s")
            raise EmbeddingError("Error") from e

    async def generate_embeddings_batch(
        self,
        texts: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate embeddings for multiple texts in batches.

        Args:
            texts: List of texts to generate embeddings for
            metadata: Optional metadata for each text

        Returns:
            List of dicts with 'text', 'embedding', and optional 'metadata'
        """
        if not texts:
            return []

        if metadata and len(metadata) != len(texts):
            msg = "Length mismatch"
            raise ValueError(msg)

        results = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            batch_metadata = (
                metadata[i : i + self.batch_size]
                if metadata
                else [None] * len(batch_texts)
            )

            logger.info(
                "Processing embedding batch %s (%s items)",
                i // self.batch_size + 1,
                len(batch_texts),
            )

            # Generate embeddings concurrently within batch
            tasks = [
                self._generate_embedding_with_metadata(text, meta)
                for text, meta in zip(batch_texts, batch_metadata, strict=False)
            ]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle results and errors
            for idx, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(
                        "Failed to generate embedding for text %s: %s",
                        i + idx,
                        result,
                    )
                    # Store None for failed embeddings
                    results.append(
                        {
                            "text": batch_texts[idx],
                            "embedding": None,
                            "metadata": batch_metadata[idx],
                            "error": str(result),
                        },
                    )
                else:
                    results.append(result)

            # Small delay between batches to avoid rate limits
            if i + self.batch_size < len(texts):
                await asyncio.sleep(0.1)

        successful = sum(1 for r in results if r.get("embedding") is not None)
        logger.info("Generated %d/%d embeddings successfully", successful, len(texts))

        return results

    async def _generate_embedding_with_metadata(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate embedding with metadata.

        Args:
            text: Text to generate embedding for
            metadata: Optional metadata

        Returns:
            Dict with text, embedding, and metadata
        """
        embedding = await self.generate_embedding(text)

        # Estimate tokens (rough approximation)
        tokens = len(text.split()) * 1.3

        return {
            "text": text,
            "embedding": embedding,
            "metadata": metadata,
            "tokens": int(tokens),
        }

    def _truncate_text(self, text: str) -> str:
        """Truncate text to fit within token limits.

        Args:
            text: Text to truncate

        Returns:
            Truncated text
        """
        # Simple approximation: ~4 characters per token
        max_chars = self.max_tokens * 4

        if len(text) <= max_chars:
            return text

        # Truncate and add ellipsis
        truncated = text[: max_chars - 3] + "..."
        logger.debug("Truncated text from %d to %d chars", len(text), len(truncated))

        return truncated

    async def test_connection(self) -> bool:
        """Test connection to OpenAI API.

        Returns:
            True if connection successful
        """
        try:
            # Try to generate a simple embedding
            await self.generate_embedding("test")
            logger.info("OpenAI API connection successful")
            return True
        except Exception:
            logger.exception("OpenAI API connection failed: %s")
            return False

    async def estimate_cost(self, num_texts: int) -> dict[str, float]:
        """Estimate cost for generating embeddings.

        Args:
            num_texts: Number of texts to generate embeddings for

        Returns:
            Dict with cost estimates
        """
        # Pricing as of 2024 (may need updates)
        pricing = {
            "text-embedding-ada-002": 0.0001,  # per 1K tokens
            "text-embedding-3-small": 0.00002,  # per 1K tokens
            "text-embedding-3-large": 0.00013,  # per 1K tokens
        }

        price_per_1k = pricing.get(self.embedding_model, 0.0001)

        # Estimate average tokens per text
        avg_tokens_per_text = self.max_tokens * 0.5  # Assume 50% usage
        total_tokens = num_texts * avg_tokens_per_text

        estimated_cost = (total_tokens / 1000) * price_per_1k

        return {
            "model": self.embedding_model,
            "num_texts": num_texts,
            "estimated_tokens": total_tokens,
            "estimated_cost_usd": round(estimated_cost, 4),
            "price_per_1k_tokens": price_per_1k,
        }
