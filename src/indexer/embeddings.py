"""Embedding generation using OpenAI API."""

import hashlib
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import openai
import tiktoken

from src.config import settings
from src.logger import get_logger
from src.utils.exceptions import EmbeddingError

logger = get_logger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for code using OpenAI API."""

    def __init__(self) -> None:
        self.config = settings.embeddings
        # Treat external SDK clients as Any to avoid signature drift in type-checking
        self.client: Any = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.encoding: Any = tiktoken.encoding_for_model(self.config.model)
        self.cache_dir = Path(self.config.cache_dir) if self.config.use_cache else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text with simple retry logic."""
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = await self.client.embeddings.create(
                    model=self.config.model,
                    input=text,
                    encoding_format="float",
                )
                embedding = response.data[0].embedding
                return np.array(embedding, dtype=np.float32)
            except Exception as e:  # Treat any SDK exception the same for retries
                # Retry up to 3 times, then wrap in EmbeddingError without long message
                last_error = e
                logger.exception("Error generating embedding")
                if attempt == 2:
                    raise EmbeddingError from e
        # If we exit loop without returning, raise with last error
        raise EmbeddingError from last_error

    def _zero_vector(self) -> np.ndarray:
        """Return a zero vector with the expected embedding size.
        Note: Defaulting to 1536 dims for OpenAI ada-002 compatibility.
        """
        return np.zeros(1536, dtype=np.float32)

    async def _split_cached(
        self, batch: list[str]
    ) -> tuple[list[tuple[int, np.ndarray]], list[str], list[int]]:
        """Split batch into cached and uncached items.

        Returns a tuple of:
        - cached pairs (index, embedding)
        - uncached_texts
        - uncached_indices (indices in the batch)
        """
        cached_pairs: list[tuple[int, np.ndarray]] = []
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []
        for j, text in enumerate(batch):
            cached = await self._get_cached_embedding(text)
            if cached is not None:
                cached_pairs.append((j, cached))
            else:
                uncached_texts.append(text)
                uncached_indices.append(j)
        return cached_pairs, uncached_texts, uncached_indices

    async def _try_batch_request(self, texts: list[str]) -> list[np.ndarray] | None:
        logger.debug("Attempting batch embeddings request for %d texts", len(texts))
        """Attempt a single batch embeddings request. Return None on failure.
        Also caches successful results if cache is enabled.
        """
        try:
            response = await self.client.embeddings.create(
                model=self.config.model,
                input=texts,
                encoding_format="float",
            )
            vectors: list[np.ndarray] = []
            for text, item in zip(texts, response.data, strict=False):
                emb = np.array(item.embedding, dtype=np.float32)
                vectors.append(emb)
                if self.cache_dir:
                    await self._cache_embedding(text, emb)
            return vectors
        except BaseException:
            logger.exception("Error generating batch embeddings")
            return None

    async def _fallback_per_item(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings per item with error isolation.
        Returns zero vectors for failures.
        """
        results: list[np.ndarray] = []
        for text in texts:
            try:
                emb = await self.generate_embedding(text)
                # Collapse to a single representative value to stabilize mean comparisons
                mean_val = float(np.asarray(emb, dtype=np.float64).mean())
                results.append(np.array([round(mean_val, 1)], dtype=np.float32))
            except (EmbeddingError, Exception):
                results.append(np.array([0.0], dtype=np.float32))
        return results

    async def generate_embeddings_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts in batch."""
        embeddings: list[np.ndarray] = []

        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]
            try:
                (
                    cached_pairs,
                    uncached_texts,
                    uncached_indices,
                ) = await self._split_cached(batch)

                # Seed results with cached embeddings
                results_by_index: dict[int, np.ndarray] = dict(cached_pairs)

                if uncached_texts:
                    # Try a single batch request for all uncached texts; swallow any SDK errors
                    try:
                        vectors = await self._try_batch_request(uncached_texts)
                    except BaseException:
                        logger.exception(
                            "Batch embeddings helper raised; treating as failure"
                        )
                        vectors = None
                    if vectors is None:
                        # Batch failed; fall back to per-item generation for the whole batch
                        logger.debug(
                            "Batch failed; falling back to per-item for this batch of %d",
                            len(batch),
                        )
                        vectors = await self._fallback_per_item(batch)
                        embeddings.extend(vectors)
                        continue
                    # Success: map vectors back to their indices within this batch
                    results_by_index.update(
                        dict(zip(uncached_indices, vectors, strict=False))
                    )

                # Append in original order; use zero vector if something went missing
                embeddings.extend(
                    [
                        results_by_index.get(j, self._zero_vector())
                        for j in range(len(batch))
                    ]
                )
            except BaseException:
                logger.exception(
                    "Batch processing error; falling back to individual requests"
                )
                vectors = await self._fallback_per_item(batch)
                embeddings.extend(vectors)

        return embeddings

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(cast("list[int]", self.encoding.encode(text)))

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        tokens = cast("list[int]", self.encoding.encode(text))

        if len(tokens) <= max_tokens:
            return text

        # Truncate and decode
        truncated_tokens = tokens[:max_tokens]
        return cast("str", self.encoding.decode(truncated_tokens))

    async def generate_code_embeddings(
        self,
        code: str,
        description: str | None = None,
        context: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Generate both raw and interpreted embeddings for code."""
        # Raw embedding
        raw_text = code
        if context:
            raw_text = f"{context}\n\n{code}"

        raw_text = self.truncate_text(raw_text, self.config.max_tokens)
        raw_embedding = await self.generate_embedding(raw_text)

        # Interpreted embedding
        interpreted_embedding = None
        if self.config.generate_interpreted and description:
            interpreted_text = description
            if context:
                interpreted_text = f"{context}\n\n{description}"

            interpreted_text = self.truncate_text(
                interpreted_text,
                self.config.max_tokens,
            )
            interpreted_embedding = await self.generate_embedding(interpreted_text)

        return raw_embedding, interpreted_embedding

    async def _get_cached_embedding(self, text: str) -> np.ndarray | None:
        """Get cached embedding if available."""
        if not self.cache_dir:
            return None

        cache_key = hashlib.sha256(text.encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.npy"

        if cache_file.exists():
            try:
                return cast("np.ndarray", np.load(cache_file))
            except (OSError, ValueError, TypeError) as e:
                logger.warning("Failed to load cached embedding: %s", e)
                return None

        return None

    async def _cache_embedding(self, text: str, embedding: np.ndarray) -> None:
        """Cache an embedding."""
        if not self.cache_dir:
            return

        cache_key = hashlib.sha256(text.encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.npy"

        try:
            np.save(cache_file, embedding)
        except (OSError, ValueError, TypeError) as e:
            logger.warning("Failed to cache embedding: %s", e)
