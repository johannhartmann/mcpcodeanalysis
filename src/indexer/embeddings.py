"""Embedding generation using OpenAI API."""

import hashlib
from pathlib import Path

import numpy as np
import openai
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential

from src.mcp_server.config import config, settings
from src.utils.exceptions import EmbeddingError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for code using OpenAI API."""

    def __init__(self) -> None:
        self.config = config.embeddings
        self.client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        self.encoding = tiktoken.encoding_for_model(self.config.model)
        self.cache_dir = Path(self.config.cache_dir) if self.config.use_cache else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        try:
            response = await self.client.embeddings.create(
                model=self.config.model,
                input=text,
                encoding_format="float",
            )

            embedding = response.data[0].embedding
            return np.array(embedding, dtype=np.float32)

        except Exception as e:
            logger.exception("Error generating embedding: %s")
            raise EmbeddingError(f"Failed to generate embedding: {e}") from e

    async def generate_embeddings_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts in batch."""
        embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]

            try:
                # Check cache first
                batch_embeddings = []
                uncached_texts = []
                uncached_indices = []

                for j, text in enumerate(batch):
                    cached = await self._get_cached_embedding(text)
                    if cached is not None:
                        batch_embeddings.append((j, cached))
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(j)

                # Generate embeddings for uncached texts
                if uncached_texts:
                    response = await self.client.embeddings.create(
                        model=self.config.model,
                        input=uncached_texts,
                        encoding_format="float",
                    )

                    for idx, embedding_data in zip(
                        uncached_indices,
                        response.data,
                        strict=False,
                    ):
                        embedding = np.array(embedding_data.embedding, dtype=np.float32)
                        batch_embeddings.append((idx, embedding))

                        # Cache the embedding
                        if self.cache_dir:
                            await self._cache_embedding(
                                uncached_texts[uncached_indices.index(idx)],
                                embedding,
                            )

                # Sort by original index and extract embeddings
                batch_embeddings.sort(key=lambda x: x[0])
                embeddings.extend([emb for _, emb in batch_embeddings])

            except Exception:
                logger.exception("Error generating batch embeddings: %s")
                # Generate individually for failed batch
                for text in batch:
                    try:
                        embedding = await self.generate_embedding(text)
                        embeddings.append(embedding)
                    except Exception:
                        # Use zero embedding as fallback
                        embeddings.append(np.zeros(1536, dtype=np.float32))

        return embeddings

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        tokens = self.encoding.encode(text)

        if len(tokens) <= max_tokens:
            return text

        # Truncate and decode
        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)

    async def generate_code_embeddings(
        self,
        code: str,
        description: str | None = None,
        context: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
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
                return np.load(cache_file)
            except Exception as e:
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
        except Exception as e:
            logger.warning("Failed to cache embedding: %s", e)
