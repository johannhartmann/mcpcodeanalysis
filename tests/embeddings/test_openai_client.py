"""Tests for OpenAI client."""

from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest
from openai.types import CreateEmbeddingResponse, Embedding
from openai.types.create_embedding_response import Usage

from src.embeddings.openai_client import OpenAIClient
from src.utils.exceptions import EmbeddingError


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = MagicMock()
    settings.openai_api_key.get_secret_value.return_value = "test-api-key"
    settings.embeddings.model = "text-embedding-ada-002"
    settings.embeddings.max_tokens = 8191
    settings.embeddings.batch_size = 100
    return settings


@pytest.fixture
def openai_client(mock_settings):
    """Create OpenAI client fixture."""
    with patch("src.embeddings.openai_client.get_settings", return_value=mock_settings):
        return OpenAIClient()


@pytest.fixture
def mock_embedding_response():
    """Create mock embedding response."""
    embedding_data = Embedding(
        index=0,
        embedding=[0.1] * 1536,  # Ada-002 dimension
        object="embedding",
    )

    usage = Usage(
        prompt_tokens=10,
        total_tokens=10,
    )

    return CreateEmbeddingResponse(
        data=[embedding_data],
        model="text-embedding-ada-002",
        object="list",
        usage=usage,
    )


class TestOpenAIClient:
    """Tests for OpenAIClient class."""

    def test_init_with_api_key(self, mock_settings) -> None:
        """Test initialization with API key."""
        with patch(
            "src.embeddings.openai_client.get_settings",
            return_value=mock_settings,
        ):
            client = OpenAIClient(api_key="custom-key")
            assert client.api_key == "custom-key"
            assert client.embedding_model == "text-embedding-ada-002"
            assert client.max_tokens == 8191
            assert client.batch_size == 100

    def test_init_from_settings(self, openai_client) -> None:
        """Test initialization from settings."""
        assert openai_client.api_key == "test-api-key"
        assert openai_client.embedding_model == "text-embedding-ada-002"

    @pytest.mark.asyncio
    async def test_generate_embedding_success(
        self,
        openai_client,
        mock_embedding_response,
    ) -> None:
        """Test successful embedding generation."""
        with patch.object(
            openai_client.client.embeddings,
            "create",
            return_value=mock_embedding_response,
        ):
            embedding = await openai_client.generate_embedding("test text")

            assert isinstance(embedding, list)
            assert len(embedding) == 1536
            assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_generate_embedding_api_error(self, openai_client) -> None:
        """Test embedding generation with API error."""
        # Create a mock request object
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.method = "POST"
        mock_request.url = "https://api.openai.com/v1/embeddings"
        
        with (
            patch.object(
                openai_client.client.embeddings,
                "create",
                side_effect=openai.APIError("API Error", mock_request, body=None),
            ),
            pytest.raises(EmbeddingError, match="Failed to generate embedding"),
        ):
            await openai_client.generate_embedding("test text")

    @pytest.mark.asyncio
    async def test_generate_embedding_unexpected_error(self, openai_client) -> None:
        """Test embedding generation with unexpected error."""
        with (
            patch.object(
                openai_client.client.embeddings,
                "create",
                side_effect=Exception("Unexpected error"),
            ),
            pytest.raises(EmbeddingError, match="Unexpected error"),
        ):
            await openai_client.generate_embedding("test text")

    def test_truncate_text_short(self, openai_client) -> None:
        """Test text truncation with short text."""
        text = "Short text"
        result = openai_client._truncate_text(text)
        assert result == text

    def test_truncate_text_long(self, openai_client) -> None:
        """Test text truncation with long text."""
        # Create text longer than max_tokens * 4
        text = "a" * (openai_client.max_tokens * 4 + 100)
        result = openai_client._truncate_text(text)

        assert len(result) < len(text)
        assert result.endswith("...")
        assert len(result) == openai_client.max_tokens * 4

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_empty(self, openai_client) -> None:
        """Test batch embedding generation with empty list."""
        results = await openai_client.generate_embeddings_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_success(
        self,
        openai_client,
        mock_embedding_response,
    ) -> None:
        """Test successful batch embedding generation."""
        texts = ["text1", "text2", "text3"]
        metadata = [{"id": 1}, {"id": 2}, {"id": 3}]

        with patch.object(
            openai_client,
            "generate_embedding",
            return_value=[0.1] * 1536,
        ):
            results = await openai_client.generate_embeddings_batch(texts, metadata)

            assert len(results) == 3
            for i, result in enumerate(results):
                assert result["text"] == texts[i]
                assert result["embedding"] == [0.1] * 1536
                assert result["metadata"] == metadata[i]

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_with_errors(self, openai_client) -> None:
        """Test batch embedding generation with some errors."""
        texts = ["text1", "text2", "text3"]

        # Mock to fail on second text
        async def mock_generate(text):
            if text == "text2":
                raise Exception("Failed for text2")
            return [0.1] * 1536

        with patch.object(
            openai_client,
            "generate_embedding",
            side_effect=mock_generate,
        ):
            results = await openai_client.generate_embeddings_batch(texts)

            assert len(results) == 3
            assert results[0]["embedding"] is not None
            assert results[1]["embedding"] is None
            assert results[1]["error"] == "Failed for text2"
            assert results[2]["embedding"] is not None

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_metadata_mismatch(
        self,
        openai_client,
    ) -> None:
        """Test batch embedding with mismatched metadata."""
        texts = ["text1", "text2"]
        metadata = [{"id": 1}]  # Only one metadata item

        with pytest.raises(ValueError, match="Metadata length must match texts length"):
            await openai_client.generate_embeddings_batch(texts, metadata)

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_large(self, openai_client) -> None:
        """Test batch embedding with more texts than batch size."""
        # Create more texts than batch size
        texts = [f"text{i}" for i in range(openai_client.batch_size + 10)]

        call_count = 0

        async def mock_generate_with_metadata(text, metadata):
            nonlocal call_count
            call_count += 1
            return {
                "text": text,
                "embedding": [0.1] * 10,  # Smaller for test
                "metadata": metadata,
            }

        with patch.object(
            openai_client,
            "_generate_embedding_with_metadata",
            side_effect=mock_generate_with_metadata,
        ):
            results = await openai_client.generate_embeddings_batch(texts)

            assert len(results) == len(texts)
            assert call_count == len(texts)

    @pytest.mark.asyncio
    async def test_test_connection_success(self, openai_client) -> None:
        """Test successful connection test."""
        with patch.object(
            openai_client,
            "generate_embedding",
            return_value=[0.1] * 1536,
        ):
            result = await openai_client.test_connection()
            assert result is True

    @pytest.mark.asyncio
    async def test_test_connection_failure(self, openai_client) -> None:
        """Test failed connection test."""
        with patch.object(
            openai_client,
            "generate_embedding",
            side_effect=Exception("Connection failed"),
        ):
            result = await openai_client.test_connection()
            assert result is False

    @pytest.mark.asyncio
    async def test_estimate_cost(self, openai_client) -> None:
        """Test cost estimation."""
        result = await openai_client.estimate_cost(1000)

        assert result["model"] == "text-embedding-ada-002"
        assert result["num_texts"] == 1000
        assert result["estimated_tokens"] > 0
        assert result["estimated_cost_usd"] > 0
        assert result["price_per_1k_tokens"] == 0.0001
