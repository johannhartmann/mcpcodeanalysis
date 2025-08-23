"""Tests for embedding generation module."""

import hashlib
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import openai
import pytest

from src.indexer.embeddings import EmbeddingGenerator
from src.utils.exceptions import EmbeddingError


@pytest.fixture
def mock_settings() -> Generator[Any, None, None]:
    """Mock settings for testing."""
    with patch("src.indexer.embeddings.settings") as mock:
        mock.embeddings.model = "text-embedding-ada-002"
        mock.embeddings.batch_size = 10
        mock.embeddings.use_cache = True
        mock.embeddings.cache_dir = "/tmp/test_cache"
        mock.embeddings.max_tokens = 8000
        mock.embeddings.generate_interpreted = True
        yield mock


@pytest.fixture
def mock_openai_client() -> Generator[Any, None, None]:
    """Mock OpenAI client."""
    with patch("src.indexer.embeddings.openai.AsyncOpenAI") as mock_class:
        mock_client = AsyncMock()
        mock_class.return_value = mock_client

        # Mock embedding response
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        yield mock_client


@pytest.fixture
def mock_tiktoken() -> Generator[Any, None, None]:
    """Mock tiktoken encoder."""
    with patch("src.indexer.embeddings.tiktoken.encoding_for_model") as mock_encoding:
        mock_encoder = MagicMock()
        mock_encoder.encode = Mock(return_value=[1, 2, 3, 4, 5])
        mock_encoder.decode = Mock(return_value="decoded text")
        mock_encoding.return_value = mock_encoder
        yield mock_encoder


@pytest.fixture
def embedding_generator(
    mock_settings: Any, mock_openai_client: Any, mock_tiktoken: Any
) -> EmbeddingGenerator:
    """Create EmbeddingGenerator instance with mocked dependencies."""
    with (
        patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
        patch("src.indexer.embeddings.Path.mkdir"),
    ):
        return EmbeddingGenerator()


@pytest.mark.asyncio
async def test_embedding_generator_initialization(
    mock_settings: Any, mock_openai_client: Any, mock_tiktoken: Any
) -> None:
    """Test EmbeddingGenerator initialization."""
    with (
        patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
        patch("src.indexer.embeddings.Path.mkdir") as mock_mkdir,
    ):
        generator = EmbeddingGenerator()

        assert generator.config == mock_settings.embeddings
        assert generator.client is not None
        assert generator.encoding is not None
        assert generator.cache_dir == Path("/tmp/test_cache")
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


@pytest.mark.asyncio
async def test_embedding_generator_no_cache(
    mock_settings: Any, mock_openai_client: Any, mock_tiktoken: Any
) -> None:
    """Test EmbeddingGenerator with caching disabled."""
    mock_settings.embeddings.use_cache = False

    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        generator = EmbeddingGenerator()
        assert generator.cache_dir is None


@pytest.mark.asyncio
async def test_generate_embedding_success(
    embedding_generator: EmbeddingGenerator, mock_openai_client: Any
) -> None:
    """Test successful embedding generation."""
    text = "This is a test function"

    result = await embedding_generator.generate_embedding(text)

    assert isinstance(result, np.ndarray)
    assert result.shape == (1536,)
    assert result.dtype == np.float32

    mock_openai_client.embeddings.create.assert_called_once_with(
        model="text-embedding-ada-002",
        input=text,
        encoding_format="float",
    )


@pytest.mark.asyncio
async def test_generate_embedding_api_error(
    embedding_generator: EmbeddingGenerator, mock_openai_client: Any
) -> None:
    """Test embedding generation with API error."""
    mock_openai_client.embeddings.create.side_effect = openai.APIError(
        message="API Error", request=cast("Any", None), body=None
    )

    with pytest.raises(EmbeddingError) as exc_info:
        await embedding_generator.generate_embedding("test text")

    assert "Failed to generate embedding" in str(exc_info.value)


@pytest.mark.asyncio
async def test_generate_embedding_retry(
    embedding_generator: EmbeddingGenerator, mock_openai_client: Any
) -> None:
    """Test embedding generation with retry logic."""
    # First two calls fail, third succeeds
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.2] * 1536)]

    mock_openai_client.embeddings.create.side_effect = [
        openai.APIError(
            message="Temporary error", request=cast("Any", None), body=None
        ),
        openai.APIError(message="Another error", request=cast("Any", None), body=None),
        mock_response,
    ]

    result = await embedding_generator.generate_embedding("test text")

    assert isinstance(result, np.ndarray)
    assert mock_openai_client.embeddings.create.call_count == 3


@pytest.mark.asyncio
async def test_generate_embeddings_batch(
    embedding_generator: EmbeddingGenerator, mock_openai_client: Any
) -> None:
    """Test batch embedding generation."""
    texts = ["text1", "text2", "text3"]

    # Mock responses for batch
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1] * 1536),
        MagicMock(embedding=[0.2] * 1536),
        MagicMock(embedding=[0.3] * 1536),
    ]
    mock_openai_client.embeddings.create.return_value = mock_response

    # Mock cache to return None (no cached embeddings)
    with patch.object(embedding_generator, "_get_cached_embedding", return_value=None):
        results = await embedding_generator.generate_embeddings_batch(texts)

    assert len(results) == 3
    assert all(isinstance(r, np.ndarray) for r in results)
    assert all(r.shape == (1536,) for r in results)


@pytest.mark.asyncio
async def test_generate_embeddings_batch_with_cache(
    embedding_generator: EmbeddingGenerator, mock_openai_client: Any
) -> None:
    """Test batch embedding generation with cached values."""
    texts = ["cached_text", "new_text1", "new_text2"]

    # Mock cache to return embedding for first text
    cached_embedding = np.array([0.5] * 1536, dtype=np.float32)

    async def mock_get_cached(text: str) -> np.ndarray | None:
        if text == "cached_text":
            return cached_embedding
        return None

    # Mock response for uncached texts
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.2] * 1536),
        MagicMock(embedding=[0.3] * 1536),
    ]
    mock_openai_client.embeddings.create.return_value = mock_response

    with (
        patch.object(
            embedding_generator, "_get_cached_embedding", side_effect=mock_get_cached
        ),
        patch.object(embedding_generator, "_cache_embedding", new_callable=AsyncMock),
    ):
        results = await embedding_generator.generate_embeddings_batch(texts)

    assert len(results) == 3
    np.testing.assert_array_equal(results[0], cached_embedding)

    # Only uncached texts should be sent to API
    mock_openai_client.embeddings.create.assert_called_once()
    call_args = mock_openai_client.embeddings.create.call_args
    assert call_args[1]["input"] == ["new_text1", "new_text2"]


@pytest.mark.asyncio
async def test_generate_embeddings_batch_partial_failure(
    embedding_generator: EmbeddingGenerator, mock_openai_client: Any
) -> None:
    """Test batch embedding with partial failure fallback."""
    texts = ["text1", "text2", "text3"]

    # Mock batch call to fail
    mock_openai_client.embeddings.create.side_effect = openai.APIError(
        message="Batch failed", request=cast("Any", None), body=None
    )

    # Mock individual calls
    _individual_responses = [
        MagicMock(data=[MagicMock(embedding=[0.1] * 1536)]),
        MagicMock(data=[MagicMock(embedding=[0.2] * 1536)]),
        openai.APIError(
            message="Individual call failed", request=cast("Any", None), body=None
        ),  # One fails
    ]

    with patch.object(embedding_generator, "generate_embedding") as mock_individual:
        mock_individual.side_effect = [
            np.array([0.1] * 1536, dtype=np.float32),
            np.array([0.2] * 1536, dtype=np.float32),
            EmbeddingError("Failed"),
        ]

        results = await embedding_generator.generate_embeddings_batch(texts)

    assert len(results) == 3
    assert results[0].mean() == 0.1
    assert results[1].mean() == 0.2
    assert results[2].mean() == 0.0  # Zero embedding for failed


@pytest.mark.asyncio
async def test_count_tokens(
    embedding_generator: EmbeddingGenerator, mock_tiktoken: Any
) -> None:
    """Test token counting."""
    text = "This is a test"
    mock_tiktoken.encode.return_value = [1, 2, 3, 4, 5]

    count = embedding_generator.count_tokens(text)

    assert count == 5
    mock_tiktoken.encode.assert_called_once_with(text)


@pytest.mark.asyncio
async def test_truncate_text_no_truncation(
    embedding_generator: EmbeddingGenerator, mock_tiktoken: Any
) -> None:
    """Test text truncation when text is within limit."""
    text = "Short text"
    mock_tiktoken.encode.return_value = [1, 2, 3]

    result = embedding_generator.truncate_text(text, max_tokens=10)

    assert result == text
    mock_tiktoken.encode.assert_called_once()
    mock_tiktoken.decode.assert_not_called()


@pytest.mark.asyncio
async def test_truncate_text_with_truncation(
    embedding_generator: EmbeddingGenerator, mock_tiktoken: Any
) -> None:
    """Test text truncation when text exceeds limit."""
    text = "This is a very long text that needs truncation"
    mock_tiktoken.encode.return_value = list(range(20))
    mock_tiktoken.decode.return_value = "Truncated text"

    result = embedding_generator.truncate_text(text, max_tokens=10)

    assert result == "Truncated text"
    mock_tiktoken.encode.assert_called_once_with(text)
    mock_tiktoken.decode.assert_called_once_with(list(range(10)))


@pytest.mark.asyncio
async def test_generate_code_embeddings_raw_only(
    embedding_generator: EmbeddingGenerator,
) -> None:
    """Test generating raw code embeddings only."""
    embedding_generator.config.generate_interpreted = False

    code = "def test(): pass"
    context = "File: test.py"

    with patch.object(embedding_generator, "generate_embedding") as mock_generate:
        mock_generate.return_value = np.array([0.1] * 1536)

        raw, interpreted = await embedding_generator.generate_code_embeddings(
            code, None, context
        )

    assert isinstance(raw, np.ndarray)
    assert interpreted is None

    # Check that context was included
    expected_text = f"{context}\n\n{code}"
    mock_generate.assert_called_once()
    call_text = mock_generate.call_args[0][0]
    assert call_text == expected_text


@pytest.mark.asyncio
async def test_generate_code_embeddings_with_interpretation(
    embedding_generator: EmbeddingGenerator,
) -> None:
    """Test generating both raw and interpreted embeddings."""
    code = "def greet(name): return f'Hello {name}'"
    description = "Function that greets a person by name"
    context = "File: greetings.py"

    with patch.object(embedding_generator, "generate_embedding") as mock_generate:
        mock_generate.side_effect = [
            np.array([0.1] * 1536),  # Raw embedding
            np.array([0.2] * 1536),  # Interpreted embedding
        ]

        raw, interpreted = await embedding_generator.generate_code_embeddings(
            code, description, context
        )

    assert isinstance(raw, np.ndarray)
    assert isinstance(interpreted, np.ndarray)
    assert not np.array_equal(raw, interpreted)

    assert mock_generate.call_count == 2

    # Check raw call
    raw_call = mock_generate.call_args_list[0][0][0]
    assert code in raw_call
    assert context in raw_call

    # Check interpreted call
    interpreted_call = mock_generate.call_args_list[1][0][0]
    assert description in interpreted_call
    assert context in interpreted_call


@pytest.mark.asyncio
async def test_get_cached_embedding_exists(
    embedding_generator: EmbeddingGenerator, tmp_path: Path
) -> None:
    """Test retrieving cached embedding."""
    embedding_generator.cache_dir = tmp_path

    # Create cached embedding
    text = "cached text"
    cache_key = hashlib.sha256(text.encode()).hexdigest()
    cached_embedding = np.array([0.5] * 1536, dtype=np.float32)

    cache_file = tmp_path / f"{cache_key}.npy"
    np.save(cache_file, cached_embedding)

    result = await embedding_generator._get_cached_embedding(text)

    assert result is not None
    np.testing.assert_array_equal(result, cached_embedding)


@pytest.mark.asyncio
async def test_get_cached_embedding_not_exists(
    embedding_generator: EmbeddingGenerator, tmp_path: Path
) -> None:
    """Test retrieving non-existent cached embedding."""
    embedding_generator.cache_dir = tmp_path

    result = await embedding_generator._get_cached_embedding("non-cached text")

    assert result is None


@pytest.mark.asyncio
async def test_get_cached_embedding_no_cache_dir(
    embedding_generator: EmbeddingGenerator,
) -> None:
    """Test cached embedding when caching is disabled."""
    embedding_generator.cache_dir = None

    result = await embedding_generator._get_cached_embedding("any text")

    assert result is None


@pytest.mark.asyncio
async def test_get_cached_embedding_corrupted(
    embedding_generator: EmbeddingGenerator, tmp_path: Path
) -> None:
    """Test handling corrupted cache file."""
    embedding_generator.cache_dir = tmp_path

    text = "corrupted cache"
    cache_key = hashlib.sha256(text.encode()).hexdigest()
    cache_file = tmp_path / f"{cache_key}.npy"

    # Write corrupted data
    cache_file.write_text("corrupted data")

    result = await embedding_generator._get_cached_embedding(text)

    assert result is None


@pytest.mark.asyncio
async def test_cache_embedding_success(
    embedding_generator: EmbeddingGenerator, tmp_path: Path
) -> None:
    """Test successfully caching an embedding."""
    embedding_generator.cache_dir = tmp_path

    text = "text to cache"
    embedding = np.array([0.3] * 1536, dtype=np.float32)

    await embedding_generator._cache_embedding(text, embedding)

    # Verify file was created
    cache_key = hashlib.sha256(text.encode()).hexdigest()
    cache_file = tmp_path / f"{cache_key}.npy"

    assert cache_file.exists()

    # Verify content
    loaded = np.load(cache_file)
    np.testing.assert_array_equal(loaded, embedding)


@pytest.mark.asyncio
async def test_cache_embedding_no_cache_dir(
    embedding_generator: EmbeddingGenerator,
) -> None:
    """Test caching when cache is disabled."""
    embedding_generator.cache_dir = None

    # Should not raise any error
    await embedding_generator._cache_embedding("text", np.array([0.1] * 1536))


@pytest.mark.asyncio
async def test_cache_embedding_write_error(
    embedding_generator: EmbeddingGenerator, tmp_path: Path
) -> None:
    """Test handling cache write errors."""
    embedding_generator.cache_dir = tmp_path

    # Make directory read-only
    tmp_path.chmod(0o444)

    try:
        # Should not raise, just log warning
        await embedding_generator._cache_embedding("text", np.array([0.1] * 1536))
    finally:
        # Restore permissions
        tmp_path.chmod(0o755)


@pytest.mark.asyncio
async def test_large_batch_processing(
    embedding_generator: EmbeddingGenerator, mock_openai_client: Any
) -> None:
    """Test processing large batches that exceed batch size."""
    # Create 25 texts (batch size is 10)
    texts = [f"text_{i}" for i in range(25)]

    # Mock responses for 3 batches
    mock_responses = []
    for i in range(3):
        batch_size = 10 if i < 2 else 5
        embeddings = [MagicMock(embedding=[float(i)] * 1536) for _ in range(batch_size)]
        response = MagicMock(data=embeddings)
        mock_responses.append(response)

    mock_openai_client.embeddings.create.side_effect = mock_responses

    with (
        patch.object(embedding_generator, "_get_cached_embedding", return_value=None),
        patch.object(embedding_generator, "_cache_embedding", new_callable=AsyncMock),
    ):
        results = await embedding_generator.generate_embeddings_batch(texts)

    assert len(results) == 25
    assert mock_openai_client.embeddings.create.call_count == 3

    # Verify batch sizes
    calls = mock_openai_client.embeddings.create.call_args_list
    assert len(calls[0][1]["input"]) == 10
    assert len(calls[1][1]["input"]) == 10
    assert len(calls[2][1]["input"]) == 5


@pytest.mark.asyncio
async def test_embedding_dimension_validation(
    embedding_generator: EmbeddingGenerator, mock_openai_client: Any
) -> None:
    """Test that embeddings have correct dimensions."""
    # Mock response with wrong dimension
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1] * 100)]  # Wrong size
    mock_openai_client.embeddings.create.return_value = mock_response

    result = await embedding_generator.generate_embedding("test")

    # Should still return numpy array with the actual dimensions
    assert result.shape == (100,)
    assert result.dtype == np.float32
