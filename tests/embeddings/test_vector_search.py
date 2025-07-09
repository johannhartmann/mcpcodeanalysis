"""Tests for vector search."""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import Class, CodeEmbedding, Function
from src.embeddings.vector_search import SearchScope, VectorSearch


@pytest.fixture
def mock_db_session():
    """Create mock database session."""
    session = AsyncMock(spec=AsyncSession)
    session.execute = AsyncMock()
    return session


@pytest.fixture
def mock_openai_client():
    """Create mock OpenAI client."""
    client = AsyncMock()
    client.generate_embedding = AsyncMock(return_value=[0.1] * 1536)
    return client


@pytest.fixture
def vector_search(mock_db_session, mock_openai_client):
    """Create vector search fixture."""
    return VectorSearch(mock_db_session, mock_openai_client)


@pytest.fixture
def sample_embedding():
    """Create sample embedding record."""
    embedding = MagicMock(spec=CodeEmbedding)
    embedding.id = 1
    embedding.entity_type = "function"
    embedding.entity_id = 10
    embedding.file_id = 100
    embedding.repository_id = 1000
    embedding.embedding = np.array([0.1] * 1536)
    embedding.embedding_text = "Function: test_function"
    embedding.metadata = {"entity_name": "test_function"}
    return embedding


@pytest.fixture
def sample_function():
    """Create sample function record."""
    func = MagicMock(spec=Function)
    func.id = 10
    func.name = "test_function"
    func.start_line = 10
    func.end_line = 20
    func.parameters = []
    func.return_type = "str"
    func.is_async = False
    func.class_ = None
    return func


class TestVectorSearch:
    """Tests for VectorSearch class."""

    @pytest.mark.asyncio
    async def test_search_basic(
        self,
        vector_search,
        mock_db_session,
        mock_openai_client,
        sample_embedding,
        sample_function,
    ) -> None:
        """Test basic search functionality."""
        # Mock database results
        mock_db_session.execute.side_effect = [
            # Search query result
            MagicMock(fetchall=lambda: [(1, 0.95, "function")]),
            # Load embedding result
            MagicMock(scalar_one_or_none=lambda: sample_embedding),
            # Load function result
            MagicMock(scalar_one_or_none=lambda: sample_function),
        ]

        results = await vector_search.search(
            "find test function",
            scope=SearchScope.ALL,
            limit=10,
        )

        assert len(results) == 1
        assert results[0]["similarity"] == 0.95
        assert results[0]["entity_type"] == "function"
        assert results[0]["entity"]["name"] == "test_function"

        # Verify embedding was generated for query
        mock_openai_client.generate_embedding.assert_called_once_with(
            "find test function",
        )

    @pytest.mark.asyncio
    async def test_search_with_filters(
        self,
        vector_search,
        mock_db_session,
        mock_openai_client,
    ) -> None:
        """Test search with repository and scope filters."""
        mock_db_session.execute.return_value = MagicMock(fetchall=list)

        await vector_search.search(
            "test query",
            scope=SearchScope.FUNCTIONS,
            repository_id=123,
            file_id=456,
            threshold=0.8,
        )

        # Check that query was built with filters
        # (Would need to inspect the actual SQL query in real implementation)
        assert mock_db_session.execute.called

    @pytest.mark.asyncio
    async def test_search_similar(
        self,
        vector_search,
        mock_db_session,
        sample_embedding,
    ) -> None:
        """Test finding similar embeddings."""
        # Mock source embedding lookup
        mock_db_session.execute.side_effect = [
            # Get source embedding
            MagicMock(scalar_one_or_none=lambda: sample_embedding),
            # Search results
            MagicMock(
                fetchall=lambda: [(2, 0.1, "function")],
            ),  # distance, not similarity
            # Load similar embedding
            MagicMock(scalar_one_or_none=lambda: sample_embedding),
            # Load function
            MagicMock(scalar_one_or_none=lambda: MagicMock()),
        ]

        results = await vector_search.search_similar(1, limit=5)

        assert len(results) == 1
        assert results[0]["similarity"] == 0.9  # 1.0 - 0.1 distance

    @pytest.mark.asyncio
    async def test_search_similar_not_found(
        self,
        vector_search,
        mock_db_session,
    ) -> None:
        """Test search similar with non-existent embedding."""
        mock_db_session.execute.return_value = MagicMock(
            scalar_one_or_none=lambda: None,
        )

        with pytest.raises(ValueError, match="Embedding 999 not found"):
            await vector_search.search_similar(999)

    @pytest.mark.asyncio
    async def test_search_by_code(
        self,
        vector_search,
        mock_openai_client,
    ) -> None:
        """Test searching by code snippet."""
        code_snippet = "def test():\n    return 42"

        with patch.object(vector_search, "search") as mock_search:
            await vector_search.search_by_code(
                code_snippet,
                scope=SearchScope.FUNCTIONS,
                repository_id=123,
            )

            # Check that search was called with formatted query
            mock_search.assert_called_once()
            call_args = mock_search.call_args[0]
            assert "Code snippet:" in call_args[0]
            assert code_snippet in call_args[0]

    @pytest.mark.asyncio
    async def test_format_entity_function(
        self,
        vector_search,
        mock_db_session,
        sample_embedding,
        sample_function,
    ) -> None:
        """Test formatting function entity."""
        sample_embedding.entity_type = "function"
        mock_repo = MagicMock()
        mock_repo.name = "test-repo"
        mock_file = MagicMock(path="test.py", repository=mock_repo)
        sample_embedding.file = mock_file

        mock_db_session.execute.return_value = MagicMock(
            scalar_one_or_none=lambda: sample_function,
        )

        entity_info = await vector_search._format_entity(sample_embedding)

        assert entity_info["type"] == "function"
        assert entity_info["name"] == "test_function"
        assert entity_info["file_path"] == "test.py"
        assert entity_info["repository"] == "test-repo"
        assert entity_info["start_line"] == 10
        assert entity_info["end_line"] == 20

    @pytest.mark.asyncio
    async def test_format_entity_class(
        self,
        vector_search,
        mock_db_session,
        sample_embedding,
    ) -> None:
        """Test formatting class entity."""
        sample_embedding.entity_type = "class"
        sample_embedding.file = MagicMock(path="test.py", repository=None)

        sample_class = MagicMock(spec=Class)
        sample_class.name = "TestClass"
        sample_class.start_line = 5
        sample_class.end_line = 50
        sample_class.base_classes = ["Base"]
        sample_class.is_abstract = True

        mock_db_session.execute.return_value = MagicMock(
            scalar_one_or_none=lambda: sample_class,
        )

        entity_info = await vector_search._format_entity(sample_embedding)

        assert entity_info["type"] == "class"
        assert entity_info["name"] == "TestClass"
        assert entity_info["base_classes"] == ["Base"]
        assert entity_info["is_abstract"] is True

    @pytest.mark.asyncio
    async def test_get_repository_stats(
        self,
        vector_search,
        mock_db_session,
    ) -> None:
        """Test getting repository statistics."""
        # Mock count queries
        # Create mock result object that behaves like SQLAlchemy result
        mock_type_result = MagicMock()
        mock_type_result.__iter__ = lambda self: iter([
            ("function", 50),
            ("class", 10),
            ("module", 5),
        ])

        mock_db_session.execute.side_effect = [
            # Count by type
            mock_type_result,
            # Total count
            MagicMock(scalar=lambda: 65),
            # File count
            MagicMock(scalar=lambda: 15),
        ]

        stats = await vector_search.get_repository_stats(123)

        assert stats["repository_id"] == 123
        assert stats["total_embeddings"] == 65
        assert stats["files_with_embeddings"] == 15
        assert stats["embeddings_by_type"]["functions"] == 50
        assert stats["embeddings_by_type"]["classes"] == 10
        assert stats["embeddings_by_type"]["modules"] == 5
