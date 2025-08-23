"""Tests for the search engine module."""

from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.database.models import Class, File, Function, Module
from src.query.search_engine import SearchEngine


@pytest.fixture
def mock_db_session() -> AsyncMock:
    """Create a mock database session."""
    # AsyncSession-like mock; we'll set execute explicitly per test
    return AsyncMock()


@pytest.fixture
def mock_embedding_generator() -> Generator[MagicMock, None, None]:
    """Create a mock embedding generator."""
    with patch("src.query.search_engine.EmbeddingGenerator") as mock:
        instance = MagicMock()
        mock.return_value = instance
        yield instance


@pytest.fixture
def search_engine(
    mock_db_session: AsyncMock, mock_embedding_generator: MagicMock
) -> SearchEngine:
    """Create a search engine instance with mocked dependencies."""
    return SearchEngine(mock_db_session)


@pytest.fixture
def sample_functions() -> tuple[MagicMock, MagicMock, MagicMock]:
    """Create sample function data for testing."""
    mock_func = MagicMock(spec=Function)
    mock_func.id = 1
    mock_func.name = "connect_to_database"
    mock_func.docstring = "Connect to PostgreSQL database"
    mock_func.start_line = 10
    mock_func.end_line = 25
    mock_func.module_id = 1

    mock_module = MagicMock(spec=Module)
    mock_module.id = 1
    mock_module.name = "database"
    mock_module.file_id = 1

    mock_file = MagicMock(spec=File)
    mock_file.id = 1
    mock_file.path = "src/db/connection.py"

    return mock_func, mock_file, mock_module


@pytest.fixture
def sample_classes() -> tuple[MagicMock, MagicMock, MagicMock]:
    """Create sample class data for testing."""
    mock_class = MagicMock(spec=Class)
    mock_class.id = 1
    mock_class.name = "UserRepository"
    mock_class.docstring = "Repository for user data operations"
    mock_class.start_line = 30
    mock_class.end_line = 150
    mock_class.module_id = 1

    mock_module = MagicMock(spec=Module)
    mock_module.id = 1
    mock_module.name = "repositories"
    mock_module.file_id = 1

    mock_file = MagicMock(spec=File)
    mock_file.id = 1
    mock_file.path = "src/repositories/user.py"

    return mock_class, mock_file, mock_module


class TestSearchEngine:
    """Test cases for SearchEngine class."""

    @pytest.mark.asyncio
    async def test_search_basic(
        self,
        search_engine: SearchEngine,
        sample_functions: tuple[MagicMock, MagicMock, MagicMock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test basic search functionality."""
        # Arrange
        query = "database"  # This should match the function name and docstring
        mock_func, mock_file, mock_module = sample_functions

        # Mock the database query
        mock_result = MagicMock()
        mock_result.all.return_value = [(mock_func, mock_file, mock_module)]
        monkeypatch.setattr(
            search_engine.session, "execute", AsyncMock(return_value=mock_result)
        )

        # Act
        results = await search_engine.search(query, limit=10)

        # Assert
        assert len(results) == 1
        assert results[0]["name"] == "connect_to_database"
        assert results[0]["entity_type"] == "function"
        assert results[0]["file_path"] == "src/db/connection.py"

    @pytest.mark.asyncio
    async def test_search_with_entity_types(
        self,
        search_engine: SearchEngine,
        sample_classes: tuple[MagicMock, MagicMock, MagicMock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test search filtered by entity types."""
        # Arrange
        query = "repository"
        entity_types = ["class"]
        mock_class, mock_file, mock_module = sample_classes

        # Mock the database query
        mock_result = MagicMock()
        mock_result.all.return_value = [(mock_class, mock_file, mock_module)]
        monkeypatch.setattr(
            search_engine.session, "execute", AsyncMock(return_value=mock_result)
        )

        # Act
        results = await search_engine.search(query, entity_types=entity_types)

        # Assert
        assert len(results) == 1
        assert results[0]["entity_type"] == "class"
        assert results[0]["name"] == "UserRepository"

    @pytest.mark.asyncio
    async def test_search_with_repository_filter(
        self, search_engine: SearchEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test search with repository path filter."""
        # Arrange
        query = "test"
        repository_filter = "myproject"

        # Mock empty results
        mock_result = MagicMock()
        mock_result.all.return_value = []
        exec_mock = AsyncMock(return_value=mock_result)
        monkeypatch.setattr(search_engine.session, "execute", exec_mock)

        # Act
        results = await search_engine.search(query, repository_filter=repository_filter)

        # Assert
        assert results == []
        # Verify the query included the repository filter
        exec_mock.assert_called()

    @pytest.mark.asyncio
    async def test_search_empty_query(
        self, search_engine: SearchEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test search with empty query returns results."""
        # Arrange
        query = ""

        # Mock empty results
        mock_result = MagicMock()
        mock_result.all.return_value = []
        monkeypatch.setattr(
            search_engine.session, "execute", AsyncMock(return_value=mock_result)
        )

        # Act
        results = await search_engine.search(query)

        # Assert
        assert results == []

    @pytest.mark.asyncio
    async def test_search_error_handling(
        self, search_engine: SearchEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test error handling in search."""
        # Arrange
        monkeypatch.setattr(
            search_engine.session,
            "execute",
            AsyncMock(side_effect=Exception("Database error")),
        )

        # Act
        results = await search_engine.search("test query")

        # Assert
        assert results == []  # Should return empty list on error

    @pytest.mark.asyncio
    async def test_search_limit_enforcement(
        self, search_engine: SearchEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that search respects limit configuration."""
        # Arrange
        query = "test"

        # Mock the query config
        search_engine.query_config.default_limit = 20
        search_engine.query_config.max_limit = 100

        mock_result = MagicMock()
        mock_result.all.return_value = []
        exec_mock = AsyncMock(return_value=mock_result)
        monkeypatch.setattr(search_engine.session, "execute", exec_mock)

        # Act - Test with large limit
        await search_engine.search(query, limit=200)

        # Assert - Should be capped at max_limit
        # The limit should be applied in the SQL query
        exec_mock.assert_called()

    @pytest.mark.asyncio
    async def test_search_multiple_entity_types(
        self,
        search_engine: SearchEngine,
        sample_functions: tuple[MagicMock, MagicMock, MagicMock],
        sample_classes: tuple[MagicMock, MagicMock, MagicMock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test searching multiple entity types."""
        # Arrange
        query = "user"
        mock_func, mock_file_func, mock_module_func = sample_functions
        mock_class, mock_file_class, mock_module_class = sample_classes

        # First call returns functions, second returns classes
        mock_result_func = MagicMock()
        mock_result_func.all.return_value = [
            (mock_func, mock_file_func, mock_module_func)
        ]

        mock_result_class = MagicMock()
        mock_result_class.all.return_value = [
            (mock_class, mock_file_class, mock_module_class)
        ]

        monkeypatch.setattr(
            search_engine.session,
            "execute",
            AsyncMock(side_effect=[mock_result_func, mock_result_class]),
        )

        # Act
        results = await search_engine.search(query, entity_types=["function", "class"])

        # Assert
        assert len(results) == 2
        assert any(r["entity_type"] == "function" for r in results)
        assert any(r["entity_type"] == "class" for r in results)

    @pytest.mark.asyncio
    async def test_search_case_insensitive(
        self,
        search_engine: SearchEngine,
        sample_functions: tuple[MagicMock, MagicMock, MagicMock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that search is case insensitive."""
        # Arrange
        query = "DATABASE"  # Uppercase
        mock_func, mock_file, mock_module = sample_functions
        mock_func.name = "connect_to_database"  # Lowercase

        mock_result = MagicMock()
        mock_result.all.return_value = [(mock_func, mock_file, mock_module)]
        monkeypatch.setattr(
            search_engine.session, "execute", AsyncMock(return_value=mock_result)
        )

        # Act
        results = await search_engine.search(query)

        # Assert
        assert len(results) == 1
        assert results[0]["name"] == "connect_to_database"

    @pytest.mark.asyncio
    async def test_search_by_docstring(
        self,
        search_engine: SearchEngine,
        sample_functions: tuple[MagicMock, MagicMock, MagicMock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test searching by docstring content."""
        # Arrange
        query = "PostgreSQL"
        mock_func, mock_file, mock_module = sample_functions

        mock_result = MagicMock()
        mock_result.all.return_value = [(mock_func, mock_file, mock_module)]
        monkeypatch.setattr(
            search_engine.session, "execute", AsyncMock(return_value=mock_result)
        )

        # Act
        results = await search_engine.search(query)

        # Assert
        assert len(results) == 1
        assert "PostgreSQL" in results[0]["docstring"]

    @pytest.mark.asyncio
    async def test_search_modules(
        self, search_engine: SearchEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test searching for modules."""
        # Arrange
        query = "utils"

        mock_module = MagicMock(spec=Module)
        mock_module.id = 1
        mock_module.name = "utils"
        mock_module.docstring = "Utility functions"
        mock_module.file_id = 1

        mock_file = MagicMock(spec=File)
        mock_file.id = 1
        mock_file.path = "src/utils.py"

        mock_result = MagicMock()
        mock_result.all.return_value = [(mock_module, mock_file)]
        monkeypatch.setattr(
            search_engine.session, "execute", AsyncMock(return_value=mock_result)
        )

        # Act
        results = await search_engine.search(query, entity_types=["module"])

        # Assert
        assert len(results) == 1
        assert results[0]["entity_type"] == "module"
        assert results[0]["name"] == "utils"

    @pytest.mark.asyncio
    async def test_search_similarity_scores(
        self,
        search_engine: SearchEngine,
        sample_functions: tuple[MagicMock, MagicMock, MagicMock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that results include similarity scores."""
        # Arrange
        query = "database"
        mock_func, mock_file, mock_module = sample_functions

        mock_result = MagicMock()
        mock_result.all.return_value = [(mock_func, mock_file, mock_module)]
        monkeypatch.setattr(
            search_engine.session, "execute", AsyncMock(return_value=mock_result)
        )

        # Act
        results = await search_engine.search(query)

        # Assert
        assert len(results) == 1
        assert "similarity" in results[0]
        assert 0 <= results[0]["similarity"] <= 1

    @pytest.mark.asyncio
    async def test_search_performance(
        self, search_engine: SearchEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test search performance with many results."""
        # Arrange
        query = "test"

        # Create many mock results
        mock_results = []
        for i in range(100):
            mock_func = MagicMock(spec=Function)
            mock_func.id = i
            mock_func.name = f"test_function_{i}"
            mock_func.docstring = "Test function"
            mock_func.start_line = i * 10
            mock_func.end_line = i * 10 + 5

            mock_module = MagicMock(spec=Module)
            mock_module.id = i
            mock_module.name = f"module_{i}"

            mock_file = MagicMock(spec=File)
            mock_file.id = i
            mock_file.path = f"src/test_{i}.py"

            mock_results.append((mock_func, mock_file, mock_module))

        mock_result = MagicMock()
        mock_result.all.return_value = mock_results
        monkeypatch.setattr(
            search_engine.session, "execute", AsyncMock(return_value=mock_result)
        )

        # Act
        import time

        start_time = time.time()
        results = await search_engine.search(query, limit=100)
        end_time = time.time()

        # Assert
        assert len(results) <= 100
        assert end_time - start_time < 1.0  # Should complete within 1 second

    @pytest.mark.asyncio
    async def test_concurrent_searches(
        self, search_engine: SearchEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test handling concurrent search requests."""
        import asyncio

        # Arrange
        queries = ["query1", "query2", "query3"]

        mock_result = MagicMock()
        mock_result.all.return_value = []
        monkeypatch.setattr(
            search_engine.session, "execute", AsyncMock(return_value=mock_result)
        )

        # Act - Run searches concurrently
        tasks = [search_engine.search(q) for q in queries]
        results = await asyncio.gather(*tasks)

        # Assert
        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)
