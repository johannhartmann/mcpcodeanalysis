"""Tests for code similarity tools."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import FastMCP
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import Class, File, Function
from src.mcp_server.tools.code_search import CodeSearchTools


@pytest.fixture
def mock_db_session():
    """Create mock database session."""
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
def mock_mcp():
    """Create mock FastMCP instance."""
    mcp = MagicMock(spec=FastMCP)
    mcp.tool = MagicMock(side_effect=lambda **kwargs: lambda func: func)
    return mcp


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings."""
    with patch("langchain_openai.OpenAIEmbeddings") as mock_class:
        mock_instance = MagicMock()
        mock_instance.aembed_query = AsyncMock(return_value=[0.1] * 1536)
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def search_tools(mock_db_session, mock_mcp, mock_embeddings):
    """Create code search tools fixture."""
    with (
        patch("src.embeddings.vector_search.settings") as mock_vector_settings,
        patch("src.embeddings.domain_search.settings") as mock_domain_settings,
        patch("src.embeddings.vector_search.OpenAIEmbeddings") as mock_openai_class,
        patch(
            "src.embeddings.domain_search.OpenAIEmbeddings"
        ) as mock_domain_openai_class,
        patch("src.embeddings.domain_search.ChatOpenAI") as mock_chat_openai,
    ):
        # Configure settings
        mock_vector_settings.openai_api_key.get_secret_value.return_value = "test-key"
        mock_vector_settings.embeddings.model = "text-embedding-ada-002"
        mock_domain_settings.openai_api_key.get_secret_value.return_value = "test-key"
        mock_domain_settings.embeddings.model = "text-embedding-ada-002"
        mock_domain_settings.llm.model = "gpt-3.5-turbo"
        mock_domain_settings.llm.temperature = 0.0

        # Use the mock embeddings fixture
        mock_openai_class.return_value = mock_embeddings
        mock_domain_openai_class.return_value = mock_embeddings
        mock_chat_openai.return_value = MagicMock()

        return CodeSearchTools(mock_db_session, mock_mcp)


class TestSimilarityTools:
    """Tests for code similarity analysis tools."""

    @pytest.mark.asyncio
    async def test_find_similar_code_by_function(self, search_tools, mock_db_session):
        """Test finding similar code by function ID."""
        # Mock the target function
        mock_function = MagicMock(spec=Function)
        mock_function.id = 10
        mock_function.name = "process_data"
        mock_function.file_id = 1
        mock_function.start_line = 50
        mock_function.end_line = 100

        func_result = MagicMock()
        func_result.scalar_one_or_none.return_value = mock_function

        # Mock file for the function
        mock_file = MagicMock(spec=File)
        mock_file.path = "/src/processor.py"

        file_result = MagicMock()
        file_result.scalar_one_or_none.return_value = mock_file

        # Mock vector search results
        similar_results = [
            {
                "similarity": 0.95,
                "entity_type": "function",
                "entity": {
                    "id": 20,
                    "name": "handle_data",
                    "file_id": 2,
                    "docstring": "Similar data processing function",
                },
                "file": {"path": "/src/handler.py"},
                "chunk": {"start_line": 100, "end_line": 150},
            },
            {
                "similarity": 0.88,
                "entity_type": "function",
                "entity": {
                    "id": 30,
                    "name": "transform_data",
                    "file_id": 3,
                    "docstring": "Transform data with similar logic",
                },
                "file": {"path": "/src/transformer.py"},
                "chunk": {"start_line": 200, "end_line": 250},
            },
        ]

        search_tools.vector_search = MagicMock()
        search_tools.vector_search.find_similar_by_entity = AsyncMock(
            return_value=similar_results
        )

        mock_db_session.execute.side_effect = [func_result, file_result]

        # Register tools and test
        await search_tools.register_tools()

        # Mock the function call
        result = await search_tools.find_similar_code(
            entity_type="function", entity_id=10, limit=5
        )

        assert len(result) == 2
        assert result[0]["name"] == "handle_data"
        assert result[0]["similarity"] == 0.95
        assert result[0]["file"] == "/src/handler.py"
        assert result[1]["name"] == "transform_data"

    @pytest.mark.asyncio
    async def test_find_similar_code_by_class(self, search_tools, mock_db_session):
        """Test finding similar code by class ID."""
        # Mock the target class
        mock_class = MagicMock(spec=Class)
        mock_class.id = 5
        mock_class.name = "DataProcessor"
        mock_class.file_id = 1

        class_result = MagicMock()
        class_result.scalar_one_or_none.return_value = mock_class

        # Mock file for the class
        mock_file = MagicMock(spec=File)
        mock_file.path = "/src/models/processor.py"

        file_result = MagicMock()
        file_result.scalar_one_or_none.return_value = mock_file

        # Mock similar classes
        similar_results = [
            {
                "similarity": 0.92,
                "entity_type": "class",
                "entity": {
                    "id": 15,
                    "name": "DataHandler",
                    "file_id": 10,
                    "docstring": "Handles data processing",
                    "method_count": 8,
                },
                "file": {"path": "/src/models/handler.py"},
                "chunk": {"start_line": 10, "end_line": 200},
            },
            {
                "similarity": 0.85,
                "entity_type": "class",
                "entity": {
                    "id": 25,
                    "name": "StreamProcessor",
                    "file_id": 20,
                    "docstring": "Process data streams",
                    "method_count": 12,
                },
                "file": {"path": "/src/streaming/processor.py"},
                "chunk": {"start_line": 50, "end_line": 300},
            },
        ]

        search_tools.vector_search = MagicMock()
        search_tools.vector_search.find_similar_by_entity = AsyncMock(
            return_value=similar_results
        )

        mock_db_session.execute.side_effect = [class_result, file_result]

        await search_tools.register_tools()

        result = await search_tools.find_similar_code(
            entity_type="class", entity_id=5, limit=10
        )

        assert len(result) == 2
        assert result[0]["name"] == "DataHandler"
        assert result[0]["type"] == "class"
        assert result[0]["similarity"] == 0.92
        assert "method_count" in result[0]

    @pytest.mark.asyncio
    async def test_find_similar_code_entity_not_found(
        self, search_tools, mock_db_session
    ):
        """Test finding similar code when entity doesn't exist."""
        # Mock entity not found
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None

        mock_db_session.execute.return_value = mock_result

        await search_tools.register_tools()

        result = await search_tools.find_similar_code(
            entity_type="function", entity_id=999
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_find_similar_code_exclude_same_file(
        self, search_tools, mock_db_session
    ):
        """Test finding similar code excluding results from same file."""
        # Mock the target function
        mock_function = MagicMock(spec=Function)
        mock_function.id = 10
        mock_function.name = "validate_input"
        mock_function.file_id = 5

        func_result = MagicMock()
        func_result.scalar_one_or_none.return_value = mock_function

        # Mock file
        mock_file = MagicMock(spec=File)
        mock_file.path = "/src/validators.py"

        file_result = MagicMock()
        file_result.scalar_one_or_none.return_value = mock_file

        # Mock results including same file and different files
        similar_results = [
            {
                "similarity": 0.98,
                "entity_type": "function",
                "entity": {
                    "id": 11,
                    "name": "validate_output",
                    "file_id": 5,  # Same file!
                },
                "file": {"path": "/src/validators.py"},
                "chunk": {"start_line": 150, "end_line": 200},
            },
            {
                "similarity": 0.90,
                "entity_type": "function",
                "entity": {
                    "id": 20,
                    "name": "check_input",
                    "file_id": 10,  # Different file
                },
                "file": {"path": "/src/checkers.py"},
                "chunk": {"start_line": 50, "end_line": 80},
            },
            {
                "similarity": 0.88,
                "entity_type": "function",
                "entity": {
                    "id": 12,
                    "name": "validate_params",
                    "file_id": 5,  # Same file!
                },
                "file": {"path": "/src/validators.py"},
                "chunk": {"start_line": 250, "end_line": 300},
            },
        ]

        search_tools.vector_search = MagicMock()
        search_tools.vector_search.find_similar_by_entity = AsyncMock(
            return_value=similar_results
        )

        mock_db_session.execute.side_effect = [func_result, file_result]

        await search_tools.register_tools()

        # Test with exclude_same_file=True (default)
        result = await search_tools.find_similar_code(
            entity_type="function", entity_id=10, limit=5
        )

        # Should only return the one from different file
        assert len(result) == 1
        assert result[0]["name"] == "check_input"
        assert result[0]["file"] == "/src/checkers.py"

    @pytest.mark.asyncio
    async def test_find_similar_code_include_same_file(
        self, search_tools, mock_db_session
    ):
        """Test finding similar code including results from same file."""
        # Mock the target function
        mock_function = MagicMock(spec=Function)
        mock_function.id = 10
        mock_function.name = "process_item"
        mock_function.file_id = 1

        func_result = MagicMock()
        func_result.scalar_one_or_none.return_value = mock_function

        # Mock file
        mock_file = MagicMock(spec=File)
        mock_file.path = "/src/processor.py"

        file_result = MagicMock()
        file_result.scalar_one_or_none.return_value = mock_file

        # Mock results from same file
        similar_results = [
            {
                "similarity": 0.95,
                "entity_type": "function",
                "entity": {
                    "id": 11,
                    "name": "process_batch",
                    "file_id": 1,  # Same file
                },
                "file": {"path": "/src/processor.py"},
                "chunk": {"start_line": 200, "end_line": 250},
            },
            {
                "similarity": 0.90,
                "entity_type": "function",
                "entity": {
                    "id": 12,
                    "name": "process_single",
                    "file_id": 1,  # Same file
                },
                "file": {"path": "/src/processor.py"},
                "chunk": {"start_line": 300, "end_line": 350},
            },
        ]

        search_tools.vector_search = MagicMock()
        search_tools.vector_search.find_similar_by_entity = AsyncMock(
            return_value=similar_results
        )

        mock_db_session.execute.side_effect = [func_result, file_result]

        await search_tools.register_tools()

        # Test with exclude_same_file=False
        result = await search_tools.find_similar_code(
            entity_type="function", entity_id=10, exclude_same_file=False, limit=5
        )

        # Should return both results from same file
        assert len(result) == 2
        assert result[0]["name"] == "process_batch"
        assert result[1]["name"] == "process_single"
        assert all(r["file"] == "/src/processor.py" for r in result)

    @pytest.mark.asyncio
    async def test_find_similar_patterns(self, search_tools):
        """Test finding similar code patterns."""
        # Mock pattern search results
        pattern_results = [
            {
                "pattern_type": "singleton",
                "confidence": 0.95,
                "entity": {
                    "name": "DatabaseManager",
                    "type": "class",
                    "file_id": 10,
                },
                "file": {"path": "/src/db/manager.py"},
                "description": "Singleton pattern implementation",
            },
            {
                "pattern_type": "singleton",
                "confidence": 0.88,
                "entity": {
                    "name": "ConfigManager",
                    "type": "class",
                    "file_id": 20,
                },
                "file": {"path": "/src/config/manager.py"},
                "description": "Another singleton implementation",
            },
            {
                "pattern_type": "factory",
                "confidence": 0.82,
                "entity": {
                    "name": "create_processor",
                    "type": "function",
                    "file_id": 30,
                },
                "file": {"path": "/src/factory.py"},
                "description": "Factory method pattern",
            },
        ]

        search_tools.pattern_analyzer = MagicMock()
        search_tools.pattern_analyzer.find_similar_patterns = AsyncMock(
            return_value=pattern_results
        )

        await search_tools.register_tools()

        # Test pattern search
        result = await search_tools.find_similar_patterns(
            pattern_type="singleton", min_confidence=0.8
        )

        # Should return only singleton patterns above confidence threshold
        assert len(result) == 2
        assert all(r["pattern_type"] == "singleton" for r in result)
        assert result[0]["confidence"] == 0.95
        assert result[0]["entity"]["name"] == "DatabaseManager"

    @pytest.mark.asyncio
    async def test_find_duplicate_code(self, search_tools):
        """Test finding duplicate or near-duplicate code."""
        # Mock duplicate detection results
        duplicate_results = [
            {
                "group_id": 1,
                "similarity": 0.98,
                "instances": [
                    {
                        "function": "validate_email",
                        "file": "/src/validators/email.py",
                        "lines": (10, 30),
                    },
                    {
                        "function": "check_email",
                        "file": "/src/utils/validation.py",
                        "lines": (50, 70),
                    },
                    {
                        "function": "is_valid_email",
                        "file": "/src/helpers.py",
                        "lines": (100, 120),
                    },
                ],
                "code_sample": "def validate_email(email):\n    # Similar validation logic...",
            },
            {
                "group_id": 2,
                "similarity": 0.95,
                "instances": [
                    {
                        "function": "parse_config",
                        "file": "/src/config/parser.py",
                        "lines": (20, 50),
                    },
                    {
                        "function": "load_config",
                        "file": "/src/settings.py",
                        "lines": (30, 60),
                    },
                ],
                "code_sample": "def parse_config(path):\n    # Similar config parsing...",
            },
        ]

        search_tools.duplicate_detector = MagicMock()
        search_tools.duplicate_detector.find_duplicates = AsyncMock(
            return_value=duplicate_results
        )

        await search_tools.register_tools()

        # Test duplicate detection
        result = await search_tools.find_duplicate_code(
            min_similarity=0.9, min_lines=10
        )

        assert len(result) == 2
        assert result[0]["similarity"] == 0.98
        assert len(result[0]["instances"]) == 3
        assert result[1]["similarity"] == 0.95
        assert len(result[1]["instances"]) == 2

    @pytest.mark.asyncio
    async def test_analyze_code_similarity_metrics(self, search_tools):
        """Test analyzing overall code similarity metrics."""
        # Mock similarity metrics
        metrics = {
            "total_functions": 500,
            "total_classes": 100,
            "duplicate_groups": 15,
            "near_duplicate_groups": 25,
            "avg_similarity_in_duplicates": 0.92,
            "most_duplicated": [
                {
                    "pattern": "validation logic",
                    "occurrences": 8,
                    "files": [
                        "/src/validators.py",
                        "/src/utils/validation.py",
                        "/src/api/validators.py",
                    ],
                },
                {
                    "pattern": "error handling",
                    "occurrences": 6,
                    "files": ["/src/errors.py", "/src/handlers.py"],
                },
            ],
            "similarity_distribution": {
                "high": 45,  # > 0.9 similarity
                "medium": 120,  # 0.7-0.9
                "low": 335,  # < 0.7
            },
        }

        search_tools.similarity_analyzer = MagicMock()
        search_tools.similarity_analyzer.get_similarity_metrics = AsyncMock(
            return_value=metrics
        )

        await search_tools.register_tools()

        result = await search_tools.analyze_code_similarity_metrics(repository_id=1)

        assert result["total_functions"] == 500
        assert result["duplicate_groups"] == 15
        assert result["avg_similarity_in_duplicates"] == 0.92
        assert len(result["most_duplicated"]) == 2
        assert result["similarity_distribution"]["high"] == 45

    @pytest.mark.asyncio
    async def test_find_similar_code_with_threshold(
        self, search_tools, mock_db_session
    ):
        """Test finding similar code with similarity threshold."""
        # Mock function
        mock_function = MagicMock(spec=Function)
        mock_function.id = 10
        mock_function.name = "calculate_score"
        mock_function.file_id = 1

        func_result = MagicMock()
        func_result.scalar_one_or_none.return_value = mock_function

        # Mock file
        mock_file = MagicMock(spec=File)
        mock_file.path = "/src/calculator.py"

        file_result = MagicMock()
        file_result.scalar_one_or_none.return_value = mock_file

        # Mock results with varying similarities
        similar_results = [
            {
                "similarity": 0.95,
                "entity_type": "function",
                "entity": {"id": 20, "name": "compute_score", "file_id": 2},
                "file": {"path": "/src/scorer.py"},
                "chunk": {"start_line": 10, "end_line": 50},
            },
            {
                "similarity": 0.75,  # Below threshold
                "entity_type": "function",
                "entity": {"id": 30, "name": "get_rating", "file_id": 3},
                "file": {"path": "/src/rating.py"},
                "chunk": {"start_line": 20, "end_line": 60},
            },
            {
                "similarity": 0.85,
                "entity_type": "function",
                "entity": {"id": 40, "name": "evaluate_score", "file_id": 4},
                "file": {"path": "/src/evaluator.py"},
                "chunk": {"start_line": 30, "end_line": 70},
            },
        ]

        search_tools.vector_search = MagicMock()
        search_tools.vector_search.find_similar_by_entity = AsyncMock(
            return_value=similar_results
        )

        mock_db_session.execute.side_effect = [func_result, file_result]

        await search_tools.register_tools()

        # Test with similarity threshold of 0.8
        result = await search_tools.find_similar_code(
            entity_type="function", entity_id=10, min_similarity=0.8
        )

        # Should only return results with similarity >= 0.8
        assert len(result) == 2
        assert result[0]["similarity"] == 0.95
        assert result[1]["similarity"] == 0.85
        assert all(r["similarity"] >= 0.8 for r in result)
