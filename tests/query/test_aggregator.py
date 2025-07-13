"""Tests for the aggregator module."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.query.aggregator import AggregationStrategy, CodeAggregator


@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    return AsyncMock()


@pytest.fixture
def aggregator(mock_db_session):
    """Create a code aggregator instance."""
    return CodeAggregator(mock_db_session)


@pytest.fixture
def sample_file_structure():
    """Create a sample file structure for testing."""
    return {
        "file_id": 1,
        "path": "src/services/user_service.py",
        "modules": [
            {
                "id": 1,
                "name": "user_service",
                "classes": [
                    {
                        "id": 1,
                        "name": "UserService",
                        "methods": [
                            {
                                "id": 1,
                                "name": "__init__",
                                "start_line": 10,
                                "end_line": 15,
                                "complexity": 1,
                            },
                            {
                                "id": 2,
                                "name": "get_user",
                                "start_line": 17,
                                "end_line": 25,
                                "complexity": 3,
                            },
                            {
                                "id": 3,
                                "name": "create_user",
                                "start_line": 27,
                                "end_line": 40,
                                "complexity": 5,
                            },
                        ],
                        "start_line": 8,
                        "end_line": 42,
                    },
                    {
                        "id": 2,
                        "name": "UserRepository",
                        "methods": [
                            {
                                "id": 4,
                                "name": "find_by_id",
                                "start_line": 50,
                                "end_line": 55,
                                "complexity": 2,
                            }
                        ],
                        "start_line": 45,
                        "end_line": 60,
                    },
                ],
                "functions": [
                    {
                        "id": 5,
                        "name": "validate_user_data",
                        "start_line": 65,
                        "end_line": 75,
                        "complexity": 4,
                    }
                ],
            }
        ],
    }


class TestCodeAggregator:
    """Test cases for CodeAggregator class."""

    @pytest.mark.asyncio
    async def test_aggregate_file_hierarchy(self, aggregator, sample_file_structure):
        """Test aggregating code into file hierarchy."""
        # Arrange
        file_id = 1
        aggregator.db_session.get = AsyncMock(
            return_value=MagicMock(id=file_id, path=sample_file_structure["path"])
        )

        # Mock the queries
        aggregator._fetch_file_structure = AsyncMock(return_value=sample_file_structure)

        # Act
        result = await aggregator.aggregate_file_hierarchy(file_id)

        # Assert
        assert result["file_id"] == 1
        assert result["path"] == "src/services/user_service.py"
        assert len(result["modules"]) == 1
        assert len(result["modules"][0]["classes"]) == 2
        assert len(result["modules"][0]["classes"][0]["methods"]) == 3

    @pytest.mark.asyncio
    async def test_aggregate_by_complexity(self, aggregator, sample_file_structure):
        """Test aggregating code by complexity levels."""
        # Arrange
        aggregator._fetch_file_structure = AsyncMock(return_value=sample_file_structure)

        # Act
        result = await aggregator.aggregate_by_complexity(
            file_ids=[1], complexity_ranges=[(1, 2), (3, 4), (5, 10)]
        )

        # Assert
        assert len(result) == 3
        assert result["low_complexity"]["range"] == (1, 2)
        assert result["low_complexity"]["count"] == 2  # __init__ and find_by_id
        assert result["medium_complexity"]["range"] == (3, 4)
        assert (
            result["medium_complexity"]["count"] == 2
        )  # get_user and validate_user_data
        assert result["high_complexity"]["range"] == (5, 10)
        assert result["high_complexity"]["count"] == 1  # create_user

    @pytest.mark.asyncio
    async def test_aggregate_class_hierarchy(self, aggregator):
        """Test aggregating class inheritance hierarchy."""
        # Arrange
        class_id = 1

        # Mock class with inheritance
        mock_class = MagicMock(
            id=1,
            name="ConcreteService",
            base_classes=["BaseService", "LoggingMixin"],
            module_id=1,
        )

        mock_base_class = MagicMock(
            id=2, name="BaseService", base_classes=["ABC"], module_id=2
        )

        aggregator.db_session.get = AsyncMock(
            side_effect=lambda model, entity_id: {
                1: mock_class,
                2: mock_base_class,
            }.get(entity_id)
        )

        aggregator._resolve_base_class = AsyncMock(
            side_effect=lambda name: {
                "BaseService": mock_base_class,
                "LoggingMixin": None,  # External class
                "ABC": None,
            }.get(name)
        )

        # Act
        result = await aggregator.aggregate_class_hierarchy(class_id)

        # Assert
        assert result["class_name"] == "ConcreteService"
        assert len(result["inheritance_chain"]) == 2
        assert result["inheritance_chain"][0] == "BaseService"
        assert result["inheritance_chain"][1] == "LoggingMixin"
        assert result["depth"] == 2

    @pytest.mark.asyncio
    async def test_aggregate_by_file_type(self, aggregator):
        """Test aggregating code by file types."""
        # Arrange
        repository_id = 1

        # Mock file statistics
        mock_stats = [
            {"extension": ".py", "count": 150, "total_lines": 25000},
            {"extension": ".js", "count": 80, "total_lines": 15000},
            {"extension": ".java", "count": 50, "total_lines": 20000},
            {"extension": ".md", "count": 30, "total_lines": 2000},
        ]

        aggregator._fetch_file_statistics = AsyncMock(return_value=mock_stats)

        # Act
        result = await aggregator.aggregate_by_file_type(repository_id)

        # Assert
        assert len(result["file_types"]) == 4
        assert result["file_types"][0]["extension"] == ".py"
        assert result["file_types"][0]["percentage"] > 40  # Python is dominant
        assert result["total_files"] == 310
        assert result["total_lines"] == 62000

    @pytest.mark.asyncio
    async def test_aggregate_functions_by_module(self, aggregator):
        """Test aggregating functions grouped by module."""
        # Arrange
        repository_id = 1

        # Mock module data
        mock_modules = [
            {
                "module_name": "auth",
                "functions": ["login", "logout", "verify_token"],
                "function_count": 3,
            },
            {
                "module_name": "database",
                "functions": ["connect", "disconnect", "execute_query"],
                "function_count": 3,
            },
        ]

        aggregator._fetch_module_functions = AsyncMock(return_value=mock_modules)

        # Act
        result = await aggregator.aggregate_functions_by_module(repository_id)

        # Assert
        assert len(result["modules"]) == 2
        assert result["modules"]["auth"]["count"] == 3
        assert "login" in result["modules"]["auth"]["functions"]
        assert result["total_functions"] == 6

    @pytest.mark.asyncio
    async def test_aggregate_code_metrics(self, aggregator):
        """Test aggregating various code metrics."""
        # Arrange
        file_ids = [1, 2, 3]

        # Mock metrics data
        mock_metrics = {
            "total_lines": 5000,
            "code_lines": 3500,
            "comment_lines": 1000,
            "blank_lines": 500,
            "average_complexity": 3.5,
            "max_complexity": 15,
            "total_functions": 120,
            "total_classes": 25,
            "test_coverage": 0.75,
        }

        aggregator._calculate_metrics = AsyncMock(return_value=mock_metrics)

        # Act
        result = await aggregator.aggregate_code_metrics(file_ids)

        # Assert
        assert result["total_lines"] == 5000
        assert result["code_to_comment_ratio"] == 3.5
        assert result["test_coverage_percent"] == 75
        assert result["average_complexity"] == 3.5

    @pytest.mark.asyncio
    async def test_aggregate_by_author(self, aggregator):
        """Test aggregating code contributions by author."""
        # Arrange
        repository_id = 1

        # Mock author data
        mock_contributions = [
            {
                "author": "john.doe@example.com",
                "commits": 150,
                "lines_added": 5000,
                "lines_removed": 2000,
                "files_modified": 80,
            },
            {
                "author": "jane.smith@example.com",
                "commits": 200,
                "lines_added": 8000,
                "lines_removed": 3000,
                "files_modified": 120,
            },
        ]

        aggregator._fetch_author_contributions = AsyncMock(
            return_value=mock_contributions
        )

        # Act
        result = await aggregator.aggregate_by_author(repository_id, limit=10)

        # Assert
        assert len(result["contributors"]) == 2
        assert (
            result["contributors"][0]["author"] == "jane.smith@example.com"
        )  # More commits
        assert result["contributors"][0]["net_lines"] == 5000
        assert result["total_commits"] == 350

    @pytest.mark.asyncio
    async def test_aggregate_imports(self, aggregator):
        """Test aggregating import statements."""
        # Arrange
        file_ids = [1, 2, 3]

        # Mock import data
        mock_imports = [
            {"module": "os", "count": 15, "files": ["file1.py", "file2.py"]},
            {"module": "sys", "count": 10, "files": ["file1.py"]},
            {"module": "pandas", "count": 8, "files": ["analysis.py"]},
            {"module": "numpy", "count": 8, "files": ["analysis.py"]},
        ]

        aggregator._fetch_imports = AsyncMock(return_value=mock_imports)

        # Act
        result = await aggregator.aggregate_imports(file_ids)

        # Assert
        assert len(result["imports"]) == 4
        assert result["imports"][0]["module"] == "os"  # Most used
        assert result["most_common"][0] == ("os", 15)
        assert "pandas" in result["external_dependencies"]
        assert "os" in result["standard_library"]

    @pytest.mark.asyncio
    async def test_aggregate_empty_results(self, aggregator):
        """Test aggregating with no results."""
        # Arrange
        aggregator._fetch_file_structure = AsyncMock(return_value=None)

        # Act
        result = await aggregator.aggregate_file_hierarchy(999)

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_aggregate_with_strategy(self, aggregator):
        """Test using different aggregation strategies."""
        # Arrange
        strategy = AggregationStrategy.HIERARCHICAL
        file_ids = [1, 2]

        # Mock based on strategy
        aggregator.aggregate_file_hierarchy = AsyncMock(
            return_value={"type": "hierarchy"}
        )

        # Act
        result = await aggregator.aggregate(file_ids, strategy=strategy)

        # Assert
        assert result["type"] == "hierarchy"
        aggregator.aggregate_file_hierarchy.assert_called_once()

    @pytest.mark.asyncio
    async def test_aggregate_large_codebase(self, aggregator):
        """Test aggregating metrics for a large codebase."""
        # Arrange
        # Simulate a large codebase with many files
        large_file_ids = list(range(1, 1001))  # 1000 files

        mock_metrics = {
            "total_lines": 250000,
            "code_lines": 180000,
            "total_functions": 5000,
            "total_classes": 800,
            "average_file_size": 250,
        }

        aggregator._calculate_metrics = AsyncMock(return_value=mock_metrics)

        # Act
        result = await aggregator.aggregate_code_metrics(large_file_ids)

        # Assert
        assert result["total_lines"] == 250000
        assert result["average_file_size"] == 250
        assert result["functions_per_class"] == 6.25  # 5000/800

    @pytest.mark.asyncio
    async def test_aggregate_circular_inheritance(self, aggregator):
        """Test handling circular inheritance in class hierarchy."""
        # Arrange
        class_id = 1

        # Create circular inheritance scenario
        visited = set()

        async def mock_resolve_base(name):
            if name in visited:
                return None  # Prevent infinite loop
            visited.add(name)

            if name == "ClassA":
                return MagicMock(name="ClassA", base_classes=["ClassB"])
            if name == "ClassB":
                return MagicMock(name="ClassB", base_classes=["ClassA"])
            return None

        aggregator._resolve_base_class = mock_resolve_base

        mock_class = MagicMock(id=1, name="ClassA", base_classes=["ClassB"])
        aggregator.db_session.get = AsyncMock(return_value=mock_class)

        # Act
        result = await aggregator.aggregate_class_hierarchy(class_id)

        # Assert
        assert result["has_circular_dependency"] is True
        assert len(result["inheritance_chain"]) <= 10  # Should limit depth

    @pytest.mark.asyncio
    async def test_aggregate_performance(self, aggregator):
        """Test aggregation performance with time limits."""
        # Arrange
        import asyncio

        # Simulate slow query
        async def slow_fetch():
            await asyncio.sleep(5)
            return {"data": "slow"}

        aggregator._fetch_file_structure = slow_fetch

        # Act & Assert
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(aggregator.aggregate_file_hierarchy(1), timeout=1.0)
