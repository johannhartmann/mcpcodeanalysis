"""Tests for the symbol finder module."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.query.symbol_finder import (
    SymbolFinder,
    SymbolType,
)


@pytest.fixture
def mock_db_session() -> AsyncMock:
    """Create a mock database session."""
    return AsyncMock()


@pytest.fixture
def symbol_finder(mock_db_session: AsyncMock) -> SymbolFinder:
    """Create a symbol finder instance."""
    return SymbolFinder(mock_db_session)


@pytest.fixture
def sample_symbols() -> dict[str, Any]:
    """Create sample symbols for testing."""
    return {
        "classes": [
            MagicMock(
                id=1,
                name="UserService",
                module_id=1,
                start_line=10,
                end_line=100,
                docstring="Service for user operations",
                base_classes=["BaseService"],
                is_abstract=False,
            ),
            MagicMock(
                id=2,
                name="UserRepository",
                module_id=1,
                start_line=110,
                end_line=200,
                docstring="Repository for user data",
                base_classes=["Repository", "CacheableMixin"],
                is_abstract=False,
            ),
        ],
        "functions": [
            MagicMock(
                id=1,
                name="get_user_by_id",
                module_id=1,
                class_id=None,
                start_line=210,
                end_line=220,
                docstring="Get user by ID",
                parameters=[{"name": "user_id", "type": "int"}],
                return_type="User",
                is_async=True,
            ),
            MagicMock(
                id=2,
                name="create_user",
                module_id=1,
                class_id=1,
                start_line=50,
                end_line=70,
                docstring="Create a new user",
                parameters=[{"name": "data", "type": "dict"}],
                return_type="User",
                is_async=False,
            ),
        ],
        "modules": [
            MagicMock(
                id=1, name="user_service", file_id=1, docstring="User service module"
            )
        ],
    }


class TestSymbolFinder:
    """Test cases for SymbolFinder class."""

    @pytest.mark.asyncio
    async def test_find_symbol_by_exact_name(
        self,
        symbol_finder: SymbolFinder,
        sample_symbols: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test finding symbol by exact name match."""
        # Arrange
        symbol_name = "UserService"

        # Mock the query
        result = AsyncMock()
        result.scalars = MagicMock(
            return_value=MagicMock(
                all=MagicMock(return_value=[sample_symbols["classes"][0]])
            )
        )
        monkeypatch.setattr(
            symbol_finder.db_session, "execute", AsyncMock(return_value=result)
        )

        # Act
        found = await symbol_finder.find_by_name(symbol_name, exact_match=True)

        # Assert
        assert len(found) == 1
        assert found[0].name == "UserService"
        assert found[0].symbol_type == SymbolType.CLASS

    @pytest.mark.asyncio
    async def test_find_symbol_by_partial_name(
        self,
        symbol_finder: SymbolFinder,
        sample_symbols: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test finding symbol by partial name match."""
        # Arrange
        partial_name = "User"

        # Mock the query to return all User-related symbols
        result = AsyncMock()
        result.scalars = MagicMock(
            return_value=MagicMock(
                all=MagicMock(return_value=sample_symbols["classes"])
            )
        )
        monkeypatch.setattr(
            symbol_finder.db_session, "execute", AsyncMock(return_value=result)
        )

        # Act
        found = await symbol_finder.find_by_name(partial_name, exact_match=False)

        # Assert
        assert len(found) == 2
        assert all("User" in symbol.name for symbol in found)

    @pytest.mark.asyncio
    async def test_find_symbol_by_type(
        self,
        symbol_finder: SymbolFinder,
        sample_symbols: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test finding symbols filtered by type."""
        # Arrange
        # Mock class query
        class_result = AsyncMock()
        class_result.scalars = MagicMock(
            return_value=MagicMock(
                all=MagicMock(return_value=sample_symbols["classes"])
            )
        )

        monkeypatch.setattr(
            symbol_finder.db_session, "execute", AsyncMock(return_value=class_result)
        )

        # Act
        found = await symbol_finder.find_by_type(SymbolType.CLASS)

        # Assert
        assert len(found) == 2
        assert all(s.symbol_type == SymbolType.CLASS for s in found)

    @pytest.mark.asyncio
    async def test_find_symbol_in_file(
        self,
        symbol_finder: SymbolFinder,
        sample_symbols: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test finding symbols within a specific file."""
        # Arrange
        file_id = 1

        # Mock the queries
        classes_mock = AsyncMock(return_value=sample_symbols["classes"])
        funcs_mock = AsyncMock(return_value=sample_symbols["functions"])
        monkeypatch.setattr(symbol_finder, "_find_classes_in_file", classes_mock)
        monkeypatch.setattr(symbol_finder, "_find_functions_in_file", funcs_mock)

        # Act
        found = await symbol_finder.find_in_file(file_id)

        # Assert
        assert len(found) == 4  # 2 classes + 2 functions
        classes_mock.assert_called_once_with(file_id)
        funcs_mock.assert_called_once_with(file_id)

    @pytest.mark.asyncio
    async def test_find_symbol_by_signature(
        self,
        symbol_finder: SymbolFinder,
        sample_symbols: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test finding function by signature."""
        # Arrange
        function_name = "create_user"
        param_types = ["dict"]

        # Mock the query
        result = AsyncMock()
        result.scalars = MagicMock(
            return_value=MagicMock(
                all=MagicMock(return_value=[sample_symbols["functions"][1]])
            )
        )
        monkeypatch.setattr(
            symbol_finder.db_session, "execute", AsyncMock(return_value=result)
        )

        # Act
        found = await symbol_finder.find_function_by_signature(
            function_name, param_types=param_types
        )

        # Assert
        assert len(found) == 1
        assert found[0].name == "create_user"
        assert found[0].parameters[0]["type"] == "dict"

    @pytest.mark.asyncio
    async def test_find_subclasses(
        self, symbol_finder: SymbolFinder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test finding all subclasses of a class."""
        # Arrange
        base_class_name = "BaseService"

        # Mock subclasses
        mock_subclasses = [
            MagicMock(id=1, name="UserService", base_classes=["BaseService"]),
            MagicMock(
                id=2, name="ProductService", base_classes=["BaseService", "Loggable"]
            ),
        ]

        result = AsyncMock()
        result.scalars = MagicMock(
            return_value=MagicMock(all=MagicMock(return_value=mock_subclasses))
        )
        monkeypatch.setattr(
            symbol_finder.db_session, "execute", AsyncMock(return_value=result)
        )

        # Act
        subclasses = await symbol_finder.find_subclasses(base_class_name)

        # Assert
        assert len(subclasses) == 2
        assert all(base_class_name in cls.base_classes for cls in subclasses)

    @pytest.mark.asyncio
    async def test_find_implementations(
        self, symbol_finder: SymbolFinder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test finding implementations of an interface/abstract class."""
        # Arrange
        interface_name = "Repository"

        # Mock implementations
        mock_implementations = [
            MagicMock(
                id=1,
                name="UserRepository",
                base_classes=["Repository", "CacheableMixin"],
            ),
            MagicMock(id=2, name="ProductRepository", base_classes=["Repository"]),
        ]

        result = AsyncMock()
        result.scalars = MagicMock(
            return_value=MagicMock(all=MagicMock(return_value=mock_implementations))
        )
        monkeypatch.setattr(
            symbol_finder.db_session, "execute", AsyncMock(return_value=result)
        )

        # Act
        implementations = await symbol_finder.find_implementations(interface_name)

        # Assert
        assert len(implementations) == 2
        assert all(interface_name in impl.base_classes for impl in implementations)

    @pytest.mark.asyncio
    async def test_find_references(
        self, symbol_finder: SymbolFinder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test finding all references to a symbol."""
        # Arrange
        symbol_id = 1
        symbol_type = SymbolType.CLASS

        # Mock references
        mock_references = [
            MagicMock(
                source_file_id=1,
                source_line=50,
                reference_type="import",
                context="from services import UserService",
            ),
            MagicMock(
                source_file_id=2,
                source_line=100,
                reference_type="instantiation",
                context="service = UserService()",
            ),
            MagicMock(
                source_file_id=3,
                source_line=200,
                reference_type="inheritance",
                context="class ExtendedUserService(UserService):",
            ),
        ]

        monkeypatch.setattr(
            symbol_finder,
            "_find_symbol_references",
            AsyncMock(return_value=mock_references),
        )

        # Act
        references = await symbol_finder.find_references(symbol_id, symbol_type)

        # Assert
        assert len(references) == 3
        assert references[0].reference_type == "import"
        assert references[1].reference_type == "instantiation"
        assert references[2].reference_type == "inheritance"

    @pytest.mark.asyncio
    async def test_find_unused_symbols(
        self, symbol_finder: SymbolFinder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test finding symbols that are never referenced."""
        # Arrange
        repository_id = 1

        # Mock unused symbols
        mock_unused = [
            MagicMock(
                id=1,
                name="deprecated_function",
                symbol_type=SymbolType.FUNCTION,
                reference_count=0,
            ),
            MagicMock(
                id=2,
                name="UnusedClass",
                symbol_type=SymbolType.CLASS,
                reference_count=0,
            ),
        ]

        monkeypatch.setattr(
            symbol_finder,
            "_find_unreferenced_symbols",
            AsyncMock(return_value=mock_unused),
        )

        # Act
        unused = await symbol_finder.find_unused_symbols(repository_id)

        # Assert
        assert len(unused) == 2
        assert all(s.reference_count == 0 for s in unused)

    @pytest.mark.asyncio
    async def test_find_overloaded_methods(
        self, symbol_finder: SymbolFinder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test finding overloaded methods in a class."""
        # Arrange
        class_id = 1

        # Mock overloaded methods
        mock_methods = [
            MagicMock(
                id=1,
                name="process",
                class_id=1,
                parameters=[{"name": "data", "type": "str"}],
            ),
            MagicMock(
                id=2,
                name="process",
                class_id=1,
                parameters=[{"name": "data", "type": "dict"}],
            ),
            MagicMock(
                id=3,
                name="process",
                class_id=1,
                parameters=[
                    {"name": "data", "type": "list"},
                    {"name": "options", "type": "dict"},
                ],
            ),
        ]

        result = AsyncMock()
        result.scalars = MagicMock(
            return_value=MagicMock(all=MagicMock(return_value=mock_methods))
        )
        monkeypatch.setattr(
            symbol_finder.db_session, "execute", AsyncMock(return_value=result)
        )

        # Act
        overloaded = await symbol_finder.find_overloaded_methods(class_id)

        # Assert
        assert "process" in overloaded
        assert len(overloaded["process"]) == 3
        assert len(overloaded["process"][2].parameters) == 2

    @pytest.mark.asyncio
    async def test_get_symbol_info(
        self,
        symbol_finder: SymbolFinder,
        sample_symbols: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test getting detailed symbol information."""
        # Arrange
        symbol_id = 1
        symbol_type = SymbolType.CLASS

        # Mock the symbol
        mock_class = sample_symbols["classes"][0]
        monkeypatch.setattr(
            symbol_finder.db_session, "get", AsyncMock(return_value=mock_class)
        )

        # Mock file and module info
        mock_module = MagicMock(file_id=1, name="user_service")
        mock_file = MagicMock(path="src/services/user_service.py", repository_id=1)

        monkeypatch.setattr(
            symbol_finder, "_get_module_info", AsyncMock(return_value=mock_module)
        )
        monkeypatch.setattr(
            symbol_finder, "_get_file_info", AsyncMock(return_value=mock_file)
        )

        # Act
        info = await symbol_finder.get_symbol_info(symbol_id, symbol_type)

        # Assert
        assert info.name == "UserService"
        assert info.symbol_type == SymbolType.CLASS
        assert info.file_path == "src/services/user_service.py"
        assert info.line_range == (10, 100)
        assert info.docstring == "Service for user operations"

    @pytest.mark.asyncio
    async def test_search_symbols_with_regex(
        self,
        symbol_finder: SymbolFinder,
        sample_symbols: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test searching symbols with regex pattern."""
        # Arrange
        pattern = r"^get_.*_by_\w+$"

        # Mock regex search
        matching_functions = [sample_symbols["functions"][0]]  # get_user_by_id

        monkeypatch.setattr(
            symbol_finder,
            "_search_by_regex",
            AsyncMock(return_value=matching_functions),
        )

        # Act
        found = await symbol_finder.search_with_regex(
            pattern, symbol_type=SymbolType.FUNCTION
        )

        # Assert
        assert len(found) == 1
        assert found[0].name == "get_user_by_id"

    @pytest.mark.asyncio
    async def test_find_async_functions(
        self,
        symbol_finder: SymbolFinder,
        sample_symbols: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test finding all async functions."""
        # Arrange
        # Mock async functions
        async_functions = [f for f in sample_symbols["functions"] if f.is_async]

        result = AsyncMock()
        result.scalars = MagicMock(
            return_value=MagicMock(all=MagicMock(return_value=async_functions))
        )
        monkeypatch.setattr(
            symbol_finder.db_session, "execute", AsyncMock(return_value=result)
        )

        # Act
        found = await symbol_finder.find_async_functions()

        # Assert
        assert len(found) == 1
        assert found[0].name == "get_user_by_id"
        assert found[0].is_async is True

    @pytest.mark.asyncio
    async def test_find_abstract_classes(
        self, symbol_finder: SymbolFinder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test finding all abstract classes."""
        # Arrange
        # Mock abstract classes
        mock_abstract = [
            MagicMock(
                id=1, name="BaseRepository", is_abstract=True, base_classes=["ABC"]
            ),
            MagicMock(id=2, name="AbstractService", is_abstract=True, base_classes=[]),
        ]

        result = AsyncMock()
        result.scalars = MagicMock(
            return_value=MagicMock(all=MagicMock(return_value=mock_abstract))
        )
        monkeypatch.setattr(
            symbol_finder.db_session, "execute", AsyncMock(return_value=result)
        )

        # Act
        found = await symbol_finder.find_abstract_classes()

        # Assert
        assert len(found) == 2
        assert all(cls.is_abstract for cls in found)

    @pytest.mark.asyncio
    async def test_symbol_not_found(
        self, symbol_finder: SymbolFinder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test handling when symbol is not found."""
        # Arrange
        result = AsyncMock()
        result.scalars = MagicMock(
            return_value=MagicMock(all=MagicMock(return_value=[]))
        )
        monkeypatch.setattr(
            symbol_finder.db_session, "execute", AsyncMock(return_value=result)
        )

        # Act
        found = await symbol_finder.find_by_name("NonExistentSymbol")

        # Assert
        assert found == []

    @pytest.mark.asyncio
    async def test_find_symbols_performance(
        self, symbol_finder: SymbolFinder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test performance with large result sets."""
        # Arrange
        # Create 1000 mock symbols
        large_symbol_set = [
            MagicMock(
                id=i,
                name=f"Symbol_{i}",
                module_id=i % 10,
                start_line=i * 10,
                end_line=i * 10 + 5,
            )
            for i in range(1000)
        ]

        result = AsyncMock()
        result.scalars = MagicMock(
            return_value=MagicMock(all=MagicMock(return_value=large_symbol_set))
        )
        monkeypatch.setattr(
            symbol_finder.db_session, "execute", AsyncMock(return_value=result)
        )

        # Act
        import time

        start_time = time.time()
        found = await symbol_finder.find_by_name("Symbol", exact_match=False)
        end_time = time.time()

        # Assert
        assert len(found) == 1000
        assert (
            end_time - start_time < 1.0
        )  # Should process 1000 symbols in under 1 second
