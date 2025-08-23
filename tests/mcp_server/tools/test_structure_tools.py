"""Tests for code structure analysis tools."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastmcp import FastMCP
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import Class, File, Function, Module
from src.mcp_server.tools.code_analysis import CodeAnalysisTools


@pytest.fixture
def mock_db_session() -> AsyncSession:
    """Create mock database session."""
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
def mock_mcp() -> FastMCP:
    """Create mock FastMCP instance."""
    mcp = MagicMock(spec=FastMCP)
    mcp.tool = MagicMock(side_effect=lambda **kwargs: lambda func: func)
    return mcp


@pytest.fixture
def analysis_tools(
    mock_db_session: AsyncSession, mock_mcp: FastMCP
) -> CodeAnalysisTools:
    """Create code analysis tools fixture."""
    return CodeAnalysisTools(mock_db_session, mock_mcp)


# Helper builders to reduce statements in tests


def build_file_module_and_results() -> tuple[MagicMock, MagicMock]:
    mock_file = MagicMock(spec=File)
    mock_file.id = 1
    mock_file.path = "/src/models/user.py"
    mock_file.language = "python"
    mock_file.size = 5000
    mock_file.lines = 200

    file_result = MagicMock()
    file_result.scalar_one_or_none.return_value = mock_file

    mock_module = MagicMock(spec=Module)
    mock_module.name = "src.models.user"
    mock_module.docstring = "User model module"

    module_result = MagicMock()
    module_result.scalar_one_or_none.return_value = mock_module

    return file_result, module_result


def build_classes_and_result() -> MagicMock:
    classes = []
    for i, (name, docstring, methods) in enumerate(
        [
            ("User", "Main user model", 15),
            ("UserProfile", "User profile extension", 8),
            ("UserPermissions", "User permissions", 5),
        ]
    ):
        cls = MagicMock(spec=Class)
        cls.name = name
        cls.docstring = docstring
        cls.method_count = methods
        cls.start_line = 20 + i * 50
        cls.end_line = cls.start_line + 40
        cls.parent_id = None
        cls.is_abstract = i == 2
        classes.append(cls)

    classes_result = MagicMock()
    classes_result.scalars.return_value.all.return_value = classes
    return classes_result


def build_functions_and_result() -> MagicMock:
    functions = []
    func_data = [
        ("create_user", "Create new user", None, 10),
        ("validate_email", "Validate email format", None, 15),
        ("get_user_by_id", "Retrieve user by ID", None, 180),
        ("__init__", "Initialize user", "User", 25),
        ("save", "Save user to database", "User", 35),
        ("delete", "Delete user", "User", 45),
    ]

    for name, docstring, parent_class, line in func_data:
        func = MagicMock(spec=Function)
        func.name = name
        func.docstring = docstring
        func.start_line = line
        func.end_line = line + 8
        func.is_method = parent_class is not None
        func.is_async = name == "save"
        func.complexity_score = 5 if name != "validate_email" else 12
        func.parent_class = parent_class
        functions.append(func)

    functions_result = MagicMock()
    functions_result.scalars.return_value.all.return_value = functions
    return functions_result


class TestStructureTools:
    """Tests for code structure analysis tools."""

    @pytest.mark.asyncio
    async def test_get_code_structure_file_not_found(
        self,
        analysis_tools: CodeAnalysisTools,
        mock_db_session: AsyncSession,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test getting structure for non-existent file."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        monkeypatch.setattr(
            mock_db_session,
            "execute",
            AsyncMock(return_value=mock_result),
        )

        result = await analysis_tools.get_code_structure("nonexistent.py")

        assert result["error"] == "File not found: nonexistent.py"

    @pytest.mark.asyncio
    async def test_get_code_structure_complete(
        self,
        analysis_tools: CodeAnalysisTools,
        mock_db_session: AsyncSession,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test getting complete code structure for a file."""
        # Mock file, module, classes, and functions via helpers
        file_result, module_result = build_file_module_and_results()
        classes_result = build_classes_and_result()
        functions_result = build_functions_and_result()

        monkeypatch.setattr(
            mock_db_session,
            "execute",
            AsyncMock(
                side_effect=[
                    file_result,
                    module_result,
                    classes_result,
                    functions_result,
                ]
            ),
        )

        result = await analysis_tools.get_code_structure("/src/models/user.py")

        assert result["file"] == "/src/models/user.py"
        assert result["language"] == "python"
        assert result["size"] == 5000
        assert result["lines"] == 200

        # Check module info
        assert result["module"]["name"] == "src.models.user"
        assert result["module"]["docstring"] == "User model module"

        # Check classes
        assert len(result["classes"]) == 3
        assert result["classes"][0]["name"] == "User"
        assert result["classes"][0]["method_count"] == 15
        assert result["classes"][2]["is_abstract"] is True

        # Check functions (only module-level)
        module_functions = [f for f in result["functions"] if not f["is_method"]]
        assert len(module_functions) == 3
        assert any(f["name"] == "create_user" for f in module_functions)

        # Check methods are associated with classes
        assert any(f["is_method"] for f in result["functions"])

    @pytest.mark.asyncio
    async def test_get_code_structure_nested_classes(
        self,
        analysis_tools: CodeAnalysisTools,
        mock_db_session: AsyncSession,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test getting code structure with nested classes."""
        # Mock file
        mock_file = MagicMock(spec=File)
        mock_file.id = 1
        mock_file.path = "/src/patterns/builder.py"

        file_result = MagicMock()
        file_result.scalar_one_or_none.return_value = mock_file

        # Mock module
        module_result = MagicMock()
        module_result.scalar_one_or_none.return_value = None

        # Mock classes with nesting
        outer_class = MagicMock(spec=Class)
        outer_class.id = 1
        outer_class.name = "CarBuilder"
        outer_class.parent_id = None
        outer_class.method_count = 5
        outer_class.start_line = 10
        outer_class.is_abstract = False

        inner_class1 = MagicMock(spec=Class)
        inner_class1.id = 2
        inner_class1.name = "EngineBuilder"
        inner_class1.parent_id = 1  # Nested in CarBuilder
        inner_class1.method_count = 3
        inner_class1.start_line = 30
        inner_class1.is_abstract = False

        inner_class2 = MagicMock(spec=Class)
        inner_class2.id = 3
        inner_class2.name = "WheelBuilder"
        inner_class2.parent_id = 1  # Also nested in CarBuilder
        inner_class2.method_count = 2
        inner_class2.start_line = 50
        inner_class2.is_abstract = False

        classes_result = MagicMock()
        classes_result.scalars.return_value.all.return_value = [
            outer_class,
            inner_class1,
            inner_class2,
        ]

        functions_result = MagicMock()
        functions_result.scalars.return_value.all.return_value = []

        monkeypatch.setattr(
            mock_db_session,
            "execute",
            AsyncMock(
                side_effect=[
                    file_result,
                    module_result,
                    classes_result,
                    functions_result,
                ]
            ),
        )

        result = await analysis_tools.get_code_structure("/src/patterns/builder.py")

        # Check nested structure
        assert len(result["classes"]) == 3
        outer = next(c for c in result["classes"] if c["name"] == "CarBuilder")
        assert outer["parent_id"] is None

        nested = [c for c in result["classes"] if c["parent_id"] == 1]
        assert len(nested) == 2
        assert all(c["name"] in ["EngineBuilder", "WheelBuilder"] for c in nested)

    @pytest.mark.asyncio
    async def test_get_module_structure(
        self,
        analysis_tools: CodeAnalysisTools,
        mock_db_session: AsyncSession,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test getting structure for an entire module."""
        # Mock module
        mock_module = MagicMock(spec=Module)
        mock_module.id = 1
        mock_module.name = "src.services"
        mock_module.docstring = "Service layer module"
        mock_module.file_id = 10

        module_result = MagicMock()
        module_result.scalar_one_or_none.return_value = mock_module

        # Mock files in module
        files = []
        for i, name in enumerate(["__init__.py", "user_service.py", "auth_service.py"]):
            f = MagicMock(spec=File)
            f.id = 10 + i
            f.path = f"/src/services/{name}"
            f.language = "python"
            f.size = 1000 * (i + 1)
            f.lines = 50 * (i + 1)
            files.append(f)

        files_result = MagicMock()
        files_result.scalars.return_value.all.return_value = files

        # Mock submodules
        submodules = []
        for _i, name in enumerate(["validators", "handlers"]):
            sub = MagicMock(spec=Module)
            sub.name = f"src.services.{name}"
            sub.docstring = f"{name} submodule"
            submodules.append(sub)

        submodules_result = MagicMock()
        submodules_result.scalars.return_value.all.return_value = submodules

        # Mock aggregate stats
        stats_result = MagicMock()
        stats_result.one.return_value = (10, 50, 200)  # classes, functions, total_lines

        monkeypatch.setattr(
            mock_db_session,
            "execute",
            AsyncMock(
                side_effect=[
                    module_result,
                    files_result,
                    submodules_result,
                    stats_result,
                ]
            ),
        )

        result = await analysis_tools.get_module_structure("src.services")

        assert result["module"]["name"] == "src.services"
        assert result["module"]["docstring"] == "Service layer module"

        # Check files
        assert len(result["files"]) == 3
        assert any(f["path"].endswith("__init__.py") for f in result["files"])

        # Check submodules
        assert len(result["submodules"]) == 2
        assert all(
            sub["name"].startswith("src.services.") for sub in result["submodules"]
        )

        # Check stats
        assert result["stats"]["total_classes"] == 10
        assert result["stats"]["total_functions"] == 50
        assert result["stats"]["total_lines"] == 200

    @pytest.mark.asyncio
    async def test_analyze_file_structure_complexity(
        self,
        analysis_tools: CodeAnalysisTools,
        mock_db_session: AsyncSession,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test analyzing file structure with complexity metrics."""
        # Mock file
        mock_file = MagicMock(spec=File)
        mock_file.id = 1
        mock_file.path = "/src/complex_module.py"
        mock_file.lines = 500

        file_result = MagicMock()
        file_result.scalar_one_or_none.return_value = mock_file

        # Mock functions with varying complexity
        functions = []
        complexity_data = [
            ("simple_function", 2),
            ("moderate_function", 8),
            ("complex_function", 15),
            ("very_complex_function", 25),
            ("extremely_complex_function", 40),
        ]

        for name, complexity in complexity_data:
            func = MagicMock(spec=Function)
            func.name = name
            func.complexity_score = complexity
            func.start_line = 10
            func.end_line = 10 + complexity * 5
            func.is_method = False
            functions.append(func)

        functions_result = MagicMock()
        functions_result.scalars.return_value.all.return_value = functions

        # Mock classes
        classes_result = MagicMock()
        classes_result.scalars.return_value.all.return_value = []

        # Mock module
        module_result = MagicMock()
        module_result.scalar_one_or_none.return_value = None

        monkeypatch.setattr(
            mock_db_session,
            "execute",
            AsyncMock(
                side_effect=[
                    file_result,
                    module_result,
                    classes_result,
                    functions_result,
                ]
            ),
        )

        result = await analysis_tools.analyze_file_complexity("/src/complex_module.py")

        assert result["file"] == "/src/complex_module.py"
        assert result["total_functions"] == 5

        # Check complexity metrics
        metrics = result["complexity_metrics"]
        assert metrics["avg_complexity"] == pytest.approx(18.0, 0.1)
        assert metrics["max_complexity"] == 40
        assert metrics["high_complexity_functions"] == 2  # Functions with score > 20

        # Check function breakdown
        assert len(result["functions_by_complexity"]["simple"]) == 1  # <= 5
        assert len(result["functions_by_complexity"]["moderate"]) == 1  # 6-10
        assert len(result["functions_by_complexity"]["complex"]) == 1  # 11-20
        assert len(result["functions_by_complexity"]["very_complex"]) == 2  # > 20

    @pytest.mark.asyncio
    async def test_get_file_dependencies_structure(
        self,
        analysis_tools: CodeAnalysisTools,
        mock_db_session: AsyncSession,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test getting file dependencies as a structure."""
        # Mock file
        mock_file = MagicMock(spec=File)
        mock_file.id = 1
        mock_file.path = "/src/api/handler.py"

        file_result = MagicMock()
        file_result.scalar_one_or_none.return_value = mock_file

        # Mock imports with structure
        from src.database.models import Import

        imports = []
        import_structure = [
            ("fastapi", ["FastAPI", "Request", "Response"], False, 0),
            ("src.models", ["User", "Product"], False, 0),
            ("src.services.user_service", ["UserService"], False, 0),
            (".", ["validators"], True, 1),
            ("..", ["common"], True, 2),
        ]

        for module, items, is_relative, level in import_structure:
            imp = MagicMock(spec=Import)
            imp.module_name = module
            imp.imported_names = ", ".join(items) if items else None
            imp.is_stdlib = False
            imp.is_local = module.startswith("src") or is_relative
            imp.is_relative = is_relative
            imp.level = level
            imports.append(imp)

        imports_result = MagicMock()
        imports_result.scalars.return_value.all.return_value = imports

        monkeypatch.setattr(
            mock_db_session,
            "execute",
            AsyncMock(side_effect=[file_result, imports_result]),
        )

        result = await analysis_tools.get_file_dependencies_structure(
            "/src/api/handler.py"
        )

        assert result["file"] == "/src/api/handler.py"

        # Check import structure
        assert len(result["imports"]["external"]) == 1
        assert result["imports"]["external"][0]["module"] == "fastapi"
        assert "FastAPI" in result["imports"]["external"][0]["items"]

        assert len(result["imports"]["internal"]) == 2
        assert any(
            imp["module"] == "src.models" for imp in result["imports"]["internal"]
        )

        assert len(result["imports"]["relative"]) == 2
        assert any(imp["level"] == 1 for imp in result["imports"]["relative"])
        assert any(imp["level"] == 2 for imp in result["imports"]["relative"])

        # Check dependency graph structure
        assert "dependency_graph" in result
        assert result["dependency_graph"]["node"] == "/src/api/handler.py"
        assert len(result["dependency_graph"]["depends_on"]) == 5

    @pytest.mark.asyncio
    async def test_get_project_structure_overview(
        self,
        analysis_tools: CodeAnalysisTools,
        mock_db_session: AsyncSession,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test getting high-level project structure overview."""
        # Mock repository stats
        stats_result = MagicMock()
        stats_result.one.return_value = (
            100,  # total_files
            50,  # total_modules
            500,  # total_functions
            100,  # total_classes
            50000,  # total_lines
        )

        # Mock language distribution
        lang_result = MagicMock()
        lang_result.__iter__ = MagicMock(
            return_value=iter(
                [
                    ("python", 80),
                    ("javascript", 15),
                    ("yaml", 5),
                ]
            )
        )

        # Mock top-level packages
        packages_result = MagicMock()
        packages_result.__iter__ = MagicMock(
            return_value=iter(
                [
                    ("src", 40),
                    ("tests", 25),
                    ("scripts", 10),
                    ("docs", 5),
                ]
            )
        )

        # Mock complexity distribution
        complexity_result = MagicMock()
        complexity_result.__iter__ = MagicMock(
            return_value=iter(
                [
                    ("simple", 300),  # complexity <= 5
                    ("moderate", 150),  # 6-10
                    ("complex", 40),  # 11-20
                    ("very_complex", 10),  # > 20
                ]
            )
        )

        monkeypatch.setattr(
            mock_db_session,
            "execute",
            AsyncMock(
                side_effect=[
                    stats_result,
                    lang_result,
                    packages_result,
                    complexity_result,
                ]
            ),
        )

        result = await analysis_tools.get_project_structure_overview(repository_id=1)

        assert result["repository_id"] == 1

        # Check stats
        stats = result["stats"]
        assert stats["total_files"] == 100
        assert stats["total_modules"] == 50
        assert stats["total_functions"] == 500
        assert stats["total_classes"] == 100
        assert stats["total_lines"] == 50000

        # Check language distribution
        assert len(result["languages"]) == 3
        assert result["languages"][0]["language"] == "python"
        assert result["languages"][0]["file_count"] == 80

        # Check package structure
        assert len(result["top_level_packages"]) == 4
        assert result["top_level_packages"][0]["name"] == "src"
        assert result["top_level_packages"][0]["file_count"] == 40

        # Check complexity distribution
        assert result["complexity_distribution"]["simple"] == 300
        assert result["complexity_distribution"]["very_complex"] == 10
