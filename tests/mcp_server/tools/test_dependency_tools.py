"""Tests for dependency analysis tools."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import FastMCP
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import File, Import, Module
from src.mcp_server.tools.analysis_tools import AnalysisTools


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
def analysis_tools(mock_db_session, mock_mcp):
    """Create analysis tools fixture."""
    with patch("src.mcp_server.tools.analysis_tools.settings") as mock_settings:
        mock_settings.openai_api_key.get_secret_value.return_value = "test-key"
        return AnalysisTools(mock_db_session, mock_mcp)


class TestDependencyTools:
    """Tests for dependency analysis tools."""

    @pytest.mark.asyncio
    async def test_analyze_dependencies_file_not_found(
        self, analysis_tools, mock_db_session
    ):
        """Test analyzing dependencies when file is not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        result = await analysis_tools.analyze_dependencies("nonexistent.py")

        assert result["error"] == "File not found: nonexistent.py"

    @pytest.mark.asyncio
    async def test_analyze_dependencies_with_imports(
        self, analysis_tools, mock_db_session
    ):
        """Test analyzing dependencies with various import types."""
        # Mock file
        mock_file = MagicMock(spec=File)
        mock_file.id = 1
        mock_file.path = "/src/utils/helpers.py"
        mock_file.repository_id = 10

        file_result = MagicMock()
        file_result.scalar_one_or_none.return_value = mock_file

        # Mock imports
        imports = []
        import_data = [
            ("os", None, "os", True, False),
            ("json", None, "json", True, False),
            ("typing", "List, Dict", "typing", True, False),
            ("numpy", None, "np", False, False),
            ("src.models", "User, Product", "src.models", False, True),
            ("src.database", None, "db", False, True),
            ("..common", "utils", "..common", False, True),
        ]

        for _i, (module, names, alias, is_stdlib, is_local) in enumerate(import_data):
            imp = MagicMock(spec=Import)
            imp.module_name = module
            imp.imported_names = names
            imp.alias = alias if alias != module else None
            imp.is_stdlib = is_stdlib
            imp.is_local = is_local
            imports.append(imp)

        imports_result = MagicMock()
        imports_result.scalars.return_value.all.return_value = imports

        # Mock module resolution for local imports
        mock_module1 = MagicMock(spec=Module)
        mock_module1.name = "src.models"
        mock_module1.file_id = 20

        mock_module2 = MagicMock(spec=Module)
        mock_module2.name = "src.database"
        mock_module2.file_id = 30

        module_results = [
            MagicMock(scalar_one_or_none=lambda: mock_module1),
            MagicMock(scalar_one_or_none=lambda: mock_module2),
            MagicMock(scalar_one_or_none=lambda: None),  # ..common not found
        ]

        # Mock files for resolved modules
        mock_file1 = MagicMock(spec=File)
        mock_file1.path = "/src/models.py"

        mock_file2 = MagicMock(spec=File)
        mock_file2.path = "/src/database.py"

        file_results = [
            MagicMock(scalar_one_or_none=lambda: mock_file1),
            MagicMock(scalar_one_or_none=lambda: mock_file2),
        ]

        # Setup mock sequence
        mock_db_session.execute.side_effect = [
            file_result,
            imports_result,
            *module_results,
            *file_results,
        ]

        result = await analysis_tools.analyze_dependencies("/src/utils/helpers.py")

        assert result["file"] == "/src/utils/helpers.py"
        assert result["total_imports"] == 7
        assert result["stdlib_imports"] == 3
        assert result["third_party_imports"] == 1
        assert result["local_imports"] == 3

        # Check categorized imports
        assert len(result["imports"]["stdlib"]) == 3
        assert "os" in result["imports"]["stdlib"]
        assert "typing (List, Dict)" in result["imports"]["stdlib"]

        assert len(result["imports"]["third_party"]) == 1
        assert "numpy as np" in result["imports"]["third_party"]

        assert len(result["imports"]["local"]) == 3

        # Check resolved dependencies
        assert len(result["resolved_dependencies"]) == 2
        assert any(d["module"] == "src.models" for d in result["resolved_dependencies"])
        assert any(
            d["file"] == "/src/database.py" for d in result["resolved_dependencies"]
        )

        # Check unresolved dependencies
        assert len(result["unresolved_dependencies"]) == 1
        assert result["unresolved_dependencies"][0] == "..common (utils)"

    @pytest.mark.asyncio
    async def test_analyze_dependencies_module_only(
        self, analysis_tools, mock_db_session
    ):
        """Test analyzing dependencies for a module file."""
        # Mock module file
        mock_file = MagicMock(spec=File)
        mock_file.id = 1
        mock_file.path = "/src/services/__init__.py"
        mock_file.repository_id = 10

        file_result = MagicMock()
        file_result.scalar_one_or_none.return_value = mock_file

        # Mock module
        mock_module = MagicMock(spec=Module)
        mock_module.name = "src.services"
        mock_module.file_id = 1

        module_result = MagicMock()
        module_result.scalar_one_or_none.return_value = mock_module

        # Mock imports from all files in the module
        module_imports = []
        for _i, (imp_module, is_local) in enumerate(
            [
                ("logging", False),
                ("src.models", True),
                ("src.utils", True),
                ("requests", False),
            ]
        ):
            imp = MagicMock(spec=Import)
            imp.module_name = imp_module
            imp.imported_names = None
            imp.alias = None
            imp.is_stdlib = imp_module == "logging"
            imp.is_local = is_local
            module_imports.append(imp)

        imports_result = MagicMock()
        imports_result.scalars.return_value.all.return_value = module_imports

        # Mock resolved modules
        mock_models = MagicMock(spec=Module)
        mock_models.name = "src.models"
        mock_models.file_id = 20

        mock_utils = MagicMock(spec=Module)
        mock_utils.name = "src.utils"
        mock_utils.file_id = 30

        module_resolutions = [
            MagicMock(scalar_one_or_none=lambda: mock_models),
            MagicMock(scalar_one_or_none=lambda: mock_utils),
        ]

        # Mock files for resolved modules
        mock_models_file = MagicMock(spec=File)
        mock_models_file.path = "/src/models.py"

        mock_utils_file = MagicMock(spec=File)
        mock_utils_file.path = "/src/utils.py"

        file_resolutions = [
            MagicMock(scalar_one_or_none=lambda: mock_models_file),
            MagicMock(scalar_one_or_none=lambda: mock_utils_file),
        ]

        mock_db_session.execute.side_effect = [
            file_result,
            module_result,
            imports_result,
            *module_resolutions,
            *file_resolutions,
        ]

        result = await analysis_tools.analyze_dependencies("/src/services/__init__.py")

        assert result["file"] == "/src/services/__init__.py"
        assert result["module"] == "src.services"
        assert result["total_imports"] == 4
        assert result["stdlib_imports"] == 1
        assert result["third_party_imports"] == 1
        assert result["local_imports"] == 2

    @pytest.mark.asyncio
    async def test_find_circular_dependencies_none_found(
        self, analysis_tools, mock_db_session
    ):
        """Test finding circular dependencies when none exist."""
        # Mock files in repository
        files = []
        for i, path in enumerate(
            ["/src/models.py", "/src/views.py", "/src/controllers.py"]
        ):
            f = MagicMock(spec=File)
            f.id = i + 1
            f.path = path
            files.append(f)

        files_result = MagicMock()
        files_result.scalars.return_value.all.return_value = files

        # Mock imports (no cycles)
        # models.py imports nothing
        # views.py imports models
        # controllers.py imports models and views
        import_results = [
            MagicMock(scalars=lambda: MagicMock(all=list)),  # models imports
            MagicMock(
                scalars=lambda: MagicMock(
                    all=lambda: [
                        MagicMock(
                            module_name="src.models", is_local=True, imported_file_id=1
                        )
                    ]
                )
            ),  # views imports
            MagicMock(
                scalars=lambda: MagicMock(
                    all=lambda: [
                        MagicMock(
                            module_name="src.models", is_local=True, imported_file_id=1
                        ),
                        MagicMock(
                            module_name="src.views", is_local=True, imported_file_id=2
                        ),
                    ]
                )
            ),  # controllers imports
        ]

        mock_db_session.execute.side_effect = [files_result, *import_results]

        result = await analysis_tools.find_circular_dependencies(repository_id=10)

        assert result["repository_id"] == 10
        assert result["circular_dependencies"] == []
        assert result["files_analyzed"] == 3

    @pytest.mark.asyncio
    async def test_find_circular_dependencies_with_cycles(
        self, analysis_tools, mock_db_session
    ):
        """Test finding circular dependencies with multiple cycles."""
        # Mock files
        files = []
        file_paths = [
            "/src/auth/user.py",
            "/src/auth/permissions.py",
            "/src/models/order.py",
            "/src/models/product.py",
            "/src/services/pricing.py",
        ]

        for i, path in enumerate(file_paths):
            f = MagicMock(spec=File)
            f.id = i + 1
            f.path = path
            files.append(f)

        files_result = MagicMock()
        files_result.scalars.return_value.all.return_value = files

        # Create circular dependencies:
        # Cycle 1: user.py -> permissions.py -> user.py
        # Cycle 2: order.py -> product.py -> pricing.py -> order.py

        def create_import(module_name, file_id):
            imp = MagicMock(spec=Import)
            imp.module_name = module_name
            imp.is_local = True
            imp.imported_file_id = file_id
            return imp

        import_results = [
            # user.py imports permissions
            MagicMock(
                scalars=lambda: MagicMock(
                    all=lambda: [create_import("src.auth.permissions", 2)]
                )
            ),
            # permissions.py imports user (cycle!)
            MagicMock(
                scalars=lambda: MagicMock(
                    all=lambda: [create_import("src.auth.user", 1)]
                )
            ),
            # order.py imports product
            MagicMock(
                scalars=lambda: MagicMock(
                    all=lambda: [create_import("src.models.product", 4)]
                )
            ),
            # product.py imports pricing
            MagicMock(
                scalars=lambda: MagicMock(
                    all=lambda: [create_import("src.services.pricing", 5)]
                )
            ),
            # pricing.py imports order (cycle!)
            MagicMock(
                scalars=lambda: MagicMock(
                    all=lambda: [create_import("src.models.order", 3)]
                )
            ),
        ]

        mock_db_session.execute.side_effect = [files_result, *import_results]

        result = await analysis_tools.find_circular_dependencies(repository_id=10)

        assert result["repository_id"] == 10
        assert len(result["circular_dependencies"]) == 2
        assert result["files_analyzed"] == 5

        # Check first cycle
        cycle1 = next(
            c
            for c in result["circular_dependencies"]
            if "/src/auth/user.py" in c["cycle"]
        )
        assert len(cycle1["cycle"]) == 3  # user -> permissions -> user
        assert cycle1["cycle"][0] == cycle1["cycle"][-1]  # Cycle closes

        # Check second cycle
        cycle2 = next(
            c
            for c in result["circular_dependencies"]
            if "/src/models/order.py" in c["cycle"]
        )
        assert len(cycle2["cycle"]) == 4  # order -> product -> pricing -> order
        assert "/src/services/pricing.py" in cycle2["cycle"]

    @pytest.mark.asyncio
    async def test_analyze_import_graph_complex(self, analysis_tools, mock_db_session):
        """Test analyzing complex import graph with metrics."""
        # Mock repository files
        files = []
        for i in range(10):
            f = MagicMock(spec=File)
            f.id = i + 1
            f.path = f"/src/module{i}.py"
            files.append(f)

        files_result = MagicMock()
        files_result.scalars.return_value.all.return_value = files

        # Create import graph:
        # module0 is imported by everyone (core module)
        # module1-3 import module0 and each other
        # module4-6 form isolated group
        # module7-9 have no imports

        import_patterns = {
            0: [],  # Core module, imports nothing
            1: [0, 2],  # Imports core and module2
            2: [0, 1, 3],  # Imports core, module1, and module3
            3: [0, 2],  # Imports core and module2
            4: [5, 6],  # Isolated group
            5: [4, 6],
            6: [4, 5],
            7: [],  # No imports
            8: [],
            9: [],
        }

        def create_import(module_name, file_id):
            imp = MagicMock(spec=Import)
            imp.module_name = module_name
            imp.is_local = True
            imp.imported_file_id = file_id
            return imp

        import_results = []
        for imported_ids in import_patterns.values():
            imports = [
                create_import(f"src.module{imp_id}", imp_id + 1)
                for imp_id in imported_ids
            ]
            import_results.append(
                MagicMock(
                    scalars=lambda imports=imports: MagicMock(all=lambda: imports)
                )
            )

        mock_db_session.execute.side_effect = [files_result, *import_results]

        result = await analysis_tools.analyze_import_graph(repository_id=10)

        assert result["repository_id"] == 10
        assert result["total_files"] == 10
        assert result["total_local_imports"] > 0

        # Check most imported (should be module0)
        assert len(result["most_imported_files"]) > 0
        assert "/src/module0.py" in [f["file"] for f in result["most_imported_files"]]

        # Check most importing (should include module2 with 3 imports)
        assert len(result["most_importing_files"]) > 0

        # Check isolated files (module7-9)
        assert result["isolated_files"] >= 3

    @pytest.mark.asyncio
    async def test_analyze_dependencies_with_relative_imports(
        self, analysis_tools, mock_db_session
    ):
        """Test analyzing dependencies with complex relative imports."""
        # Mock file in nested structure
        mock_file = MagicMock(spec=File)
        mock_file.id = 1
        mock_file.path = "/src/services/auth/handlers.py"
        mock_file.repository_id = 10

        file_result = MagicMock()
        file_result.scalar_one_or_none.return_value = mock_file

        # Mock various relative imports
        imports = []
        relative_imports = [
            (".", "validators", "current package import"),
            ("..", "utils", "parent package import"),
            ("..models", "User, Role", "cousin package import"),
            ("...common", "constants", "grandparent package import"),
        ]

        for module, names, _desc in relative_imports:
            imp = MagicMock(spec=Import)
            imp.module_name = module
            imp.imported_names = names
            imp.alias = None
            imp.is_stdlib = False
            imp.is_local = True
            imports.append(imp)

        imports_result = MagicMock()
        imports_result.scalars.return_value.all.return_value = imports

        # Mock resolved modules (some found, some not)
        mock_validators = MagicMock(spec=Module)
        mock_validators.name = "src.services.auth.validators"
        mock_validators.file_id = 20

        mock_utils = MagicMock(spec=Module)
        mock_utils.name = "src.services.utils"
        mock_utils.file_id = 30

        module_results = [
            MagicMock(scalar_one_or_none=lambda: mock_validators),
            MagicMock(scalar_one_or_none=lambda: mock_utils),
            MagicMock(scalar_one_or_none=lambda: None),  # ..models not found
            MagicMock(scalar_one_or_none=lambda: None),  # ...common not found
        ]

        # Mock files for resolved modules
        mock_validators_file = MagicMock(spec=File)
        mock_validators_file.path = "/src/services/auth/validators.py"

        mock_utils_file = MagicMock(spec=File)
        mock_utils_file.path = "/src/services/utils.py"

        file_results = [
            MagicMock(scalar_one_or_none=lambda: mock_validators_file),
            MagicMock(scalar_one_or_none=lambda: mock_utils_file),
        ]

        mock_db_session.execute.side_effect = [
            file_result,
            imports_result,
            *module_results,
            *file_results,
        ]

        result = await analysis_tools.analyze_dependencies(
            "/src/services/auth/handlers.py"
        )

        assert result["total_imports"] == 4
        assert result["local_imports"] == 4

        # Check resolved relative imports
        assert len(result["resolved_dependencies"]) == 2
        resolved_modules = [d["module"] for d in result["resolved_dependencies"]]
        assert "src.services.auth.validators" in resolved_modules
        assert "src.services.utils" in resolved_modules

        # Check unresolved relative imports
        assert len(result["unresolved_dependencies"]) == 2
        assert any("..models" in dep for dep in result["unresolved_dependencies"])
        assert any("...common" in dep for dep in result["unresolved_dependencies"])
