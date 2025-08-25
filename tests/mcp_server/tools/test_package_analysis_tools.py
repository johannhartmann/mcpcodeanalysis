"""Tests for package analysis tools."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.package_models import Package, PackageMetrics
from src.mcp_server.tools.package_analysis import (
    AnalyzePackagesRequest,
    FindCircularDependenciesRequest,
    GetPackageCouplingRequest,
    GetPackageDependenciesRequest,
    GetPackageDetailsRequest,
    GetPackageTreeRequest,
    analyze_packages,
    find_circular_dependencies,
    get_package_coupling_metrics,
    get_package_dependencies,
    get_package_details,
    get_package_tree,
)


@pytest.fixture
def mock_db_session() -> AsyncMock:
    """Create mock database session."""
    return AsyncMock(spec=AsyncSession)


class TestPackageAnalysisTools:
    """Tests for package analysis tools."""

    @pytest.mark.asyncio
    async def test_analyze_packages_existing_no_force(
        self, mock_db_session: AsyncMock
    ) -> None:
        """Test analyzing packages when analysis already exists."""
        # Mock existing packages
        mock_packages = [MagicMock(spec=Package) for _ in range(5)]

        with patch(
            "src.mcp_server.tools.package_analysis.PackageRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_repository_packages = AsyncMock(return_value=mock_packages)
            mock_repo_class.return_value = mock_repo

            request = AnalyzePackagesRequest(repository_id=1, force_refresh=False)
            result = await analyze_packages(request, mock_db_session)

            assert result["status"] == "existing"
            assert result["packages_found"] == 5
            assert "Use force_refresh=true" in result["message"]

    @pytest.mark.asyncio
    async def test_analyze_packages_force_refresh(
        self, mock_db_session: AsyncSession
    ) -> None:
        """Test analyzing packages with force refresh."""
        with (
            patch(
                "src.mcp_server.tools.package_analysis.PackageAnalyzer"
            ) as mock_analyzer_class,
        ):
            mock_analyzer = MagicMock()
            mock_analyzer.analyze_packages = AsyncMock(
                return_value={
                    "packages_found": 10,
                    "total_modules": 50,
                    "total_lines": 5000,
                    "analysis_time": 2.5,
                }
            )
            mock_analyzer_class.return_value = mock_analyzer

            request = AnalyzePackagesRequest(repository_id=1, force_refresh=True)
            result = await analyze_packages(request, mock_db_session)

            assert result["status"] == "success"
            assert result["packages_found"] == 10
            assert result["total_modules"] == 50
            assert result["total_lines"] == 5000

    @pytest.mark.asyncio
    async def test_analyze_packages_error(self, mock_db_session: AsyncMock) -> None:
        """Test error handling in analyze_packages."""
        with patch(
            "src.mcp_server.tools.package_analysis.PackageRepository"
        ) as mock_repo_class:
            mock_repo_class.side_effect = Exception("Database error")

            request = AnalyzePackagesRequest(repository_id=1)
            result = await analyze_packages(request, mock_db_session)

            assert result["status"] == "error"
            assert result["error"] == "Database error"

    @pytest.mark.asyncio
    async def test_get_package_tree_success(self, mock_db_session: AsyncMock) -> None:
        """Test getting package tree structure."""
        mock_tree = {
            "total_packages": 5,
            "max_depth": 3,
            "tree": {
                "src": {
                    "type": "package",
                    "has_init": True,
                    "children": {
                        "utils": {"type": "package", "has_init": True, "children": {}},
                        "models": {"type": "package", "has_init": True, "children": {}},
                    },
                }
            },
        }

        with patch(
            "src.mcp_server.tools.package_analysis.PackageRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_package_tree = AsyncMock(return_value=mock_tree)
            mock_repo_class.return_value = mock_repo

            request = GetPackageTreeRequest(repository_id=1)
            result = await get_package_tree(request, mock_db_session)

            assert result["status"] == "success"
            assert result["repository_id"] == 1
            assert result["total_packages"] == 5
            assert "tree" in result

    @pytest.mark.asyncio
    async def test_get_package_tree_error(self, mock_db_session: AsyncMock) -> None:
        """Test error handling in get_package_tree."""
        with patch(
            "src.mcp_server.tools.package_analysis.PackageRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_package_tree = AsyncMock(side_effect=Exception("Tree error"))
            mock_repo_class.return_value = mock_repo

            request = GetPackageTreeRequest(repository_id=1)
            result = await get_package_tree(request, mock_db_session)

            assert result["status"] == "error"
            assert result["error"] == "Tree error"

    @pytest.mark.asyncio
    async def test_get_package_details_not_found(
        self, mock_db_session: AsyncMock
    ) -> None:
        """Test getting details for non-existent package."""
        with patch(
            "src.mcp_server.tools.package_analysis.PackageRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_package_by_path = AsyncMock(return_value=None)
            mock_repo_class.return_value = mock_repo

            request = GetPackageDetailsRequest(
                repository_id=1, package_path="src/nonexistent"
            )
            result = await get_package_details(request, mock_db_session)

            assert result["status"] == "not_found"
            assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_get_package_details_success(
        self, mock_db_session: AsyncMock
    ) -> None:
        """Test getting package details successfully."""
        # Mock package
        mock_package = MagicMock(spec=Package)
        mock_package.id = 10
        mock_package.name = "utils"
        mock_package.path = "src/utils"
        mock_package.has_init = True
        mock_package.is_namespace = False
        mock_package.module_count = 5
        mock_package.subpackage_count = 2
        mock_package.total_lines = 1000
        mock_package.total_functions = 20
        mock_package.total_classes = 5
        mock_package.readme_file_id = 100
        mock_package.docstring = "Utility package"

        # Mock metrics
        mock_metrics = MagicMock(spec=PackageMetrics)
        mock_metrics.total_complexity = 50
        mock_metrics.avg_complexity = 2.5
        mock_metrics.max_complexity = 10
        mock_metrics.has_tests = True
        mock_metrics.has_docs = True
        mock_metrics.public_classes = 3
        mock_metrics.public_functions = 15

        # Mock dependencies
        mock_deps = {
            "imports": ["src.models", "src.database"],
            "imported_by": ["src.api", "src.cli"],
        }

        with patch(
            "src.mcp_server.tools.package_analysis.PackageRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_package_by_path = AsyncMock(return_value=mock_package)
            mock_repo.get_package_metrics = AsyncMock(return_value=mock_metrics)
            mock_repo.get_package_dependencies = AsyncMock(return_value=mock_deps)
            mock_repo_class.return_value = mock_repo

            request = GetPackageDetailsRequest(
                repository_id=1, package_path="src/utils"
            )
            result = await get_package_details(request, mock_db_session)

            assert result["status"] == "success"
            assert result["package"]["name"] == "utils"
            assert result["package"]["has_readme"] is True
            assert result["metrics"]["total_complexity"] == 50
            assert result["dependencies"]["imports_count"] == 2
            assert result["dependencies"]["imported_by_count"] == 2

    @pytest.mark.asyncio
    async def test_get_package_dependencies_not_found(
        self, mock_db_session: AsyncMock
    ) -> None:
        """Test getting dependencies for non-existent package."""
        with patch(
            "src.mcp_server.tools.package_analysis.PackageRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_package_by_path = AsyncMock(return_value=None)
            mock_repo_class.return_value = mock_repo

            request = GetPackageDependenciesRequest(
                repository_id=1, package_path="src/nonexistent"
            )
            result = await get_package_dependencies(request, mock_db_session)

            assert result["status"] == "not_found"

    @pytest.mark.asyncio
    async def test_get_package_dependencies_both_directions(
        self, mock_db_session: AsyncMock
    ) -> None:
        """Test getting package dependencies in both directions."""
        mock_package = MagicMock(spec=Package)
        mock_package.id = 10
        mock_package.name = "models"
        mock_package.path = "src/models"

        mock_deps = {
            "imports": [
                {"name": "database", "path": "src/database", "import_count": 5},
                {"name": "utils", "path": "src/utils", "import_count": 3},
            ],
            "imported_by": [
                {"name": "api", "path": "src/api", "import_count": 10},
            ],
        }

        with patch(
            "src.mcp_server.tools.package_analysis.PackageRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_package_by_path = AsyncMock(return_value=mock_package)
            mock_repo.get_package_dependencies = AsyncMock(return_value=mock_deps)
            mock_repo_class.return_value = mock_repo

            request = GetPackageDependenciesRequest(
                repository_id=1, package_path="src/models", direction="both"
            )
            result = await get_package_dependencies(request, mock_db_session)

            assert result["status"] == "success"
            assert result["package_name"] == "models"
            assert len(result["imports"]) == 2
            assert len(result["imported_by"]) == 1

    @pytest.mark.asyncio
    async def test_get_package_dependencies_imports_only(
        self, mock_db_session: AsyncMock
    ) -> None:
        """Test getting only import dependencies."""
        mock_package = MagicMock(spec=Package)
        mock_package.id = 10
        mock_package.name = "api"
        mock_package.path = "src/api"

        mock_deps = {
            "imports": [
                {"name": "models", "path": "src/models", "import_count": 20},
            ],
        }

        with patch(
            "src.mcp_server.tools.package_analysis.PackageRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_package_by_path = AsyncMock(return_value=mock_package)
            mock_repo.get_package_dependencies = AsyncMock(return_value=mock_deps)
            mock_repo_class.return_value = mock_repo

            request = GetPackageDependenciesRequest(
                repository_id=1, package_path="src/api", direction="imports"
            )
            result = await get_package_dependencies(request, mock_db_session)

            assert result["status"] == "success"
            assert "imports" in result
            assert len(result["imports"]) == 1

    @pytest.mark.asyncio
    async def test_find_circular_dependencies_none_found(
        self, mock_db_session: AsyncMock
    ) -> None:
        """Test finding circular dependencies when none exist."""
        with patch(
            "src.mcp_server.tools.package_analysis.PackageRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.find_circular_dependencies = AsyncMock(return_value=[])
            mock_repo_class.return_value = mock_repo

            request = FindCircularDependenciesRequest(repository_id=1)
            result = await find_circular_dependencies(request, mock_db_session)

            assert result["status"] == "success"
            assert result["circular_dependencies_found"] == 0
            assert result["cycles"] == []

    @pytest.mark.asyncio
    async def test_find_circular_dependencies_found(
        self, mock_db_session: AsyncMock
    ) -> None:
        """Test finding circular dependencies."""
        mock_cycles = [
            {
                "cycle": ["src.models", "src.utils", "src.models"],
                "packages": [
                    {"name": "models", "path": "src/models"},
                    {"name": "utils", "path": "src/utils"},
                ],
            },
            {
                "cycle": ["src.api", "src.services", "src.handlers", "src.api"],
                "packages": [
                    {"name": "api", "path": "src/api"},
                    {"name": "services", "path": "src/services"},
                    {"name": "handlers", "path": "src/handlers"},
                ],
            },
        ]

        with patch(
            "src.mcp_server.tools.package_analysis.PackageRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.find_circular_dependencies = AsyncMock(return_value=mock_cycles)
            mock_repo_class.return_value = mock_repo

            request = FindCircularDependenciesRequest(repository_id=1)
            result = await find_circular_dependencies(request, mock_db_session)

            assert result["status"] == "success"
            assert result["circular_dependencies_found"] == 2
            assert len(result["cycles"]) == 2
            assert len(result["cycles"][0]["packages"]) == 2
            assert len(result["cycles"][1]["packages"]) == 3

    @pytest.mark.asyncio
    async def test_get_package_coupling_metrics_success(
        self, mock_db_session: AsyncMock
    ) -> None:
        """Test getting package coupling metrics."""
        mock_metrics = {
            "average_coupling": 0.25,
            "max_coupling": 0.8,
            "highly_coupled_packages": [
                {
                    "package": "src/api",
                    "coupling_score": 0.8,
                    "imports": 15,
                    "imported_by": 3,
                },
                {
                    "package": "src/models",
                    "coupling_score": 0.6,
                    "imports": 5,
                    "imported_by": 10,
                },
            ],
            "loosely_coupled_packages": [
                {
                    "package": "src/utils",
                    "coupling_score": 0.1,
                    "imports": 1,
                    "imported_by": 20,
                },
            ],
            "coupling_distribution": {
                "low": 10,  # 0-0.3
                "medium": 5,  # 0.3-0.6
                "high": 2,  # 0.6-1.0
            },
        }

        with patch(
            "src.mcp_server.tools.package_analysis.PackageRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_package_coupling_metrics = AsyncMock(
                return_value=mock_metrics
            )
            mock_repo_class.return_value = mock_repo

            request = GetPackageCouplingRequest(repository_id=1)
            result = await get_package_coupling_metrics(request, mock_db_session)

            assert result["status"] == "success"
            assert result["average_coupling"] == 0.25
            assert result["max_coupling"] == 0.8
            assert len(result["highly_coupled_packages"]) == 2
            assert result["coupling_distribution"]["low"] == 10

    @pytest.mark.asyncio
    async def test_get_package_coupling_metrics_error(
        self, mock_db_session: AsyncMock
    ) -> None:
        """Test error handling in get_package_coupling_metrics."""
        with patch(
            "src.mcp_server.tools.package_analysis.PackageRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_package_coupling_metrics = AsyncMock(
                side_effect=Exception("Metrics calculation failed")
            )
            mock_repo_class.return_value = mock_repo

            request = GetPackageCouplingRequest(repository_id=1)
            result = await get_package_coupling_metrics(request, mock_db_session)

            assert result["status"] == "error"
            assert result["error"] == "Metrics calculation failed"

    @pytest.mark.asyncio
    async def test_get_package_details_no_metrics(
        self, mock_db_session: AsyncMock
    ) -> None:
        """Test getting package details when metrics don't exist."""
        mock_package = MagicMock(spec=Package)
        mock_package.id = 10
        mock_package.name = "new_package"
        mock_package.path = "src/new_package"
        mock_package.has_init = True
        mock_package.is_namespace = False
        mock_package.module_count = 1
        mock_package.subpackage_count = 0
        mock_package.total_lines = 100
        mock_package.total_functions = 5
        mock_package.total_classes = 1
        mock_package.readme_file_id = None
        mock_package.docstring = None

        mock_deps: dict[str, list[str]] = {"imports": [], "imported_by": []}

        with patch(
            "src.mcp_server.tools.package_analysis.PackageRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_package_by_path = AsyncMock(return_value=mock_package)
            mock_repo.get_package_metrics = AsyncMock(return_value=None)
            mock_repo.get_package_dependencies = AsyncMock(return_value=mock_deps)
            mock_repo_class.return_value = mock_repo

            request = GetPackageDetailsRequest(
                repository_id=1, package_path="src/new_package"
            )
            result = await get_package_details(request, mock_db_session)

            assert result["status"] == "success"
            assert result["package"]["has_readme"] is False
            assert result["metrics"] is None
            assert result["dependencies"]["imports_count"] == 0
