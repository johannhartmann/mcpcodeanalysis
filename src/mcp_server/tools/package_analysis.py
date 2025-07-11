"""MCP tools for package structure analysis."""

from typing import Any

from pydantic import BaseModel, Field

from src.database.package_repository import PackageRepository
from src.logger import get_logger
from src.scanner.package_analyzer import PackageAnalyzer

logger = get_logger(__name__)


class AnalyzePackagesRequest(BaseModel):
    """Request to analyze package structure."""

    repository_id: int = Field(description="Repository ID to analyze")
    force_refresh: bool = Field(
        default=False, description="Force re-analysis even if data exists"
    )


class GetPackageTreeRequest(BaseModel):
    """Request to get package tree structure."""

    repository_id: int = Field(description="Repository ID")


class GetPackageDetailsRequest(BaseModel):
    """Request to get details about a specific package."""

    repository_id: int = Field(description="Repository ID")
    package_path: str = Field(description="Package path (e.g., 'src/utils')")


class GetPackageDependenciesRequest(BaseModel):
    """Request to get package dependencies."""

    repository_id: int = Field(description="Repository ID")
    package_path: str = Field(description="Package path")
    direction: str = Field(
        default="both", description="Direction: 'imports', 'imported_by', or 'both'"
    )


class FindCircularDependenciesRequest(BaseModel):
    """Request to find circular dependencies."""

    repository_id: int = Field(description="Repository ID")


class GetPackageCouplingRequest(BaseModel):
    """Request to get package coupling metrics."""

    repository_id: int = Field(description="Repository ID")


async def analyze_packages(
    request: AnalyzePackagesRequest, db_session
) -> dict[str, Any]:
    """Analyze the package structure of a repository.

    This tool discovers all packages (directories with __init__.py),
    analyzes their contents, dependencies, and calculates metrics.
    """
    try:
        # Check if analysis already exists
        if not request.force_refresh:
            pkg_repo = PackageRepository(db_session)
            existing = await pkg_repo.get_repository_packages(request.repository_id)
            if existing:
                return {
                    "status": "existing",
                    "packages_found": len(existing),
                    "message": "Package analysis already exists. Use force_refresh=true to re-analyze.",
                }

        # Run package analysis
        analyzer = PackageAnalyzer(db_session, request.repository_id)
        result = await analyzer.analyze_packages()

        return {"status": "success", **result}

    except Exception as e:
        logger.exception("Error analyzing packages")
        return {"status": "error", "error": str(e)}


async def get_package_tree(
    request: GetPackageTreeRequest, db_session
) -> dict[str, Any]:
    """Get the hierarchical package structure of a repository.

    Returns a tree view of all packages and their relationships.
    """
    try:
        pkg_repo = PackageRepository(db_session)
        tree = await pkg_repo.get_package_tree(request.repository_id)

        return {"status": "success", "repository_id": request.repository_id, **tree}

    except Exception as e:
        logger.exception("Error getting package tree")
        return {"status": "error", "error": str(e)}


async def get_package_details(
    request: GetPackageDetailsRequest, db_session
) -> dict[str, Any]:
    """Get detailed information about a specific package.

    Includes modules, metrics, and documentation status.
    """
    try:
        pkg_repo = PackageRepository(db_session)
        package = await pkg_repo.get_package_by_path(
            request.repository_id, request.package_path
        )

        if not package:
            return {
                "status": "not_found",
                "error": f"Package '{request.package_path}' not found",
            }

        # Get metrics
        metrics = await pkg_repo.get_package_metrics(package.id)

        # Get dependencies summary
        deps = await pkg_repo.get_package_dependencies(package.id)

        return {
            "status": "success",
            "package": {
                "id": package.id,
                "name": package.name,
                "path": package.path,
                "has_init": package.has_init,
                "is_namespace": package.is_namespace,
                "module_count": package.module_count,
                "subpackage_count": package.subpackage_count,
                "total_lines": package.total_lines,
                "total_functions": package.total_functions,
                "total_classes": package.total_classes,
                "has_readme": bool(package.readme_file_id),
                "has_docstring": bool(package.docstring),
            },
            "metrics": (
                {
                    "total_complexity": metrics.total_complexity if metrics else None,
                    "avg_complexity": metrics.avg_complexity if metrics else None,
                    "max_complexity": metrics.max_complexity if metrics else None,
                    "has_tests": metrics.has_tests if metrics else None,
                    "has_docs": metrics.has_docs if metrics else None,
                    "public_classes": metrics.public_classes if metrics else None,
                    "public_functions": metrics.public_functions if metrics else None,
                }
                if metrics
                else None
            ),
            "dependencies": {
                "imports_count": len(deps["imports"]),
                "imported_by_count": len(deps["imported_by"]),
            },
        }

    except Exception as e:
        logger.exception("Error getting package details")
        return {"status": "error", "error": str(e)}


async def get_package_dependencies(
    request: GetPackageDependenciesRequest, db_session
) -> dict[str, Any]:
    """Get dependencies for a specific package.

    Shows what packages this package imports and what packages import it.
    """
    try:
        pkg_repo = PackageRepository(db_session)
        package = await pkg_repo.get_package_by_path(
            request.repository_id, request.package_path
        )

        if not package:
            return {
                "status": "not_found",
                "error": f"Package '{request.package_path}' not found",
            }

        deps = await pkg_repo.get_package_dependencies(package.id, request.direction)

        return {
            "status": "success",
            "package_name": package.name,
            "package_path": package.path,
            **deps,
        }

    except Exception as e:
        logger.exception("Error getting package dependencies")
        return {"status": "error", "error": str(e)}


async def find_circular_dependencies(
    request: FindCircularDependenciesRequest, db_session
) -> dict[str, Any]:
    """Find circular dependencies between packages.

    Identifies dependency cycles that could indicate design issues.
    """
    try:
        pkg_repo = PackageRepository(db_session)
        cycles = await pkg_repo.find_circular_dependencies(request.repository_id)

        return {
            "status": "success",
            "repository_id": request.repository_id,
            "circular_dependencies_found": len(cycles),
            "cycles": cycles,
        }

    except Exception as e:
        logger.exception("Error finding circular dependencies")
        return {"status": "error", "error": str(e)}


async def get_package_coupling_metrics(
    request: GetPackageCouplingRequest, db_session
) -> dict[str, Any]:
    """Get coupling metrics for all packages in a repository.

    Analyzes how tightly coupled packages are to each other.
    """
    try:
        pkg_repo = PackageRepository(db_session)
        metrics = await pkg_repo.get_package_coupling_metrics(request.repository_id)

        return {"status": "success", "repository_id": request.repository_id, **metrics}

    except Exception as e:
        logger.exception("Error getting coupling metrics")
        return {"status": "error", "error": str(e)}
