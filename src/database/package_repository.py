"""Repository for package-related database operations."""

from collections import defaultdict
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.database.package_models import Package, PackageDependency, PackageMetrics
from src.logger import get_logger

logger = get_logger(__name__)


class PackageRepository:
    """Repository for package operations."""

    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session

    async def get_package_by_path(
        self, repository_id: int, path: str
    ) -> Package | None:
        """Get a package by its path within a repository."""
        result = await self.db_session.execute(
            select(Package).where(
                Package.repository_id == repository_id, Package.path == path
            )
        )
        return result.scalar_one_or_none()

    async def get_repository_packages(
        self, repository_id: int, include_metrics: bool = False
    ) -> list[Package]:
        """Get all packages in a repository."""
        query = select(Package).where(Package.repository_id == repository_id)

        if include_metrics:
            query = query.options(selectinload(Package.metrics))

        result = await self.db_session.execute(query)
        return result.scalars().all()

    async def get_package_tree(self, repository_id: int) -> dict[str, Any]:
        """Get the complete package tree for a repository."""
        packages = await self.get_repository_packages(repository_id)

        # Build tree
        def build_tree_node(package: Package) -> dict[str, Any]:
            children = [
                build_tree_node(p) for p in packages if p.parent_id == package.id
            ]

            return {
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
                "children": children,
            }

        # Find root packages
        roots = [p for p in packages if p.parent_id is None]

        return {
            "total_packages": len(packages),
            "root_packages": [build_tree_node(root) for root in roots],
        }

    async def get_package_dependencies(
        self, package_id: int, direction: str = "both"
    ) -> dict[str, list[dict[str, Any]]]:
        """Get dependencies for a package.

        Args:
            package_id: Package ID
            direction: 'imports' (what this package imports),
                      'imported_by' (what imports this package),
                      or 'both'
        """
        result = {"imports": [], "imported_by": []}

        if direction in ("imports", "both"):
            # Get packages this package imports
            deps_result = await self.db_session.execute(
                select(PackageDependency, Package)
                .join(Package, PackageDependency.target_package_id == Package.id)
                .where(PackageDependency.source_package_id == package_id)
            )

            for dep, target_pkg in deps_result:
                result["imports"].append(
                    {
                        "package_id": target_pkg.id,
                        "package_name": target_pkg.name,
                        "package_path": target_pkg.path,
                        "import_count": dep.import_count,
                        "dependency_type": dep.dependency_type,
                    }
                )

        if direction in ("imported_by", "both"):
            # Get packages that import this package
            deps_result = await self.db_session.execute(
                select(PackageDependency, Package)
                .join(Package, PackageDependency.source_package_id == Package.id)
                .where(PackageDependency.target_package_id == package_id)
            )

            for dep, source_pkg in deps_result:
                result["imported_by"].append(
                    {
                        "package_id": source_pkg.id,
                        "package_name": source_pkg.name,
                        "package_path": source_pkg.path,
                        "import_count": dep.import_count,
                        "dependency_type": dep.dependency_type,
                    }
                )

        return result

    async def get_package_metrics(self, package_id: int) -> PackageMetrics | None:
        """Get metrics for a package."""
        result = await self.db_session.execute(
            select(PackageMetrics).where(PackageMetrics.package_id == package_id)
        )
        return result.scalar_one_or_none()

    async def find_circular_dependencies(
        self, repository_id: int
    ) -> list[list[dict[str, Any]]]:
        """Find circular dependencies between packages."""
        # Get all packages and their dependencies
        packages = await self.get_repository_packages(repository_id)
        package_map = {p.id: p for p in packages}

        # Build adjacency list
        graph = defaultdict(list)
        deps_result = await self.db_session.execute(
            select(PackageDependency).where(
                PackageDependency.source_package_id.in_(package_map.keys())
            )
        )

        for dep in deps_result.scalars():
            graph[dep.source_package_id].append(dep.target_package_id)

        # Find cycles using DFS
        cycles = []
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node: int) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:]
                    cycle_info = [
                        {
                            "package_id": pid,
                            "package_name": package_map[pid].name,
                            "package_path": package_map[pid].path,
                        }
                        for pid in cycle
                    ]
                    cycles.append(cycle_info)

            path.pop()
            rec_stack.remove(node)

        # Run DFS from each unvisited node
        for package_id in package_map:
            if package_id not in visited:
                dfs(package_id)

        return cycles

    async def get_package_coupling_metrics(self, repository_id: int) -> dict[str, Any]:
        """Calculate coupling metrics for all packages in a repository."""
        packages = await self.get_repository_packages(repository_id)

        # Get all dependencies
        deps_result = await self.db_session.execute(
            select(PackageDependency).where(
                PackageDependency.source_package_id.in_([p.id for p in packages])
            )
        )
        dependencies = deps_result.scalars().all()

        # Calculate metrics
        metrics = {
            "total_packages": len(packages),
            "total_dependencies": len(dependencies),
            "avg_dependencies_per_package": (
                len(dependencies) / len(packages) if packages else 0
            ),
            "packages_with_no_dependencies": 0,
            "packages_with_no_dependents": 0,
            "most_depended_on": [],
            "most_dependent": [],
        }

        # Count dependencies per package
        outgoing = defaultdict(int)
        incoming = defaultdict(int)

        for dep in dependencies:
            outgoing[dep.source_package_id] += 1
            incoming[dep.target_package_id] += 1

        # Find packages with no dependencies
        for package in packages:
            if package.id not in outgoing:
                metrics["packages_with_no_dependencies"] += 1
            if package.id not in incoming:
                metrics["packages_with_no_dependents"] += 1

        # Find most coupled packages
        if packages:
            sorted_by_deps = sorted(
                packages, key=lambda p: outgoing.get(p.id, 0), reverse=True
            )[:5]

            metrics["most_dependent"] = [
                {
                    "package_name": p.name,
                    "package_path": p.path,
                    "dependency_count": outgoing.get(p.id, 0),
                }
                for p in sorted_by_deps
            ]

            sorted_by_dependents = sorted(
                packages, key=lambda p: incoming.get(p.id, 0), reverse=True
            )[:5]

            metrics["most_depended_on"] = [
                {
                    "package_name": p.name,
                    "package_path": p.path,
                    "dependent_count": incoming.get(p.id, 0),
                }
                for p in sorted_by_dependents
            ]

        return metrics


