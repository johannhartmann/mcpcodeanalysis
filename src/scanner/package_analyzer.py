"""Package structure analyzer for Python projects."""

from collections import defaultdict
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import File, Import, Module, Repository
from src.database.package_models import (
    Package,
    PackageDependency,
    PackageMetrics,
    PackageModule,
)
from src.logger import get_logger

logger = get_logger(__name__)


class PackageAnalyzer:
    """Analyzes package structure and dependencies within a repository."""

    def __init__(self, db_session: AsyncSession, repository_id: int):
        self.db_session = db_session
        self.repository_id = repository_id
        self._package_cache: dict[str, Package] = {}
        self._file_cache: dict[str, File] = {}

    async def analyze_packages(self) -> dict[str, Any]:
        """Analyze the complete package structure of the repository."""
        logger.info(
            "Starting package structure analysis for repository %d", self.repository_id
        )

        # Get repository info
        repo = await self.db_session.get(Repository, self.repository_id)
        if not repo:
            msg = f"Repository {self.repository_id} not found"
            raise ValueError(msg)

        # Clean up existing package data for this repository
        await self._cleanup_existing_packages()

        # Find all Python files
        result = await self.db_session.execute(
            select(File).where(
                File.repository_id == self.repository_id,
                File.language == "python",
                ~File.is_deleted,
            )
        )
        python_files = result.scalars().all()

        # Build file cache
        self._file_cache = {f.path: f for f in python_files}

        # Find all packages (directories with __init__.py)
        packages = await self._discover_packages(python_files)

        # Create package hierarchy
        await self._create_package_hierarchy(packages)

        # Analyze package contents
        await self._analyze_package_contents()

        # Analyze dependencies between packages
        await self._analyze_package_dependencies()

        # Calculate package metrics
        await self._calculate_package_metrics()

        # Commit all changes
        await self.db_session.commit()

        return {
            "packages_found": len(self._package_cache),
            "total_files": len(python_files),
            "package_tree": await self._build_package_tree(),
        }

    async def _cleanup_existing_packages(self) -> None:
        """Remove existing package data for this repository."""
        from sqlalchemy import delete

        # Delete in order to respect foreign key constraints
        await self.db_session.execute(
            delete(PackageMetrics).where(
                PackageMetrics.package_id.in_(
                    select(Package.id).where(
                        Package.repository_id == self.repository_id
                    )
                )
            )
        )

        await self.db_session.execute(
            delete(PackageDependency).where(
                PackageDependency.source_package_id.in_(
                    select(Package.id).where(
                        Package.repository_id == self.repository_id
                    )
                )
            )
        )

        await self.db_session.execute(
            delete(PackageModule).where(
                PackageModule.package_id.in_(
                    select(Package.id).where(
                        Package.repository_id == self.repository_id
                    )
                )
            )
        )

        await self.db_session.execute(
            delete(Package).where(Package.repository_id == self.repository_id)
        )

        await self.db_session.flush()

    async def _discover_packages(  # noqa: PLR0912
        self, python_files: list[File]
    ) -> dict[str, dict[str, Any]]:
        """Discover all packages in the repository."""
        packages = {}

        # Find all __init__.py files
        init_files = [f for f in python_files if f.path.endswith("__init__.py")]

        for init_file in init_files:
            package_path = str(Path(init_file.path).parent)
            package_name = Path(package_path).name

            # Root package might be "." or empty
            if package_path == ".":
                package_path = ""
                package_name = "root"

            packages[package_path] = {
                "name": package_name,
                "path": package_path,
                "init_file": init_file,
                "modules": [],
                "subpackages": [],
            }

        # Find namespace packages (PEP 420) - directories with .py files but no __init__.py
        all_dirs = set()
        for file in python_files:
            path = Path(file.path)
            for parent in path.parents:
                if str(parent) != ".":
                    all_dirs.add(str(parent))

        # Check each directory
        for dir_path in all_dirs:
            if dir_path not in packages:
                # Check if it contains Python files
                has_py_files = any(
                    f.path.startswith(dir_path + "/")
                    and f.path.count("/", len(dir_path) + 1) == 0
                    for f in python_files
                )

                if has_py_files:
                    package_name = Path(dir_path).name
                    packages[dir_path] = {
                        "name": package_name,
                        "path": dir_path,
                        "init_file": None,
                        "is_namespace": True,
                        "modules": [],
                        "subpackages": [],
                    }

        # Find modules in each package
        for file in python_files:
            if file.path.endswith("__init__.py"):
                continue

            file_path = Path(file.path)
            parent_path = str(file_path.parent)

            if parent_path in packages:
                module_name = file_path.stem
                packages[parent_path]["modules"].append(
                    {"name": module_name, "file": file}
                )

        # Build parent-child relationships
        for pkg_path in sorted(packages.keys(), key=len, reverse=True):
            if pkg_path:
                parent_path = str(Path(pkg_path).parent)
                if parent_path in packages and parent_path != pkg_path:
                    packages[parent_path]["subpackages"].append(pkg_path)

        return packages

    async def _create_package_hierarchy(
        self, packages: dict[str, dict[str, Any]]
    ) -> None:
        """Create package records and establish hierarchy."""
        # Create packages in order (parents first)
        for pkg_path in sorted(packages.keys(), key=len):
            pkg_info = packages[pkg_path]

            # Find parent package
            parent_id = None
            if pkg_path:
                parent_path = str(Path(pkg_path).parent)
                # Handle "." as root package path
                if parent_path == ".":
                    parent_path = ""
                if parent_path in self._package_cache:
                    parent_id = self._package_cache[parent_path].id

            # Create package record
            package = Package(
                repository_id=self.repository_id,
                path=pkg_path,
                name=pkg_info["name"],
                parent_id=parent_id,
                init_file_id=(
                    pkg_info["init_file"].id if pkg_info.get("init_file") else None
                ),
                has_init=bool(pkg_info.get("init_file")),
                is_namespace=pkg_info.get("is_namespace", False),
            )

            # Extract docstring from __init__.py if available
            if pkg_info.get("init_file"):
                module_result = await self.db_session.execute(
                    select(Module).where(Module.file_id == pkg_info["init_file"].id)
                )
                module = module_result.scalar_one_or_none()
                if module:
                    package.docstring = module.docstring

            self.db_session.add(package)
            await self.db_session.flush()  # Get ID

            self._package_cache[pkg_path] = package

            # Check for README
            readme_names = ["README.md", "README.rst", "README.txt", "readme.md"]
            for readme_name in readme_names:
                readme_path = (
                    str(Path(pkg_path) / readme_name) if pkg_path else readme_name
                )
                if readme_path in self._file_cache:
                    package.readme_file_id = self._file_cache[readme_path].id
                    break

    async def _analyze_package_contents(self) -> None:
        """Analyze the contents of each package."""
        for pkg_path, package in self._package_cache.items():
            # Count direct modules
            module_count = 0
            for file_path, file in self._file_cache.items():
                if file_path.endswith("__init__.py"):
                    continue

                file_parent = str(Path(file_path).parent)
                if file_parent == pkg_path:
                    module_count += 1

                    # Create PackageModule association
                    module_name = Path(file_path).stem
                    pkg_module = PackageModule(
                        package_id=package.id,
                        file_id=file.id,
                        module_name=module_name,
                        is_public=not module_name.startswith("_"),
                    )

                    # Check for __all__ exports
                    module_result = await self.db_session.execute(
                        select(Module).where(Module.file_id == file.id)
                    )
                    module = module_result.scalar_one_or_none()
                    if module:
                        # TODO(dev): Extract __all__ from module (would need AST analysis)
                        pass

                    self.db_session.add(pkg_module)

            # Count subpackages
            subpackage_count = sum(
                1 for p in self._package_cache.values() if p.parent_id == package.id
            )

            # Update package statistics
            package.module_count = module_count
            package.subpackage_count = subpackage_count

    async def _analyze_package_dependencies(self) -> None:
        """Analyze dependencies between packages."""
        # Get all imports in the repository
        result = await self.db_session.execute(
            select(Import, File)
            .join(File)
            .where(File.repository_id == self.repository_id)
        )
        imports = result.all()

        # Track dependencies
        dependencies = defaultdict(lambda: defaultdict(list))

        for import_obj, file in imports:
            # Find source package
            source_package = self._find_package_for_file(file.path)
            if not source_package:
                continue

            # Parse import to find target package
            if import_obj.module_name:
                target_package = self._find_package_by_import(import_obj.module_name)
                if target_package and target_package.id != source_package.id:
                    dependencies[source_package.id][target_package.id].append(
                        {
                            "module": import_obj.module_name,
                            "names": import_obj.imported_names,
                            "line": import_obj.line_number,
                        }
                    )

        # Create dependency records
        for source_id, targets in dependencies.items():
            for target_id, import_list in targets.items():
                dep = PackageDependency(
                    source_package_id=source_id,
                    target_package_id=target_id,
                    import_count=len(import_list),
                    dependency_type="direct",
                    import_details=import_list,
                )
                self.db_session.add(dep)

    def _find_package_for_file(self, file_path: str) -> Package | None:
        """Find which package a file belongs to."""
        file_dir = str(Path(file_path).parent)

        # Check exact match first
        if file_dir in self._package_cache:
            return self._package_cache[file_dir]

        # Check parent directories
        path = Path(file_path)
        for parent in path.parents:
            parent_str = str(parent)
            if parent_str in self._package_cache:
                return self._package_cache[parent_str]

        return None

    def _find_package_by_import(self, module_name: str) -> Package | None:
        """Find package by import name."""
        # For relative imports within the repository
        parts = module_name.split(".")

        # Try to match against package paths
        for i in range(len(parts), 0, -1):
            potential_path = "/".join(parts[:i])
            if potential_path in self._package_cache:
                return self._package_cache[potential_path]

        # Try to match against package names
        for package in self._package_cache.values():
            if package.name == parts[0]:
                return package

        return None

    async def _calculate_package_metrics(self) -> None:
        """Calculate metrics for each package."""
        for package in self._package_cache.values():
            metrics = PackageMetrics(package_id=package.id)

            # Get all files in package and subpackages
            package_files = []
            for file_path, file in self._file_cache.items():
                if self._is_file_in_package(file_path, package.path):
                    package_files.append(file)

            if not package_files:
                self.db_session.add(metrics)
                continue

            # Get all modules, classes, and functions
            file_ids = [f.id for f in package_files]

            # Get modules
            module_result = await self.db_session.execute(
                select(Module).where(Module.file_id.in_(file_ids))
            )
            modules = module_result.scalars().all()

            # Count lines, functions, classes
            total_lines = sum(f.size or 0 for f in package_files)
            total_functions = 0
            total_classes = 0
            total_complexity = 0
            max_complexity = 0

            # Get detailed statistics from the database
            from src.database.models import Class, Function

            all_classes = []
            all_functions = []

            for module in modules:
                # Count classes
                class_result = await self.db_session.execute(
                    select(Class).where(Class.module_id == module.id)
                )
                classes = class_result.scalars().all()
                all_classes.extend(classes)
                total_classes += len(classes)

                # Count functions
                func_result = await self.db_session.execute(
                    select(Function).where(Function.module_id == module.id)
                )
                functions = func_result.scalars().all()
                all_functions.extend(functions)
                total_functions += len(functions)

                # Calculate complexity
                for func in functions:
                    if func.complexity:
                        total_complexity += func.complexity
                        max_complexity = max(max_complexity, func.complexity)

            # Update package stats
            package.total_lines = total_lines
            package.total_functions = total_functions
            package.total_classes = total_classes

            # Update metrics
            metrics.total_loc = total_lines
            metrics.total_complexity = total_complexity
            metrics.max_complexity = max_complexity
            if total_functions > 0:
                metrics.avg_complexity = total_complexity // total_functions

            # Check for tests
            metrics.has_tests = any(
                "test" in f.path or "tests" in f.path for f in package_files
            )

            # Check for docs
            metrics.has_docs = bool(package.readme_file_id or package.docstring)

            # Count public API
            public_classes = sum(
                1 for cls in all_classes if not cls.name.startswith("_")
            )
            public_functions = sum(
                1
                for func in all_functions
                if not func.name.startswith("_") and not func.class_id
            )

            metrics.public_classes = public_classes
            metrics.public_functions = public_functions

            self.db_session.add(metrics)

    def _is_file_in_package(self, file_path: str, package_path: str) -> bool:
        """Check if a file belongs to a package or its subpackages."""
        if not package_path:  # Root package
            return True
        return file_path.startswith(package_path + "/")

    async def _build_package_tree(self) -> dict[str, Any]:
        """Build a tree representation of the package structure."""

        def build_node(package: Package) -> dict[str, Any]:
            children = [
                build_node(p)
                for p in self._package_cache.values()
                if p.parent_id == package.id
            ]

            return {
                "name": package.name,
                "path": package.path,
                "modules": package.module_count,
                "subpackages": package.subpackage_count,
                "has_init": package.has_init,
                "is_namespace": package.is_namespace,
                "children": children,
            }

        # Find root packages
        roots = [p for p in self._package_cache.values() if p.parent_id is None]

        return {"packages": [build_node(root) for root in roots]}
