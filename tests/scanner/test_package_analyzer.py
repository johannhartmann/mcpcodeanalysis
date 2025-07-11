"""Tests for package structure analyzer."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import File, Module, Repository
from src.database.package_models import Package, PackageDependency
from src.scanner.package_analyzer import PackageAnalyzer


@pytest.fixture
async def test_repository(async_session: AsyncSession) -> Repository:
    """Create a test repository."""
    repo = Repository(
        github_url="https://github.com/test/repo",
        owner="test",
        name="repo",
        default_branch="main",
    )
    async_session.add(repo)
    await async_session.commit()
    return repo


@pytest.fixture
async def test_files(
    async_session: AsyncSession, test_repository: Repository
) -> list[File]:
    """Create test files representing a package structure."""
    files = [
        # Root package
        File(
            repository_id=test_repository.id,
            path="__init__.py",
            language="python",
            size=10,
            is_deleted=False,
        ),
        File(
            repository_id=test_repository.id,
            path="main.py",
            language="python",
            size=100,
            is_deleted=False,
        ),
        File(
            repository_id=test_repository.id,
            path="README.md",
            language="markdown",
            size=500,
            is_deleted=False,
        ),
        # src package
        File(
            repository_id=test_repository.id,
            path="src/__init__.py",
            language="python",
            size=20,
            is_deleted=False,
        ),
        File(
            repository_id=test_repository.id,
            path="src/utils.py",
            language="python",
            size=200,
            is_deleted=False,
        ),
        File(
            repository_id=test_repository.id,
            path="src/helpers.py",
            language="python",
            size=150,
            is_deleted=False,
        ),
        # src/core subpackage
        File(
            repository_id=test_repository.id,
            path="src/core/__init__.py",
            language="python",
            size=30,
            is_deleted=False,
        ),
        File(
            repository_id=test_repository.id,
            path="src/core/engine.py",
            language="python",
            size=300,
            is_deleted=False,
        ),
        # tests package
        File(
            repository_id=test_repository.id,
            path="tests/__init__.py",
            language="python",
            size=5,
            is_deleted=False,
        ),
        File(
            repository_id=test_repository.id,
            path="tests/test_utils.py",
            language="python",
            size=150,
            is_deleted=False,
        ),
        # Namespace package (no __init__.py)
        File(
            repository_id=test_repository.id,
            path="namespace/module1.py",
            language="python",
            size=100,
            is_deleted=False,
        ),
        File(
            repository_id=test_repository.id,
            path="namespace/module2.py",
            language="python",
            size=120,
            is_deleted=False,
        ),
    ]

    for file in files:
        async_session.add(file)
    await async_session.commit()

    # Create modules for __init__ files
    for file in files:
        if file.path.endswith("__init__.py"):
            module = Module(
                file_id=file.id,
                name=file.path.replace("/__init__.py", "").replace(
                    "__init__.py", "root"
                ),
                docstring=f"Package docstring for {file.path}",
                start_line=1,
                end_line=10,
            )
            async_session.add(module)

    await async_session.commit()
    return files


@pytest.mark.asyncio
async def test_discover_packages(
    async_session: AsyncSession,
    test_repository: Repository,
    test_files: list[File],
):
    """Test package discovery."""
    analyzer = PackageAnalyzer(async_session, test_repository.id)

    # Discover packages
    packages = await analyzer._discover_packages(test_files)

    # Should find regular packages with __init__.py
    assert "" in packages  # Root package
    assert "src" in packages
    assert "src/core" in packages
    assert "tests" in packages

    # Should find namespace package
    assert "namespace" in packages
    assert packages["namespace"]["is_namespace"] is True

    # Check package contents
    assert len(packages["src"]["modules"]) == 2  # utils.py and helpers.py
    assert len(packages["src"]["subpackages"]) == 1  # core
    assert packages["src/core"]["modules"][0]["name"] == "engine"


@pytest.mark.asyncio
async def test_analyze_packages(
    async_session: AsyncSession,
    test_repository: Repository,
    test_files: list[File],
):
    """Test complete package analysis."""
    analyzer = PackageAnalyzer(async_session, test_repository.id)

    # Run analysis
    result = await analyzer.analyze_packages()

    # Check packages were created in database
    from sqlalchemy import select

    result_db = await async_session.execute(
        select(Package).where(Package.repository_id == test_repository.id)
    )
    packages = result_db.scalars().all()

    assert result["packages_found"] == 5  # root, src, src/core, tests, namespace
    assert result["total_files"] == len(
        [f for f in test_files if f.language == "python"]
    )
    assert len(packages) == 5

    # Check hierarchy
    root_pkg = next(p for p in packages if p.path == "")
    src_pkg = next(p for p in packages if p.path == "src")
    assert src_pkg.parent_id == root_pkg.id

    # Check namespace package
    ns_pkg = next(p for p in packages if p.path == "namespace")
    assert ns_pkg.is_namespace is True
    assert ns_pkg.has_init is False


@pytest.mark.asyncio
async def test_package_metrics(
    async_session: AsyncSession,
    test_repository: Repository,
    test_files: list[File],
):
    """Test package metrics calculation."""
    # Add some classes and functions to test metrics
    src_file = next(f for f in test_files if f.path == "src/utils.py")
    module = Module(
        file_id=src_file.id,
        name="utils",
        start_line=1,
        end_line=100,
    )
    async_session.add(module)
    await async_session.commit()

    from src.database.models import Class, Function

    # Add a class
    cls = Class(
        module_id=module.id,
        name="UtilityClass",
        start_line=10,
        end_line=50,
    )
    async_session.add(cls)
    await async_session.commit()  # Commit class to get its ID

    # Add functions
    func1 = Function(
        module_id=module.id,
        name="utility_function",
        start_line=60,
        end_line=80,
        complexity=5,
    )
    func2 = Function(
        module_id=module.id,
        class_id=cls.id,
        name="method",
        start_line=20,
        end_line=30,
        complexity=3,
    )
    async_session.add_all([func1, func2])
    await async_session.commit()

    # Run analysis
    analyzer = PackageAnalyzer(async_session, test_repository.id)
    await analyzer.analyze_packages()

    # Check metrics
    from sqlalchemy import select

    from src.database.package_models import PackageMetrics

    result = await async_session.execute(
        select(PackageMetrics).join(Package).where(Package.path == "src")
    )
    src_metrics = result.scalar_one()

    assert src_metrics.total_complexity == 8  # 5 + 3
    assert src_metrics.avg_complexity == 4  # 8 / 2
    assert src_metrics.max_complexity == 5
    assert src_metrics.has_tests is False  # No test files in src
    assert src_metrics.public_classes == 1

    assert (
        src_metrics.public_functions == 1
    )  # Only module-level function (not class methods)


@pytest.mark.asyncio
async def test_package_dependencies(
    async_session: AsyncSession,
    test_repository: Repository,
    test_files: list[File],
):
    """Test package dependency analysis."""
    # Add imports to create dependencies
    from src.database.models import Import

    # src/utils.py imports from src.core
    src_utils = next(f for f in test_files if f.path == "src/utils.py")
    import1 = Import(
        file_id=src_utils.id,
        import_statement="from src.core.engine import Engine",
        module_name="src.core.engine",
        imported_names=["Engine"],
        line_number=1,
    )

    # tests/test_utils.py imports from src
    test_utils = next(f for f in test_files if f.path == "tests/test_utils.py")
    import2 = Import(
        file_id=test_utils.id,
        import_statement="from src.utils import utility_function",
        module_name="src.utils",
        imported_names=["utility_function"],
        line_number=1,
    )

    async_session.add_all([import1, import2])
    await async_session.commit()

    # Run analysis
    analyzer = PackageAnalyzer(async_session, test_repository.id)
    await analyzer.analyze_packages()

    # Check dependencies
    from sqlalchemy import select

    result = await async_session.execute(select(PackageDependency))
    deps = result.scalars().all()

    # Should have 2 dependencies
    assert len(deps) == 2

    # Get packages to check dependencies
    pkg_result = await async_session.execute(
        select(Package).where(Package.repository_id == test_repository.id)
    )
    pkgs = {p.path: p for p in pkg_result.scalars()}

    # Check src -> src/core dependency
    src_to_core = next(
        d
        for d in deps
        if d.source_package_id == pkgs["src"].id
        and d.target_package_id == pkgs["src/core"].id
    )
    assert src_to_core.import_count == 1
    assert src_to_core.dependency_type == "direct"

    # Check tests -> src dependency
    tests_to_src = next(
        d
        for d in deps
        if d.source_package_id == pkgs["tests"].id
        and d.target_package_id == pkgs["src"].id
    )
    assert tests_to_src.import_count == 1
