#!/usr/bin/env python3
"""Comprehensive test script for all MCP methods on mcpcodeanalysis repository."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy import select

from src.database.init_db import get_session_factory, init_database
from src.database.migration_models import MigrationStrategy
from src.database.models import Repository
from src.logger import get_logger

logger = get_logger(__name__)


class MCPMethodTester:
    """Test all MCP methods systematically."""

    def __init__(
        self, repository_url: str = "https://github.com/johannhartmann/mcpcodeanalysis"
    ):
        self.repository_url = repository_url
        self.repository_id = None
        self.session_factory = None
        self.results = {"passed": [], "failed": [], "warnings": []}

    async def setup(self):
        """Initialize database and find repository."""
        engine = await init_database()
        self.session_factory = get_session_factory(engine)

        async with self.session_factory() as session:
            result = await session.execute(
                select(Repository).where(Repository.github_url == self.repository_url)
            )
            repo = result.scalar_one_or_none()

            if not repo:
                msg = f"Repository {self.repository_url} not found in database"
                raise ValueError(msg)

            self.repository_id = repo.id
            print(f"Testing repository: {repo.name} (ID: {repo.id})")
            print(f"Files in repository: {await self._count_files(session)}")
            print("=" * 80)

    async def _count_files(self, session):
        """Count files in repository."""
        from sqlalchemy import func

        from src.database.models import File

        result = await session.execute(
            select(func.count(File.id)).where(File.repository_id == self.repository_id)
        )
        return result.scalar()

    async def test_core_analysis(self):
        """Test core code analysis methods."""
        print("\n### CORE CODE ANALYSIS METHODS ###\n")

        # Test 1: find_definition
        await self._test_method(
            "find_definition",
            self._test_find_definition,
            "Find class/function definitions",
        )

        # Test 2: get_code_structure
        await self._test_method(
            "get_code_structure",
            self._test_get_code_structure,
            "Get file structure analysis",
        )

        # Test 3: search_code
        await self._test_method(
            "search_code", self._test_search_code, "Natural language code search"
        )

        # Test 4: find_usage
        await self._test_method(
            "find_usage", self._test_find_usage, "Find symbol usage locations"
        )

        # Test 5: suggest_refactoring
        await self._test_method(
            "suggest_refactoring",
            self._test_suggest_refactoring,
            "AI-powered refactoring suggestions",
        )

        # Test 6: find_similar_code
        await self._test_method(
            "find_similar_code",
            self._test_find_similar_code,
            "Find similar code patterns",
        )

    async def test_package_analysis(self):
        """Test package analysis methods."""
        print("\n### PACKAGE ANALYSIS METHODS ###\n")

        # Test 7: analyze_packages
        await self._test_method(
            "analyze_packages", self._test_analyze_packages, "Analyze package structure"
        )

        # Test 8: get_package_tree
        await self._test_method(
            "get_package_tree", self._test_get_package_tree, "Get package hierarchy"
        )

        # Test 9: get_package_dependencies
        await self._test_method(
            "get_package_dependencies",
            self._test_get_package_dependencies,
            "Analyze package dependencies",
        )

        # Test 10: find_circular_dependencies
        await self._test_method(
            "find_circular_dependencies",
            self._test_find_circular_dependencies,
            "Detect circular imports",
        )

        # Test 11: get_package_coupling_metrics
        await self._test_method(
            "get_package_coupling_metrics",
            self._test_get_package_coupling,
            "Calculate coupling metrics",
        )

    async def test_migration_intelligence(self):
        """Test migration intelligence methods."""
        print("\n### MIGRATION INTELLIGENCE METHODS ###\n")

        # Test 12: analyze_migration_readiness
        await self._test_method(
            "analyze_migration_readiness",
            self._test_analyze_migration,
            "Analyze migration opportunities",
        )

        # Test 13: create_migration_plan
        await self._test_method(
            "create_migration_plan",
            self._test_create_migration_plan,
            "Create migration strategy",
        )

        # Test 14: search_migration_patterns
        await self._test_method(
            "search_migration_patterns",
            self._test_search_patterns,
            "Search pattern library",
        )

        # Test 15: get_pattern_library_stats
        await self._test_method(
            "get_pattern_library_stats",
            self._test_pattern_stats,
            "Pattern library statistics",
        )

    async def test_repository_management(self):
        """Test repository management methods."""
        print("\n### REPOSITORY MANAGEMENT METHODS ###\n")

        # Test 16: health_check
        await self._test_method(
            "health_check", self._test_health_check, "Server health status"
        )

    async def _test_method(self, name: str, test_func, description: str):
        """Test a single method and record results."""
        print(f"Testing {name}: {description}")
        try:
            result = await test_func()
            if result.get("success", True):
                self.results["passed"].append(name)
                print(f"  ✅ PASSED: {result.get('message', 'Success')}")
                if result.get("details"):
                    print(f"     Details: {result['details']}")
            else:
                self.results["failed"].append(name)
                print(f"  ❌ FAILED: {result.get('message', 'Unknown error')}")
        except (ImportError, AttributeError, RuntimeError, ValueError) as e:
            self.results["failed"].append(name)
            print(f"  ❌ ERROR: {e!s}")
        print()

    # Individual test implementations

    async def _test_find_definition(self):
        async with self.session_factory() as session:
            from src.query.symbol_finder import SymbolFinder

            finder = SymbolFinder(session)
            results = await finder.find_definitions(
                "TreeSitterParser", entity_type="class"
            )

            return {
                "success": len(results) > 0,
                "message": f"Found {len(results)} definitions",
                "details": results[0]["file_path"] if results else None,
            }

    async def _test_get_code_structure(self):
        async with self.session_factory() as session:
            from src.database.models import File
            from src.scanner.code_processor import CodeProcessor

            # Get a sample file
            result = await session.execute(
                select(File)
                .where(
                    File.repository_id == self.repository_id, File.path.like("%parser%")
                )
                .limit(1)
            )
            file = result.scalar_one_or_none()

            if not file:
                return {"success": False, "message": "No suitable file found"}

            processor = CodeProcessor(session)
            structure = await processor.get_file_structure(file)

            return {
                "success": True,
                "message": f"Analyzed {file.path}",
                "details": f"{len(structure.get('classes', []))} classes, {len(structure.get('functions', []))} functions",
            }

    async def _test_search_code(self):
        async with self.session_factory() as session:
            from src.query.search_engine import SearchEngine

            search = SearchEngine(session)
            results = await search.search("TreeSitter parser", limit=5)

            return {
                "success": True,
                "message": f"Found {len(results)} results",
                "details": "Note: Results may be empty without embeddings",
            }

    async def _test_find_usage(self):
        async with self.session_factory() as session:
            from src.mcp_server.tools.find import FindTool

            tool = FindTool(session)
            try:
                results = await tool.find_usage("TreeSitterParser", None)
                return {
                    "success": True,
                    "message": f"Found {len(results)} usage locations",
                }
            except (ImportError, AttributeError, RuntimeError, ValueError) as e:
                return {"success": False, "message": f"Method needs fixing: {e!s}"}

    async def _test_suggest_refactoring(self):
        async with self.session_factory() as session:
            from src.mcp_server.tools.analyze import AnalyzeTool

            tool = AnalyzeTool(session)
            try:
                results = await tool.suggest_refactoring(
                    "src/parser/treesitter_parser.py", None
                )
                return {
                    "success": True,
                    "message": f"Generated {len(results)} suggestions",
                }
            except (ImportError, AttributeError, RuntimeError, ValueError) as e:
                return {"success": False, "message": f"Requires LLM client: {e!s}"}

    async def _test_find_similar_code(self):
        async with self.session_factory() as session:
            from src.query.search_engine import SearchEngine

            search = SearchEngine(session)
            results = await search.search_similar_code("class Parser:", limit=5)

            return {
                "success": True,
                "message": f"Found {len(results)} similar patterns",
                "details": "Requires embeddings for best results",
            }

    async def _test_analyze_packages(self):
        async with self.session_factory() as session:
            from src.scanner.package_analyzer import PackageAnalyzer

            analyzer = PackageAnalyzer(session, self.repository_id)
            result = await analyzer.analyze_packages()

            return {
                "success": True,
                "message": f"Found {result['packages_found']} packages",
                "details": f"Total files: {result['total_files']}",
            }

    async def _test_get_package_tree(self):
        async with self.session_factory() as session:
            from src.database.package_repository import PackageRepository

            repo = PackageRepository(session)
            tree = await repo.get_package_tree(self.repository_id)

            return {
                "success": "root" in tree,
                "message": f"Package tree with {len(tree.get('root', {}).get('children', []))} root packages",
            }

    async def _test_get_package_dependencies(self):
        async with self.session_factory() as session:
            from src.database.package_repository import PackageRepository

            repo = PackageRepository(session)
            packages = await repo.get_repository_packages(self.repository_id)

            if not packages:
                return {"success": False, "message": "No packages found"}

            deps = await repo.get_package_dependencies(packages[0].id, "both")

            return {
                "success": True,
                "message": f"Analyzed dependencies for {packages[0].path} ({len(deps)} dependencies)",
            }

    async def _test_find_circular_dependencies(self):
        async with self.session_factory() as session:
            from src.database.package_repository import PackageRepository

            repo = PackageRepository(session)
            circles = await repo.find_circular_dependencies(self.repository_id)

            return {
                "success": True,
                "message": f"Found {len(circles)} circular dependencies",
            }

    async def _test_get_package_coupling(self):
        async with self.session_factory() as session:
            from src.database.package_repository import PackageRepository

            repo = PackageRepository(session)
            metrics = await repo.get_coupling_metrics(self.repository_id)

            return {
                "success": True,
                "message": f"Calculated coupling for {len(metrics)} package pairs",
            }

    async def _test_analyze_migration(self):
        async with self.session_factory() as session:
            from src.services.migration_analyzer import MigrationAnalyzer

            analyzer = MigrationAnalyzer(session)
            try:
                analysis = await analyzer.analyze_repository_for_migration(
                    self.repository_id
                )

                return {
                    "success": True,
                    "message": f"Found {len(analysis['bounded_contexts'])} contexts, {len(analysis['migration_candidates'])} candidates",
                    "details": "Note: May be empty without domain analysis",
                }
            except (ImportError, AttributeError, RuntimeError, ValueError) as e:
                return {"success": False, "message": f"Analysis error: {e!s}"}

    async def _test_create_migration_plan(self):
        async with self.session_factory() as session:
            from src.services.migration_planner import MigrationPlanner

            planner = MigrationPlanner(session)
            try:
                plan = await planner.create_migration_plan(
                    repository_id=self.repository_id,
                    name=f"Test Plan for {self.repository_url}",
                    strategy=MigrationStrategy.GRADUAL,
                    target_architecture="modular_monolith",
                    risk_tolerance="medium",
                    team_size=5,
                )

                return {
                    "success": True,
                    "message": f"Created plan with {len(plan.steps)} steps",
                }
            except (ImportError, AttributeError, RuntimeError, ValueError) as e:
                return {"success": False, "message": f"Planning error: {e!s}"}

    async def _test_search_patterns(self):
        async with self.session_factory() as session:
            from src.services.pattern_library import PatternLibrary

            library = PatternLibrary(session)
            patterns = await library.search_patterns(min_success_rate=0.0)

            return {
                "success": True,
                "message": f"Found {len(patterns)} patterns in library",
            }

    async def _test_pattern_stats(self):
        async with self.session_factory() as session:
            from src.services.pattern_library import PatternLibrary

            library = PatternLibrary(session)
            stats = await library.get_library_statistics()

            return {
                "success": True,
                "message": f"Library has {stats['total_patterns']} patterns",
                "details": f"Categories: {len(stats['categories'])}",
            }

    async def _test_health_check(self):
        async with self.session_factory() as session:
            from sqlalchemy import select

            # Simple health check
            result = await session.execute(select(1))
            db_ok = result.scalar() == 1

            return {
                "success": db_ok,
                "message": (
                    "Database connection healthy"
                    if db_ok
                    else "Database connection failed"
                ),
            }

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 80)
        print("### TEST SUMMARY ###")
        print("=" * 80)

        total = len(self.results["passed"]) + len(self.results["failed"])
        passed_pct = (len(self.results["passed"]) / total * 100) if total > 0 else 0

        print(f"\nTotal tests: {total}")
        print(f"Passed: {len(self.results['passed'])} ({passed_pct:.1f}%)")
        print(f"Failed: {len(self.results['failed'])} ({100-passed_pct:.1f}%)")

        if self.results["passed"]:
            print(f"\n✅ PASSED ({len(self.results['passed'])}):")
            for method in self.results["passed"]:
                print(f"  - {method}")

        if self.results["failed"]:
            print(f"\n❌ FAILED ({len(self.results['failed'])}):")
            for method in self.results["failed"]:
                print(f"  - {method}")

        print("\n### RECOMMENDATIONS ###")
        print("1. Generate embeddings for better search results:")
        print("   python scripts/generate_embeddings.py 1")
        print("2. Run domain analysis for migration features:")
        print("   docker exec mcp-indexer python -m src.domain.indexer")
        print("3. Add LLM client configuration for AI features")


async def main():
    """Run all tests."""
    tester = MCPMethodTester()

    try:
        await tester.setup()

        # Run all test categories
        await tester.test_core_analysis()
        await tester.test_package_analysis()
        await tester.test_migration_intelligence()
        await tester.test_repository_management()

        # Print summary
        tester.print_summary()

    except Exception as e:
        logger.exception("Test failed")
        print(f"\n❌ Test suite failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
