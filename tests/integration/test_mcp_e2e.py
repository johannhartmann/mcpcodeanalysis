"""End-to-end integration tests for MCP server."""

import tempfile
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import select

from src.database.init_db import get_session_factory, init_database
from src.database.models import Class, File, Function, Module
from src.mcp_server.tools.code_analysis import (
    AnalyzeFileRequest,
    CodeAnalysisTools,
    GetCodeRequest,
)
from src.mcp_server.tools.code_search import (
    CodeSearchTools,
)
from src.mcp_server.tools.repository_management import (
    AddRepositoryRequest,
    RepositoryManagementTools,
    ScanRepositoryRequest,
)


class TestMCPEndToEnd:
    """End-to-end integration tests for MCP server functionality."""

    @pytest_asyncio.fixture
    async def test_db_engine(self):
        """Create a test database engine."""
        # Use SQLite for tests to avoid PostgreSQL dependency
        db_url = "sqlite+aiosqlite:///:memory:"
        engine = await init_database(db_url)
        yield engine
        await engine.dispose()

    @pytest_asyncio.fixture
    async def db_session(self, test_db_engine):
        """Create a test database session."""
        factory = get_session_factory(test_db_engine)
        async with factory() as session:
            yield session

    @pytest_asyncio.fixture
    async def openai_client(self) -> None:
        """Create OpenAI client (mocked for tests)."""
        # In real tests, you might want to mock this
        # For now, return None to skip embeddings
        return

    @pytest_asyncio.fixture
    async def temp_repo_dir(self):
        """Create a temporary directory for test repositories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest_asyncio.fixture
    async def sample_repo(self, temp_repo_dir):
        """Create a sample repository for testing."""
        repo_path = temp_repo_dir / "test_repo"
        repo_path.mkdir()

        # Create a simple Python project structure
        (repo_path / "src").mkdir()
        (repo_path / "tests").mkdir()

        # Create main.py
        main_py = repo_path / "src" / "main.py"
        main_py.write_text(
            '''
"""Main module for test application."""

class Calculator:
    """A simple calculator class."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b


def main():
    """Main function."""
    calc = Calculator()
    result = calc.add(5, 3)
    print(f"5 + 3 = {result}")


if __name__ == "__main__":
    main()
''',
        )

        # Create utils.py
        utils_py = repo_path / "src" / "utils.py"
        utils_py.write_text(
            '''
"""Utility functions."""

def format_number(value: float, decimals: int = 2) -> str:
    """Format a number with specified decimal places."""
    return f"{value:.{decimals}f}"


def parse_config(config_str: str) -> dict:
    """Parse configuration string."""
    result = {}
    for line in config_str.strip().split("\\n"):
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result
''',
        )

        # Create test file
        test_file = repo_path / "tests" / "test_main.py"
        test_file.write_text(
            '''
"""Tests for main module."""

from src.main import Calculator


def test_calculator_add():
    """Test calculator addition."""
    calc = Calculator()
    assert calc.add(2, 3) == 5
    assert calc.add(-1, 1) == 0


def test_calculator_multiply():
    """Test calculator multiplication."""
    calc = Calculator()
    assert calc.multiply(3, 4) == 12
    assert calc.multiply(0, 5) == 0
''',
        )

        # Initialize git repo
        import subprocess

        subprocess.run(["git", "init"], cwd=repo_path, check=True)
        subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=repo_path,
            check=True,
        )

        return repo_path

    @pytest.mark.asyncio
    async def test_full_workflow(self, db_session, openai_client, sample_repo) -> None:
        """Test the complete workflow: add repo -> scan -> search -> analyze."""
        from fastmcp import FastMCP

        # Create MCP instance and tools
        mcp = FastMCP("Test MCP")

        # Initialize repository management tools
        repo_tools = RepositoryManagementTools(db_session, openai_client, mcp)
        await repo_tools.register_tools()

        # Step 1: Add repository
        add_request = AddRepositoryRequest(
            url=f"file://{sample_repo}",  # Use file:// URL for local repo
            scan_immediately=True,
            generate_embeddings=False,  # Skip embeddings for test
        )

        # Manually call the tool function
        add_result = await repo_tools.add_repository(add_request)
        assert add_result["success"] is True
        assert "repository_id" in add_result
        repo_id = add_result["repository_id"]

        # Step 2: Verify repository was added
        list_result = await repo_tools.list_repositories(include_stats=True)
        assert list_result["success"] is True
        assert list_result["count"] == 1
        assert list_result["repositories"][0]["id"] == repo_id

        # Step 3: Check that files were scanned
        files_result = await db_session.execute(
            select(File).where(File.repository_id == repo_id),
        )
        files = files_result.scalars().all()
        assert len(files) > 0

        # Find main.py
        main_file = next((f for f in files if f.path.endswith("main.py")), None)
        assert main_file is not None

        # Step 4: Check that code was parsed
        modules_result = await db_session.execute(
            select(Module).where(Module.file_id == main_file.id),
        )
        modules = modules_result.scalars().all()
        assert len(modules) > 0

        # Check classes were extracted
        classes_result = await db_session.execute(
            select(Class).where(Class.module_id == modules[0].id),
        )
        classes = classes_result.scalars().all()
        assert len(classes) == 1
        assert classes[0].name == "Calculator"

        # Check functions were extracted
        functions_result = await db_session.execute(
            select(Function).where(Function.module_id == modules[0].id),
        )
        functions = functions_result.scalars().all()
        assert len(functions) == 1  # main function
        assert functions[0].name == "main"

        # Step 5: Test code analysis tools
        analysis_tools = CodeAnalysisTools(db_session, mcp)
        await analysis_tools.register_tools()

        # Get code for a function
        get_code_request = GetCodeRequest(
            entity_type="class",
            entity_id=classes[0].id,
            include_context=True,
        )
        code_result = await analysis_tools.get_code(get_code_request)
        assert code_result["success"] is True
        assert "Calculator" in code_result["code"]
        assert code_result["entity"]["name"] == "Calculator"

        # Analyze file
        analyze_request = AnalyzeFileRequest(
            file_path="src/main.py",
            repository_id=repo_id,
        )
        analysis_result = await analysis_tools.analyze_file(analyze_request)
        assert analysis_result["success"] is True
        assert analysis_result["analysis"]["classes"] == 1
        assert analysis_result["analysis"]["functions"] == 1
        assert "Calculator" in [
            c["name"] for c in analysis_result["analysis"]["class_list"]
        ]

    @pytest.mark.asyncio
    async def test_incremental_scanning(
        self,
        db_session,
        openai_client,
        sample_repo,
    ) -> None:
        """Test incremental scanning after file changes."""
        import subprocess

        from fastmcp import FastMCP

        mcp = FastMCP("Test MCP")
        repo_tools = RepositoryManagementTools(db_session, openai_client, mcp)
        await repo_tools.register_tools()

        # Add repository
        add_request = AddRepositoryRequest(
            url=f"file://{sample_repo}",
            scan_immediately=True,
            generate_embeddings=False,
        )
        add_result = await repo_tools.add_repository(add_request)
        repo_id = add_result["repository_id"]

        # Get initial file count
        initial_files = await db_session.execute(
            select(File).where(File.repository_id == repo_id),
        )
        initial_count = len(initial_files.scalars().all())

        # Add a new file
        new_file = sample_repo / "src" / "helpers.py"
        new_file.write_text(
            '''
"""Helper functions."""

def greet(name: str) -> str:
    """Generate a greeting."""
    return f"Hello, {name}!"
''',
        )

        # Commit the change
        subprocess.run(["git", "add", "."], cwd=sample_repo, check=True)
        subprocess.run(
            ["git", "commit", "-m", "Add helpers"],
            cwd=sample_repo,
            check=True,
        )

        # Rescan repository
        scan_request = ScanRepositoryRequest(
            repository_id=repo_id,
            force_full_scan=False,  # Incremental scan
            generate_embeddings=False,
        )
        scan_result = await repo_tools.scan_repository(scan_request)
        assert scan_result["success"] is True

        # Check that new file was added
        new_files = await db_session.execute(
            select(File).where(File.repository_id == repo_id),
        )
        new_count = len(new_files.scalars().all())
        assert new_count == initial_count + 1

        # Verify the new file was parsed
        helper_file = next(
            (f for f in new_files.scalars().all() if f.path.endswith("helpers.py")),
            None,
        )
        assert helper_file is not None

    @pytest.mark.asyncio
    async def test_search_functionality(
        self,
        db_session,
        openai_client,
        sample_repo,
    ) -> None:
        """Test code search functionality (without embeddings)."""
        from fastmcp import FastMCP

        mcp = FastMCP("Test MCP")

        # Setup repository
        repo_tools = RepositoryManagementTools(db_session, openai_client, mcp)
        await repo_tools.register_tools()

        add_request = AddRepositoryRequest(
            url=f"file://{sample_repo}",
            scan_immediately=True,
            generate_embeddings=False,
        )
        add_result = await repo_tools.add_repository(add_request)
        repo_id = add_result["repository_id"]

        # Initialize search tools
        search_tools = CodeSearchTools(db_session, openai_client, mcp)
        await search_tools.register_tools()

        # Test keyword search
        keyword_results = await search_tools.keyword_search(
            {
                "keywords": ["Calculator", "add"],
                "scope": "all",
                "repository_id": repo_id,
                "limit": 10,
            },
        )

        assert keyword_results["success"] is True
        assert len(keyword_results["results"]) > 0

        # Should find the Calculator class
        calc_results = [
            r for r in keyword_results["results"] if r["entity"]["name"] == "Calculator"
        ]
        assert len(calc_results) > 0
        assert calc_results[0]["entity"]["type"] == "class"

    @pytest.mark.asyncio
    async def test_error_handling(self, db_session, openai_client) -> None:
        """Test error handling in various scenarios."""
        from fastmcp import FastMCP

        mcp = FastMCP("Test MCP")
        repo_tools = RepositoryManagementTools(db_session, openai_client, mcp)
        await repo_tools.register_tools()

        # Test adding invalid repository
        add_request = AddRepositoryRequest(
            url="https://github.com/nonexistent/repo",
            scan_immediately=True,
            generate_embeddings=False,
        )

        # This should fail gracefully
        add_result = await repo_tools.add_repository(add_request)
        assert add_result["success"] is False
        assert "error" in add_result

        # Test listing when no repositories exist
        list_result = await repo_tools.list_repositories()
        assert list_result["success"] is True
        assert list_result["count"] == 0

        # Test scanning non-existent repository
        scan_request = ScanRepositoryRequest(
            repository_id=999,  # Non-existent ID
            force_full_scan=True,
            generate_embeddings=False,
        )
        scan_result = await repo_tools.scan_repository(scan_request)
        assert scan_result["success"] is False
        assert "not found" in scan_result["error"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
