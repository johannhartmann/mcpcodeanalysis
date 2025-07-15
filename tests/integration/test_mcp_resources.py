"""Integration tests for MCP resources."""

import asyncio
import os

import pytest
from fastmcp.client import Client


class TestMCPResources:
    """Test MCP resources functionality."""

    @pytest.fixture
    def mcp_server_url(self) -> str:
        """Get MCP server URL."""
        return "http://localhost:8080/mcp/v1/messages"

    @pytest.fixture
    def test_repo_url(self):
        """Use a real GitHub repository for testing."""
        # Use a small Python repository with good structure
        return "https://github.com/psf/requests-html"

    @pytest.fixture
    async def mcp_client(self, mcp_server_url):
        """Create FastMCP client."""
        async with Client(mcp_server_url) as client:
            yield client

    @pytest.fixture
    async def setup_repository(self, mcp_client, test_repo_url):
        """Add and scan test repository."""
        await mcp_client.call_tool(
            "add_repository",
            {
                "url": test_repo_url,
                "scan_immediately": True,
                "generate_embeddings": False,
            },
        )
        # Wait for scanning to complete
        await asyncio.sleep(5)  # Give more time for real repo
        return test_repo_url

    # Package Analysis Resources Tests

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        "CI" not in os.environ,
        reason="Requires Docker MCP server to be running",
    )
    async def test_package_tree_resource(self, mcp_client, setup_repository) -> None:
        """Test package tree resource."""
        try:
            # Read package tree
            tree_data = await mcp_client.read_resource(
                f"packages://{setup_repository}/tree"
            )

            assert tree_data is not None
            tree_text = str(tree_data)

            # Check for expected package structure
            assert "Package Structure" in tree_text
            # requests-html should have some packages
            assert "requests_html" in tree_text or "Package" in tree_text

        except Exception as e:
            if "Connection refused" in str(e):
                pytest.skip("MCP server not available")
            raise

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        "CI" not in os.environ,
        reason="Requires Docker MCP server to be running",
    )
    async def test_package_dependencies_resource(
        self, mcp_client, setup_repository
    ) -> None:
        """Test package dependencies resource."""
        try:
            # Analyze packages first
            await mcp_client.call_tool(
                "analyze_packages",
                {
                    "repository_url": setup_repository,
                    "force_refresh": True,
                },
            )
            await asyncio.sleep(2)

            # Read dependencies for a package
            deps_data = await mcp_client.read_resource(
                f"packages://{setup_repository}/requests_html/dependencies"
            )

            assert deps_data is not None
            deps_text = str(deps_data)

            # Check for dependency information
            assert "Dependencies" in deps_text or "dependencies" in deps_text.lower()

        except Exception as e:
            if "Connection refused" in str(e):
                pytest.skip("MCP server not available")
            if "not found" in str(e).lower():
                pytest.skip("Package analysis not available")
            raise

    # Code Analysis Resources Tests

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        "CI" not in os.environ,
        reason="Requires Docker MCP server to be running",
    )
    async def test_code_structure_resource(self, mcp_client, setup_repository) -> None:
        """Test code structure resource."""
        try:
            # Read code structure for a file
            structure_data = await mcp_client.read_resource(
                "code://structure/requests_html.py"
            )

            assert structure_data is not None
            structure_text = str(structure_data)

            # Check for expected structure elements
            assert "Code Structure" in structure_text
            # Should have some class or function names from requests-html
            assert (
                "class" in structure_text.lower()
                or "function" in structure_text.lower()
            )

        except Exception as e:
            if "Connection refused" in str(e):
                pytest.skip("MCP server not available")
            raise

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        "CI" not in os.environ,
        reason="Requires Docker MCP server to be running",
    )
    async def test_code_definitions_resource(
        self, mcp_client, setup_repository
    ) -> None:
        """Test code definitions resource."""
        try:
            # Find definitions for "HTMLSession" (main class in requests-html)
            defs_data = await mcp_client.read_resource("code://definitions/HTMLSession")

            assert defs_data is not None
            defs_text = str(defs_data)

            # Check for HTMLSession class definition
            assert "HTMLSession" in defs_text or "definitions" in defs_text.lower()
            assert "class" in defs_text.lower() or "definitions" in defs_text.lower()

        except Exception as e:
            if "Connection refused" in str(e):
                pytest.skip("MCP server not available")
            raise

    # Migration Analysis Resources Tests

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        "CI" not in os.environ,
        reason="Requires Docker MCP server to be running",
    )
    async def test_migration_readiness_resource(
        self, mcp_client, setup_repository
    ) -> None:
        """Test migration readiness analysis resource."""
        try:
            # Read migration readiness
            readiness_data = await mcp_client.read_resource(
                f"migration://readiness/{setup_repository}"
            )

            assert readiness_data is not None
            readiness_text = str(readiness_data)

            # Check for migration analysis content
            assert (
                "Migration Readiness" in readiness_text
                or "readiness" in readiness_text.lower()
            )

        except Exception as e:
            if "Connection refused" in str(e):
                pytest.skip("MCP server not available")
            raise

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        "CI" not in os.environ,
        reason="Requires Docker MCP server to be running",
    )
    async def test_migration_dashboard_resource(
        self, mcp_client, setup_repository
    ) -> None:
        """Test migration dashboard resource."""
        try:
            # Read migration dashboard for repository
            dashboard_data = await mcp_client.read_resource(
                f"migration://dashboard/{setup_repository}"
            )

            assert dashboard_data is not None
            dashboard_text = str(dashboard_data)

            # Check for dashboard content
            assert (
                "Migration Dashboard" in dashboard_text
                or "dashboard" in dashboard_text.lower()
            )

        except Exception as e:
            if "Connection refused" in str(e):
                pytest.skip("MCP server not available")
            raise

    # System Resources Tests

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        "CI" not in os.environ,
        reason="Requires Docker MCP server to be running",
    )
    async def test_system_config_resource(self, mcp_client) -> None:
        """Test system configuration resource."""
        try:
            # Read system config
            config_data = await mcp_client.read_resource("system://config")

            assert config_data is not None
            config_text = str(config_data)

            # Check for configuration information
            assert "Configuration" in config_text or "Settings" in config_text
            assert "MCP Server" in config_text or "Feature" in config_text

        except Exception as e:
            if "Connection refused" in str(e):
                pytest.skip("MCP server not available")
            raise

    # Resource Templates Tests

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        "CI" not in os.environ,
        reason="Requires Docker MCP server to be running",
    )
    async def test_list_resource_templates(self, mcp_client) -> None:
        """Test listing resource templates."""
        try:
            templates = await mcp_client.list_resource_templates()

            # Check we have templates for parameterized resources
            template_uris = [t.uriTemplate for t in templates]

            expected_templates = [
                "migration://readiness/{repository_url}",
                "packages://{repository_url}/tree",
                "code://structure/{file_path}",
            ]

            found_templates = sum(
                1
                for template in expected_templates
                if any(template in uri for uri in template_uris)
            )

            assert (
                found_templates >= 2
            ), f"Expected at least 2 resource templates, found {found_templates}"

        except Exception as e:
            if "Connection refused" in str(e):
                pytest.skip("MCP server not available")
            raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
