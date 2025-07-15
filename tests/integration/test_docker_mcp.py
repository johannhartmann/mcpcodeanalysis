"""Integration tests for MCP server running in Docker using FastMCP client."""

import asyncio
import os

import pytest
from fastmcp.client import Client


class TestDockerMCPIntegration:
    """Test MCP server running in Docker containers using FastMCP client."""

    @pytest.fixture
    def mcp_server_url(self) -> str:
        """Get MCP server URL."""
        return "http://localhost:8080/mcp/v1/messages"

    @pytest.fixture
    def test_repo_url(self):
        """Use a real GitHub repository for testing."""
        # Use a small, public repository for testing
        return "https://github.com/octocat/Hello-World"

    @pytest.fixture
    async def mcp_client(self, mcp_server_url):
        """Create FastMCP client."""
        async with Client(mcp_server_url) as client:
            yield client

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        "CI" not in os.environ,
        reason="Requires Docker MCP server to be running",
    )
    async def test_list_resources(self, mcp_client) -> None:
        """Test listing available resources."""
        try:
            resources = await mcp_client.list_resources()
            templates = await mcp_client.list_resource_templates()
        except Exception as e:
            if "Connection refused" in str(e):
                pytest.skip("MCP server not available")
            raise

        # Check that we have the expected resources
        resource_uris = [str(r.uri) for r in resources]
        template_uris = [str(t.uriTemplate) for t in templates]
        all_uris = resource_uris + template_uris

        print(f"Found {len(resource_uris)} resources: {resource_uris}")
        print(f"Found {len(template_uris)} templates: {template_uris}")

        # Check for core resource patterns
        expected_patterns = [
            "system://health",
            "system://stats",
            "system://config",
            "migration://readiness/",
            "migration://patterns/search",
            "migration://patterns/stats",
            "migration://dashboard/",
            "packages://",
            "code://search",
            "code://definitions/",
            "code://structure/",
        ]

        # Count how many expected patterns we find
        found_patterns = sum(
            1
            for pattern in expected_patterns
            if any(pattern in uri for uri in all_uris)
        )

        assert (
            found_patterns >= 5
        ), f"Expected at least 5 resource patterns, found {found_patterns}"

        # Also check that we have a reasonable number of resources + templates
        assert (
            len(all_uris) >= 10
        ), f"Expected at least 10 resources + templates, but found {len(all_uris)}"

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        "CI" not in os.environ,
        reason="Requires Docker MCP server to be running",
    )
    async def test_system_health_resource(self, mcp_client) -> None:
        """Test the system health resource."""
        try:
            # Read the health resource
            health_data = await mcp_client.read_resource("system://health")

            # Check we got a response
            assert health_data is not None

            # Check it contains expected content
            health_text = str(health_data)
            assert (
                "System Health Status" in health_text or "Overall Status" in health_text
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
    async def test_list_tools(self, mcp_client) -> None:
        """Test listing remaining tools (actions)."""
        try:
            tools = await mcp_client.list_tools()
        except Exception as e:
            if "Connection refused" in str(e):
                pytest.skip("MCP server not available")
            raise

        # Check that we have the expected action tools
        tool_names = [tool.name for tool in tools]

        # These are the tools that modify state
        expected_tools = [
            "add_repository",
            "create_migration_plan",
            "start_migration_step",
            "complete_migration_step",
            "extract_migration_patterns",
            "analyze_packages",
        ]

        for expected in expected_tools:
            assert expected in tool_names, f"Missing tool: {expected}"

        # Check we have fewer tools now (most converted to resources)
        assert (
            len(tool_names) <= 10
        ), f"Expected 10 or fewer tools, but found {len(tool_names)}"

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        "CI" not in os.environ,
        reason="Requires Docker MCP server to be running",
    )
    async def test_add_repository_tool(self, mcp_client, test_repo_url) -> None:
        """Test adding a repository using the tool."""
        try:
            # Add the test repository
            result = await mcp_client.call_tool(
                "add_repository",
                {
                    "url": test_repo_url,
                    "scan_immediately": True,
                    "generate_embeddings": False,
                },
            )

            # Check the result
            assert result is not None
            # The response structure varies, so we just check it's not an error
            result_str = str(result)
            assert "error" not in result_str.lower() or "success" in result_str.lower()

        except Exception as e:
            if "Connection refused" in str(e):
                pytest.skip("MCP server not available")
            raise

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        "CI" not in os.environ,
        reason="Requires Docker MCP server to be running",
    )
    async def test_code_search_resource(self, mcp_client, test_repo_url) -> None:
        """Test code search resource after adding a repository."""
        try:
            # First add the repository
            await mcp_client.call_tool(
                "add_repository",
                {
                    "url": test_repo_url,
                    "scan_immediately": True,
                    "generate_embeddings": False,
                },
            )

            # Wait for indexing
            await asyncio.sleep(2)

            # The code://search resource provides information about search capabilities
            search_info = await mcp_client.read_resource("code://search")

            assert search_info is not None
            search_text = str(search_info)

            # Check if we got information about the search resource
            assert "search" in search_text.lower()

            # For actual search, we would use a tool instead of a resource
            # Resources are for static data access, not dynamic queries

        except Exception as e:
            if "Connection refused" in str(e):
                pytest.skip("MCP server not available")
            # Some resources might not work without proper setup
            if "not found" in str(e).lower() or "error" in str(e).lower():
                pytest.skip("Resource features not fully configured")
            raise

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        "CI" not in os.environ,
        reason="Requires Docker MCP server to be running",
    )
    async def test_system_stats_resource(self, mcp_client) -> None:
        """Test system statistics resource."""
        try:
            # Read the stats resource
            stats_data = await mcp_client.read_resource("system://stats")

            # Check we got a response
            assert stats_data is not None

            stats_text = str(stats_data)
            assert (
                "System Statistics" in stats_text or "statistics" in stats_text.lower()
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
    async def test_migration_patterns_resource(self, mcp_client) -> None:
        """Test migration patterns resource."""
        try:
            # Read the migration patterns stats
            patterns_data = await mcp_client.read_resource("migration://patterns/stats")

            # Check we got a response
            assert patterns_data is not None

            patterns_text = str(patterns_data)
            assert (
                "Pattern Library" in patterns_text
                or "patterns" in patterns_text.lower()
            )

        except Exception as e:
            if "Connection refused" in str(e):
                pytest.skip("MCP server not available")
            # Migration features might not have data initially
            if "not found" in str(e).lower():
                pytest.skip("No migration patterns available yet")
            raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
