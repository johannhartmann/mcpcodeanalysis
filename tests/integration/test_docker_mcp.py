"""Integration tests for MCP server running in Docker."""

import asyncio
import json
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest


class TestDockerMCPIntegration:
    """Test MCP server running in Docker containers."""

    @pytest.fixture
    def mcp_server_url(self) -> str:
        """Get MCP server URL."""
        return "http://localhost:8080/mcp/"

    @pytest.fixture
    def test_repo_path(self) -> Iterator[Path]:
        """Create a test Git repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir) / "test_repo"
            repo_path.mkdir()

            # Create a simple Python file
            (repo_path / "main.py").write_text(
                '''
"""Test module."""

def hello_world():
    """Print hello world."""
    print("Hello, World!")

class Calculator:
    """Simple calculator."""

    def add(self, a, b):
        """Add two numbers."""
        return a + b
''',
            )

            # Initialize git repo (config necessary for github ci)
            subprocess.run(["git", "init"], cwd=repo_path, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@invalid"],
                cwd=repo_path,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "test"], cwd=repo_path, check=True
            )
            subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
            subprocess.run(
                ["git", "commit", "-m", "Initial commit"],
                cwd=repo_path,
                check=True,
            )

            yield repo_path

    async def send_mcp_request(
        self, url: str, method: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Send an MCP request to the server."""

        initialization_request_data = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0.0.1"},
            },
        }

        initialized_request_data = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }

        request_data = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": 2,
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

        async with httpx.AsyncClient() as client:
            # Initial POST
            initialization_response = await client.post(
                url,
                json=initialization_request_data,
                headers=headers,
                timeout=30.0,
                follow_redirects=True,
            )

            # Capture session id if provided
            sid = initialization_response.headers.get("mcp-session-id")
            if sid:
                headers["mcp-session-id"] = sid

            await client.post(
                url,
                json=initialized_request_data,
                headers=headers,
                timeout=30.0,
                follow_redirects=True,
            )

            response = await client.post(
                url,
                json=request_data,
                headers=headers,
                timeout=30.0,
                follow_redirects=True,
            )

            # If initial succeeded, parse and return
            if response.status_code == 200:
                result: list[dict[str, Any]] = []
                async for line in response.aiter_lines():
                    raw = line.strip()
                    if not raw:
                        continue
                    if raw.startswith("data:"):
                        raw = raw[len("data:") :].strip()
                    if raw.startswith(":"):
                        continue
                    try:
                        data = json.loads(raw)
                        result.append(data)
                    except json.JSONDecodeError:
                        continue
                return result

            # No session id supplied and initial not 200 -> raise
            raise httpx.ConnectError(
                f"Request failed with status {response.status_code}: {response.text}"
            )

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_list_tools(self, mcp_server_url: str) -> None:
        """Test listing available tools."""
        result = await self.send_mcp_request(mcp_server_url, "tools/list")

        assert len(result) > 0
        response = result[-1]  # Get the final response

        assert "result" in response
        assert "tools" in response["result"]

        # Check that we have the expected tools
        tool_names = [tool["name"] for tool in response["result"]["tools"]]
        expected_tools = [
            "add_repository",
            "list_repositories",
            "scan_repository",
            "semantic_search",
            "keyword_search",
            "get_code",
            "analyze_file",
        ]

        for expected in expected_tools:
            assert expected in tool_names, f"Missing tool: {expected}"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_add_and_scan_repository(
        self,
        mcp_server_url: str,
        test_repo_path: Path,
    ) -> None:
        """Test adding and scanning a repository."""
        # First, list repositories to check initial state
        list_result = await self.send_mcp_request(
            mcp_server_url,
            "tools/call",
            {"name": "list_repositories", "arguments": {}},
        )

        list_response = list_result[-1]
        initial_count = len(
            list_response["result"]["content"][0]["text"].get("repositories", []),
        )

        # Add the test repository
        add_result = await self.send_mcp_request(
            mcp_server_url,
            "tools/call",
            {
                "name": "add_repository",
                "arguments": {
                    "url": f"file://{test_repo_path}",
                    "scan_immediately": True,
                    "generate_embeddings": False,
                },
            },
        )

        add_response = add_result[-1]
        assert add_response["result"]["content"][0]["text"]["success"] is True
        repo_id = add_response["result"]["content"][0]["text"]["repository_id"]

        # List repositories again to verify it was added
        list_result2 = await self.send_mcp_request(
            mcp_server_url,
            "tools/call",
            {"name": "list_repositories", "arguments": {"include_stats": True}},
        )

        list_response2 = list_result2[-1]
        final_count = list_response2["result"]["content"][0]["text"]["count"]
        assert final_count == initial_count + 1

        # Check that files were scanned
        repos = list_response2["result"]["content"][0]["text"]["repositories"]
        test_repo = next((r for r in repos if r["id"] == repo_id), None)
        assert test_repo is not None
        assert test_repo["stats"]["total_files"] > 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_search_functionality(
        self, mcp_server_url: str, test_repo_path: Path
    ) -> None:
        """Test search functionality after indexing."""
        # Add and scan repository
        add_result = await self.send_mcp_request(
            mcp_server_url,
            "tools/call",
            {
                "name": "add_repository",
                "arguments": {
                    "url": f"file://{test_repo_path}",
                    "scan_immediately": True,
                    "generate_embeddings": False,
                },
            },
        )

        add_response = add_result[-1]
        repo_id = add_response["result"]["content"][0]["text"]["repository_id"]

        # Wait a bit for indexing to complete
        await asyncio.sleep(2)

        # Search for the Calculator class
        search_result = await self.send_mcp_request(
            mcp_server_url,
            "tools/call",
            {
                "name": "keyword_search",
                "arguments": {
                    "keywords": ["Calculator"],
                    "scope": "all",
                    "repository_id": repo_id,
                    "limit": 10,
                },
            },
        )

        search_response = search_result[-1]
        assert search_response["result"]["content"][0]["text"]["success"] is True

        results = search_response["result"]["content"][0]["text"]["results"]
        assert len(results) > 0

        # Check that we found the Calculator class
        calculator_found = any(
            r["entity"]["name"] == "Calculator" and r["entity"]["type"] == "class"
            for r in results
        )
        assert calculator_found, "Calculator class not found in search results"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_code(self, mcp_server_url: str, test_repo_path: Path) -> None:
        """Test getting code for a specific entity."""
        # Add and scan repository
        add_result = await self.send_mcp_request(
            mcp_server_url,
            "tools/call",
            {
                "name": "add_repository",
                "arguments": {
                    "url": f"file://{test_repo_path}",
                    "scan_immediately": True,
                    "generate_embeddings": False,
                },
            },
        )

        repo_id = add_result[-1]["result"]["content"][0]["text"]["repository_id"]

        # Wait for indexing
        await asyncio.sleep(2)

        # Search for Calculator class to get its ID
        search_result = await self.send_mcp_request(
            mcp_server_url,
            "tools/call",
            {
                "name": "keyword_search",
                "arguments": {
                    "keywords": ["Calculator"],
                    "scope": "classes",
                    "repository_id": repo_id,
                    "limit": 10,
                },
            },
        )

        results = search_result[-1]["result"]["content"][0]["text"]["results"]
        calc_result = next(
            (r for r in results if r["entity"]["name"] == "Calculator"),
            None,
        )
        assert calc_result is not None

        entity_id = calc_result["entity"]["id"]

        # Get the code for the Calculator class
        code_result = await self.send_mcp_request(
            mcp_server_url,
            "tools/call",
            {
                "name": "get_code",
                "arguments": {
                    "entity_type": "class",
                    "entity_id": entity_id,
                    "include_context": True,
                },
            },
        )

        code_response = code_result[-1]
        assert code_response["result"]["content"][0]["text"]["success"] is True

        code = code_response["result"]["content"][0]["text"]["code"]
        assert "class Calculator:" in code
        assert "def add(" in code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
