#!/usr/bin/env python3
"""Simple test to check tool registration without database."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logger import get_logger
from src.mcp_server.server import mcp

logger = get_logger(__name__)


def test_initial_tools():
    """Test tools registered before database initialization."""
    # Get tools registered directly on mcp
    tools = mcp.tools
    tool_names = [tool.name for tool in tools]

    logger.info(f"Found {len(tools)} initially registered tools:")
    for tool in tools:
        logger.info(f"  - {tool.name}: {tool.description}")

    # Expected tools in server.py
    expected_server_tools = [
        "create_migration_plan",
        "start_migration_step",
        "complete_migration_step",
        "extract_migration_patterns",
        "analyze_packages",
    ]

    logger.info("\n=== Checking server.py tools ===")
    for tool_name in expected_server_tools:
        if tool_name in tool_names:
            logger.info(f"✓ {tool_name}")
        else:
            logger.error(f"✗ {tool_name} - MISSING")

    # Check resources
    resources = mcp.resources
    logger.info(f"\n=== Resources ===")
    logger.info(f"Found {len(resources)} resources")

    # Group by prefix
    resource_prefixes = set()
    for resource in resources:
        prefix = resource.uri.split("://")[0]
        resource_prefixes.add(prefix)

    logger.info(f"Resource types: {', '.join(sorted(resource_prefixes))}")


if __name__ == "__main__":
    test_initial_tools()
