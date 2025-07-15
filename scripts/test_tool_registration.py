#!/usr/bin/env python3
"""Test that all tools are properly registered in the MCP server."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logger import get_logger
from src.mcp_server.server import initialize_server, mcp

logger = get_logger(__name__)


async def test_tool_registration():
    """Test that all expected tools are registered."""
    # Initialize the server
    await initialize_server()

    # Get all registered tools
    tools = mcp.tools
    tool_names = [tool.name for tool in tools]

    logger.info(f"Found {len(tools)} registered tools")

    # Expected tools from each category
    expected_tools = {
        # Repository Management Tools
        "add_repository": "RepositoryManagementTools",
        "list_repositories": "RepositoryManagementTools",
        "scan_repository": "RepositoryManagementTools",
        "update_embeddings": "RepositoryManagementTools",
        "get_repository_stats": "RepositoryManagementTools",
        "delete_repository": "RepositoryManagementTools",
        # Domain Tools
        "extract_domain_model": "DomainTools",
        "find_aggregate_roots": "DomainTools",
        "analyze_bounded_context": "DomainTools",
        "suggest_ddd_refactoring": "DomainTools",
        "find_bounded_contexts": "DomainTools",
        "generate_context_map": "DomainTools",
        # Analysis Tools
        "analyze_coupling": "AnalysisTools",
        "suggest_context_splits": "AnalysisTools",
        "detect_anti_patterns": "AnalysisTools",
        "analyze_domain_evolution": "AnalysisTools",
        # Migration Tools (in server.py)
        "create_migration_plan": "server.py",
        "start_migration_step": "server.py",
        "complete_migration_step": "server.py",
        "extract_migration_patterns": "server.py",
        # Package Analysis (in server.py)
        "analyze_packages": "server.py",
    }

    # Check which tools are registered
    logger.info("\n=== Tool Registration Status ===")

    missing_tools = []
    for tool_name, source in expected_tools.items():
        if tool_name in tool_names:
            logger.info(f"✓ {tool_name} ({source})")
        else:
            logger.error(f"✗ {tool_name} ({source}) - MISSING")
            missing_tools.append(tool_name)

    # Check for unexpected tools
    unexpected_tools = []
    for tool_name in tool_names:
        if tool_name not in expected_tools:
            unexpected_tools.append(tool_name)
            logger.warning(f"? {tool_name} - UNEXPECTED")

    # Summary
    logger.info(f"\n=== Summary ===")
    logger.info(f"Total tools registered: {len(tools)}")
    logger.info(
        f"Expected tools found: {len(expected_tools) - len(missing_tools)}/{len(expected_tools)}"
    )

    if missing_tools:
        logger.error(f"Missing tools: {', '.join(missing_tools)}")

    if unexpected_tools:
        logger.warning(f"Unexpected tools: {', '.join(unexpected_tools)}")

    if not missing_tools:
        logger.info("✓ All expected tools are registered!")

    # List all resources too
    logger.info(f"\n=== Registered Resources ===")
    resources = mcp.resources
    logger.info(f"Total resources: {len(resources)}")

    # Group resources by prefix
    resource_groups = {}
    for resource in resources:
        prefix = resource.uri.split("://")[0]
        if prefix not in resource_groups:
            resource_groups[prefix] = []
        resource_groups[prefix].append(resource.uri)

    for prefix, uris in sorted(resource_groups.items()):
        logger.info(f"\n{prefix}:// ({len(uris)} resources)")
        for uri in sorted(uris)[:5]:  # Show first 5
            logger.info(f"  - {uri}")
        if len(uris) > 5:
            logger.info(f"  ... and {len(uris) - 5} more")


if __name__ == "__main__":
    asyncio.run(test_tool_registration())
