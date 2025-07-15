"""MCP Resources for read-only data access."""

from src.mcp_server.resources.code_analysis import CodeAnalysisResources
from src.mcp_server.resources.migration_analysis import MigrationAnalysisResources
from src.mcp_server.resources.package_analysis import PackageAnalysisResources
from src.mcp_server.resources.system import SystemResources

__all__ = [
    "MigrationAnalysisResources",
    "PackageAnalysisResources",
    "CodeAnalysisResources",
    "SystemResources",
]
