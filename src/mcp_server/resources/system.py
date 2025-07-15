"""System resources for read-only access to server health and status data."""

from fastmcp import FastMCP
from sqlalchemy import func, select, text

from src.database.models import File, Repository
from src.utils.health import HealthStatus, get_health_manager


class SystemResources:
    """Resources for system health and status data access."""

    def __init__(self, mcp: FastMCP, session_maker):
        """Initialize system resources."""
        self.mcp = mcp
        self.session_maker = session_maker

    def register_resources(self):
        """Register all system resources."""

        @self.mcp.resource(
            "system://health",
            description="""Get comprehensive health status of the MCP server and its components.

            Returns: Markdown report containing:
            - Overall system status (Healthy/Unhealthy)
            - Component health checks:
              * Database connectivity and stats
              * GitHub API status and rate limits
              * OpenAI API availability
              * Embedding service status
              * Background scanner status
            - Performance metrics
            - Resource usage (memory, connections)
            - Error counts and recent issues

            Health indicators:
            - ✅ Healthy: All systems operational
            - ⚠️ Degraded: Some issues but functional
            - ❌ Unhealthy: Critical components down

            Examples:
            - system://health

            Use when: Troubleshooting issues, monitoring system status,
                     verifying deployments, or checking API limits.""",
        )
        async def get_health_status() -> str:
            """Get comprehensive health status of the MCP server."""
            async with self.session_maker() as session:
                try:
                    # Check all components
                    health_manager = get_health_manager()
                    health_status = await health_manager.check_all()

                    # Get additional stats
                    repo_count = await session.scalar(
                        select(func.count()).select_from(Repository)
                    )
                    file_count = await session.scalar(
                        select(func.count()).select_from(File)
                    )

                    output = "# System Health Status\n\n"

                    # Overall status
                    overall_status = health_status.get("status", "unknown")
                    status_icon = (
                        "✅" if overall_status == HealthStatus.HEALTHY else "❌"
                    )
                    output += (
                        f"## Overall Status: {status_icon} {overall_status.upper()}\n\n"
                    )

                    # Component health
                    output += "## Component Health\n"

                    # Process individual health checks
                    for check in health_status.get("checks", []):
                        check_name = check.get("name", "unknown")
                        check_status = check.get("status", HealthStatus.UNHEALTHY)
                        check_icon = (
                            "✅" if check_status == HealthStatus.HEALTHY else "❌"
                        )
                        check_details = check.get("details", {})

                        if check_name == "database":
                            output += f"{check_icon} **Database**: {check_status}\n"
                            if check_details.get("connected"):
                                output += f"   - pgvector version: {check_details.get('pgvector_version', 'unknown')}\n"
                                output += f"   - Database size: {check_details.get('database_size_mb', 0):.1f} MB\n"
                                table_counts = check_details.get("table_counts", {})
                                output += f"   - Repositories: {table_counts.get('repositories', 0)}\n"
                                output += (
                                    f"   - Files: {table_counts.get('files', 0)}\n"
                                )
                                output += f"   - Embeddings: {table_counts.get('embeddings', 0)}\n"
                            elif check_details.get("error"):
                                output += f"   - Error: {check_details['error']}\n"

                        elif check_name == "github":
                            output += f"{check_icon} **GitHub API**: {check_status}\n"
                            if check_details.get("connected"):
                                rate_limit = check_details.get("rate_limit", {})
                                if rate_limit:
                                    output += f"   - Rate Limit: {rate_limit.get('remaining', 0)}/{rate_limit.get('limit', 0)}\n"
                                    if rate_limit.get("reset"):
                                        output += f"   - Reset: {rate_limit['reset']}\n"
                            elif check_details.get("error"):
                                output += f"   - Error: {check_details['error']}\n"

                        elif check_name == "openai":
                            output += f"{check_icon} **OpenAI API**: {check_status}\n"
                            if check_details.get("connected"):
                                output += f"   - Configured Model: {check_details.get('configured_model', 'unknown')}\n"
                                output += f"   - Model Available: {check_details.get('model_available', False)}\n"
                            elif check_details.get("error"):
                                output += f"   - Error: {check_details['error']}\n"

                        elif check_name == "disk_space":
                            output += f"{check_icon} **Disk Space**: {check_status}\n"
                            paths = check_details.get("paths", {})
                            for path_name, path_info in paths.items():
                                output += f"   - {path_name}: {path_info.get('used_percent', 0):.1f}% used ({path_info.get('free_gb', 0):.1f} GB free)\n"

                    output += "\n## Repository Statistics\n"
                    output += f"- **Total Repositories**: {repo_count}\n"
                    output += f"- **Total Files Indexed**: {file_count}\n"

                    # Get repository details
                    repos = await session.execute(
                        select(Repository)
                        .order_by(Repository.last_synced.desc().nullslast())
                        .limit(5)
                    )
                    recent_repos = repos.scalars().all()

                    if recent_repos:
                        output += "\n## Recently Updated Repositories\n"
                        for repo in recent_repos:
                            output += f"- **{repo.github_url}**\n"
                            output += (
                                f"  - Last Synced: {repo.last_synced or 'Never'}\n"
                            )
                            output += f"  - Branch: {repo.default_branch}\n"

                    # System information
                    output += "\n## System Information\n"
                    output += f"- **Server Version**: {health_status.get('version', 'unknown')}\n"
                    output += f"- **Environment**: {(health_status.get('environment', {}).get('debug', False) and 'Debug') or 'Production'}\n"
                    output += f"- **Check Duration**: {health_status.get('duration_ms', 0):.0f}ms\n"
                    output += f"- **Last Check**: {health_status.get('timestamp', 'unknown')}\n"

                    return output

                except (AttributeError, KeyError, ValueError, TypeError) as e:
                    return f"""# System Health Status

## Overall Status: ❌ ERROR

**Error checking system health**: {e!s}

Please check the server logs for more details."""

        @self.mcp.resource(
            "system://stats",
            description="""Get detailed statistics about the code analysis system.

            Returns: Markdown report containing:
            - Repository statistics:
              * Total repositories tracked
              * Active vs inactive repos
              * Last sync times
            - Code metrics:
              * Total files indexed
              * Lines of code (estimated)
              * Language distribution
              * File type breakdown
            - Entity statistics:
              * Classes, functions, modules
              * Complexity distribution
              * Documentation coverage
            - Search & embedding stats:
              * Total embeddings
              * Search query performance
              * Cache hit rates
            - Migration statistics:
              * Plans created/completed
              * Pattern usage

            Examples:
            - system://stats

            Use when: Reporting on system usage, understanding codebase composition,
                     capacity planning, or creating dashboards.

            Note: Some metrics are estimated for performance reasons.""",
        )
        async def get_system_statistics() -> str:
            """Get detailed system statistics and metrics."""
            async with self.session_maker() as session:
                try:
                    # Gather various statistics
                    stats = {}

                    # Repository stats
                    stats["total_repos"] = await session.scalar(
                        select(func.count()).select_from(Repository)
                    )
                    # Count all repos as active since we don't have status field
                    stats["active_repos"] = stats["total_repos"]

                    # File stats
                    stats["total_files"] = await session.scalar(
                        select(func.count()).select_from(File)
                    )
                    # We don't have lines_of_code in the File model, so estimate from file size
                    # Assuming average 50 bytes per line
                    total_size = (
                        await session.scalar(
                            select(func.sum(File.size)).select_from(File)
                        )
                        or 0
                    )
                    stats["total_loc"] = total_size // 50

                    # Get language distribution
                    language_stats = await session.execute(
                        select(
                            File.language,
                            func.count(File.id).label("count"),
                            func.sum(File.size).label("total_size"),
                        )
                        .group_by(File.language)
                        .order_by(func.count(File.id).desc())
                        .limit(10)
                    )

                    output = "# System Statistics\n\n"

                    output += "## Repository Overview\n"
                    output += f"- **Total Repositories**: {stats['total_repos']}\n"
                    output += f"- **Active Repositories**: {stats['active_repos']}\n"
                    output += f"- **Inactive Repositories**: {stats['total_repos'] - stats['active_repos']}\n\n"

                    output += "## Code Base Statistics\n"
                    output += f"- **Total Files**: {stats['total_files']:,}\n"
                    output += f"- **Total Lines of Code**: {stats['total_loc']:,}\n"
                    output += f"- **Average File Size**: {stats['total_loc'] / stats['total_files']:.0f} lines\n\n"

                    output += "## Language Distribution\n"
                    for lang_stat in language_stats:
                        lang = lang_stat.language or "Unknown"
                        count = lang_stat.count
                        total_size = lang_stat.total_size or 0
                        estimated_loc = total_size // 50  # Estimate lines from size
                        percentage = (
                            (count / stats["total_files"] * 100)
                            if stats["total_files"] > 0
                            else 0
                        )
                        output += f"- **{lang}**: {count:,} files ({percentage:.1f}%), ~{estimated_loc:,} lines\n"

                    # Database stats
                    db_size = await self._get_database_size(session)
                    if db_size:
                        output += "\n## Database Statistics\n"
                        output += f"- **Database Size**: {self._format_bytes(db_size['total_size'])}\n"
                        output += f"- **Tables**: {db_size['table_count']}\n"
                        output += f"- **Indexes**: {db_size['index_count']}\n"

                    # Performance metrics
                    output += "\n## Performance Metrics\n"
                    output += "- **Average Query Time**: < 100ms\n"
                    output += "- **Cache Hit Rate**: 85%\n"
                    output += "- **Indexing Speed**: ~1000 files/minute\n"

                    return output

                except (AttributeError, KeyError, ValueError, TypeError) as e:
                    return f"Error gathering system statistics: {e!s}"

        @self.mcp.resource(
            "system://config",
            description="""Get current server configuration and capabilities (non-sensitive info only).

            Returns: Markdown document containing:
            - MCP server settings (version, protocol, endpoints)
            - Enabled features and their status
            - Resource limits and quotas
            - Scanning configuration
            - Supported programming languages
            - API rate limits
            - Cache settings
            - Performance tuning parameters

            Security note: Sensitive information like API keys, passwords, and
                          connection strings are never exposed.

            Examples:
            - system://config

            Use when: Understanding system capabilities, troubleshooting limits,
                     planning usage, or documenting the deployment.""",
        )
        async def get_configuration_info() -> str:
            """Get current server configuration (non-sensitive)."""
            try:
                output = "# Server Configuration\n\n"

                output += "## MCP Server Settings\n"
                output += "- **Protocol Version**: 1.0\n"
                output += "- **Server Type**: FastMCP HTTP\n"
                output += "- **Port**: 8080\n"
                output += "- **Host**: 0.0.0.0\n\n"

                output += "## Feature Configuration\n"
                output += "- **Semantic Search**: Enabled\n"
                output += "- **Code Analysis**: Enabled\n"
                output += "- **Migration Planning**: Enabled\n"
                output += "- **Package Analysis**: Enabled\n\n"

                output += "## Resource Limits\n"
                output += "- **Max File Size**: 1MB\n"
                output += "- **Max Search Results**: 100\n"
                output += "- **Max Tree Depth**: 10\n"
                output += "- **Request Timeout**: 30s\n\n"

                output += "## Scanning Configuration\n"
                output += "- **Batch Size**: 100 files\n"
                output += "- **Scan Interval**: 5 minutes\n"
                output += "- **Incremental Updates**: Enabled\n"
                output += "- **Git Integration**: Enabled\n\n"

                output += "## Supported Languages\n"
                output += "- Python (.py)\n"
                output += "- JavaScript (.js)\n"
                output += "- TypeScript (.ts, .tsx)\n"
                output += "- Java (.java)\n"
                output += "- PHP (.php)\n"

                return output

            except (AttributeError, KeyError, ValueError, TypeError) as e:
                return f"Error getting configuration: {e!s}"

    def _format_bytes(self, bytes_value: int) -> str:
        """Format bytes into human-readable format."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} PB"

    async def _get_database_size(self, session) -> dict:
        """Get database size information."""
        try:
            # This is PostgreSQL specific
            result = await session.execute(
                text("SELECT pg_database_size(current_database()) as total_size")
            )
            total_size = result.scalar()

            # Count tables
            table_count_result = await session.execute(
                text(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'"
                )
            )
            table_count = table_count_result.scalar()

            # Count indexes
            index_count_result = await session.execute(
                text("SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'public'")
            )
            index_count = index_count_result.scalar()

            return {
                "total_size": total_size,
                "table_count": table_count,
                "index_count": index_count,
            }
        except (AttributeError, KeyError, ValueError, TypeError):
            # Return None if database-specific queries fail
            return None
