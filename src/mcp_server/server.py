"""MCP Code Analysis Server implementation - Fixed version."""

from collections.abc import AsyncGenerator
from typing import Any

from fastmcp import FastMCP
from pydantic import Field
from sqlalchemy import select

from src.database.init_db import get_session_factory, init_database
from src.database.models import Repository
from src.logger import get_logger

# Repository management tools handled via functions
from src.scanner.repository_scanner import RepositoryConfig

# Class-based tools are imported when needed
# Migration tools are imported below


logger = get_logger(__name__)

# Create FastMCP server
mcp = FastMCP("Code Analysis MCP Server")

# Global session factory
_session_factory = None


async def initialize_server() -> None:
    """Initialize the server and database."""
    global _session_factory

    if _session_factory is not None:
        return

    logger.info("Initializing server...")

    # Initialize database
    engine = await init_database()
    _session_factory = get_session_factory(engine)

    logger.info("Server initialized successfully")


async def get_db_session() -> AsyncGenerator[Any, None]:
    """Get a database session."""
    if _session_factory is None:
        await initialize_server()

    async with _session_factory() as session:
        yield session


# Register repository management tools
@mcp.tool(name="add_repository", description="Add a new repository to track")
async def add_repository(
    url: str = Field(description="Repository URL (GitHub or file://)"),
    branch: str = Field(default=None, description="Branch to track (optional)"),
    scan_immediately: bool = Field(
        default=True,
        description="Scan repository immediately",
    ),
    generate_embeddings: bool = Field(
        default=True,
        description="Generate embeddings for code",
    ),
) -> dict[str, Any]:
    """Add a new repository to track."""
    await initialize_server()

    async for session in get_db_session():
        try:
            # Check for existing repository
            result = await session.execute(
                select(Repository).where(Repository.github_url == url),
            )
            existing_repo = result.scalar_one_or_none()

            if existing_repo:
                return {
                    "success": False,
                    "error": f"Repository already exists with ID {existing_repo.id}",
                    "repository_id": existing_repo.id,
                }

            # Add repository to database
            repo = Repository(
                github_url=url,
                owner=url.split("/")[-2] if "/" in url else "local",
                name=url.split("/")[-1].replace(".git", "") if "/" in url else url,
                default_branch=branch or "main",
            )
            session.add(repo)
            await session.commit()
            await session.refresh(repo)

            scan_result = {"repository_id": repo.id}

            # Scan if requested
            if scan_immediately:
                from src.scanner.repository_scanner import RepositoryScanner

                repo_config = RepositoryConfig(
                    url=url,
                    branch=branch,
                )

                scanner = RepositoryScanner(session)
                scan_result: dict[str, Any] = await scanner.scan_repository(repo_config)

                # Generate embeddings if requested
                if generate_embeddings:
                    from src.embeddings.embedding_service import EmbeddingService

                    embedding_service = EmbeddingService(session)
                    embedding_result = (
                        await embedding_service.create_repository_embeddings(
                            scan_result["repository_id"],
                        )
                    )
                    scan_result["embeddings"] = embedding_result

            return {
                "success": True,
                "repository": {
                    "id": repo.id,
                    "url": url,
                    "branch": branch or "default",
                },
                "scan_result": scan_result,
            }
        except Exception as e:
            logger.exception("Failed to add repository: %s")
            return {
                "success": False,
                "error": str(e),
            }
    return None


# Initialize class-based tools on startup
# Migration tools will be registered via decorators below


# Migration analysis and planning tools
@mcp.tool(
    name="analyze_migration_readiness",
    description="Analyze a repository to assess migration readiness and identify opportunities",
)
async def analyze_migration_readiness(
    repository_url: str = Field(
        description="Repository URL to analyze for migration readiness"
    ),
) -> dict[str, Any]:
    """Analyze repository for migration readiness."""
    await initialize_server()

    async for session in get_db_session():
        try:
            # Get repository ID
            result = await session.execute(
                select(Repository).where(Repository.github_url == repository_url)
            )
            repository = result.scalar_one_or_none()

            if not repository:
                return {
                    "success": False,
                    "error": f"Repository {repository_url} not found. Please sync it first.",
                }

            # Create analyzer
            from src.services.migration_analyzer import MigrationAnalyzer

            analyzer = MigrationAnalyzer(session)

            # Perform analysis
            analysis = await analyzer.analyze_repository_for_migration(repository.id)

            # Format results
            result = {
                "repository": analysis["repository_name"],
                "analysis_summary": {
                    "bounded_contexts_found": len(analysis["bounded_contexts"]),
                    "migration_candidates": len(analysis["migration_candidates"]),
                    "complexity_rating": analysis["complexity_metrics"][
                        "complexity_rating"
                    ],
                    "recommended_strategy": analysis["recommended_strategy"][
                        "recommended_strategy"
                    ],
                },
                "top_contexts": analysis["bounded_contexts"][:5],  # Top 5 by readiness
                "top_candidates": analysis["migration_candidates"][
                    :10
                ],  # Top 10 candidates
                "dependency_issues": {
                    "circular_dependencies": len(
                        analysis["dependency_analysis"]["circular_dependencies"]
                    ),
                    "high_coupling_packages": len(
                        analysis["dependency_analysis"]["high_coupling_packages"]
                    ),
                    "bottlenecks": len(
                        analysis["dependency_analysis"]["dependency_bottlenecks"]
                    ),
                },
                "recommended_approach": analysis["recommended_strategy"],
                "complexity_metrics": analysis["complexity_metrics"],
            }

            return {
                "success": True,
                "message": "Migration readiness analysis completed successfully",
                "data": result,
            }

        except Exception as e:
            logger.exception("Failed to analyze repository")
            return {"success": False, "error": str(e)}
    return {}


@mcp.tool(
    name="create_migration_plan",
    description="Create a detailed migration plan for transforming a monolithic codebase",
)
async def create_migration_plan(
    repository_url: str = Field(description="Repository URL to create plan for"),
    plan_name: str = Field(description="Name for the migration plan"),
    strategy: str = Field(
        description="Migration strategy (strangler_fig, gradual, big_bang, branch_by_abstraction, parallel_run)",
        default="gradual",
    ),
    target_architecture: str = Field(
        description="Target architecture (modular_monolith, microservices, event_driven)",
        default="modular_monolith",
    ),
    team_size: int = Field(description="Available team size", default=5),
    timeline_weeks: int | None = Field(
        description="Desired timeline in weeks (optional)", default=None
    ),
    risk_tolerance: str = Field(
        description="Risk tolerance level (low, medium, high)", default="medium"
    ),
) -> dict[str, Any]:
    """Create a migration plan."""
    await initialize_server()

    async for session in get_db_session():
        try:
            # Get repository ID
            result = await session.execute(
                select(Repository).where(Repository.github_url == repository_url)
            )
            repository = result.scalar_one_or_none()

            if not repository:
                return {
                    "success": False,
                    "error": f"Repository {repository_url} not found. Please sync it first.",
                }

            # Validate strategy
            from src.database.migration_models import MigrationStrategy

            try:
                strategy_enum = MigrationStrategy(strategy)
            except ValueError:
                return {
                    "success": False,
                    "error": f"Invalid strategy '{strategy}'. "
                    f"Valid options: {', '.join(s.value for s in MigrationStrategy)}",
                }

            # Create planner
            from src.services.migration_planner import MigrationPlanner

            planner = MigrationPlanner(session)

            # Create plan
            plan = await planner.create_migration_plan(
                repository_id=repository.id,
                name=plan_name,
                strategy=strategy_enum,
                target_architecture=target_architecture,
                risk_tolerance=risk_tolerance,
                team_size=team_size,
                timeline_weeks=timeline_weeks,
            )

            # Get full plan details
            roadmap = await planner.generate_migration_roadmap(plan.id)

            result = {
                "plan_id": plan.id,
                "plan_name": plan.name,
                "strategy": plan.strategy,
                "target_architecture": plan.target_architecture,
                "summary": {
                    "total_steps": len(plan.steps),
                    "total_effort_hours": plan.total_effort_hours,
                    "timeline_weeks": roadmap["timeline"]["total_duration_weeks"],
                    "confidence_level": f"{plan.confidence_level * 100:.0f}%",
                    "risk_count": len(plan.risks),
                },
                "phases": roadmap["phases"],
                "critical_path": roadmap["critical_path"],
                "major_risks": [
                    {
                        "name": risk.name,
                        "level": risk.risk_level,
                        "mitigation": risk.mitigation_strategy,
                    }
                    for risk in plan.risks
                    if risk.risk_level in ["high", "critical"]
                ],
                "success_metrics": plan.success_metrics,
            }

            return {
                "success": True,
                "message": f"Migration plan '{plan_name}' created successfully",
                "data": result,
            }

        except Exception as e:
            logger.exception("Failed to create migration plan")
            return {"success": False, "error": str(e)}
    return {}


# Migration execution and monitoring tools
@mcp.tool(
    name="start_migration_step", description="Start execution of a migration step"
)
async def start_migration_step(
    step_id: int = Field(description="ID of the migration step to start"),
    executor_id: str | None = Field(
        description="ID of the person or system executing the step", default=None
    ),
) -> dict[str, Any]:
    """Start execution of a migration step."""
    await initialize_server()

    async for session in get_db_session():
        try:
            from src.services.migration_executor import MigrationExecutor

            executor = MigrationExecutor(session)
            result = await executor.start_migration_step(step_id, executor_id)

            if result["success"]:
                return {
                    "success": True,
                    "message": f"Started migration step: {result['step_name']}",
                    "data": result,
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Failed to start migration step"),
                }
        except Exception as e:
            logger.exception("Failed to start migration step")
            return {"success": False, "error": str(e)}
    return {}


@mcp.tool(
    name="complete_migration_step", description="Mark a migration step as completed"
)
async def complete_migration_step(
    step_id: int = Field(description="ID of the migration step to complete"),
    success: bool = Field(
        description="Whether the step completed successfully", default=True
    ),
    notes: str | None = Field(
        description="Completion notes or failure reason", default=None
    ),
    validation_results: dict[str, Any] | None = Field(
        description="Optional validation results", default=None
    ),
) -> dict[str, Any]:
    """Mark a migration step as completed."""
    await initialize_server()

    async for session in get_db_session():
        try:
            from src.services.migration_executor import MigrationExecutor

            executor = MigrationExecutor(session)
            result = await executor.complete_migration_step(
                step_id, success, notes, validation_results
            )

            if result["success"]:
                status = "successfully" if success else "with failure"
                return {
                    "success": True,
                    "message": f"Completed migration step {status}",
                    "data": result,
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Failed to complete migration step"),
                }
        except Exception as e:
            logger.exception("Failed to complete migration step")
            return {"success": False, "error": str(e)}
    return {}


@mcp.tool(
    name="track_migration_progress",
    description="Track overall progress of a migration plan",
)
async def track_migration_progress(
    plan_id: int = Field(description="ID of the migration plan to track"),
) -> dict[str, Any]:
    """Track overall progress of a migration plan."""
    await initialize_server()

    async for session in get_db_session():
        try:
            from src.services.migration_monitor import MigrationMonitor

            monitor = MigrationMonitor(session)
            progress = await monitor.track_migration_progress(plan_id)

            return {
                "success": True,
                "message": f"Migration plan is {progress['completion_percentage']:.1f}% complete",
                "data": progress,
            }
        except Exception as e:
            logger.exception("Failed to track migration progress")
            return {"success": False, "error": str(e)}
    return {}


@mcp.tool(
    name="get_migration_dashboard",
    description="Get migration dashboard with summary metrics",
)
async def get_migration_dashboard(
    repository_url: str | None = Field(
        description="Optional repository URL to filter by", default=None
    ),
) -> dict[str, Any]:
    """Get migration dashboard with summary metrics."""
    await initialize_server()

    async for session in get_db_session():
        try:
            # Get repository ID if URL provided
            repository_id = None
            if repository_url:
                result = await session.execute(
                    select(Repository).where(Repository.github_url == repository_url)
                )
                repository = result.scalar_one_or_none()
                if repository:
                    repository_id = repository.id

            from src.services.migration_monitor import MigrationMonitor

            monitor = MigrationMonitor(session)
            dashboard = await monitor.get_migration_dashboard(repository_id)

            return {
                "success": True,
                "message": f"Dashboard shows {dashboard['summary']['active_plans']} active migration plans",
                "data": dashboard,
            }
        except Exception as e:
            logger.exception("Failed to get migration dashboard")
            return {"success": False, "error": str(e)}
    return {}


# Migration knowledge management tools
@mcp.tool(
    name="extract_migration_patterns",
    description="Extract reusable patterns from completed migration plans",
)
async def extract_migration_patterns(
    plan_id: int = Field(
        description="ID of completed migration plan to extract patterns from"
    ),
) -> dict[str, Any]:
    """Extract reusable patterns from completed migration plans."""
    await initialize_server()

    async for session in get_db_session():
        try:
            from src.services.pattern_library import PatternLibrary

            library = PatternLibrary(session)
            patterns = await library.extract_patterns_from_execution(plan_id)

            return {
                "success": True,
                "message": f"Extracted {len(patterns)} patterns from migration plan",
                "data": {
                    "plan_id": plan_id,
                    "patterns_extracted": len(patterns),
                    "patterns": patterns,
                },
            }
        except Exception as e:
            logger.exception("Failed to extract patterns")
            return {"success": False, "error": str(e)}
    return {}


@mcp.tool(
    name="search_migration_patterns",
    description="Search the pattern library for migration patterns",
)
async def search_migration_patterns(
    category: str | None = Field(
        description="Pattern category to filter by", default=None
    ),
    context_type: str | None = Field(
        description="Context type to filter by", default=None
    ),
    keywords: str | None = Field(
        description="Keywords to search in name and description", default=None
    ),
    min_success_rate: float = Field(
        description="Minimum success rate (0.0 to 1.0)", default=0.7
    ),
) -> dict[str, Any]:
    """Search the pattern library for migration patterns."""
    await initialize_server()

    async for session in get_db_session():
        try:
            from src.services.pattern_library import PatternLibrary

            library = PatternLibrary(session)
            patterns = await library.search_patterns(
                category=category,
                context_type=context_type,
                keywords=keywords,
                min_success_rate=min_success_rate,
            )

            return {
                "success": True,
                "message": f"Found {len(patterns)} matching patterns",
                "data": {
                    "patterns": patterns,
                    "search_criteria": {
                        "category": category,
                        "context_type": context_type,
                        "keywords": keywords,
                        "min_success_rate": min_success_rate,
                    },
                },
            }
        except Exception as e:
            logger.exception("Failed to search patterns")
            return {"success": False, "error": str(e)}
    return {}


@mcp.tool(
    name="get_pattern_library_stats",
    description="Get statistics about the pattern library",
)
async def get_pattern_library_stats() -> dict[str, Any]:
    """Get statistics about the pattern library."""
    await initialize_server()

    async for session in get_db_session():
        try:
            from src.services.pattern_library import PatternLibrary

            library = PatternLibrary(session)
            stats = await library.get_library_statistics()

            return {
                "success": True,
                "message": "Retrieved pattern library statistics",
                "data": stats,
            }
        except Exception as e:
            logger.exception("Failed to get library stats")
            return {"success": False, "error": str(e)}
    return {}


# Register package analysis tools
# TODO: These need to be converted to @mcp.tool decorators
# mcp.add_tool(analyze_packages)
# mcp.add_tool(get_package_dependencies)
# mcp.add_tool(find_circular_dependencies)
# mcp.add_tool(get_package_coupling_metrics)
# mcp.add_tool(get_package_details)
# mcp.add_tool(get_package_tree)


# Tools from main.py for compatibility
@mcp.tool(name="search_code", description="Search for code by natural language query")
async def search_code(
    query: str = Field(description="Natural language search query"),
    limit: int = Field(default=10, description="Maximum results to return"),
) -> list[dict[str, Any]]:
    """Search for code by natural language query."""
    await initialize_server()

    try:
        from src.query.search_engine import SearchEngine

        async for session in get_db_session():
            search = SearchEngine(session)
            return await search.search(query, limit=limit)
    except Exception as e:
        logger.exception("Error in search_code")
        return [{"error": str(e)}]
    return []


@mcp.tool(name="explain_code", description="Get hierarchical code explanation")
async def explain_code(
    path: str = Field(
        description="Path to explain (can be file, directory, or entity like ClassName.method)"
    ),
) -> str:
    """Get hierarchical explanation of code."""
    await initialize_server()

    try:
        from src.mcp_server.tools.explain import ExplainTool

        tool = ExplainTool()
        return await tool.explain_code(path)
    except Exception as e:
        logger.exception("Error in explain_code")
        return f"Error explaining code: {e!s}"


@mcp.tool(name="find_definition", description="Find where a symbol is defined")
async def find_definition(
    name: str = Field(description="Name of the symbol to find"),
    file_path: str = Field(default=None, description="Optional file path to search in"),
    entity_type: str = Field(default=None, description="Optional entity type filter"),
) -> list[dict[str, Any]]:
    """Find where a symbol is defined."""
    await initialize_server()

    try:
        from src.query.symbol_finder import SymbolFinder

        async for session in get_db_session():
            finder = SymbolFinder(session)
            return await finder.find_definitions(
                name=name,
                file_path=file_path,
                entity_type=entity_type,
            )
    except Exception as e:
        logger.exception("Error in find_definition")
        return [{"error": str(e), "name": name}]
    return []


@mcp.tool(
    name="find_usage", description="Find all places where a function/class is used"
)
async def find_usage(
    function_or_class: str = Field(description="Name of the function or class"),
    repository: str = Field(
        default=None, description="Optional repository name filter"
    ),
) -> list[dict[str, Any]]:
    """Find all places where a function/class is used."""
    await initialize_server()

    try:
        from src.mcp_server.tools.find import FindTool

        async for session in get_db_session():
            tool = FindTool(session)
            return await tool.find_usage(function_or_class, repository)
    except Exception as e:
        logger.exception("Error in find_usage")
        return [{"error": str(e)}]
    return []


@mcp.tool(name="get_code_structure", description="Get the structure of a code file")
async def get_code_structure(
    file_path: str = Field(description="Path to the file relative to repository root"),
) -> dict[str, Any]:
    """Get the structure of a code file."""
    await initialize_server()

    try:
        from sqlalchemy import select

        from src.database.models import File

        async for session in get_db_session():
            # Get file from database
            result = await session.execute(select(File).where(File.path == file_path))
            file = result.scalar_one_or_none()

            if not file:
                return {"error": f"File not found: {file_path}"}

            # Get code structure using CodeProcessor
            from src.scanner.code_processor import CodeProcessor

            processor = CodeProcessor(session)
            return await processor.get_file_structure(file)

    except Exception as e:
        logger.exception("Error in get_code_structure")
        return {"error": str(e)}
    return {}


@mcp.tool(name="suggest_refactoring", description="Suggest refactoring opportunities")
async def suggest_refactoring(
    file_path: str = Field(description="Path to the file to analyze"),
    focus_area: str = Field(
        default=None, description="Optional focus area for suggestions"
    ),
) -> list[dict[str, Any]]:
    """Suggest refactoring opportunities."""
    await initialize_server()

    try:
        from src.mcp_server.tools.analyze import AnalyzeTool

        async for session in get_db_session():
            tool = AnalyzeTool(session)
            return await tool.suggest_refactoring(file_path, focus_area)
    except Exception as e:
        logger.exception("Error in suggest_refactoring")
        return [{"error": str(e)}]
    return []


@mcp.tool(name="find_similar_code", description="Find code similar to a given example")
async def find_similar_code(
    code_snippet: str = Field(description="Code snippet to find similar code for"),
    repository: str = Field(
        default=None, description="Optional repository name filter"
    ),
    limit: int = Field(default=10, description="Maximum results to return"),
) -> list[dict[str, Any]]:
    """Find code similar to a given example."""
    await initialize_server()

    try:
        from src.query.search_engine import SearchEngine

        async for session in get_db_session():
            search = SearchEngine(session)
            return await search.search_similar_code(code_snippet, limit=limit)
    except Exception as e:
        logger.exception("Error in find_similar_code")
        return [{"error": str(e)}]
    return []


@mcp.tool(name="health_check", description="Check server health and database status")
async def health_check() -> dict[str, Any]:
    """Check server health."""
    await initialize_server()

    try:
        async for session in get_db_session():
            # Check database connection
            result = await session.execute(select(1))
            db_ok = result.scalar() == 1

            # Get repository count
            repo_count_result = await session.execute(
                select(Repository).count(),
            )
            repo_count = repo_count_result.scalar()

            return {
                "status": "healthy" if db_ok else "unhealthy",
                "database": "connected" if db_ok else "disconnected",
                "repositories": repo_count,
                "version": "1.0.0",
            }
    except Exception as e:
        logger.exception("Health check failed")
        return {
            "status": "unhealthy",
            "error": str(e),
        }
    return {}


def create_server() -> FastMCP:
    """Create and configure the MCP server."""
    return mcp


if __name__ == "__main__":
    import asyncio

    # Run the server
    asyncio.run(mcp.run())
