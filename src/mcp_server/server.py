"""MCP Code Analysis Server implementation with Resources."""

from collections.abc import AsyncGenerator
from typing import Any

from fastmcp import FastMCP
from pydantic import Field
from sqlalchemy import select

from src.database.init_db import get_session_factory, init_database
from src.database.models import Repository
from src.logger import get_logger

# Import resources
from src.mcp_server.resources import (
    CodeAnalysisResources,
    MigrationAnalysisResources,
    PackageAnalysisResources,
    SystemResources,
)

# Repository management tools handled via functions

logger = get_logger(__name__)

# Create FastMCP server
mcp = FastMCP("Code Analysis MCP Server")

# Global session factory
_session_factory = None
_resources_registered = False


async def initialize_server() -> None:
    """Initialize the server and database."""
    global _session_factory, _resources_registered

    if _session_factory is not None and _resources_registered:
        return

    logger.info("Initializing server...")

    # Initialize database if needed
    if _session_factory is None:
        engine = await init_database()
        _session_factory = get_session_factory(engine)

    # Register resources if not already done
    if not _resources_registered:
        migration_resources = MigrationAnalysisResources(mcp, _session_factory)
        migration_resources.register_resources()

        package_resources = PackageAnalysisResources(mcp, _session_factory)
        package_resources.register_resources()

        code_resources = CodeAnalysisResources(mcp, _session_factory)
        code_resources.register_resources()

        system_resources = SystemResources(mcp, _session_factory)
        system_resources.register_resources()

        _resources_registered = True

    # Register advanced tool classes
    logger.info("Registering advanced tool classes...")

    # Import tool classes
    from src.mcp_server.tools.analysis_tools import AnalysisTools
    from src.mcp_server.tools.domain_tools import DomainTools
    from src.mcp_server.tools.repository_management import RepositoryManagementTools

    # Note: Tool classes will create their own sessions as needed
    # We pass the session factory so they can create sessions on demand
    # Create temporary session just for tool registration
    async with _session_factory() as temp_session:
        # Domain-Driven Design tools
        domain_tools = DomainTools(temp_session, mcp)
        await domain_tools.register_tools()
        logger.info("Registered DomainTools")

        # Advanced analysis tools
        analysis_tools = AnalysisTools(temp_session, mcp)
        await analysis_tools.register_tools()
        logger.info("Registered AnalysisTools")

        # Repository management tools
        repo_tools = RepositoryManagementTools(temp_session, mcp)
        await repo_tools.register_tools()
        logger.info("Registered RepositoryManagementTools")

    logger.info("Server initialized successfully with resources and advanced tools")


async def get_db_session() -> AsyncGenerator[Any, None]:
    """Get a database session."""
    if _session_factory is None:
        await initialize_server()

    async with _session_factory() as session:
        yield session


# =======================
# TOOLS (for actions that modify state)
# =======================


# Note: Repository management is handled by RepositoryManagementTools class


# Migration planning tools (that create/modify data)
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

        except (AttributeError, KeyError, ValueError, TypeError) as e:
            logger.exception("Failed to create migration plan")
            return {"success": False, "error": str(e)}
    return {}


# Migration execution tools
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
            step = await executor.start_migration_step(step_id, executor_id)

            return {
                "success": True,
                "message": f"Started migration step: {step.name}",
                "data": {
                    "step_id": step.id,
                    "name": step.name,
                    "status": step.status,
                    "started_at": (
                        step.started_at.isoformat() if step.started_at else None
                    ),
                },
            }

        except (AttributeError, KeyError, ValueError, TypeError) as e:
            logger.exception("Failed to start migration step")
            return {"success": False, "error": str(e)}
    return {}


@mcp.tool(
    name="complete_migration_step",
    description="Mark a migration step as completed",
)
async def complete_migration_step(
    step_id: int = Field(description="ID of the migration step to complete"),
    status: str = Field(
        description="Completion status (completed, failed, blocked)",
        default="completed",
    ),
    notes: str | None = Field(description="Notes about the completion", default=None),
    validation_results: dict[str, Any] | None = Field(
        description="Validation test results", default=None
    ),
) -> dict[str, Any]:
    """Complete a migration step."""
    await initialize_server()

    async for session in get_db_session():
        try:
            from src.database.migration_models import StepStatus
            from src.services.migration_executor import MigrationExecutor

            # Validate status
            try:
                status_enum = StepStatus(status)
            except ValueError:
                return {
                    "success": False,
                    "error": f"Invalid status '{status}'. "
                    f"Valid options: {', '.join(s.value for s in StepStatus)}",
                }

            executor = MigrationExecutor(session)
            step = await executor.complete_migration_step(
                step_id, status_enum, notes, validation_results
            )

            # Update plan progress
            from src.services.migration_monitor import MigrationMonitor

            monitor = MigrationMonitor(session)
            progress = await monitor.calculate_plan_progress(step.plan_id)

            return {
                "success": True,
                "message": f"Completed migration step: {step.name}",
                "data": {
                    "step": {
                        "id": step.id,
                        "name": step.name,
                        "status": step.status,
                        "completed_at": (
                            step.completed_at.isoformat() if step.completed_at else None
                        ),
                    },
                    "plan_progress": progress,
                },
            }

        except (AttributeError, KeyError, ValueError, TypeError) as e:
            logger.exception("Failed to complete migration step")
            return {"success": False, "error": str(e)}
    return {}


@mcp.tool(
    name="extract_migration_patterns",
    description="Extract reusable patterns from completed migration plans",
)
async def extract_migration_patterns(
    plan_id: int = Field(description="ID of the completed migration plan"),
    min_success_rate: float = Field(
        description="Minimum success rate to extract patterns",
        default=0.8,
    ),
) -> dict[str, Any]:
    """Extract migration patterns from completed plans."""
    await initialize_server()

    async for session in get_db_session():
        try:
            from src.services.pattern_library import PatternLibrary

            library = PatternLibrary(session)
            patterns = await library.extract_patterns_from_plan(
                plan_id, min_success_rate
            )

            return {
                "success": True,
                "message": f"Extracted {len(patterns)} patterns from migration plan",
                "data": {
                    "patterns_extracted": len(patterns),
                    "patterns": [
                        {
                            "name": p.name,
                            "category": p.category,
                            "context_type": p.context_type,
                            "success_rate": p.success_rate,
                            "usage_count": p.usage_count,
                        }
                        for p in patterns
                    ],
                },
            }

        except (AttributeError, KeyError, ValueError, TypeError) as e:
            logger.exception("Failed to extract migration patterns")
            return {"success": False, "error": str(e)}
    return {}


# Package analysis tool (with force refresh option)
@mcp.tool(
    name="analyze_packages",
    description="Analyze the package structure of a repository",
)
async def analyze_packages(
    repository_url: str = Field(description="Repository URL to analyze"),
    force_refresh: bool = Field(
        description="Force re-analysis even if recent data exists",
        default=False,
    ),
) -> dict[str, Any]:
    """Analyze package structure."""
    await initialize_server()

    async for session in get_db_session():
        try:
            # Get repository
            result = await session.execute(
                select(Repository).where(Repository.github_url == repository_url)
            )
            repository = result.scalar_one_or_none()

            if not repository:
                return {
                    "success": False,
                    "error": f"Repository {repository_url} not found.",
                }

            from src.scanner.package_analyzer import PackageAnalyzer

            analyzer = PackageAnalyzer(session)

            # Check if we need to analyze
            if force_refresh or not await analyzer.has_recent_analysis(repository.id):
                analysis = await analyzer.analyze_repository_packages(repository.id)

                return {
                    "success": True,
                    "message": "Package analysis completed",
                    "data": {
                        "total_packages": analysis["total_packages"],
                        "max_depth": analysis["max_depth"],
                        "circular_dependencies": len(analysis["circular_dependencies"]),
                        "high_coupling_packages": len(
                            analysis["high_coupling_packages"]
                        ),
                        "package_metrics": analysis["package_metrics"][:10],  # Top 10
                    },
                }
            return {
                "success": True,
                "message": "Recent analysis exists. Use force_refresh=true to re-analyze.",
                "data": {
                    "hint": "Access package data via resources:",
                    "resources": [
                        f"packages://{repository_url}/tree",
                        f"packages://{repository_url}/circular-dependencies",
                        f"packages://{repository_url}/<package_path>/dependencies",
                    ],
                },
            }

        except (AttributeError, KeyError, ValueError, TypeError) as e:
            logger.exception("Failed to analyze packages")
            return {"success": False, "error": str(e)}
    return {}


# Note: FastMCP doesn't have startup/shutdown decorators
# Initialization happens in create_server() before resources are registered
