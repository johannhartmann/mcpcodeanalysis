"""MCP tools for migration intelligence and planning."""

from typing import Any

from pydantic import Field
from sqlalchemy import select

from src.database.migration_models import MigrationStrategy
from src.database.models import Repository
from src.logger import get_logger

logger = get_logger(__name__)


async def analyze_migration_readiness(
    repository_url: str = Field(
        description="Repository URL to analyze for migration readiness"
    ),
    db_session=None,
) -> dict[str, Any]:
    """Analyze a repository to assess migration readiness and identify opportunities.

    This tool performs comprehensive analysis including:
    - Bounded context discovery and scoring
    - Module extraction candidates
    - Dependency analysis and hotspots
    - Complexity assessment
    - Strategy recommendations

    Use this before creating a migration plan to understand the codebase structure.
    """
    if not db_session:
        from src.database.init_db import get_session_factory

        session_factory = await get_session_factory()
        async with session_factory() as session:
            return await analyze_migration_readiness(repository_url, session)

    try:
        # Get repository ID
        result = await db_session.execute(
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

        analyzer = MigrationAnalyzer(db_session)

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
    db_session=None,
) -> dict[str, Any]:
    """Create a detailed migration plan for transforming a monolithic codebase.

    The plan includes:
    - Step-by-step migration tasks with dependencies
    - Risk assessment and mitigation strategies
    - Resource requirements and timeline
    - Success metrics and validation criteria

    Available strategies:
    - strangler_fig: Gradually replace functionality behind a facade
    - gradual: Extract modules one by one
    - big_bang: Complete rewrite (not recommended)
    - branch_by_abstraction: Create abstraction layer first
    - parallel_run: Run old and new systems in parallel
    """
    if not db_session:
        from src.database.init_db import get_session_factory

        session_factory = await get_session_factory()
        async with session_factory() as session:
            return await create_migration_plan(
                repository_url,
                plan_name,
                strategy,
                target_architecture,
                team_size,
                timeline_weeks,
                risk_tolerance,
                session,
            )

    try:
        # Get repository ID
        result = await db_session.execute(
            select(Repository).where(Repository.github_url == repository_url)
        )
        repository = result.scalar_one_or_none()

        if not repository:
            return {
                "success": False,
                "error": f"Repository {repository_url} not found. Please sync it first.",
            }

        # Validate strategy
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

        planner = MigrationPlanner(db_session)

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
