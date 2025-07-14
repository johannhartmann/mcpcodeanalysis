"""MCP tools for migration execution and monitoring."""

from typing import Any

from pydantic import Field
from sqlalchemy import select

from src.database.models import Repository
from src.logger import get_logger

logger = get_logger(__name__)


async def start_migration_step(
    step_id: int = Field(description="ID of the migration step to start"),
    executor_id: str | None = Field(
        description="ID of the person or system executing the step", default=None
    ),
    db_session=None,
) -> dict[str, Any]:
    """Start execution of a migration step.

    Use this when ready to begin implementing a specific step in the migration plan.
    """
    if not db_session:
        from src.database.init_db import get_session_factory

        session_factory = await get_session_factory()
        async with session_factory() as db_session:
            return await start_migration_step(step_id, executor_id, db_session)

    try:
        from src.services.migration_executor import MigrationExecutor

        executor = MigrationExecutor(db_session)
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
    db_session=None,
) -> dict[str, Any]:
    """Mark a migration step as completed (successfully or with failure).

    Records completion status and any validation results.
    """
    if not db_session:
        from src.database.init_db import get_session_factory

        session_factory = await get_session_factory()
        async with session_factory() as db_session:
            return await complete_migration_step(
                step_id, success, notes, validation_results, db_session
            )

    try:
        from src.services.migration_executor import MigrationExecutor

        executor = MigrationExecutor(db_session)
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


async def track_migration_progress(
    plan_id: int = Field(description="ID of the migration plan to track"),
    db_session=None,
) -> dict[str, Any]:
    """Track overall progress of a migration plan.

    Including completion percentage, time tracking, blockers, and health score.
    """
    if not db_session:
        from src.database.init_db import get_session_factory

        session_factory = await get_session_factory()
        async with session_factory() as db_session:
            return await track_migration_progress(plan_id, db_session)

    try:
        from src.services.migration_monitor import MigrationMonitor

        monitor = MigrationMonitor(db_session)
        progress = await monitor.track_migration_progress(plan_id)

        return {
            "success": True,
            "message": f"Migration plan is {progress['completion_percentage']:.1f}% complete",
            "data": progress,
        }

    except Exception as e:
        logger.exception("Failed to track migration progress")
        return {"success": False, "error": str(e)}


async def get_migration_dashboard(
    repository_url: str | None = Field(
        description="Optional repository URL to filter by", default=None
    ),
    db_session=None,
) -> dict[str, Any]:
    """Get migration dashboard with summary metrics, active plans, recent activity, alerts, and performance metrics."""
    if not db_session:
        from src.database.init_db import get_session_factory

        session_factory = await get_session_factory()
        async with session_factory() as db_session:
            return await get_migration_dashboard(repository_url, db_session)

    try:
        # Get repository ID if URL provided
        repository_id = None
        if repository_url:
            result = await db_session.execute(
                select(Repository).where(Repository.github_url == repository_url)
            )
            repository = result.scalar_one_or_none()
            if repository:
                repository_id = repository.id

        from src.services.migration_monitor import MigrationMonitor

        monitor = MigrationMonitor(db_session)
        dashboard = await monitor.get_migration_dashboard(repository_id)

        return {
            "success": True,
            "message": f"Dashboard shows {dashboard['summary']['active_plans']} active migration plans",
            "data": dashboard,
        }

    except Exception as e:
        logger.exception("Failed to get migration dashboard")
        return {"success": False, "error": str(e)}
