"""MCP tools for migration execution and monitoring."""

from typing import Any

from fastmcp import Tool
from pydantic import BaseModel, Field

from src.mcp_server.tools.utils import (
    ToolResult,
    create_error_result,
    create_success_result,
    get_repository_id,
    get_session_factory,
)
from src.services.migration_executor import MigrationExecutor
from src.services.migration_monitor import MigrationMonitor


# Tool input models
class StartMigrationStepInput(BaseModel):
    """Input for start_migration_step tool."""

    step_id: int = Field(description="ID of the migration step to start")
    executor_id: str | None = Field(
        description="ID of the person or system executing the step", default=None
    )


class CompleteMigrationStepInput(BaseModel):
    """Input for complete_migration_step tool."""

    step_id: int = Field(description="ID of the migration step to complete")
    success: bool = Field(
        description="Whether the step completed successfully", default=True
    )
    notes: str | None = Field(
        description="Completion notes or failure reason", default=None
    )
    validation_results: dict[str, Any] | None = Field(
        description="Optional validation results", default=None
    )


class TrackMigrationProgressInput(BaseModel):
    """Input for track_migration_progress tool."""

    plan_id: int = Field(description="ID of the migration plan to track")


class ValidateMigrationStepInput(BaseModel):
    """Input for validate_migration_step tool."""

    step_id: int = Field(description="ID of the step to validate")
    validation_type: str = Field(
        description="Type of validation (automated, manual, performance)",
        default="manual",
    )
    validation_criteria: dict[str, Any] = Field(
        description="Criteria to validate against",
        default_factory=dict,
    )


class RollbackMigrationStepInput(BaseModel):
    """Input for rollback_migration_step tool."""

    step_id: int = Field(description="ID of the step to rollback")
    reason: str = Field(description="Reason for the rollback")


class GetMigrationDashboardInput(BaseModel):
    """Input for get_migration_dashboard tool."""

    repository_url: str | None = Field(
        description="Optional repository URL to filter dashboard", default=None
    )


class MonitorStepExecutionInput(BaseModel):
    """Input for monitor_step_execution tool."""

    step_id: int = Field(description="ID of the step to monitor")


class GetMigrationTimelineInput(BaseModel):
    """Input for get_migration_timeline tool."""

    plan_id: int = Field(description="ID of the migration plan")


class CheckMigrationHealthInput(BaseModel):
    """Input for check_migration_health tool."""

    plan_id: int = Field(description="ID of the plan to check")


class DetectAnomaliesInput(BaseModel):
    """Input for detect_anomalies tool."""

    plan_id: int = Field(description="ID of the plan to analyze")


class GenerateStatusReportInput(BaseModel):
    """Input for generate_status_report tool."""

    plan_id: int = Field(description="ID of the plan")
    report_type: str = Field(
        description="Type of report (summary, detailed, executive)",
        default="summary",
    )


# Tool implementations
async def start_migration_step_impl(input_data: StartMigrationStepInput) -> ToolResult:
    """Start execution of a migration step."""
    try:
        session_factory = await get_session_factory()
        async with session_factory() as session:
            executor = MigrationExecutor(session)
            result = await executor.start_migration_step(
                input_data.step_id, input_data.executor_id
            )

            if result["success"]:
                return create_success_result(
                    f"Started migration step '{result['step_name']}'",
                    result,
                )
            else:
                return create_error_result(result.get("error", "Failed to start step"))

    except Exception as e:
        return create_error_result(str(e))


async def complete_migration_step_impl(
    input_data: CompleteMigrationStepInput,
) -> ToolResult:
    """Complete execution of a migration step."""
    try:
        session_factory = await get_session_factory()
        async with session_factory() as session:
            executor = MigrationExecutor(session)
            result = await executor.complete_migration_step(
                input_data.step_id,
                input_data.success,
                input_data.notes,
                input_data.validation_results,
            )

            status = "successfully" if input_data.success else "with failures"
            return create_success_result(
                f"Completed migration step {status}",
                result,
            )

    except Exception as e:
        return create_error_result(str(e))


async def track_migration_progress_impl(
    input_data: TrackMigrationProgressInput,
) -> ToolResult:
    """Track overall migration plan progress."""
    try:
        session_factory = await get_session_factory()
        async with session_factory() as session:
            executor = MigrationExecutor(session)
            progress = await executor.track_migration_progress(input_data.plan_id)

            return create_success_result(
                f"Migration plan is {progress['progress_summary']['completion_percentage']:.1f}% complete",
                progress,
            )

    except Exception as e:
        return create_error_result(str(e))


async def validate_migration_step_impl(
    input_data: ValidateMigrationStepInput,
) -> ToolResult:
    """Validate a completed migration step."""
    try:
        session_factory = await get_session_factory()
        async with session_factory() as session:
            executor = MigrationExecutor(session)
            result = await executor.validate_migration_step(
                input_data.step_id,
                input_data.validation_type,
                input_data.validation_criteria,
            )

            status = "passed" if result["passed"] else "failed"
            return create_success_result(
                f"Validation {status} for migration step",
                result,
            )

    except Exception as e:
        return create_error_result(str(e))


async def rollback_migration_step_impl(
    input_data: RollbackMigrationStepInput,
) -> ToolResult:
    """Rollback a migration step."""
    try:
        session_factory = await get_session_factory()
        async with session_factory() as session:
            executor = MigrationExecutor(session)
            result = await executor.rollback_migration_step(
                input_data.step_id, input_data.reason
            )

            if result["success"]:
                return create_success_result(
                    "Initiated rollback for migration step",
                    result,
                )
            else:
                return create_error_result(
                    result.get("error", "Failed to initiate rollback")
                )

    except Exception as e:
        return create_error_result(str(e))


async def generate_execution_report_impl(plan_id: int) -> ToolResult:
    """Generate comprehensive execution report."""
    try:
        session_factory = await get_session_factory()
        async with session_factory() as session:
            executor = MigrationExecutor(session)
            report = await executor.generate_execution_report(plan_id)

            return create_success_result(
                f"Generated execution report with {report['summary']['total_executions']} executions",
                report,
            )

    except Exception as e:
        return create_error_result(str(e))


async def get_migration_dashboard_impl(
    input_data: GetMigrationDashboardInput,
) -> ToolResult:
    """Get migration dashboard data."""
    try:
        session_factory = await get_session_factory()
        async with session_factory() as session:
            # Get repository ID if URL provided
            repository_id = None
            if input_data.repository_url:
                repository_id = await get_repository_id(
                    session, input_data.repository_url
                )

            monitor = MigrationMonitor(session)
            dashboard = await monitor.get_migration_dashboard(repository_id)

            return create_success_result(
                f"Dashboard shows {dashboard['summary']['active_plans']} active migration plans",
                dashboard,
            )

    except Exception as e:
        return create_error_result(str(e))


async def monitor_step_execution_impl(
    input_data: MonitorStepExecutionInput,
) -> ToolResult:
    """Monitor a specific step execution in real-time."""
    try:
        session_factory = await get_session_factory()
        async with session_factory() as session:
            monitor = MigrationMonitor(session)
            status = await monitor.monitor_step_execution(input_data.step_id)

            return create_success_result(
                f"Step '{status['step_name']}' is {status['progress_percentage']:.1f}% complete",
                status,
            )

    except Exception as e:
        return create_error_result(str(e))


async def get_migration_timeline_impl(
    input_data: GetMigrationTimelineInput,
) -> ToolResult:
    """Get timeline view of migration execution."""
    try:
        session_factory = await get_session_factory()
        async with session_factory() as session:
            monitor = MigrationMonitor(session)
            timeline = await monitor.get_migration_timeline(input_data.plan_id)

            return create_success_result(
                f"Timeline shows {len(timeline['timeline_events'])} events",
                timeline,
            )

    except Exception as e:
        return create_error_result(str(e))


async def check_migration_health_impl(
    input_data: CheckMigrationHealthInput,
) -> ToolResult:
    """Perform health check on migration plan."""
    try:
        session_factory = await get_session_factory()
        async with session_factory() as session:
            monitor = MigrationMonitor(session)
            health = await monitor.check_migration_health(input_data.plan_id)

            return create_success_result(
                f"Migration health: {health['status']} (score: {health['overall_health_score']})",
                health,
            )

    except Exception as e:
        return create_error_result(str(e))


async def detect_anomalies_impl(input_data: DetectAnomaliesInput) -> ToolResult:
    """Detect anomalies in migration execution."""
    try:
        session_factory = await get_session_factory()
        async with session_factory() as session:
            monitor = MigrationMonitor(session)
            anomalies = await monitor.detect_anomalies(input_data.plan_id)

            return create_success_result(
                f"Detected {len(anomalies)} anomalies in migration execution",
                {"anomalies": anomalies, "count": len(anomalies)},
            )

    except Exception as e:
        return create_error_result(str(e))


async def generate_status_report_impl(
    input_data: GenerateStatusReportInput,
) -> ToolResult:
    """Generate status report for migration plan."""
    try:
        session_factory = await get_session_factory()
        async with session_factory() as session:
            monitor = MigrationMonitor(session)
            report = await monitor.generate_status_report(
                input_data.plan_id, input_data.report_type
            )

            return create_success_result(
                f"Generated {input_data.report_type} report for migration plan",
                report,
            )

    except Exception as e:
        return create_error_result(str(e))


# Tool definitions
start_migration_step = Tool(
    name="start_migration_step",
    description="Start execution of a migration step. Use this when ready to begin implementing a specific step in the migration plan.",
    input_schema=StartMigrationStepInput,
    output_schema=ToolResult,
    fn=start_migration_step_impl,
)

complete_migration_step = Tool(
    name="complete_migration_step",
    description="Mark a migration step as completed (successfully or with failure). Records completion status and any validation results.",
    input_schema=CompleteMigrationStepInput,
    output_schema=ToolResult,
    fn=complete_migration_step_impl,
)

track_migration_progress = Tool(
    name="track_migration_progress",
    description="Track overall progress of a migration plan including completion percentage, time tracking, blockers, and health score.",
    input_schema=TrackMigrationProgressInput,
    output_schema=ToolResult,
    fn=track_migration_progress_impl,
)

validate_migration_step = Tool(
    name="validate_migration_step",
    description="Validate a completed migration step against specified criteria. Supports automated, manual, and performance validation.",
    input_schema=ValidateMigrationStepInput,
    output_schema=ToolResult,
    fn=validate_migration_step_impl,
)

rollback_migration_step = Tool(
    name="rollback_migration_step",
    description="Rollback a migration step that has failed or needs to be undone. Only works if rollback strategy is defined.",
    input_schema=RollbackMigrationStepInput,
    output_schema=ToolResult,
    fn=rollback_migration_step_impl,
)

generate_execution_report = Tool(
    name="generate_execution_report",
    description="Generate comprehensive execution report for a migration plan including all executions, validations, and issues.",
    input_schema={"plan_id": int},
    output_schema=ToolResult,
    fn=lambda plan_id: generate_execution_report_impl(plan_id),
)

get_migration_dashboard = Tool(
    name="get_migration_dashboard",
    description="Get migration dashboard with summary metrics, active plans, recent activity, alerts, and performance metrics.",
    input_schema=GetMigrationDashboardInput,
    output_schema=ToolResult,
    fn=get_migration_dashboard_impl,
)

monitor_step_execution = Tool(
    name="monitor_step_execution",
    description="Monitor a specific migration step execution in real-time, showing progress, elapsed time, and any issues.",
    input_schema=MonitorStepExecutionInput,
    output_schema=ToolResult,
    fn=monitor_step_execution_impl,
)

get_migration_timeline = Tool(
    name="get_migration_timeline",
    description="Get timeline view of migration execution showing all events, critical path, and estimated completion.",
    input_schema=GetMigrationTimelineInput,
    output_schema=ToolResult,
    fn=get_migration_timeline_impl,
)

check_migration_health = Tool(
    name="check_migration_health",
    description="Perform comprehensive health check on migration plan including schedule, resources, risks, and validations.",
    input_schema=CheckMigrationHealthInput,
    output_schema=ToolResult,
    fn=check_migration_health_impl,
)

detect_anomalies = Tool(
    name="detect_anomalies",
    description="Detect anomalies in migration execution such as repeated failures, excessive overruns, or stalled progress.",
    input_schema=DetectAnomaliesInput,
    output_schema=ToolResult,
    fn=detect_anomalies_impl,
)

generate_status_report = Tool(
    name="generate_status_report",
    description="Generate status report for migration plan. Choose report type: summary (concise overview), detailed (comprehensive), or executive (high-level).",
    input_schema=GenerateStatusReportInput,
    output_schema=ToolResult,
    fn=generate_status_report_impl,
)
