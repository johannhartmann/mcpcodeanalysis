"""Migration monitoring service for real-time tracking and alerts."""

from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.database.migration_models import (
    MigrationDependency,
    MigrationExecution,
    MigrationPlan,
    MigrationRisk,
    MigrationStep,
    MigrationStepStatus,
    MigrationValidation,
    RiskLevel,
    ValidationStatus,
)
from src.logger import get_logger

logger = get_logger(__name__)


class MigrationMonitor:
    """Service for monitoring migration execution and generating alerts."""

    def __init__(self, session: AsyncSession):
        """Initialize the migration monitor.

        Args:
            session: Database session
        """
        self.session = session

    async def get_migration_dashboard(
        self, repository_id: int | None = None
    ) -> dict[str, Any]:
        """Get migration dashboard data.

        Args:
            repository_id: Optional repository filter

        Returns:
            Dashboard data with metrics and status
        """
        logger.info("Generating migration dashboard")

        # Get active plans
        stmt = select(MigrationPlan).where(
            MigrationPlan.status.in_(
                [
                    MigrationStepStatus.IN_PROGRESS,
                    MigrationStepStatus.PENDING,
                ]
            )
        )

        if repository_id:
            stmt = stmt.where(MigrationPlan.repository_id == repository_id)

        result = await self.session.execute(stmt)
        active_plans = result.scalars().all()

        # Collect dashboard data
        dashboard = {
            "summary": await self._get_summary_metrics(repository_id),
            "active_plans": [],
            "recent_activity": await self._get_recent_activity(hours=24),
            "alerts": await self._get_active_alerts(),
            "risk_overview": await self._get_risk_overview(repository_id),
            "performance_metrics": await self._get_performance_metrics(),
        }

        # Add active plan details
        for plan in active_plans:
            plan_summary = await self._get_plan_summary(plan)
            dashboard["active_plans"].append(plan_summary)

        return dashboard

    async def monitor_step_execution(self, step_id: int) -> dict[str, Any]:
        """Monitor a specific step execution in real-time.

        Args:
            step_id: Step to monitor

        Returns:
            Real-time execution status
        """
        logger.info("Monitoring step %d execution", step_id)

        # Get step with current execution
        step = await self.session.get(
            MigrationStep,
            step_id,
            options=[
                selectinload(MigrationStep.executions),
                selectinload(MigrationStep.dependencies),
            ],
        )

        if not step:
            msg = f"Step {step_id} not found"
            raise ValueError(msg)

        # Find current execution
        current_execution = None
        for execution in step.executions:
            if execution.status == MigrationStepStatus.IN_PROGRESS:
                current_execution = execution
                break

        if not current_execution:
            return {
                "step_id": step_id,
                "step_name": step.name,
                "status": step.status.value,
                "message": "No active execution",
            }

        # Calculate progress
        elapsed_time = (
            datetime.now(UTC) - current_execution.started_at
        ).total_seconds() / 3600
        progress_percentage = min(
            (elapsed_time / step.estimated_hours * 100) if step.estimated_hours else 0,
            100,
        )

        # Check for issues
        issues = []
        if elapsed_time > step.estimated_hours * 1.2:  # 20% over estimate
            issues.append(
                {
                    "type": "overrun",
                    "severity": "warning",
                    "message": f"Execution time exceeds estimate by {(elapsed_time / step.estimated_hours - 1) * 100:.0f}%",
                }
            )

        # Get recent logs
        recent_logs = current_execution.logs[-10:] if current_execution.logs else []

        return {
            "step_id": step_id,
            "step_name": step.name,
            "execution_id": current_execution.id,
            "status": "executing",
            "started_at": current_execution.started_at,
            "elapsed_hours": round(elapsed_time, 2),
            "estimated_hours": step.estimated_hours,
            "progress_percentage": round(progress_percentage, 1),
            "executor": current_execution.executor_id,
            "recent_logs": recent_logs,
            "issues": issues,
        }

    async def get_migration_timeline(self, plan_id: int) -> dict[str, Any]:
        """Get timeline view of migration execution.

        Args:
            plan_id: Migration plan ID

        Returns:
            Timeline data
        """
        logger.info("Generating timeline for plan %d", plan_id)

        # Get plan with steps
        plan = await self.session.get(
            MigrationPlan,
            plan_id,
            options=[
                selectinload(MigrationPlan.steps).selectinload(MigrationStep.executions)
            ],
        )

        if not plan:
            msg = f"Plan {plan_id} not found"
            raise ValueError(msg)

        # Build timeline
        timeline_events = []

        for step in plan.steps:
            # Add step planned dates
            if step.estimated_start_date:
                timeline_events.append(
                    {
                        "timestamp": step.estimated_start_date,
                        "type": "planned_start",
                        "step_id": step.id,
                        "step_name": step.name,
                        "phase": step.phase,
                    }
                )

            # Add actual execution events
            for execution in step.executions:
                timeline_events.append(
                    {
                        "timestamp": execution.started_at,
                        "type": "execution_started",
                        "step_id": step.id,
                        "step_name": step.name,
                        "execution_id": execution.id,
                        "is_rollback": execution.is_rollback,
                    }
                )

                if execution.completed_at:
                    timeline_events.append(
                        {
                            "timestamp": execution.completed_at,
                            "type": "execution_completed",
                            "step_id": step.id,
                            "step_name": step.name,
                            "execution_id": execution.id,
                            "success": execution.success,
                        }
                    )

        # Sort by timestamp
        timeline_events.sort(key=lambda x: x["timestamp"])

        # Calculate critical path
        critical_path = await self._calculate_critical_path(plan)

        return {
            "plan_id": plan_id,
            "plan_name": plan.name,
            "timeline_events": timeline_events,
            "critical_path": critical_path,
            "start_date": plan.created_at,
            "estimated_end_date": self._calculate_estimated_end_date(plan),
            "current_phase": self._get_current_phase(plan),
        }

    async def check_migration_health(self, plan_id: int) -> dict[str, Any]:
        """Perform health check on migration plan.

        Args:
            plan_id: Plan to check

        Returns:
            Health check results
        """
        logger.info("Checking health for plan %d", plan_id)

        # Get plan with all data
        plan = await self.session.get(
            MigrationPlan,
            plan_id,
            options=[
                selectinload(MigrationPlan.steps),
                selectinload(MigrationPlan.risks),
            ],
        )

        if not plan:
            msg = f"Plan {plan_id} not found"
            raise ValueError(msg)

        # Perform health checks
        health_checks = {
            "schedule_health": await self._check_schedule_health(plan),
            "resource_health": await self._check_resource_health(plan),
            "risk_health": await self._check_risk_health(plan),
            "validation_health": await self._check_validation_health(plan),
            "dependency_health": await self._check_dependency_health(plan),
        }

        # Calculate overall health
        health_scores = [check["score"] for check in health_checks.values()]
        overall_health = sum(health_scores) / len(health_scores)

        # Determine status
        if overall_health >= 80:
            status = "healthy"
        elif overall_health >= 60:
            status = "warning"
        else:
            status = "critical"

        # Collect all issues
        all_issues = []
        for check in health_checks.values():
            all_issues.extend(check.get("issues", []))

        # Prioritize issues
        all_issues.sort(
            key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(
                x.get("severity", "low"), 3
            )
        )

        return {
            "plan_id": plan_id,
            "plan_name": plan.name,
            "overall_health_score": round(overall_health, 1),
            "status": status,
            "health_checks": health_checks,
            "issues": all_issues[:10],  # Top 10 issues
            "recommendations": self._generate_health_recommendations(
                health_checks, all_issues
            ),
        }

    async def detect_anomalies(self, plan_id: int) -> list[dict[str, Any]]:
        """Detect anomalies in migration execution.

        Args:
            plan_id: Plan to analyze

        Returns:
            List of detected anomalies
        """
        logger.info("Detecting anomalies for plan %d", plan_id)

        anomalies = []

        # Get plan with executions
        plan = await self.session.get(
            MigrationPlan,
            plan_id,
            options=[
                selectinload(MigrationPlan.steps).selectinload(MigrationStep.executions)
            ],
        )

        if not plan:
            msg = f"Plan {plan_id} not found"
            raise ValueError(msg)

        # Check for execution anomalies
        for step in plan.steps:
            # Multiple failed attempts
            failed_count = sum(
                1
                for e in step.executions
                if e.status == MigrationStepStatus.FAILED and not e.is_rollback
            )
            if failed_count >= 2:
                anomalies.append(
                    {
                        "type": "repeated_failures",
                        "step_id": step.id,
                        "step_name": step.name,
                        "severity": "high",
                        "description": f"Step has failed {failed_count} times",
                        "detected_at": datetime.now(UTC),
                    }
                )

            # Significant time overrun
            if step.actual_hours and step.estimated_hours:
                overrun_ratio = step.actual_hours / step.estimated_hours
                if overrun_ratio > 2.0:
                    anomalies.append(
                        {
                            "type": "excessive_overrun",
                            "step_id": step.id,
                            "step_name": step.name,
                            "severity": "medium",
                            "description": f"Execution took {overrun_ratio:.1f}x longer than estimated",
                            "detected_at": datetime.now(UTC),
                        }
                    )

            # Validation failures after success
            for execution in step.executions:
                if execution.success:
                    # Check validations
                    stmt = select(MigrationValidation).where(
                        MigrationValidation.execution_id == execution.id
                    )
                    result = await self.session.execute(stmt)
                    validations = result.scalars().all()

                    failed_validations = [
                        v for v in validations if v.status == ValidationStatus.FAILED
                    ]
                    if failed_validations:
                        anomalies.append(
                            {
                                "type": "validation_failure_after_success",
                                "step_id": step.id,
                                "step_name": step.name,
                                "severity": "high",
                                "description": "Step marked successful but validations failed",
                                "detected_at": datetime.now(UTC),
                            }
                        )

        # Check for plan-level anomalies
        # Stalled progress
        last_activity = max(
            (s.updated_at for s in plan.steps if s.updated_at), default=plan.created_at
        )
        if (datetime.now(UTC) - last_activity).days > 7:
            anomalies.append(
                {
                    "type": "stalled_progress",
                    "severity": "medium",
                    "description": "No activity in the last 7 days",
                    "detected_at": datetime.now(UTC),
                }
            )

        return anomalies

    async def generate_status_report(
        self, plan_id: int, report_type: str = "summary"
    ) -> dict[str, Any]:
        """Generate status report for migration plan.

        Args:
            plan_id: Plan ID
            report_type: Type of report (summary, detailed, executive)

        Returns:
            Status report
        """
        logger.info("Generating %s report for plan %d", report_type, plan_id)

        # Get comprehensive data
        progress = await self._get_plan_progress(plan_id)
        health = await self.check_migration_health(plan_id)
        anomalies = await self.detect_anomalies(plan_id)
        timeline = await self.get_migration_timeline(plan_id)

        if report_type == "executive":
            return self._generate_executive_report(
                plan_id, progress, health, anomalies, timeline
            )
        if report_type == "detailed":
            return self._generate_detailed_report(
                plan_id, progress, health, anomalies, timeline
            )
        # summary
        return self._generate_summary_report(
            plan_id, progress, health, anomalies, timeline
        )

    async def _get_summary_metrics(self, repository_id: int | None) -> dict[str, Any]:
        """Get summary metrics for dashboard.

        Args:
            repository_id: Optional repository filter

        Returns:
            Summary metrics
        """
        # Count plans by status
        stmt = select(MigrationPlan.status, func.count(MigrationPlan.id)).group_by(
            MigrationPlan.status
        )

        if repository_id:
            stmt = stmt.where(MigrationPlan.repository_id == repository_id)

        result = await self.session.execute(stmt)
        status_counts = dict(result.all())

        # Count total steps
        step_stmt = select(func.count(MigrationStep.id))
        if repository_id:
            step_stmt = step_stmt.join(MigrationPlan).where(
                MigrationPlan.repository_id == repository_id
            )

        step_result = await self.session.execute(step_stmt)
        total_steps = step_result.scalar() or 0

        return {
            "total_plans": sum(status_counts.values()),
            "active_plans": status_counts.get(MigrationStepStatus.IN_PROGRESS, 0),
            "completed_plans": status_counts.get(MigrationStepStatus.COMPLETED, 0),
            "failed_plans": status_counts.get(MigrationStepStatus.FAILED, 0),
            "total_steps": total_steps,
        }

    async def _get_recent_activity(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get recent migration activity.

        Args:
            hours: Hours to look back

        Returns:
            Recent activities
        """
        since = datetime.now(UTC) - timedelta(hours=hours)

        # Get recent executions
        stmt = (
            select(MigrationExecution)
            .options(selectinload(MigrationExecution.step))
            .where(MigrationExecution.started_at >= since)
            .order_by(MigrationExecution.started_at.desc())
            .limit(20)
        )

        result = await self.session.execute(stmt)
        executions = result.scalars().all()

        activities = []
        for execution in executions:
            activity = {
                "timestamp": execution.started_at,
                "type": (
                    "execution_started"
                    if not execution.is_rollback
                    else "rollback_started"
                ),
                "step_name": execution.step.name if execution.step else "Unknown",
                "executor": execution.executor_id,
                "status": execution.status.value,
            }

            if execution.completed_at:
                activity["completed_at"] = execution.completed_at
                activity["duration_hours"] = (
                    execution.completed_at - execution.started_at
                ).total_seconds() / 3600

            activities.append(activity)

        return activities

    async def _get_active_alerts(self) -> list[dict[str, Any]]:
        """Get active alerts that need attention.

        Returns:
            List of active alerts
        """
        alerts = []

        # Check for failed steps
        failed_stmt = select(MigrationStep).where(
            MigrationStep.status == MigrationStepStatus.FAILED
        )
        failed_result = await self.session.execute(failed_stmt)
        failed_steps = failed_result.scalars().all()

        alerts.extend(
            {
                "type": "step_failure",
                "severity": "high",
                "message": f"Step '{step.name}' has failed",
                "step_id": step.id,
                "timestamp": step.updated_at,
            }
            for step in failed_steps
        )

        # Check for overdue steps
        overdue_stmt = select(MigrationStep).where(
            and_(
                MigrationStep.status == MigrationStepStatus.IN_PROGRESS,
                MigrationStep.estimated_completion_date < datetime.now(UTC),
            )
        )
        overdue_result = await self.session.execute(overdue_stmt)
        overdue_steps = overdue_result.scalars().all()

        alerts.extend(
            {
                "type": "step_overdue",
                "severity": "medium",
                "message": f"Step '{step.name}' is overdue",
                "step_id": step.id,
                "timestamp": datetime.now(UTC),
            }
            for step in overdue_steps
        )

        # Check for high risks
        risk_stmt = select(MigrationRisk).where(
            MigrationRisk.risk_level == RiskLevel.CRITICAL
        )
        risk_result = await self.session.execute(risk_stmt)
        critical_risks = risk_result.scalars().all()

        alerts.extend(
            {
                "type": "critical_risk",
                "severity": "critical",
                "message": f"Critical risk: {risk.description}",
                "risk_id": risk.id,
                "timestamp": risk.identified_date,
            }
            for risk in critical_risks
        )

        return alerts

    async def _get_risk_overview(self, repository_id: int | None) -> dict[str, Any]:
        """Get risk overview for migrations.

        Args:
            repository_id: Optional repository filter

        Returns:
            Risk overview
        """
        # Count risks by level
        stmt = select(MigrationRisk.risk_level, func.count(MigrationRisk.id)).group_by(
            MigrationRisk.risk_level
        )

        if repository_id:
            stmt = stmt.join(MigrationPlan).where(
                MigrationPlan.repository_id == repository_id
            )

        result = await self.session.execute(stmt)
        risk_counts = dict(result.all())

        return {
            "total_risks": sum(risk_counts.values()),
            "critical_risks": risk_counts.get(RiskLevel.CRITICAL, 0),
            "high_risks": risk_counts.get(RiskLevel.HIGH, 0),
            "medium_risks": risk_counts.get(RiskLevel.MEDIUM, 0),
            "low_risks": risk_counts.get(RiskLevel.LOW, 0),
        }

    async def _get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics for migrations.

        Returns:
            Performance metrics
        """
        # Calculate average execution times
        stmt = select(
            func.avg(
                func.extract(
                    "epoch",
                    MigrationExecution.completed_at - MigrationExecution.started_at,
                )
                / 3600
            )
        ).where(
            and_(
                MigrationExecution.completed_at.isnot(None),
                MigrationExecution.success,
            )
        )

        result = await self.session.execute(stmt)
        avg_execution_time = result.scalar() or 0

        # Calculate success rate
        total_stmt = select(func.count(MigrationExecution.id))
        total_result = await self.session.execute(total_stmt)
        total_executions = total_result.scalar() or 0

        success_stmt = select(func.count(MigrationExecution.id)).where(
            MigrationExecution.success
        )
        success_result = await self.session.execute(success_stmt)
        successful_executions = success_result.scalar() or 0

        success_rate = (
            successful_executions / total_executions * 100
            if total_executions > 0
            else 0
        )

        return {
            "avg_execution_hours": round(avg_execution_time, 2),
            "success_rate": round(success_rate, 1),
            "total_executions": total_executions,
        }

    async def _get_plan_summary(self, plan: MigrationPlan) -> dict[str, Any]:
        """Get summary for a single plan.

        Args:
            plan: Migration plan

        Returns:
            Plan summary
        """
        # Load steps if not loaded
        if not plan.steps:
            await self.session.refresh(plan, ["steps"])

        total_steps = len(plan.steps)
        completed_steps = sum(
            1 for s in plan.steps if s.status == MigrationStepStatus.COMPLETED
        )
        in_progress_steps = sum(
            1 for s in plan.steps if s.status == MigrationStepStatus.IN_PROGRESS
        )

        return {
            "plan_id": plan.id,
            "plan_name": plan.name,
            "status": plan.status.value,
            "progress_percentage": (
                completed_steps / total_steps * 100 if total_steps > 0 else 0
            ),
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "in_progress_steps": in_progress_steps,
            "started_at": plan.created_at,
        }

    async def _calculate_critical_path(
        self, plan: MigrationPlan
    ) -> list[dict[str, Any]]:
        """Calculate critical path for migration plan.

        Args:
            plan: Migration plan

        Returns:
            Critical path steps
        """
        # Simplified critical path - steps with no parallel alternatives
        critical_steps = []

        for step in plan.steps:
            # Check if step has dependents
            dependent_stmt = select(func.count(MigrationStep.id)).where(
                MigrationStep.dependencies.any(
                    MigrationDependency.prerequisite_step_id == step.id
                )
            )
            dependent_result = await self.session.execute(dependent_stmt)
            dependent_count = dependent_result.scalar() or 0

            if dependent_count > 0:
                critical_steps.append(
                    {
                        "step_id": step.id,
                        "step_name": step.name,
                        "phase": step.phase,
                        "estimated_hours": step.estimated_hours,
                        "dependent_count": dependent_count,
                    }
                )

        return critical_steps

    def _calculate_estimated_end_date(self, plan: MigrationPlan) -> datetime | None:
        """Calculate estimated end date for plan.

        Args:
            plan: Migration plan

        Returns:
            Estimated end date
        """
        if not plan.steps:
            return None

        # Sum remaining hours
        remaining_hours = sum(
            s.estimated_hours or 0
            for s in plan.steps
            if s.status
            in [MigrationStepStatus.PENDING, MigrationStepStatus.IN_PROGRESS]
        )

        if remaining_hours == 0:
            return None

        # Assume 40 hours per week
        remaining_weeks = remaining_hours / 40
        return datetime.now(UTC) + timedelta(weeks=remaining_weeks)

    def _get_current_phase(self, plan: MigrationPlan) -> str | None:
        """Get current phase of migration.

        Args:
            plan: Migration plan

        Returns:
            Current phase
        """
        for step in plan.steps:
            if step.status == MigrationStepStatus.IN_PROGRESS:
                return step.phase

        # Find next pending step
        for step in plan.steps:
            if step.status == MigrationStepStatus.PENDING:
                return step.phase

        return None

    async def _check_schedule_health(self, plan: MigrationPlan) -> dict[str, Any]:
        """Check schedule health of plan.

        Args:
            plan: Migration plan

        Returns:
            Schedule health check
        """
        issues = []

        # Check for overdue steps
        overdue_count = 0
        for step in plan.steps:
            if (
                step.estimated_completion_date
                and step.status != MigrationStepStatus.COMPLETED
                and step.estimated_completion_date < datetime.now(UTC)
            ):
                overdue_count += 1
                issues.append(
                    {
                        "type": "overdue_step",
                        "severity": "high",
                        "description": f"Step '{step.name}' is overdue",
                    }
                )

        # Calculate schedule variance
        completed_steps = [
            s for s in plan.steps if s.status == MigrationStepStatus.COMPLETED
        ]
        if completed_steps:
            total_variance = sum(
                (s.actual_hours or 0) - (s.estimated_hours or 0)
                for s in completed_steps
            )
            avg_variance = total_variance / len(completed_steps)

            if avg_variance > 0.2:  # 20% over
                issues.append(
                    {
                        "type": "schedule_variance",
                        "severity": "medium",
                        "description": f"Average {avg_variance * 100:.0f}% over estimates",
                    }
                )

        # Calculate health score
        score = 100
        score -= overdue_count * 10
        score -= len(issues) * 5
        score = max(0, score)

        return {
            "score": score,
            "overdue_steps": overdue_count,
            "issues": issues,
        }

    async def _check_resource_health(self, _plan: MigrationPlan) -> dict[str, Any]:
        """Check resource health of plan.

        Args:
            plan: Migration plan

        Returns:
            Resource health check
        """
        issues = []

        # Check resource allocation
        # This is simplified - would check actual resource data
        score = 85  # Default good score

        return {
            "score": score,
            "issues": issues,
        }

    async def _check_risk_health(self, plan: MigrationPlan) -> dict[str, Any]:
        """Check risk health of plan.

        Args:
            plan: Migration plan

        Returns:
            Risk health check
        """
        issues = []

        # Count unmitigated risks
        unmitigated_count = sum(
            1
            for r in plan.risks
            if not r.mitigation_status or r.mitigation_status == "pending"
        )

        if unmitigated_count > 0:
            issues.append(
                {
                    "type": "unmitigated_risks",
                    "severity": "high",
                    "description": f"{unmitigated_count} risks without mitigation",
                }
            )

        # Check critical risks
        critical_count = sum(
            1 for r in plan.risks if r.risk_level == RiskLevel.CRITICAL
        )

        if critical_count > 0:
            issues.append(
                {
                    "type": "critical_risks",
                    "severity": "critical",
                    "description": f"{critical_count} critical risks identified",
                }
            )

        # Calculate score
        score = 100
        score -= critical_count * 20
        score -= unmitigated_count * 10
        score = max(0, score)

        return {
            "score": score,
            "unmitigated_risks": unmitigated_count,
            "critical_risks": critical_count,
            "issues": issues,
        }

    async def _check_validation_health(self, plan: MigrationPlan) -> dict[str, Any]:
        """Check validation health of plan.

        Args:
            plan: Migration plan

        Returns:
            Validation health check
        """
        issues = []

        # Get validation stats
        validation_stats = defaultdict(int)

        for step in plan.steps:
            for execution in step.executions:
                stmt = select(MigrationValidation).where(
                    MigrationValidation.execution_id == execution.id
                )
                result = await self.session.execute(stmt)
                validations = result.scalars().all()

                for validation in validations:
                    validation_stats[validation.status] += 1

        # Check validation failure rate
        total_validations = sum(validation_stats.values())
        failed_validations = validation_stats.get(ValidationStatus.FAILED, 0)

        if total_validations > 0:
            failure_rate = failed_validations / total_validations
            if failure_rate > 0.1:  # 10% failure rate
                issues.append(
                    {
                        "type": "high_validation_failure_rate",
                        "severity": "high",
                        "description": f"{failure_rate * 100:.0f}% validation failure rate",
                    }
                )

        # Calculate score
        score = 100
        if total_validations > 0:
            score = (1 - failure_rate) * 100

        return {
            "score": score,
            "total_validations": total_validations,
            "failed_validations": failed_validations,
            "issues": issues,
        }

    async def _check_dependency_health(self, _plan: MigrationPlan) -> dict[str, Any]:
        """Check dependency health of plan.

        Args:
            plan: Migration plan

        Returns:
            Dependency health check
        """
        issues = []

        # Check for circular dependencies (simplified)
        # In practice would do full graph analysis

        score = 90  # Default good score

        return {
            "score": score,
            "issues": issues,
        }

    def _generate_health_recommendations(
        self, health_checks: dict[str, Any], issues: list[dict[str, Any]]
    ) -> list[str]:
        """Generate recommendations based on health checks.

        Args:
            health_checks: Health check results
            issues: All issues found

        Returns:
            List of recommendations
        """
        recommendations = []

        # Check worst performing areas
        worst_score = min(check["score"] for check in health_checks.values())

        if worst_score < 60:
            recommendations.append(
                "Consider pausing migration to address critical issues"
            )

        # Specific recommendations based on issues
        issue_types = {issue["type"] for issue in issues}

        if "overdue_step" in issue_types:
            recommendations.append(
                "Review and update timeline estimates for remaining steps"
            )

        if "unmitigated_risks" in issue_types:
            recommendations.append("Develop mitigation strategies for identified risks")

        if "high_validation_failure_rate" in issue_types:
            recommendations.append("Improve testing and validation procedures")

        if "repeated_failures" in issue_types:
            recommendations.append("Investigate root causes of repeated step failures")

        return recommendations

    async def _get_plan_progress(self, plan_id: int) -> dict[str, Any]:
        """Get detailed progress for a plan.

        Args:
            plan_id: Plan ID

        Returns:
            Progress details
        """
        # This would be implemented to get detailed progress
        # For now, return basic structure
        return {
            "plan_id": plan_id,
            "progress_percentage": 0,
            "phase_progress": {},
            "milestone_status": {},
        }

    def _generate_executive_report(
        self,
        plan_id: int,
        progress: dict[str, Any],
        health: dict[str, Any],
        anomalies: list[dict[str, Any]],
        timeline: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate executive-level report.

        Args:
            plan_id: Plan ID
            progress: Progress data
            health: Health data
            anomalies: Anomalies data
            timeline: Timeline data

        Returns:
            Executive report
        """
        return {
            "report_type": "executive",
            "plan_id": plan_id,
            "generated_at": datetime.now(UTC),
            "executive_summary": {
                "overall_status": health["status"],
                "health_score": health["overall_health_score"],
                "critical_issues": len(
                    [i for i in health["issues"] if i.get("severity") == "critical"]
                ),
                "estimated_completion": timeline.get("estimated_end_date"),
            },
            "key_metrics": {
                "progress": progress.get("progress_percentage", 0),
                "anomalies": len(anomalies),
                "recommendations": health["recommendations"][:3],  # Top 3
            },
        }

    def _generate_detailed_report(
        self,
        plan_id: int,
        progress: dict[str, Any],
        health: dict[str, Any],
        anomalies: list[dict[str, Any]],
        timeline: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate detailed report.

        Args:
            plan_id: Plan ID
            progress: Progress data
            health: Health data
            anomalies: Anomalies data
            timeline: Timeline data

        Returns:
            Detailed report
        """
        return {
            "report_type": "detailed",
            "plan_id": plan_id,
            "generated_at": datetime.now(UTC),
            "progress": progress,
            "health": health,
            "anomalies": anomalies,
            "timeline": timeline,
            "all_issues": health["issues"],
            "all_recommendations": health["recommendations"],
        }

    def _generate_summary_report(
        self,
        plan_id: int,
        progress: dict[str, Any],
        health: dict[str, Any],
        anomalies: list[dict[str, Any]],
        timeline: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate summary report.

        Args:
            plan_id: Plan ID
            progress: Progress data
            health: Health data
            anomalies: Anomalies data
            timeline: Timeline data

        Returns:
            Summary report
        """
        return {
            "report_type": "summary",
            "plan_id": plan_id,
            "generated_at": datetime.now(UTC),
            "status": health["status"],
            "health_score": health["overall_health_score"],
            "progress_percentage": progress.get("progress_percentage", 0),
            "anomaly_count": len(anomalies),
            "current_phase": timeline.get("current_phase"),
            "top_issues": health["issues"][:5],  # Top 5 issues
            "next_steps": health["recommendations"][:3],  # Top 3 recommendations
        }
