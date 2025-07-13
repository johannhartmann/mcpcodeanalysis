"""Pattern library service for managing and learning from migration patterns."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.database.migration_models import (
    ExecutionStatus,
    MigrationExecution,
    MigrationPattern,
    MigrationPlan,
    MigrationStep,
)
from src.database.models import Repository
from src.llm.client import LLMClient
from src.logger import get_logger

logger = get_logger(__name__)


class PatternLibrary:
    """Service for managing migration patterns and learning from execution history."""

    def __init__(self, session: AsyncSession, llm_client: LLMClient | None = None):
        """Initialize the pattern library.

        Args:
            session: Database session
            llm_client: Optional LLM client for pattern extraction
        """
        self.session = session
        self.llm_client = llm_client

    async def extract_patterns_from_execution(
        self, plan_id: int
    ) -> list[dict[str, Any]]:
        """Extract reusable patterns from a completed migration plan.

        Args:
            plan_id: Completed migration plan to analyze

        Returns:
            List of extracted patterns
        """
        logger.info("Extracting patterns from migration plan %d", plan_id)

        # Get plan with all execution data
        plan = await self.session.get(
            MigrationPlan,
            plan_id,
            options=[
                selectinload(MigrationPlan.steps).selectinload(
                    MigrationStep.executions
                ),
                selectinload(MigrationPlan.repository),
            ],
        )

        if not plan:
            raise ValueError(f"Migration plan {plan_id} not found")

        # Analyze successful steps
        successful_steps = [
            step for step in plan.steps if step.status == ExecutionStatus.COMPLETED
        ]

        patterns = []

        # Extract step-level patterns
        for step in successful_steps:
            pattern = await self._extract_step_pattern(step, plan)
            if pattern:
                patterns.append(pattern)

        # Extract plan-level patterns
        plan_pattern = await self._extract_plan_pattern(plan)
        if plan_pattern:
            patterns.append(plan_pattern)

        # Extract dependency patterns
        dependency_patterns = await self._extract_dependency_patterns(plan)
        patterns.extend(dependency_patterns)

        return patterns

    async def add_pattern_to_library(
        self, pattern_data: dict[str, Any]
    ) -> MigrationPattern:
        """Add a new pattern to the library.

        Args:
            pattern_data: Pattern information

        Returns:
            Created pattern
        """
        logger.info("Adding pattern '%s' to library", pattern_data.get("name"))

        pattern = MigrationPattern(
            name=pattern_data["name"],
            category=pattern_data.get("category", "general"),
            description=pattern_data.get("description", ""),
            implementation_steps=pattern_data.get("implementation_steps", []),
            applicable_scenarios=pattern_data.get("applicable_scenarios", {}),
            prerequisites=pattern_data.get("prerequisites", []),
            risks=pattern_data.get("risks", []),
            best_practices=pattern_data.get("best_practices", []),
            anti_patterns=pattern_data.get("anti_patterns", []),
            success_metrics=pattern_data.get("success_metrics", {}),
            example_code=pattern_data.get("example_code"),
            tools_required=pattern_data.get("tools_required", []),
            avg_effort_hours=pattern_data.get("avg_effort_hours"),
            success_rate=pattern_data.get("success_rate", 0.0),
            usage_count=0,
        )

        self.session.add(pattern)
        await self.session.commit()
        await self.session.refresh(pattern)

        return pattern

    async def search_patterns(
        self,
        query: str | None = None,
        category: str | None = None,
        min_success_rate: float | None = None,
        applicable_to: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for patterns in the library.

        Args:
            query: Text search query
            category: Pattern category filter
            min_success_rate: Minimum success rate filter
            applicable_to: Scenario applicability filter

        Returns:
            List of matching patterns
        """
        logger.info("Searching patterns with query: %s", query)

        stmt = select(MigrationPattern)

        # Apply filters
        if category:
            stmt = stmt.where(MigrationPattern.category == category)

        if min_success_rate is not None:
            stmt = stmt.where(MigrationPattern.success_rate >= min_success_rate)

        # Text search
        if query:
            stmt = stmt.where(
                func.lower(MigrationPattern.name).contains(query.lower())
                | func.lower(MigrationPattern.description).contains(query.lower())
            )

        result = await self.session.execute(stmt)
        patterns = result.scalars().all()

        # Filter by applicability
        if applicable_to:
            patterns = [
                p for p in patterns if self._is_pattern_applicable(p, applicable_to)
            ]

        # Convert to dict and add usage stats
        pattern_results = []
        for pattern in patterns:
            pattern_dict = {
                "id": pattern.id,
                "name": pattern.name,
                "category": pattern.category,
                "description": pattern.description,
                "success_rate": pattern.success_rate,
                "usage_count": pattern.usage_count,
                "avg_effort_hours": pattern.avg_effort_hours,
                "applicable_scenarios": pattern.applicable_scenarios,
                "prerequisites": pattern.prerequisites,
                "best_practices": pattern.best_practices,
                "last_used": pattern.last_used_at,
            }
            pattern_results.append(pattern_dict)

        # Sort by relevance (usage and success rate)
        pattern_results.sort(
            key=lambda x: (x["success_rate"] * x["usage_count"]), reverse=True
        )

        return pattern_results

    async def get_pattern_recommendations(
        self, repository_id: int, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Get pattern recommendations based on repository and context.

        Args:
            repository_id: Repository to get recommendations for
            context: Current migration context

        Returns:
            List of recommended patterns with reasoning
        """
        logger.info("Getting pattern recommendations for repository %d", repository_id)

        # Analyze repository characteristics
        repo_analysis = await self._analyze_repository_characteristics(repository_id)

        # Get all patterns
        patterns = await self.search_patterns(min_success_rate=0.6)

        recommendations = []
        for pattern in patterns:
            score, reasoning = self._calculate_recommendation_score(
                pattern, repo_analysis, context
            )

            if score > 0.5:  # Threshold for recommendation
                recommendations.append(
                    {
                        "pattern": pattern,
                        "score": score,
                        "reasoning": reasoning,
                        "estimated_effort": self._adjust_effort_estimate(
                            pattern["avg_effort_hours"], repo_analysis
                        ),
                    }
                )

        # Sort by score
        recommendations.sort(key=lambda x: x["score"], reverse=True)

        return recommendations[:10]  # Top 10 recommendations

    async def update_pattern_from_execution(
        self, pattern_id: int, execution_data: dict[str, Any]
    ) -> None:
        """Update pattern statistics based on execution results.

        Args:
            pattern_id: Pattern that was used
            execution_data: Execution results
        """
        logger.info("Updating pattern %d from execution", pattern_id)

        pattern = await self.session.get(MigrationPattern, pattern_id)
        if not pattern:
            return

        # Update usage count
        pattern.usage_count += 1
        pattern.last_used_at = datetime.now(UTC)

        # Update success rate
        if execution_data.get("success"):
            # Rolling average
            pattern.success_rate = (
                (pattern.success_rate * (pattern.usage_count - 1)) + 1.0
            ) / pattern.usage_count
        else:
            pattern.success_rate = (
                pattern.success_rate * (pattern.usage_count - 1)
            ) / pattern.usage_count

        # Update average effort
        if execution_data.get("actual_hours"):
            if pattern.avg_effort_hours:
                pattern.avg_effort_hours = (
                    (pattern.avg_effort_hours * (pattern.usage_count - 1))
                    + execution_data["actual_hours"]
                ) / pattern.usage_count
            else:
                pattern.avg_effort_hours = execution_data["actual_hours"]

        # Add to learned scenarios
        if execution_data.get("scenario"):
            if not pattern.applicable_scenarios:
                pattern.applicable_scenarios = {}

            scenario_key = self._generate_scenario_key(execution_data["scenario"])
            pattern.applicable_scenarios[scenario_key] = {
                "success": execution_data.get("success", False),
                "context": execution_data["scenario"],
                "date": datetime.now(UTC).isoformat(),
            }

        await self.session.commit()

    async def generate_pattern_documentation(self, pattern_id: int) -> str:
        """Generate comprehensive documentation for a pattern.

        Args:
            pattern_id: Pattern to document

        Returns:
            Markdown documentation
        """
        logger.info("Generating documentation for pattern %d", pattern_id)

        pattern = await self.session.get(MigrationPattern, pattern_id)
        if not pattern:
            raise ValueError(f"Pattern {pattern_id} not found")

        # Get usage examples
        usage_examples = await self._get_pattern_usage_examples(pattern_id)

        doc_lines = [
            f"# {pattern.name}",
            "",
            f"**Category**: {pattern.category}",
            f"**Success Rate**: {pattern.success_rate * 100:.1f}%",
            f"**Average Effort**: {pattern.avg_effort_hours:.1f} hours",
            f"**Times Used**: {pattern.usage_count}",
            "",
            "## Description",
            pattern.description,
            "",
        ]

        # Prerequisites
        if pattern.prerequisites:
            doc_lines.extend(
                [
                    "## Prerequisites",
                    "",
                ]
            )
            for prereq in pattern.prerequisites:
                doc_lines.append(f"- {prereq}")
            doc_lines.append("")

        # Implementation steps
        if pattern.implementation_steps:
            doc_lines.extend(
                [
                    "## Implementation Steps",
                    "",
                ]
            )
            for i, step in enumerate(pattern.implementation_steps, 1):
                doc_lines.append(f"{i}. {step}")
            doc_lines.append("")

        # Best practices
        if pattern.best_practices:
            doc_lines.extend(
                [
                    "## Best Practices",
                    "",
                ]
            )
            for practice in pattern.best_practices:
                doc_lines.append(f"- {practice}")
            doc_lines.append("")

        # Anti-patterns
        if pattern.anti_patterns:
            doc_lines.extend(
                [
                    "## Anti-patterns to Avoid",
                    "",
                ]
            )
            for anti in pattern.anti_patterns:
                doc_lines.append(f"- {anti}")
            doc_lines.append("")

        # Risks
        if pattern.risks:
            doc_lines.extend(
                [
                    "## Known Risks",
                    "",
                ]
            )
            for risk in pattern.risks:
                doc_lines.append(f"- {risk}")
            doc_lines.append("")

        # Example code
        if pattern.example_code:
            doc_lines.extend(
                [
                    "## Example Code",
                    "```python",
                    pattern.example_code,
                    "```",
                    "",
                ]
            )

        # Usage examples
        if usage_examples:
            doc_lines.extend(
                [
                    "## Real-world Usage Examples",
                    "",
                ]
            )
            for example in usage_examples[:3]:  # Top 3 examples
                doc_lines.extend(
                    [
                        f"### {example['repository_name']}",
                        f"- **Context**: {example['context']}",
                        f"- **Duration**: {example['duration_hours']:.1f} hours",
                        f"- **Outcome**: {'Success' if example['success'] else 'Failed'}",
                        "",
                    ]
                )

        # Success metrics
        if pattern.success_metrics:
            doc_lines.extend(
                [
                    "## Success Metrics",
                    "",
                ]
            )
            for metric, value in pattern.success_metrics.items():
                doc_lines.append(f"- **{metric}**: {value}")
            doc_lines.append("")

        # Tools required
        if pattern.tools_required:
            doc_lines.extend(
                [
                    "## Required Tools",
                    "",
                ]
            )
            for tool in pattern.tools_required:
                doc_lines.append(f"- {tool}")

        return "\n".join(doc_lines)

    async def learn_from_failures(self, plan_id: int) -> dict[str, Any]:
        """Analyze failed migrations to extract lessons learned.

        Args:
            plan_id: Failed migration plan

        Returns:
            Lessons learned analysis
        """
        logger.info("Learning from failures in plan %d", plan_id)

        # Get plan with failures
        plan = await self.session.get(
            MigrationPlan,
            plan_id,
            options=[
                selectinload(MigrationPlan.steps).selectinload(
                    MigrationStep.executions
                ),
                selectinload(MigrationPlan.risks),
            ],
        )

        if not plan:
            raise ValueError(f"Migration plan {plan_id} not found")

        # Analyze failed steps
        failed_steps = [
            step for step in plan.steps if step.status == ExecutionStatus.FAILED
        ]

        lessons = {
            "plan_id": plan_id,
            "plan_name": plan.name,
            "failure_analysis": [],
            "root_causes": [],
            "prevention_strategies": [],
            "pattern_adjustments": [],
        }

        for step in failed_steps:
            analysis = await self._analyze_step_failure(step)
            lessons["failure_analysis"].append(analysis)

            # Extract root causes
            if analysis.get("root_cause"):
                lessons["root_causes"].append(analysis["root_cause"])

        # Identify common failure patterns
        failure_patterns = self._identify_failure_patterns(lessons["failure_analysis"])

        # Generate prevention strategies
        lessons["prevention_strategies"] = self._generate_prevention_strategies(
            failure_patterns, lessons["root_causes"]
        )

        # Suggest pattern adjustments
        lessons["pattern_adjustments"] = await self._suggest_pattern_adjustments(
            plan, failed_steps
        )

        return lessons

    async def _extract_step_pattern(
        self, step: MigrationStep, plan: MigrationPlan
    ) -> dict[str, Any] | None:
        """Extract pattern from a successful step.

        Args:
            step: Successful step
            plan: Parent plan

        Returns:
            Extracted pattern or None
        """
        # Skip if step is too specific
        if step.actual_hours and step.actual_hours < 1:
            return None

        # Use LLM if available to extract pattern
        if self.llm_client and step.description:
            try:
                prompt = f"""
                Extract a reusable migration pattern from this successful step:
                Step: {step.name}
                Description: {step.description}
                Task Type: {step.task_type}
                Strategy: {plan.strategy}

                Provide:
                1. Pattern name
                2. When to use it
                3. Key implementation steps
                4. Prerequisites
                """

                # This would use the LLM to extract pattern
                # For now, use rule-based extraction
            except Exception as e:
                logger.error("Failed to extract pattern with LLM: %s", e)

        # Rule-based pattern extraction
        pattern = {
            "name": f"{step.task_type.replace('_', ' ').title()} Pattern",
            "category": self._categorize_step(step),
            "description": f"Pattern for {step.task_type} based on successful execution",
            "implementation_steps": [step.description] if step.description else [],
            "applicable_scenarios": {
                "task_type": step.task_type,
                "phase": step.phase,
                "complexity": "medium",
            },
            "avg_effort_hours": step.actual_hours,
            "success_rate": 1.0,  # Since it succeeded
        }

        return pattern

    async def _extract_plan_pattern(self, plan: MigrationPlan) -> dict[str, Any] | None:
        """Extract pattern from overall plan.

        Args:
            plan: Completed plan

        Returns:
            Extracted pattern or None
        """
        if plan.status != ExecutionStatus.COMPLETED:
            return None

        # Calculate actual timeline
        actual_weeks = (
            (plan.updated_at - plan.created_at).days / 7 if plan.updated_at else None
        )

        pattern = {
            "name": f"{plan.strategy.replace('_', ' ').title()} Migration Pattern",
            "category": "migration_strategy",
            "description": f"Complete migration pattern using {plan.strategy} strategy",
            "implementation_steps": self._extract_plan_steps(plan),
            "applicable_scenarios": {
                "target_architecture": plan.target_architecture,
                "team_size": plan.team_size,
                "timeline_weeks": actual_weeks,
                "risk_tolerance": plan.risk_tolerance,
            },
            "prerequisites": [
                "Clear module boundaries identified",
                "Test coverage above 70%",
                "Team trained on migration strategy",
            ],
            "success_metrics": {
                "completion_rate": self._calculate_completion_rate(plan),
                "on_time_delivery": (
                    actual_weeks <= (plan.timeline_weeks or 0)
                    if actual_weeks and plan.timeline_weeks
                    else False
                ),
            },
            "avg_effort_hours": sum(
                s.actual_hours or 0 for s in plan.steps if s.actual_hours
            ),
            "success_rate": 1.0,
        }

        return pattern

    async def _extract_dependency_patterns(
        self, plan: MigrationPlan
    ) -> list[dict[str, Any]]:
        """Extract dependency handling patterns.

        Args:
            plan: Migration plan

        Returns:
            List of dependency patterns
        """
        patterns = []

        # Analyze step dependencies
        dependency_chains = self._analyze_dependency_chains(plan.steps)

        for chain_type, chain_info in dependency_chains.items():
            if chain_info["successful"]:
                pattern = {
                    "name": f"{chain_type.replace('_', ' ').title()} Dependency Pattern",
                    "category": "dependency_management",
                    "description": f"Pattern for handling {chain_type} dependencies",
                    "implementation_steps": chain_info["steps"],
                    "best_practices": chain_info["best_practices"],
                    "avg_effort_hours": chain_info["total_hours"],
                    "success_rate": 1.0,
                }
                patterns.append(pattern)

        return patterns

    async def _analyze_repository_characteristics(
        self, repository_id: int
    ) -> dict[str, Any]:
        """Analyze repository characteristics for pattern matching.

        Args:
            repository_id: Repository to analyze

        Returns:
            Repository characteristics
        """
        # Get repository with metrics
        repo = await self.session.get(Repository, repository_id)
        if not repo:
            return {}

        # Get codebase metrics
        from sqlalchemy import func

        from src.database.models import File, Function, Module

        # File count
        file_stmt = select(func.count(File.id)).where(
            File.repository_id == repository_id
        )
        file_result = await self.session.execute(file_stmt)
        file_count = file_result.scalar() or 0

        # Module count
        module_stmt = (
            select(func.count(Module.id))
            .join(File)
            .where(File.repository_id == repository_id)
        )
        module_result = await self.session.execute(module_stmt)
        module_count = module_result.scalar() or 0

        # Average complexity (simplified)
        complexity_stmt = (
            select(func.avg(Function.complexity))
            .join(Module)
            .join(File)
            .where(File.repository_id == repository_id)
        )
        complexity_result = await self.session.execute(complexity_stmt)
        avg_complexity = complexity_result.scalar() or 0

        return {
            "repository_id": repository_id,
            "repository_name": repo.name,
            "file_count": file_count,
            "module_count": module_count,
            "avg_complexity": float(avg_complexity),
            "size_category": self._categorize_size(file_count),
            "complexity_category": self._categorize_complexity(avg_complexity),
        }

    def _is_pattern_applicable(
        self, pattern: MigrationPattern, scenario: dict[str, Any]
    ) -> bool:
        """Check if pattern is applicable to scenario.

        Args:
            pattern: Pattern to check
            scenario: Current scenario

        Returns:
            True if applicable
        """
        if not pattern.applicable_scenarios:
            return True  # No specific requirements

        # Check each scenario attribute
        for key, value in scenario.items():
            pattern_value = pattern.applicable_scenarios.get(key)
            if pattern_value and pattern_value != value:
                return False

        return True

    def _calculate_recommendation_score(
        self,
        pattern: dict[str, Any],
        repo_analysis: dict[str, Any],
        context: dict[str, Any],
    ) -> tuple[float, str]:
        """Calculate recommendation score for a pattern.

        Args:
            pattern: Pattern to score
            repo_analysis: Repository characteristics
            context: Current context

        Returns:
            Score and reasoning
        """
        score = 0.0
        reasons = []

        # Success rate component (40%)
        score += pattern["success_rate"] * 0.4
        if pattern["success_rate"] > 0.8:
            reasons.append("High success rate")

        # Usage count component (20%)
        if pattern["usage_count"] > 10:
            score += 0.2
            reasons.append("Widely used pattern")
        elif pattern["usage_count"] > 5:
            score += 0.1
            reasons.append("Proven pattern")

        # Applicability component (40%)
        applicability_score = 0.0

        # Check size match
        if pattern.get("applicable_scenarios", {}).get(
            "size_category"
        ) == repo_analysis.get("size_category"):
            applicability_score += 0.5
            reasons.append("Matches repository size")

        # Check complexity match
        if pattern.get("applicable_scenarios", {}).get(
            "complexity_category"
        ) == repo_analysis.get("complexity_category"):
            applicability_score += 0.5
            reasons.append("Matches complexity level")

        score += applicability_score * 0.4

        reasoning = "; ".join(reasons) if reasons else "General applicability"
        return score, reasoning

    def _adjust_effort_estimate(
        self, base_effort: float | None, repo_analysis: dict[str, Any]
    ) -> float | None:
        """Adjust effort estimate based on repository characteristics.

        Args:
            base_effort: Base effort hours
            repo_analysis: Repository characteristics

        Returns:
            Adjusted effort estimate
        """
        if not base_effort:
            return None

        adjustment_factor = 1.0

        # Adjust for size
        size_category = repo_analysis.get("size_category", "medium")
        if size_category == "large":
            adjustment_factor *= 1.5
        elif size_category == "small":
            adjustment_factor *= 0.7

        # Adjust for complexity
        complexity_category = repo_analysis.get("complexity_category", "medium")
        if complexity_category == "high":
            adjustment_factor *= 1.3
        elif complexity_category == "low":
            adjustment_factor *= 0.8

        return base_effort * adjustment_factor

    def _generate_scenario_key(self, scenario: dict[str, Any]) -> str:
        """Generate a key for scenario storage.

        Args:
            scenario: Scenario data

        Returns:
            Scenario key
        """
        # Create a simple key from main attributes
        components = []
        for key in ["size_category", "complexity_category", "target_architecture"]:
            if key in scenario:
                components.append(f"{key}:{scenario[key]}")

        return "|".join(components) if components else "general"

    async def _get_pattern_usage_examples(
        self, pattern_id: int
    ) -> list[dict[str, Any]]:
        """Get real usage examples of a pattern.

        Args:
            pattern_id: Pattern ID

        Returns:
            List of usage examples
        """
        # Query executions that used this pattern
        # This is simplified - would need to track pattern usage in executions
        stmt = (
            select(MigrationExecution, MigrationStep, MigrationPlan, Repository)
            .join(MigrationStep)
            .join(MigrationPlan)
            .join(Repository)
            .where(MigrationExecution.success == True)
            .order_by(MigrationExecution.completed_at.desc())
            .limit(10)
        )

        result = await self.session.execute(stmt)
        executions = result.all()

        examples = []
        for execution, step, plan, repo in executions:
            if execution.completed_at and execution.started_at:
                duration = (
                    execution.completed_at - execution.started_at
                ).total_seconds() / 3600
            else:
                duration = 0

            examples.append(
                {
                    "repository_name": repo.name,
                    "context": f"{step.task_type} in {plan.strategy} migration",
                    "duration_hours": duration,
                    "success": execution.success,
                    "step_name": step.name,
                }
            )

        return examples

    async def _analyze_step_failure(self, step: MigrationStep) -> dict[str, Any]:
        """Analyze a failed step to extract lessons.

        Args:
            step: Failed step

        Returns:
            Failure analysis
        """
        analysis = {
            "step_id": step.id,
            "step_name": step.name,
            "task_type": step.task_type,
            "failure_reasons": [],
            "root_cause": None,
            "prevention_suggestions": [],
        }

        # Analyze executions
        failed_executions = [
            e for e in step.executions if e.status == ExecutionStatus.FAILED
        ]

        for execution in failed_executions:
            if execution.notes:
                analysis["failure_reasons"].append(execution.notes)

            # Extract patterns from logs
            if execution.logs:
                error_patterns = self._extract_error_patterns(execution.logs)
                analysis["failure_reasons"].extend(error_patterns)

        # Determine root cause
        if analysis["failure_reasons"]:
            analysis["root_cause"] = self._determine_root_cause(
                analysis["failure_reasons"]
            )

        # Generate prevention suggestions
        analysis["prevention_suggestions"] = self._generate_prevention_suggestions(
            step.task_type, analysis["root_cause"]
        )

        return analysis

    def _identify_failure_patterns(
        self, failure_analyses: list[dict[str, Any]]
    ) -> dict[str, list[str]]:
        """Identify common failure patterns.

        Args:
            failure_analyses: List of failure analyses

        Returns:
            Categorized failure patterns
        """
        patterns = {
            "technical": [],
            "process": [],
            "resource": [],
            "dependency": [],
        }

        for analysis in failure_analyses:
            root_cause = analysis.get("root_cause", "")

            if "dependency" in root_cause.lower() or "import" in root_cause.lower():
                patterns["dependency"].append(root_cause)
            elif "test" in root_cause.lower() or "validation" in root_cause.lower():
                patterns["technical"].append(root_cause)
            elif "time" in root_cause.lower() or "resource" in root_cause.lower():
                patterns["resource"].append(root_cause)
            else:
                patterns["process"].append(root_cause)

        return patterns

    def _generate_prevention_strategies(
        self, failure_patterns: dict[str, list[str]], root_causes: list[str]
    ) -> list[str]:
        """Generate prevention strategies based on failures.

        Args:
            failure_patterns: Categorized failure patterns
            root_causes: List of root causes

        Returns:
            Prevention strategies
        """
        strategies = []

        # Dependency failures
        if failure_patterns["dependency"]:
            strategies.extend(
                [
                    "Implement comprehensive dependency analysis before migration",
                    "Create dependency injection interfaces before extraction",
                    "Use feature flags for gradual dependency migration",
                ]
            )

        # Technical failures
        if failure_patterns["technical"]:
            strategies.extend(
                [
                    "Increase test coverage to minimum 80% before migration",
                    "Implement contract testing between modules",
                    "Add integration test suite for migrated components",
                ]
            )

        # Resource failures
        if failure_patterns["resource"]:
            strategies.extend(
                [
                    "Add 30% buffer to effort estimates",
                    "Implement parallel execution where possible",
                    "Consider phased approach for resource-intensive migrations",
                ]
            )

        # Process failures
        if failure_patterns["process"]:
            strategies.extend(
                [
                    "Establish clear rollback procedures",
                    "Implement staged validation checkpoints",
                    "Improve team communication channels",
                ]
            )

        return strategies

    async def _suggest_pattern_adjustments(
        self, plan: MigrationPlan, failed_steps: list[MigrationStep]
    ) -> list[dict[str, Any]]:
        """Suggest adjustments to patterns based on failures.

        Args:
            plan: Migration plan
            failed_steps: Failed steps

        Returns:
            Pattern adjustment suggestions
        """
        adjustments = []

        # Group failures by task type
        failures_by_type = {}
        for step in failed_steps:
            if step.task_type not in failures_by_type:
                failures_by_type[step.task_type] = []
            failures_by_type[step.task_type].append(step)

        # Suggest adjustments for each type
        for task_type, steps in failures_by_type.items():
            failure_rate = len(steps) / len(
                [s for s in plan.steps if s.task_type == task_type]
            )

            if failure_rate > 0.3:  # High failure rate
                adjustments.append(
                    {
                        "pattern_type": task_type,
                        "adjustment": "increase_validation",
                        "reason": f"High failure rate ({failure_rate * 100:.0f}%) for {task_type}",
                        "suggestions": [
                            f"Add pre-migration validation for {task_type}",
                            f"Increase effort estimate by {failure_rate * 50:.0f}%",
                            f"Add additional prerequisites check",
                        ],
                    }
                )

        return adjustments

    def _categorize_step(self, step: MigrationStep) -> str:
        """Categorize a step for pattern extraction.

        Args:
            step: Migration step

        Returns:
            Category
        """
        task_type = step.task_type.lower()

        if "extract" in task_type or "separate" in task_type:
            return "extraction"
        elif "interface" in task_type or "api" in task_type:
            return "interface_design"
        elif "refactor" in task_type:
            return "refactoring"
        elif "test" in task_type:
            return "testing"
        elif "validate" in task_type:
            return "validation"
        else:
            return "general"

    def _extract_plan_steps(self, plan: MigrationPlan) -> list[str]:
        """Extract high-level steps from plan.

        Args:
            plan: Migration plan

        Returns:
            List of implementation steps
        """
        steps = []

        # Group by phase
        phases = {}
        for step in plan.steps:
            phase = step.phase or "unspecified"
            if phase not in phases:
                phases[phase] = []
            phases[phase].append(step)

        # Create high-level steps
        for phase, phase_steps in sorted(phases.items()):
            if phase != "unspecified":
                step_summary = f"{phase}: "
                task_types = list({s.task_type for s in phase_steps})
                step_summary += ", ".join(task_types[:3])
                steps.append(step_summary)

        return steps

    def _calculate_completion_rate(self, plan: MigrationPlan) -> float:
        """Calculate completion rate for plan.

        Args:
            plan: Migration plan

        Returns:
            Completion rate (0-1)
        """
        if not plan.steps:
            return 0.0

        completed = sum(1 for s in plan.steps if s.status == ExecutionStatus.COMPLETED)
        return completed / len(plan.steps)

    def _analyze_dependency_chains(
        self, steps: list[MigrationStep]
    ) -> dict[str, dict[str, Any]]:
        """Analyze dependency chains in steps.

        Args:
            steps: Migration steps

        Returns:
            Dependency chain analysis
        """
        chains = {}

        # Find linear chains
        linear_chains = self._find_linear_chains(steps)
        if linear_chains:
            chains["linear_dependency"] = {
                "successful": True,
                "steps": [
                    "Identify clear sequence",
                    "Execute in order",
                    "Validate each step",
                ],
                "best_practices": [
                    "Clear interfaces between steps",
                    "Rollback procedures",
                ],
                "total_hours": sum(s.actual_hours or 0 for s in linear_chains[0]),
            }

        # Find parallel chains
        parallel_chains = self._find_parallel_chains(steps)
        if parallel_chains:
            chains["parallel_execution"] = {
                "successful": True,
                "steps": [
                    "Identify independent tasks",
                    "Allocate resources",
                    "Synchronize completion",
                ],
                "best_practices": ["Resource isolation", "Progress tracking"],
                "total_hours": max(
                    sum(s.actual_hours or 0 for s in chain) for chain in parallel_chains
                ),
            }

        return chains

    def _find_linear_chains(
        self, steps: list[MigrationStep]
    ) -> list[list[MigrationStep]]:
        """Find linear dependency chains.

        Args:
            steps: Migration steps

        Returns:
            List of linear chains
        """
        # Simplified - would implement graph traversal
        chains = []

        # Look for steps with single dependencies
        for step in steps:
            if len(step.dependencies) == 1:
                # This is part of a linear chain
                chain = [step]
                chains.append(chain)

        return chains

    def _find_parallel_chains(
        self, steps: list[MigrationStep]
    ) -> list[list[MigrationStep]]:
        """Find parallel execution chains.

        Args:
            steps: Migration steps

        Returns:
            List of parallel chains
        """
        # Simplified - would analyze dependency graph
        chains = []

        # Group steps by phase
        phase_groups = {}
        for step in steps:
            phase = step.phase or "unspecified"
            if phase not in phase_groups:
                phase_groups[phase] = []
            phase_groups[phase].append(step)

        # Look for phases with multiple independent steps
        for phase, phase_steps in phase_groups.items():
            if len(phase_steps) > 1:
                # Check if steps have no inter-dependencies
                independent = []
                for step in phase_steps:
                    depends_on_same_phase = any(
                        dep.depends_on_step_id in [s.id for s in phase_steps]
                        for dep in step.dependencies
                    )
                    if not depends_on_same_phase:
                        independent.append(step)

                if len(independent) > 1:
                    chains.append(independent)

        return chains

    def _categorize_size(self, file_count: int) -> str:
        """Categorize repository size.

        Args:
            file_count: Number of files

        Returns:
            Size category
        """
        if file_count < 100:
            return "small"
        elif file_count < 1000:
            return "medium"
        else:
            return "large"

    def _categorize_complexity(self, avg_complexity: float) -> str:
        """Categorize code complexity.

        Args:
            avg_complexity: Average cyclomatic complexity

        Returns:
            Complexity category
        """
        if avg_complexity < 5:
            return "low"
        elif avg_complexity < 10:
            return "medium"
        else:
            return "high"

    def _extract_error_patterns(self, logs: list[str]) -> list[str]:
        """Extract error patterns from logs.

        Args:
            logs: Execution logs

        Returns:
            Error patterns
        """
        patterns = []

        for log in logs:
            log_lower = log.lower()

            if "error" in log_lower or "failed" in log_lower:
                # Extract the error message
                if ":" in log:
                    error_msg = log.split(":", 1)[1].strip()
                    patterns.append(error_msg)
                else:
                    patterns.append(log)

        return patterns

    def _determine_root_cause(self, failure_reasons: list[str]) -> str:
        """Determine root cause from failure reasons.

        Args:
            failure_reasons: List of failure reasons

        Returns:
            Root cause
        """
        # Analyze patterns in failure reasons
        combined = " ".join(failure_reasons).lower()

        if "dependency" in combined or "import" in combined:
            return "Unresolved dependencies"
        elif "test" in combined or "assertion" in combined:
            return "Test failures"
        elif "timeout" in combined or "time" in combined:
            return "Execution timeout"
        elif "permission" in combined or "access" in combined:
            return "Permission issues"
        elif "memory" in combined or "resource" in combined:
            return "Resource constraints"
        else:
            return "Unknown failure"

    def _generate_prevention_suggestions(
        self, task_type: str, root_cause: str | None
    ) -> list[str]:
        """Generate prevention suggestions for a specific failure.

        Args:
            task_type: Type of task that failed
            root_cause: Root cause of failure

        Returns:
            Prevention suggestions
        """
        suggestions = []

        # Task-specific suggestions
        if "extract" in task_type:
            suggestions.extend(
                [
                    "Ensure module boundaries are well-defined",
                    "Create integration tests before extraction",
                    "Document module interfaces clearly",
                ]
            )
        elif "refactor" in task_type:
            suggestions.extend(
                [
                    "Increase test coverage before refactoring",
                    "Use automated refactoring tools",
                    "Perform incremental refactoring",
                ]
            )

        # Root cause specific suggestions
        if root_cause:
            if "dependency" in root_cause.lower():
                suggestions.extend(
                    [
                        "Create dependency injection interfaces",
                        "Use dependency inversion principle",
                        "Map all dependencies before migration",
                    ]
                )
            elif "test" in root_cause.lower():
                suggestions.extend(
                    [
                        "Improve test data management",
                        "Add contract testing",
                        "Implement test isolation",
                    ]
                )

        return suggestions
