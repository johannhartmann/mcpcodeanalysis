"""Migration planner service for creating detailed migration roadmaps."""

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.database.migration_models import (
    MigrationDecision,
    MigrationDependency,
    MigrationPlan,
    MigrationResourceRequirement,
    MigrationRisk,
    MigrationStep,
    MigrationStrategy,
    ResourceType,
)
from src.logger import get_logger
from src.services.migration_analyzer import MigrationAnalyzer

logger = get_logger(__name__)


class MigrationPlanner:
    """Service for creating and managing migration plans."""

    def __init__(self, session: AsyncSession):
        """Initialize the migration planner.

        Args:
            session: Database session
        """
        self.session = session
        self.analyzer = MigrationAnalyzer(session)

    async def create_migration_plan(
        self,
        repository_id: int,
        name: str,
        strategy: MigrationStrategy,
        target_architecture: str = "modular_monolith",
        risk_tolerance: str = "medium",
        team_size: int = 5,
        timeline_weeks: int | None = None,
    ) -> MigrationPlan:
        """Create a comprehensive migration plan.

        Args:
            repository_id: Repository to migrate
            name: Plan name
            strategy: Migration strategy to use
            target_architecture: Target architecture type
            risk_tolerance: Risk tolerance level (low, medium, high)
            team_size: Available team size
            timeline_weeks: Desired timeline (optional)

        Returns:
            Created migration plan
        """
        logger.info(
            "Creating migration plan '%s' for repository %d", name, repository_id
        )

        # Analyze repository
        analysis = await self.analyzer.analyze_repository_for_migration(repository_id)

        # Create plan
        plan = MigrationPlan(
            repository_id=repository_id,
            name=name,
            description=f"Migration to {target_architecture} using {strategy.value} strategy",
            strategy=strategy.value,
            target_architecture=target_architecture,
            risk_tolerance=risk_tolerance,
            team_size=team_size,
            timeline_weeks=timeline_weeks,
            complexity_score=analysis["complexity_metrics"][
                "avg_cyclomatic_complexity"
            ],
            constraints={
                "team_size": team_size,
                "risk_tolerance": risk_tolerance,
                "target_completion": timeline_weeks,
            },
        )

        self.session.add(plan)
        await self.session.flush()

        # Generate migration steps
        steps = await self._generate_migration_steps(plan, analysis)

        # Create dependencies between steps
        await self._create_step_dependencies(steps)

        # Assess risks
        risks = await self._assess_plan_risks(plan, analysis)

        # Calculate resource requirements
        await self._calculate_resource_requirements(plan, steps)

        # Calculate effort and timeline
        total_effort = sum(step.estimated_hours or 0 for step in steps)
        plan.total_effort_hours = total_effort

        # Calculate confidence level
        plan.confidence_level = self._calculate_plan_confidence(
            analysis, len(steps), len(risks)
        )

        # Set success metrics
        plan.success_metrics = {
            "module_extraction_count": len(analysis["migration_candidates"][:10]),
            "test_coverage_target": 80,
            "performance_degradation_threshold": 5,  # Max 5% performance loss
            "downtime_minutes": 0,  # Zero downtime goal
        }

        await self.session.commit()

        logger.info(
            "Created migration plan %d with %d steps, %d risks",
            plan.id,
            len(steps),
            len(risks),
        )

        return plan

    async def _generate_migration_steps(
        self, plan: MigrationPlan, analysis: dict[str, Any]
    ) -> list[MigrationStep]:
        """Generate detailed migration steps based on analysis.

        Args:
            plan: Migration plan
            analysis: Repository analysis results

        Returns:
            List of created migration steps
        """
        steps = []
        sequence = 1

        # Phase 1: Preparation steps
        if plan.strategy == MigrationStrategy.STRANGLER_FIG.value:
            # Setup proxy/routing layer
            step = MigrationStep(
                plan_id=plan.id,
                sequence_number=sequence,
                name="Setup Strangler Fig Proxy",
                description="Implement routing layer to redirect traffic between old and new implementations",
                step_type="infrastructure_setup",
                estimated_hours=40,
                validation_criteria={
                    "proxy_operational": True,
                    "zero_downtime_routing": True,
                    "monitoring_enabled": True,
                },
            )
            self.session.add(step)
            steps.append(step)
            sequence += 1

        # Add test coverage improvement step if needed
        candidates_without_tests = [
            c
            for c in analysis["migration_candidates"][:10]
            if not c["quality_metrics"]["has_tests"]
        ]
        if candidates_without_tests:
            step = MigrationStep(
                plan_id=plan.id,
                sequence_number=sequence,
                name="Improve Test Coverage",
                description="Add comprehensive tests for modules lacking coverage",
                step_type="test_creation",
                estimated_hours=len(candidates_without_tests) * 16,
                validation_criteria={
                    "test_coverage": 80,
                    "all_tests_passing": True,
                },
            )
            self.session.add(step)
            steps.append(step)
            sequence += 1

        # Phase 2: Module extraction steps
        for candidate in analysis["migration_candidates"][:10]:
            # Create interface definition step
            interface_step = MigrationStep(
                plan_id=plan.id,
                sequence_number=sequence,
                name=f"Define Interface for {candidate['name']}",
                description=f"Create clear interface/contract for {candidate['path']}",
                step_type="create_interface",
                target_package_id=candidate["id"],
                estimated_hours=8,
                validation_criteria={
                    "interface_documented": True,
                    "contract_tests_created": True,
                },
            )
            self.session.add(interface_step)
            steps.append(interface_step)
            sequence += 1

            # Create extraction step
            extract_step = MigrationStep(
                plan_id=plan.id,
                sequence_number=sequence,
                name=f"Extract {candidate['name']} Module",
                description=f"Extract and modularize {candidate['path']}",
                step_type="extract_module",
                target_package_id=candidate["id"],
                estimated_hours=candidate["estimated_effort_hours"],
                validation_criteria={
                    "module_extracted": True,
                    "tests_passing": True,
                    "no_circular_dependencies": True,
                    "interface_compliance": True,
                },
                rollback_procedure={
                    "steps": [
                        "Revert extracted module changes",
                        "Restore original module structure",
                        "Verify system functionality",
                    ]
                },
            )
            self.session.add(extract_step)
            steps.append(extract_step)
            sequence += 1

            # Add integration testing step
            if candidate["dependents"] > 3:
                test_step = MigrationStep(
                    plan_id=plan.id,
                    sequence_number=sequence,
                    name=f"Integration Test {candidate['name']}",
                    description=f"Comprehensive integration testing for {candidate['name']} with dependents",
                    step_type="integration_test",
                    target_package_id=candidate["id"],
                    estimated_hours=16,
                    validation_criteria={
                        "integration_tests_passing": True,
                        "performance_benchmarks_met": True,
                        "backwards_compatibility": True,
                    },
                )
                self.session.add(test_step)
                steps.append(test_step)
                sequence += 1

        # Phase 3: Optimization steps
        opt_step = MigrationStep(
            plan_id=plan.id,
            sequence_number=sequence,
            name="Optimize Module Communication",
            description="Optimize inter-module communication and remove redundancies",
            step_type="optimization",
            estimated_hours=24,
            validation_criteria={
                "performance_improved": True,
                "latency_reduced": True,
            },
        )
        self.session.add(opt_step)
        steps.append(opt_step)
        sequence += 1

        # Phase 4: Cleanup steps
        cleanup_step = MigrationStep(
            plan_id=plan.id,
            sequence_number=sequence,
            name="Remove Legacy Code",
            description="Clean up old implementations and deprecated code",
            step_type="cleanup",
            estimated_hours=16,
            validation_criteria={
                "legacy_code_removed": True,
                "no_dead_code": True,
                "documentation_updated": True,
            },
        )
        self.session.add(cleanup_step)
        steps.append(cleanup_step)

        await self.session.flush()
        return steps

    async def _create_step_dependencies(self, steps: list[MigrationStep]) -> None:
        """Create dependencies between migration steps.

        Args:
            steps: List of migration steps
        """
        # Map steps by type for easier dependency creation
        steps_by_type = {}
        for step in steps:
            if step.step_type not in steps_by_type:
                steps_by_type[step.step_type] = []
            steps_by_type[step.step_type].append(step)

        # Test creation must complete before extraction
        if "test_creation" in steps_by_type and "extract_module" in steps_by_type:
            for test_step in steps_by_type["test_creation"]:
                for extract_step in steps_by_type["extract_module"]:
                    dep = MigrationDependency(
                        dependent_step_id=extract_step.id,
                        prerequisite_step_id=test_step.id,
                        dependency_type="hard",
                        description="Tests must exist before extraction",
                    )
                    self.session.add(dep)

        # Interface must be defined before extraction
        interface_steps = steps_by_type.get("create_interface", [])
        extract_steps = steps_by_type.get("extract_module", [])

        for i, interface_step in enumerate(interface_steps):
            if i < len(extract_steps):
                extract_step = extract_steps[i]
                dep = MigrationDependency(
                    dependent_step_id=extract_step.id,
                    prerequisite_step_id=interface_step.id,
                    dependency_type="hard",
                    description="Interface must be defined before extraction",
                )
                self.session.add(dep)

        # Sequential dependencies for same package
        for step_type in ["extract_module", "integration_test"]:
            if step_type in steps_by_type:
                type_steps = sorted(
                    steps_by_type[step_type], key=lambda s: s.sequence_number
                )
                for i in range(1, len(type_steps)):
                    dep = MigrationDependency(
                        dependent_step_id=type_steps[i].id,
                        prerequisite_step_id=type_steps[i - 1].id,
                        dependency_type="soft",
                        description="Recommended sequential execution",
                    )
                    self.session.add(dep)

        await self.session.flush()

    async def _assess_plan_risks(
        self, plan: MigrationPlan, _analysis: dict[str, Any]
    ) -> list[MigrationRisk]:
        """Assess risks for the migration plan.

        Args:
            plan: Migration plan
            analysis: Repository analysis

        Returns:
            List of created risks
        """
        risks = []

        # Get risks from analyzer
        analyzer_risks = await self.analyzer.assess_migration_risks(
            plan.repository_id, plan.id
        )

        for risk_data in analyzer_risks:
            risk = MigrationRisk(
                plan_id=plan.id,
                risk_type=risk_data["type"],
                name=risk_data["name"],
                description=risk_data["description"],
                probability=risk_data["probability"],
                impact=risk_data["impact"],
                risk_level=risk_data["level"],
                mitigation_strategy=risk_data["mitigation"],
            )
            self.session.add(risk)
            risks.append(risk)

        # Add plan-specific risks
        if plan.timeline_weeks and plan.timeline_weeks < 12:
            risk = MigrationRisk(
                plan_id=plan.id,
                risk_type="operational",
                name="Aggressive Timeline",
                description="Short timeline may lead to rushed decisions",
                probability=0.7,
                impact=0.6,
                risk_level="high",
                mitigation_strategy="Build in buffer time and prioritize critical paths",
            )
            self.session.add(risk)
            risks.append(risk)

        if plan.team_size < 3:
            risk = MigrationRisk(
                plan_id=plan.id,
                risk_type="operational",
                name="Small Team Size",
                description="Limited team size may cause bottlenecks",
                probability=0.6,
                impact=0.7,
                risk_level="high",
                mitigation_strategy="Consider bringing in temporary resources for peak phases",
            )
            self.session.add(risk)
            risks.append(risk)

        await self.session.flush()
        return risks

    async def _calculate_resource_requirements(
        self, plan: MigrationPlan, steps: list[MigrationStep]
    ) -> list[MigrationResourceRequirement]:
        """Calculate resource requirements for the plan.

        Args:
            plan: Migration plan
            steps: Migration steps

        Returns:
            List of resource requirements
        """
        requirements = []

        # Calculate requirements by step type
        step_types = {}
        for step in steps:
            if step.step_type not in step_types:
                step_types[step.step_type] = {"steps": [], "hours": 0}
            step_types[step.step_type]["steps"].append(step)
            step_types[step.step_type]["hours"] += step.estimated_hours or 0

        # Define resource needs by step type
        resource_mapping = {
            "infrastructure_setup": {
                "roles": [ResourceType.DEVOPS, ResourceType.ARCHITECT],
                "skills": ["kubernetes", "docker", "networking", "monitoring"],
            },
            "test_creation": {
                "roles": [ResourceType.DEVELOPER, ResourceType.QA],
                "skills": ["unit_testing", "integration_testing", "tdd"],
            },
            "create_interface": {
                "roles": [ResourceType.ARCHITECT, ResourceType.DEVELOPER],
                "skills": ["api_design", "contract_design", "documentation"],
            },
            "extract_module": {
                "roles": [ResourceType.DEVELOPER, ResourceType.ARCHITECT],
                "skills": ["refactoring", "design_patterns", "testing"],
            },
            "integration_test": {
                "roles": [ResourceType.QA, ResourceType.DEVELOPER],
                "skills": ["integration_testing", "performance_testing"],
            },
            "optimization": {
                "roles": [ResourceType.DEVELOPER, ResourceType.DBA],
                "skills": ["performance_tuning", "profiling", "caching"],
            },
        }

        # Create requirements
        for step_type, info in step_types.items():
            mapping = resource_mapping.get(
                step_type, {"roles": [ResourceType.DEVELOPER], "skills": []}
            )

            for role in mapping["roles"]:
                # Calculate FTE needed
                hours = info["hours"]
                weeks = plan.timeline_weeks or 16
                fte_needed = hours / (40 * weeks)  # 40 hours per week

                req = MigrationResourceRequirement(
                    plan_id=plan.id,
                    resource_type=role.value,
                    skill_level="senior" if role == ResourceType.ARCHITECT else "mid",
                    quantity=round(fte_needed, 1),
                    specific_skills=mapping["skills"],
                )
                self.session.add(req)
                requirements.append(req)

        await self.session.flush()
        return requirements

    def _calculate_plan_confidence(
        self, analysis: dict[str, Any], step_count: int, risk_count: int
    ) -> float:
        """Calculate confidence level for the plan.

        Args:
            analysis: Repository analysis
            step_count: Number of steps
            risk_count: Number of risks

        Returns:
            Confidence level (0.0 to 1.0)
        """
        confidence = 0.5  # Base confidence

        # Good bounded contexts increase confidence
        if analysis["bounded_contexts"]:
            avg_readiness = sum(
                c["migration_readiness"] for c in analysis["bounded_contexts"][:3]
            ) / min(3, len(analysis["bounded_contexts"]))
            confidence += avg_readiness * 0.2

        # Low complexity increases confidence
        complexity_rating = analysis["complexity_metrics"]["complexity_rating"]
        if complexity_rating == "low":
            confidence += 0.2
        elif complexity_rating == "medium":
            confidence += 0.1

        # Reasonable step count
        if 10 <= step_count <= 30:
            confidence += 0.1

        # Low risk count
        if risk_count < 5:
            confidence += 0.1
        elif risk_count > 10:
            confidence -= 0.1

        return max(0.1, min(confidence, 0.95))

    async def optimize_migration_plan(
        self, plan_id: int, optimization_goals: dict[str, Any]
    ) -> MigrationPlan:
        """Optimize an existing migration plan based on goals.

        Args:
            plan_id: Plan to optimize
            optimization_goals: Goals like minimize_time, minimize_risk, maximize_quality

        Returns:
            Updated migration plan
        """
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

        logger.info("Optimizing migration plan %d", plan_id)

        # Record optimization decision
        decision = MigrationDecision(
            plan_id=plan_id,
            decision_type="optimization",
            title="Plan Optimization",
            description="Optimizing plan based on specified goals",
            alternatives_considered=list(optimization_goals.keys()),
            evaluation_criteria=optimization_goals,
            rationale=f"Optimizing for: {', '.join(optimization_goals.keys())}",
            made_by="system",
        )
        self.session.add(decision)

        # Apply optimizations
        if "minimize_time" in optimization_goals:
            await self._optimize_for_time(plan)

        if "minimize_risk" in optimization_goals:
            await self._optimize_for_risk(plan)

        if "maximize_quality" in optimization_goals:
            await self._optimize_for_quality(plan)

        # Recalculate metrics
        plan.updated_at = datetime.now(UTC)

        await self.session.commit()
        return plan

    async def _optimize_for_time(self, plan: MigrationPlan) -> None:
        """Optimize plan to minimize timeline.

        Args:
            plan: Plan to optimize
        """
        # Identify steps that can be parallelized
        for step in plan.steps:
            # Check if step has only soft dependencies
            hard_deps = [d for d in step.dependencies if d.dependency_type == "hard"]
            if not hard_deps and step.estimated_hours and step.estimated_hours > 40:
                # Can be split into parallel sub-tasks
                step.notes = "Can be parallelized across team members"

        # Update strategy if beneficial
        if plan.strategy == MigrationStrategy.GRADUAL.value:
            plan.strategy = MigrationStrategy.PARALLEL_RUN.value
            logger.info("Changed strategy to parallel_run for time optimization")

    async def _optimize_for_risk(self, plan: MigrationPlan) -> None:
        """Optimize plan to minimize risk.

        Args:
            plan: Plan to optimize
        """
        # Add validation steps after risky operations
        risky_steps = [s for s in plan.steps if s.step_type == "extract_module"]

        for step in risky_steps:
            # Check if validation step exists
            next_seq = step.sequence_number + 1
            has_validation = any(
                s.sequence_number == next_seq and s.step_type == "integration_test"
                for s in plan.steps
            )

            if not has_validation:
                # Add validation checkpoint
                step.validation_criteria["extended_validation"] = True
                step.notes = "Extended validation added for risk mitigation"

    async def _optimize_for_quality(self, plan: MigrationPlan) -> None:
        """Optimize plan to maximize quality.

        Args:
            plan: Plan to optimize
        """
        # Increase test coverage requirements
        for step in plan.steps:
            if step.validation_criteria and "test_coverage" in step.validation_criteria:
                step.validation_criteria["test_coverage"] = 90  # Increase to 90%

            # Add code review requirements
            step.validation_criteria["code_review_completed"] = True
            step.validation_criteria["documentation_updated"] = True

    async def generate_migration_roadmap(self, plan_id: int) -> dict[str, Any]:
        """Generate a visual roadmap for the migration plan.

        Args:
            plan_id: Plan to generate roadmap for

        Returns:
            Roadmap data structure
        """
        plan = await self.session.get(
            MigrationPlan,
            plan_id,
            options=[
                selectinload(MigrationPlan.steps).selectinload(
                    MigrationStep.dependencies
                ),
                selectinload(MigrationPlan.risks),
            ],
        )

        if not plan:
            msg = f"Plan {plan_id} not found"
            raise ValueError(msg)

        # Group steps by phase
        phases = self._group_steps_by_phase(plan.steps)

        # Calculate timeline
        timeline = self._calculate_timeline(plan, phases)

        # Identify critical path
        critical_path = self._find_critical_path(plan.steps)

        return {
            "plan_id": plan_id,
            "plan_name": plan.name,
            "strategy": plan.strategy,
            "phases": phases,
            "timeline": timeline,
            "critical_path": critical_path,
            "milestones": self._identify_milestones(plan, phases),
            "resource_allocation": self._generate_resource_allocation(plan, phases),
            "risk_mitigation_schedule": self._schedule_risk_mitigation(plan),
        }

    def _group_steps_by_phase(self, steps: list[MigrationStep]) -> list[dict[str, Any]]:
        """Group steps into logical phases.

        Args:
            steps: Migration steps

        Returns:
            List of phases with steps
        """
        # Define phase mapping
        phase_mapping = {
            "infrastructure_setup": "preparation",
            "test_creation": "preparation",
            "create_interface": "design",
            "extract_module": "extraction",
            "integration_test": "validation",
            "optimization": "optimization",
            "cleanup": "completion",
        }

        phases = {}
        for step in steps:
            phase = phase_mapping.get(step.step_type, "execution")
            if phase not in phases:
                phases[phase] = {
                    "name": phase.title(),
                    "steps": [],
                    "total_hours": 0,
                }

            phases[phase]["steps"].append(
                {
                    "id": step.id,
                    "name": step.name,
                    "type": step.step_type,
                    "hours": step.estimated_hours or 0,
                    "dependencies": [d.prerequisite_step_id for d in step.dependencies],
                }
            )
            phases[phase]["total_hours"] += step.estimated_hours or 0

        # Order phases
        phase_order = [
            "preparation",
            "design",
            "extraction",
            "validation",
            "optimization",
            "completion",
        ]

        return [phases[p] for p in phase_order if p in phases]

    def _calculate_timeline(
        self, plan: MigrationPlan, phases: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Calculate timeline for the migration.

        Args:
            plan: Migration plan
            phases: Grouped phases

        Returns:
            Timeline information
        """
        start_date = datetime.now(UTC)
        current_date = start_date

        timeline = {
            "start_date": start_date,
            "phases": [],
            "total_duration_weeks": 0,
        }

        for phase in phases:
            # Calculate phase duration
            phase_hours = phase["total_hours"]
            phase_weeks = (phase_hours / (40 * plan.team_size)) * 1.3  # 30% buffer

            phase_timeline = {
                "phase": phase["name"],
                "start_date": current_date,
                "end_date": current_date + timedelta(weeks=phase_weeks),
                "duration_weeks": round(phase_weeks, 1),
            }

            timeline["phases"].append(phase_timeline)
            current_date = phase_timeline["end_date"]

        timeline["end_date"] = current_date
        timeline["total_duration_weeks"] = round(
            (current_date - start_date).days / 7, 1
        )

        return timeline

    def _find_critical_path(self, steps: list[MigrationStep]) -> list[int]:
        """Find the critical path through migration steps.

        Args:
            steps: Migration steps

        Returns:
            List of step IDs in critical path
        """
        # Simple implementation - find longest dependency chain
        # In practice, would use proper CPM algorithm

        step_map = {step.id: step for step in steps}
        longest_path = []

        def find_path_from_step(step_id: int, current_path: list[int]) -> list[int]:
            step = step_map.get(step_id)
            if not step:
                return current_path

            current_path = [*current_path, step_id]

            # Find dependent steps
            dependent_ids = [
                dep.dependent_step_id
                for dep in step.dependents
                if dep.dependency_type == "hard"
            ]

            if not dependent_ids:
                return current_path

            # Recursively find longest path
            paths = [
                find_path_from_step(dep_id, current_path) for dep_id in dependent_ids
            ]

            return max(paths, key=len)

        # Find starting steps (no prerequisites)
        start_steps = [
            step
            for step in steps
            if not any(d.dependency_type == "hard" for d in step.dependencies)
        ]

        # Find longest path from each start
        for start in start_steps:
            path = find_path_from_step(start.id, [])
            if len(path) > len(longest_path):
                longest_path = path

        return longest_path

    def _identify_milestones(
        self, plan: MigrationPlan, phases: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Identify key milestones in the migration.

        Args:
            plan: Migration plan
            phases: Migration phases

        Returns:
            List of milestones
        """
        milestones = []

        # Phase completion milestones
        for i, phase in enumerate(phases):
            milestone = {
                "name": f"{phase['name']} Phase Complete",
                "description": f"All {phase['name'].lower()} tasks completed",
                "target_date": self._calculate_timeline(plan, phases)["phases"][i][
                    "end_date"
                ],
                "success_criteria": {
                    "all_phase_steps_complete": True,
                    "no_blocking_issues": True,
                },
            }
            milestones.append(milestone)

        # Add special milestones
        if plan.strategy == MigrationStrategy.STRANGLER_FIG.value:
            milestones.insert(
                0,
                {
                    "name": "Proxy Layer Operational",
                    "description": "Strangler fig proxy fully operational",
                    "success_criteria": {
                        "zero_downtime_achieved": True,
                        "monitoring_active": True,
                    },
                },
            )

        return milestones

    def _generate_resource_allocation(
        self, _plan: MigrationPlan, phases: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Generate resource allocation plan.

        Args:
            plan: Migration plan
            phases: Migration phases

        Returns:
            Resource allocation by phase
        """
        allocation = {}

        for phase in phases:
            phase_name = phase["name"]
            allocation[phase_name] = {
                "developers": 0,
                "architects": 0,
                "qa": 0,
                "devops": 0,
            }

            # Allocate based on step types
            for step in phase["steps"]:
                if step["type"] in ["extract_module", "optimization"]:
                    allocation[phase_name]["developers"] += 1
                elif step["type"] in ["create_interface", "infrastructure_setup"]:
                    allocation[phase_name]["architects"] += 1
                elif step["type"] in ["test_creation", "integration_test"]:
                    allocation[phase_name]["qa"] += 1
                elif step["type"] == "infrastructure_setup":
                    allocation[phase_name]["devops"] += 1

        return allocation

    def _schedule_risk_mitigation(self, plan: MigrationPlan) -> list[dict[str, Any]]:
        """Schedule risk mitigation activities.

        Args:
            plan: Migration plan

        Returns:
            Risk mitigation schedule
        """
        schedule = [
            {
                "risk_id": risk.id,
                "risk_name": risk.name,
                "mitigation_timing": (
                    "before_extraction" if risk.risk_type == "technical" else "ongoing"
                ),
                "mitigation_strategy": risk.mitigation_strategy,
                "owner": risk.owner or "team_lead",
            }
            for risk in plan.risks
            if risk.risk_level in ["critical", "high"]
        ]

        return schedule
