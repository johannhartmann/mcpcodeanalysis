"""Edge case and error scenario tests for migration intelligence."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.migration_models import (
    MigrationDependency,
    MigrationPlan,
    MigrationStep,
    MigrationStepStatus,
    MigrationStrategy,
)
from src.database.models import Repository
from src.database.package_models import Package, PackageDependency
from src.services.migration_analyzer import MigrationAnalyzer
from src.services.migration_executor import MigrationExecutor
from src.services.migration_monitor import MigrationMonitor
from src.services.migration_planner import MigrationPlanner
from src.services.pattern_library import PatternLibrary


@pytest.fixture
async def empty_repository(session: AsyncSession) -> Repository:
    """Create an empty repository for edge case testing."""
    repo = Repository(
        github_url="https://github.com/edge/empty-repo",
        owner="edge",
        name="empty-repo",
        default_branch="main",
    )
    session.add(repo)
    await session.commit()
    await session.refresh(repo)
    return repo


@pytest.fixture
async def circular_dependency_repository(session: AsyncSession) -> Repository:
    """Create a repository with circular dependencies."""
    repo = Repository(
        github_url="https://github.com/edge/circular-deps",
        owner="edge",
        name="circular-deps",
        default_branch="main",
    )
    session.add(repo)
    await session.commit()
    await session.refresh(repo)

    # Create packages with circular dependencies
    pkg_a = Package(
        repository_id=repo.id,
        path="src/package_a",
        name="package_a",
        module_count=5,
        total_lines=1000,
    )
    pkg_b = Package(
        repository_id=repo.id,
        path="src/package_b",
        name="package_b",
        module_count=3,
        total_lines=800,
    )
    pkg_c = Package(
        repository_id=repo.id,
        path="src/package_c",
        name="package_c",
        module_count=4,
        total_lines=600,
    )

    session.add_all([pkg_a, pkg_b, pkg_c])
    await session.commit()

    # Create circular dependencies: A -> B -> C -> A
    dep1 = PackageDependency(
        source_package_id=pkg_a.id,
        target_package_id=pkg_b.id,
        dependency_type="import",
    )
    dep2 = PackageDependency(
        source_package_id=pkg_b.id,
        target_package_id=pkg_c.id,
        dependency_type="import",
    )
    dep3 = PackageDependency(
        source_package_id=pkg_c.id,
        target_package_id=pkg_a.id,
        dependency_type="import",
    )

    session.add_all([dep1, dep2, dep3])
    await session.commit()

    return repo


@pytest.fixture
async def failed_migration_plan(
    session: AsyncSession, empty_repository: Repository
) -> MigrationPlan:
    """Create a migration plan with multiple failures."""
    plan = MigrationPlan(
        repository_id=empty_repository.id,
        name="Failed Migration Plan",
        description="Plan with multiple failures for testing",
        strategy=MigrationStrategy.GRADUAL,
        target_architecture="modular_monolith",
        team_size=5,
        created_by="test-failures",
    )
    session.add(plan)
    await session.commit()

    # Create steps with failures
    steps = []
    for i in range(5):
        step = MigrationStep(
            plan_id=plan.id,
            sequence_number=i,
            name=f"Failed Step {i}",
            description="Step that will fail",
            step_type="module_extraction",
            estimated_hours=8.0,
            status=MigrationStepStatus.FAILED if i < 3 else MigrationStepStatus.PENDING,
            notes="Test failure: dependency conflict" if i < 3 else None,
            started_at=datetime.now(UTC) - timedelta(hours=i + 1) if i < 3 else None,
            completed_at=datetime.now(UTC) - timedelta(hours=i) if i < 3 else None,
        )
        steps.append(step)

    session.add_all(steps)
    await session.commit()

    return plan


class TestMigrationEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.mark.asyncio
    async def test_empty_repository_analysis(
        self, session: AsyncSession, empty_repository: Repository
    ):
        """Test analysis of empty repository."""
        analyzer = MigrationAnalyzer(session)

        # Should handle empty repository gracefully
        result = await analyzer.analyze_repository_for_migration(empty_repository.id)

        assert result["repository_id"] == empty_repository.id
        assert result["bounded_contexts"] == []
        assert result["migration_candidates"] == []
        assert result["complexity_metrics"]["total_lines_of_code"] == 0
        assert result["recommended_strategy"] == "gradual"  # Default for empty
        assert result["readiness_score"] == 0.0

    @pytest.mark.asyncio
    async def test_circular_dependency_handling(
        self, session: AsyncSession, circular_dependency_repository: Repository
    ):
        """Test handling of circular dependencies."""
        analyzer = MigrationAnalyzer(session)

        # Analyze repository with circular dependencies
        result = await analyzer.analyze_repository_for_migration(
            circular_dependency_repository.id
        )

        # Should detect circular dependencies
        assert len(result["dependency_analysis"]["circular_dependencies"]) > 0
        assert result["dependency_analysis"]["dependency_bottlenecks"] > 0

        # Should lower readiness score due to circular deps
        assert result["readiness_score"] < 0.5

        # Risk assessment should flag circular dependencies
        risks = await analyzer.assess_migration_risks(circular_dependency_repository.id)

        circular_risk = next(
            (r for r in risks if "circular" in r["description"].lower()), None
        )
        assert circular_risk is not None
        assert circular_risk["level"] in ["high", "critical"]

    @pytest.mark.asyncio
    async def test_invalid_repository_id(self, session: AsyncSession):
        """Test handling of invalid repository ID."""
        analyzer = MigrationAnalyzer(session)

        with pytest.raises(ValueError, match="Repository 99999 not found"):
            await analyzer.analyze_repository_for_migration(99999)

    @pytest.mark.asyncio
    async def test_concurrent_step_execution_conflicts(
        self, session: AsyncSession, empty_repository: Repository
    ):
        """Test handling of concurrent execution conflicts."""
        # Create a plan with dependent steps
        planner = MigrationPlanner(session)
        plan = await planner.create_migration_plan(
            repository_id=empty_repository.id,
            name="Concurrent Conflict Test",
            description="Test concurrent execution conflicts",
            strategy=MigrationStrategy.GRADUAL,
            target_architecture="modular_monolith",
            team_size=5,
        )

        executor = MigrationExecutor(session)

        # Try to start two dependent steps concurrently
        if len(plan.steps) >= 2:
            step1 = plan.steps[0]
            step2 = plan.steps[1]

            # Add dependency: step2 depends on step1
            dep = MigrationDependency(
                dependent_step_id=step2.id,
                prerequisite_step_id=step1.id,
                dependency_type="sequential",
            )
            session.add(dep)
            await session.commit()

            # Start step1
            result1 = await executor.start_migration_step(step1.id)
            assert result1["success"] is True

            # Try to start step2 while step1 is still executing
            result2 = await executor.start_migration_step(step2.id)
            assert result2["success"] is False
            assert "dependencies not completed" in result2["message"].lower()

    @pytest.mark.asyncio
    async def test_rollback_without_strategy(
        self, session: AsyncSession, empty_repository: Repository
    ):
        """Test rollback attempt without rollback strategy defined."""
        planner = MigrationPlanner(session)
        plan = await planner.create_migration_plan(
            repository_id=empty_repository.id,
            name="No Rollback Strategy Test",
            description="Test rollback without strategy",
            strategy=MigrationStrategy.GRADUAL,
            target_architecture="modular_monolith",
            team_size=5,
        )

        executor = MigrationExecutor(session)
        step = plan.steps[0]

        # Remove rollback strategy
        step.rollback_procedure = None
        await session.commit()

        # Execute step
        await executor.start_migration_step(step.id)
        await executor.complete_migration_step(step.id, success=True)

        # Try to rollback
        result = await executor.rollback_migration_step(step.id, reason="Test rollback")

        assert result["success"] is False
        assert "no rollback strategy" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_pattern_extraction_from_failed_plan(
        self, session: AsyncSession, failed_migration_plan: MigrationPlan
    ):
        """Test pattern extraction from failed migration plan."""
        library = PatternLibrary(session)

        # Extract patterns from failed plan
        patterns = await library.extract_patterns_from_execution(
            failed_migration_plan.id
        )

        # Should not extract patterns from failed steps
        assert len(patterns) == 0 or all(
            p.get("source") != "failed_step" for p in patterns
        )

        # Learn from failures
        lessons = await library.learn_from_failures(failed_migration_plan.id)

        assert len(lessons["failure_analysis"]) >= 3  # 3 failed steps
        assert len(lessons["root_causes"]) > 0
        assert len(lessons["prevention_strategies"]) > 0

    @pytest.mark.asyncio
    async def test_resource_planning_with_zero_resources(
        self, session: AsyncSession, empty_repository: Repository
    ):
        """Test resource planning with no available resources."""
        planner = MigrationPlanner(session)
        plan = await planner.create_migration_plan(
            repository_id=empty_repository.id,
            name="Zero Resources Test",
            description="Test with no resources",
            strategy=MigrationStrategy.GRADUAL,
            target_architecture="modular_monolith",
            team_size=5,
        )

        # Plan with zero resources
        with pytest.raises(ValueError, match="No resources available"):
            await planner.plan_migration_resources(
                plan.id,
                available_developers=0,
                available_architects=0,
                available_qa=0,
            )

    @pytest.mark.asyncio
    async def test_anomaly_detection_edge_cases(
        self, session: AsyncSession, empty_repository: Repository
    ):
        """Test anomaly detection edge cases."""
        planner = MigrationPlanner(session)
        plan = await planner.create_migration_plan(
            repository_id=empty_repository.id,
            name="Anomaly Edge Cases",
            description="Test anomaly detection",
            strategy=MigrationStrategy.GRADUAL,
            target_architecture="modular_monolith",
            team_size=5,
        )

        monitor = MigrationMonitor(session)
        executor = MigrationExecutor(session)

        if plan.steps:
            step = plan.steps[0]

            # Create anomalous execution pattern
            # Multiple starts without completion
            for _ in range(5):
                await executor.start_migration_step(step.id)
                # Reset status for next attempt
                step.status = MigrationStepStatus.PENDING
                await session.commit()

            # Detect anomalies
            anomalies = await monitor.detect_anomalies(plan.id)

            # Should detect repeated starts
            repeated_starts = [a for a in anomalies if a["type"] == "repeated_starts"]
            assert len(repeated_starts) > 0

    @pytest.mark.asyncio
    async def test_optimization_with_no_dependencies(
        self, session: AsyncSession, empty_repository: Repository
    ):
        """Test optimization of plan with no step dependencies."""
        planner = MigrationPlanner(session)
        plan = await planner.create_migration_plan(
            repository_id=empty_repository.id,
            name="No Dependencies Test",
            description="All steps independent",
            strategy=MigrationStrategy.GRADUAL,
            target_architecture="modular_monolith",
            team_size=10,
        )

        # Remove all dependencies
        await session.execute(
            MigrationDependency.__table__.delete().where(
                MigrationDependency.dependent_step_id.in_([s.id for s in plan.steps])
            )
        )
        await session.commit()

        from src.utils.migration_performance import MigrationPerformanceOptimizer

        optimizer = MigrationPerformanceOptimizer(session)
        result = await optimizer.analyze_parallelization_opportunities(plan.id)

        # All steps should be in first execution level
        assert len(result["execution_levels"]) == 1
        assert result["execution_levels"][0]["step_count"] == len(plan.steps)
        # Maximum parallelization possible
        assert result["time_savings_percentage"] > 80  # High savings

    @pytest.mark.asyncio
    async def test_health_check_with_stalled_migration(
        self, session: AsyncSession, empty_repository: Repository
    ):
        """Test health check detection of stalled migration."""
        planner = MigrationPlanner(session)
        plan = await planner.create_migration_plan(
            repository_id=empty_repository.id,
            name="Stalled Migration Test",
            description="Test stalled detection",
            strategy=MigrationStrategy.GRADUAL,
            target_architecture="modular_monolith",
            team_size=5,
        )

        executor = MigrationExecutor(session)
        monitor = MigrationMonitor(session)

        if plan.steps:
            step = plan.steps[0]

            # Start step but simulate stall (no completion for long time)
            await executor.start_migration_step(step.id)

            # Manually set started_at to past
            step.started_at = datetime.now(UTC) - timedelta(days=7)
            await session.commit()

            # Check health
            health = await monitor.check_migration_health(plan.id)

            assert health["overall_health_score"] < 50  # Poor health
            assert any(
                "stalled" in check["message"].lower()
                for check in health["health_checks"]
            )

    @pytest.mark.asyncio
    async def test_interface_design_for_nonexistent_package(
        self, session: AsyncSession
    ):
        """Test interface design for non-existent package."""
        from src.services.interface_designer import InterfaceDesigner

        designer = InterfaceDesigner(session)

        with pytest.raises(ValueError, match="Package 99999 not found"):
            await designer.design_module_interface(
                99999, target_architecture="modular_monolith"
            )

    @pytest.mark.asyncio
    async def test_plan_optimization_with_conflicting_goals(
        self, session: AsyncSession, empty_repository: Repository
    ):
        """Test plan optimization with all goals set to minimize/maximize."""
        planner = MigrationPlanner(session)
        plan = await planner.create_migration_plan(
            repository_id=empty_repository.id,
            name="Conflicting Goals Test",
            description="Test conflicting optimization goals",
            strategy=MigrationStrategy.GRADUAL,
            target_architecture="modular_monolith",
            team_size=5,
        )

        # Try to optimize with conflicting goals
        optimized = await planner.optimize_migration_plan(
            plan.id,
            minimize_time=True,
            minimize_risk=True,  # Conflicts with minimize_time
            maximize_quality=True,  # Also conflicts with minimize_time
        )

        # Should still return a valid plan
        assert optimized.id == plan.id
        # But should balance conflicting goals
        assert len(optimized.steps) >= len(plan.steps)

    @pytest.mark.asyncio
    async def test_validation_with_impossible_criteria(
        self, session: AsyncSession, empty_repository: Repository
    ):
        """Test validation with impossible criteria."""
        planner = MigrationPlanner(session)
        plan = await planner.create_migration_plan(
            repository_id=empty_repository.id,
            name="Impossible Validation Test",
            description="Test impossible validation criteria",
            strategy=MigrationStrategy.GRADUAL,
            target_architecture="modular_monolith",
            team_size=5,
        )

        executor = MigrationExecutor(session)

        if plan.steps:
            step = plan.steps[0]
            await executor.start_migration_step(step.id)
            await executor.complete_migration_step(step.id, success=True)

            # Validate with impossible criteria
            result = await executor.validate_migration_step(
                step.id,
                validation_type="automated",
                validation_criteria={
                    "test_coverage_required": 200,  # Impossible
                    "response_time_ms": 0,  # Impossible
                },
            )

            assert result["passed"] is False
            assert "criteria" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_concurrent_plan_modifications(
        self, session: AsyncSession, empty_repository: Repository
    ):
        """Test handling of concurrent plan modifications."""
        planner = MigrationPlanner(session)
        plan = await planner.create_migration_plan(
            repository_id=empty_repository.id,
            name="Concurrent Modification Test",
            description="Test concurrent modifications",
            strategy=MigrationStrategy.GRADUAL,
            target_architecture="modular_monolith",
            team_size=5,
        )

        # Simulate concurrent modifications
        async def modify_plan(plan_id: int, modification: str):
            """Simulate plan modification."""
            p = await session.get(MigrationPlan, plan_id)
            if p:
                p.description = f"{p.description} - {modification}"
                await session.commit()

        # Run concurrent modifications
        tasks = [modify_plan(plan.id, f"Modification {i}") for i in range(5)]

        # Should handle concurrent modifications gracefully
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # At least some modifications should succeed
        successful = [r for r in results if not isinstance(r, Exception)]
        assert len(successful) > 0

    @pytest.mark.asyncio
    async def test_pattern_update_with_invalid_data(self, session: AsyncSession):
        """Test pattern update with invalid execution data."""
        library = PatternLibrary(session)

        # Create a test pattern
        pattern_data = {
            "name": "Test Pattern",
            "category": "extraction",
            "description": "Test pattern for invalid update",
            "implementation_steps": ["Step 1"],
            "prerequisites": [],
            "best_practices": [],
        }

        pattern = await library.add_pattern_to_library(pattern_data)

        # Try to update with invalid data
        with pytest.raises(ValueError, match="Invalid|Pattern"):
            await library.update_pattern_from_execution(
                pattern.id,
                {
                    "success": "not_a_boolean",  # Invalid type
                    "actual_hours": -10,  # Invalid value
                },
            )

    @pytest.mark.asyncio
    async def test_timeline_generation_for_empty_plan(
        self, session: AsyncSession, empty_repository: Repository
    ):
        """Test timeline generation for plan with no steps."""
        # Create plan with no steps
        plan = MigrationPlan(
            repository_id=empty_repository.id,
            name="Empty Plan",
            description="Plan with no steps",
            strategy=MigrationStrategy.GRADUAL,
            target_architecture="modular_monolith",
            team_size=5,
            created_by="test",
        )
        session.add(plan)
        await session.commit()

        monitor = MigrationMonitor(session)
        timeline = await monitor.get_migration_timeline(plan.id)

        assert timeline["timeline_events"] == []
        assert timeline["estimated_completion"] is None
        assert timeline["critical_path"] == []
