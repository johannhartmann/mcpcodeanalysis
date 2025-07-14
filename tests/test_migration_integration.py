"""Integration tests for migration intelligence features."""

import asyncio

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.migration_models import (
    MigrationPattern,
    MigrationStepStatus,
    MigrationStrategy,
)
from src.database.models import Repository
from src.services.interface_designer import InterfaceDesigner
from src.services.migration_analyzer import MigrationAnalyzer
from src.services.migration_executor import MigrationExecutor
from src.services.migration_monitor import MigrationMonitor
from src.services.migration_planner import MigrationPlanner
from src.services.pattern_library import PatternLibrary


@pytest.fixture
async def sample_repository(session: AsyncSession) -> Repository:
    """Create a sample repository for testing."""
    repo = Repository(
        github_url="https://github.com/test/monolith",
        owner="test",
        name="monolith",
        default_branch="main",
    )
    session.add(repo)
    await session.commit()
    await session.refresh(repo)
    return repo


@pytest.fixture
async def sample_pattern(session: AsyncSession) -> MigrationPattern:
    """Create a sample migration pattern."""
    pattern = MigrationPattern(
        name="Test Extraction Pattern",
        category="extraction",
        description="Pattern for extracting modules",
        implementation_steps=["Step 1", "Step 2", "Step 3"],
        applicable_scenarios={
            "size_category": "medium",
            "complexity_category": "medium",
        },
        prerequisites=["Test coverage > 80%"],
        best_practices=["Use interfaces", "Incremental approach"],
        success_rate=0.85,
        usage_count=5,
        avg_effort_hours=40.0,
    )
    session.add(pattern)
    await session.commit()
    await session.refresh(pattern)
    return pattern


class TestMigrationIntegration:
    """Integration tests for the complete migration workflow."""

    @pytest.mark.asyncio
    async def test_full_migration_workflow(
        self, session: AsyncSession, sample_repository: Repository
    ):
        """Test the complete migration workflow from analysis to execution."""
        # 1. Analyze repository for migration
        analyzer = MigrationAnalyzer(session)
        analysis = await analyzer.analyze_repository_for_migration(sample_repository.id)

        assert analysis["repository_id"] == sample_repository.id
        assert "bounded_contexts" in analysis
        assert "migration_candidates" in analysis
        assert "recommended_strategy" in analysis

        # 2. Create migration plan
        planner = MigrationPlanner(session)
        plan = await planner.create_migration_plan(
            repository_id=sample_repository.id,
            name="Test Migration Plan",
            description="Integration test plan",
            strategy=MigrationStrategy.GRADUAL,
            target_architecture="modular_monolith",
            team_size=5,
            risk_tolerance="medium",
        )

        assert plan.repository_id == sample_repository.id
        assert plan.strategy == MigrationStrategy.GRADUAL
        assert len(plan.steps) > 0

        # 3. Execute a migration step
        executor = MigrationExecutor(session)
        first_step = plan.steps[0]

        # Start execution
        start_result = await executor.start_migration_step(
            first_step.id, executor_id="test-executor"
        )
        assert start_result["success"] is True
        assert "execution_id" in start_result

        # Complete execution
        complete_result = await executor.complete_migration_step(
            first_step.id,
            success=True,
            notes="Test completion",
            validation_results={
                "type": "automated",
                "passed": True,
                "results": {"tests_passed": 10},
            },
        )
        assert complete_result["success"] is True

        # 4. Monitor progress
        monitor = MigrationMonitor(session)
        progress = await executor.track_migration_progress(plan.id)

        # Use monitor for health check
        health = await monitor.check_migration_health(plan.id)
        assert health["status"] == "healthy"

        assert progress["plan_id"] == plan.id
        assert progress["progress_summary"]["completed_steps"] == 1
        assert progress["health_score"] > 0

        # 5. Extract patterns from execution
        library = PatternLibrary(session)
        patterns = await library.extract_patterns_from_execution(plan.id)

        assert isinstance(patterns, list)
        # Pattern extraction may not always yield patterns from simple test

    @pytest.mark.asyncio
    async def test_migration_analysis_pipeline(
        self, session: AsyncSession, sample_repository: Repository
    ):
        """Test the migration analysis pipeline."""
        analyzer = MigrationAnalyzer(session)

        # Test bounded context analysis
        contexts = await analyzer._analyze_bounded_contexts(sample_repository.id)
        assert isinstance(contexts, list)

        # Test candidate identification
        candidates = await analyzer._identify_migration_candidates(sample_repository.id)
        assert isinstance(candidates, list)

        # Test dependency analysis
        dependencies = await analyzer._analyze_dependencies(sample_repository.id)
        assert "circular_dependencies" in dependencies
        assert "high_coupling_packages" in dependencies

        # Test complexity metrics
        complexity = await analyzer._calculate_complexity_metrics(sample_repository.id)
        assert "total_lines_of_code" in complexity
        assert "complexity_rating" in complexity

    @pytest.mark.asyncio
    async def test_migration_planning_features(
        self, session: AsyncSession, sample_repository: Repository
    ):
        """Test migration planning features."""
        planner = MigrationPlanner(session)

        # Create a plan
        plan = await planner.create_migration_plan(
            repository_id=sample_repository.id,
            name="Planning Test",
            description="Test planning features",
            strategy=MigrationStrategy.STRANGLER_FIG,
            target_architecture="microservices",
            team_size=3,
        )

        # Test roadmap generation
        roadmap = await planner.generate_migration_roadmap(plan.id)
        assert "phases" in roadmap
        assert "milestones" in roadmap
        assert "critical_path" in roadmap

        # Test plan optimization
        optimized = await planner.optimize_migration_plan(
            plan.id, minimize_time=True, minimize_risk=False
        )
        assert optimized.id == plan.id
        # Optimization may not change much in test scenario

        # Test resource planning
        resources = await planner.plan_migration_resources(
            plan.id,
            available_developers=3,
            available_architects=1,
            available_qa=1,
        )
        assert "allocation_plan" in resources
        assert "resource_requirements" in resources

    @pytest.mark.asyncio
    async def test_interface_design_generation(
        self, session: AsyncSession, sample_repository: Repository
    ):
        """Test interface design generation."""
        # Create a dummy package for testing
        from src.database.package_models import Package

        package = Package(
            repository_id=sample_repository.id,
            path="src/billing",
            name="billing",
            module_count=5,
            total_lines=1000,
        )
        session.add(package)
        await session.commit()

        designer = InterfaceDesigner(session)
        interface = await designer.design_module_interface(
            package.id, target_architecture="modular_monolith"
        )

        assert interface["package_id"] == package.id
        assert "public_api" in interface
        assert "data_contracts" in interface
        assert "implementation_notes" in interface

    @pytest.mark.asyncio
    async def test_execution_monitoring_integration(
        self, session: AsyncSession, sample_repository: Repository
    ):
        """Test execution and monitoring integration."""
        # Create a plan with multiple steps
        planner = MigrationPlanner(session)
        plan = await planner.create_migration_plan(
            repository_id=sample_repository.id,
            name="Monitoring Test",
            description="Test monitoring features",
            strategy=MigrationStrategy.GRADUAL,
            target_architecture="modular_monolith",
            team_size=5,
        )

        executor = MigrationExecutor(session)
        monitor = MigrationMonitor(session)

        # Execute first step
        first_step = plan.steps[0]
        await executor.start_migration_step(first_step.id)

        # Monitor execution
        step_status = await monitor.monitor_step_execution(first_step.id)
        assert step_status["step_id"] == first_step.id
        assert step_status["status"] == "executing"

        # Complete step
        await executor.complete_migration_step(first_step.id, success=True)

        # Check health
        health = await monitor.check_migration_health(plan.id)
        assert "overall_health_score" in health
        assert "health_checks" in health
        assert health["overall_health_score"] > 0

        # Generate timeline
        timeline = await monitor.get_migration_timeline(plan.id)
        assert "timeline_events" in timeline
        assert len(timeline["timeline_events"]) > 0

    @pytest.mark.asyncio
    async def test_pattern_library_integration(
        self,
        session: AsyncSession,
        sample_repository: Repository,
        sample_pattern: MigrationPattern,
    ):
        """Test pattern library integration."""
        library = PatternLibrary(session)

        # Search patterns
        patterns = await library.search_patterns(
            category="extraction", min_success_rate=0.8
        )
        assert len(patterns) >= 1
        assert patterns[0]["name"] == sample_pattern.name

        # Get recommendations
        recommendations = await library.get_pattern_recommendations(
            sample_repository.id,
            context={"team_size": 5, "complexity": "medium"},
        )
        assert isinstance(recommendations, list)

        # Update pattern from execution
        await library.update_pattern_from_execution(
            sample_pattern.id,
            execution_data={
                "success": True,
                "actual_hours": 35,
                "scenario": {"size_category": "medium"},
            },
        )

        # Verify update
        updated_pattern = await session.get(MigrationPattern, sample_pattern.id)
        assert updated_pattern.usage_count == 6
        assert updated_pattern.last_used_at is not None

    @pytest.mark.asyncio
    async def test_failure_learning_integration(
        self, session: AsyncSession, sample_repository: Repository
    ):
        """Test learning from failures."""
        # Create a plan with failed steps
        planner = MigrationPlanner(session)
        plan = await planner.create_migration_plan(
            repository_id=sample_repository.id,
            name="Failure Test",
            description="Test failure learning",
            strategy=MigrationStrategy.GRADUAL,
            target_architecture="modular_monolith",
            team_size=5,
        )

        # Mark a step as failed
        executor = MigrationExecutor(session)
        failed_step = plan.steps[0]

        await executor.start_migration_step(failed_step.id)
        await executor.complete_migration_step(
            failed_step.id,
            success=False,
            notes="Test failure: dependency issues",
        )

        # Learn from failures
        library = PatternLibrary(session)
        lessons = await library.learn_from_failures(plan.id)

        assert "failure_analysis" in lessons
        assert "root_causes" in lessons
        assert "prevention_strategies" in lessons
        assert len(lessons["failure_analysis"]) > 0

    @pytest.mark.asyncio
    async def test_risk_assessment_integration(
        self, session: AsyncSession, sample_repository: Repository
    ):
        """Test risk assessment integration."""
        analyzer = MigrationAnalyzer(session)

        # Create a plan
        planner = MigrationPlanner(session)
        plan = await planner.create_migration_plan(
            repository_id=sample_repository.id,
            name="Risk Test",
            description="Test risk assessment",
            strategy=MigrationStrategy.PARALLEL_RUN,
            target_architecture="microservices",
            team_size=3,
        )

        # Assess risks
        risks = await analyzer.assess_migration_risks(
            sample_repository.id, plan_id=plan.id
        )

        assert isinstance(risks, list)
        # Check risk structure
        if risks:
            risk = risks[0]
            assert "type" in risk
            assert "probability" in risk
            assert "impact" in risk
            assert "level" in risk
            assert "mitigation" in risk

    @pytest.mark.asyncio
    async def test_migration_validation_flow(
        self, session: AsyncSession, sample_repository: Repository
    ):
        """Test the migration validation flow."""
        # Create and execute a plan
        planner = MigrationPlanner(session)
        plan = await planner.create_migration_plan(
            repository_id=sample_repository.id,
            name="Validation Test",
            description="Test validation flow",
            strategy=MigrationStrategy.GRADUAL,
            target_architecture="modular_monolith",
            team_size=5,
        )

        executor = MigrationExecutor(session)
        step = plan.steps[0]

        # Execute step
        await executor.start_migration_step(step.id)
        await executor.complete_migration_step(step.id, success=True)

        # Validate step
        validation_result = await executor.validate_migration_step(
            step.id,
            validation_type="automated",
            validation_criteria={
                "test_coverage_required": 80,
                "performance_baseline": {"response_time_ms": 100},
            },
        )

        assert validation_result["step_id"] == step.id
        assert "validation_id" in validation_result
        assert "passed" in validation_result

    @pytest.mark.asyncio
    async def test_anomaly_detection(
        self, session: AsyncSession, sample_repository: Repository
    ):
        """Test anomaly detection in migrations."""
        # Create a plan with some anomalies
        planner = MigrationPlanner(session)
        plan = await planner.create_migration_plan(
            repository_id=sample_repository.id,
            name="Anomaly Test",
            description="Test anomaly detection",
            strategy=MigrationStrategy.GRADUAL,
            target_architecture="modular_monolith",
            team_size=5,
        )

        # Execute with some issues
        executor = MigrationExecutor(session)
        monitor = MigrationMonitor(session)

        # Create multiple failed attempts for a step
        step = plan.steps[0]

        # First failure
        await executor.start_migration_step(step.id)
        await executor.complete_migration_step(step.id, success=False)

        # Reset step for second attempt
        step.status = MigrationStepStatus.PENDING
        await session.commit()

        # Second failure
        await executor.start_migration_step(step.id)
        await executor.complete_migration_step(step.id, success=False)

        # Detect anomalies
        anomalies = await monitor.detect_anomalies(plan.id)

        assert isinstance(anomalies, list)
        # Should detect repeated failures
        repeated_failures = [a for a in anomalies if a["type"] == "repeated_failures"]
        assert len(repeated_failures) > 0

    @pytest.mark.asyncio
    async def test_concurrent_step_execution(
        self, session: AsyncSession, sample_repository: Repository
    ):
        """Test concurrent execution of independent steps."""
        planner = MigrationPlanner(session)
        plan = await planner.create_migration_plan(
            repository_id=sample_repository.id,
            name="Concurrent Test",
            description="Test concurrent execution",
            strategy=MigrationStrategy.GRADUAL,
            target_architecture="modular_monolith",
            team_size=10,  # Larger team for parallel work
        )

        executor = MigrationExecutor(session)

        # Find independent steps (no dependencies on each other)
        independent_steps = []
        for step in plan.steps:
            if not step.dependencies:
                independent_steps.append(step)
                if len(independent_steps) >= 2:
                    break

        # Execute steps concurrently
        if len(independent_steps) >= 2:
            tasks = [
                executor.start_migration_step(step.id) for step in independent_steps
            ]

            results = await asyncio.gather(*tasks)

            # All should succeed
            for result in results:
                assert result["success"] is True

    @pytest.mark.asyncio
    async def test_migration_rollback(
        self, session: AsyncSession, sample_repository: Repository
    ):
        """Test migration rollback functionality."""
        planner = MigrationPlanner(session)
        plan = await planner.create_migration_plan(
            repository_id=sample_repository.id,
            name="Rollback Test",
            description="Test rollback functionality",
            strategy=MigrationStrategy.GRADUAL,
            target_architecture="modular_monolith",
            team_size=5,
        )

        executor = MigrationExecutor(session)
        step = plan.steps[0]

        # Execute step
        await executor.start_migration_step(step.id)
        await executor.complete_migration_step(step.id, success=True)

        # Rollback step
        rollback_result = await executor.rollback_migration_step(
            step.id, reason="Test rollback"
        )

        # Rollback should succeed if strategy is defined
        if step.rollback_procedure:
            assert rollback_result["success"] is True
            assert "rollback_execution_id" in rollback_result
        else:
            assert rollback_result["success"] is False

    @pytest.mark.asyncio
    async def test_knowledge_sharing(
        self, session: AsyncSession, sample_pattern: MigrationPattern
    ):
        """Test knowledge sharing capabilities."""
        library = PatternLibrary(session)

        # Generate pattern documentation
        doc = await library.generate_pattern_documentation(sample_pattern.id)

        assert isinstance(doc, str)
        assert sample_pattern.name in doc
        assert "## Description" in doc
        assert "## Prerequisites" in doc

        # Test that documentation includes all sections
        expected_sections = [
            "Description",
            "Prerequisites",
            "Implementation Steps",
            "Best Practices",
        ]

        for section in expected_sections:
            if getattr(sample_pattern, section.lower().replace(" ", "_"), None):
                assert f"## {section}" in doc
