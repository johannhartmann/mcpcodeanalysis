"""Performance benchmark tests for migration intelligence features."""

import asyncio
import time
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.migration_models import (
    MigrationDependency,
    MigrationPlan,
    MigrationStep,
    MigrationStrategy,
)
from src.database.models import Repository
from src.database.package_models import Package, PackageDependency
from src.services.migration_analyzer import MigrationAnalyzer
from src.services.migration_planner import MigrationPlanner
from src.utils.migration_performance import MigrationPerformanceOptimizer


class BenchmarkMetrics:
    """Helper class to track benchmark metrics."""

    def __init__(self):
        self.metrics = {}

    def record(self, name: str, duration: float, **kwargs):
        """Record a benchmark metric."""
        self.metrics[name] = {
            "duration_seconds": duration,
            "timestamp": datetime.now(UTC),
            **kwargs,
        }

    def report(self) -> dict[str, Any]:
        """Generate benchmark report."""
        total_duration = sum(m["duration_seconds"] for m in self.metrics.values())
        return {
            "total_duration_seconds": total_duration,
            "metrics": self.metrics,
            "summary": {
                "slowest": max(
                    self.metrics.items(),
                    key=lambda x: x[1]["duration_seconds"],
                )[0],
                "fastest": min(
                    self.metrics.items(),
                    key=lambda x: x[1]["duration_seconds"],
                )[0],
            },
        }


@pytest.fixture
async def large_repository(session: AsyncSession) -> Repository:
    """Create a large repository for benchmarking."""
    repo = Repository(
        github_url="https://github.com/bench/large-monolith",
        owner="bench",
        name="large-monolith",
        default_branch="main",
    )
    session.add(repo)
    await session.commit()
    await session.refresh(repo)
    return repo


@pytest.fixture
async def generate_large_codebase(
    session: AsyncSession, large_repository: Repository
) -> list[Package]:
    """Generate a large simulated codebase with packages and dependencies."""
    packages = []

    # Create 1000 packages in different domains
    domains = [
        "billing",
        "auth",
        "inventory",
        "shipping",
        "customer",
        "order",
        "payment",
        "notification",
        "reporting",
        "admin",
    ]

    for domain_idx, domain in enumerate(domains):
        for i in range(100):  # 100 packages per domain
            package = Package(
                repository_id=large_repository.id,
                path=f"src/{domain}/module_{i}",
                name=f"{domain}_module_{i}",
                module_count=5 + (i % 10),
                total_lines=1000 + (i * 50),
            )
            packages.append(package)

    session.add_all(packages)
    await session.commit()

    # Refresh packages to get IDs
    for pkg in packages:
        await session.refresh(pkg)

    # Create realistic dependency patterns
    dependencies = []

    # Intra-domain dependencies (higher coupling within domains)
    for domain_idx in range(len(domains)):
        domain_packages = packages[domain_idx * 100 : (domain_idx + 1) * 100]
        for i, pkg in enumerate(domain_packages):
            # Each package depends on 2-5 others in same domain
            for j in range(2, min(5, i)):
                if i - j >= 0:
                    dep = PackageDependency(
                        source_package_id=pkg.id,
                        target_package_id=domain_packages[i - j].id,
                        dependency_type="import",
                    )
                    dependencies.append(dep)

    # Inter-domain dependencies (lower coupling between domains)
    for i, pkg in enumerate(packages):
        # Each package has 0-2 dependencies on other domains
        for _ in range(min(2, i % 3)):
            target_idx = (i * 7 + 13) % len(packages)  # Pseudo-random
            if packages[target_idx].id != pkg.id:
                dep = PackageDependency(
                    source_package_id=pkg.id,
                    target_package_id=packages[target_idx].id,
                    dependency_type="import",
                )
                dependencies.append(dep)

    session.add_all(dependencies)
    await session.commit()

    return packages


@pytest.fixture
async def complex_migration_plan(
    session: AsyncSession, large_repository: Repository
) -> MigrationPlan:
    """Create a complex migration plan with many steps and dependencies."""
    plan = MigrationPlan(
        repository_id=large_repository.id,
        name="Large-Scale Migration Benchmark",
        description="Benchmark plan with 100+ steps",
        strategy=MigrationStrategy.STRANGLER_FIG,
        target_architecture="microservices",
        team_size=10,
        created_by="benchmark-test",
    )
    session.add(plan)
    await session.commit()
    await session.refresh(plan)

    # Create 100 migration steps
    steps = []
    for i in range(100):
        step = MigrationStep(
            plan_id=plan.id,
            sequence_number=i,
            name=f"Migration Step {i}",
            description=f"Benchmark step {i}",
            step_type="module_extraction" if i % 3 == 0 else "interface_definition",
            estimated_hours=8.0 + (i % 5) * 2,
        )
        steps.append(step)

    session.add_all(steps)
    await session.commit()

    # Refresh steps to get IDs
    for step in steps:
        await session.refresh(step)

    # Create complex dependency graph
    dependencies = []
    for i, step in enumerate(steps):
        # Each step depends on 0-3 previous steps
        for j in range(min(3, i)):
            if i - j - 1 >= 0:
                dep = MigrationDependency(
                    dependent_step_id=step.id,
                    prerequisite_step_id=steps[i - j - 1].id,
                    dependency_type="sequential",
                )
                dependencies.append(dep)

    session.add_all(dependencies)
    await session.commit()

    return plan


class TestMigrationBenchmarks:
    """Benchmark tests for migration intelligence performance."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_large_repository_analysis(
        self,
        session: AsyncSession,
        large_repository: Repository,
        generate_large_codebase: list[Package],
    ):
        """Benchmark analysis of large repository (1000+ packages)."""
        metrics = BenchmarkMetrics()
        analyzer = MigrationAnalyzer(session)

        # Benchmark bounded context analysis
        start = time.time()
        contexts = await analyzer._analyze_bounded_contexts(large_repository.id)
        duration = time.time() - start
        metrics.record(
            "bounded_context_analysis",
            duration,
            package_count=len(generate_large_codebase),
            contexts_found=len(contexts),
        )

        # Benchmark candidate identification
        start = time.time()
        candidates = await analyzer._identify_migration_candidates(large_repository.id)
        duration = time.time() - start
        metrics.record(
            "candidate_identification",
            duration,
            candidates_found=len(candidates),
        )

        # Benchmark dependency analysis
        start = time.time()
        dependencies = await analyzer._analyze_dependencies(large_repository.id)
        duration = time.time() - start
        metrics.record(
            "dependency_analysis",
            duration,
            circular_deps=len(dependencies.get("circular_dependencies", [])),
        )

        # Benchmark complexity calculation
        start = time.time()
        complexity = await analyzer._calculate_complexity_metrics(large_repository.id)
        duration = time.time() - start
        metrics.record(
            "complexity_calculation",
            duration,
            total_loc=complexity.get("total_lines_of_code", 0),
        )

        # Full analysis benchmark
        start = time.time()
        full_analysis = await analyzer.analyze_repository_for_migration(
            large_repository.id
        )
        duration = time.time() - start
        metrics.record("full_analysis", duration)

        # Generate report
        report = metrics.report()

        # Performance assertions
        assert (
            report["metrics"]["full_analysis"]["duration_seconds"] < 30.0
        )  # Should complete within 30s
        assert report["metrics"]["bounded_context_analysis"]["duration_seconds"] < 10.0
        assert report["metrics"]["dependency_analysis"]["duration_seconds"] < 5.0

        print(f"\n📊 Large Repository Analysis Benchmark Report:")
        print(f"Total duration: {report['total_duration_seconds']:.2f}s")
        print(f"Slowest operation: {report['summary']['slowest']}")
        print(f"Package count: {len(generate_large_codebase)}")

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_complex_plan_optimization(
        self,
        session: AsyncSession,
        complex_migration_plan: MigrationPlan,
    ):
        """Benchmark optimization of complex migration plans."""
        metrics = BenchmarkMetrics()
        optimizer = MigrationPerformanceOptimizer(session)

        # Benchmark parallelization analysis
        start = time.time()
        parallel_analysis = await optimizer.analyze_parallelization_opportunities(
            complex_migration_plan.id
        )
        duration = time.time() - start
        metrics.record(
            "parallelization_analysis",
            duration,
            execution_levels=len(parallel_analysis["execution_levels"]),
            max_parallel=parallel_analysis["max_parallel_tasks"],
            time_savings_pct=parallel_analysis["time_savings_percentage"],
        )

        # Benchmark bottleneck identification
        start = time.time()
        bottlenecks = await optimizer.identify_migration_bottlenecks(
            complex_migration_plan.id
        )
        duration = time.time() - start
        metrics.record(
            "bottleneck_identification",
            duration,
            bottleneck_count=bottlenecks["bottleneck_count"],
        )

        # Benchmark resource optimization
        start = time.time()
        resource_opt = await optimizer.calculate_resource_optimization(
            complex_migration_plan.id,
            available_resources={
                "developers": 10,
                "architects": 2,
                "qa": 3,
            },
        )
        duration = time.time() - start
        metrics.record(
            "resource_optimization",
            duration,
            estimated_weeks=resource_opt["estimated_duration_weeks"],
        )

        # Generate report
        report = metrics.report()

        # Performance assertions
        assert report["metrics"]["parallelization_analysis"]["duration_seconds"] < 5.0
        assert report["metrics"]["bottleneck_identification"]["duration_seconds"] < 3.0
        assert report["metrics"]["resource_optimization"]["duration_seconds"] < 2.0

        print(f"\n📊 Complex Plan Optimization Benchmark Report:")
        print(f"Total duration: {report['total_duration_seconds']:.2f}s")
        print(
            f"Time savings from parallelization: {parallel_analysis['time_savings_percentage']:.1f}%"
        )
        print(f"Bottlenecks found: {bottlenecks['bottleneck_count']}")

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_concurrent_plan_creation(
        self,
        session: AsyncSession,
        large_repository: Repository,
        generate_large_codebase: list[Package],
    ):
        """Benchmark concurrent migration plan creation."""
        metrics = BenchmarkMetrics()
        planner = MigrationPlanner(session)

        # Create multiple plans concurrently
        plan_configs = [
            {
                "name": f"Concurrent Plan {i}",
                "strategy": (
                    MigrationStrategy.GRADUAL
                    if i % 2 == 0
                    else MigrationStrategy.STRANGLER_FIG
                ),
                "team_size": 5 + i,
            }
            for i in range(5)
        ]

        start = time.time()
        tasks = [
            planner.create_migration_plan(
                repository_id=large_repository.id,
                name=config["name"],
                description="Concurrent benchmark test",
                strategy=config["strategy"],
                target_architecture="modular_monolith",
                team_size=config["team_size"],
            )
            for config in plan_configs
        ]

        plans = await asyncio.gather(*tasks)
        duration = time.time() - start

        metrics.record(
            "concurrent_plan_creation",
            duration,
            plan_count=len(plans),
            avg_duration_per_plan=duration / len(plans),
        )

        # Verify all plans were created
        assert len(plans) == 5
        for plan in plans:
            assert plan.id is not None
            assert len(plan.steps) > 0

        print(f"\n📊 Concurrent Plan Creation Benchmark:")
        print(f"Created {len(plans)} plans in {duration:.2f}s")
        print(f"Average time per plan: {duration / len(plans):.2f}s")

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_step_batching_optimization(
        self,
        session: AsyncSession,
        large_repository: Repository,
        generate_large_codebase: list[Package],
    ):
        """Benchmark step batching optimization for large repositories."""
        metrics = BenchmarkMetrics()
        optimizer = MigrationPerformanceOptimizer(session)

        batch_sizes = [10, 20, 50]

        for batch_size in batch_sizes:
            start = time.time()
            result = await optimizer.optimize_step_batching(
                large_repository.id, batch_size=batch_size
            )
            duration = time.time() - start

            metrics.record(
                f"batching_size_{batch_size}",
                duration,
                batch_count=result["batch_count"],
                coupling_reduction=result["coupling_reduction"],
            )

        # Generate report
        report = metrics.report()

        # Performance assertions
        for batch_size in batch_sizes:
            assert (
                report["metrics"][f"batching_size_{batch_size}"]["duration_seconds"]
                < 10.0
            )

        print(f"\n📊 Step Batching Optimization Benchmark:")
        for batch_size in batch_sizes:
            metric = report["metrics"][f"batching_size_{batch_size}"]
            print(
                f"Batch size {batch_size}: {metric['duration_seconds']:.2f}s, "
                f"{metric['batch_count']} batches, "
                f"{metric['coupling_reduction']:.1f}% coupling reduction"
            )

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_pattern_extraction_performance(
        self,
        session: AsyncSession,
        complex_migration_plan: MigrationPlan,
    ):
        """Benchmark pattern extraction from large migration plans."""
        from src.services.pattern_library import PatternLibrary

        metrics = BenchmarkMetrics()
        library = PatternLibrary(session)

        # Mark some steps as completed for pattern extraction
        for i, step in enumerate(complex_migration_plan.steps[:20]):
            step.status = "completed"
            step.actual_hours = step.estimated_hours * (0.8 + (i % 5) * 0.1)
        await session.commit()

        # Benchmark pattern extraction
        start = time.time()
        patterns = await library.extract_patterns_from_execution(
            complex_migration_plan.id
        )
        duration = time.time() - start

        metrics.record(
            "pattern_extraction",
            duration,
            patterns_found=len(patterns),
            steps_analyzed=20,
        )

        # Benchmark pattern search
        start = time.time()
        search_results = await library.search_patterns(
            category="extraction",
            min_success_rate=0.5,
        )
        duration = time.time() - start

        metrics.record(
            "pattern_search",
            duration,
            results_count=len(search_results),
        )

        # Generate report
        report = metrics.report()

        # Performance assertions
        assert report["metrics"]["pattern_extraction"]["duration_seconds"] < 5.0
        assert report["metrics"]["pattern_search"]["duration_seconds"] < 1.0

        print(f"\n📊 Pattern Library Performance Benchmark:")
        print(
            f"Pattern extraction: {report['metrics']['pattern_extraction']['duration_seconds']:.2f}s"
        )
        print(
            f"Pattern search: {report['metrics']['pattern_search']['duration_seconds']:.2f}s"
        )

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_monitoring_dashboard_performance(
        self,
        session: AsyncSession,
        large_repository: Repository,
        complex_migration_plan: MigrationPlan,
    ):
        """Benchmark monitoring dashboard generation for multiple active migrations."""
        from src.services.migration_monitor import MigrationMonitor

        metrics = BenchmarkMetrics()
        monitor = MigrationMonitor(session)

        # Create multiple active plans
        planner = MigrationPlanner(session)
        active_plans = []

        for i in range(5):
            plan = await planner.create_migration_plan(
                repository_id=large_repository.id,
                name=f"Active Plan {i}",
                description="Dashboard benchmark",
                strategy=MigrationStrategy.GRADUAL,
                target_architecture="modular_monolith",
                team_size=5,
            )
            active_plans.append(plan)

        # Benchmark dashboard generation
        start = time.time()
        dashboard = await monitor.get_migration_dashboard()
        duration = time.time() - start

        metrics.record(
            "dashboard_generation",
            duration,
            active_plans=dashboard["summary"]["active_plans"],
            total_plans=dashboard["summary"]["total_plans"],
        )

        # Benchmark individual plan monitoring
        start = time.time()
        tasks = [monitor.check_migration_health(plan.id) for plan in active_plans]
        health_checks = await asyncio.gather(*tasks)
        duration = time.time() - start

        metrics.record(
            "concurrent_health_checks",
            duration,
            plans_checked=len(health_checks),
            avg_duration_per_check=duration / len(health_checks),
        )

        # Generate report
        report = metrics.report()

        # Performance assertions
        assert report["metrics"]["dashboard_generation"]["duration_seconds"] < 3.0
        assert report["metrics"]["concurrent_health_checks"]["duration_seconds"] < 5.0

        print(f"\n📊 Monitoring Dashboard Performance Benchmark:")
        print(
            f"Dashboard generation: {report['metrics']['dashboard_generation']['duration_seconds']:.2f}s"
        )
        print(
            f"Concurrent health checks: {report['metrics']['concurrent_health_checks']['duration_seconds']:.2f}s"
        )
        print(
            f"Average per health check: {report['metrics']['concurrent_health_checks']['avg_duration_per_check']:.2f}s"
        )

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_end_to_end_migration_workflow(
        self,
        session: AsyncSession,
        large_repository: Repository,
        generate_large_codebase: list[Package],
    ):
        """Benchmark complete end-to-end migration workflow."""
        metrics = BenchmarkMetrics()

        # 1. Analysis phase
        analyzer = MigrationAnalyzer(session)
        start = time.time()
        analysis = await analyzer.analyze_repository_for_migration(large_repository.id)
        duration = time.time() - start
        metrics.record("analysis_phase", duration)

        # 2. Planning phase
        planner = MigrationPlanner(session)
        start = time.time()
        plan = await planner.create_migration_plan(
            repository_id=large_repository.id,
            name="End-to-End Benchmark",
            description="Complete workflow test",
            strategy=MigrationStrategy.GRADUAL,
            target_architecture="modular_monolith",
            team_size=10,
        )
        roadmap = await planner.generate_migration_roadmap(plan.id)
        duration = time.time() - start
        metrics.record("planning_phase", duration, steps_created=len(plan.steps))

        # 3. Optimization phase
        optimizer = MigrationPerformanceOptimizer(session)
        start = time.time()
        parallel = await optimizer.analyze_parallelization_opportunities(plan.id)
        bottlenecks = await optimizer.identify_migration_bottlenecks(plan.id)
        duration = time.time() - start
        metrics.record("optimization_phase", duration)

        # 4. Resource planning phase
        start = time.time()
        resources = await planner.plan_migration_resources(
            plan.id,
            available_developers=10,
            available_architects=2,
            available_qa=3,
        )
        duration = time.time() - start
        metrics.record("resource_planning_phase", duration)

        # Generate final report
        report = metrics.report()
        total_duration = report["total_duration_seconds"]

        print(f"\n📊 End-to-End Migration Workflow Benchmark:")
        print(f"Total duration: {total_duration:.2f}s")
        print(
            f"Analysis phase: {report['metrics']['analysis_phase']['duration_seconds']:.2f}s"
        )
        print(
            f"Planning phase: {report['metrics']['planning_phase']['duration_seconds']:.2f}s"
        )
        print(
            f"Optimization phase: {report['metrics']['optimization_phase']['duration_seconds']:.2f}s"
        )
        print(
            f"Resource planning: {report['metrics']['resource_planning_phase']['duration_seconds']:.2f}s"
        )

        # Performance assertions for complete workflow
        assert total_duration < 60.0  # Complete workflow should finish within 1 minute

        # Save benchmark results
        benchmark_results = {
            "timestamp": datetime.now(UTC).isoformat(),
            "package_count": len(generate_large_codebase),
            "plan_steps": len(plan.steps),
            "total_duration": total_duration,
            "phases": report["metrics"],
        }

        return benchmark_results
