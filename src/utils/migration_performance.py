"""Performance optimization utilities for large-scale migrations."""

from collections import defaultdict
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.database.migration_models import (
    MigrationPlan,
    MigrationStep,
)
from src.database.package_models import Package
from src.logger import get_logger

logger = get_logger(__name__)


class MigrationPerformanceOptimizer:
    """Utilities for optimizing migration performance."""

    def __init__(self, session: AsyncSession):
        """Initialize the performance optimizer.

        Args:
            session: Database session
        """
        self.session = session

    async def analyze_parallelization_opportunities(
        self, plan_id: int
    ) -> dict[str, Any]:
        """Analyze opportunities for parallel execution in a migration plan.

        Args:
            plan_id: Migration plan to analyze

        Returns:
            Parallelization analysis with execution groups
        """
        logger.info("Analyzing parallelization opportunities for plan %d", plan_id)

        # Get plan with steps and dependencies
        plan = await self.session.get(
            MigrationPlan,
            plan_id,
            options=[
                selectinload(MigrationPlan.steps).selectinload(
                    MigrationStep.dependencies
                )
            ],
        )

        if not plan:
            raise ValueError(f"Migration plan {plan_id} not found")

        # Build dependency graph
        dep_graph = self._build_dependency_graph(plan.steps)

        # Find execution levels using topological sort
        execution_levels = self._calculate_execution_levels(plan.steps, dep_graph)

        # Calculate potential time savings
        sequential_time = sum(s.estimated_hours or 0 for s in plan.steps)
        parallel_time = self._calculate_parallel_execution_time(execution_levels)
        time_savings = sequential_time - parallel_time

        # Identify resource requirements per level
        resource_requirements = self._calculate_resource_requirements(execution_levels)

        return {
            "plan_id": plan_id,
            "execution_levels": execution_levels,
            "sequential_time_hours": sequential_time,
            "parallel_time_hours": parallel_time,
            "time_savings_hours": time_savings,
            "time_savings_percentage": (
                (time_savings / sequential_time * 100) if sequential_time > 0 else 0
            ),
            "max_parallel_tasks": max(
                len(level["steps"]) for level in execution_levels
            ),
            "resource_requirements": resource_requirements,
            "optimization_recommendations": self._generate_optimization_recommendations(
                execution_levels, resource_requirements
            ),
        }

    async def optimize_step_batching(
        self, repository_id: int, batch_size: int = 10
    ) -> dict[str, Any]:
        """Optimize migration step batching for large repositories.

        Args:
            repository_id: Repository to optimize
            batch_size: Target batch size

        Returns:
            Batching optimization results
        """
        logger.info("Optimizing step batching for repository %d", repository_id)

        # Get packages with dependencies
        stmt = (
            select(Package)
            .where(Package.repository_id == repository_id)
            .options(
                selectinload(Package.dependencies),
                selectinload(Package.dependents),
            )
        )

        result = await self.session.execute(stmt)
        packages = result.scalars().all()

        # Group packages by coupling strength
        package_groups = self._group_packages_by_coupling(packages)

        # Create optimal batches
        batches = self._create_optimal_batches(package_groups, batch_size)

        # Calculate batch metrics
        batch_metrics = []
        for i, batch in enumerate(batches):
            metrics = {
                "batch_number": i + 1,
                "package_count": len(batch),
                "total_lines": sum(p.total_lines or 0 for p in batch),
                "avg_coupling": self._calculate_avg_coupling(batch),
                "estimated_hours": self._estimate_batch_effort(batch),
            }
            batch_metrics.append(metrics)

        return {
            "repository_id": repository_id,
            "total_packages": len(packages),
            "batch_count": len(batches),
            "batch_size": batch_size,
            "batches": batch_metrics,
            "estimated_total_hours": sum(b["estimated_hours"] for b in batch_metrics),
            "coupling_reduction": self._calculate_coupling_reduction(packages, batches),
        }

    async def identify_migration_bottlenecks(self, plan_id: int) -> dict[str, Any]:
        """Identify potential bottlenecks in migration execution.

        Args:
            plan_id: Migration plan to analyze

        Returns:
            Bottleneck analysis
        """
        logger.info("Identifying bottlenecks for plan %d", plan_id)

        plan = await self.session.get(
            MigrationPlan,
            plan_id,
            options=[
                selectinload(MigrationPlan.steps).selectinload(
                    MigrationStep.dependencies
                )
            ],
        )

        if not plan:
            raise ValueError(f"Migration plan {plan_id} not found")

        bottlenecks = []

        # Find steps with many dependents (blocking steps)
        for step in plan.steps:
            dependent_count = self._count_dependent_steps(step, plan.steps)
            if dependent_count > 3:  # Threshold for bottleneck
                bottlenecks.append(
                    {
                        "step_id": step.id,
                        "step_name": step.name,
                        "type": "high_dependency",
                        "dependent_count": dependent_count,
                        "estimated_delay_hours": step.estimated_hours or 0,
                        "severity": "high" if dependent_count > 5 else "medium",
                        "mitigation": self._suggest_bottleneck_mitigation(
                            step, dependent_count
                        ),
                    }
                )

        # Find resource bottlenecks
        resource_analysis = await self._analyze_resource_bottlenecks(plan)
        bottlenecks.extend(resource_analysis)

        # Find complexity bottlenecks
        complexity_bottlenecks = self._identify_complexity_bottlenecks(plan.steps)
        bottlenecks.extend(complexity_bottlenecks)

        # Sort by severity
        bottlenecks.sort(
            key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(
                x.get("severity", "low"), 2
            )
        )

        return {
            "plan_id": plan_id,
            "bottleneck_count": len(bottlenecks),
            "bottlenecks": bottlenecks,
            "critical_path_impact": self._calculate_critical_path_impact(
                bottlenecks, plan
            ),
            "optimization_potential": self._calculate_optimization_potential(
                bottlenecks
            ),
        }

    async def generate_performance_recommendations(
        self, repository_id: int
    ) -> list[dict[str, Any]]:
        """Generate performance optimization recommendations.

        Args:
            repository_id: Repository to analyze

        Returns:
            List of performance recommendations
        """
        logger.info(
            "Generating performance recommendations for repository %d", repository_id
        )

        recommendations = []

        # Analyze repository size and complexity
        repo_stats = await self._get_repository_statistics(repository_id)

        # Large repository recommendations
        if repo_stats["file_count"] > 1000:
            recommendations.append(
                {
                    "category": "batching",
                    "priority": "high",
                    "recommendation": "Use incremental migration with batching",
                    "reason": f"Repository has {repo_stats['file_count']} files",
                    "expected_improvement": "30-50% reduction in migration time",
                    "implementation": [
                        "Group related packages into batches",
                        "Migrate batches in parallel where possible",
                        "Validate each batch before proceeding",
                    ],
                }
            )

        # High complexity recommendations
        if repo_stats["avg_complexity"] > 10:
            recommendations.append(
                {
                    "category": "refactoring",
                    "priority": "high",
                    "recommendation": "Pre-migration refactoring for complex code",
                    "reason": f"Average complexity is {repo_stats['avg_complexity']:.1f}",
                    "expected_improvement": "20-30% reduction in migration effort",
                    "implementation": [
                        "Identify and refactor complex functions",
                        "Extract common functionality",
                        "Improve test coverage for complex areas",
                    ],
                }
            )

        # Dependency recommendations
        if repo_stats["circular_dependency_count"] > 0:
            recommendations.append(
                {
                    "category": "architecture",
                    "priority": "critical",
                    "recommendation": "Resolve circular dependencies before migration",
                    "reason": f"Found {repo_stats['circular_dependency_count']} circular dependencies",
                    "expected_improvement": "Prevents migration blocking",
                    "implementation": [
                        "Map all circular dependencies",
                        "Introduce interfaces to break cycles",
                        "Refactor to use dependency inversion",
                    ],
                }
            )

        # Caching recommendations
        if repo_stats["file_count"] > 500:
            recommendations.append(
                {
                    "category": "infrastructure",
                    "priority": "medium",
                    "recommendation": "Implement analysis caching",
                    "reason": "Large codebase requires repeated analysis",
                    "expected_improvement": "60-80% faster subsequent analyses",
                    "implementation": [
                        "Cache AST parsing results",
                        "Cache dependency graphs",
                        "Implement incremental analysis",
                    ],
                }
            )

        # Parallel execution recommendations
        recommendations.append(
            {
                "category": "execution",
                "priority": "high",
                "recommendation": "Enable parallel step execution",
                "reason": "Most migrations have independent steps",
                "expected_improvement": "40-60% reduction in total time",
                "implementation": [
                    "Identify independent migration steps",
                    "Allocate resources for parallel execution",
                    "Implement progress tracking for parallel tasks",
                ],
            }
        )

        return recommendations

    async def calculate_resource_optimization(
        self, plan_id: int, available_resources: dict[str, int]
    ) -> dict[str, Any]:
        """Calculate optimal resource allocation for migration.

        Args:
            plan_id: Migration plan
            available_resources: Available resources (developers, architects, etc.)

        Returns:
            Resource optimization plan
        """
        logger.info("Calculating resource optimization for plan %d", plan_id)

        plan = await self.session.get(
            MigrationPlan,
            plan_id,
            options=[selectinload(MigrationPlan.steps)],
        )

        if not plan:
            raise ValueError(f"Migration plan {plan_id} not found")

        # Get parallelization analysis
        parallel_analysis = await self.analyze_parallelization_opportunities(plan_id)

        # Calculate resource allocation per execution level
        allocations = []
        for level in parallel_analysis["execution_levels"]:
            level_allocation = self._calculate_level_allocation(
                level, available_resources
            )
            allocations.append(level_allocation)

        # Calculate utilization metrics
        utilization = self._calculate_resource_utilization(
            allocations, available_resources
        )

        # Generate optimization suggestions
        optimization_suggestions = self._generate_resource_suggestions(
            allocations, utilization, available_resources
        )

        return {
            "plan_id": plan_id,
            "resource_allocations": allocations,
            "resource_utilization": utilization,
            "estimated_duration_weeks": self._calculate_optimized_duration(allocations),
            "bottleneck_resources": self._identify_resource_bottlenecks_from_allocation(
                allocations, available_resources
            ),
            "optimization_suggestions": optimization_suggestions,
        }

    def _build_dependency_graph(
        self, steps: list[MigrationStep]
    ) -> dict[int, set[int]]:
        """Build dependency graph from steps.

        Args:
            steps: Migration steps

        Returns:
            Dependency graph
        """
        graph = defaultdict(set)

        for step in steps:
            for dep in step.dependencies:
                graph[step.id].add(dep.prerequisite_step_id)

        return dict(graph)

    def _calculate_execution_levels(
        self, steps: list[MigrationStep], dep_graph: dict[int, set[int]]
    ) -> list[dict[str, Any]]:
        """Calculate execution levels using topological sort.

        Args:
            steps: Migration steps
            dep_graph: Dependency graph

        Returns:
            Execution levels
        """
        # Create step lookup
        step_lookup = {s.id: s for s in steps}

        # Calculate in-degree for each step
        in_degree = defaultdict(int)
        for step in steps:
            in_degree[step.id] = len(dep_graph.get(step.id, set()))

        # Find steps with no dependencies
        queue = [s.id for s in steps if in_degree[s.id] == 0]
        levels = []

        while queue:
            current_level = []
            next_queue = []

            for step_id in queue:
                step = step_lookup[step_id]
                current_level.append(step)

                # Find steps that depend on this one
                for other_step in steps:
                    if step_id in dep_graph.get(other_step.id, set()):
                        in_degree[other_step.id] -= 1
                        if in_degree[other_step.id] == 0:
                            next_queue.append(other_step.id)

            if current_level:
                levels.append(
                    {
                        "level": len(levels) + 1,
                        "steps": current_level,
                        "step_count": len(current_level),
                        "total_hours": sum(
                            s.estimated_hours or 0 for s in current_level
                        ),
                        "max_hours": max(
                            (s.estimated_hours or 0 for s in current_level), default=0
                        ),
                    }
                )

            queue = next_queue

        return levels

    def _calculate_parallel_execution_time(
        self, execution_levels: list[dict[str, Any]]
    ) -> float:
        """Calculate total time with parallel execution.

        Args:
            execution_levels: Execution levels

        Returns:
            Total parallel execution time
        """
        # For each level, time is the maximum of all steps in that level
        total_time = sum(level["max_hours"] for level in execution_levels)
        return total_time

    def _calculate_resource_requirements(
        self, execution_levels: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Calculate resource requirements per execution level.

        Args:
            execution_levels: Execution levels

        Returns:
            Resource requirements
        """
        requirements = {
            "max_parallel_developers": 0,
            "max_parallel_teams": 0,
            "peak_resource_level": 0,
            "resource_profile": [],
        }

        for level in execution_levels:
            step_count = level["step_count"]

            # Assume 1-2 developers per step
            developers_needed = step_count * 1.5
            teams_needed = max(1, step_count // 3)  # 3 steps per team

            requirements["max_parallel_developers"] = max(
                requirements["max_parallel_developers"], developers_needed
            )
            requirements["max_parallel_teams"] = max(
                requirements["max_parallel_teams"], teams_needed
            )

            requirements["resource_profile"].append(
                {
                    "level": level["level"],
                    "developers_needed": developers_needed,
                    "teams_needed": teams_needed,
                    "duration_hours": level["max_hours"],
                }
            )

        requirements["peak_resource_level"] = max(
            (
                r["level"]
                for r in requirements["resource_profile"]
                if r["developers_needed"] == requirements["max_parallel_developers"]
            ),
            default=1,
        )

        return requirements

    def _generate_optimization_recommendations(
        self,
        execution_levels: list[dict[str, Any]],
        resource_requirements: dict[str, Any],
    ) -> list[str]:
        """Generate optimization recommendations.

        Args:
            execution_levels: Execution levels
            resource_requirements: Resource requirements

        Returns:
            List of recommendations
        """
        recommendations = []

        # Check for unbalanced levels
        level_hours = [level["max_hours"] for level in execution_levels]
        if level_hours:
            avg_hours = sum(level_hours) / len(level_hours)
            max_hours = max(level_hours)

            if max_hours > avg_hours * 2:
                recommendations.append(
                    "Consider breaking down long-running steps to improve parallelization"
                )

        # Check for resource peaks
        if resource_requirements["max_parallel_developers"] > 10:
            recommendations.append(
                f"Peak resource requirement of {resource_requirements['max_parallel_developers']:.0f} developers - consider staggered execution"
            )

        # Check for serialization
        single_step_levels = sum(
            1 for level in execution_levels if level["step_count"] == 1
        )
        if single_step_levels > len(execution_levels) * 0.5:
            recommendations.append(
                "High serialization detected - review dependencies for optimization opportunities"
            )

        return recommendations

    def _group_packages_by_coupling(
        self, packages: list[Package]
    ) -> list[list[Package]]:
        """Group packages by coupling strength.

        Args:
            packages: List of packages

        Returns:
            Grouped packages
        """
        # Build coupling graph
        coupling_graph = defaultdict(set)

        for pkg in packages:
            for dep in pkg.dependencies:
                coupling_graph[pkg.id].add(dep.target_package_id)
            for dep in pkg.dependents:
                coupling_graph[dep.source_package_id].add(pkg.id)

        # Find strongly connected components
        visited = set()
        groups = []

        for pkg in packages:
            if pkg.id not in visited:
                group = self._find_coupled_group(pkg, packages, coupling_graph, visited)
                if group:
                    groups.append(group)

        return groups

    def _find_coupled_group(
        self,
        start_pkg: Package,
        all_packages: list[Package],
        coupling_graph: dict[int, set[int]],
        visited: set[int],
    ) -> list[Package]:
        """Find group of coupled packages using DFS.

        Args:
            start_pkg: Starting package
            all_packages: All packages
            coupling_graph: Coupling graph
            visited: Visited package IDs

        Returns:
            Group of coupled packages
        """
        group = []
        stack = [start_pkg.id]
        pkg_lookup = {p.id: p for p in all_packages}

        while stack:
            pkg_id = stack.pop()
            if pkg_id not in visited:
                visited.add(pkg_id)
                if pkg_id in pkg_lookup:
                    group.append(pkg_lookup[pkg_id])

                # Add coupled packages
                for coupled_id in coupling_graph.get(pkg_id, set()):
                    if coupled_id not in visited:
                        stack.append(coupled_id)

        return group

    def _create_optimal_batches(
        self, package_groups: list[list[Package]], batch_size: int
    ) -> list[list[Package]]:
        """Create optimal batches from package groups.

        Args:
            package_groups: Grouped packages
            batch_size: Target batch size

        Returns:
            List of batches
        """
        batches = []
        current_batch = []

        # Sort groups by size (largest first)
        sorted_groups = sorted(package_groups, key=len, reverse=True)

        for group in sorted_groups:
            if len(group) > batch_size:
                # Large group needs to be split
                for i in range(0, len(group), batch_size):
                    batches.append(group[i : i + batch_size])
            elif len(current_batch) + len(group) <= batch_size:
                # Add to current batch
                current_batch.extend(group)
            else:
                # Start new batch
                if current_batch:
                    batches.append(current_batch)
                current_batch = group[:]

        if current_batch:
            batches.append(current_batch)

        return batches

    def _calculate_avg_coupling(self, batch: list[Package]) -> float:
        """Calculate average coupling for a batch.

        Args:
            batch: Package batch

        Returns:
            Average coupling score
        """
        if not batch:
            return 0.0

        total_deps = sum(len(p.dependencies) + len(p.dependents) for p in batch)
        return total_deps / len(batch)

    def _estimate_batch_effort(self, batch: list[Package]) -> float:
        """Estimate effort for a batch.

        Args:
            batch: Package batch

        Returns:
            Estimated effort in hours
        """
        # Base effort on size
        base_effort = sum(p.total_lines or 0 for p in batch) / 50  # 50 lines per hour

        # Adjust for coupling
        coupling_factor = 1 + (self._calculate_avg_coupling(batch) / 10)

        return base_effort * coupling_factor

    def _calculate_coupling_reduction(
        self, original_packages: list[Package], batches: list[list[Package]]
    ) -> float:
        """Calculate coupling reduction from batching.

        Args:
            original_packages: Original packages
            batches: Batched packages

        Returns:
            Coupling reduction percentage
        """
        # Count cross-batch dependencies
        batch_lookup = {}
        for i, batch in enumerate(batches):
            for pkg in batch:
                batch_lookup[pkg.id] = i

        cross_batch_deps = 0
        total_deps = 0

        for pkg in original_packages:
            pkg_batch = batch_lookup.get(pkg.id)
            if pkg_batch is None:
                continue

            for dep in pkg.dependencies:
                total_deps += 1
                dep_batch = batch_lookup.get(dep.target_package_id)
                if dep_batch is not None and dep_batch != pkg_batch:
                    cross_batch_deps += 1

        if total_deps == 0:
            return 0.0

        return (1 - cross_batch_deps / total_deps) * 100

    def _count_dependent_steps(
        self, step: MigrationStep, all_steps: list[MigrationStep]
    ) -> int:
        """Count steps that depend on a given step.

        Args:
            step: Step to check
            all_steps: All steps

        Returns:
            Dependent step count
        """
        count = 0
        for other_step in all_steps:
            if any(
                dep.prerequisite_step_id == step.id for dep in other_step.dependencies
            ):
                count += 1
        return count

    def _suggest_bottleneck_mitigation(
        self, step: MigrationStep, dependent_count: int
    ) -> list[str]:
        """Suggest mitigation strategies for bottlenecks.

        Args:
            step: Bottleneck step
            dependent_count: Number of dependent steps

        Returns:
            Mitigation suggestions
        """
        suggestions = []

        if dependent_count > 5:
            suggestions.append(
                "Consider breaking this step into smaller parallel tasks"
            )
            suggestions.append("Prioritize this step for early execution")
            suggestions.append("Allocate best resources to this critical step")
        elif dependent_count > 3:
            suggestions.append("Review dependencies for optimization opportunities")
            suggestions.append("Consider pre-work to reduce step duration")

        if step.task_type == "interface_definition":
            suggestions.append("Define interfaces early to unblock dependent work")
        elif step.task_type == "data_migration":
            suggestions.append("Consider incremental data migration approach")

        return suggestions

    async def _analyze_resource_bottlenecks(
        self, plan: MigrationPlan
    ) -> list[dict[str, Any]]:
        """Analyze resource-based bottlenecks.

        Args:
            plan: Migration plan

        Returns:
            Resource bottlenecks
        """
        bottlenecks = []

        # Group steps by required skills
        skill_requirements = defaultdict(list)
        for step in plan.steps:
            required_skills = self._determine_required_skills(step)
            for skill in required_skills:
                skill_requirements[skill].append(step)

        # Identify bottlenecks
        for skill, steps in skill_requirements.items():
            if len(steps) > 5:  # Many steps need same skill
                bottlenecks.append(
                    {
                        "type": "resource_constraint",
                        "skill": skill,
                        "step_count": len(steps),
                        "total_hours": sum(s.estimated_hours or 0 for s in steps),
                        "severity": "high" if len(steps) > 10 else "medium",
                        "mitigation": [
                            f"Train additional team members in {skill}",
                            "Consider external expertise",
                            "Stagger execution of {skill}-intensive tasks",
                        ],
                    }
                )

        return bottlenecks

    def _identify_complexity_bottlenecks(
        self, steps: list[MigrationStep]
    ) -> list[dict[str, Any]]:
        """Identify complexity-based bottlenecks.

        Args:
            steps: Migration steps

        Returns:
            Complexity bottlenecks
        """
        bottlenecks = []

        for step in steps:
            if step.complexity_score and step.complexity_score > 8:
                bottlenecks.append(
                    {
                        "step_id": step.id,
                        "step_name": step.name,
                        "type": "high_complexity",
                        "complexity_score": step.complexity_score,
                        "severity": "high" if step.complexity_score > 9 else "medium",
                        "mitigation": [
                            "Break down into smaller steps",
                            "Assign senior developers",
                            "Additional testing and validation",
                            "Consider proof of concept first",
                        ],
                    }
                )

        return bottlenecks

    def _calculate_critical_path_impact(
        self, bottlenecks: list[dict[str, Any]], plan: MigrationPlan
    ) -> dict[str, Any]:
        """Calculate impact of bottlenecks on critical path.

        Args:
            bottlenecks: Identified bottlenecks
            plan: Migration plan

        Returns:
            Critical path impact
        """
        # Simplified critical path analysis
        total_delay = sum(
            b.get("estimated_delay_hours", 0)
            for b in bottlenecks
            if b.get("severity") in ["high", "critical"]
        )

        original_duration = sum(s.estimated_hours or 0 for s in plan.steps)

        return {
            "estimated_delay_hours": total_delay,
            "delay_percentage": (
                (total_delay / original_duration * 100) if original_duration > 0 else 0
            ),
            "affected_steps": len(
                {b.get("step_id") for b in bottlenecks if "step_id" in b}
            ),
        }

    def _calculate_optimization_potential(
        self, bottlenecks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Calculate potential optimization from addressing bottlenecks.

        Args:
            bottlenecks: Identified bottlenecks

        Returns:
            Optimization potential
        """
        high_severity = [b for b in bottlenecks if b.get("severity") == "high"]
        medium_severity = [b for b in bottlenecks if b.get("severity") == "medium"]

        # Estimate time savings
        potential_savings = (
            len(high_severity) * 10  # 10 hours per high severity
            + len(medium_severity) * 5  # 5 hours per medium severity
        )

        return {
            "potential_time_savings_hours": potential_savings,
            "high_priority_fixes": len(high_severity),
            "medium_priority_fixes": len(medium_severity),
            "estimated_effort_to_fix_hours": len(bottlenecks)
            * 8,  # 8 hours per bottleneck
        }

    async def _get_repository_statistics(self, repository_id: int) -> dict[str, Any]:
        """Get repository statistics for performance analysis.

        Args:
            repository_id: Repository ID

        Returns:
            Repository statistics
        """
        from src.database.models import File, Function, Module

        # File count
        file_stmt = select(func.count(File.id)).where(
            File.repository_id == repository_id
        )
        file_result = await self.session.execute(file_stmt)
        file_count = file_result.scalar() or 0

        # Average complexity
        complexity_stmt = (
            select(func.avg(Function.complexity))
            .join(Module)
            .join(File)
            .where(File.repository_id == repository_id)
        )
        complexity_result = await self.session.execute(complexity_stmt)
        avg_complexity = complexity_result.scalar() or 0

        # Circular dependencies (simplified check)
        # In practice, would do proper graph analysis
        circular_dependency_count = 0

        return {
            "file_count": file_count,
            "avg_complexity": float(avg_complexity),
            "circular_dependency_count": circular_dependency_count,
        }

    def _calculate_level_allocation(
        self, level: dict[str, Any], available_resources: dict[str, int]
    ) -> dict[str, Any]:
        """Calculate resource allocation for an execution level.

        Args:
            level: Execution level
            available_resources: Available resources

        Returns:
            Resource allocation
        """
        step_count = level["step_count"]

        # Allocate resources proportionally
        developers_needed = min(
            step_count * 2,  # 2 developers per step ideal
            available_resources.get("developers", 0),
        )

        architects_needed = min(
            max(1, step_count // 5),  # 1 architect per 5 steps
            available_resources.get("architects", 0),
        )

        qa_needed = min(
            step_count, available_resources.get("qa", 0)  # 1 QA per step ideal
        )

        return {
            "level": level["level"],
            "step_count": step_count,
            "allocated_developers": developers_needed,
            "allocated_architects": architects_needed,
            "allocated_qa": qa_needed,
            "duration_hours": level["max_hours"],
            "efficiency": self._calculate_allocation_efficiency(
                step_count, developers_needed, architects_needed, qa_needed
            ),
        }

    def _calculate_allocation_efficiency(
        self, step_count: int, developers: int, architects: int, qa: int
    ) -> float:
        """Calculate efficiency of resource allocation.

        Args:
            step_count: Number of steps
            developers: Allocated developers
            architects: Allocated architects
            qa: Allocated QA

        Returns:
            Efficiency score (0-1)
        """
        ideal_developers = step_count * 2
        ideal_architects = max(1, step_count // 5)
        ideal_qa = step_count

        dev_efficiency = (
            min(1.0, developers / ideal_developers) if ideal_developers > 0 else 1.0
        )
        arch_efficiency = (
            min(1.0, architects / ideal_architects) if ideal_architects > 0 else 1.0
        )
        qa_efficiency = min(1.0, qa / ideal_qa) if ideal_qa > 0 else 1.0

        # Weighted average
        return dev_efficiency * 0.5 + arch_efficiency * 0.2 + qa_efficiency * 0.3

    def _calculate_resource_utilization(
        self, allocations: list[dict[str, Any]], available_resources: dict[str, int]
    ) -> dict[str, Any]:
        """Calculate resource utilization metrics.

        Args:
            allocations: Resource allocations
            available_resources: Available resources

        Returns:
            Utilization metrics
        """
        total_dev_hours = sum(
            a["allocated_developers"] * a["duration_hours"] for a in allocations
        )
        total_arch_hours = sum(
            a["allocated_architects"] * a["duration_hours"] for a in allocations
        )
        total_qa_hours = sum(
            a["allocated_qa"] * a["duration_hours"] for a in allocations
        )

        # Available hours (assuming 40 hours/week)
        available_dev_hours = available_resources.get("developers", 0) * 40
        available_arch_hours = available_resources.get("architects", 0) * 40
        available_qa_hours = available_resources.get("qa", 0) * 40

        return {
            "developer_utilization": (
                (total_dev_hours / available_dev_hours * 100)
                if available_dev_hours > 0
                else 0
            ),
            "architect_utilization": (
                (total_arch_hours / available_arch_hours * 100)
                if available_arch_hours > 0
                else 0
            ),
            "qa_utilization": (
                (total_qa_hours / available_qa_hours * 100)
                if available_qa_hours > 0
                else 0
            ),
            "average_utilization": (
                (total_dev_hours + total_arch_hours + total_qa_hours)
                / (available_dev_hours + available_arch_hours + available_qa_hours)
                * 100
                if (available_dev_hours + available_arch_hours + available_qa_hours) > 0
                else 0
            ),
        }

    def _generate_resource_suggestions(
        self,
        allocations: list[dict[str, Any]],
        utilization: dict[str, Any],
        available_resources: dict[str, int],
    ) -> list[str]:
        """Generate resource optimization suggestions.

        Args:
            allocations: Resource allocations
            utilization: Utilization metrics
            available_resources: Available resources

        Returns:
            Optimization suggestions
        """
        suggestions = []

        # Check for over-utilization
        if utilization["developer_utilization"] > 90:
            suggestions.append(
                "Developer resources are over-utilized - consider hiring contractors or extending timeline"
            )

        if utilization["architect_utilization"] > 80:
            suggestions.append(
                "Architect resources are constrained - prioritize architectural decisions early"
            )

        # Check for under-utilization
        if utilization["qa_utilization"] < 50:
            suggestions.append(
                "QA resources are under-utilized - consider more comprehensive testing"
            )

        # Check for inefficient allocations
        inefficient_levels = [a for a in allocations if a["efficiency"] < 0.6]
        if inefficient_levels:
            suggestions.append(
                f"{len(inefficient_levels)} execution levels have inefficient resource allocation"
            )

        return suggestions

    def _calculate_optimized_duration(self, allocations: list[dict[str, Any]]) -> float:
        """Calculate optimized migration duration.

        Args:
            allocations: Resource allocations

        Returns:
            Duration in weeks
        """
        total_hours = sum(a["duration_hours"] for a in allocations)

        # Add overhead for coordination (20%)
        total_hours *= 1.2

        # Convert to weeks (40 hours/week)
        return total_hours / 40

    def _identify_resource_bottlenecks_from_allocation(
        self, allocations: list[dict[str, Any]], available_resources: dict[str, int]
    ) -> list[str]:
        """Identify resource bottlenecks from allocation.

        Args:
            allocations: Resource allocations
            available_resources: Available resources

        Returns:
            List of bottleneck resources
        """
        bottlenecks = []

        for allocation in allocations:
            if allocation["allocated_developers"] < allocation["step_count"] * 1.5:
                bottlenecks.append("developers")
                break

        for allocation in allocations:
            if allocation["step_count"] > 5 and allocation["allocated_architects"] < 1:
                bottlenecks.append("architects")
                break

        return list(set(bottlenecks))

    def _determine_required_skills(self, step: MigrationStep) -> list[str]:
        """Determine required skills for a step.

        Args:
            step: Migration step

        Returns:
            Required skills
        """
        skills = []

        task_type = step.task_type.lower()

        if "database" in task_type:
            skills.append("database_expertise")
        if "api" in task_type or "interface" in task_type:
            skills.append("api_design")
        if "refactor" in task_type:
            skills.append("refactoring")
        if "test" in task_type:
            skills.append("testing")
        if "security" in task_type:
            skills.append("security")
        if "performance" in task_type:
            skills.append("performance_optimization")

        if not skills:
            skills.append("general_development")

        return skills
