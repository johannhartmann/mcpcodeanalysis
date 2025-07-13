"""Migration analyzer service for identifying and analyzing code for migration."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.database.domain_models import BoundedContext, DomainEntity
from src.database.migration_models import (
    MigrationPattern,
    RiskLevel,
)
from src.database.models import Class, File, Function, Module, Repository
from src.database.package_models import Package, PackageDependency, PackageMetrics
from src.logger import get_logger

logger = get_logger(__name__)


class MigrationAnalyzer:
    """Service for analyzing code to identify migration opportunities and challenges."""

    def __init__(self, session: AsyncSession):
        """Initialize the migration analyzer.

        Args:
            session: Database session
        """
        self.session = session

    async def analyze_repository_for_migration(
        self, repository_id: int
    ) -> dict[str, Any]:
        """Analyze a repository to identify migration opportunities.

        Args:
            repository_id: Repository to analyze

        Returns:
            Analysis results including:
            - bounded_contexts: List of discovered contexts
            - migration_candidates: Modules/packages suitable for extraction
            - dependency_hotspots: Areas with high coupling
            - complexity_metrics: Overall complexity assessment
            - recommended_strategy: Suggested migration approach
        """
        logger.info("Starting migration analysis for repository %d", repository_id)

        # Get repository
        repo = await self.session.get(Repository, repository_id)
        if not repo:
            raise ValueError(f"Repository {repository_id} not found")

        # Analyze bounded contexts
        contexts = await self._analyze_bounded_contexts(repository_id)

        # Identify migration candidates
        candidates = await self._identify_migration_candidates(repository_id)

        # Analyze dependencies
        dependency_analysis = await self._analyze_dependencies(repository_id)

        # Calculate complexity metrics
        complexity = await self._calculate_complexity_metrics(repository_id)

        # Determine recommended strategy
        strategy = await self._recommend_migration_strategy(
            contexts, candidates, dependency_analysis, complexity
        )

        return {
            "repository_id": repository_id,
            "repository_name": repo.name,
            "analysis_date": datetime.now(UTC),
            "bounded_contexts": contexts,
            "migration_candidates": candidates,
            "dependency_analysis": dependency_analysis,
            "complexity_metrics": complexity,
            "recommended_strategy": strategy,
        }

    async def _analyze_bounded_contexts(
        self, repository_id: int
    ) -> list[dict[str, Any]]:
        """Analyze and score bounded contexts for migration.

        Args:
            repository_id: Repository to analyze

        Returns:
            List of bounded contexts with migration scores
        """
        # Query bounded contexts with their entities
        stmt = (
            select(BoundedContext)
            .join(BoundedContext.memberships)
            .join(DomainEntity)
            .where(
                DomainEntity.source_entities.any(File.repository_id == repository_id)
            )
            .options(selectinload(BoundedContext.memberships))
            .distinct()
        )

        result = await self.session.execute(stmt)
        contexts = result.scalars().all()

        context_analysis = []
        for context in contexts:
            # Calculate migration readiness score
            readiness = await self._calculate_context_readiness(context)

            # Estimate extraction complexity
            complexity = await self._estimate_extraction_complexity(context)

            # Identify dependencies
            dependencies = await self._analyze_context_dependencies(context)

            context_analysis.append(
                {
                    "id": context.id,
                    "name": context.name,
                    "type": context.context_type,
                    "entity_count": len(context.memberships),
                    "cohesion_score": context.cohesion_score or 0,
                    "coupling_score": context.coupling_score or 0,
                    "migration_readiness": readiness,
                    "extraction_complexity": complexity,
                    "dependencies": dependencies,
                    "priority": context.migration_priority or "medium",
                }
            )

        # Sort by readiness score (highest first)
        context_analysis.sort(key=lambda x: x["migration_readiness"], reverse=True)

        return context_analysis

    async def _identify_migration_candidates(
        self, repository_id: int
    ) -> list[dict[str, Any]]:
        """Identify packages and modules suitable for migration.

        Args:
            repository_id: Repository to analyze

        Returns:
            List of migration candidates with scores
        """
        # Query packages with metrics
        stmt = (
            select(Package, PackageMetrics)
            .join(PackageMetrics, Package.id == PackageMetrics.package_id)
            .where(Package.repository_id == repository_id)
            .options(
                selectinload(Package.dependencies),
                selectinload(Package.dependents),
            )
        )

        result = await self.session.execute(stmt)
        packages = result.all()

        candidates = []
        for package, metrics in packages:
            # Skip packages with too many dependencies
            if len(package.dependents) > 10:
                continue

            # Calculate migration score
            score = await self._calculate_migration_score(package, metrics)

            # Estimate effort
            effort = await self._estimate_migration_effort(package, metrics)

            candidates.append(
                {
                    "id": package.id,
                    "path": package.path,
                    "name": package.name,
                    "type": "package",
                    "size_metrics": {
                        "modules": package.module_count,
                        "lines": package.total_lines,
                        "classes": package.total_classes,
                        "functions": package.total_functions,
                    },
                    "quality_metrics": {
                        "cohesion": metrics.cohesion_score,
                        "coupling": metrics.coupling_score,
                        "has_tests": metrics.has_tests,
                        "has_docs": metrics.has_docs,
                    },
                    "migration_score": score,
                    "estimated_effort_hours": effort,
                    "dependencies": len(package.dependencies),
                    "dependents": len(package.dependents),
                }
            )

        # Sort by migration score (highest first)
        candidates.sort(key=lambda x: x["migration_score"], reverse=True)

        return candidates[:20]  # Return top 20 candidates

    async def _analyze_dependencies(self, repository_id: int) -> dict[str, Any]:
        """Analyze dependency patterns in the repository.

        Args:
            repository_id: Repository to analyze

        Returns:
            Dependency analysis results
        """
        # Find circular dependencies
        circular = await self._find_circular_dependencies(repository_id)

        # Identify high-coupling areas
        high_coupling = await self._identify_high_coupling_packages(repository_id)

        # Find dependency bottlenecks
        bottlenecks = await self._find_dependency_bottlenecks(repository_id)

        return {
            "circular_dependencies": circular,
            "high_coupling_packages": high_coupling,
            "dependency_bottlenecks": bottlenecks,
            "total_packages": await self._count_packages(repository_id),
            "avg_dependencies_per_package": await self._avg_dependencies_per_package(
                repository_id
            ),
        }

    async def _calculate_complexity_metrics(self, repository_id: int) -> dict[str, Any]:
        """Calculate overall complexity metrics for the repository.

        Args:
            repository_id: Repository to analyze

        Returns:
            Complexity metrics
        """
        # Total LOC
        loc_stmt = select(func.sum(File.size)).where(
            File.repository_id == repository_id
        )
        loc_result = await self.session.execute(loc_stmt)
        total_loc = loc_result.scalar() or 0

        # File count
        file_stmt = select(func.count(File.id)).where(
            File.repository_id == repository_id
        )
        file_result = await self.session.execute(file_stmt)
        file_count = file_result.scalar() or 0

        # Function complexity
        complexity_stmt = (
            select(
                func.avg(Function.complexity),
                func.max(Function.complexity),
                func.count(Function.id),
            )
            .join(Module)
            .join(File)
            .where(File.repository_id == repository_id)
        )

        complexity_result = await self.session.execute(complexity_stmt)
        avg_complexity, max_complexity, function_count = complexity_result.one()

        # Class count
        class_stmt = (
            select(func.count(Class.id))
            .join(Module)
            .join(File)
            .where(File.repository_id == repository_id)
        )
        class_result = await self.session.execute(class_stmt)
        class_count = class_result.scalar() or 0

        return {
            "total_lines_of_code": total_loc,
            "file_count": file_count,
            "class_count": class_count,
            "function_count": function_count,
            "avg_cyclomatic_complexity": float(avg_complexity or 0),
            "max_cyclomatic_complexity": max_complexity or 0,
            "complexity_rating": self._rate_complexity(
                avg_complexity or 0, max_complexity or 0
            ),
        }

    async def _recommend_migration_strategy(
        self,
        contexts: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
        dependencies: dict[str, Any],
        complexity: dict[str, Any],
    ) -> dict[str, Any]:
        """Recommend a migration strategy based on analysis.

        Args:
            contexts: Bounded context analysis
            candidates: Migration candidates
            dependencies: Dependency analysis
            complexity: Complexity metrics

        Returns:
            Recommended migration strategy
        """
        # Determine strategy based on codebase characteristics
        strategy_type = "gradual"  # Default

        if complexity["complexity_rating"] == "high":
            strategy_type = "strangler_fig"
        elif len(contexts) >= 5 and all(
            c["migration_readiness"] > 0.7 for c in contexts[:3]
        ):
            strategy_type = "branch_by_abstraction"
        elif dependencies["circular_dependencies"]:
            strategy_type = "gradual"

        # Identify starting points
        starting_points = []
        if contexts:
            # Start with highest readiness context
            starting_points.append(
                {
                    "type": "bounded_context",
                    "id": contexts[0]["id"],
                    "name": contexts[0]["name"],
                    "reason": "Highest migration readiness score",
                }
            )

        if candidates:
            # Add top candidate
            starting_points.append(
                {
                    "type": "package",
                    "id": candidates[0]["id"],
                    "name": candidates[0]["name"],
                    "reason": "Best migration score with low coupling",
                }
            )

        # Estimate timeline
        total_effort = sum(c["estimated_effort_hours"] for c in candidates[:10])
        timeline_weeks = (total_effort / 40) * 1.5  # Add 50% buffer

        # Identify risks
        risks = []
        if dependencies["circular_dependencies"]:
            risks.append(
                {
                    "type": "technical",
                    "level": "high",
                    "description": "Circular dependencies need resolution",
                    "mitigation": "Refactor circular dependencies before extraction",
                }
            )

        if complexity["max_cyclomatic_complexity"] > 20:
            risks.append(
                {
                    "type": "technical",
                    "level": "medium",
                    "description": "High complexity functions need simplification",
                    "mitigation": "Refactor complex functions during migration",
                }
            )

        return {
            "recommended_strategy": strategy_type,
            "starting_points": starting_points,
            "estimated_timeline_weeks": round(timeline_weeks),
            "phases": self._generate_migration_phases(
                strategy_type, contexts, candidates
            ),
            "risks": risks,
            "success_factors": [
                "Strong test coverage for migrated components",
                "Clear interface definitions between modules",
                "Incremental validation at each phase",
                "Team knowledge sharing and documentation",
            ],
        }

    async def _calculate_context_readiness(self, context: BoundedContext) -> float:
        """Calculate migration readiness score for a bounded context.

        Args:
            context: Bounded context to evaluate

        Returns:
            Readiness score (0.0 to 1.0)
        """
        score = 0.0

        # Cohesion contributes positively
        if context.cohesion_score:
            score += context.cohesion_score * 0.3

        # Low coupling contributes positively
        if context.coupling_score:
            score += (1.0 - context.coupling_score) * 0.3

        # Modularity score
        if context.modularity_score:
            score += context.modularity_score * 0.2

        # Core contexts get a bonus
        if context.context_type == "core":
            score += 0.1

        # Generic contexts are easier to extract
        if context.context_type == "generic":
            score += 0.1

        return min(score, 1.0)

    async def _estimate_extraction_complexity(self, context: BoundedContext) -> float:
        """Estimate the complexity of extracting a bounded context.

        Args:
            context: Bounded context to evaluate

        Returns:
            Complexity score (0.0 to 1.0, lower is easier)
        """
        complexity = 0.0

        # High coupling increases complexity
        if context.coupling_score:
            complexity += context.coupling_score * 0.4

        # Large contexts are more complex
        entity_count = len(context.memberships)
        if entity_count > 50:
            complexity += 0.3
        elif entity_count > 20:
            complexity += 0.2
        elif entity_count > 10:
            complexity += 0.1

        # Supporting contexts are easier
        if context.context_type == "supporting":
            complexity -= 0.1

        # External contexts are hardest
        if context.context_type == "external":
            complexity += 0.2

        return max(0.0, min(complexity, 1.0))

    async def _analyze_context_dependencies(
        self, context: BoundedContext
    ) -> dict[str, Any]:
        """Analyze dependencies for a bounded context.

        Args:
            context: Bounded context to analyze

        Returns:
            Dependency information
        """
        # This would analyze actual code dependencies
        # For now, return a simplified version
        return {
            "inbound_dependencies": 0,  # Other contexts depending on this
            "outbound_dependencies": 0,  # This context's dependencies
            "shared_kernel_candidates": [],  # Potential shared kernels
            "interface_complexity": "medium",  # Estimated interface complexity
        }

    async def _calculate_migration_score(
        self, package: Package, metrics: PackageMetrics
    ) -> float:
        """Calculate migration suitability score for a package.

        Args:
            package: Package to evaluate
            metrics: Package metrics

        Returns:
            Migration score (0.0 to 1.0)
        """
        score = 0.0

        # Cohesion is good
        if metrics.cohesion_score:
            score += (metrics.cohesion_score / 100) * 0.3

        # Low coupling is good
        if metrics.coupling_score:
            score += ((100 - metrics.coupling_score) / 100) * 0.3

        # Having tests is good
        if metrics.has_tests:
            score += 0.2

        # Having docs is good
        if metrics.has_docs:
            score += 0.1

        # Small to medium size is good
        if 5 <= package.module_count <= 20:
            score += 0.1

        return min(score, 1.0)

    async def _estimate_migration_effort(
        self, package: Package, metrics: PackageMetrics
    ) -> float:
        """Estimate migration effort in hours for a package.

        Args:
            package: Package to evaluate
            metrics: Package metrics

        Returns:
            Estimated effort in hours
        """
        # Base effort based on size
        base_effort = package.total_lines / 50  # 50 lines per hour

        # Adjust for complexity
        if metrics.avg_complexity:
            if metrics.avg_complexity > 10:
                base_effort *= 1.5
            elif metrics.avg_complexity > 5:
                base_effort *= 1.2

        # Adjust for dependencies
        dependency_count = len(package.dependencies) + len(package.dependents)
        if dependency_count > 10:
            base_effort *= 1.5
        elif dependency_count > 5:
            base_effort *= 1.3

        # Adjust for test coverage
        if not metrics.has_tests:
            base_effort *= 1.5  # Need to write tests

        return round(base_effort)

    async def _find_circular_dependencies(
        self, repository_id: int
    ) -> list[dict[str, Any]]:
        """Find circular dependencies in the repository.

        Args:
            repository_id: Repository to analyze

        Returns:
            List of circular dependency cycles
        """
        # This would implement graph cycle detection
        # For now, return empty list
        return []

    async def _identify_high_coupling_packages(
        self, repository_id: int
    ) -> list[dict[str, Any]]:
        """Identify packages with high coupling.

        Args:
            repository_id: Repository to analyze

        Returns:
            List of high-coupling packages
        """
        stmt = (
            select(Package, PackageMetrics)
            .join(PackageMetrics)
            .where(
                and_(
                    Package.repository_id == repository_id,
                    PackageMetrics.coupling_score > 70,
                )
            )
            .order_by(PackageMetrics.coupling_score.desc())
            .limit(10)
        )

        result = await self.session.execute(stmt)
        packages = result.all()

        return [
            {
                "package_id": pkg.id,
                "package_path": pkg.path,
                "coupling_score": metrics.coupling_score,
                "dependency_count": len(pkg.dependencies) + len(pkg.dependents),
            }
            for pkg, metrics in packages
        ]

    async def _find_dependency_bottlenecks(
        self, repository_id: int
    ) -> list[dict[str, Any]]:
        """Find packages that many others depend on.

        Args:
            repository_id: Repository to analyze

        Returns:
            List of bottleneck packages
        """
        # Query packages with many dependents
        stmt = (
            select(Package, func.count(PackageDependency.id).label("dependent_count"))
            .join(
                PackageDependency,
                Package.id == PackageDependency.target_package_id,
            )
            .where(Package.repository_id == repository_id)
            .group_by(Package.id)
            .having(func.count(PackageDependency.id) > 5)
            .order_by(func.count(PackageDependency.id).desc())
            .limit(10)
        )

        result = await self.session.execute(stmt)
        bottlenecks = result.all()

        return [
            {
                "package_id": pkg.id,
                "package_path": pkg.path,
                "dependent_count": count,
                "risk_level": "high" if count > 10 else "medium",
            }
            for pkg, count in bottlenecks
        ]

    async def _count_packages(self, repository_id: int) -> int:
        """Count total packages in repository.

        Args:
            repository_id: Repository to count

        Returns:
            Package count
        """
        stmt = select(func.count(Package.id)).where(
            Package.repository_id == repository_id
        )
        result = await self.session.execute(stmt)
        return result.scalar() or 0

    async def _avg_dependencies_per_package(self, repository_id: int) -> float:
        """Calculate average dependencies per package.

        Args:
            repository_id: Repository to analyze

        Returns:
            Average dependency count
        """
        stmt = (
            select(func.avg(func.count(PackageDependency.id)))
            .join(Package, Package.id == PackageDependency.source_package_id)
            .where(Package.repository_id == repository_id)
            .group_by(Package.id)
        )
        result = await self.session.execute(stmt)
        avg = result.scalar()
        return float(avg) if avg else 0.0

    def _rate_complexity(self, avg_complexity: float, max_complexity: float) -> str:
        """Rate overall complexity level.

        Args:
            avg_complexity: Average cyclomatic complexity
            max_complexity: Maximum cyclomatic complexity

        Returns:
            Complexity rating
        """
        if max_complexity > 20 or avg_complexity > 10:
            return "high"
        elif max_complexity > 10 or avg_complexity > 5:
            return "medium"
        else:
            return "low"

    def _generate_migration_phases(
        self,
        strategy_type: str,
        contexts: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Generate migration phases based on strategy.

        Args:
            strategy_type: Type of migration strategy
            contexts: Bounded contexts
            candidates: Migration candidates

        Returns:
            List of migration phases
        """
        phases = []

        if strategy_type == "strangler_fig":
            phases = [
                {
                    "phase": 1,
                    "name": "Setup and Infrastructure",
                    "description": "Establish migration infrastructure and patterns",
                    "duration_weeks": 2,
                },
                {
                    "phase": 2,
                    "name": "Interface Definition",
                    "description": "Define clear interfaces for target modules",
                    "duration_weeks": 3,
                },
                {
                    "phase": 3,
                    "name": "Incremental Extraction",
                    "description": "Extract modules one by one with validation",
                    "duration_weeks": 8,
                },
                {
                    "phase": 4,
                    "name": "Legacy Cleanup",
                    "description": "Remove old code and optimize",
                    "duration_weeks": 2,
                },
            ]
        elif strategy_type == "branch_by_abstraction":
            phases = [
                {
                    "phase": 1,
                    "name": "Abstraction Layer",
                    "description": "Create abstraction layer for target components",
                    "duration_weeks": 3,
                },
                {
                    "phase": 2,
                    "name": "Parallel Implementation",
                    "description": "Build new implementation behind abstraction",
                    "duration_weeks": 6,
                },
                {
                    "phase": 3,
                    "name": "Migration and Testing",
                    "description": "Migrate traffic and validate",
                    "duration_weeks": 4,
                },
                {
                    "phase": 4,
                    "name": "Cleanup",
                    "description": "Remove old implementation",
                    "duration_weeks": 1,
                },
            ]
        else:  # gradual
            phases = [
                {
                    "phase": 1,
                    "name": "Preparation",
                    "description": "Refactor and prepare target modules",
                    "duration_weeks": 4,
                },
                {
                    "phase": 2,
                    "name": "Module Extraction",
                    "description": "Extract modules in dependency order",
                    "duration_weeks": 10,
                },
                {
                    "phase": 3,
                    "name": "Integration",
                    "description": "Integrate and optimize extracted modules",
                    "duration_weeks": 3,
                },
            ]

        return phases

    async def identify_migration_patterns(
        self, repository_id: int
    ) -> list[dict[str, Any]]:
        """Identify applicable migration patterns for a repository.

        Args:
            repository_id: Repository to analyze

        Returns:
            List of applicable migration patterns
        """
        # Get all migration patterns
        stmt = select(MigrationPattern).where(
            MigrationPattern.success_rate > 0.7
        )  # Only successful patterns

        result = await self.session.execute(stmt)
        patterns = result.scalars().all()

        # Analyze repository characteristics
        analysis = await self.analyze_repository_for_migration(repository_id)

        applicable_patterns = []
        for pattern in patterns:
            # Check if pattern is applicable based on context
            applicability = self._check_pattern_applicability(pattern, analysis)

            if applicability > 0.5:
                applicable_patterns.append(
                    {
                        "pattern_id": pattern.id,
                        "name": pattern.name,
                        "category": pattern.category,
                        "description": pattern.description,
                        "applicability_score": applicability,
                        "success_rate": pattern.success_rate,
                        "avg_effort_hours": pattern.avg_effort_hours,
                    }
                )

        # Sort by applicability
        applicable_patterns.sort(key=lambda x: x["applicability_score"], reverse=True)

        return applicable_patterns

    def _check_pattern_applicability(
        self, pattern: MigrationPattern, analysis: dict[str, Any]
    ) -> float:
        """Check if a pattern is applicable to the analyzed codebase.

        Args:
            pattern: Migration pattern to check
            analysis: Repository analysis results

        Returns:
            Applicability score (0.0 to 1.0)
        """
        score = 0.5  # Base score

        # Check pattern context against analysis
        if pattern.applicable_scenarios:
            for scenario in pattern.applicable_scenarios:
                if (
                    scenario.get("complexity")
                    == analysis["complexity_metrics"]["complexity_rating"]
                ):
                    score += 0.2

                if (
                    scenario.get("strategy")
                    == analysis["recommended_strategy"]["recommended_strategy"]
                ):
                    score += 0.3

        return min(score, 1.0)

    async def assess_migration_risks(
        self, repository_id: int, plan_id: int | None = None
    ) -> list[dict[str, Any]]:
        """Assess migration risks for a repository or specific plan.

        Args:
            repository_id: Repository to assess
            plan_id: Optional specific migration plan

        Returns:
            List of identified risks with mitigation strategies
        """
        risks = []

        # Analyze repository if no plan specified
        if not plan_id:
            analysis = await self.analyze_repository_for_migration(repository_id)

            # Technical risks
            if analysis["complexity_metrics"]["max_cyclomatic_complexity"] > 20:
                risks.append(
                    {
                        "type": "technical",
                        "name": "High Code Complexity",
                        "description": "Complex functions may be difficult to migrate cleanly",
                        "probability": 0.7,
                        "impact": 0.6,
                        "level": self._calculate_risk_level(0.7, 0.6),
                        "mitigation": "Refactor complex functions before migration",
                    }
                )

            if analysis["dependency_analysis"]["circular_dependencies"]:
                risks.append(
                    {
                        "type": "technical",
                        "name": "Circular Dependencies",
                        "description": "Circular dependencies complicate module extraction",
                        "probability": 0.9,
                        "impact": 0.8,
                        "level": self._calculate_risk_level(0.9, 0.8),
                        "mitigation": "Break circular dependencies through interface abstraction",
                    }
                )

            # Operational risks
            if analysis["complexity_metrics"]["total_lines_of_code"] > 100000:
                risks.append(
                    {
                        "type": "operational",
                        "name": "Large Codebase",
                        "description": "Large codebase requires significant coordination",
                        "probability": 0.6,
                        "impact": 0.5,
                        "level": self._calculate_risk_level(0.6, 0.5),
                        "mitigation": "Phase migration and establish clear communication channels",
                    }
                )

            # Business risks
            if not any(
                c["quality_metrics"]["has_tests"]
                for c in analysis["migration_candidates"][:5]
            ):
                risks.append(
                    {
                        "type": "business",
                        "name": "Insufficient Test Coverage",
                        "description": "Lack of tests increases risk of regression",
                        "probability": 0.8,
                        "impact": 0.9,
                        "level": self._calculate_risk_level(0.8, 0.9),
                        "mitigation": "Implement comprehensive test suite before migration",
                    }
                )

        return risks

    def _calculate_risk_level(self, probability: float, impact: float) -> str:
        """Calculate risk level from probability and impact.

        Args:
            probability: Risk probability (0.0 to 1.0)
            impact: Risk impact (0.0 to 1.0)

        Returns:
            Risk level
        """
        risk_score = probability * impact

        if risk_score >= 0.7:
            return RiskLevel.CRITICAL.value
        elif risk_score >= 0.5:
            return RiskLevel.HIGH.value
        elif risk_score >= 0.3:
            return RiskLevel.MEDIUM.value
        else:
            return RiskLevel.LOW.value
