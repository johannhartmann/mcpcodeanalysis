"""MCP tools for migration intelligence and planning."""

from typing import Any

from fastmcp import Tool
from pydantic import BaseModel, Field

from src.database.migration_models import MigrationStrategy
from src.mcp_server.tools.utils import (
    ToolResult,
    create_error_result,
    create_success_result,
    get_repository_id,
    get_session_factory,
)
from src.services.interface_designer import InterfaceDesigner
from src.services.migration_analyzer import MigrationAnalyzer
from src.services.migration_planner import MigrationPlanner


# Tool input models
class AnalyzeMigrationInput(BaseModel):
    """Input for analyze_migration_readiness tool."""

    repository_url: str = Field(
        description="Repository URL to analyze for migration readiness"
    )


class CreateMigrationPlanInput(BaseModel):
    """Input for create_migration_plan tool."""

    repository_url: str = Field(description="Repository URL to create plan for")
    plan_name: str = Field(description="Name for the migration plan")
    strategy: str = Field(
        description="Migration strategy (strangler_fig, gradual, big_bang, branch_by_abstraction, parallel_run)",
        default="gradual",
    )
    target_architecture: str = Field(
        description="Target architecture (modular_monolith, microservices, event_driven)",
        default="modular_monolith",
    )
    team_size: int = Field(description="Available team size", default=5)
    timeline_weeks: int | None = Field(
        description="Desired timeline in weeks (optional)", default=None
    )
    risk_tolerance: str = Field(
        description="Risk tolerance level (low, medium, high)", default="medium"
    )


class OptimizeMigrationPlanInput(BaseModel):
    """Input for optimize_migration_plan tool."""

    plan_id: int = Field(description="ID of the migration plan to optimize")
    minimize_time: bool = Field(
        description="Optimize to minimize timeline", default=False
    )
    minimize_risk: bool = Field(description="Optimize to minimize risk", default=False)
    maximize_quality: bool = Field(
        description="Optimize to maximize quality", default=False
    )


class GenerateMigrationRoadmapInput(BaseModel):
    """Input for generate_migration_roadmap tool."""

    plan_id: int = Field(description="ID of the migration plan")


class IdentifyMigrationPatternsInput(BaseModel):
    """Input for identify_migration_patterns tool."""

    repository_url: str = Field(description="Repository URL to analyze")


class AssessMigrationRisksInput(BaseModel):
    """Input for assess_migration_risks tool."""

    repository_url: str = Field(description="Repository URL to assess risks for")
    plan_id: int | None = Field(
        description="Optional specific migration plan ID", default=None
    )


class AnalyzeMigrationImpactInput(BaseModel):
    """Input for analyze_migration_impact tool."""

    repository_url: str = Field(description="Repository URL to analyze")
    module_path: str = Field(description="Module or package path to analyze impact for")
    impact_type: str = Field(
        description="Type of impact analysis (extraction, refactoring, interface_change)",
        default="extraction",
    )


class EstimateMigrationEffortInput(BaseModel):
    """Input for estimate_migration_effort tool."""

    plan_id: int = Field(description="Migration plan ID to estimate effort for")
    include_testing: bool = Field(
        description="Include testing effort in estimates", default=True
    )
    include_documentation: bool = Field(
        description="Include documentation effort in estimates", default=True
    )
    productivity_factor: float = Field(
        description="Team productivity factor (0.5 to 1.5)", default=1.0
    )


class PlanMigrationResourcesInput(BaseModel):
    """Input for plan_migration_resources tool."""

    plan_id: int = Field(description="Migration plan ID to plan resources for")
    available_developers: int = Field(
        description="Number of available developers", default=5
    )
    available_architects: int = Field(
        description="Number of available architects", default=1
    )
    available_qa: int = Field(description="Number of available QA engineers", default=2)
    sprint_weeks: int = Field(description="Sprint duration in weeks", default=2)


class DesignModuleInterfaceInput(BaseModel):
    """Input for design_module_interface tool."""

    repository_url: str = Field(description="Repository URL")
    package_path: str = Field(description="Package path to design interface for")
    target_architecture: str = Field(
        description="Target architecture (modular_monolith, microservices, event_driven)",
        default="modular_monolith",
    )
    include_events: bool = Field(
        description="Include domain events in design", default=False
    )


class GenerateInterfaceDocumentationInput(BaseModel):
    """Input for generate_interface_documentation tool."""

    interface_design: dict[str, Any] = Field(
        description="Interface design from design_module_interface tool"
    )


# Migration analysis and planning tools
analyze_migration_readiness = Tool(
    name="analyze_migration_readiness",
    description="""Analyze a repository to assess migration readiness and identify opportunities.

This tool performs comprehensive analysis including:
- Bounded context discovery and scoring
- Module extraction candidates
- Dependency analysis and hotspots
- Complexity assessment
- Strategy recommendations

Use this before creating a migration plan to understand the codebase structure.""",
    parameters=AnalyzeMigrationInput,
)


create_migration_plan = Tool(
    name="create_migration_plan",
    description="""Create a detailed migration plan for transforming a monolithic codebase.

The plan includes:
- Step-by-step migration tasks with dependencies
- Risk assessment and mitigation strategies
- Resource requirements and timeline
- Success metrics and validation criteria

Available strategies:
- strangler_fig: Gradually replace functionality behind a facade
- gradual: Extract modules one by one
- big_bang: Complete rewrite (not recommended)
- branch_by_abstraction: Create abstraction layer first
- parallel_run: Run old and new systems in parallel""",
    parameters=CreateMigrationPlanInput,
)


optimize_migration_plan = Tool(
    name="optimize_migration_plan",
    description="""Optimize an existing migration plan based on specified goals.

Optimization options:
- minimize_time: Reduce timeline through parallelization
- minimize_risk: Add validation steps and reduce complexity
- maximize_quality: Increase test coverage and documentation requirements

Multiple goals can be selected simultaneously.""",
    parameters=OptimizeMigrationPlanInput,
)


generate_migration_roadmap = Tool(
    name="generate_migration_roadmap",
    description="""Generate a visual roadmap for a migration plan.

The roadmap includes:
- Phase breakdown with timelines
- Critical path identification
- Resource allocation by phase
- Milestone tracking
- Risk mitigation schedule

Use this to communicate the migration plan to stakeholders.""",
    parameters=GenerateMigrationRoadmapInput,
)


identify_migration_patterns = Tool(
    name="identify_migration_patterns",
    description="""Identify proven migration patterns applicable to a codebase.

Analyzes the repository characteristics and suggests:
- Applicable migration patterns from the pattern library
- Success rates and typical effort
- Pattern-specific implementation guidance

Use this to leverage best practices from successful migrations.""",
    parameters=IdentifyMigrationPatternsInput,
)


# Risk assessment and impact analysis tools
assess_migration_risks = Tool(
    name="assess_migration_risks",
    description="""Assess risks associated with migrating a codebase.

Risk assessment includes:
- Technical risks (complexity, dependencies, tech debt)
- Operational risks (team skills, timeline, coordination)
- Business risks (downtime, performance, functionality)

Each risk includes:
- Probability and impact scores
- Risk level (low, medium, high, critical)
- Mitigation strategies

Use this to proactively identify and plan for migration challenges.""",
    parameters=AssessMigrationRisksInput,
)


analyze_migration_impact = Tool(
    name="analyze_migration_impact",
    description="""Analyze the impact of migrating a specific module or package.

Impact analysis includes:
- Direct dependencies (what depends on this module)
- Transitive dependencies (indirect impacts)
- Interface changes required
- Data migration needs
- Performance implications
- Testing requirements

Impact types:
- extraction: Impact of extracting module
- refactoring: Impact of refactoring module
- interface_change: Impact of changing module interface

Use this to understand ripple effects before making changes.""",
    parameters=AnalyzeMigrationImpactInput,
)


# Effort estimation and resource planning tools
estimate_migration_effort = Tool(
    name="estimate_migration_effort",
    description="""Estimate detailed effort required for a migration plan.

Effort estimation includes:
- Development hours by phase and step
- Testing effort (unit, integration, performance)
- Documentation requirements
- Coordination and planning overhead
- Buffer for unknowns and risks

Factors considered:
- Code complexity and size
- Dependency complexity
- Team experience and productivity
- Testing requirements
- Risk factors

Use this for accurate project planning and budgeting.""",
    parameters=EstimateMigrationEffortInput,
)


plan_migration_resources = Tool(
    name="plan_migration_resources",
    description="""Plan resource allocation for a migration project.

Resource planning includes:
- Team composition by sprint/phase
- Role assignments (developers, architects, QA)
- Skill requirements mapping
- Capacity planning and utilization
- Timeline with resource constraints

Generates:
- Sprint-by-sprint resource allocation
- Role responsibility matrix
- Skills gap analysis
- Resource conflict identification

Use this to ensure adequate staffing and skills for migration.""",
    parameters=PlanMigrationResourcesInput,
)


# Interface and contract design tools
design_module_interface = Tool(
    name="design_module_interface",
    description="""Design clean interfaces and contracts for a module.

Interface design includes:
- Public API specification
- Data transfer objects (DTOs)
- Domain events (if applicable)
- Dependency interfaces
- Implementation guidelines

Design principles:
- Clear separation of concerns
- Minimal coupling
- Explicit contracts
- Versioning strategy

Use this to create well-defined module boundaries.""",
    parameters=DesignModuleInterfaceInput,
)


generate_interface_documentation = Tool(
    name="generate_interface_documentation",
    description="""Generate comprehensive documentation for a module interface.

Documentation includes:
- API reference with examples
- Data contract specifications
- Integration guidelines
- Migration instructions
- Testing strategies

Output format: Markdown documentation ready for publishing.

Use this to document interfaces for development teams.""",
    parameters=GenerateInterfaceDocumentationInput,
)


# Tool implementations
@analyze_migration_readiness.on_call
async def analyze_migration_readiness_impl(input: AnalyzeMigrationInput) -> ToolResult:
    """Analyze repository for migration readiness."""
    session_factory = await get_session_factory()

    async with session_factory() as session:
        try:
            # Get repository ID
            repo_id = await get_repository_id(session, input.repository_url)
            if not repo_id:
                return create_error_result(
                    f"Repository {input.repository_url} not found. Please sync it first."
                )

            # Create analyzer
            analyzer = MigrationAnalyzer(session)

            # Perform analysis
            analysis = await analyzer.analyze_repository_for_migration(repo_id)

            # Format results
            result = {
                "repository": analysis["repository_name"],
                "analysis_summary": {
                    "bounded_contexts_found": len(analysis["bounded_contexts"]),
                    "migration_candidates": len(analysis["migration_candidates"]),
                    "complexity_rating": analysis["complexity_metrics"][
                        "complexity_rating"
                    ],
                    "recommended_strategy": analysis["recommended_strategy"][
                        "recommended_strategy"
                    ],
                },
                "top_contexts": analysis["bounded_contexts"][:5],  # Top 5 by readiness
                "top_candidates": analysis["migration_candidates"][
                    :10
                ],  # Top 10 candidates
                "dependency_issues": {
                    "circular_dependencies": len(
                        analysis["dependency_analysis"]["circular_dependencies"]
                    ),
                    "high_coupling_packages": len(
                        analysis["dependency_analysis"]["high_coupling_packages"]
                    ),
                    "bottlenecks": len(
                        analysis["dependency_analysis"]["dependency_bottlenecks"]
                    ),
                },
                "recommended_approach": analysis["recommended_strategy"],
                "complexity_metrics": analysis["complexity_metrics"],
            }

            return create_success_result(
                result, "Migration readiness analysis completed successfully"
            )

        except Exception as e:
            return create_error_result(f"Failed to analyze repository: {str(e)}")


@create_migration_plan.on_call
async def create_migration_plan_impl(input: CreateMigrationPlanInput) -> ToolResult:
    """Create a migration plan."""
    session_factory = await get_session_factory()

    async with session_factory() as session:
        try:
            # Get repository ID
            repo_id = await get_repository_id(session, input.repository_url)
            if not repo_id:
                return create_error_result(
                    f"Repository {input.repository_url} not found. Please sync it first."
                )

            # Validate strategy
            try:
                strategy = MigrationStrategy(input.strategy)
            except ValueError:
                return create_error_result(
                    f"Invalid strategy '{input.strategy}'. "
                    f"Valid options: {', '.join(s.value for s in MigrationStrategy)}"
                )

            # Create planner
            planner = MigrationPlanner(session)

            # Create plan
            plan = await planner.create_migration_plan(
                repository_id=repo_id,
                name=input.plan_name,
                strategy=strategy,
                target_architecture=input.target_architecture,
                risk_tolerance=input.risk_tolerance,
                team_size=input.team_size,
                timeline_weeks=input.timeline_weeks,
            )

            # Get full plan details
            roadmap = await planner.generate_migration_roadmap(plan.id)

            result = {
                "plan_id": plan.id,
                "plan_name": plan.name,
                "strategy": plan.strategy,
                "target_architecture": plan.target_architecture,
                "summary": {
                    "total_steps": len(plan.steps),
                    "total_effort_hours": plan.total_effort_hours,
                    "timeline_weeks": roadmap["timeline"]["total_duration_weeks"],
                    "confidence_level": f"{plan.confidence_level * 100:.0f}%",
                    "risk_count": len(plan.risks),
                },
                "phases": roadmap["phases"],
                "critical_path": roadmap["critical_path"],
                "major_risks": [
                    {
                        "name": risk.name,
                        "level": risk.risk_level,
                        "mitigation": risk.mitigation_strategy,
                    }
                    for risk in plan.risks
                    if risk.risk_level in ["high", "critical"]
                ],
                "success_metrics": plan.success_metrics,
            }

            return create_success_result(
                result, f"Migration plan '{input.plan_name}' created successfully"
            )

        except Exception as e:
            return create_error_result(f"Failed to create migration plan: {str(e)}")


@optimize_migration_plan.on_call
async def optimize_migration_plan_impl(
    input: OptimizeMigrationPlanInput,
) -> ToolResult:
    """Optimize a migration plan."""
    session_factory = await get_session_factory()

    async with session_factory() as session:
        try:
            # Create planner
            planner = MigrationPlanner(session)

            # Build optimization goals
            goals = {}
            if input.minimize_time:
                goals["minimize_time"] = True
            if input.minimize_risk:
                goals["minimize_risk"] = True
            if input.maximize_quality:
                goals["maximize_quality"] = True

            if not goals:
                return create_error_result(
                    "At least one optimization goal must be specified"
                )

            # Optimize plan
            plan = await planner.optimize_migration_plan(input.plan_id, goals)

            # Get updated roadmap
            roadmap = await planner.generate_migration_roadmap(plan.id)

            result = {
                "plan_id": plan.id,
                "optimization_applied": list(goals.keys()),
                "updated_metrics": {
                    "total_effort_hours": plan.total_effort_hours,
                    "timeline_weeks": roadmap["timeline"]["total_duration_weeks"],
                    "confidence_level": f"{plan.confidence_level * 100:.0f}%",
                },
                "changes_made": [
                    decision.description
                    for decision in plan.decisions
                    if decision.decision_type == "optimization"
                ],
            }

            return create_success_result(
                result, "Migration plan optimized successfully"
            )

        except ValueError as e:
            return create_error_result(str(e))
        except Exception as e:
            return create_error_result(f"Failed to optimize plan: {str(e)}")


@generate_migration_roadmap.on_call
async def generate_migration_roadmap_impl(
    input: GenerateMigrationRoadmapInput,
) -> ToolResult:
    """Generate a migration roadmap."""
    session_factory = await get_session_factory()

    async with session_factory() as session:
        try:
            # Create planner
            planner = MigrationPlanner(session)

            # Generate roadmap
            roadmap = await planner.generate_migration_roadmap(input.plan_id)

            # Format for visualization
            result = {
                "plan_name": roadmap["plan_name"],
                "strategy": roadmap["strategy"],
                "timeline": roadmap["timeline"],
                "phases": [
                    {
                        "name": phase["name"],
                        "duration_weeks": phase["total_hours"] / 40,
                        "step_count": len(phase["steps"]),
                        "key_activities": [
                            step["name"] for step in phase["steps"][:3]
                        ],  # Top 3
                    }
                    for phase in roadmap["phases"]
                ],
                "milestones": roadmap["milestones"],
                "critical_path_length": len(roadmap["critical_path"]),
                "resource_requirements": roadmap["resource_allocation"],
                "risk_mitigation_schedule": roadmap["risk_mitigation_schedule"],
            }

            return create_success_result(result, "Migration roadmap generated")

        except ValueError as e:
            return create_error_result(str(e))
        except Exception as e:
            return create_error_result(f"Failed to generate roadmap: {str(e)}")


@identify_migration_patterns.on_call
async def identify_migration_patterns_impl(
    input: IdentifyMigrationPatternsInput,
) -> ToolResult:
    """Identify applicable migration patterns."""
    session_factory = await get_session_factory()

    async with session_factory() as session:
        try:
            # Get repository ID
            repo_id = await get_repository_id(session, input.repository_url)
            if not repo_id:
                return create_error_result(
                    f"Repository {input.repository_url} not found. Please sync it first."
                )

            # Create analyzer
            analyzer = MigrationAnalyzer(session)

            # Identify patterns
            patterns = await analyzer.identify_migration_patterns(repo_id)

            # Format results
            result = {
                "repository_url": input.repository_url,
                "applicable_patterns": [
                    {
                        "name": pattern["name"],
                        "category": pattern["category"],
                        "description": pattern["description"],
                        "applicability": f"{pattern['applicability_score'] * 100:.0f}%",
                        "success_rate": f"{pattern['success_rate'] * 100:.0f}%",
                        "typical_effort_hours": pattern["avg_effort_hours"],
                    }
                    for pattern in patterns[:10]  # Top 10 patterns
                ],
                "pattern_summary": {
                    "total_applicable": len(patterns),
                    "high_confidence": len(
                        [p for p in patterns if p["applicability_score"] > 0.8]
                    ),
                    "categories": list({p["category"] for p in patterns}),
                },
            }

            return create_success_result(
                result, "Migration patterns identified successfully"
            )

        except Exception as e:
            return create_error_result(f"Failed to identify patterns: {str(e)}")


@assess_migration_risks.on_call
async def assess_migration_risks_impl(input: AssessMigrationRisksInput) -> ToolResult:
    """Assess migration risks."""
    session_factory = await get_session_factory()

    async with session_factory() as session:
        try:
            # Get repository ID
            repo_id = await get_repository_id(session, input.repository_url)
            if not repo_id:
                return create_error_result(
                    f"Repository {input.repository_url} not found. Please sync it first."
                )

            # Create analyzer
            analyzer = MigrationAnalyzer(session)

            # Assess risks
            risks = await analyzer.assess_migration_risks(repo_id, input.plan_id)

            # Group risks by type and level
            risks_by_type = {"technical": [], "operational": [], "business": []}
            risks_by_level = {"critical": [], "high": [], "medium": [], "low": []}

            for risk in risks:
                risks_by_type[risk["type"]].append(risk)
                risks_by_level[risk["level"]].append(risk)

            # Calculate overall risk score
            total_risk_score = (
                sum(risk["probability"] * risk["impact"] for risk in risks) / len(risks)
                if risks
                else 0
            )

            result = {
                "repository_url": input.repository_url,
                "risk_summary": {
                    "total_risks": len(risks),
                    "critical_risks": len(risks_by_level["critical"]),
                    "high_risks": len(risks_by_level["high"]),
                    "overall_risk_score": f"{total_risk_score:.2f}",
                    "risk_level": analyzer._calculate_risk_level(total_risk_score, 1.0),
                },
                "risks_by_type": {
                    risk_type: [
                        {
                            "name": r["name"],
                            "level": r["level"],
                            "probability": f"{r['probability'] * 100:.0f}%",
                            "impact": f"{r['impact'] * 100:.0f}%",
                            "mitigation": r["mitigation"],
                        }
                        for r in type_risks
                    ]
                    for risk_type, type_risks in risks_by_type.items()
                },
                "top_risks": [
                    {
                        "name": risk["name"],
                        "type": risk["type"],
                        "level": risk["level"],
                        "description": risk["description"],
                        "mitigation": risk["mitigation"],
                    }
                    for risk in sorted(
                        risks,
                        key=lambda r: r["probability"] * r["impact"],
                        reverse=True,
                    )[:5]
                ],
                "mitigation_priorities": [
                    risk["name"]
                    for risk in risks_by_level["critical"] + risks_by_level["high"]
                ],
            }

            return create_success_result(result, "Migration risk assessment completed")

        except Exception as e:
            return create_error_result(f"Failed to assess risks: {str(e)}")


@analyze_migration_impact.on_call
async def analyze_migration_impact_impl(
    input: AnalyzeMigrationImpactInput,
) -> ToolResult:
    """Analyze migration impact."""
    session_factory = await get_session_factory()

    async with session_factory() as session:
        try:
            # Get repository ID
            repo_id = await get_repository_id(session, input.repository_url)
            if not repo_id:
                return create_error_result(
                    f"Repository {input.repository_url} not found. Please sync it first."
                )

            # Find the package/module
            from sqlalchemy import select

            from src.database.package_models import Package, PackageDependency

            stmt = select(Package).where(
                Package.repository_id == repo_id,
                Package.path == input.module_path,
            )
            result = await session.execute(stmt)
            package = result.scalar_one_or_none()

            if not package:
                return create_error_result(
                    f"Module/package '{input.module_path}' not found in repository"
                )

            # Analyze direct dependencies
            # Get packages that depend on this one
            dependent_stmt = (
                select(Package)
                .join(
                    PackageDependency,
                    Package.id == PackageDependency.source_package_id,
                )
                .where(PackageDependency.target_package_id == package.id)
            )
            dependent_result = await session.execute(dependent_stmt)
            dependents = dependent_result.scalars().all()

            # Get packages this one depends on
            dependency_stmt = (
                select(Package)
                .join(
                    PackageDependency,
                    Package.id == PackageDependency.target_package_id,
                )
                .where(PackageDependency.source_package_id == package.id)
            )
            dependency_result = await session.execute(dependency_stmt)
            dependencies = dependency_result.scalars().all()

            # Calculate impact metrics
            impact_score = len(dependents) * 0.3 + len(dependencies) * 0.2

            # Determine interface changes needed
            interface_changes = []
            if input.impact_type == "extraction":
                interface_changes = [
                    "Create public API for extracted module",
                    "Define data contracts for communication",
                    "Implement dependency injection for loose coupling",
                ]
            elif input.impact_type == "refactoring":
                interface_changes = [
                    "Update method signatures if needed",
                    "Ensure backward compatibility",
                    "Add deprecation warnings for changed APIs",
                ]
            elif input.impact_type == "interface_change":
                interface_changes = [
                    "Version the interface changes",
                    "Update all dependent modules",
                    "Provide migration guide for consumers",
                ]

            # Testing requirements
            testing_requirements = {
                "unit_tests": f"Update tests for {package.name}",
                "integration_tests": f"Test interactions with {len(dependents)} dependent modules",
                "contract_tests": "Verify interface contracts are maintained",
                "performance_tests": "Benchmark before/after performance",
            }

            result = {
                "module": input.module_path,
                "impact_type": input.impact_type,
                "impact_summary": {
                    "direct_dependents": len(dependents),
                    "direct_dependencies": len(dependencies),
                    "impact_score": f"{impact_score:.2f}",
                    "complexity": (
                        "high"
                        if impact_score > 5
                        else "medium" if impact_score > 2 else "low"
                    ),
                },
                "affected_modules": {
                    "dependents": [
                        {
                            "path": dep.path,
                            "name": dep.name,
                            "coupling_strength": (
                                "high"
                                if dep.path.startswith(package.path.rsplit("/", 1)[0])
                                else "medium"
                            ),
                        }
                        for dep in dependents[:10]  # Top 10
                    ],
                    "dependencies": [
                        {"path": dep.path, "name": dep.name}
                        for dep in dependencies[:10]  # Top 10
                    ],
                },
                "interface_changes_required": interface_changes,
                "testing_requirements": testing_requirements,
                "migration_steps": [
                    f"1. Analyze current usage patterns in {len(dependents)} dependent modules",
                    f"2. Design new interface for {input.impact_type}",
                    "3. Implement changes with backward compatibility",
                    "4. Update and run all tests",
                    "5. Migrate dependent modules incrementally",
                    "6. Remove deprecated code after migration",
                ],
                "estimated_effort": {
                    "refactoring_hours": 8 + len(dependents) * 2,
                    "testing_hours": 16 + len(dependents) * 4,
                    "total_hours": 24 + len(dependents) * 6,
                },
            }

            return create_success_result(
                result, f"Impact analysis for {input.module_path} completed"
            )

        except Exception as e:
            return create_error_result(f"Failed to analyze impact: {str(e)}")


@estimate_migration_effort.on_call
async def estimate_migration_effort_impl(
    input: EstimateMigrationEffortInput,
) -> ToolResult:
    """Estimate migration effort."""
    session_factory = await get_session_factory()

    async with session_factory() as session:
        try:
            from sqlalchemy.orm import selectinload

            from src.database.migration_models import MigrationPlan

            # Get plan with steps
            plan = await session.get(
                MigrationPlan,
                input.plan_id,
                options=[
                    selectinload(MigrationPlan.steps),
                    selectinload(MigrationPlan.risks),
                ],
            )

            if not plan:
                return create_error_result(f"Migration plan {input.plan_id} not found")

            # Calculate base development effort
            dev_hours = sum(step.estimated_hours or 0 for step in plan.steps)

            # Adjust for productivity
            adjusted_dev_hours = dev_hours / input.productivity_factor

            # Calculate testing effort
            testing_hours = 0
            if input.include_testing:
                # Unit tests: 40% of dev time
                unit_test_hours = adjusted_dev_hours * 0.4
                # Integration tests: 60% of dev time
                integration_test_hours = adjusted_dev_hours * 0.6
                # Performance tests: 20% of dev time
                performance_test_hours = adjusted_dev_hours * 0.2
                testing_hours = (
                    unit_test_hours + integration_test_hours + performance_test_hours
                )

            # Calculate documentation effort
            doc_hours = 0
            if input.include_documentation:
                # Technical docs: 15% of dev time
                tech_doc_hours = adjusted_dev_hours * 0.15
                # User docs: 10% of dev time
                user_doc_hours = adjusted_dev_hours * 0.10
                doc_hours = tech_doc_hours + user_doc_hours

            # Calculate overhead
            # Planning and coordination: 20% of total
            planning_hours = (adjusted_dev_hours + testing_hours + doc_hours) * 0.2
            # Code reviews: 10% of dev time
            review_hours = adjusted_dev_hours * 0.1

            # Risk buffer based on risk assessment
            risk_buffer_percent = 0.1  # Base 10%
            high_risk_count = len(
                [r for r in plan.risks if r.risk_level in ["high", "critical"]]
            )
            risk_buffer_percent += high_risk_count * 0.05  # +5% per high risk
            risk_buffer_hours = (
                adjusted_dev_hours + testing_hours + doc_hours + planning_hours
            ) * risk_buffer_percent

            # Total effort
            total_hours = (
                adjusted_dev_hours
                + testing_hours
                + doc_hours
                + planning_hours
                + review_hours
                + risk_buffer_hours
            )

            # Break down by phase
            phase_breakdown = {}
            step_types = {}
            for step in plan.steps:
                if step.step_type not in step_types:
                    step_types[step.step_type] = []
                step_types[step.step_type].append(step)

            for step_type, steps in step_types.items():
                phase_hours = sum(s.estimated_hours or 0 for s in steps)
                phase_breakdown[step_type] = {
                    "development": phase_hours / input.productivity_factor,
                    "testing": phase_hours * 0.6 if input.include_testing else 0,
                    "documentation": (
                        phase_hours * 0.1 if input.include_documentation else 0
                    ),
                }

            result = {
                "plan_id": input.plan_id,
                "effort_summary": {
                    "total_hours": round(total_hours),
                    "total_person_days": round(total_hours / 8),
                    "total_person_weeks": round(total_hours / 40),
                    "productivity_factor": input.productivity_factor,
                },
                "effort_breakdown": {
                    "development": round(adjusted_dev_hours),
                    "testing": round(testing_hours),
                    "documentation": round(doc_hours),
                    "planning_coordination": round(planning_hours),
                    "code_reviews": round(review_hours),
                    "risk_buffer": round(risk_buffer_hours),
                },
                "testing_breakdown": {
                    "unit_tests": (
                        round(unit_test_hours) if input.include_testing else 0
                    ),
                    "integration_tests": (
                        round(integration_test_hours) if input.include_testing else 0
                    ),
                    "performance_tests": (
                        round(performance_test_hours) if input.include_testing else 0
                    ),
                },
                "phase_breakdown": phase_breakdown,
                "assumptions": [
                    f"Productivity factor: {input.productivity_factor}",
                    f"Testing included: {input.include_testing}",
                    f"Documentation included: {input.include_documentation}",
                    f"Risk buffer: {risk_buffer_percent * 100:.0f}% based on {high_risk_count} high risks",
                ],
            }

            return create_success_result(
                result, "Migration effort estimation completed"
            )

        except Exception as e:
            return create_error_result(f"Failed to estimate effort: {str(e)}")


@plan_migration_resources.on_call
async def plan_migration_resources_impl(
    input: PlanMigrationResourcesInput,
) -> ToolResult:
    """Plan migration resources."""
    session_factory = await get_session_factory()

    async with session_factory() as session:
        try:
            from sqlalchemy.orm import selectinload

            from src.database.migration_models import MigrationPlan
            from src.services.migration_planner import MigrationPlanner

            # Get plan
            plan = await session.get(
                MigrationPlan,
                input.plan_id,
                options=[
                    selectinload(MigrationPlan.steps),
                ],
            )

            if not plan:
                return create_error_result(f"Migration plan {input.plan_id} not found")

            # Get roadmap for timeline
            planner = MigrationPlanner(session)
            roadmap = await planner.generate_migration_roadmap(input.plan_id)

            # Calculate total capacity
            total_team_size = (
                input.available_developers
                + input.available_architects
                + input.available_qa
            )
            sprint_capacity_hours = total_team_size * input.sprint_weeks * 40

            # Calculate sprints needed
            total_hours = plan.total_effort_hours or 0
            sprints_needed = int((total_hours / sprint_capacity_hours) + 0.5)

            # Allocate resources by sprint
            sprint_plan = []
            remaining_phases = list(roadmap["phases"])
            current_sprint = 1

            while remaining_phases and current_sprint <= sprints_needed:
                sprint = {
                    "sprint": current_sprint,
                    "weeks": input.sprint_weeks,
                    "phases": [],
                    "allocation": {
                        "developers": 0,
                        "architects": 0,
                        "qa": 0,
                    },
                    "focus_areas": [],
                }

                # Allocate phases to sprint
                sprint_hours = 0
                while remaining_phases and sprint_hours < sprint_capacity_hours * 0.8:
                    phase = remaining_phases[0]
                    phase_hours = phase["total_hours"]

                    if sprint_hours + phase_hours <= sprint_capacity_hours:
                        sprint["phases"].append(phase["name"])
                        sprint_hours += phase_hours
                        remaining_phases.pop(0)

                        # Determine allocation based on phase
                        if phase["name"] in ["Design", "Preparation"]:
                            sprint["allocation"]["architects"] = min(
                                input.available_architects, 2
                            )
                            sprint["allocation"]["developers"] = (
                                input.available_developers - 2
                            )
                        elif phase["name"] in ["Validation", "Testing"]:
                            sprint["allocation"]["qa"] = input.available_qa
                            sprint["allocation"]["developers"] = (
                                input.available_developers - 1
                            )
                        else:
                            sprint["allocation"][
                                "developers"
                            ] = input.available_developers
                            sprint["allocation"]["architects"] = 1
                            sprint["allocation"]["qa"] = max(1, input.available_qa // 2)

                        sprint["focus_areas"] = _determine_focus_areas(phase["name"])
                    else:
                        break

                sprint_plan.append(sprint)
                current_sprint += 1

            # Identify skill requirements
            skill_requirements = {
                "developers": [
                    "refactoring",
                    "design_patterns",
                    "testing",
                    "api_design",
                ],
                "architects": [
                    "system_design",
                    "interface_design",
                    "migration_patterns",
                    "integration_patterns",
                ],
                "qa": [
                    "test_automation",
                    "integration_testing",
                    "performance_testing",
                    "contract_testing",
                ],
            }

            # Calculate utilization
            total_available_hours = (
                total_team_size * sprints_needed * input.sprint_weeks * 40
            )
            utilization = (
                (total_hours / total_available_hours) * 100
                if total_available_hours > 0
                else 0
            )

            result = {
                "plan_id": input.plan_id,
                "resource_summary": {
                    "team_size": total_team_size,
                    "sprints_needed": sprints_needed,
                    "total_weeks": sprints_needed * input.sprint_weeks,
                    "utilization_percent": f"{utilization:.1f}%",
                },
                "sprint_plan": sprint_plan,
                "skill_requirements": skill_requirements,
                "resource_risks": _identify_resource_risks(
                    sprints_needed, utilization, total_team_size
                ),
                "recommendations": [
                    "Consider pair programming for knowledge transfer",
                    "Rotate team members across phases for cross-training",
                    "Maintain at least 20% capacity buffer for unknowns",
                    "Schedule regular architecture reviews",
                ],
            }

            return create_success_result(
                result, "Resource planning completed successfully"
            )

        except Exception as e:
            return create_error_result(f"Failed to plan resources: {str(e)}")


def _determine_focus_areas(phase_name: str) -> list[str]:
    """Determine focus areas for a phase."""
    focus_map = {
        "Preparation": ["test_coverage", "dependency_analysis", "documentation"],
        "Design": ["interface_design", "contract_definition", "api_specification"],
        "Extraction": ["refactoring", "modularization", "decoupling"],
        "Validation": [
            "integration_testing",
            "performance_testing",
            "regression_testing",
        ],
        "Optimization": ["performance_tuning", "code_cleanup", "documentation"],
    }
    return focus_map.get(phase_name, ["implementation"])


def _identify_resource_risks(
    sprints_needed: int, utilization: float, team_size: int
) -> list[str]:
    """Identify resource-related risks."""
    risks = []

    if utilization > 85:
        risks.append("High utilization - limited buffer for issues")
    if sprints_needed > 10:
        risks.append("Long timeline - risk of team changes")
    if team_size < 5:
        risks.append("Small team - potential bottlenecks")

    return risks


@design_module_interface.on_call
async def design_module_interface_impl(
    input: DesignModuleInterfaceInput,
) -> ToolResult:
    """Design module interface."""
    session_factory = await get_session_factory()

    async with session_factory() as session:
        try:
            from sqlalchemy import select

            from src.database.package_models import Package

            # Get repository ID
            repo_id = await get_repository_id(session, input.repository_url)
            if not repo_id:
                return create_error_result(
                    f"Repository {input.repository_url} not found. Please sync it first."
                )

            # Find package
            stmt = select(Package).where(
                Package.repository_id == repo_id,
                Package.path == input.package_path,
            )
            result = await session.execute(stmt)
            package = result.scalar_one_or_none()

            if not package:
                return create_error_result(
                    f"Package '{input.package_path}' not found in repository"
                )

            # Create interface designer
            designer = InterfaceDesigner(session)

            # Design interface
            interface_design = await designer.design_module_interface(
                package.id,
                input.target_architecture,
            )

            # Add events if requested
            if input.include_events and input.target_architecture == "modular_monolith":
                interface_design["events"] = [
                    {
                        "name": f"{package.name}ModuleInitialized",
                        "description": "Raised when module is initialized",
                        "payload": {"module_name": "str", "version": "str"},
                    }
                ]

            return create_success_result(
                interface_design, f"Interface design for {input.package_path} completed"
            )

        except Exception as e:
            return create_error_result(f"Failed to design interface: {str(e)}")


@generate_interface_documentation.on_call
async def generate_interface_documentation_impl(
    input: GenerateInterfaceDocumentationInput,
) -> ToolResult:
    """Generate interface documentation."""
    session_factory = await get_session_factory()

    async with session_factory() as session:
        try:
            # Create interface designer
            designer = InterfaceDesigner(session)

            # Generate documentation
            documentation = await designer.generate_interface_documentation(
                input.interface_design
            )

            result = {
                "documentation": documentation,
                "format": "markdown",
                "sections": [
                    "Public API",
                    "Data Contracts",
                    "Domain Events" if input.interface_design.get("events") else None,
                    "Dependencies",
                    "Implementation Notes",
                ],
            }

            # Remove None sections
            result["sections"] = [s for s in result["sections"] if s]

            return create_success_result(
                result, "Interface documentation generated successfully"
            )

        except Exception as e:
            return create_error_result(f"Failed to generate documentation: {str(e)}")
