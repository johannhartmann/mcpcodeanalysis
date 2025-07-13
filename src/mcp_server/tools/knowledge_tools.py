"""MCP tools for migration knowledge management and learning."""

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
from src.services.pattern_library import PatternLibrary


# Tool input models
class ExtractPatternsInput(BaseModel):
    """Input for extract_migration_patterns tool."""

    plan_id: int = Field(
        description="ID of completed migration plan to extract patterns from"
    )


class AddPatternInput(BaseModel):
    """Input for add_pattern_to_library tool."""

    name: str = Field(description="Name of the pattern")
    category: str = Field(
        description="Pattern category (extraction, refactoring, interface_design, etc.)",
        default="general",
    )
    description: str = Field(description="Detailed description of the pattern")
    implementation_steps: list[str] = Field(
        description="Step-by-step implementation guide",
        default_factory=list,
    )
    prerequisites: list[str] = Field(
        description="Prerequisites for using this pattern",
        default_factory=list,
    )
    best_practices: list[str] = Field(
        description="Best practices when applying this pattern",
        default_factory=list,
    )
    anti_patterns: list[str] = Field(
        description="Anti-patterns to avoid",
        default_factory=list,
    )
    example_code: str | None = Field(
        description="Example code demonstrating the pattern",
        default=None,
    )
    tools_required: list[str] = Field(
        description="Tools required for this pattern",
        default_factory=list,
    )
    avg_effort_hours: float | None = Field(
        description="Average effort in hours",
        default=None,
    )


class SearchPatternsInput(BaseModel):
    """Input for search_patterns tool."""

    query: str | None = Field(
        description="Text search query",
        default=None,
    )
    category: str | None = Field(
        description="Filter by pattern category",
        default=None,
    )
    min_success_rate: float | None = Field(
        description="Minimum success rate (0-1)",
        default=None,
    )
    applicable_to: dict[str, Any] | None = Field(
        description="Scenario applicability filter",
        default=None,
    )


class GetPatternRecommendationsInput(BaseModel):
    """Input for get_pattern_recommendations tool."""

    repository_url: str = Field(description="Repository to get recommendations for")
    context: dict[str, Any] = Field(
        description="Current migration context (team_size, timeline, complexity, etc.)",
        default_factory=dict,
    )


class UpdatePatternInput(BaseModel):
    """Input for update_pattern_from_execution tool."""

    pattern_id: int = Field(description="ID of the pattern that was used")
    execution_data: dict[str, Any] = Field(
        description="Execution results including success, actual_hours, scenario"
    )


class LearnFromFailuresInput(BaseModel):
    """Input for learn_from_failures tool."""

    plan_id: int = Field(description="ID of failed migration plan to analyze")


class GeneratePatternDocInput(BaseModel):
    """Input for generate_pattern_documentation tool."""

    pattern_id: int = Field(description="ID of the pattern to document")


class ShareKnowledgeInput(BaseModel):
    """Input for share_migration_knowledge tool."""

    knowledge_type: str = Field(
        description="Type of knowledge to share (patterns, lessons, best_practices)",
        default="patterns",
    )
    format: str = Field(
        description="Output format (markdown, json)",
        default="markdown",
    )
    filter_category: str | None = Field(
        description="Optional category filter",
        default=None,
    )


# Tool implementations
async def extract_migration_patterns_impl(
    input_data: ExtractPatternsInput,
) -> ToolResult:
    """Extract reusable patterns from completed migration."""
    try:
        session_factory = await get_session_factory()
        async with session_factory() as session:
            library = PatternLibrary(session)
            patterns = await library.extract_patterns_from_execution(input_data.plan_id)

            if patterns:
                return create_success_result(
                    f"Extracted {len(patterns)} patterns from migration plan",
                    {
                        "patterns": patterns,
                        "count": len(patterns),
                        "categories": list(
                            {p.get("category", "general") for p in patterns}
                        ),
                    },
                )
            else:
                return create_success_result(
                    "No patterns extracted from migration plan",
                    {"patterns": [], "count": 0},
                )

    except Exception as e:
        return create_error_result(str(e))


async def add_pattern_to_library_impl(input_data: AddPatternInput) -> ToolResult:
    """Add a new pattern to the knowledge library."""
    try:
        session_factory = await get_session_factory()
        async with session_factory() as session:
            library = PatternLibrary(session)

            pattern_data = {
                "name": input_data.name,
                "category": input_data.category,
                "description": input_data.description,
                "implementation_steps": input_data.implementation_steps,
                "prerequisites": input_data.prerequisites,
                "best_practices": input_data.best_practices,
                "anti_patterns": input_data.anti_patterns,
                "example_code": input_data.example_code,
                "tools_required": input_data.tools_required,
                "avg_effort_hours": input_data.avg_effort_hours,
            }

            pattern = await library.add_pattern_to_library(pattern_data)

            return create_success_result(
                f"Added pattern '{pattern.name}' to library",
                {
                    "pattern_id": pattern.id,
                    "name": pattern.name,
                    "category": pattern.category,
                },
            )

    except Exception as e:
        return create_error_result(str(e))


async def search_patterns_impl(input_data: SearchPatternsInput) -> ToolResult:
    """Search for patterns in the knowledge library."""
    try:
        session_factory = await get_session_factory()
        async with session_factory() as session:
            library = PatternLibrary(session)
            patterns = await library.search_patterns(
                query=input_data.query,
                category=input_data.category,
                min_success_rate=input_data.min_success_rate,
                applicable_to=input_data.applicable_to,
            )

            return create_success_result(
                f"Found {len(patterns)} matching patterns",
                {
                    "patterns": patterns,
                    "count": len(patterns),
                    "top_pattern": patterns[0] if patterns else None,
                },
            )

    except Exception as e:
        return create_error_result(str(e))


async def get_pattern_recommendations_impl(
    input_data: GetPatternRecommendationsInput,
) -> ToolResult:
    """Get pattern recommendations for a repository."""
    try:
        session_factory = await get_session_factory()
        async with session_factory() as session:
            repository_id = await get_repository_id(session, input_data.repository_url)

            library = PatternLibrary(session)
            recommendations = await library.get_pattern_recommendations(
                repository_id, input_data.context
            )

            return create_success_result(
                f"Generated {len(recommendations)} pattern recommendations",
                {
                    "recommendations": recommendations,
                    "count": len(recommendations),
                    "top_recommendation": (
                        recommendations[0] if recommendations else None
                    ),
                },
            )

    except Exception as e:
        return create_error_result(str(e))


async def update_pattern_from_execution_impl(
    input_data: UpdatePatternInput,
) -> ToolResult:
    """Update pattern statistics based on execution results."""
    try:
        session_factory = await get_session_factory()
        async with session_factory() as session:
            library = PatternLibrary(session)
            await library.update_pattern_from_execution(
                input_data.pattern_id, input_data.execution_data
            )

            return create_success_result(
                f"Updated pattern {input_data.pattern_id} with execution results",
                {
                    "pattern_id": input_data.pattern_id,
                    "success": input_data.execution_data.get("success", False),
                },
            )

    except Exception as e:
        return create_error_result(str(e))


async def learn_from_failures_impl(input_data: LearnFromFailuresInput) -> ToolResult:
    """Analyze failed migrations to extract lessons learned."""
    try:
        session_factory = await get_session_factory()
        async with session_factory() as session:
            library = PatternLibrary(session)
            lessons = await library.learn_from_failures(input_data.plan_id)

            failure_count = len(lessons.get("failure_analysis", []))
            prevention_count = len(lessons.get("prevention_strategies", []))

            return create_success_result(
                f"Analyzed {failure_count} failures and generated {prevention_count} prevention strategies",
                lessons,
            )

    except Exception as e:
        return create_error_result(str(e))


async def generate_pattern_documentation_impl(
    input_data: GeneratePatternDocInput,
) -> ToolResult:
    """Generate comprehensive documentation for a pattern."""
    try:
        session_factory = await get_session_factory()
        async with session_factory() as session:
            library = PatternLibrary(session)
            documentation = await library.generate_pattern_documentation(
                input_data.pattern_id
            )

            return create_success_result(
                "Generated pattern documentation",
                {
                    "pattern_id": input_data.pattern_id,
                    "documentation": documentation,
                    "format": "markdown",
                },
            )

    except Exception as e:
        return create_error_result(str(e))


async def share_migration_knowledge_impl(
    input_data: ShareKnowledgeInput,
) -> ToolResult:
    """Share migration knowledge in various formats."""
    try:
        session_factory = await get_session_factory()
        async with session_factory() as session:
            library = PatternLibrary(session)

            if input_data.knowledge_type == "patterns":
                # Get all patterns
                patterns = await library.search_patterns(
                    category=input_data.filter_category,
                    min_success_rate=0.5,
                )

                if input_data.format == "markdown":
                    # Generate markdown report
                    report_lines = [
                        "# Migration Pattern Library",
                        "",
                        f"Total patterns: {len(patterns)}",
                        "",
                    ]

                    # Group by category
                    by_category = {}
                    for pattern in patterns:
                        cat = pattern.get("category", "general")
                        if cat not in by_category:
                            by_category[cat] = []
                        by_category[cat].append(pattern)

                    for category, cat_patterns in sorted(by_category.items()):
                        report_lines.extend(
                            [
                                f"## {category.replace('_', ' ').title()}",
                                "",
                            ]
                        )

                        for pattern in sorted(
                            cat_patterns,
                            key=lambda x: x.get("success_rate", 0),
                            reverse=True,
                        ):
                            report_lines.extend(
                                [
                                    f"### {pattern['name']}",
                                    f"- Success Rate: {pattern.get('success_rate', 0) * 100:.0f}%",
                                    f"- Usage Count: {pattern.get('usage_count', 0)}",
                                    f"- Description: {pattern.get('description', 'N/A')}",
                                    "",
                                ]
                            )

                    content = "\n".join(report_lines)
                else:
                    # JSON format
                    content = {
                        "patterns": patterns,
                        "total": len(patterns),
                        "categories": (
                            list(by_category.keys())
                            if "by_category" in locals()
                            else []
                        ),
                    }

                return create_success_result(
                    f"Shared {len(patterns)} patterns",
                    {
                        "knowledge_type": input_data.knowledge_type,
                        "format": input_data.format,
                        "content": content,
                        "item_count": len(patterns),
                    },
                )

            elif input_data.knowledge_type == "best_practices":
                # Aggregate best practices from all patterns
                patterns = await library.search_patterns()
                all_practices = []

                for pattern in patterns:
                    practices = pattern.get("best_practices", [])
                    for practice in practices:
                        all_practices.append(
                            {
                                "practice": practice,
                                "pattern": pattern["name"],
                                "category": pattern.get("category", "general"),
                            }
                        )

                if input_data.format == "markdown":
                    report_lines = [
                        "# Migration Best Practices",
                        "",
                        f"Collected from {len(patterns)} patterns",
                        "",
                    ]

                    # Group by category
                    by_category = {}
                    for item in all_practices:
                        cat = item["category"]
                        if cat not in by_category:
                            by_category[cat] = []
                        by_category[cat].append(item)

                    for category, items in sorted(by_category.items()):
                        report_lines.extend(
                            [
                                f"## {category.replace('_', ' ').title()}",
                                "",
                            ]
                        )

                        for item in items:
                            report_lines.append(
                                f"- {item['practice']} (from {item['pattern']})"
                            )

                        report_lines.append("")

                    content = "\n".join(report_lines)
                else:
                    content = {
                        "best_practices": all_practices,
                        "total": len(all_practices),
                        "source_patterns": len(patterns),
                    }

                return create_success_result(
                    f"Shared {len(all_practices)} best practices",
                    {
                        "knowledge_type": input_data.knowledge_type,
                        "format": input_data.format,
                        "content": content,
                        "item_count": len(all_practices),
                    },
                )

            else:
                return create_error_result(
                    f"Unknown knowledge type: {input_data.knowledge_type}"
                )

    except Exception as e:
        return create_error_result(str(e))


# Tool definitions
extract_migration_patterns = Tool(
    name="extract_migration_patterns",
    description="Extract reusable patterns from a completed migration plan. Analyzes successful steps and strategies to build the pattern library.",
    input_schema=ExtractPatternsInput,
    output_schema=ToolResult,
    fn=extract_migration_patterns_impl,
)

add_pattern_to_library = Tool(
    name="add_pattern_to_library",
    description="Add a new migration pattern to the knowledge library. Use this to document proven patterns and strategies.",
    input_schema=AddPatternInput,
    output_schema=ToolResult,
    fn=add_pattern_to_library_impl,
)

search_patterns = Tool(
    name="search_patterns",
    description="Search the pattern library by text, category, success rate, or applicability. Find proven patterns for your migration needs.",
    input_schema=SearchPatternsInput,
    output_schema=ToolResult,
    fn=search_patterns_impl,
)

get_pattern_recommendations = Tool(
    name="get_pattern_recommendations",
    description="Get pattern recommendations based on repository characteristics and migration context. Uses ML to match patterns to your specific needs.",
    input_schema=GetPatternRecommendationsInput,
    output_schema=ToolResult,
    fn=get_pattern_recommendations_impl,
)

update_pattern_from_execution = Tool(
    name="update_pattern_from_execution",
    description="Update pattern statistics based on execution results. Helps the system learn and improve recommendations over time.",
    input_schema=UpdatePatternInput,
    output_schema=ToolResult,
    fn=update_pattern_from_execution_impl,
)

learn_from_failures = Tool(
    name="learn_from_failures",
    description="Analyze failed migrations to extract lessons learned. Identifies root causes and generates prevention strategies.",
    input_schema=LearnFromFailuresInput,
    output_schema=ToolResult,
    fn=learn_from_failures_impl,
)

generate_pattern_documentation = Tool(
    name="generate_pattern_documentation",
    description="Generate comprehensive documentation for a migration pattern including examples, best practices, and usage history.",
    input_schema=GeneratePatternDocInput,
    output_schema=ToolResult,
    fn=generate_pattern_documentation_impl,
)

share_migration_knowledge = Tool(
    name="share_migration_knowledge",
    description="Share migration knowledge (patterns, lessons, best practices) in various formats. Great for team training and knowledge transfer.",
    input_schema=ShareKnowledgeInput,
    output_schema=ToolResult,
    fn=share_migration_knowledge_impl,
)
