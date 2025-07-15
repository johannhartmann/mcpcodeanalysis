"""Migration analysis resources for read-only access to migration data."""

from fastmcp import FastMCP
from sqlalchemy import func, select

from src.services.migration_analyzer import MigrationAnalyzer
from src.services.pattern_library import PatternLibrary


class MigrationAnalysisResources:
    """Resources for migration analysis data access."""

    def __init__(self, mcp: FastMCP, session_maker):
        """Initialize migration analysis resources."""
        self.mcp = mcp
        self.session_maker = session_maker

    def register_resources(self):
        """Register all migration analysis resources."""

        @self.mcp.resource("migration://readiness/{repository_url}")
        async def get_migration_readiness(repository_url: str) -> str:
            """Get migration readiness analysis for a repository."""
            async with self.session_maker() as session:
                analyzer = MigrationAnalyzer(session)

                try:
                    # Analyze repository for migration readiness
                    readiness = await analyzer.analyze_repository(repository_url)

                    return f"""# Migration Readiness Analysis

**Repository**: {repository_url}
**Overall Readiness Score**: {readiness.get('overall_score', 0):.1f}/10

## Bounded Contexts Identified
{self._format_bounded_contexts(readiness.get('bounded_contexts', []))}

## Migration Candidates
{self._format_migration_candidates(readiness.get('candidates', []))}

## Complexity Metrics
- **Total Files**: {readiness.get('metrics', {}).get('total_files', 0)}
- **Total Lines**: {readiness.get('metrics', {}).get('total_lines', 0)}
- **Cyclomatic Complexity**: {readiness.get('metrics', {}).get('avg_complexity', 0):.1f}
- **Coupling Score**: {readiness.get('metrics', {}).get('coupling_score', 0):.1f}

## Recommendations
{self._format_recommendations(readiness.get('recommendations', []))}
"""
                except Exception as e:
                    return f"Error analyzing migration readiness: {str(e)}"

        @self.mcp.resource("migration://patterns/search")
        async def search_migration_patterns() -> str:
            """Search migration patterns in the pattern library."""
            async with self.session_maker() as session:
                library = PatternLibrary(session)

                try:
                    # For now, return all patterns
                    # In a real implementation, this would handle query params through the request context
                    patterns = await library.search_patterns()

                    if not patterns:
                        return "No migration patterns found matching the criteria."

                    result = "# Migration Patterns Search Results\n\n"
                    for pattern in patterns:
                        result += f"""## {pattern['name']}
**Category**: {pattern['category']}
**Context Type**: {pattern['context_type']}
**Success Rate**: {pattern['success_rate']:.1%}
**Usage Count**: {pattern['usage_count']}

### Description
{pattern['description']}

### Implementation
```{pattern.get('language', 'python')}
{pattern['implementation']}
```

---

"""
                    return result

                except Exception as e:
                    return f"Error searching migration patterns: {str(e)}"

        @self.mcp.resource("migration://patterns/stats")
        async def get_pattern_library_stats() -> str:
            """Get statistics about the pattern library."""
            async with self.session_maker() as session:
                library = PatternLibrary(session)

                try:
                    # Get pattern statistics directly from database
                    from src.database.migration_models import MigrationPattern

                    # Total patterns
                    total_patterns = (
                        await session.scalar(
                            select(func.count()).select_from(MigrationPattern)
                        )
                        or 0
                    )

                    # Total usage
                    total_usage = (
                        await session.scalar(
                            select(func.sum(MigrationPattern.usage_count)).select_from(
                                MigrationPattern
                            )
                        )
                        or 0
                    )

                    # Average success rate
                    avg_success_rate = (
                        await session.scalar(
                            select(func.avg(MigrationPattern.success_rate)).select_from(
                                MigrationPattern
                            )
                        )
                        or 0.0
                    )

                    # Patterns by category
                    category_stats = await session.execute(
                        select(
                            MigrationPattern.category,
                            func.count(MigrationPattern.id).label("count"),
                            func.avg(MigrationPattern.success_rate).label(
                                "avg_success_rate"
                            ),
                        )
                        .group_by(MigrationPattern.category)
                        .order_by(func.count(MigrationPattern.id).desc())
                    )

                    # Top patterns by usage
                    top_patterns = await session.execute(
                        select(MigrationPattern)
                        .order_by(MigrationPattern.usage_count.desc())
                        .limit(5)
                    )

                    # Recent patterns
                    recent_patterns = await session.execute(
                        select(MigrationPattern)
                        .order_by(MigrationPattern.created_at.desc())
                        .limit(5)
                    )

                    return f"""# Pattern Library Statistics

## Overview
- **Total Patterns**: {total_patterns}
- **Total Usage**: {total_usage}
- **Average Success Rate**: {avg_success_rate:.1%}

## Patterns by Category
{self._format_category_stats_from_results(category_stats)}

## Top Patterns by Usage
{self._format_patterns_list(top_patterns.scalars().all())}

## Recent Patterns
{self._format_patterns_list(recent_patterns.scalars().all())}
"""
                except Exception as e:
                    return f"Error getting pattern library stats: {str(e)}"

        @self.mcp.resource("migration://dashboard/{repository_url}")
        async def get_migration_dashboard(repository_url: str | None = None) -> str:
            """Get migration dashboard with summary metrics."""
            async with self.session_maker() as session:
                try:
                    # For now, return a placeholder dashboard
                    # In a real implementation, this would query migration plans
                    result = "# Migration Dashboard\n\n"

                    if repository_url:
                        result += f"**Repository**: {repository_url}\n\n"

                    result += """## Summary
- **Total Plans**: 0
- **Active Plans**: 0
- **Completed Plans**: 0

## Status
No migration plans have been created yet.

To create a migration plan, use the `create_migration_plan` tool.
"""

                    return result

                except Exception as e:
                    return f"Error getting migration dashboard: {str(e)}"

    def _format_category_stats_from_results(self, category_stats) -> str:
        """Format category statistics from query results."""
        output = ""
        for stat in category_stats:
            output += f"- **{stat.category}**: {stat.count} patterns, {stat.avg_success_rate:.1%} success rate\n"
        return output or "No category statistics available"

    def _format_patterns_list(self, patterns: list) -> str:
        """Format a list of patterns."""
        if not patterns:
            return "No patterns available"

        output = ""
        for pattern in patterns:
            output += f"- **{pattern.name}** ({pattern.category}): "
            output += (
                f"{pattern.usage_count} uses, {pattern.success_rate:.1%} success rate\n"
            )
        return output

    def _format_bounded_contexts(self, contexts: list) -> str:
        """Format bounded contexts for display."""
        if not contexts:
            return "No bounded contexts identified."

        result = ""
        for ctx in contexts:
            result += f"- **{ctx['name']}**: {ctx['description']}\n"
            result += f"  - Entities: {', '.join(ctx.get('entities', []))}\n"
            result += f"  - Cohesion Score: {ctx.get('cohesion_score', 0):.2f}\n"
        return result

    def _format_migration_candidates(self, candidates: list) -> str:
        """Format migration candidates for display."""
        if not candidates:
            return "No migration candidates identified."

        result = ""
        for candidate in candidates:
            result += f"- **{candidate['name']}**\n"
            result += f"  - Type: {candidate['type']}\n"
            result += f"  - Priority: {candidate['priority']}\n"
            result += f"  - Estimated Effort: {candidate['effort']}\n"
        return result

    def _format_recommendations(self, recommendations: list) -> str:
        """Format recommendations for display."""
        if not recommendations:
            return "No specific recommendations."

        return "\n".join(f"- {rec}" for rec in recommendations)

    def _format_category_stats(self, by_category: dict) -> str:
        """Format category statistics."""
        if not by_category:
            return "No category data available."

        result = ""
        for category, stats in by_category.items():
            result += f"- **{category}**: {stats['count']} patterns "
            result += f"(avg success: {stats['avg_success_rate']:.1%})\n"
        return result

    def _format_context_stats(self, by_context: dict) -> str:
        """Format context type statistics."""
        if not by_context:
            return "No context type data available."

        result = ""
        for context_type, count in by_context.items():
            result += f"- **{context_type}**: {count} patterns\n"
        return result

    def _format_top_patterns(self, patterns: list) -> str:
        """Format top patterns by usage."""
        if not patterns:
            return "No usage data available."

        result = ""
        for i, pattern in enumerate(patterns[:5], 1):
            result += f"{i}. **{pattern['name']}** - {pattern['usage_count']} uses "
            result += f"({pattern['success_rate']:.1%} success)\n"
        return result

    def _format_recent_patterns(self, patterns: list) -> str:
        """Format recently added patterns."""
        if not patterns:
            return "No recent patterns."

        result = ""
        for pattern in patterns[:5]:
            result += f"- **{pattern['name']}** ({pattern['category']}) "
            result += f"- Added {pattern['created_at']}\n"
        return result

    def _calculate_progress(self, plan: dict) -> float:
        """Calculate overall progress of a migration plan."""
        steps = plan.get("steps", [])
        if not steps:
            return 0.0

        completed = sum(1 for s in steps if s.get("status") == "completed")
        return completed / len(steps)

    def _get_current_phase(self, plan: dict) -> str:
        """Get the current phase of a migration plan."""
        steps = plan.get("steps", [])
        for step in steps:
            if step.get("status") == "in_progress":
                return step.get("name", "Unknown")

        # Find next pending step
        for step in steps:
            if step.get("status") == "pending":
                return f"Next: {step.get('name', 'Unknown')}"

        return (
            "Completed"
            if all(s.get("status") == "completed" for s in steps)
            else "Not started"
        )

    def _format_step_progress(self, steps: list) -> str:
        """Format individual step progress."""
        if not steps:
            return "No steps defined."

        result = ""
        for step in steps:
            status_icon = {
                "completed": "✅",
                "in_progress": "🔄",
                "pending": "⏳",
                "failed": "❌",
            }.get(step.get("status", "pending"), "❓")

            result += f"{status_icon} **{step['name']}**\n"
            if step.get("completed_at"):
                result += f"   Completed: {step['completed_at']}\n"
            if step.get("notes"):
                result += f"   Notes: {step['notes']}\n"

        return result

    def _format_risks(self, risks: list) -> str:
        """Format risk assessment."""
        if not risks:
            return "No risks identified."

        result = ""
        for risk in risks:
            severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                risk.get("severity", "medium"), "⚪"
            )

            result += f"{severity_emoji} **{risk['name']}** ({risk.get('severity', 'medium')} severity)\n"
            result += f"   {risk.get('description', '')}\n"
            if risk.get("mitigation"):
                result += f"   Mitigation: {risk['mitigation']}\n"

        return result

    def _estimate_completion(self, plan: dict, current_progress: float) -> str:
        """Estimate completion date based on progress."""
        if current_progress >= 1.0:
            return "Migration completed!"

        if current_progress == 0:
            return "Migration not yet started."

        # Simple estimation based on elapsed time and progress

        created = plan.get("created_at")
        if not created:
            return "Unable to estimate completion."

        # This is a simplified estimation
        # In a real implementation, you'd use actual velocity data
        return "Estimated completion: Based on current velocity"
