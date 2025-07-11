"""Advanced domain analysis MCP tools."""

from typing import Any

from fastmcp import FastMCP
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.pattern_analyzer import DomainPatternAnalyzer
from src.logger import get_logger

logger = get_logger(__name__)

# Constants for analysis thresholds
MODERATE_COUPLING_THRESHOLD = 3
HIGH_COUPLING_THRESHOLD = 5
CRITICAL_ISSUE_COUNT_THRESHOLD = 5
HIGH_COUPLING_PAIRS_THRESHOLD = 3
MAX_ISSUES_TO_DISPLAY = 10


# Pydantic models for tool parameters
class AnalyzeCouplingRequest(BaseModel):
    """Request to analyze cross-context coupling."""

    repository_id: int | None = Field(
        None,
        description="Optional repository ID to filter analysis",
    )


class SuggestContextSplitsRequest(BaseModel):
    """Request to suggest context splits."""

    min_entities: int = Field(
        default=20,
        description="Minimum entities for a context to be considered",
    )
    max_cohesion_threshold: float = Field(
        default=0.4,
        description="Maximum cohesion score to suggest split",
    )


class DetectAntiPatternsRequest(BaseModel):
    """Request to detect anti-patterns."""

    repository_id: int | None = Field(
        None,
        description="Optional repository ID to filter analysis",
    )


class AnalyzeEvolutionRequest(BaseModel):
    """Request to analyze domain evolution."""

    repository_id: int = Field(..., description="Repository ID to analyze")
    days: int = Field(default=30, description="Number of days to look back")


class AnalysisTools:
    """Advanced domain analysis tools."""

    def __init__(
        self,
        db_session: AsyncSession,
        mcp: FastMCP,
    ) -> None:
        """Initialize analysis tools.

        Args:
            db_session: Database session
            mcp: FastMCP instance
        """
        self.db_session = db_session
        self.mcp = mcp
        self.pattern_analyzer = DomainPatternAnalyzer(db_session)

    async def register_tools(self) -> None:
        """Register all analysis tools."""

        @self.mcp.tool(
            name="analyze_coupling",
            description="Analyze coupling between bounded contexts with metrics and recommendations",
        )
        async def analyze_coupling(
            request: AnalyzeCouplingRequest,
        ) -> dict[str, Any]:
            """Analyze cross-context coupling."""
            try:
                return await self.pattern_analyzer.analyze_cross_context_coupling(
                    request.repository_id,
                )
            except Exception as e:
                logger.exception("Error analyzing coupling")
                return {"error": str(e)}

        @self.mcp.tool(
            name="suggest_context_splits",
            description="Suggest how to split large bounded contexts based on cohesion analysis",
        )
        async def suggest_context_splits(
            request: SuggestContextSplitsRequest,
        ) -> list[dict[str, Any]]:
            """Suggest context splits."""
            try:
                return await self.pattern_analyzer.suggest_context_splits(
                    request.min_entities,
                    request.max_cohesion_threshold,
                )
            except Exception as e:
                logger.exception("Error suggesting splits")
                return [{"error": str(e)}]

        @self.mcp.tool(
            name="detect_anti_patterns",
            description="Detect DDD anti-patterns like anemic models, god objects, and circular dependencies",
        )
        async def detect_anti_patterns(
            request: DetectAntiPatternsRequest,
        ) -> dict[str, list[dict[str, Any]]]:
            """Detect anti-patterns."""
            try:
                return await self.pattern_analyzer.detect_anti_patterns(
                    request.repository_id,
                )
            except Exception as e:
                logger.exception("Error detecting anti-patterns")
                return {"error": [{"message": str(e)}]}

        @self.mcp.tool(
            name="analyze_domain_evolution",
            description="Analyze how the domain model has evolved over time",
        )
        async def analyze_domain_evolution(
            request: AnalyzeEvolutionRequest,
        ) -> dict[str, Any]:
            """Analyze domain evolution."""
            try:
                return await self.pattern_analyzer.analyze_evolution(
                    request.repository_id,
                    request.days,
                )
            except Exception as e:
                logger.exception("Error analyzing evolution")
                return {"error": str(e)}

        @self.mcp.tool(
            name="get_domain_metrics",
            description="Get comprehensive domain health metrics and insights",
        )
        async def get_domain_metrics(
            request: AnalyzeCouplingRequest,  # Reuse for repository_id
        ) -> dict[str, Any]:
            """Get comprehensive domain metrics."""
            try:
                # Combine multiple analyses for a health report
                coupling = await self.pattern_analyzer.analyze_cross_context_coupling(
                    request.repository_id,
                )
                anti_patterns = await self.pattern_analyzer.detect_anti_patterns(
                    request.repository_id,
                )

                # Count issues by severity
                severity_counts = {
                    "high": 0,
                    "medium": 0,
                    "low": 0,
                }

                for issues in anti_patterns.values():
                    if isinstance(issues, list):
                        for issue in issues:
                            severity = issue.get("severity", "medium")
                            severity_counts[severity] += 1

                # Generate insights
                insights = []

                if (
                    coupling["metrics"]["average_coupling"]
                    > MODERATE_COUPLING_THRESHOLD
                ):
                    insights.append(
                        {
                            "type": "high_coupling",
                            "message": "High average coupling between contexts indicates potential architectural issues",
                            "recommendation": "Consider introducing anti-corruption layers or event-driven communication",
                        },
                    )

                if severity_counts["high"] > CRITICAL_ISSUE_COUNT_THRESHOLD:
                    insights.append(
                        {
                            "type": "many_critical_issues",
                            "message": f"Found {severity_counts['high']} high-severity anti-patterns",
                            "recommendation": "Prioritize fixing high-severity issues like missing aggregate roots and circular dependencies",
                        },
                    )

                if len(coupling["high_coupling_pairs"]) > HIGH_COUPLING_PAIRS_THRESHOLD:
                    insights.append(
                        {
                            "type": "chatty_contexts",
                            "message": "Multiple context pairs have high coupling",
                            "recommendation": "Review if these contexts have the right boundaries or should be merged",
                        },
                    )

                return {
                    "metrics": {
                        "average_context_coupling": coupling["metrics"][
                            "average_coupling"
                        ],
                        "max_context_coupling": coupling["metrics"]["max_coupling"],
                        "coupling_distribution": coupling["metrics"][
                            "coupling_distribution"
                        ],
                        "anti_pattern_counts": severity_counts,
                        "total_contexts": len(coupling["contexts"]),
                        "high_coupling_pairs": len(coupling["high_coupling_pairs"]),
                    },
                    "health_score": self._calculate_health_score(
                        coupling,
                        severity_counts,
                    ),
                    "insights": insights,
                    "top_issues": self._get_top_issues(
                        coupling,
                        anti_patterns,
                    ),
                }
            except Exception as e:
                logger.exception("Error getting domain metrics")
                return {"error": str(e)}

    def _calculate_health_score(
        self,
        coupling: dict[str, Any],
        severity_counts: dict[str, int],
    ) -> float:
        """Calculate overall domain health score (0-100)."""
        score = 100.0

        # Deduct for coupling issues
        avg_coupling = coupling["metrics"]["average_coupling"]
        if avg_coupling > HIGH_COUPLING_THRESHOLD:
            score -= 30
        elif avg_coupling > MODERATE_COUPLING_THRESHOLD:
            score -= 20
        elif avg_coupling > 1:
            score -= 10

        # Deduct for anti-patterns
        score -= severity_counts["high"] * 5
        score -= severity_counts["medium"] * 2
        score -= severity_counts["low"] * 0.5

        # Deduct for high coupling pairs
        score -= len(coupling["high_coupling_pairs"]) * 3

        return max(0, min(100, score))

    def _get_top_issues(
        self,
        coupling: dict[str, Any],
        anti_patterns: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Get top issues to address."""
        issues = []

        # Add high coupling pairs
        issues.extend(
            {
                "type": "high_coupling",
                "severity": "high",
                "description": f"{pair['source']} -> {pair['target']} ({pair['relationship_count']} relationships)",
                "recommendation": pair["recommendation"],
            }
            for pair in coupling["high_coupling_pairs"][:3]
        )

        # Add critical anti-patterns
        for pattern_type, pattern_issues in anti_patterns.items():
            if isinstance(pattern_issues, list):
                for issue in pattern_issues:
                    if issue.get("severity") == "high":
                        issues.append(
                            {
                                "type": pattern_type,
                                "severity": "high",
                                "description": issue.get(
                                    "message",
                                    issue.get("issue", ""),
                                ),
                                "recommendation": issue.get("recommendation", ""),
                                "entity": issue.get("entity")
                                or issue.get("context", ""),
                            },
                        )

                    if len(issues) >= MAX_ISSUES_TO_DISPLAY:
                        break

            if len(issues) >= MAX_ISSUES_TO_DISPLAY:
                break

        return issues[:10]
