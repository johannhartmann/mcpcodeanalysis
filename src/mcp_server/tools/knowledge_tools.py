"""MCP tools for migration knowledge management and learning."""

from typing import Any

from pydantic import Field

from src.logger import get_logger

logger = get_logger(__name__)


async def extract_migration_patterns(
    plan_id: int = Field(
        description="ID of completed migration plan to extract patterns from"
    ),
    db_session=None,
) -> dict[str, Any]:
    """Extract reusable patterns from completed migration plans.

    Analyzes successful migration executions to identify:
    - Common steps and sequences
    - Effective strategies
    - Risk mitigation approaches
    - Performance optimization patterns

    Use this to build a knowledge base of proven migration patterns.
    """
    if not db_session:
        from src.database.init_db import get_session_factory

        session_factory = await get_session_factory()
        async with session_factory() as session:
            return await extract_migration_patterns(plan_id, session)

    try:
        from src.services.pattern_library import PatternLibrary

        library = PatternLibrary(db_session)
        patterns = await library.extract_patterns_from_execution(plan_id)

        return {
            "success": True,
            "message": f"Extracted {len(patterns)} patterns from migration plan",
            "data": {
                "plan_id": plan_id,
                "patterns_extracted": len(patterns),
                "patterns": patterns,
            },
        }

    except (AttributeError, KeyError, ValueError, TypeError) as e:
        logger.exception("Failed to extract patterns")
        return {"success": False, "error": str(e)}


async def search_migration_patterns(
    category: str | None = Field(
        description="Pattern category to filter by", default=None
    ),
    context_type: str | None = Field(
        description="Context type to filter by", default=None
    ),
    keywords: str | None = Field(
        description="Keywords to search in name and description", default=None
    ),
    min_success_rate: float = Field(
        description="Minimum success rate (0.0 to 1.0)", default=0.7
    ),
    db_session=None,
) -> dict[str, Any]:
    """Search the pattern library for migration patterns.

    Search by:
    - Category (extraction, refactoring, interface_design, etc.)
    - Context (bounded_context, module, class, function)
    - Keywords in name or description
    - Success rate threshold

    Returns patterns sorted by relevance and success rate.
    """
    if not db_session:
        from src.database.init_db import get_session_factory

        session_factory = await get_session_factory()
        async with session_factory() as session:
            return await search_migration_patterns(
                category, context_type, keywords, min_success_rate, session
            )

    try:
        from src.services.pattern_library import PatternLibrary

        library = PatternLibrary(db_session)
        patterns = await library.search_patterns(
            category=category,
            context_type=context_type,
            keywords=keywords,
            min_success_rate=min_success_rate,
        )

        return {
            "success": True,
            "message": f"Found {len(patterns)} matching patterns",
            "data": {
                "patterns": patterns,
                "search_criteria": {
                    "category": category,
                    "context_type": context_type,
                    "keywords": keywords,
                    "min_success_rate": min_success_rate,
                },
            },
        }

    except (AttributeError, KeyError, ValueError, TypeError) as e:
        logger.exception("Failed to search patterns")
        return {"success": False, "error": str(e)}


async def get_pattern_library_stats(db_session=None) -> dict[str, Any]:
    """Get statistics about the pattern library.

    Returns:
    - Total patterns by category
    - Average success rates
    - Most used patterns
    - Recently added patterns
    - Pattern effectiveness metrics

    Use this to understand the available knowledge base.
    """
    if not db_session:
        from src.database.init_db import get_session_factory

        session_factory = await get_session_factory()
        async with session_factory() as session:
            return await get_pattern_library_stats(session)

    try:
        from src.services.pattern_library import PatternLibrary

        library = PatternLibrary(db_session)
        stats = await library.get_library_statistics()

        return {
            "success": True,
            "message": "Retrieved pattern library statistics",
            "data": stats,
        }

    except (AttributeError, KeyError, ValueError, TypeError) as e:
        logger.exception("Failed to get library stats")
        return {"success": False, "error": str(e)}
