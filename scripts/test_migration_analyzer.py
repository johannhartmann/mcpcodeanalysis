#!/usr/bin/env python3
"""Test migration analyzer after fixes."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy import select

from src.database.init_db import get_session_factory, init_database
from src.database.models import Repository
from src.logger import get_logger

logger = get_logger(__name__)


async def test_migration_analyzer():
    """Test the migration analyzer."""
    engine = await init_database()
    session_factory = get_session_factory(engine)

    async with session_factory() as session:
        # Get repository
        result = await session.execute(
            select(Repository).where(
                Repository.github_url
                == "https://github.com/johannhartmann/mcpcodeanalysis"
            )
        )
        repo = result.scalar_one_or_none()

        if not repo:
            print("❌ Repository not found")
            return None

        print(f"Testing migration analyzer for repository: {repo.name} (ID: {repo.id})")
        print("=" * 80)

        try:
            from src.services.migration_analyzer import MigrationAnalyzer

            analyzer = MigrationAnalyzer(session)

            # Test analyze_repository_for_migration
            print("\nTesting analyze_repository_for_migration...")
            analysis = await analyzer.analyze_repository_for_migration(repo.id)

            print("✅ Success! Analysis completed")
            print(f"  - Bounded contexts found: {len(analysis['bounded_contexts'])}")
            print(
                f"  - Migration candidates found: {len(analysis['migration_candidates'])}"
            )
            print(
                f"  - Recommended strategy: {analysis['recommended_strategy'].get('approach', 'N/A')}"
            )

            # Show some details
            if analysis["bounded_contexts"]:
                print("\nBounded Contexts:")
                for ctx in analysis["bounded_contexts"][:3]:
                    print(
                        f"  - {ctx['name']} (readiness: {ctx.get('migration_readiness', 0):.2f})"
                    )

            if analysis["migration_candidates"]:
                print("\nTop Migration Candidates:")
                for candidate in analysis["migration_candidates"][:3]:
                    print(
                        f"  - {candidate['name']} (score: {candidate.get('migration_score', 0):.2f})"
                    )

            # Test pattern library as well
            print("\n\nTesting pattern library...")
            from src.services.pattern_library import PatternLibrary

            library = PatternLibrary(session)

            # Search patterns
            patterns = await library.search_patterns(min_success_rate=0.0)
            print(f"✅ Pattern search succeeded! Found {len(patterns)} patterns")

            # Get statistics
            stats = await library.get_library_statistics()
            print("✅ Pattern statistics succeeded!")
            print(f"  - Total patterns: {stats['total_patterns']}")
            print(f"  - Categories: {len(stats['categories'])}")
            if "avg_success_rate" in stats:
                print(f"  - Average success rate: {stats['avg_success_rate']:.2f}")

            return True

        except Exception as e:
            print(f"❌ Error: {e!s}")
            logger.exception("Detailed error")

            # Try to test pattern library separately
            print("\n\nTrying pattern library separately...")
            async with session_factory() as new_session:
                try:
                    from src.services.pattern_library import PatternLibrary

                    library = PatternLibrary(new_session)
                    patterns = await library.search_patterns(min_success_rate=0.0)
                    print(
                        f"✅ Pattern library works independently! Found {len(patterns)} patterns"
                    )
                except (ImportError, AttributeError, RuntimeError, ValueError) as e2:
                    print(f"❌ Pattern library also failed: {e2!s}")

            return False


async def main():
    """Run test."""
    success = await test_migration_analyzer()
    if success:
        print("\n✅ All migration intelligence tools are working!")
    else:
        print("\n❌ Some issues remain")


if __name__ == "__main__":
    asyncio.run(main())
