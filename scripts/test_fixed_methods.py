#!/usr/bin/env python3
"""Test the two fixed MCP methods."""

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


async def test_find_usage():
    """Test the fixed find_usage method."""
    print("\n=== Testing find_usage ===")

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

        from src.mcp_server.tools.find import FindTool

        tool = FindTool(session)

        try:
            # Test finding usage of a common class
            results = await tool.find_usage("PackageAnalyzer", None)
            print(f"✅ find_usage succeeded! Found {len(results)} usage locations")

            # Show first few results
            for i, usage in enumerate(results[:3]):
                usage_type = usage.get("type", "unknown")
                file_path = usage.get("location", {}).get("file", "unknown")
                line_num = usage.get("location", {}).get("line", "?")
                context = usage.get("context", "No context")
                print(f"  {i+1}. {usage_type} in {file_path}:{line_num}")
                print(f"     Context: {context}")

            return True

        except Exception as e:
            print(f"❌ find_usage failed: {e!s}")
            logger.exception("find_usage error details")
            return False


async def test_find_similar_code():
    """Test the fixed find_similar_code method."""
    print("\n=== Testing find_similar_code ===")

    engine = await init_database()
    session_factory = get_session_factory(engine)

    async with session_factory() as session:
        from src.query.search_engine import SearchEngine

        search = SearchEngine(session)

        try:
            # Test with a simple code pattern
            results = await search.search_similar_code(
                "async def analyze", limit=5, threshold=0.5
            )

            print(
                f"✅ find_similar_code succeeded! Found {len(results)} similar patterns"
            )

            if not results:
                print(
                    "  Note: No results found. This may be because embeddings haven't been generated yet."
                )
                print("  Run: python scripts/generate_embeddings.py 1")
            else:
                # Show results
                for i, result in enumerate(results):
                    print(
                        f"  {i+1}. {result['name']} ({result['entity_type']}) - similarity: {result['similarity']:.3f}"
                    )
                    print(f"     File: {result['file_path']}")

            return True

        except Exception as e:
            print(f"❌ find_similar_code failed: {e!s}")
            logger.exception("find_similar_code error details")
            return False


async def main():
    """Run tests for fixed methods."""
    print("Testing fixed MCP methods...")

    # Test both methods
    find_usage_ok = await test_find_usage()
    similar_code_ok = await test_find_similar_code()

    print("\n=== Summary ===")
    print(f"find_usage: {'✅ PASSED' if find_usage_ok else '❌ FAILED'}")
    print(f"find_similar_code: {'✅ PASSED' if similar_code_ok else '❌ FAILED'}")

    if find_usage_ok and similar_code_ok:
        print("\n🎉 All fixes are working!")
    else:
        print("\n⚠️  Some issues remain")


if __name__ == "__main__":
    asyncio.run(main())
