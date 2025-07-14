#!/usr/bin/env python3
"""Simple test for find_usage method."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.database.init_db import get_session_factory, init_database
from src.logger import get_logger
from src.mcp_server.tools.find import FindTool

logger = get_logger(__name__)


async def main():
    """Test find_usage."""
    engine = await init_database()
    session_factory = get_session_factory(engine)

    async with session_factory() as session:
        tool = FindTool(session)

        print("Testing find_usage with 'SearchEngine'...")
        try:
            results = await tool.find_usage("SearchEngine", None)
            print(f"✅ Success! Found {len(results)} usage locations")

            for i, usage in enumerate(results[:5]):
                print(f"\n{i+1}. Usage details:")
                print(f"   Type: {usage.get('type', 'unknown')}")
                print(f"   Location: {usage.get('location', {})}")
                print(f"   Context: {usage.get('context', 'N/A')}")
                print(f"   Statement: {usage.get('statement', 'N/A')}")

        except Exception as e:
            print(f"❌ Failed: {e}")
            logger.exception("Detailed error")


if __name__ == "__main__":
    asyncio.run(main())
