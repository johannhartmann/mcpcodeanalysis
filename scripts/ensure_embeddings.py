#!/usr/bin/env python3
"""Script to ensure all repositories have embeddings generated."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import and_, func, select

from src.database.init_db import get_session_factory, init_database
from src.database.models import CodeEmbedding, File, Repository
from src.embeddings.embedding_service import EmbeddingService
from src.logger import get_logger

logger = get_logger(__name__)


async def check_repository_embeddings(session, repo):
    """Check if a repository has embeddings for all files."""
    # Get file count
    file_count_result = await session.execute(
        select(func.count(File.id)).where(
            and_(
                File.repository_id == repo.id,
                File.language == "python",  # Only Python files need embeddings
                ~File.is_deleted,
            )
        )
    )
    file_count = file_count_result.scalar() or 0

    # Get embedding count for this repository
    embedding_count_result = await session.execute(
        select(func.count(func.distinct(CodeEmbedding.file_id)))
        .join(File)
        .where(
            and_(
                File.repository_id == repo.id,
                ~File.is_deleted,
            )
        )
    )
    embedding_count = embedding_count_result.scalar() or 0

    return {
        "repository_id": repo.id,
        "name": repo.name,
        "url": repo.github_url,
        "python_files": file_count,
        "files_with_embeddings": embedding_count,
        "missing_embeddings": file_count - embedding_count,
        "has_all_embeddings": file_count == embedding_count,
    }


async def ensure_all_embeddings():
    """Ensure all repositories have embeddings generated."""
    # Initialize database
    engine = await init_database()
    session_factory = get_session_factory(engine)

    async with session_factory() as session:
        # Get all repositories
        result = await session.execute(select(Repository).order_by(Repository.name))
        repositories = result.scalars().all()

        if not repositories:
            logger.info("No repositories found.")
            return

        logger.info(f"Checking embeddings for {len(repositories)} repositories...")

        # Check each repository
        repositories_needing_embeddings = []

        for repo in repositories:
            stats = await check_repository_embeddings(session, repo)

            logger.info(
                f"{repo.name}: {stats['files_with_embeddings']}/{stats['python_files']} "
                f"Python files have embeddings"
            )

            if not stats["has_all_embeddings"] and stats["python_files"] > 0:
                repositories_needing_embeddings.append(stats)

        # Generate embeddings for repositories that need them
        if repositories_needing_embeddings:
            logger.info(
                f"\n{len(repositories_needing_embeddings)} repositories need embeddings:"
            )

            embedding_service = EmbeddingService(session)

            for repo_stats in repositories_needing_embeddings:
                logger.info(
                    f"\nGenerating embeddings for {repo_stats['name']} "
                    f"({repo_stats['missing_embeddings']} files)..."
                )

                try:
                    result = await embedding_service.create_repository_embeddings(
                        repo_stats["repository_id"]
                    )

                    logger.info(
                        f"✓ Generated embeddings: {result['embeddings_created']} created, "
                        f"{result['embeddings_failed']} failed"
                    )

                    if result["failed_files"]:
                        logger.warning(
                            f"Failed files: {', '.join(result['failed_files'][:5])}"
                            f"{' and more...' if len(result['failed_files']) > 5 else ''}"
                        )

                except Exception as e:
                    logger.error(
                        f"✗ Failed to generate embeddings for {repo_stats['name']}: {e}"
                    )
        else:
            logger.info("\n✓ All repositories have embeddings!")

        # Final summary
        logger.info("\n=== Final Summary ===")
        for repo in repositories:
            stats = await check_repository_embeddings(session, repo)
            status = "✓" if stats["has_all_embeddings"] else "✗"
            logger.info(
                f"{status} {repo.name}: {stats['files_with_embeddings']}/{stats['python_files']} "
                f"Python files have embeddings"
            )


if __name__ == "__main__":
    asyncio.run(ensure_all_embeddings())
