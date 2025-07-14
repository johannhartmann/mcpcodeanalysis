#!/usr/bin/env python3
"""Generate embeddings for a repository."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.database.init_db import get_session_factory, init_database
from src.embeddings.embedding_service import EmbeddingService
from src.logger import get_logger

logger = get_logger(__name__)


async def generate_embeddings(repository_id: int):
    """Generate embeddings for a repository."""
    logger.info("Starting embedding generation for repository %d", repository_id)

    # Initialize database
    engine = await init_database()
    session_factory = get_session_factory(engine)

    async with session_factory() as session:
        # Create embedding service
        embedding_service = EmbeddingService(session)

        # Generate embeddings
        result = await embedding_service.create_repository_embeddings(repository_id)

        logger.info("Embedding generation complete: %s", result)
        return result


if __name__ == "__main__":
    repo_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1  # Default to repository ID 1

    result = asyncio.run(generate_embeddings(repo_id))
    print("\nEmbedding generation results:")
    print(f"Total entities: {result.get('total_entities', 0)}")
    print(f"Embeddings created: {result.get('embeddings_created', 0)}")
    print(f"Embeddings updated: {result.get('embeddings_updated', 0)}")
    print(f"Errors: {result.get('errors', 0)}")
