"""Session manager for parallel processing with proper connection pooling."""

import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Any, TypeVar

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class ParallelSessionManager:
    """Manages database sessions for parallel processing operations."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        """Initialize with session factory."""
        self.session_factory = session_factory
        self._semaphore = asyncio.Semaphore(10)  # Limit concurrent sessions

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session with proper resource management."""
        async with self._semaphore, self.session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            else:
                await session.commit()

    async def execute_parallel(
        self,
        items: list[T],
        func: Callable[[T, AsyncSession], Any],
        batch_size: int = 5,
    ) -> list[Any]:
        """
        Execute a function in parallel on a list of items.

        Args:
            items: List of items to process
            func: Async function that takes (item, session) and returns result
            batch_size: Number of items to process in parallel

        Returns:
            List of results from processing each item
        """
        results = []

        # Process items in batches to avoid overwhelming the database
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]

            async def process_item(item: T) -> Any:
                async with self.get_session() as session:
                    return await func(item, session)

            # Execute batch in parallel
            batch_results = await asyncio.gather(
                *[process_item(item) for item in batch],
                return_exceptions=True
            )

            # Handle exceptions and collect results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(
                        "Error processing item %d in batch %d: %s",
                        j,
                        i // batch_size,
                        result
                    )
                    results.append(None)
                else:
                    results.append(result)

        return results

    async def execute_parallel_with_context(
        self,
        items: list[T],
        func: Callable[[T, AsyncSession], Any],
        batch_size: int = 5,
        context: dict[str, Any] | None = None,
    ) -> list[Any]:
        """
        Execute a function in parallel with shared context.

        Args:
            items: List of items to process
            func: Async function that takes (item, session) and returns result
            batch_size: Number of items to process in parallel
            context: Shared context dictionary

        Returns:
            List of results from processing each item
        """
        if context is None:
            context = {}

        results = []

        # Process items in batches
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]

            async def process_item_with_context(item: T) -> Any:
                async with self.get_session() as session:
                    # Add session to context
                    item_context = {**context, "session": session}
                    return await func(item, session, item_context)

            # Execute batch in parallel
            batch_results = await asyncio.gather(
                *[process_item_with_context(item) for item in batch],
                return_exceptions=True
            )

            # Handle exceptions and collect results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(
                        "Error processing item %d in batch %d: %s",
                        j,
                        i // batch_size,
                        result
                    )
                    results.append(None)
                else:
                    results.append(result)

        return results

    async def bulk_insert(
        self,
        items: list[Any],
        batch_size: int = 100,
    ) -> None:
        """
        Bulk insert items into the database.

        Args:
            items: List of SQLAlchemy model instances
            batch_size: Number of items to insert per batch
        """
        async with self.get_session() as session:
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                session.add_all(batch)
                await session.flush()  # Flush to get IDs

                logger.debug(
                    "Bulk inserted batch %d-%d (%d items)",
                    i,
                    min(i + batch_size, len(items)),
                    len(batch)
                )

    async def bulk_update(
        self,
        model_class: type,
        updates: list[dict[str, Any]],
        batch_size: int = 100,
    ) -> None:
        """
        Bulk update items in the database.

        Args:
            model_class: SQLAlchemy model class
            updates: List of update dictionaries with 'id' and update values
            batch_size: Number of items to update per batch
        """
        async with self.get_session() as session:
            for i in range(0, len(updates), batch_size):
                batch = updates[i:i + batch_size]

                # Prepare bulk update
                await session.bulk_update_mappings(model_class, batch)

                logger.debug(
                    "Bulk updated batch %d-%d (%d items)",
                    i,
                    min(i + batch_size, len(updates)),
                    len(batch)
                )

    async def process_files_parallel(
        self,
        file_paths: list[str],
        processor_func: Callable[[str, AsyncSession], Any],
        batch_size: int = 5,
    ) -> list[Any]:
        """
        Process multiple files in parallel with database sessions.

        Args:
            file_paths: List of file paths to process
            processor_func: Function that processes a file with a session
            batch_size: Number of files to process in parallel

        Returns:
            List of processing results
        """
        logger.info("Processing %d files in parallel (batch size: %d)", len(file_paths), batch_size)

        return await self.execute_parallel(
            file_paths,
            processor_func,
            batch_size=batch_size
        )

    def set_concurrency_limit(self, limit: int) -> None:
        """Set the maximum number of concurrent sessions."""
        self._semaphore = asyncio.Semaphore(limit)
        logger.info("Set concurrency limit to %d", limit)