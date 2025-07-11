"""Tests for parallel session manager."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.database.session_manager import ParallelSessionManager


class TestParallelSessionManager:
    """Tests for ParallelSessionManager class."""

    @pytest.fixture
    def mock_session_factory(self):
        """Create a mock session factory."""
        session_factory = MagicMock()
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        session_factory.return_value = mock_session
        return session_factory

    @pytest.fixture
    def session_manager(self, mock_session_factory):
        """Create a session manager with mock factory."""
        return ParallelSessionManager(mock_session_factory)

    @pytest.mark.asyncio
    async def test_get_session_context_manager(self, session_manager, mock_session_factory):
        """Test that get_session works as context manager."""
        async with session_manager.get_session() as session:
            assert session is not None
            # Verify session factory was called
            mock_session_factory.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_session_commit_on_success(self, session_manager, mock_session_factory):
        """Test that session commits on successful completion."""
        mock_session = mock_session_factory.return_value.__aenter__.return_value

        async with session_manager.get_session():
            pass

        mock_session.commit.assert_called_once()
        mock_session.rollback.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_session_rollback_on_exception(self, session_manager, mock_session_factory):
        """Test that session rolls back on exception."""
        mock_session = mock_session_factory.return_value.__aenter__.return_value

        with pytest.raises(ValueError, match="Test exception"):
            async with session_manager.get_session():
                raise ValueError("Test exception")

        mock_session.rollback.assert_called_once()
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_parallel_basic(self, session_manager):
        """Test basic parallel execution."""
        items = [1, 2, 3, 4, 5]

        async def test_func(item, session):
            return item * 2

        results = await session_manager.execute_parallel(
            items, test_func, batch_size=2
        )

        assert results == [2, 4, 6, 8, 10]

    @pytest.mark.asyncio
    async def test_execute_parallel_with_exception(self, session_manager):
        """Test parallel execution with exceptions."""
        items = [1, 2, 3, 4, 5]

        async def test_func(item, session):
            if item == 3:
                raise ValueError("Test exception")
            return item * 2

        results = await session_manager.execute_parallel(
            items, test_func, batch_size=2
        )

        # Should have None for the failed item
        assert results == [2, 4, None, 8, 10]

    @pytest.mark.asyncio
    async def test_execute_parallel_batch_processing(self, session_manager):
        """Test that items are processed in batches."""
        items = list(range(10))
        processed_items = []

        async def test_func(item, session):
            processed_items.append(item)
            await asyncio.sleep(0.01)  # Small delay
            return item

        results = await session_manager.execute_parallel(
            items, test_func, batch_size=3
        )

        assert len(results) == 10
        assert set(results) == set(items)
        assert set(processed_items) == set(items)

    @pytest.mark.asyncio
    async def test_process_files_parallel(self, session_manager):
        """Test file processing in parallel."""
        file_paths = ["/path/to/file1.py", "/path/to/file2.py", "/path/to/file3.py"]

        async def process_file(file_path, session):
            return f"processed_{file_path}"

        results = await session_manager.process_files_parallel(
            file_paths, process_file, batch_size=2
        )

        expected = [f"processed_{path}" for path in file_paths]
        assert results == expected

    @pytest.mark.asyncio
    async def test_bulk_insert(self, session_manager, mock_session_factory):
        """Test bulk insert functionality."""
        mock_session = mock_session_factory.return_value.__aenter__.return_value

        items = [MagicMock() for _ in range(10)]

        await session_manager.bulk_insert(items, batch_size=3)

        # Should call add_all and flush for each batch
        assert mock_session.add_all.call_count == 4  # 10 items / 3 batch_size = 4 batches
        assert mock_session.flush.call_count == 4

    @pytest.mark.asyncio
    async def test_bulk_update(self, session_manager, mock_session_factory):
        """Test bulk update functionality."""
        mock_session = mock_session_factory.return_value.__aenter__.return_value

        model_class = MagicMock()
        updates = [{"id": i, "name": f"item_{i}"} for i in range(10)]

        await session_manager.bulk_update(model_class, updates, batch_size=3)

        # Should call bulk_update_mappings for each batch
        assert mock_session.bulk_update_mappings.call_count == 4  # 10 items / 3 batch_size = 4 batches

    def test_set_concurrency_limit(self, session_manager):
        """Test setting concurrency limit."""
        session_manager.set_concurrency_limit(5)
        assert session_manager._semaphore._value == 5

    @pytest.mark.asyncio
    async def test_concurrency_limit_enforced(self, session_manager):
        """Test that concurrency limit is enforced."""
        session_manager.set_concurrency_limit(2)

        active_sessions = []

        async def track_session(item, session):
            active_sessions.append(session)
            await asyncio.sleep(0.1)  # Hold session for a bit
            active_sessions.remove(session)
            return item

        # Start processing 5 items but only 2 should be active at once
        task = asyncio.create_task(
            session_manager.execute_parallel(
                [1, 2, 3, 4, 5], track_session, batch_size=5
            )
        )

        # Give some time for processing to start
        await asyncio.sleep(0.05)

        # Should never have more than 2 active sessions
        assert len(active_sessions) <= 2

        # Wait for completion
        await task

        # All sessions should be closed
        assert len(active_sessions) == 0
