"""Simple tests for parallel processing functionality."""

from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import File
from src.scanner.code_processor import CodeProcessor


@pytest.mark.asyncio
async def test_parallel_processing_enabled(async_session: AsyncSession) -> None:
    """Test that parallel processing can be enabled and disabled."""
    processor = CodeProcessor(
        async_session,
        repository_path="/test",
        enable_parallel=False,
    )

    # Should start disabled
    assert processor._use_parallel is False

    # Enable parallel processing
    processor.enable_parallel_processing(True)
    assert processor._use_parallel is True

    # Disable again
    processor.enable_parallel_processing(False)
    assert processor._use_parallel is False


@pytest.mark.asyncio
async def test_parallel_processing_uses_correct_method(
    async_session: AsyncSession,
) -> None:
    """Test that parallel processing uses the correct method based on settings."""
    # Create mock files
    mock_files = []
    for i in range(10):
        file = MagicMock(spec=File)
        file.id = i + 1
        file.path = f"test_{i}.py"
        mock_files.append(file)

    # Test sequential processing
    processor_seq = CodeProcessor(
        async_session,
        repository_path="/test",
        enable_parallel=False,
    )

    # Mock the sequential method
    from typing import cast

    pseq_any = cast("Any", processor_seq)
    pseq_any._process_files_sequential = AsyncMock(return_value=[])
    pseq_any._process_files_parallel = AsyncMock(return_value=[])

    await processor_seq.process_files(mock_files)

    # Should have called sequential method
    assert pseq_any._process_files_sequential.await_count == 1
    assert pseq_any._process_files_parallel.await_count == 0

    # Test parallel processing
    processor_par = CodeProcessor(
        async_session,
        repository_path="/test",
        enable_parallel=True,
    )

    # Mock the methods
    ppar_any = cast("Any", processor_par)
    ppar_any._process_files_sequential = AsyncMock(return_value=[])
    ppar_any._process_files_parallel = AsyncMock(return_value=[])

    await processor_par.process_files(mock_files)

    # Should have called parallel method
    assert ppar_any._process_files_parallel.await_count == 1
    assert ppar_any._process_files_sequential.await_count == 0


@pytest.mark.asyncio
async def test_parallel_processing_batch_size_calculation(
    async_session: AsyncSession,
) -> None:
    """Test that batch size is calculated correctly."""
    processor = CodeProcessor(
        async_session,
        repository_path="/test",
        enable_parallel=True,
    )

    # Mock necessary components
    with patch("src.database.session_manager.ParallelSessionManager") as mock_pm:
        mock_pm_instance = AsyncMock()

        # Capture the batch_size argument
        captured_batch_size: int | None = None

        async def capture_batch_size(
            items: list[Any],
            func: Callable[[Any, Any], Awaitable[dict[str, Any]]],
            batch_size: int,
        ) -> list[dict[str, Any]]:
            nonlocal captured_batch_size
            captured_batch_size = batch_size
            return [{} for _ in items]

        mock_pm_instance.execute_parallel = capture_batch_size
        mock_pm.return_value = mock_pm_instance

        with patch("sqlalchemy.ext.asyncio.async_sessionmaker"):
            # Test different file counts
            test_cases = [
                (5, 2),  # min(10, max(2, 5//4=1)) = 2
                (10, 2),  # min(10, max(2, 10//4=2)) = 2
                (20, 5),  # min(10, max(2, 20//4=5)) = 5
                (50, 10),  # min(10, max(2, 50//4=12)) = 10
            ]

            for file_count, expected_batch_size in test_cases:
                mock_files = [MagicMock(spec=File, id=i) for i in range(file_count)]
                await processor._process_files_parallel(mock_files)
                print(
                    f"File count: {file_count}, Expected: {expected_batch_size}, Got: {captured_batch_size}"
                )
                assert captured_batch_size == expected_batch_size, (
                    f"Expected batch size {expected_batch_size} for {file_count} files, got {captured_batch_size}"
                )
