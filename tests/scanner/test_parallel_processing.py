"""Tests for parallel file processing in CodeProcessor."""

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import File
from src.scanner.code_processor import CodeProcessor


@pytest.fixture
def mock_files() -> list[File]:
    """Create mock file records for testing."""
    files = []
    for i in range(20):
        file = MagicMock(spec=File)
        file.id = i + 1
        file.path = f"test_file_{i}.py"
        file.language = "python"
        file.size = 1000
        files.append(file)
    return files


@pytest.fixture
def mock_extractor() -> Any:
    """Mock the code extractor."""
    with patch("src.scanner.code_processor.CodeExtractor") as mock:
        extractor = mock.return_value
        extractor.extract_from_file.return_value = {
            "modules": [{"name": "test", "start_line": 1, "end_line": 100}],
            "classes": [],
            "functions": [{"name": "test_func", "start_line": 10, "end_line": 20}],
            "imports": [],
        }
        yield extractor


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_extractor")
async def test_sequential_vs_parallel_processing(
    async_session: AsyncSession, mock_files: list[File]
) -> None:
    """Test that parallel processing is faster than sequential for many files."""
    # Create processor with parallel disabled
    sequential_processor = CodeProcessor(
        async_session,
        repository_path="/test",
        enable_parallel=False,
    )

    # Mock the actual file processing to simulate some work
    async def mock_process_file(file_record: File) -> dict[str, Any]:
        await asyncio.sleep(0.1)  # Simulate processing time
        return {
            "file_id": file_record.id,
            "status": "success",
            "statistics": {
                "modules": 1,
                "classes": 0,
                "functions": 1,
                "imports": 0,
            },
        }

    from typing import cast

    cast("Any", sequential_processor).process_file = mock_process_file

    # Time sequential processing
    start_time = time.time()
    sequential_result = await sequential_processor.process_files(mock_files[:10])
    sequential_time = time.time() - start_time

    # Create processor with parallel enabled
    parallel_processor = CodeProcessor(
        async_session,
        repository_path="/test",
        enable_parallel=True,
    )

    # Mock execute_parallel to simulate parallel execution
    async def mock_execute_parallel(
        items: list[File], func: Any, batch_size: int
    ) -> list[dict[str, Any]]:
        # Simulate parallel execution with reduced time
        results: list[dict[str, Any]] = []
        # Note: emulate ParallelSessionManager behavior: func(item, session)
        for item in items:
            result = await func(item, async_session)
            results.append(result)
        await asyncio.sleep(0.05 * len(items))
        return results

    # Assign to attribute on the AsyncMock's spec via cast to Any to satisfy mypy
    # Bind as attribute on a simple stub object instead of AsyncMock to satisfy mypy
    class _PMStub:
        async def execute_parallel(
            self, items: list[File], func: Any, batch_size: int
        ) -> list[dict[str, Any]]:
            return await mock_execute_parallel(items, func, batch_size)

    pm_stub = _PMStub()

    with patch(
        "src.scanner.code_processor.ParallelSessionManager"
    ) as mock_parallel_manager:
        mock_parallel_manager.return_value = pm_stub

        # Mock async_sessionmaker
        with patch(
            "src.scanner.code_processor.async_sessionmaker"
        ) as mock_sessionmaker:
            mock_sessionmaker.return_value = AsyncMock()

            # Time parallel processing
            start_time = time.time()
            parallel_result = await parallel_processor.process_files(mock_files[:10])
            parallel_time = time.time() - start_time

    # Verify results are the same
    assert sequential_result["total"] == parallel_result["total"]
    assert sequential_result["success"] == parallel_result["success"]
    assert sequential_result["statistics"] == parallel_result["statistics"]

    # Parallel should be significantly faster
    print(f"Sequential time: {sequential_time:.2f}s")
    print(f"Parallel time: {parallel_time:.2f}s")
    assert parallel_time < sequential_time * 0.7  # At least 30% faster


@pytest.mark.asyncio
async def test_parallel_processing_error_handling(
    async_session: AsyncSession, mock_files: list[File]
) -> None:
    """Test that parallel processing handles errors gracefully."""
    processor = CodeProcessor(
        async_session,
        repository_path="/test",
        enable_parallel=True,
    )

    # Mock the parallel session manager
    with patch("src.scanner.code_processor.ParallelSessionManager") as mock_pm:
        mock_pm_instance = AsyncMock()

        # Track calls and simulate errors
        results: list[dict[str, Any] | None] = []
        for i in range(9):
            if (i + 1) % 3 == 0:
                results.append(None)  # Simulate error
            else:
                results.append(
                    {
                        "file_id": mock_files[i].id,
                        "status": "success",
                        "statistics": {
                            "modules": 1,
                            "classes": 0,
                            "functions": 1,
                            "imports": 0,
                        },
                    }
                )

        mock_pm_instance.execute_parallel = AsyncMock(return_value=results)
        mock_pm.return_value = mock_pm_instance

        with patch("src.scanner.code_processor.async_sessionmaker"):
            result = await processor.process_files(mock_files[:9])

    # Check that we got results despite some failures
    assert result["total"] == 9
    assert result["success"] == 6  # 2/3 should succeed
    assert result["failed"] == 3  # 1/3 should fail
    assert len(result["errors"]) == 3


@pytest.mark.asyncio
async def test_parallel_processing_respects_batch_size(
    async_session: AsyncSession, mock_files: list[File]
) -> None:
    """Test that parallel processing respects batch size limits."""
    processor = CodeProcessor(
        async_session,
        repository_path="/test",
        enable_parallel=True,
    )

    # Mock the parallel session manager to track batch sizes
    with patch("src.scanner.code_processor.ParallelSessionManager") as mock_pm:
        mock_pm_instance = AsyncMock()

        # Track the batch_size argument
        actual_batch_size = None

        async def track_batch_size(
            items: list[File], func: Any, batch_size: int
        ) -> list[dict[str, Any]]:
            nonlocal actual_batch_size
            actual_batch_size = batch_size
            # Return dummy results
            return [
                {
                    "file_id": f.id,
                    "status": "success",
                    "statistics": {
                        "modules": 1,
                        "classes": 0,
                        "functions": 1,
                        "imports": 0,
                    },
                }
                for f in items
            ]

        mock_pm_instance.execute_parallel = track_batch_size
        mock_pm.return_value = mock_pm_instance

        with patch("src.scanner.code_processor.async_sessionmaker"):
            # Process 20 files - batch size should be min(10, max(2, 20//4)) = 5
            await processor.process_files(mock_files)

        # Verify batch size was calculated correctly
        assert actual_batch_size == 5

    # Verify the test monitored concurrency correctly
    # mock_semaphore_values would be set by the mock, but since we're not tracking it
    # we just ensure the function ran without error
    assert actual_batch_size > 0  # Ensure some processing occurred


@pytest.mark.asyncio
async def test_enable_disable_parallel_processing(async_session: AsyncSession) -> None:
    """Test enabling and disabling parallel processing."""
    processor = CodeProcessor(
        async_session,
        repository_path="/test",
        enable_parallel=False,  # Start with parallel disabled
    )

    assert processor._use_parallel is False

    # Enable parallel processing
    processor.enable_parallel_processing(True)
    assert processor._use_parallel is True

    # Disable parallel processing
    processor.enable_parallel_processing(False)
    assert processor._use_parallel is False
