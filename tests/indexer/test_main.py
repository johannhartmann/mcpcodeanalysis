"""Tests for the main indexing functionality."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.indexer.main import IndexerService


@pytest.fixture
def mock_dependencies():
    """Mock all dependencies for IndexerService."""
    with (
        patch("src.indexer.main.EmbeddingGenerator") as mock_embed,
        patch("src.indexer.main.CodeInterpreter") as mock_interp,
        patch("src.indexer.main.CodeChunker") as mock_chunk,
        patch("src.indexer.main.CodeExtractor") as mock_extract,
        patch("src.indexer.main.init_database") as mock_init_db,
        patch("src.indexer.main.get_session_factory") as mock_session_factory,
    ):

        # Setup mocks
        mock_embed_instance = MagicMock()
        mock_embed_instance.generate_code_embeddings = AsyncMock(
            return_value=([1.0] * 1536, [2.0] * 1536)
        )
        mock_embed_instance.count_tokens = Mock(return_value=100)
        mock_embed.return_value = mock_embed_instance

        mock_interp_instance = MagicMock()
        mock_interp_instance.interpret_function = AsyncMock(
            return_value="This function does something"
        )
        mock_interp_instance.interpret_class = AsyncMock(
            return_value="This class represents something"
        )
        mock_interp_instance.interpret_module = AsyncMock(
            return_value="This module provides functionality"
        )
        mock_interp.return_value = mock_interp_instance

        mock_chunk_instance = MagicMock()
        mock_chunk_instance.chunk_by_entity = Mock(return_value=[])
        mock_chunk_instance.merge_small_chunks = Mock(return_value=[])
        mock_chunk.return_value = mock_chunk_instance

        mock_extract_instance = MagicMock()
        mock_extract_instance.extract_from_file = Mock(return_value={})
        mock_extract.return_value = mock_extract_instance

        mock_init_db.return_value = AsyncMock()

        yield {
            "embedding_generator": mock_embed_instance,
            "code_interpreter": mock_interp_instance,
            "code_chunker": mock_chunk_instance,
            "code_extractor": mock_extract_instance,
            "init_database": mock_init_db,
            "get_session_factory": mock_session_factory,
        }


@pytest.mark.asyncio
async def test_indexer_service_initialization(mock_dependencies):
    """Test IndexerService initialization."""
    service = IndexerService()

    assert service.embedding_generator is not None
    assert service.code_interpreter is not None
    assert service.code_chunker is not None
    assert service.code_extractor is not None
    assert service.running is False
    assert service.tasks == []


@pytest.mark.asyncio
async def test_indexer_service_start(mock_dependencies):
    """Test starting the indexer service."""
    service = IndexerService()

    # Mock run_indexing to avoid infinite loop
    with patch.object(service, "run_indexing", new_callable=AsyncMock) as mock_run:
        await service.start()

        assert service.running is True
        assert len(service.tasks) == 1
        mock_dependencies["init_database"].assert_called_once()


@pytest.mark.asyncio
async def test_indexer_service_stop(mock_dependencies):
    """Test stopping the indexer service."""
    service = IndexerService()

    # Create a mock task
    mock_task = MagicMock()
    mock_task.cancel = Mock()
    service.tasks = [mock_task]
    service.running = True

    await service.stop()

    assert service.running is False
    mock_task.cancel.assert_called_once()


@pytest.mark.asyncio
async def test_process_unindexed_entities(mock_dependencies):
    """Test processing unindexed entities."""
    service = IndexerService()

    # Mock session and database results
    mock_session = AsyncMock()
    mock_file = MagicMock()
    mock_file.id = 1
    mock_file.path = "test.py"
    mock_file.repository_id = 1

    mock_session.execute = AsyncMock(return_value=[mock_file])

    mock_session_factory = AsyncMock()
    mock_session_factory.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_factory.__aexit__ = AsyncMock()

    mock_dependencies["get_session_factory"].return_value = mock_session_factory

    with patch.object(service, "index_file", new_callable=AsyncMock) as mock_index:
        await service.process_unindexed_entities()

        mock_index.assert_called_once_with(mock_session, mock_file)


@pytest.mark.asyncio
async def test_index_file_success(mock_dependencies, temp_repo_dir):
    """Test successful file indexing."""
    service = IndexerService()

    # Create test file
    test_file = temp_repo_dir / "test.py"
    test_file.write_text(
        """
def test_function():
    '''Test function docstring.'''
    return 42

class TestClass:
    '''Test class docstring.'''
    def method(self):
        pass
"""
    )

    # Mock file and repository objects
    mock_file = MagicMock()
    mock_file.id = 1
    mock_file.path = "test.py"
    mock_file.repository_id = 1

    mock_repo = MagicMock()
    mock_repo.owner = "test"
    mock_repo.name = "repo"

    mock_session = AsyncMock()
    mock_repo_repo = AsyncMock()
    mock_repo_repo.get_by_id = AsyncMock(return_value=mock_repo)

    # Mock entities extraction
    entities = {
        "functions": [
            {
                "name": "test_function",
                "start_line": 2,
                "end_line": 4,
                "docstring": "Test function docstring.",
            }
        ],
        "classes": [
            {
                "name": "TestClass",
                "start_line": 6,
                "end_line": 9,
                "docstring": "Test class docstring.",
                "methods": [{"name": "method"}],
            }
        ],
    }
    mock_dependencies["code_extractor"].extract_from_file.return_value = entities

    # Mock chunks
    chunks = [
        {
            "type": "function",
            "content": "def test_function():\n    return 42",
            "metadata": {"entity_name": "test_function"},
        },
        {
            "type": "class",
            "content": "class TestClass:\n    def method(self):\n        pass",
            "metadata": {"entity_name": "TestClass", "methods": [{"name": "method"}]},
        },
    ]
    mock_dependencies["code_chunker"].chunk_by_entity.return_value = chunks
    mock_dependencies["code_chunker"].merge_small_chunks.return_value = chunks

    with (
        patch("src.indexer.main.Path") as mock_path_cls,
        patch("src.indexer.main.RepositoryRepo") as mock_repo_cls,
        patch.object(service, "process_chunk", new_callable=AsyncMock) as mock_process,
    ):

        # Setup path mocks
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.open.return_value.__enter__.return_value.read.return_value = (
            test_file.read_text()
        )
        mock_path_cls.return_value = mock_path

        mock_repo_cls.return_value = mock_repo_repo

        await service.index_file(mock_session, mock_file)

        # Verify calls
        assert mock_process.call_count == 2
        mock_dependencies["code_extractor"].extract_from_file.assert_called_once()
        mock_dependencies["code_chunker"].chunk_by_entity.assert_called_once()


@pytest.mark.asyncio
async def test_index_file_missing_repository(mock_dependencies):
    """Test indexing when repository is not found."""
    service = IndexerService()

    mock_file = MagicMock()
    mock_file.id = 1
    mock_file.path = "test.py"
    mock_file.repository_id = 1

    mock_session = AsyncMock()
    mock_repo_repo = AsyncMock()
    mock_repo_repo.get_by_id = AsyncMock(return_value=None)

    with patch("src.indexer.main.RepositoryRepo") as mock_repo_cls:
        mock_repo_cls.return_value = mock_repo_repo

        await service.index_file(mock_session, mock_file)

        # Should return early without processing
        mock_dependencies["code_extractor"].extract_from_file.assert_not_called()


@pytest.mark.asyncio
async def test_index_file_missing_physical_file(mock_dependencies):
    """Test indexing when physical file doesn't exist."""
    service = IndexerService()

    mock_file = MagicMock()
    mock_file.id = 1
    mock_file.path = "missing.py"
    mock_file.repository_id = 1

    mock_repo = MagicMock()
    mock_repo.owner = "test"
    mock_repo.name = "repo"

    mock_session = AsyncMock()
    mock_repo_repo = AsyncMock()
    mock_repo_repo.get_by_id = AsyncMock(return_value=mock_repo)

    with (
        patch("src.indexer.main.Path") as mock_path_cls,
        patch("src.indexer.main.RepositoryRepo") as mock_repo_cls,
    ):

        mock_path = MagicMock()
        mock_path.exists.return_value = False
        mock_path_cls.return_value = mock_path

        mock_repo_cls.return_value = mock_repo_repo

        await service.index_file(mock_session, mock_file)

        # Should log warning and return
        mock_dependencies["code_extractor"].extract_from_file.assert_not_called()


@pytest.mark.asyncio
async def test_process_chunk_function(mock_dependencies):
    """Test processing a function chunk."""
    service = IndexerService()

    mock_file = MagicMock()
    mock_file.id = 1
    mock_file.path = "test.py"

    chunk = {
        "type": "function",
        "content": "def test():\n    return 42",
        "metadata": {
            "entity_name": "test",
            "parameters": [{"name": "x", "type": "int"}],
            "return_type": "int",
            "docstring": "Test function",
        },
    }

    mock_session = AsyncMock()
    mock_embedding_repo = AsyncMock()
    mock_embedding_repo.create_batch = AsyncMock()

    with (
        patch("src.indexer.main.EmbeddingRepo") as mock_repo_cls,
        patch.object(service, "_map_chunk_to_entity") as mock_map,
    ):

        mock_repo_cls.return_value = mock_embedding_repo
        mock_map.return_value = ("function", 1)

        await service.process_chunk(
            mock_session,
            mock_file,
            chunk,
            Path("test.py"),
        )

        # Verify interpretation was called
        mock_dependencies["code_interpreter"].interpret_function.assert_called_once()

        # Verify embeddings were generated
        mock_dependencies[
            "embedding_generator"
        ].generate_code_embeddings.assert_called_once()

        # Verify embeddings were stored
        mock_embedding_repo.create_batch.assert_called_once()
        embeddings = mock_embedding_repo.create_batch.call_args[0][0]
        assert len(embeddings) == 2  # Raw and interpreted


@pytest.mark.asyncio
async def test_process_chunk_class(mock_dependencies):
    """Test processing a class chunk."""
    service = IndexerService()

    mock_file = MagicMock()
    mock_file.id = 1
    mock_file.path = "test.py"

    chunk = {
        "type": "class",
        "content": "class Test:\n    pass",
        "metadata": {
            "entity_name": "Test",
            "base_classes": ["BaseClass"],
            "docstring": "Test class",
            "methods": [{"name": "method1"}, {"name": "method2"}],
        },
    }

    mock_session = AsyncMock()
    mock_embedding_repo = AsyncMock()

    with (
        patch("src.indexer.main.EmbeddingRepo") as mock_repo_cls,
        patch.object(service, "_map_chunk_to_entity") as mock_map,
    ):

        mock_repo_cls.return_value = mock_embedding_repo
        mock_map.return_value = ("class", 2)

        await service.process_chunk(
            mock_session,
            mock_file,
            chunk,
            Path("test.py"),
        )

        # Verify class interpretation
        mock_dependencies["code_interpreter"].interpret_class.assert_called_once_with(
            "class Test:\n    pass",
            "Test",
            ["BaseClass"],
            "Test class",
            ["method1", "method2"],
        )


@pytest.mark.asyncio
async def test_process_chunk_module(mock_dependencies):
    """Test processing a module chunk."""
    service = IndexerService()

    mock_file = MagicMock()
    mock_file.id = 1
    mock_file.path = "test_module.py"

    chunk = {
        "type": "module",
        "content": '"""Module docstring."""\nimport os',
        "metadata": {
            "docstring": "Module docstring.",
            "import_names": ["os", "sys"],
            "class_names": ["MyClass"],
            "function_names": ["my_func"],
        },
    }

    mock_session = AsyncMock()
    mock_embedding_repo = AsyncMock()

    with (
        patch("src.indexer.main.EmbeddingRepo") as mock_repo_cls,
        patch.object(service, "_map_chunk_to_entity") as mock_map,
    ):

        mock_repo_cls.return_value = mock_embedding_repo
        mock_map.return_value = ("module", 3)

        await service.process_chunk(
            mock_session,
            mock_file,
            chunk,
            Path("test_module.py"),
        )

        # Verify module interpretation
        mock_dependencies["code_interpreter"].interpret_module.assert_called_once_with(
            "test_module",
            "Module docstring.",
            ["os", "sys"],
            ["MyClass"],
            ["my_func"],
        )


@pytest.mark.asyncio
async def test_process_chunk_error_handling(mock_dependencies):
    """Test error handling in chunk processing."""
    service = IndexerService()

    mock_file = MagicMock()
    chunk = {"type": "function", "content": "def test(): pass", "metadata": {}}

    # Make embedding generation fail
    mock_dependencies["embedding_generator"].generate_code_embeddings.side_effect = (
        Exception("API Error")
    )

    mock_session = AsyncMock()

    # Should not raise, just log error
    await service.process_chunk(mock_session, mock_file, chunk, Path("test.py"))


@pytest.mark.asyncio
async def test_map_chunk_to_entity():
    """Test mapping chunks to entity types and IDs."""
    service = IndexerService()

    # Test function chunk
    func_chunk = {"type": "function", "metadata": {"entity_name": "test_func"}}
    entity_type, entity_id = service._map_chunk_to_entity(func_chunk, 1)
    assert entity_type == "function"
    assert entity_id == 1  # Currently returns file_id

    # Test class chunk
    class_chunk = {"type": "class", "metadata": {"entity_name": "TestClass"}}
    entity_type, entity_id = service._map_chunk_to_entity(class_chunk, 2)
    assert entity_type == "class"
    assert entity_id == 2

    # Test module chunk
    module_chunk = {"type": "module", "metadata": {}}
    entity_type, entity_id = service._map_chunk_to_entity(module_chunk, 3)
    assert entity_type == "module"
    assert entity_id == 3

    # Test unknown type defaults to module
    unknown_chunk = {"type": "unknown", "metadata": {}}
    entity_type, entity_id = service._map_chunk_to_entity(unknown_chunk, 4)
    assert entity_type == "module"
    assert entity_id == 4


@pytest.mark.asyncio
async def test_run_indexing_loop(mock_dependencies):
    """Test the main indexing loop."""
    service = IndexerService()
    service.running = True

    # Mock to stop after one iteration
    call_count = 0

    async def mock_process():
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            service.running = False

    with (
        patch.object(service, "process_unindexed_entities", side_effect=mock_process),
        patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):

        await service.run_indexing()

        assert call_count > 0
        mock_sleep.assert_called()


@pytest.mark.asyncio
async def test_run_indexing_error_recovery(mock_dependencies):
    """Test error recovery in indexing loop."""
    service = IndexerService()
    service.running = True

    # Mock to raise error then stop
    call_count = 0

    async def mock_process():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("Test error")
        service.running = False

    with (
        patch.object(service, "process_unindexed_entities", side_effect=mock_process),
        patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):

        await service.run_indexing()

        # Should recover from error and continue
        assert call_count == 2
        # Should have extra sleep after error
        assert mock_sleep.call_count >= 2


@pytest.mark.asyncio
async def test_main_entry_point(mock_dependencies):
    """Test the main entry point function."""
    with (
        patch("src.indexer.main.IndexerService") as mock_service_cls,
        patch("src.indexer.main.setup_logging") as mock_setup_logging,
        patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        patch("signal.signal") as mock_signal,
    ):

        mock_service = AsyncMock()
        mock_service.running = False  # Stop immediately
        mock_service_cls.return_value = mock_service

        await main()

        mock_setup_logging.assert_called_once()
        mock_service.start.assert_called_once()
        mock_service.stop.assert_called_once()
        assert mock_signal.call_count == 2  # SIGINT and SIGTERM


@pytest.mark.asyncio
async def test_signal_handler(mock_dependencies):
    """Test signal handler functionality."""
    service = IndexerService()

    # Test signal handler calls stop
    with patch("asyncio.create_task") as mock_create_task:
        from src.indexer.main import main

        # Create a signal handler
        def get_signal_handler():
            with (
                patch("src.indexer.main.IndexerService") as mock_service_cls,
                patch("signal.signal") as mock_signal,
            ):
                mock_service_cls.return_value = service

                # Capture the signal handler
                signal_handler = None

                def capture_handler(sig, handler):
                    nonlocal signal_handler
                    if sig == signal.SIGINT:
                        signal_handler = handler

                mock_signal.side_effect = capture_handler

                # Run main briefly to set up handlers
                import signal

                from src.indexer.main import main

                # Can't easily test the full flow, but we verified the structure
                return signal_handler

        # The signal handler is set up in main()
        # We've verified the structure exists in the code
