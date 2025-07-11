"""Tests for code processor."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from src.database.models import File
from src.scanner.code_processor import CodeProcessor


@pytest.fixture
def mock_db_session():
    """Create mock database session."""
    session = AsyncMock(spec=AsyncSession)
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.flush = AsyncMock()
    session.execute = AsyncMock()
    return session


@pytest.fixture
def code_processor(mock_db_session):
    """Create code processor fixture."""
    return CodeProcessor(mock_db_session)


@pytest.fixture
def mock_file_record():
    """Create mock file record."""
    file_record = MagicMock(spec=File)
    file_record.id = 1
    file_record.path = "src/test.py"
    file_record.parsed_at = None
    file_record.parse_error = None
    return file_record


@pytest.fixture
def sample_entities():
    """Sample extracted entities."""
    return {
        "modules": [
            {
                "name": "test",
                "docstring": "Test module",
                "start_line": 1,
                "end_line": 100,
            },
        ],
        "classes": [
            {
                "name": "TestClass",
                "docstring": "Test class",
                "base_classes": ["Base"],
                "decorators": [],
                "start_line": 10,
                "end_line": 50,
                "is_abstract": False,
            },
        ],
        "functions": [
            {
                "name": "test_func",
                "parameters": [],
                "return_type": "str",
                "docstring": "Test function",
                "decorators": [],
                "is_async": False,
                "is_generator": False,
                "is_property": False,
                "is_staticmethod": False,
                "is_classmethod": False,
                "start_line": 60,
                "end_line": 65,
                "complexity": 1,
                "class_name": None,
            },
        ],
        "imports": [
            {
                "import_statement": "import os",
                "imported_from": None,
                "imported_names": ["os"],
                "is_relative": False,
                "level": 0,
                "line_number": 3,
            },
        ],
    }


class TestCodeProcessor:
    """Tests for CodeProcessor class."""

    @pytest.mark.asyncio
    async def test_process_file_unsupported(
        self,
        code_processor,
        mock_file_record,
    ) -> None:
        """Test processing unsupported file type."""
        mock_file_record.path = "test.txt"

        with patch.object(
            code_processor.parser_factory,
            "is_supported",
            return_value=False,
        ):
            result = await code_processor.process_file(mock_file_record)

            assert result["status"] == "skipped"
            assert result["reason"] == "unsupported_file_type"

    @pytest.mark.asyncio
    async def test_process_file_success(
        self,
        code_processor,
        mock_file_record,
        sample_entities,
        mock_db_session,
    ) -> None:
        """Test successful file processing."""
        with (
            patch.object(
                code_processor.parser_factory,
                "is_supported",
                return_value=True,
            ),
            patch.object(
                code_processor,
                "_extract_entities",
                return_value=sample_entities,
            ),
            patch.object(
                code_processor,
                "_store_entities",
                return_value={
                    "modules": 1,
                    "classes": 1,
                    "functions": 1,
                    "imports": 1,
                },
            ),
        ):
            result = await code_processor.process_file(mock_file_record)

            assert result["status"] == "success"
            assert result["statistics"]["modules"] == 1
            mock_db_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_process_file_extraction_failed(
        self,
        code_processor,
        mock_file_record,
    ) -> None:
        """Test file processing with extraction failure."""
        with (
            patch.object(
                code_processor.parser_factory,
                "is_supported",
                return_value=True,
            ),
            patch.object(code_processor, "_extract_entities", return_value=None),
        ):
            result = await code_processor.process_file(mock_file_record)

            assert result["status"] == "failed"
            assert result["reason"] == "extraction_failed"

    @pytest.mark.asyncio
    async def test_process_file_error(
        self,
        code_processor,
        mock_file_record,
        mock_db_session,
    ) -> None:
        """Test file processing with error."""
        with (
            patch.object(
                code_processor.parser_factory,
                "is_supported",
                return_value=True,
            ),
            patch.object(
                code_processor,
                "_extract_entities",
                side_effect=Exception("Test error"),
            ),
        ):
            result = await code_processor.process_file(mock_file_record)

            assert result["status"] == "failed"
            assert result["reason"] == "processing_error"
            assert "Test error" in result["error"]
            mock_db_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_extract_entities(self, code_processor, sample_entities) -> None:
        """Test entity extraction."""
        file_path = Path("test.py")

        with patch.object(
            code_processor.code_extractor,
            "extract_from_file",
            return_value=sample_entities,
        ):
            result = await code_processor._extract_entities(file_path, 1)
            assert result == sample_entities

    @pytest.mark.asyncio
    async def test_store_entities(
        self,
        code_processor,
        mock_file_record,
        sample_entities,
        mock_db_session,
    ) -> None:
        """Test storing entities in database."""
        with patch.object(code_processor, "_clear_file_entities"):
            stats = await code_processor._store_entities(
                sample_entities,
                mock_file_record,
            )

            assert stats["modules"] == 1
            assert stats["classes"] == 1
            assert stats["functions"] == 1
            assert stats["imports"] == 1

            # Check that entities were added
            assert mock_db_session.add.call_count >= 4
            mock_db_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_clear_file_entities(self, code_processor, mock_db_session) -> None:
        """Test clearing existing file entities."""
        # Mock the module query
        mock_modules = MagicMock()
        mock_modules.scalars.return_value = [MagicMock(id=1), MagicMock(id=2)]
        mock_db_session.execute.return_value = mock_modules

        await code_processor._clear_file_entities(1)

        # Should execute at least 3 queries (imports, modules query, then conditional deletes)
        assert mock_db_session.execute.call_count >= 3
        mock_db_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_process_files(self, code_processor, mock_file_record) -> None:
        """Test processing multiple files."""
        file_records = [mock_file_record, MagicMock(spec=File)]
        file_records[1].id = 2
        file_records[1].path = "test2.py"

        with patch.object(
            code_processor,
            "process_file",
            side_effect=[
                {"status": "success", "statistics": {"modules": 1, "classes": 2}},
                {"status": "failed", "error": "Test error"},
            ],
        ):
            result = await code_processor.process_files(file_records)

            assert result["total"] == 2
            assert result["success"] == 1
            assert result["failed"] == 1
            assert result["statistics"]["modules"] == 1
            assert result["statistics"]["classes"] == 2
            assert len(result["errors"]) == 1

    @pytest.mark.asyncio
    async def test_get_file_structure(
        self,
        code_processor,
        mock_file_record,
        mock_db_session,
    ) -> None:
        """Test getting file structure."""
        # Mock database queries
        mock_modules = [
            MagicMock(id=1, name="test", docstring="Test", start_line=1, end_line=100),
        ]
        mock_classes = [
            MagicMock(
                id=1,
                name="TestClass",
                base_classes=["Base"],
                is_abstract=False,
                start_line=10,
                end_line=50,
            ),
        ]
        mock_functions = [
            MagicMock(
                id=1,
                name="test_func",
                class_id=None,
                parameters=[],
                return_type=None,
                is_async=False,
                start_line=60,
                end_line=70,
            ),
        ]
        mock_imports = [
            MagicMock(
                id=1,
                import_statement="import os",
                module_name=None,
                imported_names=["os"],
                line_number=3,
            ),
        ]

        mock_db_session.execute.side_effect = [
            MagicMock(scalars=lambda: mock_modules),
            MagicMock(scalars=lambda: mock_classes),
            MagicMock(scalars=lambda: mock_functions),
            MagicMock(scalars=lambda: mock_imports),
        ]

        structure = await code_processor.get_file_structure(mock_file_record)

        assert structure["file"]["id"] == mock_file_record.id
        assert len(structure["modules"]) == 1
        assert len(structure["classes"]) == 1
        assert len(structure["functions"]) == 1
        assert len(structure["imports"]) == 1
