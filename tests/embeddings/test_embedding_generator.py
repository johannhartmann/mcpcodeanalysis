"""Tests for embedding generator."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.embeddings.embedding_generator import EmbeddingGenerator


@pytest.fixture
def mock_embeddings():
    """Create mock OpenAIEmbeddings."""
    with patch("src.embeddings.embedding_generator.OpenAIEmbeddings") as mock_class:
        mock_instance = MagicMock()
        mock_instance.aembed_query = AsyncMock(return_value=[0.1] * 1536)
        mock_instance.aembed_documents = AsyncMock(return_value=[[0.1] * 1536])
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def embedding_generator(mock_embeddings):
    """Create embedding generator fixture."""
    with patch("src.embeddings.embedding_generator.get_settings"):
        return EmbeddingGenerator()


@pytest.fixture
def sample_function_data():
    """Sample function data."""
    return {
        "name": "test_function",
        "parameters": [
            {"name": "arg1", "type": "str", "default": None},
            {"name": "arg2", "type": "int", "default": "10"},
        ],
        "return_type": "Optional[str]",
        "docstring": "Test function that does something",
        "decorators": ["lru_cache"],
        "is_async": True,
        "is_generator": False,
        "is_property": False,
        "is_staticmethod": False,
        "is_classmethod": False,
        "class_name": None,
        "start_line": 10,
        "end_line": 20,
    }


@pytest.fixture
def sample_class_data():
    """Sample class data."""
    return {
        "name": "TestClass",
        "docstring": "Test class for unit testing",
        "base_classes": ["BaseClass", "Mixin"],
        "decorators": ["dataclass"],
        "is_abstract": False,
        "start_line": 30,
        "end_line": 80,
        "methods": [
            {"name": "__init__", "parameters": []},
            {"name": "public_method", "parameters": []},
            {"name": "_private_method", "parameters": []},
        ],
    }


@pytest.fixture
def sample_module_data():
    """Sample module data."""
    return {
        "name": "test_module",
        "docstring": "Test module for unit testing",
        "start_line": 1,
        "end_line": 100,
    }


class TestEmbeddingGenerator:
    """Tests for EmbeddingGenerator class."""

    def test_prepare_function_text_simple(
        self,
        embedding_generator,
        sample_function_data,
    ) -> None:
        """Test preparing function text."""
        text = embedding_generator.prepare_function_text(
            sample_function_data,
            "test.py",
        )

        assert "Function: test_function" in text
        assert "File: test.py" in text
        assert (
            "Signature: test_function(arg1: str, arg2: int = 10) -> Optional[str]"
            in text
        )
        assert "Type: async" in text
        assert "Description: Test function that does something" in text
        assert "Decorators: lru_cache" in text

    def test_prepare_function_text_method(
        self,
        embedding_generator,
        sample_function_data,
    ) -> None:
        """Test preparing method text."""
        sample_function_data["class_name"] = "TestClass"
        sample_function_data["is_staticmethod"] = True

        text = embedding_generator.prepare_function_text(
            sample_function_data,
            "test.py",
        )

        assert "Class: TestClass" in text
        assert "static method" in text

    def test_prepare_class_text(
        self,
        embedding_generator,
        sample_class_data,
    ) -> None:
        """Test preparing class text."""
        text = embedding_generator.prepare_class_text(
            sample_class_data,
            "test.py",
        )

        assert "Class: TestClass" in text
        assert "File: test.py" in text
        assert "Inherits from: BaseClass, Mixin" in text
        assert "Description: Test class for unit testing" in text
        assert "Decorators: dataclass" in text
        assert "Public methods: public_method" in text
        assert "_private_method" not in text  # Private methods excluded

    def test_prepare_class_text_abstract(
        self,
        embedding_generator,
        sample_class_data,
    ) -> None:
        """Test preparing abstract class text."""
        sample_class_data["is_abstract"] = True

        text = embedding_generator.prepare_class_text(
            sample_class_data,
            "test.py",
        )

        assert "Type: abstract class" in text

    def test_prepare_module_text(
        self,
        embedding_generator,
        sample_module_data,
    ) -> None:
        """Test preparing module text."""
        summary = {"classes": 5, "functions": 10, "imports": 3}

        text = embedding_generator.prepare_module_text(
            sample_module_data,
            "test.py",
            summary,
        )

        assert "Module: test_module" in text
        assert "File: test.py" in text
        assert "Description: Test module for unit testing" in text
        assert "Contains: 5 classes, 10 functions, 3 imports" in text

    def test_prepare_code_chunk_text(self, embedding_generator) -> None:
        """Test preparing code chunk text."""
        code = """def test_function():
    return 42"""

        text = embedding_generator.prepare_code_chunk_text(
            code,
            "function",
            "test_function",
            "test.py",
            10,
            12,
        )

        assert "Function: test_function" in text
        assert "File: test.py" in text
        assert "Lines: 10-12" in text
        assert code in text

    @pytest.mark.asyncio
    async def test_generate_function_embeddings(
        self,
        embedding_generator,
        mock_embeddings,
        sample_function_data,
    ) -> None:
        """Test generating function embeddings."""
        functions = [sample_function_data]

        mock_embeddings.aembed_documents.return_value = [[0.1] * 1536]

        results = await embedding_generator.generate_function_embeddings(
            functions,
            "test.py",
        )

        assert len(results) == 1
        assert results[0]["embedding"] == [0.1] * 1536

        # Check that embeddings were generated
        mock_embeddings.aembed_documents.assert_called_once()
        # Get the text that was embedded
        call_args = mock_embeddings.aembed_documents.call_args
        texts = call_args[0][0]
        assert len(texts) == 1
        assert "test_function" in texts[0]

    @pytest.mark.asyncio
    async def test_generate_class_embeddings(
        self,
        embedding_generator,
        mock_embeddings,
        sample_class_data,
    ) -> None:
        """Test generating class embeddings."""
        classes = [sample_class_data]

        mock_embeddings.aembed_documents.return_value = [[0.2] * 1536]

        results = await embedding_generator.generate_class_embeddings(
            classes,
            "test.py",
        )

        assert len(results) == 1
        assert results[0]["embedding"] == [0.2] * 1536

        # Check that embeddings were generated
        mock_embeddings.aembed_documents.assert_called_once()
        # Get the text that was embedded
        call_args = mock_embeddings.aembed_documents.call_args
        texts = call_args[0][0]
        assert len(texts) == 1
        assert "TestClass" in texts[0]

    @pytest.mark.asyncio
    async def test_generate_module_embedding(
        self,
        embedding_generator,
        mock_embeddings,
        sample_module_data,
    ) -> None:
        """Test generating module embedding."""
        summary = {"classes": 2, "functions": 5}

        result = await embedding_generator.generate_module_embedding(
            sample_module_data,
            "test.py",
            summary,
        )

        assert result["embedding"] == [0.1] * 1536
        assert result["metadata"]["entity_type"] == "module"
        assert result["metadata"]["entity_name"] == "test_module"

        # Check that summary was included in text
        call_args = mock_embeddings.aembed_query.call_args
        text = call_args[0][0]
        assert "2 classes" in text
        assert "5 functions" in text

    @pytest.mark.asyncio
    async def test_generate_code_chunk_embedding(
        self,
        embedding_generator,
        mock_embeddings,
    ) -> None:
        """Test generating code chunk embedding."""
        code = "def test(): pass"

        result = await embedding_generator.generate_code_chunk_embedding(
            code,
            "function",
            "test",
            "test.py",
            1,
            1,
            {"custom": "metadata"},
        )

        assert result["embedding"] == [0.1] * 1536
        assert result["metadata"]["entity_type"] == "function"
        assert result["metadata"]["entity_name"] == "test"
        assert result["metadata"]["code_lines"] == 1
        assert result["metadata"]["custom"] == "metadata"
