"""Tests for TreeSitter parser."""

from unittest.mock import MagicMock, patch

import pytest
import tree_sitter

from src.parser.treesitter_parser import PythonParser, TreeSitterParser


@pytest.fixture
def base_parser():
    """Create base TreeSitter parser fixture."""
    return TreeSitterParser()


@pytest.fixture
def python_parser():
    """Create Python TreeSitter parser fixture."""
    return PythonParser()


@pytest.fixture
def sample_python_code() -> bytes:
    """Sample Python code for testing."""
    return b'''"""Module docstring."""

import os
from typing import List

class TestClass:
    """Test class docstring."""

    def __init__(self):
        """Initialize."""
        pass

    @property
    def name(self) -> str:
        """Get name."""
        return "test"

    @staticmethod
    def static_method():
        """Static method."""
        pass

def test_function(arg1: str, arg2: int = 10) -> str:
    """Test function."""
    return arg1 * arg2

async def async_func():
    """Async function."""
    yield 1
    yield 2
'''


class TestTreeSitterParser:
    """Tests for TreeSitterParser base class."""

    def test_init(self, base_parser) -> None:
        """Test parser initialization."""
        assert base_parser.parser is not None
        assert isinstance(base_parser.parser, tree_sitter.Parser)
        assert base_parser.language is None

    def test_parse_content_no_language(self, base_parser) -> None:
        """Test parsing without language set."""
        with pytest.raises(ValueError, match="Language not set"):
            base_parser.parse_content(b"test code")

    def test_get_node_text(self, base_parser) -> None:
        """Test extracting text from node."""
        content = b"hello world"
        mock_node = MagicMock()
        mock_node.start_byte = 0
        mock_node.end_byte = 5

        text = base_parser.get_node_text(mock_node, content)
        assert text == "hello"

    def test_get_node_location(self, base_parser) -> None:
        """Test getting node line numbers."""
        mock_node = MagicMock()
        mock_node.start_point = (10, 0)  # 0-based line number
        mock_node.end_point = (15, 20)

        start_line, end_line = base_parser.get_node_location(mock_node)
        assert start_line == 11  # 1-based
        assert end_line == 16

    def test_find_nodes_by_type(self, base_parser) -> None:
        """Test finding nodes by type."""
        # Create mock tree structure
        root = MagicMock()
        root.type = "module"

        child1 = MagicMock()
        child1.type = "function_definition"
        child1.children = []

        child2 = MagicMock()
        child2.type = "class_definition"
        child2.children = []

        root.children = [child1, child2]

        # Find function definitions
        functions = base_parser.find_nodes_by_type(root, "function_definition")
        assert len(functions) == 1
        assert functions[0] == child1

        # Find class definitions
        classes = base_parser.find_nodes_by_type(root, "class_definition")
        assert len(classes) == 1
        assert classes[0] == child2


class TestPythonParser:
    """Tests for Python-specific TreeSitter parser."""

    def test_init(self, python_parser) -> None:
        """Test Python parser initialization."""
        assert python_parser.language is not None
        assert python_parser.parser is not None

    def test_parse_content(self, python_parser, sample_python_code) -> None:
        """Test parsing Python content."""
        tree = python_parser.parse_content(sample_python_code)
        assert tree is not None
        assert tree.root_node is not None

    def test_extract_imports(self, python_parser, sample_python_code) -> None:
        """Test extracting imports."""
        tree = python_parser.parse_content(sample_python_code)
        imports = python_parser.extract_imports(tree, sample_python_code)

        assert len(imports) >= 2

        # Check import os
        import_os = next((i for i in imports if "import os" in i["import_statement"]), None)
        assert import_os is not None
        assert import_os["imported_names"] == ["os"]
        assert not import_os["is_relative"]

        # Check from import
        from_import = next((i for i in imports if "from typing" in i["import_statement"]), None)
        assert from_import is not None
        assert "List" in from_import["imported_names"]

    def test_extract_functions(self, python_parser, sample_python_code) -> None:
        """Test extracting functions."""
        tree = python_parser.parse_content(sample_python_code)
        functions = python_parser.extract_functions(tree, sample_python_code)

        assert len(functions) >= 2

        # Check test_function
        test_func = next((f for f in functions if f["name"] == "test_function"), None)
        assert test_func is not None
        assert test_func["docstring"] == "Test function."
        assert len(test_func["parameters"]) == 2
        assert test_func["return_type"] == "str"
        assert not test_func["is_async"]

        # Check async function
        async_func = next((f for f in functions if f["name"] == "async_func"), None)
        assert async_func is not None
        assert async_func["is_async"]
        assert async_func["is_generator"]

    def test_extract_classes(self, python_parser, sample_python_code) -> None:
        """Test extracting classes."""
        tree = python_parser.parse_content(sample_python_code)
        classes = python_parser.extract_classes(tree, sample_python_code)

        assert len(classes) == 1

        test_class = classes[0]
        assert test_class["name"] == "TestClass"
        assert test_class["docstring"] == "Test class docstring."
        assert test_class["base_classes"] == []
        assert len(test_class["methods"]) >= 3

        # Check methods
        method_names = [m["name"] for m in test_class["methods"]]
        assert "__init__" in method_names
        assert "name" in method_names
        assert "static_method" in method_names

        # Check property method
        name_method = next(m for m in test_class["methods"] if m["name"] == "name")
        assert name_method["is_property"]
        assert name_method["return_type"] == "str"

        # Check static method
        static_method = next(m for m in test_class["methods"] if m["name"] == "static_method")
        assert static_method["is_staticmethod"]

    def test_extract_module_info(self, python_parser, sample_python_code) -> None:
        """Test extracting complete module info."""
        tree = python_parser.parse_content(sample_python_code)
        module_info = python_parser.extract_module_info(tree, sample_python_code)

        assert module_info["docstring"] == "Module docstring."
        assert len(module_info["imports"]) >= 2
        assert len(module_info["classes"]) == 1
        assert len(module_info["functions"]) >= 2

    def test_extract_parameters(self, python_parser) -> None:
        """Test parameter extraction."""
        # Create mock parameter nodes
        params_node = MagicMock()

        # Simple identifier parameter
        param1 = MagicMock()
        param1.type = "identifier"

        # Typed parameter
        param2 = MagicMock()
        param2.type = "typed_parameter"
        param2_name = MagicMock()
        param2_name.type = "identifier"
        param2_type = MagicMock()
        param2_type.type = "type"
        param2.children = [param2_name, MagicMock(type=":"), param2_type]

        params_node.children = [param1, MagicMock(type=","), param2]

        with patch.object(python_parser, "get_node_text") as mock_get_text:
            mock_get_text.side_effect = lambda n, c: {
                param1: "arg1",
                param2_name: "arg2",
                param2_type: "str",
            }.get(n, "")

            params = python_parser._extract_parameters(params_node, b"")

            assert len(params) == 2
            assert params[0]["name"] == "arg1"
            assert params[0]["type"] is None
            assert params[1]["name"] == "arg2"
            assert params[1]["type"] == "str"

    def test_get_docstring(self, python_parser) -> None:
        """Test docstring extraction."""
        # Create mock node structure
        block_node = MagicMock()
        block_node.type = "block"

        expr_stmt = MagicMock()
        expr_stmt.type = "expression_statement"

        string_node = MagicMock()
        string_node.type = "string"

        expr_stmt.children = [string_node]
        block_node.children = [expr_stmt, MagicMock()]  # Docstring + other statements

        node = MagicMock()
        node.children = [MagicMock(), block_node]  # Header + block

        with patch.object(python_parser, "get_node_text") as mock_get_text:
            mock_get_text.return_value = '"""Test docstring."""'

            docstring = python_parser.get_docstring(node, b"")
            assert docstring == "Test docstring."
