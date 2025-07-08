"""Tests for Python code parser."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.parser.python_parser import PythonCodeParser
from src.utils.exceptions import ParserError


@pytest.fixture
def python_parser():
    """Create Python parser fixture."""
    return PythonCodeParser()


@pytest.fixture
def sample_python_code():
    """Sample Python code for testing."""
    return '''"""Module docstring for testing."""

import os
import sys
from typing import List, Optional
from collections import defaultdict

# Constants
DEBUG = True
VERSION = "1.0.0"


class SampleClass:
    """A sample class for testing."""
    
    def __init__(self, name: str):
        """Initialize the sample class."""
        self.name = name
    
    @property
    def display_name(self) -> str:
        """Get display name."""
        return f"Sample: {self.name}"
    
    @staticmethod
    def static_method(value: int) -> int:
        """A static method."""
        return value * 2
    
    async def async_method(self, items: List[str]) -> None:
        """An async method."""
        for item in items:
            await self.process_item(item)
    
    async def process_item(self, item: str) -> None:
        """Process an item."""
        pass


def sample_function(arg1: str, arg2: int = 10) -> Optional[str]:
    """A sample function with type hints."""
    if arg2 > 0:
        return arg1 * arg2
    return None


async def async_generator(count: int):
    """An async generator function."""
    for i in range(count):
        yield i


class AbstractBase:
    """Abstract base class."""
    
    def abstract_method(self):
        """Must be implemented by subclasses."""
        raise NotImplementedError
'''


@pytest.fixture
def temp_python_file(tmp_path, sample_python_code):
    """Create a temporary Python file."""
    file_path = tmp_path / "test_sample.py"
    file_path.write_text(sample_python_code)
    return file_path


class TestPythonCodeParser:
    """Tests for PythonCodeParser class."""
    
    def test_parse_file_success(self, python_parser, temp_python_file):
        """Test successful file parsing."""
        result = python_parser.parse_file(temp_python_file)
        
        assert result["file_path"] == str(temp_python_file)
        assert result["file_name"] == "test_sample.py"
        assert result["module_name"] == "test_sample"
        assert result["docstring"] == "Module docstring for testing."
        assert "imports" in result
        assert "classes" in result
        assert "functions" in result
    
    def test_parse_file_not_found(self, python_parser):
        """Test parsing non-existent file."""
        with pytest.raises(ParserError) as exc_info:
            python_parser.parse_file(Path("/nonexistent/file.py"))
        
        assert "Failed to parse Python file" in str(exc_info.value)
    
    def test_extract_imports(self, python_parser, temp_python_file):
        """Test import extraction."""
        result = python_parser.parse_file(temp_python_file)
        imports = result["imports"]
        
        assert len(imports) >= 4
        
        # Check import statements
        import_statements = [imp["import_statement"] for imp in imports]
        assert "import os" in import_statements
        assert "import sys" in import_statements
        assert any("from typing import" in stmt for stmt in import_statements)
        assert any("from collections import" in stmt for stmt in import_statements)
    
    def test_extract_classes(self, python_parser, temp_python_file):
        """Test class extraction."""
        result = python_parser.parse_file(temp_python_file)
        classes = result["classes"]
        
        assert len(classes) == 2
        
        # Check SampleClass
        sample_class = next(c for c in classes if c["name"] == "SampleClass")
        assert sample_class["docstring"] == "A sample class for testing."
        assert sample_class["base_classes"] == []
        assert not sample_class["is_abstract"]
        assert len(sample_class["methods"]) == 5  # Including __init__
        
        # Check AbstractBase
        abstract_class = next(c for c in classes if c["name"] == "AbstractBase")
        assert abstract_class["docstring"] == "Abstract base class."
    
    def test_extract_methods(self, python_parser, temp_python_file):
        """Test method extraction."""
        result = python_parser.parse_file(temp_python_file)
        sample_class = next(c for c in result["classes"] if c["name"] == "SampleClass")
        methods = sample_class["methods"]
        
        # Check __init__
        init_method = next(m for m in methods if m["name"] == "__init__")
        assert len(init_method["parameters"]) == 1
        assert init_method["parameters"][0]["name"] == "name"
        assert init_method["parameters"][0]["type"] == "str"
        
        # Check property
        property_method = next(m for m in methods if m["name"] == "display_name")
        assert property_method["is_property"]
        assert property_method["return_type"] == "str"
        
        # Check static method
        static_method = next(m for m in methods if m["name"] == "static_method")
        assert static_method["is_staticmethod"]
        
        # Check async method
        async_method = next(m for m in methods if m["name"] == "async_method")
        assert async_method["is_async"]
    
    def test_extract_functions(self, python_parser, temp_python_file):
        """Test function extraction."""
        result = python_parser.parse_file(temp_python_file)
        functions = result["functions"]
        
        assert len(functions) == 2
        
        # Check sample_function
        sample_func = next(f for f in functions if f["name"] == "sample_function")
        assert sample_func["docstring"] == "A sample function with type hints."
        assert len(sample_func["parameters"]) == 2
        assert sample_func["parameters"][0]["name"] == "arg1"
        assert sample_func["parameters"][0]["type"] == "str"
        assert sample_func["parameters"][1]["default"] == "10"
        assert sample_func["return_type"] == "Optional[str]"
        
        # Check async generator
        async_gen = next(f for f in functions if f["name"] == "async_generator")
        assert async_gen["is_async"]
        assert async_gen["is_generator"]
    
    def test_extract_entities(self, python_parser, temp_python_file):
        """Test entity extraction for database storage."""
        entities = python_parser.extract_entities(temp_python_file, file_id=1)
        
        assert "modules" in entities
        assert "classes" in entities
        assert "functions" in entities
        assert "imports" in entities
        
        # Check module
        assert len(entities["modules"]) == 1
        module = entities["modules"][0]
        assert module["file_id"] == 1
        assert module["name"] == "test_sample"
        
        # Check that all entities have file_id
        for class_data in entities["classes"]:
            assert "name" in class_data
            assert "start_line" in class_data
            assert "end_line" in class_data
        
        for func_data in entities["functions"]:
            assert "name" in func_data
            assert "parameters" in func_data
    
    def test_get_code_chunk(self, python_parser, temp_python_file):
        """Test getting code chunks."""
        # Get a specific function
        chunk = python_parser.get_code_chunk(temp_python_file, 58, 62)
        assert "def sample_function" in chunk
        assert "return arg1 * arg2" in chunk
        
        # Get with context
        chunk_with_context = python_parser.get_code_chunk(
            temp_python_file, 58, 62, context_lines=2
        )
        assert len(chunk_with_context.split("\n")) > len(chunk.split("\n"))
    
    def test_calculate_complexity(self, python_parser):
        """Test complexity calculation."""
        func_info = {
            "start_line": 10,
            "end_line": 50,
        }
        
        complexity = python_parser._calculate_complexity(func_info)
        assert complexity == 4  # (50-10+1) // 10 = 4
        
        # Test small function
        small_func = {
            "start_line": 1,
            "end_line": 5,
        }
        complexity = python_parser._calculate_complexity(small_func)
        assert complexity == 1  # Minimum complexity