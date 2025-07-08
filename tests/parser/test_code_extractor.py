"""Tests for code extractor."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.parser.code_extractor import CodeExtractor


@pytest.fixture
def code_extractor():
    """Create code extractor fixture."""
    return CodeExtractor()


@pytest.fixture
def sample_entities():
    """Sample extracted entities."""
    return {
        "modules": [{
            "name": "test_module",
            "docstring": "Test module docstring",
            "file_id": 1,
            "start_line": 1,
            "end_line": 100,
        }],
        "classes": [{
            "name": "TestClass",
            "docstring": "Test class for unit testing",
            "base_classes": ["BaseClass", "Mixin"],
            "is_abstract": False,
            "decorators": ["dataclass"],
            "start_line": 10,
            "end_line": 50,
        }],
        "functions": [{
            "name": "test_function",
            "parameters": [
                {"name": "arg1", "type": "str", "default": None},
                {"name": "arg2", "type": "int", "default": "10"},
            ],
            "return_type": "Optional[str]",
            "docstring": "Test function that does something",
            "is_async": True,
            "is_generator": False,
            "is_property": False,
            "is_staticmethod": False,
            "is_classmethod": False,
            "class_name": None,
            "start_line": 60,
            "end_line": 70,
        }, {
            "name": "method",
            "parameters": [{"name": "self"}, {"name": "value"}],
            "return_type": None,
            "docstring": "A method of TestClass",
            "is_async": False,
            "is_generator": False,
            "is_property": True,
            "is_staticmethod": False,
            "is_classmethod": False,
            "class_name": "TestClass",
            "start_line": 20,
            "end_line": 25,
        }],
        "imports": [{
            "import_statement": "import os",
            "imported_from": None,
            "imported_names": ["os"],
            "is_relative": False,
            "level": 0,
            "line_number": 3,
        }],
    }


class TestCodeExtractor:
    """Tests for CodeExtractor class."""
    
    def test_init(self, code_extractor):
        """Test code extractor initialization."""
        assert code_extractor.parsers is not None
        assert ".py" in code_extractor.parsers
    
    def test_extract_from_file_success(self, code_extractor, tmp_path, sample_entities):
        """Test successful entity extraction."""
        test_file = tmp_path / "test.py"
        test_file.write_text("# Test file")
        
        with patch.object(
            code_extractor.parsers[".py"],
            "extract_entities",
            return_value=sample_entities
        ):
            result = code_extractor.extract_from_file(test_file, file_id=1)
            
            assert result == sample_entities
    
    def test_extract_from_file_unsupported(self, code_extractor, tmp_path):
        """Test extraction from unsupported file type."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Not a Python file")
        
        result = code_extractor.extract_from_file(test_file, file_id=1)
        assert result is None
    
    def test_extract_from_file_error(self, code_extractor, tmp_path):
        """Test extraction with parser error."""
        test_file = tmp_path / "test.py"
        test_file.write_text("# Test file")
        
        with patch.object(
            code_extractor.parsers[".py"],
            "extract_entities",
            side_effect=Exception("Parser error")
        ):
            result = code_extractor.extract_from_file(test_file, file_id=1)
            assert result is None
    
    def test_get_entity_content(self, code_extractor, tmp_path):
        """Test getting entity content."""
        test_file = tmp_path / "test.py"
        test_content = """def test_function():
    '''Test function.'''
    return 42

class TestClass:
    pass
"""
        test_file.write_text(test_content)
        
        with patch.object(
            code_extractor.parsers[".py"],
            "get_code_chunk"
        ) as mock_get_chunk:
            mock_get_chunk.side_effect = lambda f, start, end, context=0: (
                test_content.split("\n")[start-1:end] if context == 0
                else test_content
            )
            
            # Get raw content
            raw, contextual = code_extractor.get_entity_content(
                test_file, "function", 1, 3, include_context=False
            )
            assert raw == contextual
            
            # Get with context
            raw, contextual = code_extractor.get_entity_content(
                test_file, "function", 1, 3, include_context=True
            )
            assert contextual == test_content
    
    def test_build_entity_description_module(self, code_extractor):
        """Test building module description."""
        module_data = {
            "name": "test_module",
            "docstring": "This is a test module for unit testing purposes.",
        }
        
        description = code_extractor.build_entity_description(
            "module", module_data, Path("test/module.py")
        )
        
        assert "Python module 'test_module'" in description
        assert "test/module.py" in description
        assert "Purpose: This is a test module" in description
    
    def test_build_entity_description_class(self, code_extractor):
        """Test building class description."""
        class_data = {
            "name": "TestClass",
            "base_classes": ["BaseClass", "Mixin"],
            "is_abstract": True,
            "docstring": "Abstract test class.",
        }
        
        description = code_extractor.build_entity_description(
            "class", class_data, Path("test.py")
        )
        
        assert "Class 'TestClass'" in description
        assert "inherits from BaseClass, Mixin" in description
        assert "(abstract)" in description
        assert "Purpose: Abstract test class" in description
    
    def test_build_entity_description_function(self, code_extractor):
        """Test building function description."""
        func_data = {
            "name": "test_function",
            "parameters": [
                {"name": "arg1"},
                {"name": "arg2"},
            ],
            "return_type": "str",
            "is_async": True,
            "is_generator": True,
            "docstring": "Test function that generates values.",
        }
        
        description = code_extractor.build_entity_description(
            "function", func_data, Path("test.py")
        )
        
        assert "Function 'test_function'" in description
        assert "with parameters: arg1, arg2" in description
        assert "returns str" in description
        assert "(async, generator)" in description
        assert "Purpose: Test function" in description
    
    def test_build_entity_description_method(self, code_extractor):
        """Test building method description."""
        method_data = {
            "name": "method",
            "class_name": "TestClass",
            "parameters": [],
            "is_property": True,
            "is_staticmethod": False,
            "docstring": "Property method.",
        }
        
        description = code_extractor.build_entity_description(
            "function", method_data, Path("test.py")
        )
        
        assert "Method 'method'" in description
        assert "(property)" in description
    
    def test_aggregate_class_info(self, code_extractor):
        """Test aggregating class information."""
        class_data = {
            "name": "TestClass",
            "docstring": "Main test class.",
            "base_classes": ["Base"],
        }
        
        methods = [
            {
                "name": f"method{i}",
                "class_name": "TestClass",
                "parameters": [],
                "docstring": f"Method {i}",
            }
            for i in range(15)
        ]
        
        result = code_extractor.aggregate_class_info(class_data, methods)
        
        assert "Class 'TestClass'" in result
        assert "Methods (15):" in result
        assert "- Method 'method0'" in result
        assert "- Method 'method9'" in result
        assert "... and 5 more methods" in result
    
    def test_aggregate_module_info(self, code_extractor):
        """Test aggregating module information."""
        module_data = {
            "name": "test_module",
            "docstring": "Test module.",
        }
        
        classes = [{"name": f"Class{i}", "docstring": f"Class {i}"} for i in range(7)]
        functions = [{"name": f"func{i}", "parameters": []} for i in range(8)]
        
        result = code_extractor.aggregate_module_info(
            module_data, classes, functions, Path("test.py")
        )
        
        assert "Python module 'test_module'" in result
        assert "Classes (7):" in result
        assert "- Class 'Class0'" in result
        assert "... and 2 more classes" in result
        assert "Functions (8):" in result
        assert "- Function 'func0'" in result
        assert "... and 3 more functions" in result