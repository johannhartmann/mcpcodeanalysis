"""Tests for PHP code parser."""

from pathlib import Path

import pytest

from src.parser.php_parser import PHPCodeParser
from src.utils.exceptions import ParserError


@pytest.fixture
def php_parser():
    """Create PHP parser fixture."""
    return PHPCodeParser()


@pytest.fixture
def sample_php_file():
    """Path to sample PHP file."""
    return Path(__file__).parent.parent / "fixtures" / "sample.php"


def test_parse_php_file(php_parser, sample_php_file):
    """Test parsing a complete PHP file."""
    result = php_parser.parse_file(sample_php_file)

    assert result is not None
    assert result["file_name"] == "sample.php"
    assert result["module_name"] == "sample"
    assert "docstring" in result
    assert "imports" in result
    assert "classes" in result
    assert "functions" in result


def test_extract_imports(php_parser, sample_php_file):
    """Test extracting use statements from PHP."""
    result = php_parser.parse_file(sample_php_file)
    imports = result["imports"]

    assert len(imports) > 0

    # Check for specific imports
    import_froms = [imp["imported_from"] for imp in imports]
    assert "DateTime" in import_froms
    assert "Exception" in import_froms
    assert "Sample\\Base\\AbstractClass" in import_froms

    # Check aliased import
    aliased = [
        imp for imp in imports if "Sample\\Utils\\Helper" in imp["imported_from"]
    ]
    assert len(aliased) > 0
    assert aliased[0]["imported_names"] == ["UtilHelper"]


def test_extract_classes(php_parser, sample_php_file):
    """Test extracting class definitions from PHP."""
    result = php_parser.parse_file(sample_php_file)
    classes = result["classes"]

    assert len(classes) >= 2  # BaseClass and SampleClass

    # Find SampleClass
    sample_class = next((c for c in classes if c["name"] == "SampleClass"), None)
    assert sample_class is not None
    # Extends BaseClass and implements SampleInterface
    assert set(sample_class["base_classes"]) == {"BaseClass", "SampleInterface"}
    assert sample_class["docstring"] is not None
    assert "Main sample class" in sample_class["docstring"]

    # Find abstract class
    base_class = next((c for c in classes if c["name"] == "BaseClass"), None)
    assert base_class is not None
    assert base_class["is_abstract"] is True

    # Check traits usage
    sample_class = next((c for c in classes if c["name"] == "SampleClass"), None)
    assert "traits" in sample_class
    assert "LoggerTrait" in sample_class["traits"]


def test_extract_traits(php_parser, sample_php_file):
    """Test extracting trait definitions from PHP."""
    result = php_parser.parse_file(sample_php_file)

    assert "traits" in result
    traits = result["traits"]
    assert len(traits) >= 1

    # Find LoggerTrait
    logger_trait = next((t for t in traits if t["name"] == "LoggerTrait"), None)
    assert logger_trait is not None
    assert logger_trait["docstring"] is not None
    assert "Sample trait" in logger_trait["docstring"]

    # Check trait methods
    assert len(logger_trait["methods"]) >= 1
    log_method = next((m for m in logger_trait["methods"] if m["name"] == "log"), None)
    assert log_method is not None
    assert len(log_method["parameters"]) == 1
    assert log_method["parameters"][0]["name"] == "message"
    assert log_method["return_type"] == "void"


def test_extract_methods(php_parser, sample_php_file):
    """Test extracting methods from PHP classes."""
    result = php_parser.parse_file(sample_php_file)
    classes = result["classes"]

    sample_class = next((c for c in classes if c["name"] == "SampleClass"), None)
    methods = sample_class["methods"]

    assert len(methods) > 0

    # Check constructor
    constructor = next((m for m in methods if m["name"] == "__construct"), None)
    assert constructor is not None
    assert len(constructor["parameters"]) == 2
    assert constructor["parameters"][0]["name"] == "name"
    assert constructor["parameters"][0]["type"] == "string"
    assert constructor["parameters"][1]["name"] == "date"
    assert constructor["parameters"][1]["type"] == "?DateTime"

    # Check static method
    static_method = next((m for m in methods if m["name"] == "getInstanceCount"), None)
    assert static_method is not None
    assert static_method["is_staticmethod"] is True
    assert static_method["return_type"] == "int"

    # Check regular method with docstring
    process_method = next((m for m in methods if m["name"] == "process"), None)
    assert process_method is not None
    assert process_method["docstring"] is not None
    assert "Process data" in process_method["docstring"]


def test_extract_functions(php_parser, sample_php_file):
    """Test extracting standalone functions from PHP."""
    result = php_parser.parse_file(sample_php_file)
    functions = result["functions"]

    # Should have module-level functions
    module_functions = [f for f in functions if "class_name" not in f]
    assert len(module_functions) >= 2

    # Check processItems function
    process_func = next(
        (f for f in module_functions if f["name"] == "processItems"), None
    )
    assert process_func is not None
    assert len(process_func["parameters"]) == 2
    assert process_func["return_type"] == "array"

    # Check function with default parameters
    format_func = next(
        (f for f in module_functions if f["name"] == "formatString"), None
    )
    assert format_func is not None
    assert format_func["parameters"][0]["default"] == '"START"'
    assert format_func["parameters"][1]["default"] == '"END"'


def test_extract_entities(php_parser, sample_php_file, tmp_path):
    """Test extracting all entities for database storage."""
    # Create a temporary file ID
    file_id = 1

    entities = php_parser.extract_entities(sample_php_file, file_id)

    assert "modules" in entities
    assert "classes" in entities
    assert "functions" in entities
    assert "imports" in entities

    # Check module
    assert len(entities["modules"]) == 1
    module = entities["modules"][0]
    assert module["file_id"] == file_id
    assert module["name"] == "sample"

    # Check classes
    assert len(entities["classes"]) >= 2

    # Check functions (including methods)
    assert len(entities["functions"]) > 5

    # Check imports
    assert len(entities["imports"]) >= 4


def test_parse_invalid_file(php_parser, tmp_path):
    """Test parsing a non-existent file."""
    invalid_file = tmp_path / "nonexistent.php"

    with pytest.raises(ParserError):
        php_parser.parse_file(invalid_file)


def test_get_code_chunk(php_parser, sample_php_file):
    """Test getting code chunks from PHP file."""
    # Get a specific chunk (e.g., lines 50-60)
    chunk = php_parser.get_code_chunk(sample_php_file, 50, 60)

    assert chunk is not None
    assert len(chunk) > 0
    assert "class SampleClass" in chunk

    # Test with context
    chunk_with_context = php_parser.get_code_chunk(
        sample_php_file, 55, 57, context_lines=2
    )
    lines = chunk_with_context.split("\n")
    assert len(lines) >= 7  # 3 lines + 2 context before + 2 context after
