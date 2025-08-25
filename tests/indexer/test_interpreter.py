"""Tests for the code interpreter module."""

from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.prompts import PromptTemplate

from src.indexer.interpreter import CodeInterpreter


@pytest.fixture
def mock_settings() -> Generator[Any, None, None]:
    """Mock settings for testing."""
    with patch("src.indexer.interpreter.settings") as mock:
        mock.llm = {"model": "gpt-4", "temperature": 0.3}
        yield mock


@pytest.fixture
def mock_llm() -> Generator[Any, None, None]:
    """Mock ChatOpenAI LLM."""
    with patch("src.indexer.interpreter.ChatOpenAI") as mock_class:
        mock_instance = AsyncMock()
        mock_class.return_value = mock_instance

        # Mock invoke response
        mock_response = MagicMock()
        mock_response.content = "Generated interpretation"
        mock_instance.ainvoke = AsyncMock(return_value=mock_response)

        yield mock_instance


@pytest.fixture
def code_interpreter(mock_settings: Any, mock_llm: Any) -> CodeInterpreter:
    """Create CodeInterpreter instance with mocked dependencies."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        return CodeInterpreter()


def test_code_interpreter_initialization(mock_settings: Any, mock_llm: Any) -> None:
    """Test CodeInterpreter initialization."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        interpreter = CodeInterpreter()

        assert interpreter.llm is not None
        assert isinstance(interpreter.function_prompt, PromptTemplate)
        assert isinstance(interpreter.class_prompt, PromptTemplate)
        assert isinstance(interpreter.module_prompt, PromptTemplate)


def test_code_interpreter_default_model() -> None:
    """Test CodeInterpreter with default model when settings missing."""
    with (
        patch("src.indexer.interpreter.settings") as mock_settings,
        patch("src.indexer.interpreter.ChatOpenAI") as mock_llm_class,
        patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
    ):
        # Simulate missing llm settings
        mock_settings.llm = {}

        _ = CodeInterpreter()

        # Should use default values
        mock_llm_class.assert_called_once_with(
            model="gpt-4.1",
            temperature=0.3,
            api_key="test-key",
        )


@pytest.mark.asyncio
async def test_interpret_function_success(
    code_interpreter: CodeInterpreter, mock_llm: Any
) -> None:
    """Test successful function interpretation."""
    code = """
def calculate_sum(a: int, b: int) -> int:
    '''Calculate the sum of two numbers.'''
    return a + b
"""

    params = [
        {"name": "a", "type": "int"},
        {"name": "b", "type": "int"},
    ]

    mock_llm.ainvoke.return_value.content = (
        "This function calculates the sum of two integers."
    )

    _ = await code_interpreter.interpret_function(
        code=code,
        name="calculate_sum",
        params=params,
        return_type="int",
        docstring="Calculate the sum of two numbers.",
    )

    assert mock_llm.ainvoke.called

    # Verify prompt was called correctly
    mock_llm.ainvoke.assert_called_once()
    prompt_value = mock_llm.ainvoke.call_args[0][0]

    # Check key elements in prompt
    assert "calculate_sum" in prompt_value
    assert "a: int, b: int" in prompt_value
    assert "int" in prompt_value
    assert "Calculate the sum of two numbers." in prompt_value


@pytest.mark.asyncio
async def test_interpret_function_no_params(
    code_interpreter: CodeInterpreter, mock_llm: Any
) -> None:
    """Test function interpretation with no parameters."""
    code = "def get_timestamp(): return time.time()"

    _ = await code_interpreter.interpret_function(
        code=code,
        name="get_timestamp",
        params=[],
        return_type=None,
        docstring=None,
    )

    # Verify prompt formatting for edge cases
    prompt_value = mock_llm.ainvoke.call_args[0][0]
    assert "Parameters: None" in prompt_value
    assert "Return Type: None" in prompt_value
    assert "Docstring: No docstring provided" in prompt_value


@pytest.mark.asyncio
async def test_interpret_function_error_handling(
    code_interpreter: CodeInterpreter, mock_llm: Any
) -> None:
    """Test function interpretation error handling."""
    mock_llm.ainvoke.side_effect = Exception("LLM Error")

    _ = await code_interpreter.interpret_function(
        code="def test(): pass",
        name="test",
        params=[{"name": "x"}, {"name": "y"}],
    )

    # Should return fallback description
    assert isinstance(_, str)


@pytest.mark.asyncio
async def test_interpret_class_success(
    code_interpreter: CodeInterpreter, mock_llm: Any
) -> None:
    """Test successful class interpretation."""
    code = """
class Calculator:
    '''A simple calculator class.'''

    def __init__(self):
        self.result = 0

    def add(self, x, y):
        return x + y

    def multiply(self, x, y):
        return x * y
"""

    mock_llm.ainvoke.return_value.content = (
        "A calculator class providing basic arithmetic operations."
    )

    _ = await code_interpreter.interpret_class(
        code=code,
        name="Calculator",
        base_classes=[],
        docstring="A simple calculator class.",
        methods=["__init__", "add", "multiply"],
    )

    assert isinstance(_, str)

    # Verify prompt elements
    prompt_value = mock_llm.ainvoke.call_args[0][0]
    assert "Calculator" in prompt_value
    assert "A simple calculator class." in prompt_value
    assert "__init__, add, multiply" in prompt_value


@pytest.mark.asyncio
async def test_interpret_class_with_inheritance(
    code_interpreter: CodeInterpreter, mock_llm: Any
) -> None:
    """Test class interpretation with base classes."""
    code = "class MyList(list, UserList): pass"

    _ = await code_interpreter.interpret_class(
        code=code,
        name="MyList",
        base_classes=["list", "UserList"],
        docstring=None,
        methods=[],
    )

    prompt_value = mock_llm.ainvoke.call_args[0][0]
    assert "Base Classes: list, UserList" in prompt_value
    assert "Methods: None" in prompt_value


@pytest.mark.asyncio
async def test_interpret_class_truncated_code(
    code_interpreter: CodeInterpreter, mock_llm: Any
) -> None:
    """Test class interpretation with very long code."""
    # Create code longer than 3000 characters
    long_code = "class LongClass:\n" + "\n".join(
        [f"    def method{i}(self): pass" for i in range(200)]
    )

    _ = await code_interpreter.interpret_class(
        code=long_code,
        name="LongClass",
        base_classes=[],
        methods=[f"method{i}" for i in range(50)],
    )

    # Verify code was truncated
    prompt_value = mock_llm.ainvoke.call_args[0][0]
    code_section = prompt_value.split("```python")[1].split("```")[0]
    assert len(code_section) <= 3000


@pytest.mark.asyncio
async def test_interpret_class_error_handling(
    code_interpreter: CodeInterpreter, mock_llm: Any
) -> None:
    """Test class interpretation error handling."""
    mock_llm.ainvoke.side_effect = Exception("LLM Error")

    _ = await code_interpreter.interpret_class(
        code="class Test: pass",
        name="Test",
        base_classes=[],
        methods=["method1", "method2", "method3"],
    )

    assert isinstance(_, str)


@pytest.mark.asyncio
async def test_interpret_module_success(
    code_interpreter: CodeInterpreter, mock_llm: Any
) -> None:
    """Test successful module interpretation."""
    mock_llm.ainvoke.return_value.content = "A utility module for file operations."

    _ = await code_interpreter.interpret_module(
        name="file_utils",
        docstring="File utility functions.",
        imports=["os", "pathlib", "shutil"],
        classes=["FileHandler", "DirectoryWalker"],
        functions=["read_file", "write_file", "copy_file"],
    )

    assert isinstance(_, str)

    # Verify prompt elements
    prompt_value = mock_llm.ainvoke.call_args[0][0]
    assert "file_utils" in prompt_value
    assert "File utility functions." in prompt_value
    assert "os, pathlib, shutil" in prompt_value
    assert "FileHandler, DirectoryWalker" in prompt_value
    assert "read_file, write_file, copy_file" in prompt_value


@pytest.mark.asyncio
async def test_interpret_module_no_content(
    code_interpreter: CodeInterpreter, mock_llm: Any
) -> None:
    """Test module interpretation with minimal content."""
    _ = await code_interpreter.interpret_module(
        name="empty_module",
        docstring=None,
        imports=None,
        classes=None,
        functions=None,
    )

    prompt_value = mock_llm.ainvoke.call_args[0][0]
    assert "Module Name: empty_module" in prompt_value
    assert "Docstring: No module docstring" in prompt_value
    assert "Key Imports: None" in prompt_value
    assert "Classes: None" in prompt_value
    assert "Functions: None" in prompt_value


@pytest.mark.asyncio
async def test_interpret_module_truncated_lists(
    code_interpreter: CodeInterpreter, mock_llm: Any
) -> None:
    """Test module interpretation with long lists truncated to 10 items."""
    imports = [f"import{i}" for i in range(20)]
    classes = [f"Class{i}" for i in range(20)]
    functions = [f"func{i}" for i in range(20)]

    _ = await code_interpreter.interpret_module(
        name="large_module",
        imports=imports,
        classes=classes,
        functions=functions,
    )

    prompt_value = mock_llm.ainvoke.call_args[0][0]

    # Should only include first 10 of each
    for i in range(10):
        assert f"import{i}" in prompt_value
        assert f"Class{i}" in prompt_value
        assert f"func{i}" in prompt_value

    # Should not include beyond 10
    assert "import10" not in prompt_value
    assert "Class10" not in prompt_value
    assert "func10" not in prompt_value


@pytest.mark.asyncio
async def test_interpret_module_error_handling(
    code_interpreter: CodeInterpreter, mock_llm: Any
) -> None:
    """Test module interpretation error handling."""
    mock_llm.ainvoke.side_effect = Exception("LLM Error")

    _ = await code_interpreter.interpret_module(
        name="test_module",
        classes=["A", "B"],
        functions=["f1", "f2", "f3"],
    )

    assert isinstance(_, str)


@pytest.mark.asyncio
async def test_batch_interpret_functions(
    code_interpreter: CodeInterpreter, mock_llm: Any
) -> None:
    """Test batch interpretation of functions."""
    entities: list[dict[str, Any]] = [
        {
            "code": "def func1(): pass",
            "name": "func1",
            "parameters": [],
            "return_type": None,
            "docstring": "First function",
        },
        {
            "code": "def func2(x): return x",
            "name": "func2",
            "parameters": [{"name": "x"}],
            "return_type": "Any",
            "docstring": None,
        },
    ]

    mock_llm.ainvoke.side_effect = [
        MagicMock(content="Interpretation 1"),
        MagicMock(content="Interpretation 2"),
    ]

    results = await code_interpreter.batch_interpret(entities, "function")

    assert len(results) == 2
    assert results[0] == "Interpretation 1"
    assert results[1] == "Interpretation 2"
    assert mock_llm.ainvoke.call_count == 2


@pytest.mark.asyncio
async def test_batch_interpret_classes(
    code_interpreter: CodeInterpreter, mock_llm: Any
) -> None:
    """Test batch interpretation of classes."""
    entities: list[dict[str, Any]] = [
        {
            "code": "class A: pass",
            "name": "A",
            "base_classes": [],
            "docstring": "Class A",
            "method_names": [],
        },
        {
            "code": "class B(A): pass",
            "name": "B",
            "base_classes": ["A"],
            "docstring": None,
            "method_names": ["method1"],
        },
    ]

    mock_llm.ainvoke.side_effect = [
        MagicMock(content="Class A interpretation"),
        MagicMock(content="Class B interpretation"),
    ]

    results = await code_interpreter.batch_interpret(entities, "class")

    assert len(results) == 2
    assert results[0] == "Class A interpretation"
    assert results[1] == "Class B interpretation"


@pytest.mark.asyncio
async def test_batch_interpret_modules(
    code_interpreter: CodeInterpreter, mock_llm: Any
) -> None:
    """Test batch interpretation of modules."""
    entities: list[dict[str, Any]] = [
        {
            "name": "module1",
            "docstring": "Module 1",
            "import_names": ["os"],
            "class_names": ["A"],
            "function_names": ["f"],
        },
    ]

    mock_llm.ainvoke.return_value.content = "Module interpretation"

    results = await code_interpreter.batch_interpret(entities, "module")

    assert len(results) == 1
    assert isinstance(results[0], str)


@pytest.mark.asyncio
async def test_batch_interpret_unknown_type(
    code_interpreter: CodeInterpreter, mock_llm: Any
) -> None:
    """Test batch interpretation with unknown entity type."""
    entities = [
        {"name": "unknown_entity"},
    ]

    results = await code_interpreter.batch_interpret(entities, "unknown_type")

    assert len(results) == 1
    assert results[0] == "A unknown_type named unknown_entity"

    # Should not call LLM for unknown types
    mock_llm.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_batch_interpret_mixed_success_failure(
    code_interpreter: CodeInterpreter, mock_llm: Any
) -> None:
    """Test batch interpretation with mixed success and failure."""
    entities: list[dict[str, Any]] = [
        {
            "code": "def func1(): pass",
            "name": "func1",
            "parameters": [],
        },
        {
            "code": "def func2(): pass",
            "name": "func2",
            "parameters": [],
        },
    ]

    # First succeeds, second fails
    mock_llm.ainvoke.side_effect = [
        MagicMock(content="Success"),
        Exception("Failed"),
    ]

    results = await code_interpreter.batch_interpret(entities, "function")

    assert len(results) == 2
    assert results[0] == "Success"
    assert results[1] == "Function func2 that takes 0 parameters"


@pytest.mark.asyncio
async def test_interpret_with_response_without_content_attr(
    code_interpreter: CodeInterpreter, mock_llm: Any
) -> None:
    """Test handling responses without content attribute."""
    # Mock response without content attribute
    mock_llm.ainvoke.return_value = "Plain string response"

    _ = await code_interpreter.interpret_function(
        code="def test(): pass",
        name="test",
        params=[],
    )

    assert isinstance(_, str)


@pytest.mark.asyncio
async def test_prompt_template_variables() -> None:
    """Test that prompt templates have correct variables."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        _ = CodeInterpreter()

        # Check function prompt
        assert set(CodeInterpreter().function_prompt.input_variables) == {
            "code",
            "name",
            "params",
            "return_type",
            "docstring",
        }

        # Check class prompt
        assert set(CodeInterpreter().class_prompt.input_variables) == {
            "code",
            "name",
            "base_classes",
            "docstring",
            "methods",
        }

        # Check module prompt
        assert set(CodeInterpreter().module_prompt.input_variables) == {
            "name",
            "docstring",
            "imports",
            "classes",
            "functions",
        }
