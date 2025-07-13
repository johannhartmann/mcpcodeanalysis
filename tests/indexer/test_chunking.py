"""Tests for text chunking strategies."""

from unittest.mock import patch

import pytest

from src.indexer.chunking import MAX_MODULE_DOCSTRING_SEARCH_LINES, CodeChunker


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    with patch("src.indexer.chunking.settings") as mock:
        mock.parser.chunk_size = 100
        mock.embeddings.max_tokens = 8000
        yield mock


@pytest.fixture
def code_chunker(mock_settings):
    """Create CodeChunker instance with mocked settings."""
    return CodeChunker()


def test_code_chunker_initialization(mock_settings):
    """Test CodeChunker initialization."""
    chunker = CodeChunker()
    assert chunker.chunk_size == 100
    assert chunker.max_tokens == 8000


def test_chunk_by_entity_functions(code_chunker):
    """Test chunking by function entities."""
    entities = {
        "functions": [
            {
                "name": "test_func",
                "start_line": 1,
                "end_line": 5,
                "parameters": [{"name": "x", "type": "int"}],
                "return_type": "int",
                "docstring": "Test function",
                "is_async": False,
                "is_generator": False,
            },
            {
                "name": "async_func",
                "start_line": 7,
                "end_line": 10,
                "parameters": [],
                "is_async": True,
            },
        ],
        "classes": [],
        "imports": [],
    }

    file_content = """def test_func(x: int) -> int:
    '''Test function'''
    result = x * 2
    return result

async def async_func():
    '''Async function'''
    await asyncio.sleep(1)
    return True
"""

    chunks = code_chunker.chunk_by_entity(entities, file_content)

    assert len(chunks) == 2

    # Check first function chunk
    chunk1 = chunks[0]
    assert chunk1["type"] == "function"
    assert "def test_func" in chunk1["content"]
    assert chunk1["metadata"]["entity_name"] == "test_func"
    assert chunk1["metadata"]["parameters"] == [{"name": "x", "type": "int"}]
    assert chunk1["metadata"]["return_type"] == "int"
    assert chunk1["metadata"]["is_async"] is False

    # Check second function chunk
    chunk2 = chunks[1]
    assert chunk2["type"] == "function"
    assert "async def async_func" in chunk2["content"]
    assert chunk2["metadata"]["is_async"] is True


def test_chunk_by_entity_classes(code_chunker):
    """Test chunking by class entities."""
    entities = {
        "classes": [
            {
                "name": "TestClass",
                "start_line": 1,
                "end_line": 10,
                "base_classes": ["BaseClass"],
                "docstring": "Test class",
                "methods": [
                    {"name": "method1", "start_line": 5, "end_line": 7},
                    {"name": "method2", "start_line": 8, "end_line": 10},
                ],
                "is_abstract": False,
            },
        ],
        "functions": [],
    }

    file_content = """class TestClass(BaseClass):
    '''Test class'''

    def method1(self):
        '''Method 1'''
        pass

    def method2(self):
        '''Method 2'''
        return 42
"""

    chunks = code_chunker.chunk_by_entity(entities, file_content)

    assert len(chunks) == 1

    chunk = chunks[0]
    assert chunk["type"] == "class"
    assert "class TestClass" in chunk["content"]
    assert chunk["metadata"]["entity_name"] == "TestClass"
    assert chunk["metadata"]["base_classes"] == ["BaseClass"]
    assert chunk["metadata"]["method_count"] == 2
    assert chunk["metadata"]["is_abstract"] is False


def test_chunk_by_entity_large_class(code_chunker):
    """Test chunking large class creates separate method chunks."""
    # Create a large class
    methods = []
    method_lines = []
    for i in range(20):
        methods.append(
            {
                "name": f"method{i}",
                "start_line": i * 6 + 3,
                "end_line": i * 6 + 8,
            }
        )
        method_lines.extend(
            [
                f"    def method{i}(self):",
                f"        '''Method {i}'''",
                "        # Some implementation",
                "        result = self.process()",
                "        return result",
                "",
            ]
        )

    entities = {
        "classes": [
            {
                "name": "LargeClass",
                "start_line": 1,
                "end_line": 130,  # Large class
                "base_classes": [],
                "methods": methods,
            },
        ],
    }

    file_content = "class LargeClass:\n    '''Large class'''\n" + "\n".join(
        method_lines
    )

    chunks = code_chunker.chunk_by_entity(entities, file_content)

    # Should have 1 class chunk + individual method chunks
    assert len(chunks) > 1

    # First chunk should be the entire class
    assert chunks[0]["type"] == "class"
    assert chunks[0]["metadata"]["entity_name"] == "LargeClass"

    # Subsequent chunks should be methods
    method_chunks = [c for c in chunks[1:] if c["type"] == "method"]
    assert len(method_chunks) == len(methods)

    # Check first method chunk
    if method_chunks:
        assert method_chunks[0]["type"] == "method"
        assert method_chunks[0]["metadata"]["parent_class"] == "LargeClass"


def test_chunk_by_entity_module(code_chunker):
    """Test chunking module-level code."""
    entities = {
        "imports": [
            {"module": "os"},
            {"module": "sys"},
        ],
        "classes": [{"name": "MyClass"}],
        "functions": [{"name": "my_func"}],
    }

    file_content = '''"""
Module docstring explaining the purpose.
"""

import os
import sys
from typing import List

# Module constants
DEFAULT_VALUE = 42

def my_func():
    pass

class MyClass:
    pass
'''

    chunks = code_chunker.chunk_by_entity(entities, file_content)

    # Should have a module chunk
    module_chunks = [c for c in chunks if c["type"] == "module"]
    assert len(module_chunks) == 1

    module_chunk = module_chunks[0]
    assert '"""' in module_chunk["content"]
    assert "import os" in module_chunk["content"]
    assert module_chunk["metadata"]["import_count"] == 2
    assert module_chunk["metadata"]["class_count"] == 1
    assert module_chunk["metadata"]["function_count"] == 1


def test_chunk_by_entity_no_module_docstring(code_chunker):
    """Test module chunk when no clear module section."""
    entities = {"imports": [], "classes": [], "functions": []}

    file_content = """def first_function():
    pass

class FirstClass:
    pass
"""

    chunks = code_chunker.chunk_by_entity(entities, file_content)

    # Should not create module chunk if no clear module section
    module_chunks = [c for c in chunks if c["type"] == "module"]
    assert len(module_chunks) == 0


def test_chunk_by_entity_with_context(code_chunker):
    """Test that context lines are included."""
    entities = {
        "functions": [
            {
                "name": "process_data",
                "start_line": 5,
                "end_line": 7,
            },
        ],
    }

    file_content = """# File header
import numpy as np

# Important function
def process_data(data):
    '''Process the data'''
    return np.array(data)

# Another section
def other_func():
    pass
"""

    chunks = code_chunker.chunk_by_entity(entities, file_content)

    chunk = chunks[0]
    # Should include context before (3 lines) and after (1 line)
    assert "import numpy" in chunk["content"]
    assert "# Important function" in chunk["content"]
    assert "# Another section" in chunk["content"]


def test_chunk_by_lines(code_chunker):
    """Test chunking by fixed line count."""
    file_content = "\n".join([f"Line {i}" for i in range(200)])

    chunks = code_chunker.chunk_by_lines(file_content, overlap=20)

    # Check chunks were created
    assert len(chunks) > 1

    # Check first chunk
    first_chunk = chunks[0]
    assert first_chunk["type"] == "lines"
    assert first_chunk["start_line"] == 1
    assert first_chunk["metadata"]["has_overlap"] is False

    # Check second chunk has overlap
    if len(chunks) > 1:
        second_chunk = chunks[1]
        assert second_chunk["type"] == "lines"
        assert second_chunk["metadata"]["has_overlap"] is True

        # Verify overlap exists
        first_lines = first_chunk["content"].split("\n")
        second_lines = second_chunk["content"].split("\n")

        # Last 20 lines of first chunk should match first 20 of second
        assert len(set(first_lines[-20:]) & set(second_lines[:20])) > 0


def test_chunk_by_lines_small_file(code_chunker):
    """Test chunking small file that fits in one chunk."""
    file_content = "\n".join([f"Line {i}" for i in range(50)])

    chunks = code_chunker.chunk_by_lines(file_content)

    assert len(chunks) == 1
    assert chunks[0]["start_line"] == 1
    assert chunks[0]["end_line"] == 50


def test_merge_small_chunks(code_chunker):
    """Test merging small chunks."""
    chunks = [
        {
            "type": "function",
            "content": "def f1(): pass",
            "start_line": 1,
            "end_line": 1,
            "metadata": {"entity_name": "f1"},
        },
        {
            "type": "function",
            "content": "def f2(): return 1",
            "start_line": 3,
            "end_line": 3,
            "metadata": {"entity_name": "f2"},
        },
        {
            "type": "class",
            "content": "class LargeClass:\n"
            + "\n".join(["    def method(): pass"] * 20),
            "start_line": 5,
            "end_line": 30,
            "metadata": {"entity_name": "LargeClass"},
        },
        {
            "type": "method",
            "content": "def small(): pass",
            "start_line": 32,
            "end_line": 32,
            "metadata": {"entity_name": "small"},
        },
    ]

    merged = code_chunker.merge_small_chunks(chunks, min_size=10)

    # Small functions should be merged
    assert len(merged) < len(chunks)

    # Large class should remain separate
    class_chunks = [
        c for c in merged if c["metadata"].get("entity_name") == "LargeClass"
    ]
    assert len(class_chunks) == 1

    # Check merged chunk
    merged_chunks = [c for c in merged if c["type"] == "merged"]
    if merged_chunks:
        merged_chunk = merged_chunks[0]
        assert "f1" in str(merged_chunk["metadata"]["merged_entities"])
        assert "f2" in str(merged_chunk["metadata"]["merged_entities"])


def test_merge_small_chunks_all_large(code_chunker):
    """Test merge when all chunks are large."""
    chunks = [
        {
            "type": "function",
            "content": "def large_func():\n" + "\n".join(["    # code"] * 20),
            "start_line": 1,
            "end_line": 25,
            "metadata": {"entity_name": "large_func"},
        },
        {
            "type": "class",
            "content": "class LargeClass:\n" + "\n".join(["    pass"] * 20),
            "start_line": 30,
            "end_line": 55,
            "metadata": {"entity_name": "LargeClass"},
        },
    ]

    merged = code_chunker.merge_small_chunks(chunks, min_size=10)

    # No merging should occur
    assert len(merged) == len(chunks)
    assert all(c["type"] != "merged" for c in merged)


def test_merge_chunks_function(code_chunker):
    """Test the merge operation between two chunks."""
    chunk1 = {
        "type": "function",
        "content": "def func1():\n    return 1",
        "start_line": 1,
        "end_line": 2,
        "metadata": {"entity_name": "func1"},
    }

    chunk2 = {
        "type": "function",
        "content": "def func2():\n    return 2",
        "start_line": 4,
        "end_line": 5,
        "metadata": {"entity_name": "func2"},
    }

    merged = code_chunker._merge_chunks(chunk1, chunk2)

    assert merged["type"] == "merged"
    assert merged["start_line"] == 1
    assert merged["end_line"] == 5
    assert "func1" in merged["content"]
    assert "func2" in merged["content"]
    assert merged["metadata"]["merged_types"] == ["function", "function"]
    assert merged["metadata"]["merged_entities"] == ["func1", "func2"]


def test_create_module_chunk_edge_cases(code_chunker):
    """Test module chunk creation edge cases."""
    # Test with imports but no docstring
    entities = {"imports": [{"module": "os"}]}
    file_content = """import os
import sys

def main():
    pass
"""

    chunks = code_chunker.chunk_by_entity(entities, file_content)
    module_chunks = [c for c in chunks if c["type"] == "module"]

    if module_chunks:
        assert "import os" in module_chunks[0]["content"]

    # Test with very long module section
    long_imports = "\n".join([f"import module{i}" for i in range(60)])
    file_content = f'"""\nLong module\n"""\n{long_imports}\n\ndef func():\n    pass'

    chunks = code_chunker.chunk_by_entity(entities, file_content)
    module_chunks = [c for c in chunks if c["type"] == "module"]

    if module_chunks:
        # Should stop at MAX_MODULE_DOCSTRING_SEARCH_LINES
        assert module_chunks[0]["end_line"] <= MAX_MODULE_DOCSTRING_SEARCH_LINES


def test_entity_chunk_metadata_completeness(code_chunker):
    """Test that entity chunks have complete metadata."""
    entities = {
        "functions": [
            {
                "name": "test_func",
                "start_line": 1,
                "end_line": 3,
                "parameters": [{"name": "x", "type": "int"}],
                "return_type": "str",
                "docstring": "Test",
                "is_async": True,
                "is_generator": True,
            },
        ],
        "classes": [
            {
                "name": "TestClass",
                "start_line": 5,
                "end_line": 10,
                "base_classes": ["A", "B"],
                "docstring": "Class",
                "methods": [{"name": "m1"}, {"name": "m2"}],
                "is_abstract": True,
            },
        ],
    }

    file_content = """async def test_func(x: int) -> str:
    '''Test'''
    yield str(x)

class TestClass(A, B):
    '''Class'''

    def m1(self):
        pass

    def m2(self):
        pass
"""

    chunks = code_chunker.chunk_by_entity(entities, file_content)

    # Check function metadata
    func_chunk = next(c for c in chunks if c["type"] == "function")
    meta = func_chunk["metadata"]
    assert meta["entity_name"] == "test_func"
    assert meta["has_docstring"] is True
    assert meta["parameters"] == [{"name": "x", "type": "int"}]
    assert meta["return_type"] == "str"
    assert meta["is_async"] is True
    assert meta["is_generator"] is True

    # Check class metadata
    class_chunk = next(c for c in chunks if c["type"] == "class")
    meta = class_chunk["metadata"]
    assert meta["entity_name"] == "TestClass"
    assert meta["has_docstring"] is True
    assert meta["base_classes"] == ["A", "B"]
    assert meta["method_count"] == 2
    assert meta["is_abstract"] is True


def test_chunk_by_entity_empty_file(code_chunker):
    """Test chunking empty file."""
    entities = {"functions": [], "classes": [], "imports": []}
    file_content = ""

    chunks = code_chunker.chunk_by_entity(entities, file_content)

    assert len(chunks) == 0


def test_chunk_by_entity_no_entities(code_chunker):
    """Test chunking file with no recognized entities."""
    entities = {"functions": [], "classes": [], "imports": []}
    file_content = """# Just comments and constants
CONSTANT = 42

# More comments
DATA = [1, 2, 3]
"""

    chunks = code_chunker.chunk_by_entity(entities, file_content)

    # Might create module chunk or no chunks
    assert len(chunks) <= 1


def test_performance_large_file(code_chunker):
    """Test performance with large file."""
    # Create a large file with many entities
    entities = {
        "functions": [
            {
                "name": f"func{i}",
                "start_line": i * 5 + 1,
                "end_line": i * 5 + 4,
            }
            for i in range(100)
        ],
        "classes": [],
    }

    lines = []
    for i in range(100):
        lines.extend(
            [
                f"def func{i}():",
                "    '''Docstring'''",
                "    result = process()",
                "    return result",
                "",
            ]
        )

    file_content = "\n".join(lines)

    import time

    start = time.time()
    chunks = code_chunker.chunk_by_entity(entities, file_content)
    duration = time.time() - start

    assert len(chunks) == 100
    assert duration < 1.0  # Should complete quickly
