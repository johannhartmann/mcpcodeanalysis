"""Test parser detection of special method types."""

import tempfile
from pathlib import Path

import pytest

from src.parser.python_parser import PythonCodeParser


@pytest.fixture
def python_parser() -> PythonCodeParser:
    """Create Python parser instance."""
    return PythonCodeParser()


def test_property_detection(python_parser: PythonCodeParser) -> None:
    """Test detection of property methods."""
    code = """
class MyClass:
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = val

    @value.deleter
    def value(self):
        del self._value

    @property
    @functools.lru_cache()
    def cached_value(self):
        return expensive_computation()
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()

        entities = python_parser.extract_entities(Path(f.name), file_id=1)

        # Should have 4 methods all marked as properties
        methods = [f for f in entities["functions"] if f.get("class_name") == "MyClass"]
        assert len(methods) == 4

        for method in methods:
            assert method["is_property"] is True
            assert method["name"] in ["value", "cached_value"]

        # Check specific decorators
        getter = next(m for m in methods if m["decorators"] == ["property"])
        assert getter["name"] == "value"

        setter = next(m for m in methods if "value.setter" in m["decorators"])
        assert setter["name"] == "value"

        deleter = next((m for m in methods if "value.deleter" in m["decorators"]), None)
        assert deleter is not None
        assert deleter["name"] == "value"

        cached = next(m for m in methods if "functools.lru_cache" in m["decorators"])
        assert cached["name"] == "cached_value"
        assert "property" in cached["decorators"]

        Path(f.name).unlink()


def test_static_and_class_methods(python_parser: PythonCodeParser) -> None:
    """Test detection of static and class methods."""
    code = """
class MyClass:
    @staticmethod
    def static_method(x, y):
        return x + y

    @classmethod
    def class_method(cls, value):
        return cls(value)

    def instance_method(self, value):
        return self.process(value)
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()

        entities = python_parser.extract_entities(Path(f.name), file_id=1)

        methods = [f for f in entities["functions"] if f.get("class_name") == "MyClass"]
        assert len(methods) == 3

        static = next(m for m in methods if m["name"] == "static_method")
        assert static["is_staticmethod"] is True
        assert static["is_classmethod"] is False
        assert "staticmethod" in static["decorators"]

        classm = next(m for m in methods if m["name"] == "class_method")
        assert classm["is_classmethod"] is True
        assert classm["is_staticmethod"] is False
        assert "classmethod" in classm["decorators"]

        instance = next(m for m in methods if m["name"] == "instance_method")
        assert instance["is_staticmethod"] is False
        assert instance["is_classmethod"] is False
        assert instance["decorators"] == []

        Path(f.name).unlink()


def test_generator_detection(python_parser: PythonCodeParser) -> None:
    """Test detection of generator functions."""
    code = """
def simple_generator():
    yield 1
    yield 2

def generator_expression():
    return (x*2 for x in range(10))

def yield_from_generator():
    yield from range(5)

async def async_generator():
    for i in range(5):
        yield i

def expression_yield():
    value = yield
    result = yield value * 2
    return result

def not_generator():
    return [x*2 for x in range(10)]

class MyClass:
    def method_generator(self):
        for i in range(3):
            yield i ** 2
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()

        entities = python_parser.extract_entities(Path(f.name), file_id=1)

        # Check module-level functions
        generators = [
            "simple_generator",
            "generator_expression",
            "yield_from_generator",
            "async_generator",
            "expression_yield",
        ]
        non_generators = ["not_generator"]

        for func in entities["functions"]:
            if func["name"] in generators:
                assert func["is_generator"] is True, (
                    f"{func['name']} should be a generator"
                )
            elif func["name"] in non_generators:
                assert func["is_generator"] is False, (
                    f"{func['name']} should not be a generator"
                )

        # Check async generator
        async_gen = next(
            f for f in entities["functions"] if f["name"] == "async_generator"
        )
        assert async_gen["is_async"] is True
        assert async_gen["is_generator"] is True

        # Check method generator
        method_gen = next(
            f for f in entities["functions"] if f["name"] == "method_generator"
        )
        assert method_gen["is_generator"] is True
        assert method_gen["class_name"] == "MyClass"

        Path(f.name).unlink()


def test_async_functions(python_parser: PythonCodeParser) -> None:
    """Test detection of async functions."""
    code = """
async def async_function():
    await some_operation()
    return 42

async def async_generator():
    for i in range(5):
        yield i

class MyClass:
    async def async_method(self):
        return await self.fetch_data()

    @property
    async def async_property(self):  # Note: invalid but parser should detect
        return await self.get_value()
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()

        entities = python_parser.extract_entities(Path(f.name), file_id=1)

        async_funcs = [f for f in entities["functions"] if f["is_async"]]
        assert len(async_funcs) == 4

        # Check specific functions
        async_func = next(f for f in async_funcs if f["name"] == "async_function")
        assert async_func["is_generator"] is False

        async_gen = next(f for f in async_funcs if f["name"] == "async_generator")
        assert async_gen["is_generator"] is True

        async_method = next(f for f in async_funcs if f["name"] == "async_method")
        assert async_method["class_name"] == "MyClass"

        # Invalid async property should still be detected
        async_prop = next(f for f in async_funcs if f["name"] == "async_property")
        assert async_prop["is_property"] is True
        assert async_prop["is_async"] is True

        Path(f.name).unlink()


def test_complex_decorators(python_parser: PythonCodeParser) -> None:
    """Test functions with multiple decorators."""
    code = """
import functools
from typing import cached_property

class MyClass:
    @property
    @functools.lru_cache(maxsize=128)
    def cached_property(self):
        return self.compute()

    @staticmethod
    @functools.wraps(some_function)
    def wrapped_static(x):
        return x * 2

    @classmethod
    @deprecated("Use new_method instead")
    @log_calls
    def old_class_method(cls):
        return cls()
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()

        entities = python_parser.extract_entities(Path(f.name), file_id=1)

        methods = [f for f in entities["functions"] if f.get("class_name") == "MyClass"]

        cached = next(m for m in methods if m["name"] == "cached_property")
        assert cached["is_property"] is True
        assert "property" in cached["decorators"]
        assert "functools.lru_cache" in cached["decorators"]

        static = next(m for m in methods if m["name"] == "wrapped_static")
        assert static["is_staticmethod"] is True
        assert "staticmethod" in static["decorators"]
        assert "functools.wraps" in static["decorators"]

        classm = next(m for m in methods if m["name"] == "old_class_method")
        assert classm["is_classmethod"] is True
        assert "classmethod" in classm["decorators"]
        assert "deprecated" in classm["decorators"]
        assert "log_calls" in classm["decorators"]

        Path(f.name).unlink()
