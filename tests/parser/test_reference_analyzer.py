"""Tests for reference analyzer."""

import ast
from pathlib import Path

import pytest

from src.parser.reference_analyzer import ReferenceAnalyzer, ReferenceType


@pytest.fixture
def sample_code():
    """Sample Python code for testing."""
    return '''
import os
from pathlib import Path
from typing import List, Dict

class MyClass(BaseClass):
    """Test class."""

    def __init__(self, name: str):
        self.name = name
        self.path = Path(name)

    @property
    def full_path(self) -> Path:
        return self.path / "data"

    def process(self, items: List[str]) -> Dict[str, int]:
        result = {}
        for item in items:
            result[item] = len(item)
        return result

def main():
    obj = MyClass("test")
    data = obj.process(["a", "bb", "ccc"])
    print(data)

if __name__ == "__main__":
    main()
'''


def test_reference_analyzer_basic(sample_code):
    """Test basic reference analysis."""
    tree = ast.parse(sample_code)
    analyzer = ReferenceAnalyzer("test_module", Path("test.py"))
    references = analyzer.analyze(tree)

    # Check that we found references
    assert len(references) > 0

    # Group by type
    by_type = {}
    for ref in references:
        ref_type = ref["reference_type"]
        if ref_type not in by_type:
            by_type[ref_type] = []
        by_type[ref_type].append(ref)

    # Check imports
    assert ReferenceType.IMPORT in by_type
    import_refs = by_type[ReferenceType.IMPORT]
    assert len(import_refs) >= 4  # os, Path, List, Dict

    # Check inheritance
    assert ReferenceType.CLASS_INHERIT in by_type
    inherit_refs = by_type[ReferenceType.CLASS_INHERIT]
    assert len(inherit_refs) == 1
    assert inherit_refs[0]["target_name"].endswith("BaseClass")

    # Check type hints
    assert ReferenceType.TYPE_HINT in by_type
    type_refs = by_type[ReferenceType.TYPE_HINT]
    assert len(type_refs) >= 3  # str, Path, List, Dict

    # Check calls
    assert ReferenceType.FUNCTION_CALL in by_type
    call_refs = by_type[ReferenceType.FUNCTION_CALL]
    assert any(ref["target_name"].endswith("print") for ref in call_refs)
    assert any(ref["target_name"].endswith("len") for ref in call_refs)


def test_reference_analyzer_imports():
    """Test import reference extraction."""
    code = """
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from . import local_module
from ..parent import parent_module
"""
    tree = ast.parse(code)
    analyzer = ReferenceAnalyzer("test_module", Path("test.py"))
    references = analyzer.analyze(tree)

    # Filter import references
    imports = [
        ref for ref in references if ref["reference_type"] == ReferenceType.IMPORT
    ]

    # Check import count
    assert len(imports) >= 7

    # Check specific imports
    import_names = [ref["target_name"] for ref in imports]
    assert "os" in import_names
    assert "sys" in import_names
    assert "pathlib.Path" in import_names
    assert "typing.List" in import_names


def test_reference_analyzer_class_inheritance():
    """Test class inheritance reference extraction."""
    code = """
class BaseClass:
    pass

class MyClass(BaseClass):
    pass

class MultipleInheritance(BaseClass, dict):
    pass
"""
    tree = ast.parse(code)
    analyzer = ReferenceAnalyzer("test_module", Path("test.py"))
    references = analyzer.analyze(tree)

    # Filter inheritance references
    inherits = [
        ref
        for ref in references
        if ref["reference_type"] == ReferenceType.CLASS_INHERIT
    ]

    # Check inheritance count
    assert (
        len(inherits) == 3
    )  # MyClass -> BaseClass, MultipleInheritance -> BaseClass, MultipleInheritance -> dict

    # Check specific inheritance
    inherit_targets = [ref["target_name"] for ref in inherits]
    assert any("BaseClass" in target for target in inherit_targets)
    assert any("dict" in target for target in inherit_targets)


def test_reference_analyzer_function_calls():
    """Test function call reference extraction."""
    code = """
def helper():
    return 42

def main():
    result = helper()
    print(result)
    len([1, 2, 3])
"""
    tree = ast.parse(code)
    analyzer = ReferenceAnalyzer("test_module", Path("test.py"))
    references = analyzer.analyze(tree)

    # Filter call references
    calls = [
        ref
        for ref in references
        if ref["reference_type"] == ReferenceType.FUNCTION_CALL
    ]

    # Check call count
    assert len(calls) >= 3  # helper(), print(), len()

    # Check specific calls
    call_targets = [ref["target_name"] for ref in calls]
    assert any("helper" in target for target in call_targets)
    assert any("print" in target for target in call_targets)
    assert any("len" in target for target in call_targets)


def test_reference_analyzer_type_hints():
    """Test type hint reference extraction."""
    code = """
from typing import List, Dict, Optional

def func(items: List[str]) -> Dict[str, int]:
    pass

class MyClass:
    def method(self, value: Optional[str] = None) -> bool:
        return True
"""
    tree = ast.parse(code)
    analyzer = ReferenceAnalyzer("test_module", Path("test.py"))
    references = analyzer.analyze(tree)

    # Filter type hint references
    type_hints = [
        ref for ref in references if ref["reference_type"] == ReferenceType.TYPE_HINT
    ]

    # Check type hint count
    assert len(type_hints) >= 3  # List, Dict, Optional, bool

    # Check specific type hints
    hint_targets = [ref["target_name"] for ref in type_hints]
    assert any("List" in target for target in hint_targets)
    assert any("Dict" in target for target in hint_targets)
