"""Tests for cyclomatic complexity calculator."""

import tree_sitter
import tree_sitter_python as tspython

from src.parser.complexity_calculator import ComplexityCalculator


class TestComplexityCalculator:
    """Tests for ComplexityCalculator class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.calculator = ComplexityCalculator()
        self.language = tree_sitter.Language(tspython.language())
        self.parser = tree_sitter.Parser(self.language)

    def _parse_function(self, code: str) -> tree_sitter.Node:
        """Parse a function and return its node."""
        tree = self.parser.parse(code.encode())
        # Find the first function definition
        for node in tree.root_node.children:
            if node.type == "function_definition":
                return node
        raise ValueError("No function found in code")

    def test_simple_function(self) -> None:
        """Test complexity of a simple function."""
        code = """
def simple_function():
    return 42
"""
        node = self._parse_function(code)
        complexity = self.calculator.calculate_complexity(node, code.encode())
        assert complexity == 1

    def test_single_if_statement(self) -> None:
        """Test function with single if statement."""
        code = """
def function_with_if(x):
    if x > 0:
        return True
    return False
"""
        node = self._parse_function(code)
        complexity = self.calculator.calculate_complexity(node, code.encode())
        assert complexity == 2  # 1 base + 1 if

    def test_if_else_statement(self) -> None:
        """Test function with if-else statement."""
        code = """
def function_with_if_else(x):
    if x > 0:
        return "positive"
    else:
        return "non-positive"
"""
        node = self._parse_function(code)
        complexity = self.calculator.calculate_complexity(node, code.encode())
        assert complexity == 2  # 1 base + 1 if (else doesn't add complexity)

    def test_if_elif_else_statement(self) -> None:
        """Test function with if-elif-else statement."""
        code = """
def function_with_elif(x):
    if x > 0:
        return "positive"
    elif x < 0:
        return "negative"
    else:
        return "zero"
"""
        node = self._parse_function(code)
        complexity = self.calculator.calculate_complexity(node, code.encode())
        assert complexity == 3  # 1 base + 1 if + 1 elif

    def test_for_loop(self) -> None:
        """Test function with for loop."""
        code = """
def function_with_for(items):
    total = 0
    for item in items:
        total += item
    return total
"""
        node = self._parse_function(code)
        complexity = self.calculator.calculate_complexity(node, code.encode())
        assert complexity == 2  # 1 base + 1 for

    def test_while_loop(self) -> None:
        """Test function with while loop."""
        code = """
def function_with_while(n):
    count = 0
    while n > 0:
        count += 1
        n -= 1
    return count
"""
        node = self._parse_function(code)
        complexity = self.calculator.calculate_complexity(node, code.encode())
        assert complexity == 2  # 1 base + 1 while

    def test_nested_loops(self) -> None:
        """Test function with nested loops."""
        code = """
def function_with_nested_loops(matrix):
    total = 0
    for row in matrix:
        for value in row:
            if value > 0:
                total += value
    return total
"""
        node = self._parse_function(code)
        complexity = self.calculator.calculate_complexity(node, code.encode())
        assert complexity == 4  # 1 base + 1 for + 1 for + 1 if

    def test_exception_handling(self) -> None:
        """Test function with exception handling."""
        code = """
def function_with_try_except(x):
    try:
        result = 10 / x
    except ZeroDivisionError:
        return None
    except ValueError:
        return -1
    return result
"""
        node = self._parse_function(code)
        complexity = self.calculator.calculate_complexity(node, code.encode())
        assert complexity == 3  # 1 base + 2 except clauses

    def test_boolean_operators(self) -> None:
        """Test function with boolean operators."""
        code = """
def function_with_boolean_ops(a, b, c):
    if a and b:
        return True
    if b or c:
        return True
    return False
"""
        node = self._parse_function(code)
        complexity = self.calculator.calculate_complexity(node, code.encode())
        assert complexity == 5  # 1 base + 2 if + 1 and + 1 or

    def test_ternary_operator(self) -> None:
        """Test function with ternary operator."""
        code = """
def function_with_ternary(x):
    return "positive" if x > 0 else "non-positive"
"""
        node = self._parse_function(code)
        complexity = self.calculator.calculate_complexity(node, code.encode())
        assert complexity == 2  # 1 base + 1 conditional expression

    def test_list_comprehension(self) -> None:
        """Test function with list comprehension."""
        code = """
def function_with_list_comp(items):
    return [x * 2 for x in items if x > 0]
"""
        node = self._parse_function(code)
        complexity = self.calculator.calculate_complexity(node, code.encode())
        assert complexity == 2  # 1 base + 1 list comprehension

    def test_assert_statement(self) -> None:
        """Test function with assert statement."""
        code = """
def function_with_assert(x):
    assert x > 0, "x must be positive"
    return x * 2
"""
        node = self._parse_function(code)
        complexity = self.calculator.calculate_complexity(node, code.encode())
        assert complexity == 2  # 1 base + 1 assert

    def test_with_statement(self) -> None:
        """Test function with with statement."""
        code = """
def function_with_context_manager(filename):
    with open(filename) as f:
        return f.read()
"""
        node = self._parse_function(code)
        complexity = self.calculator.calculate_complexity(node, code.encode())
        assert complexity == 1  # with doesn't add complexity for single context manager

    def test_multiple_with_items(self) -> None:
        """Test function with multiple context managers."""
        code = """
def function_with_multiple_contexts(file1, file2):
    with open(file1) as f1, open(file2) as f2:
        return f1.read() + f2.read()
"""
        node = self._parse_function(code)
        complexity = self.calculator.calculate_complexity(node, code.encode())
        assert complexity == 2  # 1 base + 1 for second context manager

    def test_lambda_with_conditional(self) -> None:
        """Test function containing lambda with conditional."""
        code = """
def function_with_lambda():
    return lambda x: x if x > 0 else -x
"""
        node = self._parse_function(code)
        complexity = self.calculator.calculate_complexity(node, code.encode())
        # Note: The lambda itself adds 1, and the conditional inside adds 1
        assert complexity == 3  # 1 base + 1 lambda + 1 conditional

    def test_complex_function(self) -> None:
        """Test a complex function with multiple control flow elements."""
        code = """
def complex_function(data, threshold):
    if not data:
        return None

    results = []
    for item in data:
        if item > threshold:
            try:
                value = process(item)
                if value and (value > 10 or value < -10):
                    results.append(value)
            except ValueError:
                continue
            except Exception:
                break
        elif item < 0:
            results.append(0)

    return results if results else None
"""
        node = self._parse_function(code)
        complexity = self.calculator.calculate_complexity(node, code.encode())
        # 1 base + 1 if (not data) + 1 for + 1 if (item > threshold) +
        # 1 if (value and ...) + 1 and + 1 or + 2 except + 1 elif + 1 conditional
        assert complexity == 11

    def test_get_complexity_level(self) -> None:
        """Test complexity level categorization."""
        assert self.calculator.get_complexity_level(1) == "simple"
        assert self.calculator.get_complexity_level(5) == "simple"
        assert self.calculator.get_complexity_level(10) == "simple"
        assert self.calculator.get_complexity_level(11) == "moderate"
        assert self.calculator.get_complexity_level(20) == "moderate"
        assert self.calculator.get_complexity_level(21) == "complex"
        assert self.calculator.get_complexity_level(50) == "complex"
        assert self.calculator.get_complexity_level(51) == "very complex"
        assert self.calculator.get_complexity_level(100) == "very complex"
