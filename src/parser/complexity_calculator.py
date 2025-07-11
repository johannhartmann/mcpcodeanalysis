"""Cyclomatic complexity calculator for code analysis."""

from typing import ClassVar

import tree_sitter

from src.logger import get_logger

logger = get_logger(__name__)


class ComplexityCalculator:
    """Calculate cyclomatic complexity for functions."""

    # Node types that increase complexity
    COMPLEXITY_NODES: ClassVar[set[str]] = {
        # Conditionals
        "if_statement",
        "elif_clause",
        "else_clause",  # Only counts if it contains another if
        "conditional_expression",  # ternary operator
        # Loops
        "for_statement",
        "while_statement",
        # Exception handling
        "except_clause",
        # Boolean operators (each one adds a path)
        "and",
        "or",
        # Other control flow
        "match_statement",  # Python 3.10+ pattern matching
        "case_clause",
        # Comprehensions (each adds complexity)
        "list_comprehension",
        "set_comprehension",
        "dictionary_comprehension",
        "generator_expression",
    }

    # Nodes that are handled specially
    SPECIAL_NODES: ClassVar[set[str]] = {
        "assert_statement",  # Adds 1 (can fail)
        "with_statement",  # Adds 1 (can fail in __enter__)
        "lambda",  # Lambdas with conditionals
    }

    def calculate_complexity(
        self, function_node: tree_sitter.Node, content: bytes
    ) -> int:
        """
        Calculate cyclomatic complexity for a function.

        Cyclomatic complexity = E - N + 2P
        Where:
        - E = number of edges in the control flow graph
        - N = number of nodes in the control flow graph
        - P = number of connected components (usually 1 for a function)

        In practice, we use the simpler formula:
        M = 1 + number of decision points

        Args:
            function_node: TreeSitter node for the function
            content: Source code content

        Returns:
            Cyclomatic complexity score
        """
        # Base complexity is 1
        complexity = 1

        # Find the function body (block node)
        body_node = None
        for child in function_node.children:
            if child.type == "block":
                body_node = child
                break

        if not body_node:
            # No body means it's likely a stub or abstract method
            return 1

        # Count complexity nodes
        complexity += self._count_complexity_nodes(body_node, content)

        return complexity

    def _count_complexity_nodes(self, node: tree_sitter.Node, content: bytes) -> int:
        """Recursively count nodes that add to complexity."""
        count = 0

        # Check if this node adds complexity
        if node.type in self.COMPLEXITY_NODES:
            # Special handling for certain nodes
            if node.type == "else_clause":
                # Only count else if it contains an if (elif)
                if self._contains_if_statement(node):
                    count += 1
            elif node.type in {"and", "or"}:
                # Boolean operators add complexity
                count += 1
            else:
                count += 1

        # Special node handling
        if node.type in self.SPECIAL_NODES:
            if node.type == "assert_statement":
                count += 1
            elif node.type == "with_statement":
                # Count number of context managers (comma-separated)
                count += self._count_with_items(node)
            elif node.type == "lambda" and self._lambda_has_conditional(node):
                # Lambda contains conditionals
                count += 1

        # Recursively check children
        for child in node.children:
            count += self._count_complexity_nodes(child, content)

        return count

    def _contains_if_statement(self, else_node: tree_sitter.Node) -> bool:
        """Check if an else clause contains an if statement (making it elif-like)."""
        return any(child.type == "if_statement" for child in else_node.children)

    def _count_with_items(self, with_node: tree_sitter.Node) -> int:
        """Count number of context managers in a with statement."""
        count = 0
        for child in with_node.children:
            if child.type == "with_clause":
                # Count with_items inside with_clause
                for subchild in child.children:
                    if subchild.type == "with_item":
                        count += 1
        return max(count - 1, 0)  # First item doesn't add complexity

    def _lambda_has_conditional(self, lambda_node: tree_sitter.Node) -> bool:
        """Check if a lambda expression contains a conditional."""
        for child in lambda_node.children:
            if child.type == "conditional_expression":
                return True
            # Recursively check for nested conditionals
            if self._node_has_conditional(child):
                return True
        return False

    def _node_has_conditional(self, node: tree_sitter.Node) -> bool:
        """Recursively check if a node contains any conditional."""
        if node.type == "conditional_expression":
            return True
        return any(self._node_has_conditional(child) for child in node.children)

    def get_complexity_level(self, complexity: int) -> str:
        """
        Get human-readable complexity level.

        Based on common standards:
        - 1-10: Simple, low risk
        - 11-20: Moderate complexity
        - 21-50: Complex, refactoring suggested
        - 50+: Very complex, high risk
        """
        if complexity <= 10:
            return "simple"
        if complexity <= 20:
            return "moderate"
        if complexity <= 50:
            return "complex"
        return "very complex"
