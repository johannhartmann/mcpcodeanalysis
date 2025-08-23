"""Cyclomatic complexity calculator for code analysis."""

# Note: To avoid recursion during plugin initialization, ComplexityCalculator can be
# constructed with use_plugin=False. This prevents querying the LanguagePluginRegistry
# when calculators are created inside TreeSitter-based parsers that may be instantiated
# during plugin registration.


from typing import ClassVar

import tree_sitter

from src.logger import get_logger
from src.parser.plugin_registry import LanguagePluginRegistry

logger = get_logger(__name__)


class ComplexityCalculator:
    """Calculate cyclomatic complexity for functions."""

    # Python node types that increase complexity
    PYTHON_COMPLEXITY_NODES: ClassVar[set[str]] = {
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

    # PHP node types that increase complexity
    PHP_COMPLEXITY_NODES: ClassVar[set[str]] = {
        # Conditionals
        "if_statement",
        "elseif_clause",
        "else_clause",
        "conditional_expression",
        "switch_statement",
        "case_statement",
        # Loops
        "for_statement",
        "foreach_statement",
        "while_statement",
        "do_statement",
        # Exception handling
        "catch_clause",
        # Boolean operators
        "binary_expression",  # Will filter for && and ||
        # Other control flow
        "match_expression",  # PHP 8 match
        "match_conditional_expression",
    }

    # Java node types that increase complexity
    JAVA_COMPLEXITY_NODES: ClassVar[set[str]] = {
        # Conditionals
        "if_statement",
        "else_clause",
        "ternary_expression",
        "switch_expression",
        "switch_label",
        # Loops
        "for_statement",
        "enhanced_for_statement",
        "while_statement",
        "do_statement",
        # Exception handling
        "catch_clause",
        # Boolean operators
        "binary_expression",  # Will filter for && and ||
        # Other control flow
        "throw_statement",
        "assert_statement",
    }

    # TypeScript node types that increase complexity
    TYPESCRIPT_COMPLEXITY_NODES: ClassVar[set[str]] = {
        # Conditionals
        "if_statement",
        "else_clause",
        "ternary_expression",
        "conditional_expression",
        "switch_statement",
        "case_clause",
        # Loops
        "for_statement",
        "for_in_statement",
        "for_of_statement",
        "while_statement",
        "do_statement",
        # Exception handling
        "catch_clause",
        "finally_clause",
        # Boolean operators
        "binary_expression",  # Will filter for && and ||
        "logical_expression",
        # Functions and async
        "function_declaration",
        "function_expression",
        "arrow_function",
        "method_definition",
        "generator_function",
        "async_function",
        # TypeScript-specific
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
        "namespace_declaration",
        "module_declaration",
        # Control flow
        "break_statement",
        "continue_statement",
        "return_statement",
        "throw_statement",
        # Modern JavaScript/TypeScript constructs
        "await_expression",
        "yield_expression",
        "optional_chaining_expression",
        "nullish_coalescing_expression",
    }

    # JavaScript node types that increase complexity
    JAVASCRIPT_COMPLEXITY_NODES: ClassVar[set[str]] = {
        # Conditionals
        "if_statement",
        "else_clause",
        "ternary_expression",
        "conditional_expression",
        "switch_statement",
        "case_clause",
        # Loops
        "for_statement",
        "for_in_statement",
        "for_of_statement",
        "while_statement",
        "do_statement",
        # Exception handling
        "catch_clause",
        "finally_clause",
        # Boolean operators
        "binary_expression",  # Will filter for && and ||
        "logical_expression",
        # Functions and async
        "function_declaration",
        "function_expression",
        "arrow_function",
        "method_definition",
        "generator_function",
        "async_function",
        # Control flow
        "break_statement",
        "continue_statement",
        "return_statement",
        "throw_statement",
        # Modern JavaScript constructs
        "await_expression",
        "yield_expression",
        "optional_chaining_expression",  # ?.
        "nullish_coalescing_expression",  # ??
        "template_literal",
        # Class and object patterns
        "class_declaration",
        "class_expression",
        "object_pattern",
        "array_pattern",
        "assignment_pattern",
    }

    # Default to Python for backward compatibility
    COMPLEXITY_NODES = PYTHON_COMPLEXITY_NODES

    # Nodes that are handled specially
    SPECIAL_NODES: ClassVar[set[str]] = {
        "assert_statement",  # Adds 1 (can fail)
        "with_statement",  # Adds 1 (can fail in __enter__)
        "lambda",  # Lambdas with conditionals
        "try_statement",  # Try blocks
    }

    def __init__(self, language: str = "python", use_plugin: bool = True) -> None:
        """Initialize calculator for specific language.

        Args:
            language: Logical language name.
            use_plugin: When False, avoid consulting the plugin registry to prevent
                recursion during early initialization. Falls back to built-in sets.
        """
        self.language = language.lower()

        if use_plugin:
            # Get complexity nodes from language plugin
            plugin = LanguagePluginRegistry.get_plugin(self.language)
        else:
            plugin = None

        if plugin:
            self.COMPLEXITY_NODES = plugin.get_complexity_nodes()
        # Fallback to language-specific complexity nodes
        elif self.language == "typescript":
            self.COMPLEXITY_NODES = self.TYPESCRIPT_COMPLEXITY_NODES
        elif self.language == "javascript":
            self.COMPLEXITY_NODES = self.JAVASCRIPT_COMPLEXITY_NODES
        elif self.language == "php":
            self.COMPLEXITY_NODES = self.PHP_COMPLEXITY_NODES
        elif self.language == "java":
            self.COMPLEXITY_NODES = self.JAVA_COMPLEXITY_NODES
        else:
            # Final fallback to Python complexity nodes
            logger.warning(
                "No plugin found for language %s, using Python complexity nodes",
                self.language,
            )
            self.COMPLEXITY_NODES = self.PYTHON_COMPLEXITY_NODES

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
            count += self._handle_complexity_node(node, content)

        # Special node handling
        if node.type in self.SPECIAL_NODES:
            count += self._handle_special_node(node)

        # Recursively check children
        for child in node.children:
            count += self._count_complexity_nodes(child, content)

        return count

    def _handle_complexity_node(self, node: tree_sitter.Node, content: bytes) -> int:
        """Handle nodes that add to complexity."""
        if node.type == "else_clause":
            # Only count else if it contains an if (elif)
            return 1 if self._contains_if_statement(node) else 0
        if node.type in {"and", "or"}:
            # Boolean operators add complexity
            return 1
        if node.type == "binary_expression":
            # For PHP and Java, check if it's a logical operator
            operator = self._get_binary_operator(node, content)
            return 1 if operator in ["&&", "||", "and", "or"] else 0
        return 1

    def _handle_special_node(self, node: tree_sitter.Node) -> int:
        """Handle special nodes that add to complexity."""
        if node.type == "assert_statement":
            return 1
        if node.type == "with_statement":
            # Count number of context managers (comma-separated)
            return self._count_with_items(node)
        if node.type == "lambda" and self._lambda_has_conditional(node):
            # Lambda contains conditionals
            return 1
        if node.type == "try_statement":
            # Each catch/except clause adds a path
            return self._count_exception_handlers(node)
        return 0

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

    def _get_binary_operator(self, node: tree_sitter.Node, content: bytes) -> str:
        """Extract the operator from a binary expression node."""
        # Binary expressions typically have structure: left operator right
        for child in node.children:
            # The operator is usually a direct child between operands
            if child.type not in [
                "identifier",
                "call_expression",
                "member_expression",
                "literal",
                "string",
                "integer",
                "boolean",
                "parenthesized_expression",
                "binary_expression",
            ]:
                return content[child.start_byte : child.end_byte].decode(
                    "utf-8", errors="ignore"
                )
        return ""

    def _count_exception_handlers(self, try_node: tree_sitter.Node) -> int:
        """Count the number of exception handlers in a try statement."""
        count = 0
        for child in try_node.children:
            if child.type in ["catch_clause", "except_clause", "finally_clause"]:
                count += 1
        return max(count - 1, 0)  # First handler doesn't add complexity

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
