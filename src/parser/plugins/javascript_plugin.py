"""JavaScript language plugin implementation."""

try:
    from src.parser.javascript_parser import JavaScriptCodeParser

    JAVASCRIPT_PARSER_AVAILABLE = True
except ImportError:
    JAVASCRIPT_PARSER_AVAILABLE = False
    JavaScriptCodeParser = None

from src.logger import get_logger
from src.parser.base_parser import BaseParser
from src.parser.language_config import LanguageConfig, LanguageRegistry
from src.parser.language_plugin import LanguagePlugin

logger = get_logger(__name__)


class JavaScriptLanguagePlugin(LanguagePlugin):
    """Language plugin for JavaScript support."""

    def get_language_config(self) -> LanguageConfig:
        """Get JavaScript language configuration."""
        config = LanguageRegistry.get_language("javascript")
        if config is None:
            # Fallback configuration if not found in registry
            config = LanguageConfig(
                name="javascript",
                display_name="JavaScript",
                extensions=[".js", ".jsx", ".mjs"],
                parser_available=True,
                features={
                    "classes": True,
                    "functions": True,
                    "imports": True,
                    "modules": True,
                    "docstrings": False,  # JavaScript uses JSDoc comments
                    "type_hints": False,
                },
            )
        return config

    def create_parser(self) -> BaseParser:
        """Create JavaScript parser instance."""
        if not JAVASCRIPT_PARSER_AVAILABLE or JavaScriptCodeParser is None:
            logger.error(
                "JavaScript parser not available. Install tree-sitter-javascript to enable JavaScript support."
            )
            raise ImportError("JavaScript parser not available")
        return JavaScriptCodeParser()

    def get_complexity_nodes(self) -> set[str]:
        """Get JavaScript-specific complexity node types.

        Returns TreeSitter node types that contribute to cyclomatic complexity
        in JavaScript code, including conditionals, loops, exception handling,
        and modern JavaScript constructs.
        """
        return {
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

    def get_analysis_priority(self) -> int:
        """JavaScript gets medium priority."""
        return 140
