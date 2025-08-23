"""TypeScript language plugin implementation."""

from collections.abc import Callable

from src.logger import get_logger
from src.parser.base_parser import BaseParser
from src.parser.language_config import LanguageConfig, LanguageRegistry
from src.parser.language_plugin import LanguagePlugin

logger = get_logger(__name__)

# Prepare a factory to create the TypeScript parser without tripping mypy on
# BaseParser's constructor signature.
TypeScriptParserFactory: Callable[[], BaseParser] | None = None

try:
    from src.parser.typescript_parser import (
        TypeScriptCodeParser as _TypeScriptCodeParser,
    )

    TYPESCRIPT_PARSER_AVAILABLE = True

    def _make_ts_parser() -> BaseParser:
        return _TypeScriptCodeParser()

    TypeScriptParserFactory = _make_ts_parser
except ImportError:
    TYPESCRIPT_PARSER_AVAILABLE = False


class TypeScriptLanguagePlugin(LanguagePlugin):
    """Language plugin for TypeScript support."""

    def get_language_config(self) -> LanguageConfig:
        """Get TypeScript language configuration."""
        config = LanguageRegistry.get_language("typescript")
        if config is None:
            # Fallback configuration if not found in registry
            config = LanguageConfig(
                name="typescript",
                display_name="TypeScript",
                extensions=[".ts", ".tsx", ".d.ts"],
                parser_available=True,
                features={
                    "classes": True,
                    "functions": True,
                    "imports": True,
                    "modules": True,
                    "docstrings": False,  # TypeScript uses JSDoc comments
                    "type_hints": True,
                },
            )
        return config

    def create_parser(self) -> BaseParser:
        """Create TypeScript parser instance."""
        if not TYPESCRIPT_PARSER_AVAILABLE or TypeScriptParserFactory is None:
            logger.error(
                "TypeScript parser not available. Install tree-sitter-typescript to enable TypeScript support."
            )
            msg = "TypeScript parser not available"
            raise ImportError(msg)
        return TypeScriptParserFactory()

    def get_complexity_nodes(self) -> set[str]:
        """Get TypeScript-specific complexity node types.

        Returns TreeSitter node types that contribute to cyclomatic complexity
        in TypeScript code, including conditionals, loops, exception handling,
        and TypeScript-specific constructs.
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

    def get_analysis_priority(self) -> int:
        """TypeScript gets medium-high priority."""
        return 160
