"""Python language plugin implementation."""

from src.parser.base_parser import BaseParser
from src.parser.language_config import LanguageConfig, LanguageRegistry
from src.parser.language_plugin import LanguagePlugin
from src.parser.python_parser import PythonCodeParser


class PythonLanguagePlugin(LanguagePlugin):
    """Language plugin for Python support."""

    def get_language_config(self) -> LanguageConfig:
        """Get Python language configuration."""
        config = LanguageRegistry.get_language("python")
        if config is None:
            # Fallback configuration if not found in registry
            config = LanguageConfig(
                name="python",
                display_name="Python",
                extensions=[".py", ".pyw", ".pyi"],
                parser_available=True,
                features={
                    "classes": True,
                    "functions": True,
                    "imports": True,
                    "modules": True,
                    "docstrings": True,
                    "type_hints": True,
                },
            )
        return config

    def create_parser(self) -> BaseParser:
        """Create Python parser instance."""
        return PythonCodeParser()

    def get_complexity_nodes(self) -> set[str]:
        """Get Python-specific complexity node types.

        Returns TreeSitter node types that contribute to cyclomatic complexity
        in Python code, including conditionals, loops, exception handling,
        and comprehensions.
        """
        return {
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
            # Additional Python-specific constructs
            "assert_statement",  # Adds 1 (can fail)
            "with_statement",  # Adds 1 (can fail in __enter__)
            "lambda",  # Lambdas with conditionals
            "try_statement",  # Try blocks
        }

    def get_analysis_priority(self) -> int:
        """Python gets high priority as it's the primary language."""
        return 200
