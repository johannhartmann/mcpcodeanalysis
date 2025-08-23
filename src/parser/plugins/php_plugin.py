"""PHP language plugin implementation."""

from typing import cast

from src.parser.base_parser import BaseParser
from src.parser.language_config import LanguageConfig, LanguageRegistry
from src.parser.language_plugin import LanguagePlugin
from src.parser.php_parser import PHPCodeParser


class PHPLanguagePlugin(LanguagePlugin):
    """Language plugin for PHP support."""

    def get_language_config(self) -> LanguageConfig:
        """Get PHP language configuration."""
        config = LanguageRegistry.get_language("php")
        if config is None:
            # Fallback configuration if not found in registry
            config = LanguageConfig(
                name="php",
                display_name="PHP",
                extensions=[".php"],
                parser_available=True,
                features={
                    "classes": True,
                    "functions": True,
                    "imports": True,  # use/require
                    "modules": True,  # namespaces
                    "docstrings": True,  # PHPDoc
                    "type_hints": True,  # PHP 7+ has type hints
                },
            )
        return config

    def create_parser(self) -> BaseParser:
        """Create PHP parser instance."""
        return cast("BaseParser", PHPCodeParser())

    def get_complexity_nodes(self) -> set[str]:
        """Get PHP-specific complexity node types.

        Returns TreeSitter node types that contribute to cyclomatic complexity
        in PHP code, including conditionals, loops, exception handling,
        and PHP-specific constructs.
        """
        return {
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
            # PHP-specific constructs
            "ternary_expression",
            "null_coalescing_expression",  # ??
            "null_coalescing_assignment_expression",  # ??=
        }

    def get_analysis_priority(self) -> int:
        """PHP gets medium priority."""
        return 150
