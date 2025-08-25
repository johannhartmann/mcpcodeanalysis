"""Java language plugin implementation."""

from typing import cast

from src.parser.base_parser import BaseParser
from src.parser.java_parser import JavaCodeParser
from src.parser.language_config import LanguageConfig, LanguageRegistry
from src.parser.language_plugin import LanguagePlugin


class JavaLanguagePlugin(LanguagePlugin):
    """Language plugin for Java support."""

    def get_language_config(self) -> LanguageConfig:
        """Get Java language configuration."""
        config = LanguageRegistry.get_language("java")
        if config is None:
            # Fallback configuration if not found in registry
            config = LanguageConfig(
                name="java",
                display_name="Java",
                extensions=[".java"],
                parser_available=True,
                features={
                    "classes": True,
                    "functions": True,  # Methods in Java
                    "imports": True,
                    "modules": False,  # Java uses packages
                    "docstrings": True,  # JavaDoc
                    "type_hints": True,  # Java is strongly typed
                },
            )
        return config

    def create_parser(self) -> BaseParser:
        """Create Java parser instance."""
        return cast("BaseParser", JavaCodeParser())

    def get_complexity_nodes(self) -> set[str]:
        """Get Java-specific complexity node types.

        Returns TreeSitter node types that contribute to cyclomatic complexity
        in Java code, including conditionals, loops, exception handling,
        and Java-specific constructs.
        """
        return {
            # Conditionals
            "if_statement",
            "else_clause",
            "ternary_expression",
            "switch_expression",
            "switch_label",
            # Loops
            "for_statement",
            "enhanced_for_statement",  # for-each loop
            "while_statement",
            "do_statement",
            # Exception handling
            "catch_clause",
            "try_with_resources_statement",
            # Boolean operators
            "binary_expression",  # Will filter for && and ||
            # Other control flow
            "throw_statement",
            "assert_statement",
            # Java-specific constructs
            "instanceof_expression",
            "conditional_expression",
            "lambda_expression",
            "method_reference",
        }

    def get_analysis_priority(self) -> int:
        """Java gets medium priority."""
        return 150
