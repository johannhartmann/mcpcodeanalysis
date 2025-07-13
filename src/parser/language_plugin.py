"""Language plugin system for extensible multi-language support."""

from abc import ABC, abstractmethod
from typing import Any

from src.parser.base_parser import BaseParser
from src.parser.language_config import LanguageConfig


class LanguagePlugin(ABC):
    """Abstract base class for language-specific plugins.

    Language plugins provide a standardized way to add support for new
    programming languages without modifying core code. Each plugin defines:
    - Language configuration and capabilities
    - Parser instance creation
    - Language-specific complexity rules
    - Feature flags and processing options
    """

    @abstractmethod
    def get_language_config(self) -> LanguageConfig:
        """Get the language configuration for this plugin.

        Returns:
            LanguageConfig object with language metadata and capabilities
        """

    @abstractmethod
    def create_parser(self) -> BaseParser:
        """Create a parser instance for this language.

        Returns:
            BaseParser instance configured for this language
        """

    @abstractmethod
    def get_complexity_nodes(self) -> set[str]:
        """Get TreeSitter node types that contribute to complexity.

        Returns:
            Set of node type names that should be counted for complexity
        """

    def get_language_name(self) -> str:
        """Get the language name from configuration."""
        return self.get_language_config().name

    def get_supported_extensions(self) -> list[str]:
        """Get file extensions supported by this language."""
        return self.get_language_config().extensions

    def supports_feature(self, feature: str) -> bool:
        """Check if this language supports a specific feature.

        Args:
            feature: Feature name (e.g., 'classes', 'functions', 'type_hints')

        Returns:
            True if feature is supported
        """
        config = self.get_language_config()
        return config.features.get(feature, False)

    def get_parser_options(self) -> dict[str, Any]:
        """Get language-specific parser options.

        Override this method to provide custom parsing options
        such as encoding, syntax variants, or preprocessing rules.

        Returns:
            Dictionary of parser options
        """
        return {}

    def get_analysis_priority(self) -> int:
        """Get analysis priority for this language.

        Higher priority languages are processed first during
        multi-language analysis. Override to customize ordering.

        Returns:
            Priority level (higher = more priority)
        """
        return 100  # Default priority

    def __str__(self) -> str:
        """String representation of the plugin."""
        config = self.get_language_config()
        return f"{config.display_name}Plugin"

    def __repr__(self) -> str:
        """Detailed string representation."""
        config = self.get_language_config()
        return (
            f"LanguagePlugin(name='{config.name}', "
            f"extensions={config.extensions}, "
            f"parser_available={config.parser_available})"
        )
