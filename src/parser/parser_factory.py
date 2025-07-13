"""Factory for creating language-specific parsers."""

from pathlib import Path
from typing import ClassVar

from src.logger import get_logger
from src.parser.plugin_registry import LanguagePluginRegistry

logger = get_logger(__name__)


class ParserFactory:
    """Factory for creating language-specific parsers."""

    # Legacy mappings maintained for backward compatibility
    _parsers: ClassVar[dict[str, type]] = {}
    _language_parsers: ClassVar[dict[str, type]] = {}

    @classmethod
    def create_parser(cls, file_path: Path) -> object | None:
        """Create a parser for the given file based on its extension."""
        # Use plugin registry for parser creation
        plugin = LanguagePluginRegistry.get_plugin_by_file_path(file_path)
        if plugin:
            logger.debug("Creating parser for %s using %s", file_path, plugin)
            return plugin.create_parser()

        # Fallback to legacy mapping for backward compatibility
        extension = file_path.suffix.lower()
        parser_class = cls._parsers.get(extension)
        if parser_class:
            logger.debug(
                "Creating %s for %s (legacy)", parser_class.__name__, file_path
            )
            return parser_class()

        logger.warning("No parser available for extension: %s", extension)
        return None

    @classmethod
    def create_parser_by_language(cls, language: str) -> object | None:
        """Create a parser for the given language."""
        # Use plugin registry for parser creation
        plugin = LanguagePluginRegistry.get_plugin(language)
        if plugin:
            logger.debug("Creating parser for language %s using %s", language, plugin)
            return plugin.create_parser()

        # Fallback to legacy mapping for backward compatibility
        language_lower = language.lower()
        parser_class = cls._language_parsers.get(language_lower)
        if parser_class:
            logger.debug(
                "Creating %s for language: %s (legacy)",
                parser_class.__name__,
                language,
            )
            return parser_class()

        logger.warning("No parser available for language: %s", language)
        return None

    @classmethod
    def is_supported(cls, file_path: Path) -> bool:
        """Check if a file type is supported for parsing."""
        # Use plugin registry to check support
        return LanguagePluginRegistry.is_supported(file_path)

    @classmethod
    def is_language_supported(cls, language: str) -> bool:
        """Check if a language is supported for parsing."""
        # Use plugin registry to check language support
        return LanguagePluginRegistry.is_language_supported(language)

    @classmethod
    def get_supported_extensions(cls) -> list[str]:
        """Get list of supported file extensions."""
        # Use plugin registry to get supported extensions
        return LanguagePluginRegistry.get_supported_extensions()

    @classmethod
    def get_supported_languages(cls) -> list[str]:
        """Get list of supported languages."""
        # Use plugin registry to get supported languages
        return LanguagePluginRegistry.get_supported_languages()

    @classmethod
    def register_parser(cls, extension: str, parser_class: type) -> None:
        """Register a new parser for a file extension."""
        cls._parsers[extension.lower()] = parser_class
        logger.info(
            "Registered parser %s for extension: %s",
            parser_class.__name__,
            extension,
        )

    @classmethod
    def register_language_parser(cls, language: str, parser_class: type) -> None:
        """Register a new parser for a language."""
        cls._language_parsers[language.lower()] = parser_class
        logger.info(
            "Registered parser %s for language: %s",
            parser_class.__name__,
            language,
        )
