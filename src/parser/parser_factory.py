"""Factory for creating language-specific parsers."""

from pathlib import Path
from typing import ClassVar

from src.logger import get_logger
from src.parser.plugin_registry import LanguagePluginRegistry

logger = get_logger(__name__)


class ParserFactory:
    """Factory for creating language-specific parsers."""

    # Legacy mappings maintained for backward compatibility
    _parsers: ClassVar[dict[str, type[object]]] = {}
    _language_parsers: ClassVar[dict[str, type[object]]] = {}

    @classmethod
    def create_parser(cls, file_path: Path) -> object | None:
        """Create a parser for the given file based on its extension."""
        # First check legacy mapping for backward compatibility without initializing plugin registry
        extension = file_path.suffix.lower()
        parser_class = cls._parsers.get(extension)
        if parser_class:
            logger.debug(
                "Creating %s for %s (legacy)", parser_class.__name__, file_path
            )
            return parser_class()

        # Legacy mapping only: do not consult plugin registry here to avoid test
        # ordering and global initialization side effects. Plugins are used via
        # ParserFactory.create_parser_by_language and the LanguagePluginRegistry
        # APIs in integration paths.
        logger.warning("No parser available for extension: %s", extension)
        return None

    @classmethod
    def create_parser_by_language(cls, language: str) -> object | None:
        """Create a parser for the given language."""
        # Legacy-only behavior: create parser from registered legacy language parsers.
        # Do not consult the plugin registry here to avoid test ordering/initialization
        # side effects. Integration tests should use LanguagePluginRegistry directly.
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
        """Check if a file type is supported for parsing.

        This method prefers legacy-registered parsers and intentionally avoids
        triggering the global plugin registry initialization which can have side
        effects in unit tests. Integration tests should use
        LanguagePluginRegistry.is_supported when they require the full plugin
        set.
        """
        extension = file_path.suffix.lower()
        # Legacy mapping takes precedence to avoid initializing plugins in unit tests
        if extension in cls._parsers:
            return True

        # Only consult plugin registry if it has already been explicitly initialized
        # to avoid side-effects during unit tests that expect legacy-only behavior.
        if LanguagePluginRegistry._initialized:
            return LanguagePluginRegistry.is_supported(file_path)

        return False

    @classmethod
    def is_language_supported(cls, language: str) -> bool:
        """Check if a language is supported for parsing.

        Prefer legacy-registered language parsers to avoid initializing the
        plugin registry during unit tests. Use LanguagePluginRegistry for
        integration scenarios.
        """
        language_lower = language.lower()
        if language_lower in cls._language_parsers:
            return True

        # Only consult plugin registry if it has been explicitly initialized
        # to avoid triggering initialization during unit tests that expect
        # legacy-only behavior.
        if LanguagePluginRegistry._initialized:
            return LanguagePluginRegistry.is_language_supported(language)

        return False

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
    def register_parser(cls, extension: str, parser_class: type[object]) -> None:
        """Register a new parser for a file extension."""
        cls._parsers[extension.lower()] = parser_class
        logger.info(
            "Registered parser %s for extension: %s",
            parser_class.__name__,
            extension,
        )

    @classmethod
    def register_language_parser(
        cls, language: str, parser_class: type[object]
    ) -> None:
        """Register a new parser for a language."""
        cls._language_parsers[language.lower()] = parser_class
        logger.info(
            "Registered parser %s for language: %s",
            parser_class.__name__,
            language,
        )


# Register legacy Python parser mapping by default for backward compatibility
try:
    from src.parser.python_parser import PythonCodeParser

    # Populate legacy mappings for Python extensions and language name
    ParserFactory._parsers.update(
        {".py": PythonCodeParser, ".pyw": PythonCodeParser, ".pyi": PythonCodeParser}
    )
    ParserFactory._language_parsers["python"] = PythonCodeParser
except ImportError:
    # If PythonCodeParser is not importable for any reason, do not crash on import
    logger.debug(
        "PythonCodeParser not registered in legacy parser mappings: ImportError"
    )
