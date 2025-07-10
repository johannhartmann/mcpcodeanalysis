"""Factory for creating language-specific parsers."""

from pathlib import Path
from typing import ClassVar

from src.logger import get_logger
from src.parser.python_parser import PythonCodeParser

logger = get_logger(__name__)


class ParserFactory:
    """Factory for creating language-specific parsers."""

    # Mapping of file extensions to parser classes
    _parsers: ClassVar[dict[str, type]] = {
        ".py": PythonCodeParser,
        ".pyw": PythonCodeParser,
        ".pyi": PythonCodeParser,
    }

    # Mapping of language names to parser classes
    _language_parsers: ClassVar[dict[str, type]] = {
        "python": PythonCodeParser,
    }

    @classmethod
    def create_parser(cls, file_path: Path) -> object | None:
        """Create a parser for the given file based on its extension."""
        extension = file_path.suffix.lower()

        parser_class = cls._parsers.get(extension)
        if parser_class:
            logger.debug("Creating %s for %s", parser_class.__name__, file_path)
            return parser_class()

        logger.warning("No parser available for extension: %s", extension)
        return None

    @classmethod
    def create_parser_by_language(cls, language: str) -> object | None:
        """Create a parser for the given language."""
        language_lower = language.lower()

        parser_class = cls._language_parsers.get(language_lower)
        if parser_class:
            logger.debug(
                "Creating %s for language: %s",
                parser_class.__name__,
                language,
            )
            return parser_class()

        logger.warning("No parser available for language: %s", language)
        return None

    @classmethod
    def is_supported(cls, file_path: Path) -> bool:
        """Check if a file type is supported for parsing."""
        return file_path.suffix.lower() in cls._parsers

    @classmethod
    def is_language_supported(cls, language: str) -> bool:
        """Check if a language is supported for parsing."""
        return language.lower() in cls._language_parsers

    @classmethod
    def get_supported_extensions(cls) -> list[str]:
        """Get list of supported file extensions."""
        return list(cls._parsers.keys())

    @classmethod
    def get_supported_languages(cls) -> list[str]:
        """Get list of supported languages."""
        return list(cls._language_parsers.keys())

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
