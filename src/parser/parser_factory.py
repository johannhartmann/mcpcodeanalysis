"""Factory for creating language-specific parsers."""

from pathlib import Path
from typing import Dict, Optional, Type

from src.parser.python_parser import PythonCodeParser
from src.utils.exceptions import ParserError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ParserFactory:
    """Factory for creating language-specific parsers."""
    
    # Mapping of file extensions to parser classes
    _parsers: Dict[str, Type] = {
        ".py": PythonCodeParser,
        ".pyw": PythonCodeParser,
        ".pyi": PythonCodeParser,
    }
    
    # Mapping of language names to parser classes
    _language_parsers: Dict[str, Type] = {
        "python": PythonCodeParser,
    }
    
    @classmethod
    def create_parser(cls, file_path: Path) -> Optional[object]:
        """Create a parser for the given file based on its extension."""
        extension = file_path.suffix.lower()
        
        parser_class = cls._parsers.get(extension)
        if parser_class:
            logger.debug(f"Creating {parser_class.__name__} for {file_path}")
            return parser_class()
        
        logger.warning(f"No parser available for extension: {extension}")
        return None
    
    @classmethod
    def create_parser_by_language(cls, language: str) -> Optional[object]:
        """Create a parser for the given language."""
        language_lower = language.lower()
        
        parser_class = cls._language_parsers.get(language_lower)
        if parser_class:
            logger.debug(f"Creating {parser_class.__name__} for language: {language}")
            return parser_class()
        
        logger.warning(f"No parser available for language: {language}")
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
    def register_parser(cls, extension: str, parser_class: Type) -> None:
        """Register a new parser for a file extension."""
        cls._parsers[extension.lower()] = parser_class
        logger.info(f"Registered parser {parser_class.__name__} for extension: {extension}")
    
    @classmethod
    def register_language_parser(cls, language: str, parser_class: Type) -> None:
        """Register a new parser for a language."""
        cls._language_parsers[language.lower()] = parser_class
        logger.info(f"Registered parser {parser_class.__name__} for language: {language}")