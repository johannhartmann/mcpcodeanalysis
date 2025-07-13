"""Language configuration for multi-language support."""

from dataclasses import dataclass
from typing import ClassVar

from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LanguageConfig:
    """Configuration for a supported language."""

    name: str
    display_name: str
    extensions: list[str]
    parser_available: bool = False
    features: dict[str, bool] = None

    def __post_init__(self) -> None:
        if self.features is None:
            self.features = {
                "classes": False,
                "functions": False,
                "imports": False,
                "modules": False,
                "docstrings": False,
                "type_hints": False,
            }


class LanguageRegistry:
    """Registry for supported languages."""

    # Default language configurations
    _languages: ClassVar[dict[str, LanguageConfig]] = {
        "python": LanguageConfig(
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
        ),
        "javascript": LanguageConfig(
            name="javascript",
            display_name="JavaScript",
            extensions=[".js", ".jsx", ".mjs"],
            parser_available=False,  # Will be True when parser is implemented
            features={
                "classes": True,
                "functions": True,
                "imports": True,
                "modules": True,
                "docstrings": False,  # JS uses different comment style
                "type_hints": False,
            },
        ),
        "typescript": LanguageConfig(
            name="typescript",
            display_name="TypeScript",
            extensions=[".ts", ".tsx", ".d.ts"],
            parser_available=False,  # Will be True when parser is implemented
            features={
                "classes": True,
                "functions": True,
                "imports": True,
                "modules": True,
                "docstrings": False,
                "type_hints": True,
            },
        ),
        "java": LanguageConfig(
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
        ),
        "go": LanguageConfig(
            name="go",
            display_name="Go",
            extensions=[".go"],
            parser_available=False,
            features={
                "classes": False,  # Go uses structs
                "functions": True,
                "imports": True,
                "modules": True,  # Go modules/packages
                "docstrings": True,  # Go doc comments
                "type_hints": True,  # Go is strongly typed
            },
        ),
        "rust": LanguageConfig(
            name="rust",
            display_name="Rust",
            extensions=[".rs"],
            parser_available=False,
            features={
                "classes": False,  # Rust uses structs/traits
                "functions": True,
                "imports": True,  # use statements
                "modules": True,
                "docstrings": True,  # /// doc comments
                "type_hints": True,  # Rust is strongly typed
            },
        ),
        "cpp": LanguageConfig(
            name="cpp",
            display_name="C++",
            extensions=[".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx", ".h"],
            parser_available=False,
            features={
                "classes": True,
                "functions": True,
                "imports": True,  # #include
                "modules": False,  # C++ uses namespaces
                "docstrings": True,  # Doxygen style
                "type_hints": True,
            },
        ),
        "c": LanguageConfig(
            name="c",
            display_name="C",
            extensions=[".c", ".h"],
            parser_available=False,
            features={
                "classes": False,  # C doesn't have classes
                "functions": True,
                "imports": True,  # #include
                "modules": False,
                "docstrings": True,  # Doxygen style
                "type_hints": True,
            },
        ),
        "csharp": LanguageConfig(
            name="csharp",
            display_name="C#",
            extensions=[".cs"],
            parser_available=False,
            features={
                "classes": True,
                "functions": True,  # Methods
                "imports": True,  # using statements
                "modules": True,  # namespaces
                "docstrings": True,  # XML doc comments
                "type_hints": True,
            },
        ),
        "ruby": LanguageConfig(
            name="ruby",
            display_name="Ruby",
            extensions=[".rb", ".rake"],
            parser_available=False,
            features={
                "classes": True,
                "functions": True,  # Methods
                "imports": True,  # require/include
                "modules": True,
                "docstrings": True,  # RDoc
                "type_hints": False,  # Ruby is dynamically typed
            },
        ),
        "php": LanguageConfig(
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
        ),
    }

    @classmethod
    def get_language(cls, name: str) -> LanguageConfig | None:
        """Get language configuration by name."""
        return cls._languages.get(name.lower())

    @classmethod
    def get_language_by_extension(cls, extension: str) -> LanguageConfig | None:
        """Get language configuration by file extension."""
        ext_lower = extension.lower()
        for lang in cls._languages.values():
            if ext_lower in lang.extensions:
                return lang
        return None

    @classmethod
    def get_supported_languages(cls) -> list[str]:
        """Get list of all supported language names."""
        return list(cls._languages.keys())

    @classmethod
    def get_available_languages(cls) -> list[str]:
        """Get list of languages with available parsers."""
        return [name for name, lang in cls._languages.items() if lang.parser_available]

    @classmethod
    def get_supported_extensions(cls) -> set[str]:
        """Get all supported file extensions."""
        extensions = set()
        for lang in cls._languages.values():
            extensions.update(lang.extensions)
        return extensions

    @classmethod
    def get_available_extensions(cls) -> set[str]:
        """Get extensions for languages with available parsers."""
        extensions = set()
        for lang in cls._languages.values():
            if lang.parser_available:
                extensions.update(lang.extensions)
        return extensions

    @classmethod
    def register_language(cls, config: LanguageConfig) -> None:
        """Register a new language configuration."""
        cls._languages[config.name.lower()] = config
        logger.info(
            "Registered language %s with extensions %s",
            config.display_name,
            config.extensions,
        )

    @classmethod
    def update_language(cls, name: str, **kwargs) -> None:
        """Update an existing language configuration."""
        lang = cls._languages.get(name.lower())
        if lang:
            for key, value in kwargs.items():
                if hasattr(lang, key):
                    setattr(lang, key, value)
            logger.info("Updated language configuration for %s", name)
        else:
            logger.warning("Language %s not found for update", name)

    @classmethod
    def is_extension_supported(cls, extension: str) -> bool:
        """Check if a file extension is supported."""
        return extension.lower() in cls.get_supported_extensions()

    @classmethod
    def is_extension_available(cls, extension: str) -> bool:
        """Check if a file extension has an available parser."""
        return extension.lower() in cls.get_available_extensions()
