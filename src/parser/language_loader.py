"""Load language configurations from settings."""

from src.config import settings
from src.logger import get_logger
from src.parser.language_config import LanguageRegistry

logger = get_logger(__name__)


def load_language_configuration() -> None:
    """Load language configuration from settings and update registry."""
    # Get configured languages from settings
    configured_languages = settings.get("parser.languages", ["python"])

    logger.info("Loading language configuration for: %s", configured_languages)

    # Update registry to mark configured languages as enabled
    for lang_name in configured_languages:
        lang_config = LanguageRegistry.get_language(lang_name)
        if lang_config:
            # For now, only Python has a parser available
            # This will be updated as we add more parsers
            if lang_name == "python":
                lang_config.parser_available = True
                logger.info("Enabled parser for language: %s", lang_name)
            else:
                logger.warning(
                    "Language %s is configured but parser not yet implemented",
                    lang_name,
                )
        else:
            logger.warning("Unknown language in configuration: %s", lang_name)

    # Log summary
    available_langs = LanguageRegistry.get_available_languages()
    available_exts = LanguageRegistry.get_available_extensions()

    logger.info(
        "Language support initialized: %d languages, %d extensions",
        len(available_langs),
        len(available_exts),
    )
    logger.debug("Available languages: %s", available_langs)
    logger.debug("Available extensions: %s", sorted(available_exts))


def get_configured_extensions() -> set[str]:
    """Get file extensions for configured languages."""
    configured_languages = settings.get("parser.languages", ["python"])
    extensions = set()

    for lang_name in configured_languages:
        lang_config = LanguageRegistry.get_language(lang_name)
        if lang_config and lang_config.parser_available:
            extensions.update(lang_config.extensions)

    return extensions


def is_language_enabled(language: str) -> bool:
    """Check if a language is enabled in configuration."""
    configured_languages = settings.get("parser.languages", ["python"])
    return language.lower() in [lang.lower() for lang in configured_languages]


# Initialize language configuration on module import
load_language_configuration()
