"""Registry for language plugins providing centralized plugin management."""

from pathlib import Path
from typing import ClassVar

from src.logger import get_logger
from src.parser.language_plugin import LanguagePlugin

logger = get_logger(__name__)


class LanguagePluginRegistry:
    """Central registry for language plugins.

    This registry manages language plugins and provides a single point
    of access for language-specific functionality. It supports:
    - Plugin registration and discovery
    - Language detection from file paths
    - Plugin retrieval by language name or file extension
    - Validation of plugin compatibility
    """

    _plugins: ClassVar[dict[str, LanguagePlugin]] = {}
    _extension_map: ClassVar[dict[str, str]] = {}
    _initialized: ClassVar[bool] = False

    @classmethod
    def register_plugin(cls, plugin: LanguagePlugin) -> None:
        """Register a language plugin.

        Args:
            plugin: LanguagePlugin instance to register

        Raises:
            ValueError: If plugin configuration is invalid
        """
        try:
            config = plugin.get_language_config()
            language_name = config.name.lower()

            # Validate plugin configuration
            if not config.name:
                raise ValueError("Plugin must have a non-empty name")
            if not config.extensions:
                raise ValueError("Plugin must support at least one file extension")

            # Test parser creation if available
            if config.parser_available:
                try:
                    test_parser = plugin.create_parser()
                    if test_parser is None:
                        logger.warning(
                            "Plugin %s claims parser_available=True but create_parser() returned None",
                            config.display_name,
                        )
                        config.parser_available = False
                except Exception as e:
                    logger.warning(
                        "Plugin %s claims parser_available=True but create_parser() failed: %s",
                        config.display_name,
                        e,
                    )
                    config.parser_available = False

            # Register plugin
            cls._plugins[language_name] = plugin

            # Map file extensions to language
            for ext in config.extensions:
                ext_lower = ext.lower()
                if ext_lower in cls._extension_map:
                    existing_lang = cls._extension_map[ext_lower]
                    logger.warning(
                        "Extension %s already mapped to %s, overriding with %s",
                        ext,
                        existing_lang,
                        language_name,
                    )
                cls._extension_map[ext_lower] = language_name

            logger.info(
                "Registered %s plugin for extensions: %s (parser_available=%s)",
                config.display_name,
                config.extensions,
                config.parser_available,
            )

        except Exception as e:
            logger.error("Failed to register plugin %s: %s", type(plugin).__name__, e)
            raise

    @classmethod
    def get_plugin(cls, language: str) -> LanguagePlugin | None:
        """Get plugin by language name.

        Args:
            language: Language name (case-insensitive)

        Returns:
            LanguagePlugin instance or None if not found
        """
        cls._ensure_initialized()
        return cls._plugins.get(language.lower())

    @classmethod
    def get_plugin_by_extension(cls, extension: str) -> LanguagePlugin | None:
        """Get plugin by file extension.

        Args:
            extension: File extension (with or without leading dot)

        Returns:
            LanguagePlugin instance or None if not found
        """
        cls._ensure_initialized()

        # Normalize extension
        ext = extension if extension.startswith(".") else f".{extension}"
        ext_lower = ext.lower()

        language = cls._extension_map.get(ext_lower)
        if language:
            return cls._plugins.get(language)

        return None

    @classmethod
    def get_plugin_by_file_path(cls, file_path: Path) -> LanguagePlugin | None:
        """Get plugin by file path.

        Args:
            file_path: Path to source file

        Returns:
            LanguagePlugin instance or None if not supported
        """
        return cls.get_plugin_by_extension(file_path.suffix)

    @classmethod
    def detect_language(cls, file_path: Path) -> str | None:
        """Detect language from file path.

        Args:
            file_path: Path to source file

        Returns:
            Language name or None if not supported
        """
        cls._ensure_initialized()

        ext_lower = file_path.suffix.lower()
        return cls._extension_map.get(ext_lower)

    @classmethod
    def is_supported(cls, file_path: Path) -> bool:
        """Check if file type is supported.

        Args:
            file_path: Path to check

        Returns:
            True if file type has a registered plugin
        """
        return cls.get_plugin_by_file_path(file_path) is not None

    @classmethod
    def is_language_supported(cls, language: str) -> bool:
        """Check if language is supported.

        Args:
            language: Language name

        Returns:
            True if language has a registered plugin
        """
        return cls.get_plugin(language) is not None

    @classmethod
    def get_supported_languages(cls) -> list[str]:
        """Get list of all supported languages.

        Returns:
            Sorted list of language names
        """
        cls._ensure_initialized()
        return sorted(cls._plugins.keys())

    @classmethod
    def get_supported_extensions(cls) -> list[str]:
        """Get list of all supported file extensions.

        Returns:
            Sorted list of file extensions
        """
        cls._ensure_initialized()
        return sorted(cls._extension_map.keys())

    @classmethod
    def get_plugins_with_feature(cls, feature: str) -> list[LanguagePlugin]:
        """Get plugins that support a specific feature.

        Args:
            feature: Feature name (e.g., 'classes', 'type_hints')

        Returns:
            List of plugins supporting the feature
        """
        cls._ensure_initialized()
        return [
            plugin
            for plugin in cls._plugins.values()
            if plugin.supports_feature(feature)
        ]

    @classmethod
    def get_plugin_info(cls) -> dict[str, dict[str, any]]:
        """Get information about all registered plugins.

        Returns:
            Dictionary mapping language names to plugin info
        """
        cls._ensure_initialized()
        info = {}

        for language, plugin in cls._plugins.items():
            config = plugin.get_language_config()
            info[language] = {
                "display_name": config.display_name,
                "extensions": config.extensions,
                "parser_available": config.parser_available,
                "features": config.features,
                "plugin_class": plugin.__class__.__name__,
            }

        return info

    @classmethod
    def clear_plugins(cls) -> None:
        """Clear all registered plugins (mainly for testing)."""
        cls._plugins.clear()
        cls._extension_map.clear()
        cls._initialized = False

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Ensure plugins are initialized."""
        if not cls._initialized:
            cls._initialize_default_plugins()
            cls._initialized = True

    @classmethod
    def _initialize_default_plugins(cls) -> None:
        """Initialize default plugins."""
        # List of plugin classes to register with error handling
        plugin_classes = [
            ("PythonLanguagePlugin", "src.parser.plugins.python_plugin"),
            ("PHPLanguagePlugin", "src.parser.plugins.php_plugin"),
            ("JavaLanguagePlugin", "src.parser.plugins.java_plugin"),
            ("TypeScriptLanguagePlugin", "src.parser.plugins.typescript_plugin"),
            ("JavaScriptLanguagePlugin", "src.parser.plugins.javascript_plugin"),
        ]

        for plugin_name, module_path in plugin_classes:
            try:
                # Import plugin module
                module = __import__(module_path, fromlist=[plugin_name])
                plugin_class = getattr(module, plugin_name)

                # Create and register plugin instance
                plugin_instance = plugin_class()
                cls.register_plugin(plugin_instance)

            except ImportError as e:
                logger.warning(
                    "Failed to import %s from %s: %s", plugin_name, module_path, e
                )
            except Exception as e:
                logger.error("Failed to register %s plugin: %s", plugin_name, e)

    @classmethod
    def auto_discover_plugins(cls) -> None:
        """Auto-discover plugins from the plugin directory.

        This method provides extensibility for future plugin systems
        where plugins can be loaded dynamically from files.
        """
        # Future enhancement: scan for plugin files and load them
        logger.debug("Auto-discovery not yet implemented")
