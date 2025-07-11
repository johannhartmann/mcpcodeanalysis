"""Tests for language configuration."""

from src.parser.language_config import LanguageConfig, LanguageRegistry


class TestLanguageConfig:
    """Tests for LanguageConfig class."""

    def test_language_config_creation(self) -> None:
        """Test creating a language configuration."""
        config = LanguageConfig(
            name="test",
            display_name="Test Language",
            extensions=[".test", ".tst"],
            parser_available=True,
        )

        assert config.name == "test"
        assert config.display_name == "Test Language"
        assert config.extensions == [".test", ".tst"]
        assert config.parser_available is True
        assert config.features["classes"] is False
        assert config.features["functions"] is False

    def test_language_config_with_features(self) -> None:
        """Test creating a language configuration with features."""
        features = {
            "classes": True,
            "functions": True,
            "imports": False,
            "modules": True,
            "docstrings": False,
            "type_hints": True,
        }

        config = LanguageConfig(
            name="test",
            display_name="Test Language",
            extensions=[".test"],
            features=features,
        )

        assert config.features == features


class TestLanguageRegistry:
    """Tests for LanguageRegistry class."""

    def test_get_language(self) -> None:
        """Test getting a language by name."""
        python = LanguageRegistry.get_language("python")
        assert python is not None
        assert python.name == "python"
        assert python.display_name == "Python"
        assert ".py" in python.extensions

        # Test case insensitive
        python_upper = LanguageRegistry.get_language("PYTHON")
        assert python_upper is not None
        assert python_upper.name == "python"

        # Test non-existent language
        unknown = LanguageRegistry.get_language("unknown")
        assert unknown is None

    def test_get_language_by_extension(self) -> None:
        """Test getting a language by file extension."""
        python = LanguageRegistry.get_language_by_extension(".py")
        assert python is not None
        assert python.name == "python"

        # Test case insensitive
        python_upper = LanguageRegistry.get_language_by_extension(".PY")
        assert python_upper is not None
        assert python_upper.name == "python"

        # Test non-existent extension
        unknown = LanguageRegistry.get_language_by_extension(".unknown")
        assert unknown is None

    def test_get_supported_languages(self) -> None:
        """Test getting list of supported languages."""
        languages = LanguageRegistry.get_supported_languages()
        assert isinstance(languages, list)
        assert "python" in languages
        assert "javascript" in languages
        assert "typescript" in languages
        assert len(languages) >= 10  # We defined at least 10 languages

    def test_get_available_languages(self) -> None:
        """Test getting list of available languages."""
        available = LanguageRegistry.get_available_languages()
        assert isinstance(available, list)
        # By default, only Python should be available
        assert "python" in available
        assert len(available) >= 1

    def test_get_supported_extensions(self) -> None:
        """Test getting all supported extensions."""
        extensions = LanguageRegistry.get_supported_extensions()
        assert isinstance(extensions, set)
        assert ".py" in extensions
        assert ".js" in extensions
        assert ".ts" in extensions
        assert ".java" in extensions
        assert len(extensions) >= 20  # Multiple extensions across languages

    def test_get_available_extensions(self) -> None:
        """Test getting extensions for available parsers."""
        extensions = LanguageRegistry.get_available_extensions()
        assert isinstance(extensions, set)
        # By default, only Python extensions should be available
        assert ".py" in extensions
        assert ".pyw" in extensions
        assert ".pyi" in extensions

    def test_register_language(self) -> None:
        """Test registering a new language."""
        # Create a copy of the registry to avoid side effects
        original_languages = LanguageRegistry._languages.copy()

        try:
            # Register a new language
            config = LanguageConfig(
                name="newlang",
                display_name="New Language",
                extensions=[".new", ".nlg"],
                parser_available=True,
            )

            LanguageRegistry.register_language(config)

            # Check it was registered
            newlang = LanguageRegistry.get_language("newlang")
            assert newlang is not None
            assert newlang.display_name == "New Language"
            assert newlang.extensions == [".new", ".nlg"]

            # Check extensions are available
            assert LanguageRegistry.get_language_by_extension(".new") == newlang
            assert LanguageRegistry.get_language_by_extension(".nlg") == newlang

        finally:
            # Restore original registry
            LanguageRegistry._languages = original_languages

    def test_update_language(self) -> None:
        """Test updating an existing language."""
        # Create a copy of the registry to avoid side effects
        original_languages = LanguageRegistry._languages.copy()

        try:
            # Update Python to mark it as unavailable (for testing)
            LanguageRegistry.update_language("python", parser_available=False)

            python = LanguageRegistry.get_language("python")
            assert python is not None
            assert python.parser_available is False

            # Update with multiple attributes
            LanguageRegistry.update_language(
                "python",
                parser_available=True,
                display_name="Python Programming Language",
            )

            python = LanguageRegistry.get_language("python")
            assert python.parser_available is True
            assert python.display_name == "Python Programming Language"

            # Try updating non-existent language
            LanguageRegistry.update_language("nonexistent", parser_available=True)
            # Should not raise an error

        finally:
            # Restore original registry
            LanguageRegistry._languages = original_languages

    def test_is_extension_supported(self) -> None:
        """Test checking if an extension is supported."""
        assert LanguageRegistry.is_extension_supported(".py") is True
        assert (
            LanguageRegistry.is_extension_supported(".PY") is True
        )  # Case insensitive
        assert LanguageRegistry.is_extension_supported(".js") is True
        assert LanguageRegistry.is_extension_supported(".unknown") is False

    def test_is_extension_available(self) -> None:
        """Test checking if an extension has an available parser."""
        assert LanguageRegistry.is_extension_available(".py") is True
        assert (
            LanguageRegistry.is_extension_available(".PY") is True
        )  # Case insensitive
        # JavaScript parser not available by default
        assert LanguageRegistry.is_extension_available(".js") is False
        assert LanguageRegistry.is_extension_available(".unknown") is False
