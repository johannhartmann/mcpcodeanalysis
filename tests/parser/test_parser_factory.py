"""Tests for parser factory."""

from pathlib import Path

from src.parser.parser_factory import ParserFactory
from src.parser.python_parser import PythonCodeParser


class TestParserFactory:
    """Tests for ParserFactory class."""

    def test_create_parser_python(self) -> None:
        """Test creating Python parser."""
        parser = ParserFactory.create_parser(Path("test.py"))
        assert parser is not None
        assert isinstance(parser, PythonCodeParser)

        # Test other Python extensions
        parser = ParserFactory.create_parser(Path("test.pyw"))
        assert isinstance(parser, PythonCodeParser)

        parser = ParserFactory.create_parser(Path("test.pyi"))
        assert isinstance(parser, PythonCodeParser)

    def test_create_parser_unsupported(self) -> None:
        """Test creating parser for unsupported file type."""
        parser = ParserFactory.create_parser(Path("test.txt"))
        assert parser is None

        parser = ParserFactory.create_parser(Path("test.java"))
        assert parser is None

    def test_create_parser_by_language(self) -> None:
        """Test creating parser by language name."""
        parser = ParserFactory.create_parser_by_language("python")
        assert parser is not None
        assert isinstance(parser, PythonCodeParser)

        # Test case insensitive
        parser = ParserFactory.create_parser_by_language("Python")
        assert isinstance(parser, PythonCodeParser)

        parser = ParserFactory.create_parser_by_language("PYTHON")
        assert isinstance(parser, PythonCodeParser)

    def test_create_parser_by_language_unsupported(self) -> None:
        """Test creating parser for unsupported language."""
        parser = ParserFactory.create_parser_by_language("java")
        assert parser is None

        parser = ParserFactory.create_parser_by_language("unknown")
        assert parser is None

    def test_is_supported(self) -> None:
        """Test checking if file type is supported."""
        # Ensure plugin registry state does not affect this unit test
        from src.parser.plugin_registry import LanguagePluginRegistry

        LanguagePluginRegistry.clear_plugins()

        assert ParserFactory.is_supported(Path("test.py"))
        assert ParserFactory.is_supported(Path("test.pyw"))
        assert ParserFactory.is_supported(Path("test.pyi"))
        assert not ParserFactory.is_supported(Path("test.txt"))
        assert not ParserFactory.is_supported(Path("test.java"))

    def test_is_language_supported(self) -> None:
        """Test checking if language is supported."""
        assert ParserFactory.is_language_supported("python")
        assert ParserFactory.is_language_supported("Python")
        assert not ParserFactory.is_language_supported("java")
        assert not ParserFactory.is_language_supported("unknown")

    def test_get_supported_extensions(self) -> None:
        """Test getting supported file extensions."""
        extensions = ParserFactory.get_supported_extensions()
        assert ".py" in extensions
        assert ".pyw" in extensions
        assert ".pyi" in extensions
        assert len(extensions) >= 3

    def test_get_supported_languages(self) -> None:
        """Test getting supported languages."""
        languages = ParserFactory.get_supported_languages()
        assert "python" in languages
        assert len(languages) >= 1

    def test_register_parser(self) -> None:
        """Test registering a new parser."""

        # Create mock parser class
        class MockParser:
            pass

        # Register for new extension
        ParserFactory.register_parser(".mock", MockParser)

        # Test that it's registered
        assert ParserFactory.is_supported(Path("test.mock"))
        parser = ParserFactory.create_parser(Path("test.mock"))
        assert isinstance(parser, MockParser)

        # Clean up
        del ParserFactory._parsers[".mock"]

    def test_register_language_parser(self) -> None:
        """Test registering a new language parser."""

        # Create mock parser class
        class MockLanguageParser:
            pass

        # Register for new language
        ParserFactory.register_language_parser("mocklang", MockLanguageParser)

        # Test that it's registered
        assert ParserFactory.is_language_supported("mocklang")
        parser = ParserFactory.create_parser_by_language("mocklang")
        assert isinstance(parser, MockLanguageParser)

        # Clean up
        del ParserFactory._language_parsers["mocklang"]
