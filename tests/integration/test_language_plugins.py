"""Integration tests for language plugin system."""

from pathlib import Path

import pytest

from src.parser.plugin_registry import LanguagePluginRegistry


class TestLanguagePlugins:
    """Test language plugin system integration."""

    def test_plugin_registry_initialization(self) -> None:
        """Test that plugin registry initializes correctly."""
        # Clear registry to test initialization
        LanguagePluginRegistry.clear_plugins()

        # Initialize plugins
        LanguagePluginRegistry._ensure_initialized()

        # Check that core languages are registered
        supported_languages = LanguagePluginRegistry.get_supported_languages()
        assert "python" in supported_languages
        assert "php" in supported_languages
        assert "java" in supported_languages

    def test_python_plugin_works(self) -> None:
        """Test that Python plugin works correctly."""
        plugin = LanguagePluginRegistry.get_plugin("python")
        assert plugin is not None

        config = plugin.get_language_config()
        assert config.name == "python"
        assert config.parser_available is True
        assert config.features["classes"] is True
        assert config.features["functions"] is True

        # Test parser creation
        parser = plugin.create_parser()
        assert parser is not None

        # Test complexity nodes
        complexity_nodes = plugin.get_complexity_nodes()
        assert "if_statement" in complexity_nodes
        assert "for_statement" in complexity_nodes

    def test_php_plugin_works(self) -> None:
        """Test that PHP plugin works correctly."""
        plugin = LanguagePluginRegistry.get_plugin("php")
        assert plugin is not None

        config = plugin.get_language_config()
        assert config.name == "php"
        assert config.parser_available is True

        parser = plugin.create_parser()
        assert parser is not None

    def test_java_plugin_works(self) -> None:
        """Test that Java plugin works correctly."""
        plugin = LanguagePluginRegistry.get_plugin("java")
        assert plugin is not None

        config = plugin.get_language_config()
        assert config.name == "java"
        assert config.parser_available is True

        parser = plugin.create_parser()
        assert parser is not None

    def test_typescript_plugin_availability(self) -> None:
        """Test TypeScript plugin availability."""
        plugin = LanguagePluginRegistry.get_plugin("typescript")

        # Plugin should exist even if TreeSitter library is missing
        assert plugin is not None

        config = plugin.get_language_config()
        assert config.name == "typescript"
        assert ".ts" in config.extensions
        assert ".tsx" in config.extensions

        # Test complexity nodes are defined
        complexity_nodes = plugin.get_complexity_nodes()
        assert "if_statement" in complexity_nodes
        assert "interface_declaration" in complexity_nodes

    def test_javascript_plugin_availability(self) -> None:
        """Test JavaScript plugin availability."""
        plugin = LanguagePluginRegistry.get_plugin("javascript")

        # Plugin should exist even if TreeSitter library is missing
        assert plugin is not None

        config = plugin.get_language_config()
        assert config.name == "javascript"
        assert ".js" in config.extensions
        assert ".jsx" in config.extensions

        # Test complexity nodes are defined
        complexity_nodes = plugin.get_complexity_nodes()
        assert "if_statement" in complexity_nodes
        assert "arrow_function" in complexity_nodes

    def test_file_path_language_detection(self) -> None:
        """Test language detection from file paths."""
        # Test extension-based detection
        test_cases = [
            (Path("test.py"), "python"),
            (Path("test.php"), "php"),
            (Path("test.java"), "java"),
            (Path("test.ts"), "typescript"),
            (Path("test.tsx"), "typescript"),
            (Path("test.js"), "javascript"),
            (Path("test.jsx"), "javascript"),
            (Path("test.mjs"), "javascript"),
        ]

        for file_path, expected_lang in test_cases:
            detected = LanguagePluginRegistry.detect_language(file_path)
            assert (
                detected == expected_lang
            ), f"Failed for {file_path}: expected {expected_lang}, got {detected}"

    def test_plugin_by_file_path(self) -> None:
        """Test getting plugins by file path."""
        test_cases = [
            (Path("test.py"), "python"),
            (Path("test.php"), "php"),
            (Path("test.java"), "java"),
        ]

        for file_path, expected_lang in test_cases:
            plugin = LanguagePluginRegistry.get_plugin_by_file_path(file_path)
            assert plugin is not None
            assert plugin.get_language_name() == expected_lang

    def test_feature_based_capabilities(self) -> None:
        """Test feature-based language capabilities."""
        # Test that OOP languages support domain analysis
        oop_languages = ["python", "php", "java", "typescript", "javascript"]

        for lang in oop_languages:
            plugin = LanguagePluginRegistry.get_plugin(lang)
            if plugin:  # May not be available if TreeSitter libs missing
                assert plugin.supports_feature(
                    "classes"
                ), f"{lang} should support classes"
                assert plugin.supports_feature(
                    "functions"
                ), f"{lang} should support functions"

    def test_unsupported_language(self) -> None:
        """Test handling of unsupported languages."""
        plugin = LanguagePluginRegistry.get_plugin("nonexistent")
        assert plugin is None

        plugin = LanguagePluginRegistry.get_plugin_by_extension(".xyz")
        assert plugin is None

    def test_plugin_info_retrieval(self) -> None:
        """Test plugin information retrieval."""
        info = LanguagePluginRegistry.get_plugin_info()

        assert isinstance(info, dict)
        assert "python" in info

        python_info = info["python"]
        assert "display_name" in python_info
        assert "extensions" in python_info
        assert "parser_available" in python_info
        assert python_info["display_name"] == "Python"

    @pytest.mark.skipif(
        False,  # TreeSitter is base infrastructure and should be available
        reason="TreeSitter libraries should be available",
    )
    def test_typescript_parser_creation(self) -> None:
        """Test TypeScript parser creation (requires TreeSitter library)."""
        plugin = LanguagePluginRegistry.get_plugin("typescript")
        assert plugin is not None
        parser = plugin.create_parser()
        assert parser is not None
        assert parser.get_language_name() == "typescript"

    @pytest.mark.skipif(
        False,  # TreeSitter is base infrastructure and should be available
        reason="TreeSitter libraries should be available",
    )
    def test_javascript_parser_creation(self) -> None:
        """Test JavaScript parser creation (requires TreeSitter library)."""
        plugin = LanguagePluginRegistry.get_plugin("javascript")
        assert plugin is not None
        parser = plugin.create_parser()
        assert parser is not None
        assert parser.get_language_name() == "javascript"

    def test_complexity_calculator_plugin_integration(self) -> None:
        """Test that ComplexityCalculator uses plugin system."""
        from src.parser.complexity_calculator import ComplexityCalculator

        # Test that calculator gets complexity nodes from plugins
        calc_python = ComplexityCalculator("python")
        assert hasattr(calc_python, "COMPLEXITY_NODES")
        assert "if_statement" in calc_python.COMPLEXITY_NODES

        calc_ts = ComplexityCalculator("typescript")
        assert "interface_declaration" in calc_ts.COMPLEXITY_NODES

        calc_js = ComplexityCalculator("javascript")
        assert "arrow_function" in calc_js.COMPLEXITY_NODES

    def test_code_extractor_plugin_integration(self) -> None:
        """Test that CodeExtractor uses plugin registry."""
        from src.parser.code_extractor import CodeExtractor

        extractor = CodeExtractor()

        # Test that extractor uses plugin registry
        assert hasattr(extractor, "plugin_registry")

        # Test file support check
        assert extractor.plugin_registry.is_supported(Path("test.py"))
        assert extractor.plugin_registry.is_supported(Path("test.php"))
        assert extractor.plugin_registry.is_supported(Path("test.java"))

    def test_parser_factory_plugin_integration(self) -> None:
        """Test that ParserFactory uses plugin registry."""
        from src.parser.parser_factory import ParserFactory

        # Test support detection
        assert ParserFactory.is_supported(Path("test.py"))
        assert ParserFactory.is_supported(Path("test.php"))
        assert ParserFactory.is_supported(Path("test.java"))

        # Test language support
        assert ParserFactory.is_language_supported("python")
        assert ParserFactory.is_language_supported("php")
        assert ParserFactory.is_language_supported("java")

        # Test extension and language lists
        extensions = ParserFactory.get_supported_extensions()
        assert ".py" in extensions
        assert ".php" in extensions
        assert ".java" in extensions

        languages = ParserFactory.get_supported_languages()
        assert "python" in languages
        assert "php" in languages
        assert "java" in languages
