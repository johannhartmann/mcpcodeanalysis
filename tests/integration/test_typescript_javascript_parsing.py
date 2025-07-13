"""End-to-end tests for TypeScript and JavaScript parsing."""

import tempfile
from pathlib import Path

import pytest

from src.parser.plugin_registry import LanguagePluginRegistry


class TestTypeScriptJavaScriptParsing:
    """Test TypeScript and JavaScript parsing end-to-end."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_typescript_plugin_basic_functionality(self):
        """Test basic TypeScript plugin functionality."""
        plugin = LanguagePluginRegistry.get_plugin("typescript")
        assert plugin is not None

        # Test configuration
        config = plugin.get_language_config()
        assert config.name == "typescript"
        assert config.display_name == "TypeScript"
        assert ".ts" in config.extensions
        assert ".tsx" in config.extensions
        assert ".d.ts" in config.extensions

        # Test features
        assert config.features["classes"] is True
        assert config.features["functions"] is True
        assert config.features["type_hints"] is True
        assert config.features["imports"] is True

        # Test complexity nodes
        complexity_nodes = plugin.get_complexity_nodes()
        expected_nodes = {
            "if_statement",
            "switch_statement",
            "for_statement",
            "while_statement",
            "catch_clause",
            "interface_declaration",
            "arrow_function",
            "async_function",
        }
        assert expected_nodes.issubset(complexity_nodes)

    def test_javascript_plugin_basic_functionality(self):
        """Test basic JavaScript plugin functionality."""
        plugin = LanguagePluginRegistry.get_plugin("javascript")
        assert plugin is not None

        # Test configuration
        config = plugin.get_language_config()
        assert config.name == "javascript"
        assert config.display_name == "JavaScript"
        assert ".js" in config.extensions
        assert ".jsx" in config.extensions
        assert ".mjs" in config.extensions

        # Test features
        assert config.features["classes"] is True
        assert config.features["functions"] is True
        assert (
            config.features["type_hints"] is False
        )  # JavaScript doesn't have type hints
        assert config.features["imports"] is True

        # Test complexity nodes
        complexity_nodes = plugin.get_complexity_nodes()
        expected_nodes = {
            "if_statement",
            "switch_statement",
            "for_statement",
            "while_statement",
            "catch_clause",
            "arrow_function",
            "async_function",
            "class_declaration",
        }
        assert expected_nodes.issubset(complexity_nodes)

    @pytest.mark.skipif(
        False,  # TreeSitter is base infrastructure and should be available
        reason="TreeSitter libraries should be available",
    )
    def test_typescript_parser_creation_and_usage(self, temp_dir):
        """Test TypeScript parser creation and basic usage."""
        plugin = LanguagePluginRegistry.get_plugin("typescript")

        try:
            parser = plugin.create_parser()
            assert parser is not None
            assert parser.get_language_name() == "typescript"
            assert ".ts" in parser.get_file_extensions()

            # Create a simple TypeScript file
            ts_file = temp_dir / "test.ts"
            ts_content = """
interface User {
    name: string;
    age: number;
}

class UserService {
    private users: User[] = [];

    constructor() {
        this.users = [];
    }

    addUser(user: User): void {
        this.users.push(user);
    }

    getUser(name: string): User | undefined {
        return this.users.find(u => u.name === name);
    }
}

function createUser(name: string, age: number): User {
    return { name, age };
}

export { User, UserService, createUser };
"""
            ts_file.write_text(ts_content)

            # Test parsing
            result = parser.parse_file(ts_file)
            assert result is not None
            assert result.language == "typescript"
            assert result.success

        except ImportError:
            pytest.skip("tree-sitter-typescript not available")

    @pytest.mark.skipif(
        False,  # TreeSitter is base infrastructure and should be available
        reason="TreeSitter libraries should be available",
    )
    def test_javascript_parser_creation_and_usage(self, temp_dir):
        """Test JavaScript parser creation and basic usage."""
        plugin = LanguagePluginRegistry.get_plugin("javascript")

        try:
            parser = plugin.create_parser()
            assert parser is not None
            assert parser.get_language_name() == "javascript"
            assert ".js" in parser.get_file_extensions()

            # Create a simple JavaScript file
            js_file = temp_dir / "test.js"
            js_content = """
class UserService {
    constructor() {
        this.users = [];
    }

    addUser(user) {
        this.users.push(user);
    }

    async getUser(name) {
        return this.users.find(u => u.name === name);
    }

    static create() {
        return new UserService();
    }
}

function createUser(name, age) {
    return { name, age };
}

const getUserById = (id) => {
    return users.find(user => user.id === id);
};

export { UserService, createUser, getUserById };
"""
            js_file.write_text(js_content)

            # Test parsing
            result = parser.parse_file(js_file)
            assert result is not None
            assert result.language == "javascript"
            assert result.success

        except ImportError:
            pytest.skip("tree-sitter-javascript not available")

    def test_typescript_complexity_calculation(self):
        """Test TypeScript complexity calculation."""
        from src.parser.complexity_calculator import ComplexityCalculator

        calculator = ComplexityCalculator("typescript")

        # Verify TypeScript-specific complexity nodes are included
        ts_nodes = {
            "interface_declaration",
            "type_alias_declaration",
            "enum_declaration",
            "optional_chaining_expression",
            "nullish_coalescing_expression",
        }

        for node in ts_nodes:
            assert (
                node in calculator.COMPLEXITY_NODES
            ), f"Missing TypeScript node: {node}"

    def test_javascript_complexity_calculation(self):
        """Test JavaScript complexity calculation."""
        from src.parser.complexity_calculator import ComplexityCalculator

        calculator = ComplexityCalculator("javascript")

        # Verify JavaScript-specific complexity nodes are included
        js_nodes = {
            "arrow_function",
            "async_function",
            "class_declaration",
            "optional_chaining_expression",
            "nullish_coalescing_expression",
            "template_literal",
        }

        for node in js_nodes:
            assert (
                node in calculator.COMPLEXITY_NODES
            ), f"Missing JavaScript node: {node}"

    def test_domain_analysis_feature_detection(self):
        """Test that TypeScript/JavaScript support domain analysis."""
        from src.parser.plugin_registry import LanguagePluginRegistry

        # Both TypeScript and JavaScript should support domain analysis
        # since they have classes and functions

        ts_plugin = LanguagePluginRegistry.get_plugin("typescript")
        assert ts_plugin.supports_feature("classes")
        assert ts_plugin.supports_feature("functions")

        js_plugin = LanguagePluginRegistry.get_plugin("javascript")
        assert js_plugin.supports_feature("classes")
        assert js_plugin.supports_feature("functions")

    def test_code_extractor_integration(self):
        """Test that CodeExtractor can handle TypeScript/JavaScript files."""
        from src.parser.code_extractor import CodeExtractor

        extractor = CodeExtractor()

        # Test file type detection
        test_files = [
            Path("test.ts"),
            Path("test.tsx"),
            Path("test.js"),
            Path("test.jsx"),
            Path("test.mjs"),
        ]

        for file_path in test_files:
            plugin = extractor.plugin_registry.get_plugin_by_file_path(file_path)
            assert plugin is not None, f"No plugin found for {file_path}"

            # Verify language detection
            if file_path.suffix in [".ts", ".tsx"]:
                assert plugin.get_language_name() == "typescript"
            else:
                assert plugin.get_language_name() == "javascript"

    def test_error_handling_missing_libraries(self):
        """Test error handling when TreeSitter libraries are missing."""
        # This test validates that the system degrades gracefully
        # when tree-sitter-typescript or tree-sitter-javascript are not installed

        ts_plugin = LanguagePluginRegistry.get_plugin("typescript")
        js_plugin = LanguagePluginRegistry.get_plugin("javascript")

        # Plugins should exist even if parsers can't be created
        assert ts_plugin is not None
        assert js_plugin is not None

        # Configuration should work
        ts_config = ts_plugin.get_language_config()
        js_config = js_plugin.get_language_config()

        assert ts_config.name == "typescript"
        assert js_config.name == "javascript"

        # Complexity nodes should be available
        assert len(ts_plugin.get_complexity_nodes()) > 0
        assert len(js_plugin.get_complexity_nodes()) > 0

    def test_language_priority_system(self):
        """Test language analysis priority system."""
        ts_plugin = LanguagePluginRegistry.get_plugin("typescript")
        js_plugin = LanguagePluginRegistry.get_plugin("javascript")
        python_plugin = LanguagePluginRegistry.get_plugin("python")

        # TypeScript should have higher priority than JavaScript
        assert ts_plugin.get_analysis_priority() > js_plugin.get_analysis_priority()

        # Python should have highest priority
        assert python_plugin.get_analysis_priority() > ts_plugin.get_analysis_priority()

    def test_file_extension_mappings(self):
        """Test that file extensions map to correct languages."""
        test_cases = [
            (".ts", "typescript"),
            (".tsx", "typescript"),
            (".d.ts", "typescript"),
            (".js", "javascript"),
            (".jsx", "javascript"),
            (".mjs", "javascript"),
        ]

        for extension, expected_lang in test_cases:
            plugin = LanguagePluginRegistry.get_plugin_by_extension(extension)
            assert plugin is not None, f"No plugin for extension {extension}"
            assert (
                plugin.get_language_name() == expected_lang
            ), f"Wrong language for {extension}: expected {expected_lang}, got {plugin.get_language_name()}"
