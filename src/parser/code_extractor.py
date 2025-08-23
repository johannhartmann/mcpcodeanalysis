"""Code entity extractor for building structured representations."""

from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

from src.logger import get_logger

logger = get_logger(__name__)

# Display limits
MAX_DISPLAY_METHODS = 10
MAX_DISPLAY_CLASSES = 5
MAX_DISPLAY_FUNCTIONS = 5


@runtime_checkable
class _ParserProtocol(Protocol):
    """Minimal parser protocol used by CodeExtractor.

    We intentionally keep this lightweight so tests can patch methods
    without requiring heavy parser dependencies to be imported/initialized.
    """

    def extract_entities(
        self, file_path: Path, file_id: int
    ) -> dict[str, list[Any]]: ...

    def get_code_chunk(
        self,
        file_path: Path,
        start_line: int,
        end_line: int,
        context_lines: int = 0,
    ) -> str: ...


class _MiniPlugin:
    """Minimal plugin object exposing only get_language_name()."""

    def __init__(self, language: str) -> None:
        self._language = language

    def get_language_name(self) -> str:  # pragma: no cover - trivial
        return self._language


class _LightweightRegistry:
    """Tiny registry to satisfy integration points without heavy imports.

    This avoids initializing the full LanguagePluginRegistry (TreeSitter, etc.)
    in scopes/tests that only need basic language detection for file suffixes.
    """

    _ext_to_lang: ClassVar[dict[str, str]] = {
        ".py": "python",
        ".pyw": "python",
        ".pyi": "python",
        ".php": "php",
        ".java": "java",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
        ".mjs": "javascript",
    }

    def is_supported(self, file_path: Path) -> bool:
        return file_path.suffix in self._ext_to_lang

    def get_plugin_by_file_path(self, file_path: Path) -> _MiniPlugin | None:
        lang = self._ext_to_lang.get(file_path.suffix)
        return _MiniPlugin(lang) if lang else None

    def get_plugin_by_extension(self, extension: str) -> _MiniPlugin | None:
        ext = extension if extension.startswith(".") else f".{extension}"
        lang = self._ext_to_lang.get(ext)
        return _MiniPlugin(lang) if lang else None


class _DefaultPythonParser:
    """Lazy adapter around the Python parser implementing the protocol.

    Methods import the heavy implementation on demand to avoid import-time
    failures in environments without TreeSitter artifacts. Tests patch these
    methods directly, so they often won't execute.
    """

    def extract_entities(self, file_path: Path, file_id: int) -> dict[str, list[Any]]:
        from src.parser.python_parser import PythonCodeParser

        return PythonCodeParser().extract_entities(file_path, file_id)

    def get_code_chunk(
        self,
        file_path: Path,
        start_line: int,
        end_line: int,
        context_lines: int = 0,
    ) -> str:
        from src.parser.python_parser import PythonCodeParser

        return PythonCodeParser().get_code_chunk(
            file_path, start_line, end_line, context_lines
        )


class CodeExtractor:
    """Extract and structure code entities for analysis."""

    def __init__(self) -> None:
        # Parsers keyed by file suffix (e.g., ".py")
        self.parsers: dict[str, _ParserProtocol] = {
            ".py": _DefaultPythonParser(),
        }
        # Expose a lightweight registry for integration points/tests
        self.plugin_registry = _LightweightRegistry()

    # Lightweight helpers used by aggregator tests
    def _read_lines(self, file_path: Path, start_line: int, end_line: int) -> str:
        """Safely read a range of lines from a file (1-indexed, inclusive)."""
        try:
            with file_path.open("r", encoding="utf-8") as f:
                lines = f.readlines()
            s = max(1, int(start_line)) - 1
            e = max(s, int(end_line))
            return "".join(lines[s:e])
        except Exception:
            logger.exception("Failed to read code from %s", file_path)
            return ""

    def extract_function_code(
        self, file_path: Path, start_line: int, end_line: int
    ) -> str:
        """Extract raw function code block from file.

        Falls back to simple line slicing if language-specific parser isn't available.
        """
        return self._read_lines(file_path, start_line, end_line)

    def extract_class_code(
        self, file_path: Path, start_line: int, end_line: int
    ) -> str:
        """Extract raw class code block from file.

        Falls back to simple line slicing if language-specific parser isn't available.
        """
        return self._read_lines(file_path, start_line, end_line)

    def extract_from_file(
        self,
        file_path: Path,
        file_id: int,
    ) -> dict[str, list[Any]] | None:
        """Extract code entities from a file."""
        suffix = file_path.suffix
        parser = self.parsers.get(suffix)
        if parser is None:
            logger.warning("No parser available for file type: %s", suffix)
            return None

        try:
            return parser.extract_entities(file_path, file_id)
        except Exception:
            logger.exception("Failed to extract entities from %s", file_path)
            return None

    def get_entity_content(
        self,
        file_path: Path,
        entity_type: str,
        start_line: int,
        end_line: int,
        *,
        include_context: bool = True,
    ) -> tuple[str, str]:
        """Get the raw and contextual content for a code entity."""
        suffix = file_path.suffix

        parser = self.parsers.get(suffix)
        if parser is None:
            return "", ""

        # Get raw content
        raw_content = parser.get_code_chunk(file_path, start_line, end_line)

        if not include_context:
            return raw_content, raw_content

        # Get contextual content based on entity type
        context_lines = 3 if entity_type == "function" else 5
        contextual_content = parser.get_code_chunk(
            file_path,
            start_line,
            end_line,
            context_lines,
        )

        return raw_content, contextual_content

    def build_entity_description(
        self,
        entity_type: str,
        entity_data: dict[str, Any],
        file_path: Path,
    ) -> str:
        """Build a natural language description of a code entity."""
        if entity_type == "module":
            return self._describe_module(entity_data, file_path)
        if entity_type == "class":
            return self._describe_class(entity_data)
        if entity_type == "function":
            return self._describe_function(entity_data)
        return f"A {entity_type} named {entity_data.get('name', 'unknown')}"

    def _describe_module(self, module_data: dict[str, Any], file_path: Path) -> str:
        """Build description for a module."""
        # Determine language from file extension
        language = "Python"
        if file_path.suffix == ".php":
            language = "PHP"
        elif file_path.suffix == ".java":
            language = "Java"

        parts = [f"{language} module '{module_data['name']}' from {file_path}"]

        if module_data.get("docstring"):
            parts.append(f"Purpose: {module_data['docstring'][:200]}")

        return ". ".join(parts)

    def _describe_class(self, class_data: dict[str, Any]) -> str:
        """Build description for a class."""
        parts = [f"Class '{class_data['name']}'"]

        if class_data.get("base_classes"):
            parts.append(f"inherits from {', '.join(class_data['base_classes'])}")

        if class_data.get("is_abstract"):
            parts.append("(abstract)")

        if class_data.get("docstring"):
            parts.append(f"Purpose: {class_data['docstring'][:200]}")

        return ". ".join(parts)

    def _describe_function(self, func_data: dict[str, Any]) -> str:
        """Build description for a function."""
        func_type = "Method" if func_data.get("class_name") else "Function"
        parts = [f"{func_type} '{func_data['name']}'"]

        # Add parameter info
        params = func_data.get("parameters", [])
        if params:
            param_names = [p["name"] for p in params if p.get("name")]
            parts.append(f"with parameters: {', '.join(param_names)}")

        # Add return type if available
        if func_data.get("return_type"):
            parts.append(f"returns {func_data['return_type']}")

        # Add special properties
        properties = []
        if func_data.get("is_async"):
            properties.append("async")
        if func_data.get("is_generator"):
            properties.append("generator")
        if func_data.get("is_property"):
            properties.append("property")
        if func_data.get("is_staticmethod"):
            properties.append("static method")
        if func_data.get("is_classmethod"):
            properties.append("class method")

        if properties:
            parts.append(f"({', '.join(properties)})")

        # Add docstring excerpt
        if func_data.get("docstring"):
            parts.append(f"Purpose: {func_data['docstring'][:200]}")

        return ". ".join(parts)

    def aggregate_class_info(
        self,
        class_data: dict[str, Any],
        methods: list[dict[str, Any]],
    ) -> str:
        """Aggregate information about a class and its methods."""
        parts = [self._describe_class(class_data)]

        if methods:
            parts.append(f"\n\nMethods ({len(methods)}):")
            for method in methods[:MAX_DISPLAY_METHODS]:  # Limit to first 10 methods
                parts.append(f"- {self._describe_function(method)}")

            if len(methods) > MAX_DISPLAY_METHODS:
                parts.append(
                    f"... and {len(methods) - MAX_DISPLAY_METHODS} more methods"
                )

        return "\n".join(parts)

    def aggregate_module_info(
        self,
        module_data: dict[str, Any],
        classes: list[dict[str, Any]],
        functions: list[dict[str, Any]],
        file_path: Path,
    ) -> str:
        """Aggregate information about a module."""
        parts = [self._describe_module(module_data, file_path)]

        if classes:
            parts.append(f"\n\nClasses ({len(classes)}):")
            for cls in classes[:MAX_DISPLAY_CLASSES]:  # Limit to first 5 classes
                parts.append(f"- {self._describe_class(cls)}")

            if len(classes) > MAX_DISPLAY_CLASSES:
                parts.append(
                    f"... and {len(classes) - MAX_DISPLAY_CLASSES} more classes"
                )

        if functions:
            parts.append(f"\n\nFunctions ({len(functions)}):")
            for func in functions[:MAX_DISPLAY_FUNCTIONS]:  # Limit to first 5 functions
                parts.append(f"- {self._describe_function(func)}")

            if len(functions) > MAX_DISPLAY_FUNCTIONS:
                parts.append(
                    f"... and {len(functions) - MAX_DISPLAY_FUNCTIONS} more functions"
                )

        return "\n".join(parts)
