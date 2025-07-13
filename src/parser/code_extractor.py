"""Code entity extractor for building structured representations."""

from pathlib import Path
from typing import Any

from src.logger import get_logger
from src.parser.java_parser import JavaCodeParser
from src.parser.php_parser import PHPCodeParser
from src.parser.python_parser import PythonCodeParser

logger = get_logger(__name__)

# Display limits
MAX_DISPLAY_METHODS = 10
MAX_DISPLAY_CLASSES = 5
MAX_DISPLAY_FUNCTIONS = 5


class CodeExtractor:
    """Extract and structure code entities for analysis."""

    def __init__(self) -> None:
        self.parsers = {
            ".py": PythonCodeParser(),
            ".php": PHPCodeParser(),
            ".java": JavaCodeParser(),
        }

    def extract_from_file(
        self,
        file_path: Path,
        file_id: int,
    ) -> dict[str, list[Any]] | None:
        """Extract code entities from a file."""
        suffix = file_path.suffix

        if suffix not in self.parsers:
            logger.warning("No parser available for file type: %s", suffix)
            return None

        parser = self.parsers[suffix]

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

        if suffix not in self.parsers:
            return "", ""

        parser = self.parsers[suffix]

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
