"""Python code parser using TreeSitter."""

from pathlib import Path
from typing import Any

from src.parser.treesitter_parser import PythonParser as TreeSitterPythonParser
from src.utils.exceptions import ParserError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PythonCodeParser:
    """Parser for Python code files."""

    def __init__(self) -> None:
        self.parser = TreeSitterPythonParser()

    def parse_file(self, file_path: Path) -> dict[str, Any]:
        """Parse a Python file and extract all code entities."""
        try:
            # Read file content
            with file_path.open("rb") as f:
                content = f.read()

            # Parse with TreeSitter
            tree = self.parser.parse_content(content)
            if not tree:
                msg = f"Failed to parse file: {file_path}"
                raise ParserError(msg, str(file_path))  # noqa: TRY301

            # Extract module information
            module_info = self.parser.extract_module_info(tree, content)

            # Add file metadata
            module_info["file_path"] = str(file_path)
            module_info["file_name"] = file_path.name
            module_info["module_name"] = file_path.stem

        except Exception as e:
            logger.exception("Error parsing Python file %s", file_path)
            msg = f"Failed to parse Python file: {file_path}"
            raise ParserError(
                msg,
                str(file_path),
            ) from e
        else:
            return module_info

    def extract_entities(self, file_path: Path, file_id: int) -> dict[str, list[Any]]:
        """Extract all entities from a Python file for database storage."""
        module_info = self.parse_file(file_path)

        entities = {
            "modules": [],
            "classes": [],
            "functions": [],
            "imports": [],
        }

        # Create module entity
        module_data = {
            "file_id": file_id,
            "name": module_info["module_name"],
            "docstring": module_info["docstring"],
            "start_line": 1,
            "end_line": self._count_lines(file_path),
        }
        entities["modules"].append(module_data)

        # Process imports
        for import_info in module_info["imports"]:
            import_data = {
                "file_id": file_id,
                "import_statement": import_info["import_statement"],
                "imported_from": import_info["imported_from"],
                "imported_names": import_info["imported_names"],
                "is_relative": import_info["is_relative"],
                "level": import_info["level"],
                "line_number": import_info["line_number"],
            }
            entities["imports"].append(import_data)

        # Process module-level functions
        for func_info in module_info["functions"]:
            func_data = self._process_function(func_info)
            entities["functions"].append(func_data)

        # Process classes and their methods
        for class_info in module_info["classes"]:
            class_data = {
                "name": class_info["name"],
                "docstring": class_info["docstring"],
                "base_classes": class_info["base_classes"],
                "decorators": class_info["decorators"],
                "start_line": class_info["start_line"],
                "end_line": class_info["end_line"],
                "is_abstract": class_info["is_abstract"],
            }
            entities["classes"].append(class_data)

            # Process methods
            for method_info in class_info["methods"]:
                method_data = self._process_function(method_info)
                method_data["class_name"] = class_info["name"]
                entities["functions"].append(method_data)

        return entities

    def _process_function(self, func_info: dict[str, Any]) -> dict[str, Any]:
        """Process function information for database storage."""
        return {
            "name": func_info["name"],
            "parameters": func_info["parameters"],
            "return_type": func_info["return_type"],
            "docstring": func_info["docstring"],
            "decorators": func_info["decorators"],
            "is_async": func_info["is_async"],
            "is_generator": func_info["is_generator"],
            "is_property": func_info["is_property"],
            "is_staticmethod": func_info["is_staticmethod"],
            "is_classmethod": func_info["is_classmethod"],
            "start_line": func_info["start_line"],
            "end_line": func_info["end_line"],
            "complexity": self._calculate_complexity(func_info),
        }

    def _calculate_complexity(self, func_info: dict[str, Any]) -> int:
        """Calculate cyclomatic complexity of a function."""
        # Simple approximation based on function size
        # TODO(@dev): Implement proper cyclomatic complexity calculation
        lines = func_info["end_line"] - func_info["start_line"] + 1
        return max(1, lines // 10)

    def _count_lines(self, file_path: Path) -> int:
        """Count lines in a file."""
        with file_path.open(encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)

    def get_code_chunk(
        self,
        file_path: Path,
        start_line: int,
        end_line: int,
        context_lines: int = 0,
    ) -> str:
        """Get a chunk of code from a file with optional context."""
        try:
            with file_path.open(encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            # Adjust for context
            start_idx = max(0, start_line - 1 - context_lines)
            end_idx = min(len(lines), end_line + context_lines)

            return "".join(lines[start_idx:end_idx])
        except Exception:
            logger.exception("Error getting code chunk from %s", file_path)
            return ""
