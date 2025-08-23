"""JavaScript code parser using TreeSitter."""

from pathlib import Path
from typing import Any

try:
    import tree_sitter_javascript as tsjavascript

    JAVASCRIPT_AVAILABLE = True
except ImportError:
    JAVASCRIPT_AVAILABLE = False

import tree_sitter
from tree_sitter import Tree

from src.logger import get_logger
from src.parser.base_parser import BaseParser, ElementType, ParsedElement
from src.parser.treesitter_parser import JavaScriptParser as TreeSitterJavaScriptParser
from src.utils.exceptions import ParsingError

logger = get_logger(__name__)


class JavaScriptCodeParser(BaseParser):
    """Parser for JavaScript code files."""

    def __init__(self) -> None:
        if not JAVASCRIPT_AVAILABLE:
            msg = "tree-sitter-javascript not available"
            raise ImportError(msg)

        super().__init__(tree_sitter.Language(tsjavascript.language()))
        self.js_parser = TreeSitterJavaScriptParser()

    def get_language_name(self) -> str:
        """Get the name of the programming language."""
        return "javascript"

    def get_file_extensions(self) -> set[str]:
        """Get supported file extensions."""
        return {".js", ".jsx", ".mjs"}

    def _extract_elements(
        self, tree: Tree, content: str, file_path: Path
    ) -> ParsedElement:
        """Extract code elements from JavaScript parse tree."""
        # Create root module element
        root = ParsedElement(
            type=ElementType.MODULE,
            name=file_path.stem,
            start_line=1,
            end_line=len(content.splitlines()) if content else 1,
            start_column=0,
            end_column=0,
            text=content,
        )

        if not tree or not tree.root_node:
            return root

        try:
            # Use TreeSitter parser to extract module info
            module_info = self.js_parser.extract_module_info(tree, content.encode())

            # Extract classes
            for cls_info in module_info.get("classes", []):
                cls_element = ParsedElement(
                    type=ElementType.CLASS,
                    name=cls_info["name"],
                    start_line=cls_info["start_line"],
                    end_line=cls_info["end_line"],
                    start_column=0,
                    end_column=0,
                    text=cls_info.get("text", ""),
                    metadata={
                        "base_classes": cls_info.get("base_classes", []),
                        "access_modifier": "public",  # JavaScript doesn't have access modifiers
                    },
                )
                root.add_child(cls_element)

                # Add methods to class
                for method_info in cls_info.get("methods", []):
                    method_element = ParsedElement(
                        type=ElementType.METHOD,
                        name=method_info["name"],
                        start_line=method_info["start_line"],
                        end_line=method_info["end_line"],
                        start_column=0,
                        end_column=0,
                        text=method_info.get("text", ""),
                        metadata={
                            "parameters": method_info.get("parameters", []),
                            "is_async": method_info.get("is_async", False),
                            "access_modifier": "public",
                        },
                    )
                    cls_element.add_child(method_element)

            # Extract standalone functions
            for func_info in module_info.get("functions", []):
                func_element = ParsedElement(
                    type=ElementType.FUNCTION,
                    name=func_info["name"],
                    start_line=func_info["start_line"],
                    end_line=func_info["end_line"],
                    start_column=0,
                    end_column=0,
                    text=func_info.get("text", ""),
                    metadata={
                        "parameters": func_info.get("parameters", []),
                        "is_async": func_info.get("is_async", False),
                    },
                )
                root.add_child(func_element)

        except (AttributeError, ValueError, TypeError) as e:
            logger.warning(
                "Error extracting JavaScript elements from %s: %s", file_path, e
            )

        return root

    def _extract_imports(self, tree: Tree, content: str) -> list[str]:
        """Extract import statements from JavaScript parse tree."""
        imports = []

        if not tree or not tree.root_node:
            return imports

        try:
            # Use TreeSitter parser to extract module info
            module_info = self.js_parser.extract_module_info(tree, content.encode())

            # Convert import info to strings
            for imp_info in module_info.get("imports", []):
                module_name = imp_info.get("module", "")
                imported_items = imp_info.get("imports", [])
                is_default = imp_info.get("is_default", False)
                is_namespace = imp_info.get("is_namespace", False)

                if is_namespace:
                    imports.append(
                        f"import * as {imported_items[0] if imported_items else 'ns'} from {module_name}"
                    )
                elif is_default:
                    imports.append(
                        f"import {imported_items[0] if imported_items else 'default'} from {module_name}"
                    )
                elif imported_items:
                    imports.append(
                        f"import {{ {', '.join(imported_items)} }} from {module_name}"
                    )
                else:
                    imports.append(f"import {module_name}")

        except (AttributeError, ValueError, TypeError) as e:
            logger.warning("Error extracting JavaScript imports: %s", e)

        return imports

    def _extract_references(self, tree: Tree, content: str) -> list[dict[str, Any]]:
        """Extract code references from JavaScript parse tree."""
        references = []

        if not tree or not tree.root_node:
            return references

        try:
            # Get references from the TreeSitter JavaScript parser
            js_references = self.js_parser.extract_references(tree, content.encode())

            # Convert TreeSitter references to standard format
            for ref in js_references:
                reference = {
                    "type": ref.get("type", "unknown"),
                    "source": ref.get("source", ""),
                    "target": ref.get("target", ""),
                    "line": ref.get("line", 1),
                    "column": ref.get("column", 0),
                    "module": ref.get("module", ""),
                }
                references.append(reference)

        except (AttributeError, ValueError, TypeError) as e:
            logger.warning("Error extracting JavaScript references: %s", e)

        return references

    def extract_entities(self, file_path: Path, file_id: int) -> dict[str, list[Any]]:
        """Extract code entities for database storage."""
        try:
            # Use the new BaseParser interface to parse
            parse_result = self.parse_file(file_path)

            if not parse_result.success:
                logger.warning(
                    "Failed to parse JavaScript file %s: %s",
                    file_path,
                    parse_result.errors,
                )
                return {"modules": [], "classes": [], "functions": [], "imports": []}

            entities = {
                "modules": [
                    {
                        "name": parse_result.root_element.name,
                        "file_id": file_id,
                        "docstring": "",  # JavaScript doesn't have module docstrings
                        "start_line": parse_result.root_element.start_line,
                        "end_line": parse_result.root_element.end_line,
                        "imports": len(parse_result.imports),
                        "exports": 0,  # TODO(parser): Extract exports
                    }
                ],
                "classes": [],
                "functions": [],
                "imports": [],
            }

            # Process classes from ParseResult
            for cls_element in parse_result.find_elements(ElementType.CLASS):
                class_entity = {
                    "name": cls_element.name,
                    "file_id": file_id,
                    "start_line": cls_element.start_line,
                    "end_line": cls_element.end_line,
                    "docstring": "",
                    "base_classes": cls_element.metadata.get("base_classes", []),
                    "interfaces": [],  # JavaScript doesn't have interfaces
                    "methods": len(cls_element.find_children(ElementType.METHOD)),
                    "properties": 0,  # TODO(parser): Extract properties
                    "access_modifier": "public",  # JavaScript doesn't have access modifiers
                    "decorators": [],  # JavaScript doesn't have decorators
                }
                entities["classes"].append(class_entity)

                # Add methods as functions
                for method in cls_element.find_children(ElementType.METHOD):
                    method_entity = {
                        "name": f"{cls_element.name}.{method.name}",
                        "file_id": file_id,
                        "start_line": method.start_line,
                        "end_line": method.end_line,
                        "docstring": "",
                        "parameters": method.metadata.get("parameters", []),
                        "return_type": None,  # JavaScript doesn't have explicit return types
                        "complexity": 1,  # TODO(parser): Calculate complexity
                        "access_modifier": "public",
                        "decorators": [],
                        "is_async": method.metadata.get("is_async", False),
                        "parent_class": cls_element.name,
                    }
                    entities["functions"].append(method_entity)

            # Process standalone functions
            for func_element in parse_result.find_elements(ElementType.FUNCTION):
                func_entity = {
                    "name": func_element.name,
                    "file_id": file_id,
                    "start_line": func_element.start_line,
                    "end_line": func_element.end_line,
                    "docstring": "",
                    "parameters": func_element.metadata.get("parameters", []),
                    "return_type": None,  # JavaScript doesn't have explicit return types
                    "complexity": 1,  # TODO(parser): Calculate complexity
                    "access_modifier": "public",
                    "decorators": [],
                    "is_async": func_element.metadata.get("is_async", False),
                    "parent_class": None,
                }
                entities["functions"].append(func_entity)

            # Process imports from ParseResult
            for i, import_stmt in enumerate(parse_result.imports):
                import_entity = {
                    "module": import_stmt,  # This is the full import statement
                    "file_id": file_id,
                    "line": i + 1,  # Approximate line number
                    "imported_items": [],  # TODO(parser): Parse import statement
                    "alias": None,
                    "is_default": False,
                    "is_namespace": False,
                }
                entities["imports"].append(import_entity)

            return entities

        except (AttributeError, ValueError, TypeError) as e:
            logger.exception(
                "Error extracting entities from JavaScript file %s", file_path
            )
            msg = f"Failed to extract entities from JavaScript file: {file_path}"
            raise ParsingError(
                msg,
                str(file_path),
            ) from e
