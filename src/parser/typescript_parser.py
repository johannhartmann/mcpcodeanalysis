"""TypeScript code parser using TreeSitter."""

from pathlib import Path
from typing import Any

try:
    import tree_sitter_typescript as tstypescript

    TYPESCRIPT_AVAILABLE = True
except ImportError:
    TYPESCRIPT_AVAILABLE = False

import tree_sitter
from tree_sitter import Node, Tree

from src.logger import get_logger
from src.parser.base_parser import BaseParser, ElementType, ParsedElement
from src.parser.treesitter_parser import TypeScriptParser as TreeSitterTypeScriptParser
from src.utils.exceptions import ParsingError

logger = get_logger(__name__)


class TypeScriptCodeParser(BaseParser):
    """Parser for TypeScript code files."""

    def __init__(self) -> None:
        if not TYPESCRIPT_AVAILABLE:
            msg = "tree-sitter-typescript not available"
            raise ImportError(msg)

        super().__init__(tree_sitter.Language(tstypescript.language_typescript()))
        self.ts_parser = TreeSitterTypeScriptParser()

    def get_language_name(self) -> str:
        """Get the name of the programming language."""
        return "typescript"

    def get_file_extensions(self) -> set[str]:
        """Get supported file extensions."""
        return {".ts", ".tsx", ".d.ts"}

    def _extract_elements(
        self, tree: Tree, content: str, file_path: Path
    ) -> ParsedElement:
        """Extract code elements from TypeScript parse tree."""
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
            # Extract classes using direct TreeSitter traversal
            self._extract_classes_direct(tree.root_node, content, root)

            # Extract functions using direct TreeSitter traversal
            self._extract_functions_direct(tree.root_node, content, root)

        except (AttributeError, ValueError, TypeError) as e:
            logger.warning(
                "Error extracting TypeScript elements from %s: %s", file_path, e
            )

        return root

    def _extract_imports(self, tree: Tree, content: str) -> list[str]:
        """Extract import statements from TypeScript parse tree."""
        imports: list[str] = []

        if not tree or not tree.root_node:
            return imports

        try:
            # Simple direct extraction of import statements
            self._extract_imports_direct(tree.root_node, content, imports)

        except (AttributeError, ValueError, TypeError) as e:
            logger.warning("Error extracting TypeScript imports: %s", e)

        return imports

    def _extract_imports_direct(
        self, node: Node, content: str, imports: list[str]
    ) -> None:
        """Extract imports using direct TreeSitter traversal."""
        if node.type == "import_statement":
            import_text = self._get_node_text(node, content).strip()
            imports.append(import_text)

        # Recursively search children
        for child in node.children:
            self._extract_imports_direct(child, content, imports)

    def _extract_references(
        self,
        tree: Tree,
        content: str,  # noqa: ARG002
    ) -> list[dict[str, Any]]:
        """Extract code references from TypeScript parse tree."""
        references: list[dict[str, Any]] = []

        if not tree or not tree.root_node:
            return references

        try:
            # For now, return empty references - will implement properly later
            # This is a placeholder to avoid errors
            pass

        except (AttributeError, ValueError, TypeError) as e:
            logger.warning("Error extracting TypeScript references: %s", e)

        return references

    def _extract_classes_direct(
        self, node: Node, content: str, root: ParsedElement
    ) -> None:
        """Extract classes using direct TreeSitter node traversal."""
        if node.type in ("class_declaration", "abstract_class_declaration"):
            # Find class name by searching descendants for common name node types
            class_name = "UnknownClass"
            from tree_sitter import Node as TSNode

            def _find_name(n: TSNode) -> str | None:
                if n.type in (
                    "type_identifier",
                    "identifier",
                    "property_identifier",
                    "name",
                ):
                    return self._get_node_text(n, content)
                for ch in n.children:
                    res = _find_name(ch)
                    if res:
                        return res
                return None

            found_name = _find_name(node)
            if found_name:
                class_name = found_name

            # Attempt to extract base classes and interfaces from heritage clauses
            base_classes: list[str] = []
            interfaces: list[str] = []

            def _collect_heritage(n: Node) -> None:
                text = self._get_node_text(n, content).strip()
                # If clause contains 'extends' or 'implements', try to parse identifiers
                if "extends" in text or "implements" in text:
                    # Remove the keywords and split by comma
                    cleaned = text.replace("extends", "").replace("implements", "")
                    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
                    for part in parts:
                        # Extract a simple name (strip generics and other tokens)
                        import re

                        m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)", part)
                        if m:
                            name = m.group(1)
                            # Skip common language keywords that might appear
                            if name in ("extends", "implements", "class"):
                                continue
                            if "extends" in text and name not in base_classes:
                                base_classes.append(name)
                            elif "implements" in text and name not in interfaces:
                                interfaces.append(name)

                for ch in n.children:
                    _collect_heritage(ch)

            _collect_heritage(node)

            cls_element = ParsedElement(
                type=ElementType.CLASS,
                name=class_name,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                start_column=node.start_point[1],
                end_column=node.end_point[1],
                text=self._get_node_text(node, content),
                metadata={
                    "base_classes": base_classes,
                    "interfaces": interfaces,
                    "access_modifier": "public",
                    "decorators": [],
                },
            )
            root.add_child(cls_element)

            # Extract methods from class body
            self._extract_methods_from_class(node, content, cls_element)

        # Recursively search children
        for child in node.children:
            self._extract_classes_direct(child, content, root)

    def _extract_functions_direct(
        self, node: Node, content: str, root: ParsedElement
    ) -> None:
        """Extract standalone functions using direct TreeSitter node traversal."""
        if node.type in ("function_declaration", "function_expression"):
            # Find function name
            func_name = "UnknownFunction"
            for child in node.children:
                if child.type == "identifier":
                    func_name = self._get_node_text(child, content)
                    break

            func_element = ParsedElement(
                type=ElementType.FUNCTION,
                name=func_name,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                start_column=node.start_point[1],
                end_column=node.end_point[1],
                text=self._get_node_text(node, content),
                metadata={
                    "parameters": [],
                    "return_type": "any",
                    "is_async": False,
                    "decorators": [],
                },
            )
            root.add_child(func_element)

        # Handle arrow functions in variable / lexical declarations (const/let/var)
        elif node.type in (
            "variable_declaration",
            "variable_statement",
            "lexical_declaration",
            "lexical_declaration_statement",
            "variable_declaration_statement",
        ):
            # Walk descendants and extract any variable_declarator we find
            nodes = [node]
            while nodes:
                cur = nodes.pop()
                for ch in cur.children:
                    if ch.type == "variable_declarator":
                        self._extract_arrow_function(ch, content, root)
                    else:
                        nodes.append(ch)

        # Recursively search children
        for child in node.children:
            self._extract_functions_direct(child, content, root)

    def _extract_methods_from_class(
        self, class_node: Node, content: str, cls_element: ParsedElement
    ) -> None:
        """Extract methods from a class body."""
        for child in class_node.children:
            if child.type == "class_body":
                for member in child.children:
                    if member.type in ("method_definition", "method_signature"):
                        # Find method name, accept several identifier node types
                        method_name = "UnknownMethod"
                        for subchild in member.children:
                            if subchild.type in (
                                "property_identifier",
                                "identifier",
                                "private_identifier",
                            ):
                                method_name = self._get_node_text(subchild, content)
                                break

                        # Fallback: if constructor is present as a child token
                        if method_name == "UnknownMethod":
                            text = self._get_node_text(member, content)
                            if text.strip().startswith("constructor"):
                                method_name = "constructor"

                        method_element = ParsedElement(
                            type=ElementType.METHOD,
                            name=method_name,
                            start_line=member.start_point[0] + 1,
                            end_line=member.end_point[0] + 1,
                            start_column=member.start_point[1],
                            end_column=member.end_point[1],
                            text=self._get_node_text(member, content),
                            metadata={
                                "parameters": [],
                                "return_type": "any",
                                "access_modifier": "public",
                                "is_async": False,
                                "decorators": [],
                            },
                        )
                        cls_element.add_child(method_element)

    def _extract_arrow_function(
        self, declarator_node: Node, content: str, root: ParsedElement
    ) -> None:
        """Extract arrow function from variable declarator."""
        func_name = "UnknownFunction"
        arrow_node = None

        # Walk descendants to find the arrow_function node and the identifier (name)
        def _find_arrow(n: Node) -> None:
            nonlocal arrow_node, func_name
            for ch in n.children:
                if ch.type == "arrow_function":
                    arrow_node = ch
                if ch.type == "identifier" and func_name == "UnknownFunction":
                    func_name = self._get_node_text(ch, content)
                # Recurse
                _find_arrow(ch)

        _find_arrow(declarator_node)

        if arrow_node:
            func_element = ParsedElement(
                type=ElementType.FUNCTION,
                name=func_name,
                start_line=arrow_node.start_point[0] + 1,
                end_line=arrow_node.end_point[0] + 1,
                start_column=arrow_node.start_point[1],
                end_column=arrow_node.end_point[1],
                text=self._get_node_text(arrow_node, content),
                metadata={
                    "parameters": [],
                    "return_type": "any",
                    "is_async": False,
                    "decorators": [],
                },
            )
            root.add_child(func_element)

    def extract_entities(self, file_path: Path, file_id: int) -> dict[str, list[Any]]:
        """Extract code entities for database storage."""
        try:
            # Use the new BaseParser interface to parse
            parse_result = self.parse_file(file_path)

            if not parse_result.success:
                logger.warning(
                    "Failed to parse TypeScript file %s: %s",
                    file_path,
                    parse_result.errors,
                )
                return {"modules": [], "classes": [], "functions": [], "imports": []}

            entities = {
                "modules": [
                    {
                        "name": parse_result.root_element.name,
                        "file_id": file_id,
                        "docstring": "",  # TypeScript doesn't have module docstrings
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
                    "interfaces": cls_element.metadata.get("interfaces", []),
                    "methods": len(cls_element.find_children(ElementType.METHOD)),
                    "properties": 0,  # TODO(parser): Extract properties
                    "access_modifier": cls_element.metadata.get(
                        "access_modifier", "public"
                    ),
                    "decorators": cls_element.metadata.get("decorators", []),
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
                        "return_type": method.metadata.get("return_type", "any"),
                        "complexity": 1,  # TODO(parser): Calculate complexity
                        "access_modifier": method.metadata.get(
                            "access_modifier", "public"
                        ),
                        "decorators": method.metadata.get("decorators", []),
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
                    "return_type": func_element.metadata.get("return_type", "any"),
                    "complexity": 1,  # TODO(parser): Calculate complexity
                    "access_modifier": func_element.metadata.get(
                        "access_modifier", "public"
                    ),
                    "decorators": func_element.metadata.get("decorators", []),
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
                "Error extracting entities from TypeScript file %s", file_path
            )
            msg = f"Failed to extract entities from TypeScript file: {file_path}"
            raise ParsingError(
                msg,
                str(file_path),
            ) from e
