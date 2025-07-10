"""TreeSitter parser for code analysis."""

from pathlib import Path
from typing import Any

import tree_sitter
import tree_sitter_python as tspython

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TreeSitterParser:
    """Base TreeSitter parser."""

    def __init__(self, language=None) -> None:
        self.language = language
        if language:
            self.parser = tree_sitter.Parser(language)
        else:
            self.parser = tree_sitter.Parser()

    def parse_file(self, file_path: Path) -> tree_sitter.Tree | None:
        """Parse a file and return the syntax tree."""
        try:
            with file_path.open("rb") as f:
                content = f.read()
            return self.parse_content(content)
        except Exception:
            logger.exception("Error parsing file %s", file_path)
            return None

    def parse_content(self, content: bytes) -> tree_sitter.Tree | None:
        """Parse content and return the syntax tree."""
        if not self.language:
            msg = "Language not set for parser"
            raise ValueError(msg)

        try:
            return self.parser.parse(content)
        except Exception:
            logger.exception("Error parsing content: %s")
            return None

    def get_node_text(self, node: tree_sitter.Node, content: bytes) -> str:
        """Get text content of a node."""
        return content[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")

    def get_node_location(self, node: tree_sitter.Node) -> tuple[int, int]:
        """Get start and end line numbers of a node."""
        return node.start_point[0] + 1, node.end_point[0] + 1

    def find_nodes_by_type(
        self,
        node: tree_sitter.Node,
        node_type: str,
        max_depth: int | None = None,
    ) -> list[tree_sitter.Node]:
        """Find all nodes of a specific type."""
        results = []

        def traverse(n: tree_sitter.Node, depth: int = 0) -> None:
            if max_depth is not None and depth > max_depth:
                return

            if n.type == node_type:
                results.append(n)

            for child in n.children:
                traverse(child, depth + 1)

        traverse(node)
        return results

    def get_docstring(self, node: tree_sitter.Node, content: bytes) -> str | None:
        """Extract docstring from a node."""
        # Look for string as first statement
        for child in node.children:
            if child.type == "block":
                for stmt in child.children:
                    if stmt.type == "expression_statement":
                        for expr in stmt.children:
                            if expr.type == "string":
                                docstring = self.get_node_text(expr, content)
                                # Remove quotes
                                if docstring.startswith(('"""', "'''")):
                                    return docstring[3:-3].strip()
                                if docstring.startswith(('"', "'")):
                                    return docstring[1:-1].strip()
                        break
                break
        return None


class PythonParser(TreeSitterParser):
    """Python-specific TreeSitter parser."""

    def __init__(self) -> None:
        self.language = tree_sitter.Language(tspython.language())
        super().__init__(self.language)

    def extract_imports(  # noqa: PLR0912
        self,
        tree: tree_sitter.Tree,
        content: bytes,
    ) -> list[dict[str, Any]]:
        """Extract import statements."""
        imports = []

        # Find import statements
        import_nodes = self.find_nodes_by_type(tree.root_node, "import_statement")
        for node in import_nodes:
            import_data = {
                "import_statement": self.get_node_text(node, content),
                "imported_from": None,
                "imported_names": [],
                "is_relative": False,
                "level": 0,
                "line_number": node.start_point[0] + 1,
            }

            # Extract imported names
            for child in node.children:
                if child.type in {"dotted_name", "identifier"}:
                    import_data["imported_names"].append(
                        self.get_node_text(child, content),
                    )

            imports.append(import_data)

        # Find from imports
        from_import_nodes = self.find_nodes_by_type(
            tree.root_node,
            "import_from_statement",
        )
        for node in from_import_nodes:
            import_data = {
                "import_statement": self.get_node_text(node, content),
                "imported_from": None,
                "imported_names": [],
                "is_relative": False,
                "level": 0,
                "line_number": node.start_point[0] + 1,
            }

            # Extract module name and imported names
            found_from = False
            found_import = False
            has_relative = False
            for child in node.children:
                if child.type == "from":
                    found_from = True
                elif child.type == "import":
                    found_import = True
                elif child.type == "relative_import":
                    import_data["is_relative"] = True
                    has_relative = True
                    # Count dots for relative level
                    # The relative_import node contains import_prefix child with dots
                    for rel_child in child.children:
                        if rel_child.type == "import_prefix":
                            dots = self.get_node_text(rel_child, content)
                            import_data["level"] = len(dots)
                        elif rel_child.type == "dotted_name":
                            # This is the module name after dots
                            import_data["imported_from"] = self.get_node_text(
                                rel_child, content
                            )
                elif child.type == "dotted_name":
                    if found_from and not found_import and not has_relative:
                        # First dotted_name after 'from' (and not after relative import) is the module
                        import_data["imported_from"] = self.get_node_text(
                            child, content
                        )
                    else:
                        # After 'import' or if we have relative import, it's an imported item
                        import_data["imported_names"].append(
                            self.get_node_text(child, content),
                        )
                elif child.type in {"import_list", "identifier"}:
                    if child.type == "identifier":
                        import_data["imported_names"].append(
                            self.get_node_text(child, content),
                        )
                    else:
                        # Handle import list
                        for name_node in child.children:
                            if name_node.type in {"identifier", "dotted_name"}:
                                import_data["imported_names"].append(
                                    self.get_node_text(name_node, content),
                                )

            imports.append(import_data)

        return imports

    def extract_functions(  # noqa: PLR0912
        self,
        tree: tree_sitter.Tree,
        content: bytes,
        parent_class: tree_sitter.Node | None = None,
    ) -> list[dict[str, Any]]:
        """Extract function definitions."""
        functions = []
        root = parent_class if parent_class else tree.root_node

        function_nodes = self.find_nodes_by_type(
            root,
            "function_definition",
            max_depth=2 if parent_class else None,
        )

        for node in function_nodes:
            func_data = {
                "name": None,
                "parameters": [],
                "return_type": None,
                "docstring": None,
                "decorators": [],
                "is_async": False,
                "is_generator": False,
                "is_property": False,
                "is_staticmethod": False,
                "is_classmethod": False,
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
            }

            # Check for async
            for child in node.children:
                if child.type == "async":
                    func_data["is_async"] = True
                    break

            # Extract decorators
            decorator_nodes = []
            prev_sibling = node.prev_sibling
            while prev_sibling and prev_sibling.type == "decorator":
                decorator_nodes.insert(0, prev_sibling)
                prev_sibling = prev_sibling.prev_sibling

            for dec_node in decorator_nodes:
                decorator_text = self.get_node_text(dec_node, content).strip()
                if decorator_text.startswith("@"):
                    decorator_name = decorator_text[1:].split("(")[0]
                    func_data["decorators"].append(decorator_name)

                    # Check for special decorators
                    if decorator_name == "property":
                        func_data["is_property"] = True
                    elif decorator_name == "staticmethod":
                        func_data["is_staticmethod"] = True
                    elif decorator_name == "classmethod":
                        func_data["is_classmethod"] = True

            # Extract function details
            for child in node.children:
                if child.type == "identifier":
                    func_data["name"] = self.get_node_text(child, content)
                elif child.type == "parameters":
                    func_data["parameters"] = self._extract_parameters(child, content)
                elif child.type == "type":
                    # Return type annotation
                    func_data["return_type"] = (
                        self.get_node_text(child, content).strip("->").strip()
                    )
                elif child.type == "block" and (
                    self.find_nodes_by_type(child, "yield_statement")
                    or self.find_nodes_by_type(child, "yield_expression")
                    or self.find_nodes_by_type(child, "yield")
                ):
                    # Check if it's a generator by looking for yield nodes
                    func_data["is_generator"] = True

            # Extract docstring
            func_data["docstring"] = self.get_docstring(node, content)

            functions.append(func_data)

        return functions

    def extract_classes(  # noqa: PLR0912
        self,
        tree: tree_sitter.Tree,
        content: bytes,
    ) -> list[dict[str, Any]]:
        """Extract class definitions."""
        classes = []

        class_nodes = self.find_nodes_by_type(tree.root_node, "class_definition")

        for node in class_nodes:
            class_data = {
                "name": None,
                "docstring": None,
                "base_classes": [],
                "decorators": [],
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "is_abstract": False,
                "methods": [],
            }

            # Extract decorators
            decorator_nodes = []
            prev_sibling = node.prev_sibling
            while prev_sibling and prev_sibling.type == "decorator":
                decorator_nodes.insert(0, prev_sibling)
                prev_sibling = prev_sibling.prev_sibling

            for dec_node in decorator_nodes:
                decorator_text = self.get_node_text(dec_node, content).strip()
                if decorator_text.startswith("@"):
                    decorator_name = decorator_text[1:].split("(")[0]
                    class_data["decorators"].append(decorator_name)

                    # Check for abstract class decorators
                    if "abstract" in decorator_name.lower():
                        class_data["is_abstract"] = True

            # Extract class details
            for child in node.children:
                if child.type == "identifier":
                    class_data["name"] = self.get_node_text(child, content)
                elif child.type == "argument_list":
                    # Extract base classes
                    for arg in child.children:
                        if arg.type in {"identifier", "attribute"}:
                            class_data["base_classes"].append(
                                self.get_node_text(arg, content),
                            )

            # Extract docstring
            class_data["docstring"] = self.get_docstring(node, content)

            # Extract methods
            class_data["methods"] = self.extract_functions(tree, content, node)

            # Check if abstract based on methods
            if not class_data["is_abstract"]:
                for method in class_data["methods"]:
                    if "abstractmethod" in method.get("decorators", []):
                        class_data["is_abstract"] = True
                        break

            classes.append(class_data)

        return classes

    def extract_module_info(
        self,
        tree: tree_sitter.Tree,
        content: bytes,
    ) -> dict[str, Any]:
        """Extract module-level information."""
        return {
            "docstring": self._get_module_docstring(tree, content),
            "imports": self.extract_imports(tree, content),
            "classes": self.extract_classes(tree, content),
            "functions": self.extract_functions(tree, content),
        }

    def _extract_parameters(
        self,
        params_node: tree_sitter.Node,
        content: bytes,
    ) -> list[dict[str, Any]]:
        """Extract function parameters."""
        parameters = []

        for child in params_node.children:
            if child.type in (
                "identifier",
                "typed_parameter",
                "typed_default_parameter",
                "default_parameter",
            ):
                param_data = {
                    "name": None,
                    "type": None,
                    "default": None,
                }

                if child.type == "identifier":
                    param_data["name"] = self.get_node_text(child, content)
                else:
                    # Handle typed and default parameters
                    for subchild in child.children:
                        if subchild.type == "identifier":
                            param_data["name"] = self.get_node_text(subchild, content)
                        elif subchild.type == "type":
                            param_data["type"] = (
                                self.get_node_text(subchild, content).strip(":").strip()
                            )
                        elif subchild.type not in (":", "="):
                            # Default value
                            param_data["default"] = self.get_node_text(
                                subchild,
                                content,
                            )

                if param_data["name"] and param_data["name"] not in (
                    "self",
                    "cls",
                    "*",
                    "**",
                    "/",
                ):
                    parameters.append(param_data)

        return parameters

    def _get_module_docstring(
        self,
        tree: tree_sitter.Tree,
        content: bytes,
    ) -> str | None:
        """Extract module-level docstring."""
        # Module docstring is the first string in the file
        for child in tree.root_node.children:
            if child.type == "expression_statement":
                for expr in child.children:
                    if expr.type == "string":
                        docstring = self.get_node_text(expr, content)
                        # Remove quotes
                        if docstring.startswith(('"""', "'''")):
                            return docstring[3:-3].strip()
                        if docstring.startswith(('"', "'")):
                            return docstring[1:-1].strip()
                break
        return None
