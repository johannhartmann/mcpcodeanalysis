"""TreeSitter parser for code analysis."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, TypedDict

import tree_sitter
import tree_sitter_java as tsjava
import tree_sitter_php as tsphp
import tree_sitter_python as tspython

from src.logger import get_logger
from src.parser.complexity_calculator import ComplexityCalculator

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

# Optional TreeSitter language imports with error handling
JAVASCRIPT_AVAILABLE = False
TYPESCRIPT_AVAILABLE = False

# These module variables are optional and may be None at runtime
tsjavascript: ModuleType | None = None
tstypescript: ModuleType | None = None

try:
    tsjavascript = import_module("tree_sitter_javascript")
    JAVASCRIPT_AVAILABLE = True
except ImportError:
    JAVASCRIPT_AVAILABLE = False

try:
    tstypescript = import_module("tree_sitter_typescript")
    TYPESCRIPT_AVAILABLE = True
except ImportError:
    TYPESCRIPT_AVAILABLE = False

logger = get_logger(__name__)


class TreeSitterParser:
    """Base TreeSitter parser."""

    def __init__(self, language: tree_sitter.Language | None = None) -> None:
        self.language: tree_sitter.Language | None = language
        if language is not None:
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
            logger.exception("Error parsing content: %s", content[:100])
            return None

    def require_language(self) -> tree_sitter.Language:
        """Return the configured language or raise if missing (for typing)."""
        if self.language is None:
            msg = "Language not set for parser"
            raise ValueError(msg)
        return self.language

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


class PyImportData(TypedDict, total=False):
    import_statement: str
    imported_from: str | None
    imported_names: list[str]
    is_relative: bool
    level: int
    line_number: int


class PyFuncParam(TypedDict, total=False):
    name: str | None
    type: str | None
    default: str | None


class PyFuncData(TypedDict, total=False):
    name: str | None
    parameters: list[PyFuncParam]
    return_type: str | None
    docstring: str | None
    decorators: list[str]
    is_async: bool
    is_generator: bool
    is_property: bool
    is_staticmethod: bool
    is_classmethod: bool
    start_line: int
    end_line: int
    complexity: int


class PyClassData(TypedDict, total=False):
    name: str | None
    docstring: str | None
    base_classes: list[str]
    decorators: list[str]
    start_line: int
    end_line: int
    is_abstract: bool
    methods: list[PyFuncData]


class PythonParser(TreeSitterParser):
    """Python-specific TreeSitter parser."""

    def __init__(self) -> None:
        self.language = tree_sitter.Language(tspython.language())
        super().__init__(self.language)
        self.complexity_calculator = ComplexityCalculator(use_plugin=False)

    def extract_imports(  # noqa: PLR0912
        self,
        tree: tree_sitter.Tree,
        content: bytes,
    ) -> list[dict[str, Any]]:
        """Extract import statements."""
        imports: list[dict[str, Any]] = []

        # Find import statements
        import_nodes = self.find_nodes_by_type(tree.root_node, "import_statement")
        for node in import_nodes:
            import_data: dict[str, Any] = {
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
        functions: list[dict[str, Any]] = []

        root = parent_class if parent_class else tree.root_node

        # When extracting module-level functions, we need to exclude those inside classes
        if parent_class:
            function_nodes = self.find_nodes_by_type(
                root,
                "function_definition",
                max_depth=4,  # Increased to handle decorated methods
            )
        else:
            # For module-level, find all functions then filter out those inside classes
            all_function_nodes = self.find_nodes_by_type(root, "function_definition")
            class_nodes = self.find_nodes_by_type(root, "class_definition")

            # Create a set of functions that are inside classes
            class_function_nodes = set()
            for class_node in class_nodes:
                class_functions = self.find_nodes_by_type(
                    class_node, "function_definition"
                )
                class_function_nodes.update(class_functions)

            # Only keep module-level functions
            function_nodes = [
                f for f in all_function_nodes if f not in class_function_nodes
            ]

        for node in function_nodes:
            func_data: dict[str, Any] = {
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
            decorator_nodes: list[tree_sitter.Node] = []

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
                    elif decorator_name.endswith((".setter", ".deleter")):
                        # Property setters and deleters
                        func_data["is_property"] = True

            # Normalize convenience flag
            func_data["is_static"] = func_data.get("is_staticmethod", False)

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
                    (
                        self.find_nodes_by_type(child, "yield_statement")
                        or self.find_nodes_by_type(child, "yield_expression")
                        or self.find_nodes_by_type(child, "yield")
                    )
                    or self._contains_generator_return(child, content)
                ):
                    # Generator detected by yield nodes
                    func_data["is_generator"] = True

            # Extract docstring
            func_data["docstring"] = self.get_docstring(node, content)

            # Calculate cyclomatic complexity
            func_data["complexity"] = self.complexity_calculator.calculate_complexity(
                node, content
            )

            functions.append(func_data)

        return functions

    def extract_classes(  # noqa: PLR0912
        self,
        tree: tree_sitter.Tree,
        content: bytes,
    ) -> list[dict[str, Any]]:
        """Extract class definitions."""
        classes: list[dict[str, Any]] = []

        class_nodes = self.find_nodes_by_type(tree.root_node, "class_definition")

        for node in class_nodes:
            class_data: dict[str, Any] = {
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
            decorator_nodes: list[tree_sitter.Node] = []
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
        parameters: list[dict[str, Any]] = []

        for child in params_node.children:
            if child.type in (
                "identifier",
                "typed_parameter",
                "typed_default_parameter",
                "default_parameter",
            ):
                param_data: dict[str, Any] = {
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

    def _contains_generator_return(
        self,
        block_node: tree_sitter.Node,
        _content: bytes,
    ) -> bool:
        """Check if a block contains a return statement with a generator expression."""
        return_nodes = self.find_nodes_by_type(block_node, "return_statement")
        for return_node in return_nodes:
            # Check for generator expressions in return statements
            if self.find_nodes_by_type(return_node, "generator_expression"):
                return True
            # Check for yield from expressions
            for child in return_node.children:
                if child.type == "yield_from_expression":
                    return True
        return False


class PHPParser(TreeSitterParser):
    """PHP-specific TreeSitter parser."""

    def __init__(self) -> None:
        self.language = tree_sitter.Language(tsphp.language_php())
        super().__init__(self.language)
        self.complexity_calculator = ComplexityCalculator(language="php")
        self.references: list[dict[str, Any]] = []

    def extract_imports(
        self,
        tree: tree_sitter.Tree,
        content: bytes,
    ) -> list[dict[str, Any]]:
        """Extract use statements (imports) in PHP."""
        imports = []

        # Find use statements
        use_nodes = self.find_nodes_by_type(tree.root_node, "namespace_use_declaration")
        for node in use_nodes:
            import_data: dict[str, Any]

            import_data = {
                "import_statement": self.get_node_text(node, content),
                "imported_from": None,
                "imported_names": [],
                "is_relative": False,
                "level": 0,
                "line_number": node.start_point[0] + 1,
            }

            # Extract namespace/class names
            for child in node.children:
                if child.type == "namespace_use_clause":
                    # Extract the name from the use clause
                    alias_name = None
                    for clause_child in child.children:
                        if clause_child.type == "name":
                            # This could be either the import name or alias
                            if import_data["imported_from"] is None:
                                name = self.get_node_text(clause_child, content)
                                import_data["imported_from"] = name
                                import_data["imported_names"].append(name)
                            else:
                                # This is the alias
                                alias_name = self.get_node_text(clause_child, content)
                        elif clause_child.type == "qualified_name":
                            name = self.get_node_text(clause_child, content)
                            import_data["imported_from"] = name
                            # Last part is the imported name
                            parts = name.split("\\")
                            if parts:
                                import_data["imported_names"].append(parts[-1])

                    # If we found an alias, replace the imported names
                    if alias_name:
                        import_data["imported_names"] = [alias_name]

            imports.append(import_data)

        return imports

    def extract_functions(
        self,
        tree: tree_sitter.Tree,
        content: bytes,
        parent_class: tree_sitter.Node | None = None,
    ) -> list[dict[str, Any]]:
        """Extract function/method definitions."""
        functions = []
        root = parent_class if parent_class else tree.root_node

        # Find function declarations
        if parent_class:
            function_nodes = self.find_nodes_by_type(
                root,
                "method_declaration",
                max_depth=4,
            )
            func_data: dict[str, Any]

        else:
            # Module-level functions
            function_nodes = self.find_nodes_by_type(root, "function_definition")

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

            # Extract function details
            for child in node.children:
                if child.type == "name":
                    func_data["name"] = self.get_node_text(child, content)
                elif child.type == "formal_parameters":
                    func_data["parameters"] = self._extract_parameters(child, content)
                elif child.type in ("type", "primitive_type", "named_type"):
                    # Return type annotation
                    func_data["return_type"] = self.get_node_text(child, content)
                elif child.type == "static_modifier":
                    # Static method
                    func_data["is_staticmethod"] = True

            # Extract docstring (PHP DocBlock)
            func_data["docstring"] = self._get_php_docstring(node, content)

            # Calculate cyclomatic complexity
            func_data["complexity"] = self.complexity_calculator.calculate_complexity(
                node, content
            )

            functions.append(func_data)

        return functions

    def _extract_base_classes(
        self, node: tree_sitter.Node, content: bytes
    ) -> list[str]:
        """Extract base classes from base_clause node."""
        return [
            self.get_node_text(child, content)
            for child in node.children
            if child.type == "name"
        ]

    def _extract_interfaces(self, node: tree_sitter.Node, content: bytes) -> list[str]:
        """Extract interfaces from class_interface_clause node."""
        return [
            self.get_node_text(child, content)
            for child in node.children
            if child.type == "name"
        ]

    def _extract_class_traits(
        self, node: tree_sitter.Node, content: bytes
    ) -> list[str]:
        """Extract traits used by the class."""
        use_nodes = self.find_nodes_by_type(node, "use_declaration")
        return [
            self.get_node_text(child, content)
            for use_node in use_nodes
            for child in use_node.children
            if child.type == "name"
        ]

    def _process_class_modifiers(
        self, node: tree_sitter.Node, content: bytes, class_data: dict[str, Any]
    ) -> None:
        """Process class modifiers (abstract, final)."""
        modifier = self.get_node_text(node, content)
        if modifier == "abstract":
            class_data["is_abstract"] = True

    def extract_classes(
        self,
        tree: tree_sitter.Tree,
        content: bytes,
    ) -> list[dict[str, Any]]:
        """Extract class definitions."""
        classes: list[dict[str, Any]] = []

        class_nodes = self.find_nodes_by_type(tree.root_node, "class_declaration")

        for node in class_nodes:
            class_data: dict[str, Any] = {
                "name": None,
                "docstring": None,
                "base_classes": [],
                "decorators": [],
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "is_abstract": False,
                "methods": [],
                "traits": [],
            }

            # Process child nodes using a dispatch approach
            for child in node.children:
                if child.type == "name":
                    class_data["name"] = self.get_node_text(child, content)
                elif child.type == "base_clause":
                    class_data["base_classes"].extend(
                        self._extract_base_classes(child, content)
                    )
                elif child.type == "class_interface_clause":
                    class_data["base_classes"].extend(
                        self._extract_interfaces(child, content)
                    )
                elif child.type in ["abstract_modifier", "final_modifier"]:
                    self._process_class_modifiers(child, content, class_data)
                elif child.type == "declaration_list":
                    class_data["traits"] = self._extract_class_traits(child, content)

            # Extract docstring and methods
            class_data["docstring"] = self._get_php_docstring(node, content)
            class_data["methods"] = self.extract_functions(tree, content, node)

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
            "traits": self.extract_traits(tree, content),
        }

    def _extract_parameters(
        self,
        params_node: tree_sitter.Node,
        content: bytes,
    ) -> list[dict[str, Any]]:
        """Extract function parameters."""
        parameters: list[dict[str, Any]] = []

        for child in params_node.children:
            if child.type == "simple_parameter":
                param_data: dict[str, Any] = {
                    "name": None,
                    "type": None,
                    "default": None,
                }

                has_default = False
                for subchild in child.children:
                    if subchild.type == "variable_name":
                        param_data["name"] = self.get_node_text(
                            subchild, content
                        ).lstrip("$")
                    elif subchild.type in [
                        "primitive_type",
                        "named_type",
                        "optional_type",
                    ]:
                        param_data["type"] = self.get_node_text(subchild, content)
                    elif subchild.type == "=":
                        has_default = True
                    elif has_default and subchild.type not in [
                        "variable_name",
                        "primitive_type",
                        "named_type",
                        "optional_type",
                        "=",
                    ]:
                        # This is the default value
                        param_data["default"] = self.get_node_text(subchild, content)

                if param_data["name"]:
                    parameters.append(param_data)

        return parameters

    def _get_module_docstring(
        self,
        tree: tree_sitter.Tree,
        content: bytes,
    ) -> str | None:
        """Extract module-level docstring (first comment block)."""
        # In PHP, look for the first comment block in the file
        for child in tree.root_node.children:
            if child.type == "comment":
                text = self.get_node_text(child, content)
                if text.startswith("/**"):
                    return self._parse_docblock(text)
        return None

    def _get_php_docstring(
        self,
        node: tree_sitter.Node,
        content: bytes,
    ) -> str | None:
        """Extract PHP DocBlock comment."""
        # Look for comment just before the node
        prev_sibling = node.prev_sibling
        if prev_sibling and prev_sibling.type == "comment":
            text = prev_sibling.text
            if text is not None and text.startswith(b"/**"):
                return self._parse_docblock(self.get_node_text(prev_sibling, content))
        return None

    def _parse_docblock(self, docblock: str) -> str:
        """Parse PHP DocBlock to extract description."""
        lines = docblock.strip().split("\n")
        if not lines:
            return ""

        # Remove /** and */
        if lines[0].strip() == "/**":
            lines = lines[1:]
        if lines and lines[-1].strip() == "*/":
            lines = lines[:-1]

        # Remove leading asterisks and spaces
        cleaned_lines = []
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith("*"):
                stripped_line = stripped_line[1:].strip()
            # Skip annotation lines like @param, @return
            if not stripped_line.startswith("@"):
                cleaned_lines.append(stripped_line)

        return " ".join(cleaned_lines).strip()

    def extract_traits(
        self,
        tree: tree_sitter.Tree,
        content: bytes,
    ) -> list[dict[str, Any]]:
        """Extract trait definitions."""
        traits = []

        trait_nodes = self.find_nodes_by_type(tree.root_node, "trait_declaration")

        for node in trait_nodes:
            trait_data: dict[str, Any] = {
                "name": None,
                "docstring": None,
                "methods": [],
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
            }

            # Extract trait details
            for child in node.children:
                if child.type == "name":
                    trait_data["name"] = self.get_node_text(child, content)

            # Extract docstring
            trait_data["docstring"] = self._get_php_docstring(node, content)

            # Extract methods
            trait_data["methods"] = self.extract_functions(tree, content, node)

            traits.append(trait_data)

        return traits

    def extract_references(
        self, tree: tree_sitter.Tree, content: bytes
    ) -> list[dict[str, Any]]:
        """Extract references from PHP code."""
        references = []

        # Extract import references from use statements
        references.extend(
            [
                {
                    "type": "import",
                    "source": "module",
                    "target": imp["imported_from"],
                    "line": imp["line_number"],
                }
                for imp in self.extract_imports(tree, content)
                if imp["imported_from"]
            ]
        )

        # Extract class inheritance references
        classes = self.extract_classes(tree, content)
        for cls in classes:
            # Base class references
            references.extend(
                [
                    {
                        "type": "inherit",
                        "source": cls["name"],
                        "target": base,
                        "line": cls["start_line"],
                    }
                    for base in cls.get("base_classes", [])
                ]
            )

            # Trait usage references
            references.extend(
                [
                    {
                        "type": "trait_use",
                        "source": cls["name"],
                        "target": trait,
                        "line": cls["start_line"],
                    }
                    for trait in cls.get("traits", [])
                ]
            )

        # Extract type references from functions
        functions = self.extract_functions(tree, content)
        for func in functions:
            # Return type references
            if func.get("return_type") and func["return_type"] not in [
                "void",
                "string",
                "int",
                "bool",
                "float",
                "array",
                "mixed",
            ]:
                references.append(
                    {
                        "type": "type_use",
                        "source": func["name"],
                        "target": func["return_type"].lstrip("?"),
                        "line": func["start_line"],
                    }
                )

            # Parameter type references
            references.extend(
                [
                    {
                        "type": "type_use",
                        "source": func["name"],
                        "target": param["type"].lstrip("?"),
                        "line": func["start_line"],
                    }
                    for param in func.get("parameters", [])
                    if param.get("type")
                    and param["type"]
                    not in [
                        "string",
                        "int",
                        "bool",
                        "float",
                        "array",
                        "mixed",
                    ]
                ]
            )

        return references


class JavaParser(TreeSitterParser):
    """Java-specific TreeSitter parser."""

    def __init__(self) -> None:
        self.language = tree_sitter.Language(tsjava.language())
        super().__init__(self.language)
        self.complexity_calculator = ComplexityCalculator(language="java")

    def extract_imports(
        self,
        tree: tree_sitter.Tree,
        content: bytes,
    ) -> list[dict[str, Any]]:
        """Extract import statements."""
        imports = []

        # Find import declarations
        import_nodes = self.find_nodes_by_type(tree.root_node, "import_declaration")
        for node in import_nodes:
            import_data: dict[str, Any] = {
                "import_statement": self.get_node_text(node, content),
                "imported_from": None,
                "imported_names": [],
                "is_relative": False,
                "level": 0,
                "line_number": node.start_point[0] + 1,
            }

            # Extract imported class/package
            found_wildcard = False
            for child in node.children:
                if child.type == "scoped_identifier":
                    full_name = self.get_node_text(child, content)
                    import_data["imported_from"] = full_name
                    # Last part is the imported name
                    parts = full_name.split(".")
                    if parts:
                        import_data["imported_names"].append(parts[-1])
                elif child.type == "asterisk":
                    found_wildcard = True

            # If we found a wildcard, update the imported names
            if found_wildcard:
                import_data["imported_names"] = ["*"]

            imports.append(import_data)

        return imports

    def _extract_method_modifiers(
        self, node: tree_sitter.Node, content: bytes
    ) -> tuple[bool, list[str]]:
        """Extract method modifiers and annotations."""
        is_static = False
        decorators = []

        for child in node.children:
            if child.type == "modifiers":
                modifiers_text = self.get_node_text(child, content)
                if "static" in modifiers_text:
                    is_static = True

                # Extract annotations
                for mod_child in child.children:
                    if mod_child.type == "marker_annotation":
                        annotation_text = self.get_node_text(mod_child, content)
                        decorators.append(annotation_text.lstrip("@"))
                    elif mod_child.type == "annotation":
                        annotation_text = self.get_node_text(mod_child, content)
                        if annotation_text.startswith("@"):
                            annotation_name = annotation_text[1:].split("(")[0]
                            decorators.append(annotation_name)
                break

        return is_static, decorators

    def _extract_method_signature(
        self, node: tree_sitter.Node, content: bytes
    ) -> tuple[str | None, str | None, list[Any]]:
        """Extract method name, return type, and parameters."""
        name = None
        return_type = None
        parameters = []

        # Java return types for method signature
        return_type_nodes = {
            "type_identifier",
            "primitive_type",
            "integral_type",
            "floating_point_type",
            "boolean_type",
            "void_type",
            "generic_type",
            "array_type",
        }

        found_identifier = False
        for i, child in enumerate(node.children):
            if child.type == "identifier":
                name = self.get_node_text(child, content)
                found_identifier = True
                # Look for return type in previous siblings
                if i > 0 and node.children[i - 1].type in return_type_nodes:
                    return_type = self.get_node_text(node.children[i - 1], content)
            elif child.type == "formal_parameters" and found_identifier:
                parameters = self._extract_parameters(child, content)

        return name, return_type, parameters

    def _resolve_constructor_name(
        self, node: tree_sitter.Node, content: bytes
    ) -> str | None:
        """Resolve constructor name from parent class."""
        parent = node.parent
        class_types = {"class_declaration", "interface_declaration", "enum_declaration"}

        while parent and parent.type not in class_types:
            parent = parent.parent

        if parent:
            for child in parent.children:
                if child.type == "identifier":
                    return self.get_node_text(child, content)

        return None

    def extract_functions(
        self,
        tree: tree_sitter.Tree,
        content: bytes,
        parent_class: tree_sitter.Node | None = None,
    ) -> list[dict[str, Any]]:
        """Extract method definitions."""
        functions = []
        root = parent_class if parent_class else tree.root_node

        # Find method declarations and constructors
        function_nodes = self.find_nodes_by_type(
            root, "method_declaration", max_depth=4
        )
        constructor_nodes = self.find_nodes_by_type(
            root, "constructor_declaration", max_depth=4
        )
        function_nodes.extend(constructor_nodes)

        for node in function_nodes:
            func_data: dict[str, Any] = {
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

            # Extract modifiers and annotations
            is_static, decorators = self._extract_method_modifiers(node, content)
            func_data["is_staticmethod"] = is_static
            func_data["is_static"] = is_static
            func_data["decorators"] = decorators

            # Extract method signature
            name, return_type, parameters = self._extract_method_signature(
                node, content
            )
            func_data["name"] = name
            func_data["return_type"] = return_type
            func_data["parameters"] = parameters

            # For constructors, resolve name from parent class
            if node.type == "constructor_declaration" and not func_data["name"]:
                func_data["name"] = self._resolve_constructor_name(node, content)

            # Extract docstring and complexity
            func_data["docstring"] = self._get_javadoc(node, content)
            func_data["complexity"] = self.complexity_calculator.calculate_complexity(
                node, content
            )

            functions.append(func_data)

        return functions

    def _extract_superclass(self, node: tree_sitter.Node, content: bytes) -> list[str]:
        """Extract superclass from superclass node."""
        return [
            self.get_node_text(child, content)
            for child in node.children
            if child.type == "type_identifier"
        ]

    def _extract_java_interfaces(
        self, node: tree_sitter.Node, content: bytes
    ) -> list[str]:
        """Extract interfaces from interfaces node."""
        return [
            self.get_node_text(type_child, content)
            for child in node.children
            if child.type == "type_list"
            for type_child in child.children
            if type_child.type == "type_identifier"
        ]

    def _process_java_modifiers(
        self, node: tree_sitter.Node, content: bytes
    ) -> tuple[bool, list[str]]:
        """Process Java class modifiers and annotations."""
        is_abstract = False
        decorators = []

        modifiers_text = self.get_node_text(node, content)
        if "abstract" in modifiers_text:
            is_abstract = True

        # Extract annotations line by line
        for line in modifiers_text.split("\n"):
            stripped_line = line.strip()
            if stripped_line.startswith("@"):
                annotation = stripped_line[1:].split("(")[0]
                decorators.append(annotation)

        return is_abstract, decorators

    def _collect_class_nodes(self, tree: tree_sitter.Tree) -> list[tree_sitter.Node]:
        """Collect all class-like nodes (classes, interfaces, enums)."""
        class_nodes = self.find_nodes_by_type(tree.root_node, "class_declaration")
        interface_nodes = self.find_nodes_by_type(
            tree.root_node, "interface_declaration"
        )
        enum_nodes = self.find_nodes_by_type(tree.root_node, "enum_declaration")
        return class_nodes + interface_nodes + enum_nodes

    def extract_classes(
        self,
        tree: tree_sitter.Tree,
        content: bytes,
    ) -> list[dict[str, Any]]:
        """Extract class definitions."""
        classes = []
        all_class_nodes = self._collect_class_nodes(tree)

        for node in all_class_nodes:
            class_data: dict[str, Any]

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

            # Process child nodes
            for child in node.children:
                if child.type == "identifier":
                    class_data["name"] = self.get_node_text(child, content)
                elif child.type == "superclass":
                    class_data["base_classes"].extend(
                        self._extract_superclass(child, content)
                    )
                elif child.type == "interfaces":
                    class_data["base_classes"].extend(
                        self._extract_java_interfaces(child, content)
                    )
                elif child.type == "modifiers":
                    is_abstract, decorators = self._process_java_modifiers(
                        child, content
                    )
                    class_data["is_abstract"] = is_abstract
                    class_data["decorators"] = decorators

            # Extract docstring and methods
            class_data["docstring"] = self._get_javadoc(node, content)
            class_data["methods"] = self.extract_functions(tree, content, node)

            classes.append(class_data)

        return classes

    def extract_module_info(
        self,
        tree: tree_sitter.Tree,
        content: bytes,
    ) -> dict[str, Any]:
        """Extract module-level information."""
        module_info = {
            "docstring": self._get_module_docstring(tree, content),
            "imports": self.extract_imports(tree, content),
            "classes": self.extract_classes(tree, content),
            "functions": self.extract_functions(tree, content),
        }

        # Extract package declaration
        package_nodes = self.find_nodes_by_type(tree.root_node, "package_declaration")
        if package_nodes:
            package_node = package_nodes[0]  # Should only be one package declaration
            for child in package_node.children:
                if child.type == "scoped_identifier":
                    module_info["package"] = self.get_node_text(child, content)
                    break

        return module_info

    def _extract_parameters(
        self,
        params_node: tree_sitter.Node,
        content: bytes,
    ) -> list[dict[str, Any]]:
        """Extract function parameters."""
        parameters: list[dict[str, Any]] = []

        for child in params_node.children:
            if child.type in ("formal_parameter", "spread_parameter"):
                param_data: dict[str, Any] = {
                    "name": None,
                    "type": None,
                    "default": None,
                }

                # Handle varargs (spread_parameter)
                if child.type == "spread_parameter":
                    # Varargs parameter has ... before the type
                    for subchild in child.children:
                        if subchild.type == "variable_declarator":
                            param_data["name"] = self.get_node_text(subchild, content)
                        elif subchild.type in (
                            "type_identifier",
                            "integral_type",
                            "primitive_type",
                            "array_type",
                            "generic_type",
                        ):
                            # Add ... to indicate varargs
                            param_data["type"] = (
                                self.get_node_text(subchild, content) + "..."
                            )
                else:
                    # Regular parameter
                    found_identifier = False
                    for subchild in child.children:
                        if subchild.type == "identifier":
                            param_data["name"] = self.get_node_text(subchild, content)
                            found_identifier = True
                        elif subchild.type in (
                            "type_identifier",
                            "integral_type",
                            "primitive_type",
                            "array_type",
                            "generic_type",
                        ):
                            # Type comes before identifier in Java
                            if not found_identifier:
                                param_data["type"] = self.get_node_text(
                                    subchild, content
                                )

                if param_data["name"]:
                    parameters.append(param_data)

        return parameters

    def _get_module_docstring(
        self,
        tree: tree_sitter.Tree,
        content: bytes,
    ) -> str | None:
        """Extract module-level docstring (first JavaDoc comment)."""
        # In Java, look for the first block comment in the file
        for child in tree.root_node.children:
            if child.type == "block_comment":
                text = child.text
                if text is not None and text.startswith(b"/**"):
                    return self._parse_javadoc(self.get_node_text(child, content))
        return None

    def _get_javadoc(
        self,
        node: tree_sitter.Node,
        content: bytes,
    ) -> str | None:
        """Extract JavaDoc comment."""
        # Look for comment just before the node
        prev_sibling = node.prev_sibling
        if prev_sibling and prev_sibling.type == "block_comment":
            text = prev_sibling.text
            if text is not None and text.startswith(b"/**"):
                return self._parse_javadoc(self.get_node_text(prev_sibling, content))
        return None

    def _parse_javadoc(self, javadoc: str) -> str:
        """Parse JavaDoc to extract description."""
        lines = javadoc.strip().split("\n")
        if not lines:
            return ""

        # Remove /** and */
        if lines[0].strip() == "/**":
            lines = lines[1:]
        if lines and lines[-1].strip() == "*/":
            lines = lines[:-1]

        # Remove leading asterisks and spaces
        cleaned_lines = []
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith("*"):
                stripped_line = stripped_line[1:].strip()
            # Skip annotation lines like @param, @return
            if not stripped_line.startswith("@"):
                cleaned_lines.append(stripped_line)

        return " ".join(cleaned_lines).strip()

    def extract_references(
        self, tree: tree_sitter.Tree, content: bytes
    ) -> list[dict[str, Any]]:
        """Extract references from Java code."""
        references = []

        # Extract import references
        references.extend(
            [
                {
                    "type": "import",
                    "source": "module",
                    "target": imp["imported_from"],
                    "line": imp["line_number"],
                }
                for imp in self.extract_imports(tree, content)
                if imp["imported_from"]
            ]
        )

        # Extract class inheritance and interface implementation references
        classes = self.extract_classes(tree, content)
        for cls in classes:
            # Base class and interface references
            for base in cls.get("base_classes", []):
                ref_type = (
                    "implement"
                    if "interface" in cls.get("name", "").lower()
                    else "inherit"
                )
                references.append(
                    {
                        "type": ref_type,
                        "source": cls["name"],
                        "target": base,
                        "line": cls["start_line"],
                    }
                )

        # Extract type references from methods
        for cls in classes:
            for method in cls.get("methods", []):
                # Return type references
                if method.get("return_type") and method["return_type"] not in [
                    "void",
                    "int",
                    "long",
                    "short",
                    "byte",
                    "float",
                    "double",
                    "boolean",
                    "char",
                ]:
                    references.append(
                        {
                            "type": "type_use",
                            "source": f"{cls['name']}.{method['name']}",
                            "target": method["return_type"],
                            "line": method["start_line"],
                        }
                    )

                # Parameter type references
                for param in method.get("parameters", []):
                    if param.get("type") and param["type"] not in [
                        "int",
                        "long",
                        "short",
                        "byte",
                        "float",
                        "double",
                        "boolean",
                        "char",
                    ]:
                        # Handle generic types and arrays
                        param_type = (
                            param["type"]
                            .split("<")[0]
                            .split("[")[0]
                            .removesuffix("...")
                        )
                        if param_type not in ["String"]:
                            references.append(
                                {
                                    "type": "type_use",
                                    "source": f"{cls['name']}.{method['name']}",
                                    "target": param_type,
                                    "line": method["start_line"],
                                }
                            )

        return references


class TypeScriptParser(TreeSitterParser):
    """TypeScript-specific TreeSitter parser."""

    def __init__(self) -> None:
        if not TYPESCRIPT_AVAILABLE or tstypescript is None:
            msg = "tree-sitter-typescript not available. Install with: pip install tree-sitter-typescript"
            raise ImportError(msg)
        super().__init__(tree_sitter.Language(tstypescript.language_typescript()))

    def extract_module_info(
        self,
        tree: tree_sitter.Tree,
        content: bytes,
    ) -> dict[str, Any]:
        """Extract module information from TypeScript code."""
        return {
            "classes": self._extract_classes(tree, content),
            "functions": self._extract_functions(tree, content),
            "imports": self._extract_imports(tree, content),
            "exports": self._extract_exports(tree, content),
            "interfaces": self._extract_interfaces(tree, content),
            "types": self._extract_type_aliases(tree, content),
            "enums": self._extract_enums(tree, content),
        }

    def _extract_classes(
        self,
        tree: tree_sitter.Tree,
        content: bytes,
    ) -> list[dict[str, Any]]:
        """Extract class definitions from TypeScript code."""
        classes = []
        query = tree_sitter.Query(
            self.require_language(),
            """
            (class_declaration
                name: (type_identifier) @class_name
                body: (class_body) @class_body) @class
            """,
        )

        for match in query.matches(tree.root_node):
            captures = match[1]

            if "class_name" in captures and "class_body" in captures:
                class_node = captures["class"][0]
                class_name = self.get_node_text(captures["class_name"][0], content)

                class_info = {
                    "name": class_name,
                    "start_line": class_node.start_point[0] + 1,
                    "end_line": class_node.end_point[0] + 1,
                    "methods": self._extract_class_methods(
                        captures["class_body"][0], content
                    ),
                    "properties": self._extract_class_properties(
                        captures["class_body"][0], content
                    ),
                    "constructors": self._extract_constructors(
                        captures["class_body"][0], content
                    ),
                    "base_classes": self._extract_extends_clause(class_node, content),
                    "interfaces": self._extract_implements_clause(class_node, content),
                    "decorators": self._extract_decorators(class_node, content),
                    "access_modifier": self._extract_class_modifiers(
                        class_node, content
                    ),
                }
                classes.append(class_info)

        return classes

    def _extract_functions(
        self,
        tree: tree_sitter.Tree,
        content: bytes,
    ) -> list[dict[str, Any]]:
        """Extract function definitions from TypeScript code."""
        functions = []

        # Function declarations
        query = tree_sitter.Query(
            self.require_language(),
            """
            (function_declaration
                name: (identifier) @function_name
                parameters: (formal_parameters) @params
                body: (statement_block) @body) @function
            """,
        )

        for match in query.matches(tree.root_node):
            captures = match[1]

            if "function_name" in captures:
                func_node = captures["function"][0]
                func_name = self.get_node_text(captures["function_name"][0], content)

                params_nodes = captures.get("params")
                params_node = params_nodes[0] if params_nodes else None
                body_nodes = captures.get("body")
                body_node = body_nodes[0] if body_nodes else func_node

                func_info = {
                    "name": func_name,
                    "start_line": func_node.start_point[0] + 1,
                    "end_line": func_node.end_point[0] + 1,
                    "parameters": self._extract_ts_parameters(params_node, content),
                    "return_type": self._extract_return_type(func_node, content),
                    "access_modifier": "public",  # Default for functions
                    "decorators": self._extract_decorators(func_node, content),
                    "is_async": self._is_async_function(func_node, content),
                    "complexity": ComplexityCalculator(
                        "typescript"
                    ).calculate_complexity(body_node, content),
                }
                functions.append(func_info)

        # Arrow functions assigned to variables
        arrow_query = tree_sitter.Query(
            self.require_language(),
            """
            (variable_declaration
                (variable_declarator
                    name: (identifier) @var_name
                    value: (arrow_function) @arrow_func))
            """,
        )

        for match in arrow_query.matches(tree.root_node):
            captures = match[1]

            if "var_name" in captures and "arrow_func" in captures:
                var_name = self.get_node_text(captures["var_name"][0], content)
                arrow_node = captures["arrow_func"][0]

                func_info = {
                    "name": var_name,
                    "start_line": arrow_node.start_point[0] + 1,
                    "end_line": arrow_node.end_point[0] + 1,
                    "parameters": self._extract_arrow_parameters(arrow_node, content),
                    "return_type": self._extract_arrow_return_type(arrow_node, content),
                    "access_modifier": "public",
                    "decorators": [],
                    "is_async": self._is_async_function(arrow_node, content),
                    "complexity": ComplexityCalculator(
                        "typescript"
                    ).calculate_complexity(arrow_node, content),
                }
                functions.append(func_info)

        return functions

    def _extract_imports(
        self,
        tree: tree_sitter.Tree,
        content: bytes,
    ) -> list[dict[str, Any]]:
        """Extract import statements from TypeScript code."""
        imports = []
        query = tree_sitter.Query(
            self.require_language(),
            """
            (import_statement
                source: (string) @source) @import
            """,
        )

        for match in query.matches(tree.root_node):
            captures = match[1]

            if "source" in captures:
                import_node = captures["import"][0]
                source = self.get_node_text(captures["source"][0], content).strip("\"'")

                import_info = {
                    "module": source,
                    "line": import_node.start_point[0] + 1,
                    "imports": self._extract_import_specifiers(import_node, content),
                    "is_default": self._has_default_import(import_node, content),
                    "is_namespace": self._has_namespace_import(import_node, content),
                }
                imports.append(import_info)

        return imports

    def _extract_exports(
        self,
        tree: tree_sitter.Tree,
        content: bytes,
    ) -> list[dict[str, Any]]:
        """Extract export statements from TypeScript code."""
        exports = []
        query = tree_sitter.Query(
            self.require_language(),
            """
            (export_statement) @export
            """,
        )

        for match in query.matches(tree.root_node):
            captures = match[1]
            export_node = captures["export"][0]

            export_info = {
                "line": export_node.start_point[0] + 1,
                "type": self._get_export_type(export_node, content),
                "name": self._get_export_name(export_node, content),
                "is_default": self._is_default_export(export_node, content),
            }
            exports.append(export_info)

        return exports

    def _extract_interfaces(
        self,
        tree: tree_sitter.Tree,
        content: bytes,
    ) -> list[dict[str, Any]]:
        """Extract interface definitions from TypeScript code."""
        interfaces = []
        query = tree_sitter.Query(
            self.require_language(),
            """
            (interface_declaration
                name: (type_identifier) @interface_name
                body: (object_type) @interface_body) @interface
            """,
        )

        for match in query.matches(tree.root_node):
            captures = match[1]

            if "interface_name" in captures:
                interface_node = captures["interface"][0]
                interface_name = self.get_node_text(
                    captures["interface_name"][0], content
                )

                interface_body_nodes = captures.get("interface_body")
                interface_body_node = (
                    interface_body_nodes[0] if interface_body_nodes else None
                )

                interface_info = {
                    "name": interface_name,
                    "start_line": interface_node.start_point[0] + 1,
                    "end_line": interface_node.end_point[0] + 1,
                    "properties": self._extract_interface_properties(
                        interface_body_node, content
                    ),
                    "methods": self._extract_interface_methods(
                        interface_body_node, content
                    ),
                    "extends": self._extract_interface_extends(interface_node, content),
                }
                interfaces.append(interface_info)

        return interfaces

    def _extract_type_aliases(
        self,
        tree: tree_sitter.Tree,
        content: bytes,
    ) -> list[dict[str, Any]]:
        """Extract type alias definitions from TypeScript code."""
        types = []
        query = tree_sitter.Query(
            self.require_language(),
            """
            (type_alias_declaration
                name: (type_identifier) @type_name) @type_alias
            """,
        )

        for match in query.matches(tree.root_node):
            captures = match[1]

            if "type_name" in captures:
                type_node = captures["type_alias"][0]
                type_name = self.get_node_text(captures["type_name"][0], content)

                type_info = {
                    "name": type_name,
                    "line": type_node.start_point[0] + 1,
                    "definition": self._extract_type_definition(type_node, content),
                }
                types.append(type_info)

        return types

    def _extract_enums(
        self,
        tree: tree_sitter.Tree,
        content: bytes,
    ) -> list[dict[str, Any]]:
        """Extract enum definitions from TypeScript code."""
        enums = []
        query = tree_sitter.Query(
            self.require_language(),
            """
            (enum_declaration
                name: (identifier) @enum_name
                body: (enum_body) @enum_body) @enum
            """,
        )

        for match in query.matches(tree.root_node):
            captures = match[1]

            if "enum_name" in captures:
                enum_node = captures["enum"][0]
                enum_name = self.get_node_text(captures["enum_name"][0], content)

                enum_body_nodes = captures.get("enum_body")
                enum_body_node = enum_body_nodes[0] if enum_body_nodes else None

                enum_info = {
                    "name": enum_name,
                    "start_line": enum_node.start_point[0] + 1,
                    "end_line": enum_node.end_point[0] + 1,
                    "members": self._extract_enum_members(enum_body_node, content),
                }
                enums.append(enum_info)

        return enums

    def _extract_ts_parameters(
        self,
        params_node: tree_sitter.Node | None,
        content: bytes,
    ) -> list[dict[str, Any]]:
        """Extract function parameters with TypeScript types."""
        if not params_node:
            return []

        parameters = []
        for child in params_node.children:
            if child.type in ["required_parameter", "optional_parameter"]:
                param_data: dict[str, Any] = {
                    "name": None,
                    "type": None,
                    "default": None,
                    "optional": child.type == "optional_parameter",
                }

                for subchild in child.children:
                    if subchild.type == "identifier":
                        param_data["name"] = self.get_node_text(subchild, content)
                    elif subchild.type == "type_annotation":
                        param_data["type"] = self._extract_type_annotation(
                            subchild, content
                        )
                    elif subchild.type in ["number", "string", "true", "false", "null"]:
                        param_data["default"] = self.get_node_text(subchild, content)

                if param_data["name"]:
                    parameters.append(param_data)

        return parameters

    def _extract_type_annotation(
        self,
        type_node: tree_sitter.Node,
        content: bytes,
    ) -> str:
        """Extract TypeScript type annotation."""
        # Find the actual type within the type annotation
        for child in type_node.children:
            if child.type != ":":
                return self.get_node_text(child, content)
        return "any"

    def _extract_class_methods(
        self, body_node: tree_sitter.Node | None, content: bytes
    ) -> list[dict[str, Any]]:
        """Extract methods from TypeScript class body."""
        if not body_node:
            return []

        methods = []
        for child in body_node.children:
            if child.type == "method_definition":
                method_name = None
                method_params = []
                return_type = "any"
                access_modifier = "public"
                is_static = False

                for subchild in child.children:
                    if subchild.type == "property_identifier":
                        method_name = self.get_node_text(subchild, content)
                    elif subchild.type == "formal_parameters":
                        method_params = self._extract_ts_parameters(subchild, content)
                    elif subchild.type == "type_annotation":
                        return_type = self._extract_type_annotation(subchild, content)
                    elif subchild.type in ["public", "private", "protected"]:
                        access_modifier = subchild.type
                    elif subchild.type == "static":
                        is_static = True

                if method_name:
                    methods.append(
                        {
                            "name": method_name,
                            "parameters": method_params,
                            "return_type": return_type,
                            "access_modifier": access_modifier,
                            "is_static": is_static,
                            "start_line": child.start_point[0] + 1,
                            "end_line": child.end_point[0] + 1,
                            "complexity": ComplexityCalculator(
                                "typescript"
                            ).calculate_complexity(child, content),
                        }
                    )

        return methods

    def _extract_class_properties(
        self, body_node: tree_sitter.Node | None, content: bytes
    ) -> list[dict[str, Any]]:
        """Extract properties from TypeScript class body."""
        if not body_node:
            return []

        properties = []
        for child in body_node.children:
            if child.type == "field_definition":
                prop_name = None
                prop_type = "any"
                access_modifier = "public"
                is_static = False

                for subchild in child.children:
                    if subchild.type == "property_identifier":
                        prop_name = self.get_node_text(subchild, content)
                    elif subchild.type == "type_annotation":
                        prop_type = self._extract_type_annotation(subchild, content)
                    elif subchild.type in ["public", "private", "protected"]:
                        access_modifier = subchild.type
                    elif subchild.type == "static":
                        is_static = True

                if prop_name:
                    properties.append(
                        {
                            "name": prop_name,
                            "type": prop_type,
                            "access_modifier": access_modifier,
                            "is_static": is_static,
                        }
                    )

        return properties

    def _extract_constructors(
        self, body_node: tree_sitter.Node | None, content: bytes
    ) -> list[dict[str, Any]]:
        """Extract constructors from TypeScript class body."""
        if not body_node:
            return []

        constructors = []
        for child in body_node.children:
            if child.type == "method_definition":
                # Check if this is a constructor
                for subchild in child.children:
                    if (
                        subchild.type == "property_identifier"
                        and self.get_node_text(subchild, content) == "constructor"
                    ):
                        constructor_params = []
                        access_modifier = "public"

                        for param_child in child.children:
                            if param_child.type == "formal_parameters":
                                constructor_params = self._extract_ts_parameters(
                                    param_child, content
                                )
                            elif param_child.type in ["public", "private", "protected"]:
                                access_modifier = param_child.type

                        constructors.append(
                            {
                                "parameters": constructor_params,
                                "access_modifier": access_modifier,
                                "start_line": child.start_point[0] + 1,
                                "end_line": child.end_point[0] + 1,
                            }
                        )
                        break

        return constructors

    def _extract_extends_clause(
        self, class_node: tree_sitter.Node, content: bytes
    ) -> list[str]:
        """Extract extends clause from TypeScript class."""
        for child in class_node.children:
            if child.type == "class_heritage":
                for heritage_child in child.children:
                    if heritage_child.type == "extends_clause":
                        for extends_child in heritage_child.children:
                            if extends_child.type == "type_identifier":
                                return [self.get_node_text(extends_child, content)]
        return []

    def _extract_implements_clause(
        self, class_node: tree_sitter.Node, content: bytes
    ) -> list[str]:
        """Extract implements clause from TypeScript class."""
        interfaces = []
        for child in class_node.children:
            if child.type == "class_heritage":
                for heritage_child in child.children:
                    if heritage_child.type == "implements_clause":
                        interfaces.extend(
                            [
                                self.get_node_text(impl_child, content)
                                for impl_child in heritage_child.children
                                if impl_child.type == "type_identifier"
                            ]
                        )
        return interfaces

    def _extract_decorators(self, node: tree_sitter.Node, content: bytes) -> list[str]:
        """Extract decorators from TypeScript node."""
        decorators = []
        for child in node.children:
            if child.type == "decorator":
                decorator_name = ""
                for dec_child in child.children:
                    if dec_child.type == "identifier":
                        decorator_name = self.get_node_text(dec_child, content)
                        break
                if decorator_name:
                    decorators.append(decorator_name)
        return decorators

    def _extract_class_modifiers(self, node: tree_sitter.Node, _: bytes) -> str:
        """Extract class access modifiers."""
        for child in node.children:
            if child.type in ["public", "private", "protected"]:
                return child.type
        return "public"

    def _extract_return_type(self, func_node: tree_sitter.Node, content: bytes) -> str:
        """Extract return type from TypeScript function."""
        for child in func_node.children:
            if child.type == "type_annotation":
                return self._extract_type_annotation(child, content)
        return "any"

    def _is_async_function(self, func_node: tree_sitter.Node, _: bytes) -> bool:
        """Check if function is async."""
        return any(child.type == "async" for child in func_node.children)

    def _extract_arrow_parameters(
        self, arrow_node: tree_sitter.Node, content: bytes
    ) -> list[dict[str, Any]]:
        """Extract parameters from arrow function."""
        for child in arrow_node.children:
            if child.type == "formal_parameters":
                return self._extract_ts_parameters(child, content)
            if child.type == "identifier":
                # Single parameter without parentheses
                return [
                    {
                        "name": self.get_node_text(child, content),
                        "type": "any",
                        "default": None,
                        "optional": False,
                    }
                ]
        return []

    def _extract_arrow_return_type(
        self, arrow_node: tree_sitter.Node, content: bytes
    ) -> str:
        """Extract return type from arrow function."""
        for child in arrow_node.children:
            if child.type == "type_annotation":
                return self._extract_type_annotation(child, content)
        return "any"

    def _extract_import_specifiers(
        self, import_node: tree_sitter.Node, content: bytes
    ) -> list[str]:
        """Extract import specifiers from import statement."""
        imports = []
        for child in import_node.children:
            if child.type == "import_clause":
                for clause_child in child.children:
                    if clause_child.type == "named_imports":
                        for named_child in clause_child.children:
                            if named_child.type == "import_specifier":
                                imports.extend(
                                    [
                                        self.get_node_text(spec_child, content)
                                        for spec_child in named_child.children
                                        if spec_child.type == "identifier"
                                    ]
                                )
                    elif clause_child.type == "identifier":
                        imports.append(self.get_node_text(clause_child, content))
        return imports

    def _has_default_import(self, import_node: tree_sitter.Node, _: bytes) -> bool:
        """Check if import has default import."""
        for child in import_node.children:
            if child.type == "import_clause":
                for clause_child in child.children:
                    if clause_child.type == "identifier":
                        return True
        return False

    def _has_namespace_import(self, import_node: tree_sitter.Node, _: bytes) -> bool:
        """Check if import has namespace import."""
        for child in import_node.children:
            if child.type == "import_clause":
                for clause_child in child.children:
                    if clause_child.type == "namespace_import":
                        return True
        return False

    def _get_export_type(self, export_node: tree_sitter.Node, _: bytes) -> str:
        """Get export type."""
        for child in export_node.children:
            if child.type == "default":
                return "default"
        return "named"

    def _get_export_name(self, export_node: tree_sitter.Node, content: bytes) -> str:
        """Get export name."""
        for child in export_node.children:
            if child.type == "identifier":
                return self.get_node_text(child, content)
            if child.type == "class_declaration":
                for class_child in child.children:
                    if class_child.type == "type_identifier":
                        return self.get_node_text(class_child, content)
        return ""

    def _is_default_export(self, export_node: tree_sitter.Node, content: bytes) -> bool:
        """Check if export is default export."""
        return self._get_export_type(export_node, content) == "default"

    def _extract_interface_properties(
        self, body_node: tree_sitter.Node | None, content: bytes
    ) -> list[dict[str, Any]]:
        """Extract properties from TypeScript interface."""
        if not body_node:
            return []

        properties = []
        for child in body_node.children:
            if child.type == "property_signature":
                prop_name = None
                prop_type = "any"
                optional = False

                for prop_child in child.children:
                    if prop_child.type == "property_identifier":
                        prop_name = self.get_node_text(prop_child, content)
                    elif prop_child.type == "type_annotation":
                        prop_type = self._extract_type_annotation(prop_child, content)
                    elif prop_child.type == "?":
                        optional = True

                if prop_name:
                    properties.append(
                        {
                            "name": prop_name,
                            "type": prop_type,
                            "optional": optional,
                        }
                    )

        return properties

    def _extract_interface_methods(
        self, body_node: tree_sitter.Node | None, content: bytes
    ) -> list[dict[str, Any]]:
        """Extract methods from TypeScript interface."""
        if not body_node:
            return []

        methods = []
        for child in body_node.children:
            if child.type == "method_signature":
                method_name = None
                method_params = []
                return_type = "any"

                for method_child in child.children:
                    if method_child.type == "property_identifier":
                        method_name = self.get_node_text(method_child, content)
                    elif method_child.type == "formal_parameters":
                        method_params = self._extract_ts_parameters(
                            method_child, content
                        )
                    elif method_child.type == "type_annotation":
                        return_type = self._extract_type_annotation(
                            method_child, content
                        )

                if method_name:
                    methods.append(
                        {
                            "name": method_name,
                            "parameters": method_params,
                            "return_type": return_type,
                        }
                    )

        return methods

    def _extract_interface_extends(
        self, interface_node: tree_sitter.Node, content: bytes
    ) -> list[str]:
        """Extract extends clause from TypeScript interface."""
        extends = []
        for child in interface_node.children:
            if child.type == "extends_clause":
                extends.extend(
                    [
                        self.get_node_text(extends_child, content)
                        for extends_child in child.children
                        if extends_child.type == "type_identifier"
                    ]
                )
        return extends

    def _extract_type_definition(
        self, type_node: tree_sitter.Node, content: bytes
    ) -> str:
        """Extract type definition from TypeScript type alias."""
        for child in type_node.children:
            if child.type not in ["type", "type_identifier", "="]:
                return self.get_node_text(child, content)
        return "any"

    def _extract_enum_members(
        self, body_node: tree_sitter.Node | None, content: bytes
    ) -> list[dict[str, Any]]:
        """Extract members from TypeScript enum."""
        if not body_node:
            return []

        members = []
        for child in body_node.children:
            if child.type == "property_identifier":
                member_name = self.get_node_text(child, content)
                members.append(
                    {
                        "name": member_name,
                        "value": None,  # Could be enhanced to extract values
                    }
                )

        return members

    def extract_references(
        self,
        tree: tree_sitter.Tree,
        content: bytes,
    ) -> list[dict[str, Any]]:
        """Extract code references from TypeScript code."""
        references = []

        # Extract module information for context
        module_info = self.extract_module_info(tree, content)

        # Extract type references from interfaces
        for interface in module_info.get("interfaces", []):
            # Interface inheritance references
            references.extend(
                [
                    {
                        "type": "interface_extends",
                        "source": interface["name"],
                        "target": extended,
                        "line": interface["start_line"],
                    }
                    for extended in interface.get("extends", [])
                ]
            )

            # Property type references
            for prop in interface.get("properties", []):
                if prop.get("type") and prop["type"] not in [
                    "string",
                    "number",
                    "boolean",
                    "any",
                    "void",
                    "null",
                    "undefined",
                ]:
                    # Clean up generic types and arrays
                    prop_type = prop["type"].split("<")[0].split("[")[0]
                    references.append(
                        {
                            "type": "type_use",
                            "source": f"{interface['name']}.{prop['name']}",
                            "target": prop_type,
                            "line": interface["start_line"],
                        }
                    )

        # Extract type references from classes
        for cls in module_info.get("classes", []):
            # Class inheritance references
            references.extend(
                [
                    {
                        "type": "inherit",
                        "source": cls["name"],
                        "target": base,
                        "line": cls["start_line"],
                    }
                    for base in cls.get("base_classes", [])
                ]
            )

            # Interface implementation references
            references.extend(
                [
                    {
                        "type": "implement",
                        "source": cls["name"],
                        "target": impl,
                        "line": cls["start_line"],
                    }
                    for impl in cls.get("interfaces", [])
                ]
            )

        # Extract type references from functions
        for func in module_info.get("functions", []):
            # Return type references
            if func.get("return_type") and func["return_type"] not in [
                "any",
                "void",
                "string",
                "number",
                "boolean",
            ]:
                return_type = func["return_type"].split("<")[0].split("[")[0]
                references.append(
                    {
                        "type": "type_use",
                        "source": func["name"],
                        "target": return_type,
                        "line": func["start_line"],
                    }
                )

            # Parameter type references
            for param in func.get("parameters", []):
                if param.get("type") and param["type"] not in [
                    "any",
                    "string",
                    "number",
                    "boolean",
                ]:
                    param_type = param["type"].split("<")[0].split("[")[0]
                    references.append(
                        {
                            "type": "type_use",
                            "source": func["name"],
                            "target": param_type,
                            "line": func["start_line"],
                        }
                    )

        # Extract import references
        for imp in module_info.get("imports", []):
            references.extend(
                [
                    {
                        "type": "import",
                        "source": "module",
                        "target": imported_item,
                        "line": imp["line"],
                        "module": imp["module"],
                    }
                    for imported_item in imp.get("imports", [])
                ]
            )

        return references


class JavaScriptParser(TreeSitterParser):
    """JavaScript-specific TreeSitter parser."""

    def __init__(self) -> None:
        if not JAVASCRIPT_AVAILABLE or tsjavascript is None:
            msg = "tree-sitter-javascript not available. Install with: pip install tree-sitter-javascript"
            raise ImportError(msg)
        super().__init__(tree_sitter.Language(tsjavascript.language()))

    def extract_module_info(
        self,
        tree: tree_sitter.Tree,
        content: bytes,
    ) -> dict[str, Any]:
        """Extract module information from JavaScript code."""
        return {
            "classes": self._extract_classes(tree, content),
            "functions": self._extract_functions(tree, content),
            "imports": self._extract_imports(tree, content),
            "exports": self._extract_exports(tree, content),
        }

    def _extract_classes(
        self,
        tree: tree_sitter.Tree,
        content: bytes,
    ) -> list[dict[str, Any]]:
        """Extract class definitions from JavaScript code."""
        classes = []
        query = tree_sitter.Query(
            self.require_language(),
            """
            (class_declaration
                name: (identifier) @class_name
                body: (class_body) @class_body) @class
            """,
        )

        for match in query.matches(tree.root_node):
            captures = match[1]

            if "class_name" in captures and "class_body" in captures:
                class_node = captures["class"][0]
                class_name = self.get_node_text(captures["class_name"][0], content)

                class_info = {
                    "name": class_name,
                    "start_line": class_node.start_point[0] + 1,
                    "end_line": class_node.end_point[0] + 1,
                    "text": self.get_node_text(class_node, content),
                    "methods": self._extract_js_class_methods(
                        captures["class_body"][0], content
                    ),
                    "properties": self._extract_js_class_properties(
                        captures["class_body"][0], content
                    ),
                    "constructors": self._extract_js_constructors(
                        captures["class_body"][0], content
                    ),
                    "base_classes": self._extract_js_extends_clause(
                        class_node, content
                    ),
                }
                classes.append(class_info)

        return classes

    def _extract_functions(
        self,
        tree: tree_sitter.Tree,
        content: bytes,
    ) -> list[dict[str, Any]]:
        """Extract function definitions from JavaScript code."""
        functions = []

        # Function declarations
        query = tree_sitter.Query(
            self.require_language(),
            """
            (function_declaration
                name: (identifier) @function_name
                parameters: (formal_parameters) @params
                body: (statement_block) @body) @function
            """,
        )

        for match in query.matches(tree.root_node):
            captures = match[1]

            if "function_name" in captures:
                func_node = captures["function"][0]
                func_name = self.get_node_text(captures["function_name"][0], content)

                params_nodes = captures.get("params")
                params_node = params_nodes[0] if params_nodes else None
                body_nodes = captures.get("body")
                body_node = body_nodes[0] if body_nodes else func_node

                func_info = {
                    "name": func_name,
                    "start_line": func_node.start_point[0] + 1,
                    "end_line": func_node.end_point[0] + 1,
                    "parameters": self._extract_js_parameters(params_node, content),
                    "is_async": self._is_async_function(func_node, content),
                    "complexity": ComplexityCalculator(
                        "javascript"
                    ).calculate_complexity(body_node, content),
                }
                functions.append(func_info)

        # Arrow functions assigned to variables (handles both var/let/const and lexical declarations)
        arrow_query = tree_sitter.Query(
            self.require_language(),
            """
            (variable_declaration
                (variable_declarator
                    name: (identifier) @var_name
                    value: (arrow_function) @arrow_func))
            """,
        )

        for match in arrow_query.matches(tree.root_node):
            captures = match[1]

            if "var_name" in captures and "arrow_func" in captures:
                var_name = self.get_node_text(captures["var_name"][0], content)
                arrow_node = captures["arrow_func"][0]

                func_info = {
                    "name": var_name,
                    "start_line": arrow_node.start_point[0] + 1,
                    "end_line": arrow_node.end_point[0] + 1,
                    "parameters": self._extract_js_arrow_parameters(
                        arrow_node, content
                    ),
                    "is_async": self._is_async_function(arrow_node, content),
                    "complexity": ComplexityCalculator(
                        "javascript"
                    ).calculate_complexity(arrow_node, content),
                }
                functions.append(func_info)

        # Also handle lexical_declaration forms produced by some grammars (e.g., 'const')
        lexical_arrow_query = tree_sitter.Query(
            self.require_language(),
            """
            (lexical_declaration
                (variable_declarator
                    name: (identifier) @var_name
                    value: (arrow_function) @arrow_func))
            """,
        )

        for match in lexical_arrow_query.matches(tree.root_node):
            captures = match[1]
            if "var_name" in captures and "arrow_func" in captures:
                var_name = self.get_node_text(captures["var_name"][0], content)
                arrow_node = captures["arrow_func"][0]

                func_info = {
                    "name": var_name,
                    "start_line": arrow_node.start_point[0] + 1,
                    "end_line": arrow_node.end_point[0] + 1,
                    "parameters": self._extract_js_arrow_parameters(
                        arrow_node, content
                    ),
                    "is_async": self._is_async_function(arrow_node, content),
                    "complexity": ComplexityCalculator(
                        "javascript"
                    ).calculate_complexity(arrow_node, content),
                }
                functions.append(func_info)

        return functions

    # Helper utilities for JavaScript exports/imports/async
    def _is_async_function(self, func_node: tree_sitter.Node, _: bytes) -> bool:
        """Check if function is async (JavaScript)."""
        return any(child.type == "async" for child in func_node.children)

    def _has_default_import(self, import_node: tree_sitter.Node, _: bytes) -> bool:
        """Check if import has default import (JavaScript)."""
        for child in import_node.children:
            if child.type == "import_clause":
                for clause_child in child.children:
                    if clause_child.type == "identifier":
                        return True
        return False

    def _has_namespace_import(self, import_node: tree_sitter.Node, _: bytes) -> bool:
        """Check if import has namespace import (JavaScript)."""
        for child in import_node.children:
            if child.type == "import_clause":
                for clause_child in child.children:
                    if clause_child.type == "namespace_import":
                        return True
        return False

    def _get_export_type(self, export_node: tree_sitter.Node, _: bytes) -> str:
        """Get export type (JavaScript)."""
        for child in export_node.children:
            if child.type == "default":
                return "default"
        return "named"

    def _get_export_name(self, export_node: tree_sitter.Node, content: bytes) -> str:
        """Get export name (JavaScript)."""
        for child in export_node.children:
            if child.type == "identifier":
                return self.get_node_text(child, content)
            if child.type == "class_declaration":
                for class_child in child.children:
                    if class_child.type == "identifier":
                        return self.get_node_text(class_child, content)
        return ""

    def _is_default_export(self, export_node: tree_sitter.Node, content: bytes) -> bool:
        """Check if export is default export (JavaScript)."""
        return self._get_export_type(export_node, content) == "default"

    def _extract_imports(
        self,
        tree: tree_sitter.Tree,
        content: bytes,
    ) -> list[dict[str, Any]]:
        """Extract import statements from JavaScript code."""
        imports = []
        query = tree_sitter.Query(
            self.require_language(),
            """
            (import_statement
                source: (string) @source) @import
            """,
        )

        for match in query.matches(tree.root_node):
            captures = match[1]

            if "source" in captures:
                import_node = captures["import"][0]
                source = self.get_node_text(captures["source"][0], content).strip("\"'")

                import_info = {
                    "module": source,
                    "line": import_node.start_point[0] + 1,
                    "imports": self._extract_js_import_specifiers(import_node, content),
                    "is_default": self._has_default_import(import_node, content),
                    "is_namespace": self._has_namespace_import(import_node, content),
                }
                imports.append(import_info)

        return imports

    def _extract_exports(
        self,
        tree: tree_sitter.Tree,
        content: bytes,
    ) -> list[dict[str, Any]]:
        """Extract export statements from JavaScript code."""
        exports = []
        query = tree_sitter.Query(
            self.require_language(),
            """
            (export_statement) @export
            """,
        )

        for match in query.matches(tree.root_node):
            captures = match[1]
            export_node = captures["export"][0]

            export_info = {
                "line": export_node.start_point[0] + 1,
                "type": self._get_export_type(export_node, content),
                "name": self._get_export_name(export_node, content),
                "is_default": self._is_default_export(export_node, content),
            }
            exports.append(export_info)

        return exports

    def _extract_js_parameters(
        self,
        params_node: tree_sitter.Node | None,
        content: bytes,
    ) -> list[dict[str, Any]]:
        """Extract function parameters from JavaScript code."""
        if not params_node:
            return []

        parameters = []
        for child in params_node.children:
            if child.type == "identifier":
                param_data = {
                    "name": self.get_node_text(child, content),
                    "type": None,  # JavaScript doesn't have explicit types
                    "default": None,
                }
                parameters.append(param_data)
            elif child.type == "assignment_pattern":
                # Parameter with default value
                param_name = None
                default_value = None
                for subchild in child.children:
                    if subchild.type == "identifier":
                        param_name = self.get_node_text(subchild, content)
                    elif subchild.type != "=":
                        default_value = self.get_node_text(subchild, content)

                if param_name:
                    param_data = {
                        "name": param_name,
                        "type": None,
                        "default": default_value,
                    }
                    parameters.append(param_data)

        return parameters

    def _extract_js_class_methods(
        self, body_node: tree_sitter.Node | None, content: bytes
    ) -> list[dict[str, Any]]:
        """Extract methods from JavaScript class body."""
        if not body_node:
            return []

        methods = []
        for child in body_node.children:
            if child.type == "method_definition":
                method_name = None
                method_params = []
                is_static = False
                is_async = False

                for subchild in child.children:
                    if subchild.type == "property_identifier":
                        method_name = self.get_node_text(subchild, content)
                    elif subchild.type == "formal_parameters":
                        method_params = self._extract_js_parameters(subchild, content)
                    elif subchild.type == "static":
                        is_static = True
                    elif subchild.type == "async":
                        is_async = True

                if method_name:
                    methods.append(
                        {
                            "name": method_name,
                            "parameters": method_params,
                            "is_static": is_static,
                            "is_async": is_async,
                            "start_line": child.start_point[0] + 1,
                            "end_line": child.end_point[0] + 1,
                            "complexity": ComplexityCalculator(
                                "javascript"
                            ).calculate_complexity(child, content),
                        }
                    )

        return methods

    def _extract_js_class_properties(
        self, body_node: tree_sitter.Node | None, content: bytes
    ) -> list[dict[str, Any]]:
        """Extract properties from JavaScript class body."""
        if not body_node:
            return []

        properties = []
        for child in body_node.children:
            if child.type == "field_definition":
                prop_name = None
                is_static = False

                for subchild in child.children:
                    if subchild.type == "property_identifier":
                        prop_name = self.get_node_text(subchild, content)
                    elif subchild.type == "static":
                        is_static = True

                if prop_name:
                    properties.append(
                        {
                            "name": prop_name,
                            "is_static": is_static,
                        }
                    )

        return properties

    def _extract_js_constructors(
        self, body_node: tree_sitter.Node | None, content: bytes
    ) -> list[dict[str, Any]]:
        """Extract constructors from JavaScript class body."""
        if not body_node:
            return []

        constructors = []
        for child in body_node.children:
            if child.type == "method_definition":
                # Check if this is a constructor
                for subchild in child.children:
                    if (
                        subchild.type == "property_identifier"
                        and self.get_node_text(subchild, content) == "constructor"
                    ):
                        constructor_params = []

                        for param_child in child.children:
                            if param_child.type == "formal_parameters":
                                constructor_params = self._extract_js_parameters(
                                    param_child, content
                                )

                        constructors.append(
                            {
                                "parameters": constructor_params,
                                "start_line": child.start_point[0] + 1,
                                "end_line": child.end_point[0] + 1,
                            }
                        )
                        break

        return constructors

    def _extract_js_extends_clause(
        self, class_node: tree_sitter.Node, content: bytes
    ) -> list[str]:
        """Extract extends clause from JavaScript class."""
        for child in class_node.children:
            if child.type == "class_heritage":
                for heritage_child in child.children:
                    if heritage_child.type == "extends_clause":
                        for extends_child in heritage_child.children:
                            # Node types can vary between grammars (identifier, type_identifier, property_identifier, scoped_identifier)
                            if extends_child.type in {
                                "identifier",
                                "type_identifier",
                                "property_identifier",
                                "scoped_identifier",
                                "qualified_identifier",
                            }:
                                return [self.get_node_text(extends_child, content)]
        return []

    def _extract_js_arrow_parameters(
        self, arrow_node: tree_sitter.Node, content: bytes
    ) -> list[dict[str, Any]]:
        """Extract parameters from JavaScript arrow function."""
        for child in arrow_node.children:
            if child.type == "formal_parameters":
                return self._extract_js_parameters(child, content)
            if child.type == "identifier":
                # Single parameter without parentheses
                return [
                    {
                        "name": self.get_node_text(child, content),
                        "type": None,
                        "default": None,
                    }
                ]
        return []

    def _extract_js_import_specifiers(
        self, import_node: tree_sitter.Node, content: bytes
    ) -> list[str]:
        """Extract import specifiers from JavaScript import statement."""
        imports = []
        for child in import_node.children:
            if child.type == "import_clause":
                for clause_child in child.children:
                    if clause_child.type == "named_imports":
                        for named_child in clause_child.children:
                            if named_child.type == "import_specifier":
                                imports.extend(
                                    [
                                        self.get_node_text(spec_child, content)
                                        for spec_child in named_child.children
                                        if spec_child.type == "identifier"
                                    ]
                                )
                    elif clause_child.type == "identifier":
                        imports.append(self.get_node_text(clause_child, content))
        return imports

    def extract_references(
        self,
        tree: tree_sitter.Tree,
        content: bytes,
    ) -> list[dict[str, Any]]:
        """Extract code references from JavaScript code."""
        references = []

        # Extract module information for context
        module_info = self.extract_module_info(tree, content)

        # Extract class inheritance references
        for cls in module_info.get("classes", []):
            # Class inheritance references
            references.extend(
                [
                    {
                        "type": "inherit",
                        "source": cls["name"],
                        "target": base,
                        "line": cls["start_line"],
                    }
                    for base in cls.get("base_classes", [])
                ]
            )

        # Extract import references
        for imp in module_info.get("imports", []):
            references.extend(
                [
                    {
                        "type": "import",
                        "source": "module",
                        "target": imported_item,
                        "line": imp["line"],
                        "module": imp["module"],
                    }
                    for imported_item in imp.get("imports", [])
                ]
            )

        # Extract function call references (simplified)
        # This could be enhanced to analyze function calls within the AST
        references.extend(
            [
                {
                    "type": "function_definition",
                    "source": "module",
                    "target": func["name"],
                    "line": func["start_line"],
                }
                for func in module_info.get("functions", [])
            ]
        )

        return references
