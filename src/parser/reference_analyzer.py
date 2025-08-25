"""Analyze code references and dependencies."""

import ast
from pathlib import Path
from typing import Any

from src.logger import get_logger

logger = get_logger(__name__)


class ReferenceType:
    """Types of references between code entities."""

    IMPORT = "import"
    FUNCTION_CALL = "call"
    CLASS_INHERIT = "inherit"
    INSTANTIATE = "instantiate"
    TYPE_HINT = "type_hint"
    DECORATOR = "decorator"
    ATTRIBUTE = "attribute"


class ReferenceAnalyzer(ast.NodeVisitor):
    """Extract references between code entities using AST analysis."""

    def __init__(self, module_path: str, file_path: Path) -> None:
        """Initialize reference analyzer.

        Args:
            module_path: Python module path (e.g. 'src.parser.reference_analyzer')
            file_path: Path to the Python file
        """
        self.module_path = module_path
        self.file_path = file_path
        self.references: list[dict[str, Any]] = []
        self.current_class: str | None = None
        self.current_function: str | None = None
        self.imports: dict[str, str] = {}  # alias -> full module path

    def analyze(self, tree: ast.AST) -> list[dict[str, Any]]:
        """Analyze AST and extract references.

        Args:
            tree: Python AST

        Returns:
            List of reference dictionaries
        """
        self.references = []
        self.imports = {}
        self.visit(tree)
        return self.references

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        """Process import statements."""
        for alias in node.names:
            import_name = alias.name
            as_name = alias.asname or alias.name
            self.imports[as_name] = import_name

            self.references.append(
                {
                    "source_type": "module",
                    "source_name": self.module_path,
                    "source_line": node.lineno,
                    "target_type": "module",
                    "target_name": import_name,
                    "reference_type": ReferenceType.IMPORT,
                    "context": f"import {import_name}",
                }
            )

        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        """Process from...import statements."""
        module = node.module or ""
        level = node.level

        # Handle relative imports
        if level > 0:
            parts = self.module_path.split(".")
            if level <= len(parts):
                base_parts = parts[:-level]
                if module:
                    base_parts.append(module)
                module = ".".join(base_parts)

        for alias in node.names:
            import_name = alias.name
            as_name = alias.asname or alias.name

            if import_name == "*":
                # Star import
                self.imports[module] = module
                target_type = "module"
                target_name = module
            else:
                # Specific import
                full_name = f"{module}.{import_name}" if module else import_name
                self.imports[as_name] = full_name

                # Determine if it's likely a class/function based on naming
                target_type = "class" if import_name[0].isupper() else "function"
                target_name = full_name

            self.references.append(
                {
                    "source_type": self._get_current_source_type(),
                    "source_name": self._get_current_source_name(),
                    "source_line": node.lineno,
                    "target_type": target_type,
                    "target_name": target_name,
                    "reference_type": ReferenceType.IMPORT,
                    "context": ast.unparse(node),
                }
            )

        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        """Process class definitions."""
        old_class = self.current_class
        self.current_class = node.name

        # Analyze base classes
        for base in node.bases:
            base_name = self._get_name_from_node(base)
            if base_name:
                resolved_name = self._resolve_name(base_name)
                self.references.append(
                    {
                        "source_type": "class",
                        "source_name": f"{self.module_path}.{node.name}",
                        "source_line": node.lineno,
                        "target_type": "class",
                        "target_name": resolved_name,
                        "reference_type": ReferenceType.CLASS_INHERIT,
                        "context": f"class {node.name}({base_name})",
                    }
                )

        # Analyze decorators
        for decorator in node.decorator_list:
            dec_name = self._get_name_from_node(decorator)
            if dec_name:
                resolved_name = self._resolve_name(dec_name)
                self.references.append(
                    {
                        "source_type": "class",
                        "source_name": f"{self.module_path}.{node.name}",
                        "source_line": decorator.lineno,
                        "target_type": "function",  # Decorators are usually functions
                        "target_name": resolved_name,
                        "reference_type": ReferenceType.DECORATOR,
                        "context": f"@{dec_name}",
                    }
                )

        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        """Process function definitions."""
        old_function = self.current_function
        self.current_function = node.name

        # Analyze decorators
        for decorator in node.decorator_list:
            dec_name = self._get_name_from_node(decorator)
            if dec_name:
                resolved_name = self._resolve_name(dec_name)
                self.references.append(
                    {
                        "source_type": "function",
                        "source_name": self._get_current_source_name(),
                        "source_line": decorator.lineno,
                        "target_type": "function",
                        "target_name": resolved_name,
                        "reference_type": ReferenceType.DECORATOR,
                        "context": f"@{dec_name}",
                    }
                )

        # Analyze type hints
        if node.returns:
            type_name = self._get_name_from_node(node.returns)
            if type_name:
                resolved_name = self._resolve_name(type_name)
                self.references.append(
                    {
                        "source_type": "function",
                        "source_name": self._get_current_source_name(),
                        "source_line": node.lineno,
                        "target_type": "class",  # Type hints usually refer to classes
                        "target_name": resolved_name,
                        "reference_type": ReferenceType.TYPE_HINT,
                        "context": f"-> {type_name}",
                    }
                )

        # Analyze parameter type hints
        for arg in node.args.args:
            if arg.annotation:
                type_name = self._get_name_from_node(arg.annotation)
                if type_name:
                    resolved_name = self._resolve_name(type_name)
                    self.references.append(
                        {
                            "source_type": "function",
                            "source_name": self._get_current_source_name(),
                            "source_line": node.lineno,
                            "target_type": "class",
                            "target_name": resolved_name,
                            "reference_type": ReferenceType.TYPE_HINT,
                            "context": f"{arg.arg}: {type_name}",
                        }
                    )

        self.generic_visit(node)
        self.current_function = old_function

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        """Process function/class calls."""
        call_name = self._get_name_from_node(node.func)
        if call_name:
            resolved_name = self._resolve_name(call_name)

            # Determine if it's a class instantiation or function call
            if call_name[0].isupper() or resolved_name.split(".")[-1][0].isupper():
                ref_type = ReferenceType.INSTANTIATE
                target_type = "class"
            else:
                ref_type = ReferenceType.FUNCTION_CALL
                target_type = "function"

            self.references.append(
                {
                    "source_type": self._get_current_source_type(),
                    "source_name": self._get_current_source_name(),
                    "source_line": node.lineno,
                    "target_type": target_type,
                    "target_name": resolved_name,
                    "reference_type": ref_type,
                    "context": ast.unparse(node)[:100],  # Limit context length
                }
            )

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        """Process attribute access."""
        # Only track module-level attribute access for now
        if isinstance(node.value, ast.Name):
            base_name = node.value.id
            if base_name in self.imports:
                full_name = f"{self.imports[base_name]}.{node.attr}"
                self.references.append(
                    {
                        "source_type": self._get_current_source_type(),
                        "source_name": self._get_current_source_name(),
                        "source_line": node.lineno,
                        "target_type": "function",  # Could be function or class
                        "target_name": full_name,
                        "reference_type": ReferenceType.ATTRIBUTE,
                        "context": f"{base_name}.{node.attr}",
                    }
                )

        self.generic_visit(node)

    def _get_name_from_node(self, node: ast.AST) -> str | None:
        """Extract name from various AST nodes."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parts: list[str] = []
            current: ast.expr = node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
                return ".".join(reversed(parts))
        elif isinstance(node, ast.Call):
            return self._get_name_from_node(node.func)
        elif isinstance(node, ast.Subscript):
            # Handle Generic[T] style type hints
            return self._get_name_from_node(node.value)
        return None

    def _resolve_name(self, name: str) -> str:
        """Resolve a name using import information."""
        parts = name.split(".")
        first_part = parts[0]

        if first_part in self.imports:
            resolved = self.imports[first_part]
            if len(parts) > 1:
                resolved = f"{resolved}.{'.'.join(parts[1:])}"
            return resolved

        # If not imported, assume it's in the current module
        if "." not in name:
            if self.current_class and self.current_function:
                # Could be a local reference
                return f"{self.module_path}.{self.current_class}.{name}"
            if self.current_class:
                return f"{self.module_path}.{self.current_class}.{name}"
            return f"{self.module_path}.{name}"

        return name

    def _get_current_source_type(self) -> str:
        """Get the current source entity type."""
        if self.current_function:
            return "function"
        if self.current_class:
            return "class"
        return "module"

    def _get_current_source_name(self) -> str:
        """Get the current source entity name."""
        parts = [self.module_path]
        if self.current_class:
            parts.append(self.current_class)
        if self.current_function:
            parts.append(self.current_function)
        return ".".join(parts)
