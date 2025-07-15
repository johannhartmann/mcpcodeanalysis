"""Base parser interface and data structures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from tree_sitter import Language, Node, Parser, Tree

from src.logger import get_logger

logger = get_logger(__name__)


class ElementType(Enum):
    """Types of code elements that can be parsed."""

    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    IMPORT = "import"
    DECORATOR = "decorator"
    COMMENT = "comment"
    DOCSTRING = "docstring"


@dataclass
class ParsedElement:
    """Represents a parsed code element."""

    type: ElementType
    name: str
    start_line: int
    end_line: int
    start_column: int
    end_column: int
    text: str
    parent: Optional["ParsedElement"] = None
    children: list["ParsedElement"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def line_count(self) -> int:
        """Get number of lines in this element."""
        return self.end_line - self.start_line + 1

    @property
    def full_name(self) -> str:
        """Get fully qualified name including parent context."""
        if self.parent and self.parent.type != ElementType.MODULE:
            return f"{self.parent.full_name}.{self.name}"
        return self.name

    def add_child(self, child: "ParsedElement") -> None:
        """Add a child element."""
        child.parent = self
        self.children.append(child)

    def find_children(self, element_type: ElementType) -> list["ParsedElement"]:
        """Find all children of a specific type."""
        return [c for c in self.children if c.type == element_type]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "type": self.type.value,
            "name": self.name,
            "full_name": self.full_name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "start_column": self.start_column,
            "end_column": self.end_column,
            "line_count": self.line_count,
            "text": self.text,
            "metadata": self.metadata,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class ParseResult:
    """Result of parsing a file."""

    file_path: Path
    language: str
    parsed_at: datetime
    root_element: ParsedElement
    imports: list[str] = field(default_factory=list)
    dependencies: set[str] = field(default_factory=set)
    references: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if parsing was successful."""
        return len(self.errors) == 0

    @property
    def all_elements(self) -> list[ParsedElement]:
        """Get all parsed elements recursively."""
        elements = []

        def collect_elements(element: ParsedElement) -> None:
            elements.append(element)
            for child in element.children:
                collect_elements(child)

        collect_elements(self.root_element)
        return elements

    def find_elements(self, element_type: ElementType) -> list[ParsedElement]:
        """Find all elements of a specific type."""
        return [e for e in self.all_elements if e.type == element_type]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "file_path": str(self.file_path),
            "language": self.language,
            "parsed_at": self.parsed_at.isoformat(),
            "success": self.success,
            "root_element": self.root_element.to_dict(),
            "imports": self.imports,
            "dependencies": list(self.dependencies),
            "references": self.references,
            "errors": self.errors,
            "statistics": {
                "total_elements": len(self.all_elements),
                "classes": len(self.find_elements(ElementType.CLASS)),
                "functions": len(self.find_elements(ElementType.FUNCTION)),
                "methods": len(self.find_elements(ElementType.METHOD)),
                "references": len(self.references),
            },
        }


class BaseParser(ABC):
    """Abstract base class for language-specific parsers."""

    def __init__(self, language: Language) -> None:
        self.language = language
        self.parser = Parser(language)

    @abstractmethod
    def get_language_name(self) -> str:
        """Get the name of the programming language."""

    @abstractmethod
    def get_file_extensions(self) -> set[str]:
        """Get supported file extensions."""

    def parse_file(self, file_path: Path) -> ParseResult:
        """Parse a file and extract code elements."""
        logger.info("Parsing file: %s", file_path)

        try:
            # Read file content
            content = file_path.read_text(encoding="utf-8")

            # Parse with TreeSitter
            tree = self.parser.parse(bytes(content, "utf8"))

            # Extract elements
            root_element = self._extract_elements(tree, content, file_path)

            # Extract imports and dependencies
            imports = self._extract_imports(tree, content)
            dependencies = self._extract_dependencies(imports)

            # Extract references
            references = self._extract_references(tree, content)

            result = ParseResult(
                file_path=file_path,
                language=self.get_language_name(),
                parsed_at=datetime.now(tz=UTC),
                root_element=root_element,
                imports=imports,
                dependencies=dependencies,
                references=references,
            )

            logger.info(
                "Successfully parsed %s: %s elements found",
                file_path,
                len(result.all_elements),
            )

            return result

        except (AttributeError, KeyError, ValueError, TypeError) as e:
            logger.exception("Error parsing file %s", file_path)

            # Return error result
            return ParseResult(
                file_path=file_path,
                language=self.get_language_name(),
                parsed_at=datetime.now(tz=UTC),
                root_element=ParsedElement(
                    type=ElementType.MODULE,
                    name=file_path.stem,
                    start_line=1,
                    end_line=1,
                    start_column=0,
                    end_column=0,
                    text="",
                ),
                errors=[
                    {
                        "type": "parse_error",
                        "message": str(e),
                    },
                ],
            )

    def parse_string(self, content: str, filename: str = "unknown") -> ParseResult:
        """Parse a string of code."""
        # Create temporary result
        tree = self.parser.parse(bytes(content, "utf8"))

        root_element = self._extract_elements(tree, content, Path(filename))
        imports = self._extract_imports(tree, content)
        dependencies = self._extract_dependencies(imports)
        references = self._extract_references(tree, content)

        return ParseResult(
            file_path=Path(filename),
            language=self.get_language_name(),
            parsed_at=datetime.now(tz=UTC),
            root_element=root_element,
            imports=imports,
            dependencies=dependencies,
            references=references,
        )

    @abstractmethod
    def _extract_elements(
        self,
        tree: Tree,
        content: str,
        file_path: Path,
    ) -> ParsedElement:
        """Extract code elements from the parse tree."""

    @abstractmethod
    def _extract_imports(self, tree: Tree, content: str) -> list[str]:
        """Extract import statements."""

    @abstractmethod
    def _extract_references(self, tree: Tree, content: str) -> list[dict[str, Any]]:
        """Extract code references between entities."""

    def _extract_dependencies(self, imports: list[str]) -> set[str]:
        """Extract external dependencies from imports."""
        dependencies = set()

        for import_stmt in imports:
            # Extract the base module name
            parts = import_stmt.split()
            if parts and parts[0] in ["import", "from"]:
                if parts[0] == "import":
                    if len(parts) > 1:
                        module = parts[1].split(".")[0]
                        dependencies.add(module)
                elif parts[0] == "from" and len(parts) > 1:
                    module = parts[1].split(".")[0]
                    dependencies.add(module)

        # Filter out relative imports and standard library modules
        return {
            dep
            for dep in dependencies
            if not dep.startswith(".") and dep not in self._get_stdlib_modules()
        }

    def _get_stdlib_modules(self) -> set[str]:
        """Get set of standard library module names."""
        # This is a simplified list - in production, use stdlib_list package
        return {
            "os",
            "sys",
            "json",
            "re",
            "math",
            "datetime",
            "collections",
            "itertools",
            "functools",
            "pathlib",
            "typing",
            "asyncio",
            "logging",
            "unittest",
            "urllib",
            "http",
            "email",
            "csv",
            "xml",
            "html",
            "io",
            "time",
            "random",
            "string",
            "textwrap",
            "copy",
            "pickle",
            "shelve",
            "sqlite3",
            "zlib",
            "gzip",
            "tarfile",
            "zipfile",
            "configparser",
            "argparse",
            "subprocess",
            "multiprocessing",
            "threading",
            "queue",
            "socket",
            "ssl",
            "select",
            "selectors",
            "signal",
            "mmap",
            "ctypes",
            "struct",
            "codecs",
            "locale",
            "gettext",
            "base64",
            "binascii",
            "hashlib",
            "hmac",
            "secrets",
            "uuid",
            "platform",
            "errno",
            "warnings",
            "traceback",
            "inspect",
            "ast",
            "dis",
            "types",
            "dataclasses",
            "enum",
            "abc",
            "contextlib",
            "weakref",
            "gc",
            "importlib",
        }

    def _get_node_text(self, node: Node, content: str) -> str:
        """Get text content of a node."""
        return content[node.start_byte : node.end_byte]

    def _create_element(
        self,
        node: Node,
        content: str,
        element_type: ElementType,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> ParsedElement:
        """Create a ParsedElement from a TreeSitter node."""
        return ParsedElement(
            type=element_type,
            name=name,
            start_line=node.start_point[0] + 1,  # TreeSitter uses 0-based line numbers
            end_line=node.end_point[0] + 1,
            start_column=node.start_point[1],
            end_column=node.end_point[1],
            text=self._get_node_text(node, content),
            metadata=metadata or {},
        )
