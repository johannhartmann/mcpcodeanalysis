"""Base parser interface and data structures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import tree_sitter
from tree_sitter import Language, Node, Parser, Tree

from src.utils.logger import get_logger

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
    children: List["ParsedElement"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
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
    
    def find_children(self, element_type: ElementType) -> List["ParsedElement"]:
        """Find all children of a specific type."""
        return [c for c in self.children if c.type == element_type]
    
    def to_dict(self) -> Dict[str, Any]:
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
    imports: List[str] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def success(self) -> bool:
        """Check if parsing was successful."""
        return len(self.errors) == 0
    
    @property
    def all_elements(self) -> List[ParsedElement]:
        """Get all parsed elements recursively."""
        elements = []
        
        def collect_elements(element: ParsedElement):
            elements.append(element)
            for child in element.children:
                collect_elements(child)
        
        collect_elements(self.root_element)
        return elements
    
    def find_elements(self, element_type: ElementType) -> List[ParsedElement]:
        """Find all elements of a specific type."""
        return [e for e in self.all_elements if e.type == element_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "file_path": str(self.file_path),
            "language": self.language,
            "parsed_at": self.parsed_at.isoformat(),
            "success": self.success,
            "root_element": self.root_element.to_dict(),
            "imports": self.imports,
            "dependencies": list(self.dependencies),
            "errors": self.errors,
            "statistics": {
                "total_elements": len(self.all_elements),
                "classes": len(self.find_elements(ElementType.CLASS)),
                "functions": len(self.find_elements(ElementType.FUNCTION)),
                "methods": len(self.find_elements(ElementType.METHOD)),
            },
        }


class BaseParser(ABC):
    """Abstract base class for language-specific parsers."""
    
    def __init__(self, language: Language):
        self.language = language
        self.parser = Parser()
        self.parser.set_language(language)
    
    @abstractmethod
    def get_language_name(self) -> str:
        """Get the name of the programming language."""
        pass
    
    @abstractmethod
    def get_file_extensions(self) -> Set[str]:
        """Get supported file extensions."""
        pass
    
    def parse_file(self, file_path: Path) -> ParseResult:
        """Parse a file and extract code elements."""
        logger.info(f"Parsing file: {file_path}")
        
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
            
            result = ParseResult(
                file_path=file_path,
                language=self.get_language_name(),
                parsed_at=datetime.now(),
                root_element=root_element,
                imports=imports,
                dependencies=dependencies,
            )
            
            logger.info(
                f"Successfully parsed {file_path}: "
                f"{len(result.all_elements)} elements found"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            
            # Return error result
            return ParseResult(
                file_path=file_path,
                language=self.get_language_name(),
                parsed_at=datetime.now(),
                root_element=ParsedElement(
                    type=ElementType.MODULE,
                    name=file_path.stem,
                    start_line=1,
                    end_line=1,
                    start_column=0,
                    end_column=0,
                    text="",
                ),
                errors=[{
                    "type": "parse_error",
                    "message": str(e),
                }],
            )
    
    def parse_string(self, content: str, filename: str = "unknown") -> ParseResult:
        """Parse a string of code."""
        # Create temporary result
        tree = self.parser.parse(bytes(content, "utf8"))
        
        root_element = self._extract_elements(tree, content, Path(filename))
        imports = self._extract_imports(tree, content)
        dependencies = self._extract_dependencies(imports)
        
        return ParseResult(
            file_path=Path(filename),
            language=self.get_language_name(),
            parsed_at=datetime.now(),
            root_element=root_element,
            imports=imports,
            dependencies=dependencies,
        )
    
    @abstractmethod
    def _extract_elements(self, tree: Tree, content: str, file_path: Path) -> ParsedElement:
        """Extract code elements from the parse tree."""
        pass
    
    @abstractmethod
    def _extract_imports(self, tree: Tree, content: str) -> List[str]:
        """Extract import statements."""
        pass
    
    def _extract_dependencies(self, imports: List[str]) -> Set[str]:
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
        dependencies = {
            dep for dep in dependencies
            if not dep.startswith(".") and dep not in self._get_stdlib_modules()
        }
        
        return dependencies
    
    def _get_stdlib_modules(self) -> Set[str]:
        """Get set of standard library module names."""
        # This is a simplified list - in production, use stdlib_list package
        return {
            "os", "sys", "json", "re", "math", "datetime", "collections",
            "itertools", "functools", "pathlib", "typing", "asyncio",
            "logging", "unittest", "urllib", "http", "email", "csv",
            "xml", "html", "io", "time", "random", "string", "textwrap",
            "copy", "pickle", "shelve", "sqlite3", "zlib", "gzip",
            "tarfile", "zipfile", "configparser", "argparse", "subprocess",
            "multiprocessing", "threading", "queue", "socket", "ssl",
            "select", "selectors", "signal", "mmap", "ctypes", "struct",
            "codecs", "locale", "gettext", "base64", "binascii", "hashlib",
            "hmac", "secrets", "uuid", "platform", "errno", "warnings",
            "traceback", "inspect", "ast", "dis", "types", "dataclasses",
            "enum", "abc", "contextlib", "weakref", "gc", "importlib",
        }
    
    def _get_node_text(self, node: Node, content: str) -> str:
        """Get text content of a node."""
        return content[node.start_byte:node.end_byte]
    
    def _create_element(
        self,
        node: Node,
        content: str,
        element_type: ElementType,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
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