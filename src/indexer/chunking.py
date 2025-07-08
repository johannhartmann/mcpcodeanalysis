"""Code chunking strategies for embedding generation."""

from typing import Any, Dict, List, Tuple

from src.mcp_server.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CodeChunker:
    """Chunk code into logical units for embedding."""
    
    def __init__(self) -> None:
        self.chunk_size = config.parser.chunk_size
        self.max_tokens = config.embeddings.max_tokens
    
    def chunk_by_entity(
        self,
        entities: Dict[str, List[Dict[str, Any]]],
        file_content: str
    ) -> List[Dict[str, Any]]:
        """Chunk code by logical entities (functions, classes, etc.)."""
        chunks = []
        lines = file_content.split("\n")
        
        # Process functions
        for func in entities.get("functions", []):
            chunk = self._create_entity_chunk(
                "function",
                func,
                lines,
                include_context=True
            )
            chunks.append(chunk)
        
        # Process classes
        for cls in entities.get("classes", []):
            # Create chunk for entire class
            chunk = self._create_entity_chunk(
                "class",
                cls,
                lines,
                include_context=True
            )
            chunks.append(chunk)
            
            # Also create chunks for individual methods if class is large
            if cls.get("end_line", 0) - cls.get("start_line", 0) > self.chunk_size:
                for method in cls.get("methods", []):
                    method_chunk = self._create_entity_chunk(
                        "method",
                        method,
                        lines,
                        include_context=True,
                        parent_class=cls["name"]
                    )
                    chunks.append(method_chunk)
        
        # Process module-level code
        module_chunk = self._create_module_chunk(entities, lines)
        if module_chunk:
            chunks.append(module_chunk)
        
        return chunks
    
    def chunk_by_lines(
        self,
        file_content: str,
        overlap: int = 20
    ) -> List[Dict[str, Any]]:
        """Chunk code by line count with overlap."""
        lines = file_content.split("\n")
        chunks = []
        
        for i in range(0, len(lines), self.chunk_size - overlap):
            start_line = i + 1
            end_line = min(i + self.chunk_size, len(lines))
            
            chunk_lines = lines[i:end_line]
            chunk_content = "\n".join(chunk_lines)
            
            chunks.append({
                "type": "lines",
                "content": chunk_content,
                "start_line": start_line,
                "end_line": end_line,
                "metadata": {
                    "line_count": len(chunk_lines),
                    "has_overlap": i > 0
                }
            })
        
        return chunks
    
    def _create_entity_chunk(
        self,
        entity_type: str,
        entity: Dict[str, Any],
        lines: List[str],
        include_context: bool = False,
        parent_class: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a chunk for a code entity."""
        start_line = entity.get("start_line", 1) - 1
        end_line = entity.get("end_line", len(lines))
        
        # Include context lines if requested
        if include_context:
            context_before = 3
            context_after = 1
            start_line = max(0, start_line - context_before)
            end_line = min(len(lines), end_line + context_after)
        
        chunk_lines = lines[start_line:end_line]
        chunk_content = "\n".join(chunk_lines)
        
        # Build metadata
        metadata = {
            "entity_name": entity.get("name", "unknown"),
            "entity_type": entity_type,
            "has_docstring": bool(entity.get("docstring")),
            "line_count": len(chunk_lines),
        }
        
        if parent_class:
            metadata["parent_class"] = parent_class
        
        if entity_type == "function" or entity_type == "method":
            metadata["parameters"] = entity.get("parameters", [])
            metadata["return_type"] = entity.get("return_type")
            metadata["is_async"] = entity.get("is_async", False)
            metadata["is_generator"] = entity.get("is_generator", False)
        elif entity_type == "class":
            metadata["base_classes"] = entity.get("base_classes", [])
            metadata["method_count"] = len(entity.get("methods", []))
            metadata["is_abstract"] = entity.get("is_abstract", False)
        
        return {
            "type": entity_type,
            "content": chunk_content,
            "start_line": start_line + 1,
            "end_line": end_line,
            "metadata": metadata
        }
    
    def _create_module_chunk(
        self,
        entities: Dict[str, List[Dict[str, Any]]],
        lines: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Create a chunk for module-level code."""
        # Find module docstring and imports
        module_end_line = 0
        
        # Look for module docstring
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith(('"""', "'''", "#", "import", "from")):
                module_end_line = i
                break
            elif i > 50:  # Don't look too far
                module_end_line = 50
                break
        
        if module_end_line == 0:
            return None
        
        chunk_content = "\n".join(lines[:module_end_line])
        
        return {
            "type": "module",
            "content": chunk_content,
            "start_line": 1,
            "end_line": module_end_line,
            "metadata": {
                "import_count": len(entities.get("imports", [])),
                "class_count": len(entities.get("classes", [])),
                "function_count": len(entities.get("functions", [])),
            }
        }
    
    def merge_small_chunks(
        self,
        chunks: List[Dict[str, Any]],
        min_size: int = 10
    ) -> List[Dict[str, Any]]:
        """Merge small chunks to improve efficiency."""
        merged = []
        buffer = None
        
        for chunk in chunks:
            chunk_size = chunk["end_line"] - chunk["start_line"] + 1
            
            if chunk_size < min_size and chunk["type"] in ("function", "method"):
                if buffer is None:
                    buffer = chunk
                else:
                    # Merge with buffer
                    buffer = self._merge_chunks(buffer, chunk)
            else:
                if buffer:
                    merged.append(buffer)
                    buffer = None
                merged.append(chunk)
        
        if buffer:
            merged.append(buffer)
        
        return merged
    
    def _merge_chunks(
        self,
        chunk1: Dict[str, Any],
        chunk2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge two chunks."""
        return {
            "type": "merged",
            "content": chunk1["content"] + "\n\n" + chunk2["content"],
            "start_line": chunk1["start_line"],
            "end_line": chunk2["end_line"],
            "metadata": {
                "merged_types": [chunk1["type"], chunk2["type"]],
                "merged_entities": [
                    chunk1["metadata"].get("entity_name"),
                    chunk2["metadata"].get("entity_name")
                ]
            }
        }