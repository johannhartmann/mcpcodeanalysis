"""Generate embeddings for code entities."""

from typing import Any

from src.embeddings.openai_client import OpenAIClient
from src.parser.code_extractor import CodeExtractor
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Constants
MAX_DISPLAY_METHODS = 10


class EmbeddingGenerator:
    """Generate embeddings for various code entities."""

    def __init__(self, openai_client: OpenAIClient | None = None) -> None:
        """Initialize embedding generator.

        Args:
            openai_client: OpenAI client instance. Creates new if not provided.
        """
        self.openai_client = openai_client or OpenAIClient()
        self.code_extractor = CodeExtractor()

    def prepare_function_text(
        self,
        function_data: dict[str, Any],
        file_path: str,
    ) -> str:
        """Prepare function text for embedding.

        Args:
            function_data: Function entity data
            file_path: Path to the source file

        Returns:
            Formatted text for embedding
        """
        parts = []

        # Add context
        parts.append(f"Function: {function_data['name']}")
        parts.append(f"File: {file_path}")

        # Add signature
        params = function_data.get("parameters", [])
        param_strs = []
        for param in params:
            param_str = param["name"]
            if param.get("type"):
                param_str += f": {param['type']}"
            if param.get("default"):
                param_str += f" = {param['default']}"
            param_strs.append(param_str)

        signature = f"{function_data['name']}({', '.join(param_strs)})"
        if function_data.get("return_type"):
            signature += f" -> {function_data['return_type']}"
        parts.append(f"Signature: {signature}")

        # Add properties
        properties = []
        if function_data.get("is_async"):
            properties.append("async")
        if function_data.get("is_generator"):
            properties.append("generator")
        if function_data.get("is_property"):
            properties.append("property")
        if function_data.get("is_staticmethod"):
            properties.append("static method")
        if function_data.get("is_classmethod"):
            properties.append("class method")

        if properties:
            parts.append(f"Type: {', '.join(properties)}")

        # Add class context if method
        if function_data.get("class_name"):
            parts.append(f"Class: {function_data['class_name']}")

        # Add docstring
        if function_data.get("docstring"):
            parts.append(f"Description: {function_data['docstring']}")

        # Add decorators
        decorators = function_data.get("decorators", [])
        if decorators:
            parts.append(f"Decorators: {', '.join(decorators)}")

        return "\n".join(parts)

    def prepare_class_text(self, class_data: dict[str, Any], file_path: str) -> str:
        """Prepare class text for embedding.

        Args:
            class_data: Class entity data
            file_path: Path to the source file

        Returns:
            Formatted text for embedding
        """
        parts = []

        # Add context
        parts.append(f"Class: {class_data['name']}")
        parts.append(f"File: {file_path}")

        # Add inheritance
        if class_data.get("base_classes"):
            parts.append(f"Inherits from: {', '.join(class_data['base_classes'])}")

        # Add properties
        if class_data.get("is_abstract"):
            parts.append("Type: abstract class")

        # Add docstring
        if class_data.get("docstring"):
            parts.append(f"Description: {class_data['docstring']}")

        # Add decorators
        decorators = class_data.get("decorators", [])
        if decorators:
            parts.append(f"Decorators: {', '.join(decorators)}")

        # Add method summary
        methods = class_data.get("methods", [])
        if methods:
            method_names = [m["name"] for m in methods if not m["name"].startswith("_")]
            if method_names:
                parts.append(f"Public methods: {', '.join(method_names[:MAX_DISPLAY_METHODS])}")
                if len(method_names) > MAX_DISPLAY_METHODS:
                    parts.append(f"... and {len(method_names) - MAX_DISPLAY_METHODS} more")

        return "\n".join(parts)

    def prepare_module_text(
        self,
        module_data: dict[str, Any],
        file_path: str,
        summary: dict[str, int] | None = None,
    ) -> str:
        """Prepare module text for embedding.

        Args:
            module_data: Module entity data
            file_path: Path to the source file
            summary: Optional summary statistics

        Returns:
            Formatted text for embedding
        """
        parts = []

        # Add context
        parts.append(f"Module: {module_data['name']}")
        parts.append(f"File: {file_path}")

        # Add docstring
        if module_data.get("docstring"):
            parts.append(f"Description: {module_data['docstring']}")

        # Add summary if provided
        if summary:
            summary_parts = []
            if summary.get("classes", 0) > 0:
                summary_parts.append(f"{summary['classes']} classes")
            if summary.get("functions", 0) > 0:
                summary_parts.append(f"{summary['functions']} functions")
            if summary.get("imports", 0) > 0:
                summary_parts.append(f"{summary['imports']} imports")

            if summary_parts:
                parts.append(f"Contains: {', '.join(summary_parts)}")

        return "\n".join(parts)

    def prepare_code_chunk_text(
        self,
        code: str,
        entity_type: str,
        entity_name: str,
        file_path: str,
        start_line: int,
        end_line: int,
    ) -> str:
        """Prepare code chunk text for embedding.

        Args:
            code: Raw code content
            entity_type: Type of entity (function, class, etc.)
            entity_name: Name of the entity
            file_path: Path to the source file
            start_line: Starting line number
            end_line: Ending line number

        Returns:
            Formatted text for embedding
        """
        parts = []

        # Add context
        parts.append(f"{entity_type.capitalize()}: {entity_name}")
        parts.append(f"File: {file_path}")
        parts.append(f"Lines: {start_line}-{end_line}")
        parts.append("---")
        parts.append(code)

        return "\n".join(parts)

    async def generate_function_embeddings(
        self,
        functions: list[dict[str, Any]],
        file_path: str,
    ) -> list[dict[str, Any]]:
        """Generate embeddings for functions.

        Args:
            functions: List of function entities
            file_path: Path to the source file

        Returns:
            List of embedding results
        """
        if not functions:
            return []

        logger.info("Generating embeddings for %d functions", len(functions))

        # Prepare texts
        texts = []
        metadata = []

        for func in functions:
            text = self.prepare_function_text(func, file_path)
            texts.append(text)

            metadata.append(
                {
                    "entity_type": "function",
                    "entity_name": func["name"],
                    "file_path": file_path,
                    "start_line": func.get("start_line"),
                    "end_line": func.get("end_line"),
                    "is_method": func.get("class_name") is not None,
                    "class_name": func.get("class_name"),
                },
            )

        # Generate embeddings
        return await self.openai_client.generate_embeddings_batch(texts, metadata)

    async def generate_class_embeddings(
        self,
        classes: list[dict[str, Any]],
        file_path: str,
    ) -> list[dict[str, Any]]:
        """Generate embeddings for classes.

        Args:
            classes: List of class entities
            file_path: Path to the source file

        Returns:
            List of embedding results
        """
        if not classes:
            return []

        logger.info("Generating embeddings for %d classes", len(classes))

        # Prepare texts
        texts = []
        metadata = []

        for cls in classes:
            text = self.prepare_class_text(cls, file_path)
            texts.append(text)

            metadata.append(
                {
                    "entity_type": "class",
                    "entity_name": cls["name"],
                    "file_path": file_path,
                    "start_line": cls.get("start_line"),
                    "end_line": cls.get("end_line"),
                    "is_abstract": cls.get("is_abstract", False),
                },
            )

        # Generate embeddings
        return await self.openai_client.generate_embeddings_batch(texts, metadata)

    async def generate_module_embedding(
        self,
        module_data: dict[str, Any],
        file_path: str,
        summary: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Generate embedding for a module.

        Args:
            module_data: Module entity data
            file_path: Path to the source file
            summary: Optional summary statistics

        Returns:
            Embedding result
        """
        logger.info("Generating embedding for module: %s", module_data["name"])

        text = self.prepare_module_text(module_data, file_path, summary)

        metadata = {
            "entity_type": "module",
            "entity_name": module_data["name"],
            "file_path": file_path,
            "start_line": module_data.get("start_line", 1),
            "end_line": module_data.get("end_line"),
        }

        embedding = await self.openai_client.generate_embedding(text)

        # Estimate tokens
        tokens = len(text.split()) * 1.3

        return {
            "text": text,
            "embedding": embedding,
            "metadata": metadata,
            "tokens": int(tokens),
        }

    async def generate_code_chunk_embedding(
        self,
        code: str,
        entity_type: str,
        entity_name: str,
        file_path: str,
        start_line: int,
        end_line: int,
        additional_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate embedding for a code chunk.

        Args:
            code: Raw code content
            entity_type: Type of entity
            entity_name: Name of the entity
            file_path: Path to the source file
            start_line: Starting line number
            end_line: Ending line number
            additional_metadata: Additional metadata to include

        Returns:
            Embedding result
        """
        logger.debug(
            f"Generating embedding for {entity_type}: {entity_name} "
            f"({start_line}-{end_line})",
        )

        text = self.prepare_code_chunk_text(
            code,
            entity_type,
            entity_name,
            file_path,
            start_line,
            end_line,
        )

        metadata = {
            "entity_type": entity_type,
            "entity_name": entity_name,
            "file_path": file_path,
            "start_line": start_line,
            "end_line": end_line,
            "code_lines": end_line - start_line + 1,
        }

        if additional_metadata:
            metadata.update(additional_metadata)

        embedding = await self.openai_client.generate_embedding(text)

        # Estimate tokens
        tokens = len(text.split()) * 1.3

        return {
            "text": text,
            "embedding": embedding,
            "metadata": metadata,
            "tokens": int(tokens),
        }
