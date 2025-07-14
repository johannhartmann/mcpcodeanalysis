"""Generate embeddings for code entities."""

from typing import Any

from langchain_openai import OpenAIEmbeddings

from src.config import settings
from src.logger import get_logger
from src.parser.code_extractor import CodeExtractor

logger = get_logger(__name__)

# Constants
MAX_DISPLAY_METHODS = 10


class EmbeddingGenerator:
    """Generate embeddings for various code entities."""

    def __init__(self) -> None:
        """Initialize embedding generator."""
        # settings imported globally from src.config
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key.get_secret_value(),
            model=settings.embeddings.model,
        )
        self.code_extractor = CodeExtractor()
        self.code_interpreter = None  # Lazy loaded

    def _get_code_interpreter(self):
        """Get code interpreter instance (lazy loaded)."""
        if self.code_interpreter is None:
            from src.embeddings.code_interpreter import CodeInterpreter

            self.code_interpreter = CodeInterpreter()
        return self.code_interpreter

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
                parts.append(
                    f"Public methods: {', '.join(method_names[:MAX_DISPLAY_METHODS])}",
                )
                if len(method_names) > MAX_DISPLAY_METHODS:
                    parts.append(
                        f"... and {len(method_names) - MAX_DISPLAY_METHODS} more",
                    )

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
        embeddings = await self.embeddings.aembed_documents(texts)

        # Combine with metadata
        results = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings, strict=False)):
            results.append(
                {
                    "text": text,
                    "embedding": embedding,
                    "metadata": metadata[i] if metadata else None,
                }
            )

        return results

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
        embeddings = await self.embeddings.aembed_documents(texts)

        # Combine with metadata
        results = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings, strict=False)):
            results.append(
                {
                    "text": text,
                    "embedding": embedding,
                    "metadata": metadata[i] if metadata else None,
                }
            )

        return results

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

        embedding = await self.embeddings.aembed_query(text)

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
            "Generating embedding for %s: %s (%s-%s)",
            entity_type,
            entity_name,
            start_line,
            end_line,
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

        embedding = await self.embeddings.aembed_query(text)

        # Estimate tokens
        tokens = len(text.split()) * 1.3

        return {
            "text": text,
            "embedding": embedding,
            "metadata": metadata,
            "tokens": int(tokens),
        }

    def prepare_domain_entity_text(
        self,
        entity_data: dict[str, Any],
    ) -> str:
        """Prepare domain entity text for embedding.

        Args:
            entity_data: Domain entity data with name, type, description, etc.

        Returns:
            Formatted text for embedding
        """
        parts = []

        # Basic information
        parts.append(f"Domain Entity: {entity_data['name']}")
        parts.append(f"Type: {entity_data['entity_type']}")

        if entity_data.get("description"):
            parts.append(f"Description: {entity_data['description']}")

        # Business context
        if entity_data.get("business_rules"):
            rules = entity_data["business_rules"]
            if isinstance(rules, list):
                parts.append(f"Business Rules: {', '.join(rules)}")
            else:
                parts.append(f"Business Rules: {rules}")

        if entity_data.get("invariants"):
            invariants = entity_data["invariants"]
            if isinstance(invariants, list):
                parts.append(f"Invariants: {', '.join(invariants)}")
            else:
                parts.append(f"Invariants: {invariants}")

        if entity_data.get("responsibilities"):
            resp = entity_data["responsibilities"]
            if isinstance(resp, list):
                parts.append(f"Responsibilities: {', '.join(resp)}")
            else:
                parts.append(f"Responsibilities: {resp}")

        # Implementation details
        if entity_data.get("module_path"):
            parts.append(f"Module: {entity_data['module_path']}")

        if entity_data.get("class_name"):
            parts.append(f"Implementation: {entity_data['class_name']}")

        # Relationships
        if entity_data.get("bounded_context"):
            parts.append(f"Bounded Context: {entity_data['bounded_context']}")

        if entity_data.get("aggregate_root"):
            parts.append(f"Aggregate Root: {entity_data['aggregate_root']}")

        return "\n".join(parts)

    async def generate_domain_entity_embedding(
        self,
        entity_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate embedding for a domain entity.

        Args:
            entity_data: Domain entity data

        Returns:
            Embedding result with text, vector, and metadata
        """
        logger.debug(
            "Generating embedding for domain entity: %s (%s)",
            entity_data.get("name"),
            entity_data.get("entity_type"),
        )

        text = self.prepare_domain_entity_text(entity_data)

        metadata = {
            "type": "domain_entity",
            "entity_type": entity_data.get("entity_type"),
            "name": entity_data.get("name"),
            "bounded_context": entity_data.get("bounded_context"),
            "module_path": entity_data.get("module_path"),
            "class_name": entity_data.get("class_name"),
        }

        embedding = await self.embeddings.aembed_query(text)

        # Estimate tokens
        tokens = len(text.split()) * 1.3

        return {
            "text": text,
            "embedding": embedding,
            "metadata": metadata,
            "tokens": int(tokens),
        }

    async def generate_interpreted_function_embedding(
        self,
        function_data: dict[str, Any],
        code: str,
        file_path: str,
    ) -> dict[str, Any]:
        """Generate embedding for a function using LLM interpretation.

        Args:
            function_data: Function metadata
            code: Function source code
            file_path: Path to the source file

        Returns:
            Embedding result with interpreted text
        """
        logger.info(
            "Generating interpreted embedding for function: %s", function_data["name"]
        )

        # Build context for interpreter
        context = {
            "file_path": file_path,
            "class_name": function_data.get("class_name"),
        }

        # Get LLM interpretation
        code_interpreter = self._get_code_interpreter()
        interpretation = await code_interpreter.interpret_function(
            code=code,
            function_name=function_data["name"],
            docstring=function_data.get("docstring"),
            context=context,
        )

        # Build signature for metadata
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

        # Create search-optimized text
        metadata = {
            "signature": signature,
            "file_path": file_path,
        }

        text = await code_interpreter.create_search_optimized_text(
            entity_type="function",
            entity_name=function_data["name"],
            code=code if len(code) < 1000 else None,  # Include short code only
            interpretation=interpretation,
            metadata=metadata,
        )

        # Generate embedding
        embedding = await self.embeddings.aembed_query(text)

        return {
            "text": text,
            "embedding": embedding,
            "metadata": {
                "entity_type": "function",
                "entity_name": function_data["name"],
                "file_path": file_path,
                "start_line": function_data.get("start_line"),
                "end_line": function_data.get("end_line"),
                "is_method": function_data.get("class_name") is not None,
                "class_name": function_data.get("class_name"),
                "has_interpretation": True,
            },
            "tokens": len(text.split()) * 1.3,
        }

    async def generate_interpreted_class_embedding(
        self,
        class_data: dict[str, Any],
        code: str,
        file_path: str,
    ) -> dict[str, Any]:
        """Generate embedding for a class using LLM interpretation.

        Args:
            class_data: Class metadata
            code: Class source code
            file_path: Path to the source file

        Returns:
            Embedding result with interpreted text
        """
        logger.info(
            "Generating interpreted embedding for class: %s", class_data["name"]
        )

        # Build context
        context = {
            "file_path": file_path,
        }

        # Get methods summary
        methods_summary = class_data.get("methods", [])

        # Get LLM interpretation
        code_interpreter = self._get_code_interpreter()
        interpretation = await code_interpreter.interpret_class(
            code=code,
            class_name=class_data["name"],
            docstring=class_data.get("docstring"),
            methods_summary=methods_summary,
            context=context,
        )

        # Create search-optimized text
        metadata = {
            "file_path": file_path,
            "base_classes": class_data.get("base_classes", []),
        }

        text = await code_interpreter.create_search_optimized_text(
            entity_type="class",
            entity_name=class_data["name"],
            code=code if len(code) < 2000 else None,  # Include moderately sized classes
            interpretation=interpretation,
            metadata=metadata,
        )

        # Generate embedding
        embedding = await self.embeddings.aembed_query(text)

        return {
            "text": text,
            "embedding": embedding,
            "metadata": {
                "entity_type": "class",
                "entity_name": class_data["name"],
                "file_path": file_path,
                "start_line": class_data.get("start_line"),
                "end_line": class_data.get("end_line"),
                "is_abstract": class_data.get("is_abstract", False),
                "has_interpretation": True,
            },
            "tokens": len(text.split()) * 1.3,
        }

    async def generate_interpreted_module_embedding(
        self,
        module_data: dict[str, Any],
        file_path: str,
        imports: list[str] | None = None,
        classes: list[dict[str, Any]] | None = None,
        functions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Generate embedding for a module using LLM interpretation.

        Args:
            module_data: Module metadata
            file_path: Path to the source file
            imports: List of import statements
            classes: Summary of classes in module
            functions: Summary of functions in module

        Returns:
            Embedding result with interpreted text
        """
        logger.info(
            "Generating interpreted embedding for module: %s", module_data["name"]
        )

        # Get LLM interpretation
        code_interpreter = self._get_code_interpreter()
        interpretation = await code_interpreter.interpret_module(
            module_name=module_data["name"],
            file_path=file_path,
            docstring=module_data.get("docstring"),
            imports=imports,
            classes=classes,
            functions=functions,
        )

        # Create search-optimized text
        metadata = {
            "file_path": file_path,
        }

        text = await code_interpreter.create_search_optimized_text(
            entity_type="module",
            entity_name=module_data["name"],
            code=None,  # Module code usually too large
            interpretation=interpretation,
            metadata=metadata,
        )

        # Generate embedding
        embedding = await self.embeddings.aembed_query(text)

        return {
            "text": text,
            "embedding": embedding,
            "metadata": {
                "entity_type": "module",
                "entity_name": module_data["name"],
                "file_path": file_path,
                "start_line": module_data.get("start_line", 1),
                "end_line": module_data.get("end_line"),
                "has_interpretation": True,
            },
            "tokens": len(text.split()) * 1.3,
        }
