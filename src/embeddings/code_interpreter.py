"""LLM-based code interpreter for generating meaningful code explanations."""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)


class CodeInterpreter:
    """Interpret code using LLM to understand what it actually does."""

    def __init__(self) -> None:
        """Initialize code interpreter."""
        self.llm = ChatOpenAI(
            openai_api_key=settings.openai_api_key.get_secret_value(),
            model=settings.llm.model,
            temperature=0.1,  # Low temperature for consistent interpretations
        )

    async def interpret_function(
        self,
        code: str,
        function_name: str,
        docstring: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Interpret what a function does based on its implementation.

        Args:
            code: The function's source code
            function_name: Name of the function
            docstring: Existing docstring (if any)
            context: Additional context (file path, class name, etc.)

        Returns:
            Natural language description of what the function does
        """
        context_info = context or {}

        prompt = f"""Analyze this Python function and explain what it does based on its implementation.
Focus on the actual behavior, not just the signature.

Function: {function_name}
{f"File: {context_info.get('file_path')}" if context_info.get('file_path') else ""}
{f"Class: {context_info.get('class_name')}" if context_info.get('class_name') else ""}
{f"Existing docstring: {docstring}" if docstring else "No docstring provided"}

Code:
```python
{code}
```

Provide a concise but comprehensive explanation that includes:
1. What the function does (its purpose and behavior)
2. Key implementation details (algorithms, patterns, important logic)
3. Side effects or external interactions (database, files, network, etc.)
4. Error handling approach
5. Any notable design patterns or techniques used

Keep the explanation focused on what would help someone searching for this functionality."""

        try:
            messages = [
                SystemMessage(
                    content="You are a code analysis expert. Analyze code and explain what it does based on the implementation, not just documentation."
                ),
                HumanMessage(content=prompt),
            ]

            response = await self.llm.ainvoke(messages)
            return response.content.strip()

        except (ValueError, RuntimeError, AttributeError):
            logger.exception("Failed to interpret function %s", function_name)
            # Fallback to basic description
            return f"Function {function_name} - interpretation failed. {docstring or 'No description available.'}"

    async def interpret_class(
        self,
        code: str,
        class_name: str,
        docstring: str | None = None,
        methods_summary: list[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Interpret what a class does based on its implementation.

        Args:
            code: The class source code
            class_name: Name of the class
            docstring: Existing docstring (if any)
            methods_summary: Summary of class methods
            context: Additional context

        Returns:
            Natural language description of what the class does
        """
        context_info = context or {}
        methods_info = ""

        if methods_summary:
            method_list = [
                f"- {m['name']}: {m.get('docstring', 'No description')}"
                for m in methods_summary[:10]
            ]
            methods_info = "Key methods:\n" + "\n".join(method_list)

        prompt = f"""Analyze this Python class and explain what it does based on its implementation.
Focus on the class's purpose, responsibilities, and how it's used.

Class: {class_name}
{f"File: {context_info.get('file_path')}" if context_info.get('file_path') else ""}
{f"Existing docstring: {docstring}" if docstring else "No docstring provided"}

{methods_info}

Code:
```python
{code}
```

Provide a concise but comprehensive explanation that includes:
1. The class's primary purpose and responsibilities
2. Key design patterns implemented (e.g., singleton, factory, observer)
3. Main functionality provided by its methods
4. State management and data it maintains
5. External dependencies and interactions
6. How this class fits into the larger system architecture

Keep the explanation focused on what would help someone searching for this functionality."""

        try:
            messages = [
                SystemMessage(
                    content="You are a code analysis expert. Analyze code and explain what it does based on the implementation."
                ),
                HumanMessage(content=prompt),
            ]

            response = await self.llm.ainvoke(messages)
            return response.content.strip()

        except (ValueError, RuntimeError, AttributeError):
            logger.exception("Failed to interpret class %s", class_name)
            return f"Class {class_name} - interpretation failed. {docstring or 'No description available.'}"

    async def interpret_module(
        self,
        module_name: str,
        file_path: str,
        docstring: str | None = None,
        imports: list[str] | None = None,
        classes: list[dict[str, Any]] | None = None,
        functions: list[dict[str, Any]] | None = None,
    ) -> str:
        """Interpret what a module does based on its contents.

        Args:
            module_name: Name of the module
            file_path: Path to the module file
            docstring: Module docstring (if any)
            imports: List of import statements
            classes: Summary of classes in the module
            functions: Summary of functions in the module

        Returns:
            Natural language description of what the module does
        """
        # Build a summary of the module contents
        import_summary = (
            f"Imports: {', '.join(imports[:10])}" if imports else "No imports"
        )
        class_summary = (
            f"Classes: {', '.join([c['name'] for c in classes[:10]])}"
            if classes
            else "No classes"
        )
        function_summary = (
            f"Functions: {', '.join([f['name'] for f in functions[:10]])}"
            if functions
            else "No functions"
        )

        prompt = f"""Analyze this Python module and explain its purpose and functionality.

Module: {module_name}
File: {file_path}
{f"Existing docstring: {docstring}" if docstring else "No docstring provided"}

Module contents:
{import_summary}
{class_summary}
{function_summary}

Based on the module's contents, provide a concise explanation that includes:
1. The module's primary purpose and role in the system
2. Key functionality it provides
3. Main abstractions or concepts it implements
4. External dependencies and integrations
5. How this module relates to other parts of the system

Keep the explanation focused on what would help someone searching for this functionality."""

        try:
            messages = [
                SystemMessage(
                    content="You are a code analysis expert. Analyze modules and explain their purpose based on their contents."
                ),
                HumanMessage(content=prompt),
            ]

            response = await self.llm.ainvoke(messages)
            return response.content.strip()

        except (ValueError, RuntimeError, AttributeError):
            logger.exception("Failed to interpret module %s", module_name)
            return f"Module {module_name} - interpretation failed. {docstring or 'No description available.'}"

    async def create_search_optimized_text(
        self,
        entity_type: str,
        entity_name: str,
        code: str | None,
        interpretation: str,
        metadata: dict[str, Any],
    ) -> str:
        """Create search-optimized text combining code, interpretation, and metadata.

        Args:
            entity_type: Type of entity (function, class, module)
            entity_name: Name of the entity
            code: Source code (if available)
            interpretation: LLM interpretation of what the code does
            metadata: Additional metadata (signature, file path, etc.)

        Returns:
            Search-optimized text for embedding
        """
        parts = []

        # Basic identification
        parts.append(f"{entity_type.capitalize()}: {entity_name}")

        if metadata.get("file_path"):
            parts.append(f"Location: {metadata['file_path']}")

        # Add signature for functions
        if entity_type == "function" and metadata.get("signature"):
            parts.append(f"Signature: {metadata['signature']}")

        # Add the interpretation
        parts.append("\nWhat it does:")
        parts.append(interpretation)

        # Add key implementation details if available
        if code and len(code) < 1000:  # Include short code snippets
            parts.append("\nImplementation snippet:")
            parts.append(code[:500] + "..." if len(code) > 500 else code)

        # Add searchable keywords based on common patterns
        keywords = self._extract_keywords(interpretation, code)
        if keywords:
            parts.append(f"\nKey concepts: {', '.join(keywords)}")

        return "\n".join(parts)

    def _extract_keywords(self, interpretation: str, code: str | None) -> list[str]:
        """Extract searchable keywords from interpretation and code."""
        keywords = []

        # Common patterns to look for
        patterns = {
            "validation": ["validate", "valid", "check", "verify"],
            "authentication": ["auth", "login", "token", "credential"],
            "database": ["query", "database", "sql", "orm", "model"],
            "api": ["api", "endpoint", "route", "rest", "graphql"],
            "cache": ["cache", "redis", "memcache", "memoize"],
            "async": ["async", "await", "concurrent", "parallel"],
            "error handling": ["try", "except", "error", "exception"],
            "logging": ["log", "logger", "audit", "track"],
            "configuration": ["config", "setting", "env", "option"],
            "testing": ["test", "mock", "assert", "fixture"],
        }

        text_to_check = interpretation.lower()
        if code:
            text_to_check += " " + code.lower()

        for concept, terms in patterns.items():
            if any(term in text_to_check for term in terms):
                keywords.append(concept)

        return keywords
