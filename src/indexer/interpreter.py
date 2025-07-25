"""Code interpreter for generating natural language descriptions."""

from typing import Any

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)


class CodeInterpreter:
    """Generate natural language interpretations of code."""

    def __init__(self) -> None:
        # Use gpt-4.1 as default model for interpretations
        import os

        model = getattr(settings, "llm", {}).get("model", "gpt-4.1")
        temperature = getattr(settings, "llm", {}).get("temperature", 0.3)

        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

        # Prompts for different entity types
        self.function_prompt = PromptTemplate(
            input_variables=["code", "name", "params", "return_type", "docstring"],
            template="""Analyze this Python function and provide a concise natural language description of what it does.

Function Name: {name}
Parameters: {params}
Return Type: {return_type}
Docstring: {docstring}

Code:
```python
{code}
```

Provide a clear, concise description (2-3 sentences) explaining:
1. What the function does
2. What inputs it takes
3. What it returns
4. Any important side effects or behaviors

Description:""",
        )

        self.class_prompt = PromptTemplate(
            input_variables=["code", "name", "base_classes", "docstring", "methods"],
            template="""Analyze this Python class and provide a concise natural language description.

Class Name: {name}
Base Classes: {base_classes}
Docstring: {docstring}
Methods: {methods}

Code:
```python
{code}
```

Provide a clear, concise description (3-4 sentences) explaining:
1. The purpose and responsibility of this class
2. Key functionality it provides
3. How it relates to its base classes (if any)
4. The main operations it supports

Description:""",
        )

        self.module_prompt = PromptTemplate(
            input_variables=["name", "docstring", "imports", "classes", "functions"],
            template="""Analyze this Python module and provide a concise natural language description.

Module Name: {name}
Docstring: {docstring}
Key Imports: {imports}
Classes: {classes}
Functions: {functions}

Provide a clear, concise description (3-5 sentences) explaining:
1. The overall purpose of this module
2. What functionality it provides
3. The main components (classes and functions)
4. How it fits into the larger system

Description:""",
        )

    async def interpret_function(
        self,
        code: str,
        name: str,
        params: list[dict[str, Any]],
        return_type: str | None = None,
        docstring: str | None = None,
    ) -> str:
        """Generate natural language interpretation of a function."""
        try:
            # Format parameters
            param_str = ", ".join(
                [f"{p['name']}: {p.get('type', 'Any')}" for p in params],
            )

            prompt_value = self.function_prompt.format(
                code=code,
                name=name,
                params=param_str or "None",
                return_type=return_type or "None",
                docstring=docstring or "No docstring provided",
            )
            result = await self.llm.ainvoke(prompt_value)
            result = result.content if hasattr(result, "content") else str(result)

            return result.strip()

        except Exception:
            logger.exception("Error interpreting function %s", name)
            return f"Function {name} that takes {len(params)} parameters"

    async def interpret_class(
        self,
        code: str,
        name: str,
        base_classes: list[str],
        docstring: str | None = None,
        methods: list[str] | None = None,
    ) -> str:
        """Generate natural language interpretation of a class."""
        try:
            prompt_value = self.class_prompt.format(
                code=code[:3000],  # Limit code length
                name=name,
                base_classes=", ".join(base_classes) if base_classes else "None",
                docstring=docstring or "No docstring provided",
                methods=", ".join(methods[:10]) if methods else "None",
            )
            result = await self.llm.ainvoke(prompt_value)
            result = result.content if hasattr(result, "content") else str(result)

            return result.strip()

        except Exception:
            logger.exception("Error interpreting class %s", name)
            return f"Class {name} with {len(methods or [])} methods"

    async def interpret_module(
        self,
        name: str,
        docstring: str | None = None,
        imports: list[str] | None = None,
        classes: list[str] | None = None,
        functions: list[str] | None = None,
    ) -> str:
        """Generate natural language interpretation of a module."""
        try:
            prompt_value = self.module_prompt.format(
                name=name,
                docstring=docstring or "No module docstring",
                imports=", ".join(imports[:10]) if imports else "None",
                classes=", ".join(classes[:10]) if classes else "None",
                functions=", ".join(functions[:10]) if functions else "None",
            )
            result = await self.llm.ainvoke(prompt_value)
            result = result.content if hasattr(result, "content") else str(result)

            return result.strip()

        except Exception:
            logger.exception("Error interpreting module %s", name)
            return f"Module {name} containing {len(classes or [])} classes and {len(functions or [])} functions"

    async def batch_interpret(
        self,
        entities: list[dict[str, Any]],
        entity_type: str,
    ) -> list[str]:
        """Interpret multiple entities in batch."""
        interpretations = []

        for entity in entities:
            if entity_type == "function":
                interpretation = await self.interpret_function(
                    entity.get("code", ""),
                    entity.get("name", "unknown"),
                    entity.get("parameters", []),
                    entity.get("return_type"),
                    entity.get("docstring"),
                )
            elif entity_type == "class":
                interpretation = await self.interpret_class(
                    entity.get("code", ""),
                    entity.get("name", "unknown"),
                    entity.get("base_classes", []),
                    entity.get("docstring"),
                    entity.get("method_names", []),
                )
            elif entity_type == "module":
                interpretation = await self.interpret_module(
                    entity.get("name", "unknown"),
                    entity.get("docstring"),
                    entity.get("import_names", []),
                    entity.get("class_names", []),
                    entity.get("function_names", []),
                )
            else:
                interpretation = (
                    f"A {entity_type} named {entity.get('name', 'unknown')}"
                )

            interpretations.append(interpretation)

        return interpretations
