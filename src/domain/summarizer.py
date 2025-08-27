"""Hierarchical summarization of code using domain knowledge."""

import json
from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.config import settings
from src.database.domain_models import (
    BoundedContext,
    DomainEntity,
    DomainSummary,
)
from src.database.models import Class, Function, Module
from src.logger import get_logger
from src.utils.exceptions import NotFoundError

logger = get_logger(__name__)


class HierarchicalSummarizer:
    """Generate hierarchical summaries of code with domain context."""

    def __init__(
        self,
        db_session: AsyncSession,
        llm: Any = None,
    ) -> None:
        """Initialize the summarizer.

        Args:
            db_session: Database session
            llm: Optional LLM instance (for testing)
        """
        self.db_session = db_session

        if llm is not None:
            self.llm = llm
        else:
            # Try to create from settings
            try:
                # Handle different ways settings might store the API key
                openai_key = None
                if hasattr(settings, "OPENAI_API_KEY"):
                    openai_key = settings.OPENAI_API_KEY
                elif hasattr(settings, "openai_api_key"):
                    if hasattr(settings.openai_api_key, "get_secret_value"):
                        openai_key = settings.openai_api_key.get_secret_value()
                    else:
                        openai_key = settings.openai_api_key

                if not openai_key:
                    msg = "OpenAI API key not found"
                    raise ValueError(msg)  # noqa: TRY301

                # Treat ChatOpenAI client as Any for type stability
                llm_client = cast("Any", ChatOpenAI)
                self.llm = llm_client(
                    api_key=openai_key,
                    model=settings.llm.model,
                    temperature=settings.llm.temperature,
                )
            except (AttributeError, KeyError, ValueError) as e:
                logger.warning("Failed to initialize LLM: %s", e)
                self.llm = None

    async def summarize_function(
        self,
        function_id: int,
        *,
        include_domain_context: bool = True,
    ) -> DomainSummary:
        """Generate domain-aware summary of a function.

        Args:
            function_id: Function ID
            include_domain_context: Whether to include domain entity context

        Returns:
            Domain summary object
        """
        # Load function
        result = await self.db_session.execute(
            select(Function)
            .where(Function.id == function_id)
            .options(
                selectinload(Function.module).selectinload(Module.file),
                selectinload(Function.parent_class),
            ),
        )
        function = result.scalar_one_or_none()

        if not function:
            msg = "Function not found"
            raise NotFoundError(msg)

        # Get related domain entities
        domain_entities = []
        if include_domain_context:
            domain_entities = await self._get_related_domain_entities(
                "function",
                function_id,
            )

        # Generate summary
        summary_data = await self._generate_function_summary(
            function,
            domain_entities,
        )

        # Create summary record
        summary = DomainSummary(
            level="function",
            entity_type="function",
            entity_id=function_id,
            business_summary=summary_data["business_summary"],
            technical_summary=summary_data["technical_summary"],
            domain_concepts=summary_data["domain_concepts"],
        )

        self.db_session.add(summary)
        await self.db_session.commit()

        return summary

    async def summarize_class(
        self,
        class_id: int,
        *,
        include_methods: bool = True,
    ) -> DomainSummary:
        """Generate domain-aware summary of a class.

        Args:
            class_id: Class ID
            include_methods: Whether to include method summaries

        Returns:
            Domain summary object
        """
        # Load class with methods
        result = await self.db_session.execute(
            select(Class)
            .where(Class.id == class_id)
            .options(
                selectinload(Class.module).selectinload(Module.file),
                selectinload(Class.methods),
            ),
        )
        class_obj = result.scalar_one_or_none()

        if not class_obj:
            msg = "Class not found"
            raise NotFoundError(msg)

        # Get method summaries if requested
        method_summaries = []
        if include_methods:
            for method in class_obj.methods:
                # Check if summary exists
                result = await self.db_session.execute(
                    select(DomainSummary).where(
                        DomainSummary.entity_type == "function",
                        DomainSummary.entity_id == method.id,
                    ),
                )
                method_summary = result.scalar_one_or_none()

                if not method_summary:
                    method_summary = await self.summarize_function(method.id)

                method_summaries.append(method_summary)

        # Get related domain entities
        domain_entities = await self._get_related_domain_entities(
            "class",
            class_id,
        )

        # Generate summary
        summary_data = await self._generate_class_summary(
            class_obj,
            domain_entities,
            method_summaries,
        )

        # Create summary record
        summary = DomainSummary(
            level="class",
            entity_type="class",
            entity_id=class_id,
            business_summary=summary_data["business_summary"],
            technical_summary=summary_data["technical_summary"],
            domain_concepts=summary_data["domain_concepts"],
        )

        self.db_session.add(summary)
        await self.db_session.commit()

        return summary

    async def summarize_module(
        self,
        module_id: int,
    ) -> DomainSummary:
        """Generate domain-aware summary of a module.

        Args:
            module_id: Module ID

        Returns:
            Domain summary object
        """
        # Load module with classes and functions
        result = await self.db_session.execute(
            select(Module)
            .where(Module.id == module_id)
            .options(
                selectinload(Module.file),
                selectinload(Module.classes),
                selectinload(Module.functions),
            ),
        )
        module = result.scalar_one_or_none()

        if not module:
            msg = "Module not found"
            raise NotFoundError(msg)

        # Get summaries for classes and functions
        class_summaries = []
        for class_obj in module.classes:
            result = await self.db_session.execute(
                select(DomainSummary).where(
                    DomainSummary.entity_type == "class",
                    DomainSummary.entity_id == class_obj.id,
                ),
            )
            class_summary = result.scalar_one_or_none()

            if not class_summary:
                class_summary = await self.summarize_class(class_obj.id)

            class_summaries.append(class_summary)

        function_summaries = []
        for function in module.functions:
            result = await self.db_session.execute(
                select(DomainSummary).where(
                    DomainSummary.entity_type == "function",
                    DomainSummary.entity_id == function.id,
                ),
            )
            func_summary = result.scalar_one_or_none()

            if not func_summary:
                func_summary = await self.summarize_function(function.id)

            function_summaries.append(func_summary)

        # Get related domain entities
        domain_entities = await self._get_related_domain_entities(
            "module",
            module_id,
        )

        # Generate summary
        summary_data = await self._generate_module_summary(
            module,
            domain_entities,
            class_summaries,
            function_summaries,
        )

        # Create summary record
        summary = DomainSummary(
            level="module",
            entity_type="module",
            entity_id=module_id,
            business_summary=summary_data["business_summary"],
            technical_summary=summary_data["technical_summary"],
            domain_concepts=summary_data["domain_concepts"],
        )

        self.db_session.add(summary)
        await self.db_session.commit()

        return summary

    async def summarize_bounded_context(
        self,
        context_id: int,
    ) -> DomainSummary:
        """Generate summary of a bounded context.

        Args:
            context_id: Bounded context ID

        Returns:
            Domain summary object
        """
        # Load context with entities
        result = await self.db_session.execute(
            select(BoundedContext)
            .where(BoundedContext.id == context_id)
            .options(selectinload(BoundedContext.memberships)),
        )
        context = result.scalar_one_or_none()

        if not context:
            msg = "Context not found"
            raise NotFoundError(msg)

        # Load domain entities
        entity_ids = [m.domain_entity_id for m in context.memberships]
        result = await self.db_session.execute(
            select(DomainEntity).where(DomainEntity.id.in_(entity_ids)),
        )
        domain_entities: list[DomainEntity] = list(result.scalars().all())

        # Generate summary
        summary_data = await self._generate_context_summary(
            context,
            domain_entities,
        )

        # Update context with summary
        context.summary = summary_data["business_summary"]

        # Create summary record
        summary = DomainSummary(
            level="context",
            entity_type="bounded_context",
            entity_id=context_id,
            business_summary=summary_data["business_summary"],
            technical_summary=summary_data["technical_summary"],
            domain_concepts=summary_data["domain_concepts"],
        )

        self.db_session.add(summary)
        await self.db_session.commit()

        return summary

    async def _get_related_domain_entities(
        self,
        _entity_type: str,
        _entity_id: int,
    ) -> list[DomainEntity]:
        """Get domain entities related to a code entity."""
        # This is simplified - would need to map based on source_entities
        result = await self.db_session.execute(
            select(DomainEntity).limit(10),  # Placeholder
        )
        return list(result.scalars().all())

    async def _generate_function_summary(
        self,
        function: Function,
        domain_entities: list[DomainEntity],
    ) -> dict[str, Any]:
        """Generate summary for a function using LLM."""
        # Build context
        context_parts = []

        # Function details
        context_parts.append(f"Function: {function.name}")
        if function.parent_class:
            context_parts.append(f"Class: {function.parent_class.name}")
        if function.docstring:
            context_parts.append(f"Docstring: {function.docstring}")
        if function.parameters:
            context_parts.append(f"Parameters: {function.parameters}")
        if function.return_type:
            context_parts.append(f"Returns: {function.return_type}")

        # Domain context
        if domain_entities:
            entity_names = [str(e.name) for e in domain_entities[:5]]
            context_parts.append(f"Related domain entities: {', '.join(entity_names)}")

        context = "\n".join(context_parts)

        # Generate summaries
        prompt = f"""Given this function:
{context}

Generate:
1. A business-oriented summary (what business capability it provides)
2. A technical summary (how it works)
3. List of domain concepts involved

Output as JSON:
{{
  "business_summary": "...",
  "technical_summary": "...",
  "domain_concepts": ["concept1", "concept2"]
}}"""

        try:
            messages = [
                SystemMessage(
                    content="You are analyzing code to extract business and technical summaries."
                ),
                HumanMessage(content=prompt),
            ]

            # mypy: response.content is Any; coerce to str before json loads

            response = await self.llm.ainvoke(
                messages,
                config={"configurable": {"response_format": {"type": "json_object"}}},
            )

            result: dict[str, Any] = cast(
                "dict[str, Any]", json.loads(str(response.content))
            )
            return result

        except Exception:
            logger.exception("Error generating function summary: %s")
            return {
                "business_summary": f"Function {function.name} implementation",
                "technical_summary": function.docstring or "No description available",
                "domain_concepts": [],
            }

    async def _generate_class_summary(
        self,
        class_obj: Class,
        domain_entities: list[DomainEntity],
        method_summaries: list[DomainSummary],
    ) -> dict[str, Any]:
        """Generate summary for a class using LLM."""
        # Aggregate method summaries
        method_concepts: set[str] = set()
        method_descriptions = []

        for summary in method_summaries[:10]:  # Limit to prevent token overflow
            method_concepts.update(summary.domain_concepts)
            method_descriptions.append(summary.business_summary)

        safe_method_descriptions: list[str] = [str(d) for d in method_descriptions[:5]]
        related_domain_names: list[str] = [str(e.name) for e in domain_entities[:5]]
        context = f"""Class: {class_obj.name}
Base classes: {class_obj.base_classes}
Docstring: {class_obj.docstring or "None"}
Number of methods: {len(method_summaries)}
Key methods: {", ".join(safe_method_descriptions)}
Related domain entities: {", ".join(related_domain_names)}"""

        prompt = f"""Analyze this class and generate a hierarchical summary:
{context}

Consider the method summaries and domain entities to understand the class's role.

Output as JSON:
{{
  "business_summary": "What business capability this class provides",
  "technical_summary": "Technical implementation and patterns used",
  "domain_concepts": ["main concepts this class deals with"]
}}"""

        try:
            messages = [
                SystemMessage(
                    content="You are analyzing classes to understand their business and technical roles."
                ),
                HumanMessage(content=prompt),
            ]
            response = await self.llm.ainvoke(
                messages,
                config={"configurable": {"response_format": {"type": "json_object"}}},
            )

            result: dict[str, Any] = cast(
                "dict[str, Any]", json.loads(str(response.content))
            )
            # Add method concepts
            result["domain_concepts"] = list(
                set(result.get("domain_concepts", [])) | method_concepts,
            )
            return result

        except Exception:
            logger.exception("Error generating class summary: %s")
            return {
                "business_summary": f"Class {class_obj.name} implementation",
                "technical_summary": class_obj.docstring or "No description available",
                "domain_concepts": list(method_concepts),
            }

    async def _generate_module_summary(
        self,
        module: Module,
        domain_entities: list[DomainEntity],
        class_summaries: list[DomainSummary],
        function_summaries: list[DomainSummary],
    ) -> dict[str, Any]:
        """Generate summary for a module using LLM."""
        # Aggregate concepts
        all_concepts: set[str] = set()
        for summary in class_summaries + function_summaries:
            all_concepts.update(summary.domain_concepts)

        concepts_list: list[str] = list(all_concepts)[:20]
        related_names: list[str] = [str(e.name) for e in domain_entities[:10]]
        context = f"""Module: {module.name}
File: {module.file.path if module.file else "Unknown"}
Classes: {len(class_summaries)}
Functions: {len(function_summaries)}
Domain concepts found: {", ".join(concepts_list)}
Related domain entities: {", ".join(related_names)}"""

        prompt = f"""Analyze this module and generate a high-level summary:
{context}

Synthesize the information to understand the module's purpose and architecture.

Output as JSON:
{{
  "business_summary": "What business capabilities this module provides",
  "technical_summary": "Technical architecture and patterns",
  "domain_concepts": ["key domain concepts in priority order"]
}}"""

        try:
            messages = [
                SystemMessage(
                    content="You are analyzing modules to understand their role in the system."
                ),
                HumanMessage(content=prompt),
            ]
            response = await self.llm.ainvoke(
                messages,
                config={"configurable": {"response_format": {"type": "json_object"}}},
            )

            result: dict[str, Any] = cast(
                "dict[str, Any]", json.loads(str(response.content))
            )
            return result

        except Exception:
            logger.exception("Error generating module summary: %s")
            return {
                "business_summary": f"Module {module.name}",
                "technical_summary": module.docstring or "Module implementation",
                "domain_concepts": list(all_concepts)[:10],
            }

    async def _generate_context_summary(
        self,
        context: BoundedContext,
        domain_entities: list[DomainEntity],
    ) -> dict[str, Any]:
        """Generate summary for a bounded context."""
        # Collect entity information
        entity_types: dict[str, int] = {}
        for entity in domain_entities:
            key = str(entity.entity_type)
            entity_types[key] = entity_types.get(key, 0) + 1

        core_concepts_list: list[str] = [
            str(c) for c in list(context.core_concepts)[:10]
        ]
        entity_types_items: list[str] = [f"{k!s}: {v}" for k, v in entity_types.items()]
        context_info = f"""Bounded Context: {context.name}
Description: {context.description}
Core concepts: {", ".join(core_concepts_list)}
Number of entities: {len(domain_entities)}
Entity types: {", ".join(entity_types_items)}
Cohesion score: {context.cohesion_score:.2f}
Coupling score: {context.coupling_score:.2f}"""

        prompt = f"""Analyze this bounded context and generate a comprehensive summary:
{context_info}

Key entities:
{chr(10).join([f"- {e.name!s} ({e.entity_type!s}): {e.description!s}" for e in domain_entities[:15]])}

Generate:
1. A business-oriented summary of what this context handles
2. Technical patterns and architecture used
3. Most important domain concepts

Output as JSON:
{{
  "business_summary": "...",
  "technical_summary": "...",
  "domain_concepts": ["..."]
}}"""

        try:
            messages = [
                SystemMessage(
                    content="You are analyzing bounded contexts in a Domain-Driven Design system."
                ),
                HumanMessage(content=prompt),
            ]
            response = await self.llm.ainvoke(
                messages,
                config={"configurable": {"response_format": {"type": "json_object"}}},
            )

            result: dict[str, Any] = cast(
                "dict[str, Any]", json.loads(str(response.content))
            )
            return result

        except Exception:
            logger.exception("Error generating context summary: %s")
            return {
                "business_summary": context.description
                or f"Bounded context {context.name}",
                "technical_summary": f"Contains {len(domain_entities)} domain entities",
                "domain_concepts": context.core_concepts[:10],
            }
