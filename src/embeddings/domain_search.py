"""Domain-aware semantic search using knowledge graph."""

from enum import Enum
from typing import Any

import numpy as np
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.database.domain_models import (
    BoundedContext,
    BoundedContextMembership,
    DomainEntity,
)
from src.database.models import Class, CodeEmbedding, File, Function, Module
from src.embeddings.openai_client import OpenAIClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Constants
MIN_CONCEPT_WORD_LENGTH = 3


class DomainSearchScope(Enum):
    """Scope for domain-aware search."""

    ALL = "all"
    BOUNDED_CONTEXT = "bounded_context"
    AGGREGATE = "aggregate"
    DOMAIN_SERVICE = "domain_service"
    ENTITY = "entity"
    VALUE_OBJECT = "value_object"


class DomainAwareSearch:
    """Perform semantic search with domain knowledge enhancement."""

    def __init__(
        self,
        db_session: AsyncSession,
        openai_client: OpenAIClient | None = None,
    ) -> None:
        """Initialize domain-aware search.

        Args:
            db_session: Database session
            openai_client: OpenAI client for query embeddings
        """
        self.db_session = db_session
        self.openai_client = openai_client or OpenAIClient()

    async def search_with_domain_context(
        self,
        query: str,
        *,
        scope: DomainSearchScope = DomainSearchScope.ALL,
        bounded_context: str | None = None,
        limit: int = 10,
        include_related: bool = True,
    ) -> list[dict[str, Any]]:
        """Search with domain knowledge enhancement.

        Args:
            query: Natural language search query
            scope: Domain search scope
            bounded_context: Optional context to search within
            limit: Maximum results
            include_related: Include related domain entities

        Returns:
            Search results enhanced with domain information
        """
        logger.info(
            "Domain search for '%s' in scope %s (context=%s)",
            query,
            scope.value,
            bounded_context,
        )

        # Step 1: Extract domain concepts from query
        domain_concepts = await self._extract_query_concepts(query)

        # Step 2: Find relevant domain entities
        relevant_entities = await self._find_relevant_entities(
            domain_concepts,
            scope,
            bounded_context,
        )

        # Step 3: Generate enhanced query embedding
        enhanced_query = await self._enhance_query_with_domain(
            query,
            relevant_entities,
        )
        query_embedding = await self.openai_client.generate_embedding(enhanced_query)

        # Step 4: Search with domain-weighted scoring
        results = await self._domain_weighted_search(
            query_embedding,
            relevant_entities,
            limit,
        )

        # Step 5: Enhance results with domain context
        if include_related:
            results = await self._enhance_results_with_domain(results)

        return results

    async def find_implementation(
        self,
        domain_entity_name: str,
    ) -> list[dict[str, Any]]:
        """Find code that implements a domain entity.

        Args:
            domain_entity_name: Name of domain entity

        Returns:
            Code locations implementing the entity
        """
        # Find domain entity
        result = await self.db_session.execute(
            select(DomainEntity).where(DomainEntity.name == domain_entity_name),
        )
        entity = result.scalar_one_or_none()

        if not entity:
            return []

        # Get source code entities
        implementations = []

        for source_id in entity.source_entities:
            # Get file info
            file_result = await self.db_session.execute(
                select(File).where(File.id == source_id),
            )
            file = file_result.scalar_one_or_none()

            if file:
                # Get specific code entities
                code_entities = await self._get_code_entities_in_file(file.id)

                implementations.append(
                    {
                        "file": file.path,
                        "domain_entity": {
                            "name": entity.name,
                            "type": entity.entity_type,
                            "description": entity.description,
                        },
                        "code_entities": code_entities,
                    },
                )

        return implementations

    async def search_by_business_capability(
        self,
        capability: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search for code by business capability.

        Args:
            capability: Business capability description
            limit: Maximum results

        Returns:
            Code implementing the capability
        """
        # Create capability-focused query
        enhanced_query = f"""
        Find code that implements the following business capability:
        {capability}

        Focus on domain entities, aggregates, and services that handle this capability.
        """

        # Use domain-aware search
        return await self.search_with_domain_context(
            enhanced_query,
            scope=DomainSearchScope.ALL,
            limit=limit,
        )

    async def _extract_query_concepts(
        self,
        query: str,
    ) -> list[str]:
        """Extract domain concepts from search query using LLM."""
        prompt = f"""
        Extract domain concepts and entities from this search query:
        "{query}"

        Return a JSON list of domain concepts mentioned or implied.
        Focus on business terms, not technical terms.

        Example: "find payment processing" -> ["Payment", "Order", "Transaction"]
        """

        try:
            response = await self.openai_client.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "Extract domain concepts from queries.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )

            import json

            result = json.loads(response)
            return result.get("concepts", [])

        except Exception:
            logger.exception("Error extracting concepts: %s")
            # Fallback to simple extraction
            return [word.capitalize() for word in query.split() if len(word) > MIN_CONCEPT_WORD_LENGTH]

    async def _find_relevant_entities(
        self,
        concepts: list[str],
        scope: DomainSearchScope,
        bounded_context: str | None,
    ) -> list[DomainEntity]:
        """Find domain entities relevant to the concepts."""
        query = select(DomainEntity)

        # Filter by scope
        if scope != DomainSearchScope.ALL:
            entity_types = {
                DomainSearchScope.AGGREGATE: ["aggregate_root"],
                DomainSearchScope.ENTITY: ["entity", "aggregate_root"],
                DomainSearchScope.VALUE_OBJECT: ["value_object"],
                DomainSearchScope.DOMAIN_SERVICE: ["domain_service"],
            }
            if scope in entity_types:
                query = query.where(DomainEntity.entity_type.in_(entity_types[scope]))

        # Filter by bounded context if specified
        if bounded_context:
            # Get context
            ctx_result = await self.db_session.execute(
                select(BoundedContext).where(BoundedContext.name == bounded_context),
            )
            context = ctx_result.scalar_one_or_none()

            if context:
                # Get entity IDs in context
                membership_result = await self.db_session.execute(
                    select(BoundedContextMembership.domain_entity_id).where(
                        BoundedContextMembership.bounded_context_id == context.id,
                    ),
                )
                entity_ids = [row[0] for row in membership_result]
                query = query.where(DomainEntity.id.in_(entity_ids))

        # Search by concept names
        if concepts:
            concept_conditions = [
                DomainEntity.name.ilike(f"%{concept}%") for concept in concepts
            ]
            query = query.where(func.or_(*concept_conditions))

        result = await self.db_session.execute(query.limit(20))
        return result.scalars().all()

    async def _enhance_query_with_domain(
        self,
        query: str,
        entities: list[DomainEntity],
    ) -> str:
        """Enhance query with domain knowledge."""
        if not entities:
            return query

        # Add domain context
        entity_context = []
        for entity in entities[:5]:  # Limit to prevent token overflow
            context = f"{entity.name} ({entity.entity_type})"
            if entity.description:
                context += f": {entity.description[:100]}"
            entity_context.append(context)

        enhanced = f"""
        {query}

        Related domain concepts:
        {chr(10).join(entity_context)}
        """

        return enhanced.strip()

    async def _domain_weighted_search(
        self,
        query_embedding: list[float],
        relevant_entities: list[DomainEntity],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Search with domain-weighted scoring."""
        # Get source entity IDs
        source_ids = set()
        for entity in relevant_entities:
            source_ids.update(entity.source_entities)

        # Convert to numpy array
        query_vector = np.array(query_embedding)

        # Search embeddings with boost for domain-relevant code
        query = select(
            CodeEmbedding.id,
            (1 - CodeEmbedding.embedding.cosine_distance(query_vector)).label(
                "base_similarity",
            ),
            CodeEmbedding.entity_type,
            CodeEmbedding.file_id,
            func.case(
                (CodeEmbedding.file_id.in_(list(source_ids)), 1.2),
                else_=1.0,
            ).label("domain_boost"),
        )

        # Calculate final score
        query = query.add_columns(
            (text("base_similarity * domain_boost")).label("final_score"),
        )

        # Order by final score
        query = query.order_by(text("final_score DESC")).limit(limit)

        result = await self.db_session.execute(query)
        rows = result.fetchall()

        # Format results
        results = []
        for row in rows:
            embedding_id = row[0]
            similarity = float(row[1])
            entity_type = row[2]
            domain_boost = float(row[4])
            final_score = float(row[5])

            # Load full embedding
            emb_result = await self.db_session.execute(
                select(CodeEmbedding)
                .where(CodeEmbedding.id == embedding_id)
                .options(selectinload(CodeEmbedding.file)),
            )
            embedding = emb_result.scalar_one_or_none()

            if embedding:
                results.append(
                    {
                        "embedding_id": embedding_id,
                        "base_similarity": similarity,
                        "domain_boost": domain_boost,
                        "final_score": final_score,
                        "entity_type": entity_type,
                        "text": embedding.content,
                        "file_path": embedding.file.path if embedding.file else None,
                    },
                )

        return results

    async def _enhance_results_with_domain(
        self,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Enhance search results with domain context."""
        for result in results:
            # Find domain entities related to this code
            file_path = result.get("file_path")
            if not file_path:
                continue

            # Get file
            file_result = await self.db_session.execute(
                select(File).where(File.path == file_path),
            )
            file = file_result.scalar_one_or_none()

            if not file:
                continue

            # Find domain entities
            entity_result = await self.db_session.execute(
                select(DomainEntity)
                .where(DomainEntity.source_entities.contains([file.id]))
                .limit(5),
            )
            domain_entities = entity_result.scalars().all()

            if domain_entities:
                result["domain_context"] = {
                    "entities": [
                        {
                            "name": e.name,
                            "type": e.entity_type,
                            "description": e.description,
                        }
                        for e in domain_entities
                    ],
                }

                # Find bounded context
                if domain_entities:
                    membership_result = await self.db_session.execute(
                        select(BoundedContextMembership)
                        .where(
                            BoundedContextMembership.domain_entity_id
                            == domain_entities[0].id,
                        )
                        .options(selectinload(BoundedContextMembership.bounded_context))
                        .limit(1),
                    )
                    membership = membership_result.scalar_one_or_none()

                    if membership:
                        result["domain_context"]["bounded_context"] = {
                            "name": membership.bounded_context.name,
                            "description": membership.bounded_context.description,
                        }

        return results

    async def _get_code_entities_in_file(
        self,
        file_id: int,
    ) -> list[dict[str, Any]]:
        """Get code entities in a file."""
        entities = []

        # Get classes
        class_result = await self.db_session.execute(
            select(Class)
            .join(Module, Class.module_id == Module.id)
            .where(Module.file_id == file_id)
            .limit(10),
        )

        for cls in class_result.scalars().all():
            entities.append(
                {
                    "type": "class",
                    "name": cls.name,
                    "line": cls.start_line,
                },
            )

        # Get functions
        func_result = await self.db_session.execute(
            select(Function)
            .join(Module, Function.module_id == Module.id)
            .where(Module.file_id == file_id)
            .limit(10),
        )

        for func in func_result.scalars().all():
            entities.append(
                {
                    "type": "function",
                    "name": func.name,
                    "line": func.start_line,
                },
            )

        return entities
