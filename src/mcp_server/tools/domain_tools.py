"""Domain-driven MCP tools for semantic code analysis."""

from typing import Any

from fastmcp import FastMCP
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.database.domain_models import (
    BoundedContext,
    BoundedContextMembership,
    DomainEntity,
    DomainRelationship,
    DomainSummary,
)
from src.database.models import File
from src.embeddings.openai_client import OpenAIClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Pydantic models for tool parameters
class ExtractDomainModelRequest(BaseModel):
    """Request to extract domain model."""

    code_path: str = Field(..., description="Path to file or module to analyze")
    include_relationships: bool = Field(
        default=True,
        description="Whether to extract relationships",
    )


class FindAggregateRootsRequest(BaseModel):
    """Request to find aggregate roots."""

    context_name: str | None = Field(
        None,
        description="Optional bounded context to search within",
    )


class AnalyzeBoundedContextRequest(BaseModel):
    """Request to analyze a bounded context."""

    context_name: str = Field(..., description="Name of the bounded context")


class SuggestDDDRefactoringRequest(BaseModel):
    """Request for DDD refactoring suggestions."""

    code_path: str = Field(..., description="Path to analyze")


class FindBoundedContextsRequest(BaseModel):
    """Request to find bounded contexts."""

    min_entities: int = Field(
        default=3,
        description="Minimum number of entities for a context",
    )


class GenerateContextMapRequest(BaseModel):
    """Request to generate context map."""

    output_format: str = Field(
        default="json",
        description="Output format: json, mermaid, or plantuml",
    )


class DomainTools:
    """MCP tools for domain-driven code analysis."""

    def __init__(
        self,
        db_session: AsyncSession,
        openai_client: OpenAIClient | None,
        mcp: FastMCP,
    ) -> None:
        """Initialize domain tools.

        Args:
            db_session: Database session
            openai_client: OpenAI client for LLM operations
            mcp: FastMCP instance
        """
        self.db_session = db_session
        self.openai_client = openai_client
        self.mcp = mcp

    async def register_tools(self) -> None:
        """Register all domain-driven tools."""

        @self.mcp.tool(
            name="extract_domain_model",
            description="Extract domain entities and relationships from code using LLM analysis",
        )
        async def extract_domain_model(
            request: ExtractDomainModelRequest,
        ) -> dict[str, Any]:
            """Extract domain model from code."""
            return await self.extract_domain_model(
                request.code_path,
                request.include_relationships,
            )

        @self.mcp.tool(
            name="find_aggregate_roots",
            description="Find aggregate roots in the codebase using domain analysis",
        )
        async def find_aggregate_roots(
            request: FindAggregateRootsRequest,
        ) -> list[dict[str, Any]]:
            """Find aggregate roots."""
            return await self.find_aggregate_roots(request.context_name)

        @self.mcp.tool(
            name="analyze_bounded_context",
            description="Analyze a bounded context and its relationships",
        )
        async def analyze_bounded_context(
            request: AnalyzeBoundedContextRequest,
        ) -> dict[str, Any]:
            """Analyze bounded context."""
            return await self.analyze_bounded_context(request.context_name)

        @self.mcp.tool(
            name="suggest_ddd_refactoring",
            description="Suggest Domain-Driven Design refactoring improvements",
        )
        async def suggest_ddd_refactoring(
            request: SuggestDDDRefactoringRequest,
        ) -> list[dict[str, Any]]:
            """Suggest DDD refactoring."""
            return await self.suggest_ddd_refactoring(request.code_path)

        @self.mcp.tool(
            name="find_bounded_contexts",
            description="Find all bounded contexts in the codebase",
        )
        async def find_bounded_contexts(
            request: FindBoundedContextsRequest,
        ) -> list[dict[str, Any]]:
            """Find bounded contexts."""
            return await self.find_bounded_contexts(request.min_entities)

        @self.mcp.tool(
            name="generate_context_map",
            description="Generate a context map showing relationships between bounded contexts",
        )
        async def generate_context_map(
            request: GenerateContextMapRequest,
        ) -> dict[str, Any]:
            """Generate context map."""
            return await self.generate_context_map(request.output_format)

    async def extract_domain_model(
        self,
        code_path: str,
        include_relationships: bool = True,
    ) -> dict[str, Any]:
        """
        Extract domain entities and relationships from code.

        Args:
            code_path: Path to file or module to analyze
            include_relationships: Whether to extract relationships

        Returns:
            Dictionary containing domain model
        """
        try:
            # Find the file
            result = await self.db_session.execute(
                select(File).where(File.path.endswith(code_path)),
            )
            file = result.scalar_one_or_none()

            if not file:
                return {
                    "error": f"File not found: {code_path}",
                    "entities": [],
                    "relationships": [],
                }

            # Check if already indexed
            result = await self.db_session.execute(
                select(DomainEntity).where(
                    DomainEntity.source_entities.contains([file.id]),
                ),
            )
            existing_entities = result.scalars().all()

            if existing_entities:
                # Return existing domain model
                entities = [
                    {
                        "name": e.name,
                        "type": e.entity_type,
                        "description": e.description,
                        "business_rules": e.business_rules,
                        "invariants": e.invariants,
                        "responsibilities": e.responsibilities,
                    }
                    for e in existing_entities
                ]

                relationships = []
                if include_relationships:
                    # Get relationships
                    entity_ids = [e.id for e in existing_entities]
                    result = await self.db_session.execute(
                        select(DomainRelationship)
                        .where(DomainRelationship.source_entity_id.in_(entity_ids))
                        .options(
                            selectinload(DomainRelationship.source_entity),
                            selectinload(DomainRelationship.target_entity),
                        ),
                    )

                    for rel in result.scalars().all():
                        relationships.append(
                            {
                                "source": rel.source_entity.name,
                                "target": rel.target_entity.name,
                                "type": rel.relationship_type,
                                "description": rel.description,
                            },
                        )

                return {
                    "file": code_path,
                    "entities": entities,
                    "relationships": relationships,
                    "source": "cached",
                }

            # Extract new domain model
            from src.domain.indexer import DomainIndexer

            indexer = DomainIndexer(self.db_session, self.openai_client)
            result = await indexer.index_file(file.id)

            if result["status"] != "success":
                return {
                    "error": f"Failed to extract domain model: {result.get('error')}",
                    "entities": [],
                    "relationships": [],
                }

            # Fetch the extracted entities
            result = await self.db_session.execute(
                select(DomainEntity).where(
                    DomainEntity.source_entities.contains([file.id]),
                ),
            )
            entities = result.scalars().all()

            entity_data = [
                {
                    "name": e.name,
                    "type": e.entity_type,
                    "description": e.description,
                    "business_rules": e.business_rules,
                    "invariants": e.invariants,
                    "responsibilities": e.responsibilities,
                }
                for e in entities
            ]

            relationship_data = []
            if include_relationships and entities:
                entity_ids = [e.id for e in entities]
                result = await self.db_session.execute(
                    select(DomainRelationship)
                    .where(DomainRelationship.source_entity_id.in_(entity_ids))
                    .options(
                        selectinload(DomainRelationship.source_entity),
                        selectinload(DomainRelationship.target_entity),
                    ),
                )

                for rel in result.scalars().all():
                    relationship_data.append(
                        {
                            "source": rel.source_entity.name,
                            "target": rel.target_entity.name,
                            "type": rel.relationship_type,
                            "description": rel.description,
                        },
                    )

            return {
                "file": code_path,
                "entities": entity_data,
                "relationships": relationship_data,
                "source": "extracted",
            }

        except Exception as e:
            logger.exception("Error extracting domain model: %s")
            return {
                "error": str(e),
                "entities": [],
                "relationships": [],
            }

    async def find_aggregate_roots(
        self,
        context_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Find aggregate roots in the codebase.

        Args:
            context_name: Optional bounded context to search within

        Returns:
            List of aggregate roots with their details
        """
        try:
            query = select(DomainEntity).where(
                DomainEntity.entity_type == "aggregate_root",
            )

            # Filter by context if specified
            if context_name:
                # Find context
                context_result = await self.db_session.execute(
                    select(BoundedContext).where(BoundedContext.name == context_name),
                )
                context = context_result.scalar_one_or_none()

                if not context:
                    return []

                # Get entity IDs in context
                membership_result = await self.db_session.execute(
                    select(BoundedContextMembership.domain_entity_id).where(
                        BoundedContextMembership.bounded_context_id == context.id,
                    ),
                )
                entity_ids = [row[0] for row in membership_result]

                query = query.where(DomainEntity.id.in_(entity_ids))

            result = await self.db_session.execute(query)
            aggregates = result.scalars().all()

            aggregate_data = []
            for agg in aggregates:
                # Get related entities (members of the aggregate)
                member_result = await self.db_session.execute(
                    select(DomainEntity)
                    .join(
                        DomainRelationship,
                        DomainRelationship.target_entity_id == DomainEntity.id,
                    )
                    .where(
                        DomainRelationship.source_entity_id == agg.id,
                        DomainRelationship.relationship_type.in_(
                            ["aggregates", "composed_of"],
                        ),
                    ),
                )
                members = member_result.scalars().all()

                aggregate_data.append(
                    {
                        "name": agg.name,
                        "description": agg.description,
                        "invariants": agg.invariants,
                        "members": [
                            {"name": m.name, "type": m.entity_type} for m in members
                        ],
                        "source_files": await self._get_source_files(agg),
                    },
                )

            return aggregate_data

        except Exception:
            logger.exception("Error finding aggregate roots: %s")
            return []

    async def analyze_bounded_context(
        self,
        context_name: str,
    ) -> dict[str, Any]:
        """
        Analyze a bounded context and its relationships.

        Args:
            context_name: Name of the bounded context

        Returns:
            Context analysis including entities, relationships, and metrics
        """
        try:
            # Find context
            result = await self.db_session.execute(
                select(BoundedContext)
                .where(BoundedContext.name == context_name)
                .options(selectinload(BoundedContext.memberships)),
            )
            context = result.scalar_one_or_none()

            if not context:
                return {"error": f"Bounded context not found: {context_name}"}

            # Get entities in context
            entity_ids = [m.domain_entity_id for m in context.memberships]
            entity_result = await self.db_session.execute(
                select(DomainEntity).where(DomainEntity.id.in_(entity_ids)),
            )
            entities = entity_result.scalars().all()

            # Group entities by type
            entities_by_type = {}
            for entity in entities:
                entity_type = entity.entity_type
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = []
                entities_by_type[entity_type].append(
                    {
                        "name": entity.name,
                        "description": entity.description,
                    },
                )

            # Get relationships within context
            rel_result = await self.db_session.execute(
                select(DomainRelationship)
                .where(
                    DomainRelationship.source_entity_id.in_(entity_ids),
                    DomainRelationship.target_entity_id.in_(entity_ids),
                )
                .options(
                    selectinload(DomainRelationship.source_entity),
                    selectinload(DomainRelationship.target_entity),
                ),
            )
            internal_relationships = rel_result.scalars().all()

            # Get external relationships
            ext_rel_result = await self.db_session.execute(
                select(DomainRelationship)
                .where(
                    DomainRelationship.source_entity_id.in_(entity_ids),
                    ~DomainRelationship.target_entity_id.in_(entity_ids),
                )
                .options(
                    selectinload(DomainRelationship.source_entity),
                    selectinload(DomainRelationship.target_entity),
                ),
            )
            external_relationships = ext_rel_result.scalars().all()

            # Get summary if exists
            summary_result = await self.db_session.execute(
                select(DomainSummary).where(
                    DomainSummary.entity_type == "bounded_context",
                    DomainSummary.entity_id == context.id,
                ),
            )
            summary = summary_result.scalar_one_or_none()

            return {
                "name": context.name,
                "description": context.description,
                "ubiquitous_language": context.ubiquitous_language,
                "core_concepts": context.core_concepts,
                "entities_by_type": entities_by_type,
                "total_entities": len(entities),
                "internal_relationships": len(internal_relationships),
                "external_dependencies": len(external_relationships),
                "cohesion_score": context.cohesion_score,
                "coupling_score": context.coupling_score,
                "modularity_score": context.modularity_score,
                "summary": summary.business_summary if summary else None,
            }

        except Exception as e:
            logger.exception("Error analyzing bounded context: %s")
            return {"error": str(e)}

    async def suggest_ddd_refactoring(
        self,
        code_path: str,
    ) -> list[dict[str, Any]]:
        """
        Suggest Domain-Driven Design refactoring improvements.

        Args:
            code_path: Path to analyze

        Returns:
            List of DDD-based refactoring suggestions
        """
        try:
            # Find file
            result = await self.db_session.execute(
                select(File).where(File.path.endswith(code_path)),
            )
            file = result.scalar_one_or_none()

            if not file:
                return [{"error": f"File not found: {code_path}"}]

            # Get domain entities from this file
            result = await self.db_session.execute(
                select(DomainEntity).where(
                    DomainEntity.source_entities.contains([file.id]),
                ),
            )
            entities = result.scalars().all()

            suggestions = []

            # Check for missing aggregate roots
            has_aggregate = any(e.entity_type == "aggregate_root" for e in entities)
            if entities and not has_aggregate:
                suggestions.append(
                    {
                        "type": "missing_aggregate",
                        "severity": "high",
                        "message": "No aggregate root found",
                        "suggestion": "Identify the main entity that maintains consistency and make it an aggregate root",
                        "entities": [e.name for e in entities],
                    },
                )

            # Check for anemic domain models
            for entity in entities:
                if entity.entity_type in ["entity", "aggregate_root"] and not entity.business_rules and not entity.invariants:
                        suggestions.append(
                            {
                                "type": "anemic_domain_model",
                                "severity": "medium",
                                "entity": entity.name,
                                "message": f"Entity '{entity.name}' has no business rules or invariants",
                                "suggestion": "Move business logic into the entity to create a rich domain model",
                            },
                        )

            # Check for missing value objects
            # Simple heuristic: entities with few responsibilities might be value objects
            for entity in entities:
                if entity.entity_type == "entity" and len(entity.responsibilities) <= 1:
                    suggestions.append(
                        {
                            "type": "potential_value_object",
                            "severity": "low",
                            "entity": entity.name,
                            "message": f"Entity '{entity.name}' might be better as a value object",
                            "suggestion": "Consider making this a value object if it has no identity and is defined by its attributes",
                        },
                    )

            # Check for missing domain services
            # Look for entities with too many responsibilities
            for entity in entities:
                if len(entity.responsibilities) > 5:
                    suggestions.append(
                        {
                            "type": "bloated_entity",
                            "severity": "medium",
                            "entity": entity.name,
                            "message": f"Entity '{entity.name}' has too many responsibilities ({len(entity.responsibilities)})",
                            "suggestion": "Extract some responsibilities into domain services",
                            "responsibilities": [*entity.responsibilities[:5], "..."],
                        },
                    )

            # Check bounded context cohesion
            if entities:
                # Find which contexts these entities belong to
                entity_ids = [e.id for e in entities]
                membership_result = await self.db_session.execute(
                    select(BoundedContextMembership)
                    .where(BoundedContextMembership.domain_entity_id.in_(entity_ids))
                    .options(selectinload(BoundedContextMembership.bounded_context)),
                )
                memberships = membership_result.scalars().all()

                contexts = {m.bounded_context.name for m in memberships}
                if len(contexts) > 1:
                    suggestions.append(
                        {
                            "type": "context_boundary_violation",
                            "severity": "high",
                            "message": f"File contains entities from {len(contexts)} different bounded contexts",
                            "contexts": list(contexts),
                            "suggestion": "Split this file so each file contains entities from only one bounded context",
                        },
                    )

            return suggestions

        except Exception as e:
            logger.exception("Error suggesting DDD refactoring: %s")
            return [{"error": str(e)}]

    async def find_bounded_contexts(
        self,
        min_entities: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Find all bounded contexts in the codebase.

        Args:
            min_entities: Minimum number of entities for a context

        Returns:
            List of bounded contexts with their details
        """
        try:
            # Get all contexts
            result = await self.db_session.execute(
                select(BoundedContext).options(
                    selectinload(BoundedContext.memberships),
                ),
            )
            contexts = result.scalars().all()

            context_data = []
            for context in contexts:
                if len(context.memberships) >= min_entities:
                    context_data.append(
                        {
                            "name": context.name,
                            "description": context.description,
                            "entity_count": len(context.memberships),
                            "core_concepts": context.core_concepts[:5],
                            "cohesion_score": context.cohesion_score,
                            "type": context.context_type,
                        },
                    )

            # Sort by cohesion score
            context_data.sort(key=lambda x: x.get("cohesion_score", 0), reverse=True)

            return context_data

        except Exception:
            logger.exception("Error finding bounded contexts: %s")
            return []

    async def generate_context_map(
        self,
        output_format: str = "json",
    ) -> dict[str, Any]:
        """
        Generate a context map showing relationships between bounded contexts.

        Args:
            output_format: Output format (json, mermaid, or plantuml)

        Returns:
            Context map in requested format
        """
        try:
            # Get all contexts
            ctx_result = await self.db_session.execute(select(BoundedContext))
            contexts = ctx_result.scalars().all()

            # Get context relationships
            from src.database.domain_models import ContextRelationship

            rel_result = await self.db_session.execute(
                select(ContextRelationship).options(
                    selectinload(ContextRelationship.source_context),
                    selectinload(ContextRelationship.target_context),
                ),
            )
            relationships = rel_result.scalars().all()

            if output_format == "json":
                return {
                    "contexts": [
                        {
                            "id": ctx.id,
                            "name": ctx.name,
                            "type": ctx.context_type,
                            "description": ctx.description,
                        }
                        for ctx in contexts
                    ],
                    "relationships": [
                        {
                            "source": rel.source_context.name,
                            "target": rel.target_context.name,
                            "type": rel.relationship_type,
                            "description": rel.description,
                        }
                        for rel in relationships
                    ],
                }

            if output_format == "mermaid":
                # Generate Mermaid diagram
                lines = ["graph TD"]

                # Add contexts
                for ctx in contexts:
                    shape = "[[" if ctx.context_type == "core" else "["
                    shape_end = "]]" if ctx.context_type == "core" else "]"
                    lines.append(f"    {ctx.name}{shape}{ctx.name}{shape_end}")

                # Add relationships
                for rel in relationships:
                    arrow = self._get_mermaid_arrow(rel.relationship_type)
                    lines.append(
                        f"    {rel.source_context.name} {arrow} {rel.target_context.name}",
                    )

                return {"diagram": "\n".join(lines)}

            if output_format == "plantuml":
                # Generate PlantUML diagram
                lines = [
                    "@startuml",
                    "!define RECTANGLE skinparam rectangleBackgroundColor",
                    "",
                ]

                # Add contexts
                for ctx in contexts:
                    stereotype = "<<Core>>" if ctx.context_type == "core" else ""
                    lines.append(f'package "{ctx.name}" {stereotype} {{')
                    lines.append("}")
                    lines.append("")

                # Add relationships
                for rel in relationships:
                    arrow = self._get_plantuml_arrow(rel.relationship_type)
                    lines.append(
                        f'"{rel.source_context.name}" {arrow} "{rel.target_context.name}"',
                    )

                lines.append("@enduml")
                return {"diagram": "\n".join(lines)}

            return {"error": f"Unsupported format: {output_format}"}

        except Exception as e:
            logger.exception("Error generating context map: %s")
            return {"error": str(e)}

    async def _get_source_files(
        self,
        entity: DomainEntity,
    ) -> list[str]:
        """Get source files for a domain entity."""
        if not entity.source_entities:
            return []

        result = await self.db_session.execute(
            select(File.path).where(File.id.in_(entity.source_entities)),
        )
        return [row[0] for row in result]

    def _get_mermaid_arrow(self, relationship_type: str) -> str:
        """Get Mermaid arrow for relationship type."""
        arrows = {
            "shared_kernel": "-.->|SK|",
            "customer_supplier": "-->|C/S|",
            "conformist": "-->|CF|",
            "anti_corruption_layer": "-->|ACL|",
            "open_host_service": "-->|OHS|",
            "published_language": "-->|PL|",
            "partnership": "<-->|P|",
            "big_ball_of_mud": "~~~|BBoM|",
        }
        return arrows.get(relationship_type, "-->")

    def _get_plantuml_arrow(self, relationship_type: str) -> str:
        """Get PlantUML arrow for relationship type."""
        arrows = {
            "shared_kernel": "..> : <<Shared Kernel>>",
            "customer_supplier": "--> : <<Customer/Supplier>>",
            "conformist": "--> : <<Conformist>>",
            "anti_corruption_layer": "--> : <<ACL>>",
            "open_host_service": "--> : <<OHS>>",
            "published_language": "--> : <<Published Language>>",
            "partnership": "<--> : <<Partnership>>",
            "big_ball_of_mud": "~~~ : <<BBoM>>",
        }
        return arrows.get(relationship_type, "-->")
