"""Domain-driven design models for semantic code analysis."""

from __future__ import annotations

from datetime import datetime
from typing import Any

try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    # For SQLite tests, use JSON instead of Vector
    def Vector(_dim: int) -> type[JSON]:  # noqa: N802
        return JSON


from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)

try:
    pass
except ImportError:
    # For SQLite, we'll use JSON instead
    def ARRAY(_item_type: Any) -> Any:  # noqa: N802
        return JSON


from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.models import Base


class DomainEntity(Base):
    """Domain entities extracted by LLM from code."""

    __tablename__ = "domain_entities"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    entity_type: Mapped[str] = mapped_column(
        Enum(
            "aggregate_root",
            "entity",
            "value_object",
            "domain_service",
            "domain_event",
            "command",
            "query",
            "policy",
            "factory",
            "repository_interface",
            name="domain_entity_type",
        ),
        nullable=False,
    )
    description: Mapped[str | None] = mapped_column(Text)
    business_rules: Mapped[list[str]] = mapped_column(
        JSON, default=list
    )  # List of business rules
    invariants: Mapped[list[str]] = mapped_column(
        JSON, default=list
    )  # List of invariants to maintain
    responsibilities: Mapped[list[str]] = mapped_column(
        JSON, default=list
    )  # What this entity is responsible for
    ubiquitous_language: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict
    )  # Domain terms used

    # Tracking which code entities implement this domain entity
    # Use ARRAY for PostgreSQL, JSON for SQLite
    source_entities: Mapped[list[int]] = mapped_column(
        JSON, default=list
    )  # IDs from code_entities
    confidence_score: Mapped[float] = mapped_column(
        Float, default=1.0
    )  # LLM confidence in extraction

    # Embeddings for semantic search
    concept_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(1536)
    )  # Semantic embedding of the concept

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )
    extraction_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict
    )  # LLM model, prompts used, etc.

    # Relationships
    source_relationships: Mapped[list[DomainRelationship]] = relationship(
        "DomainRelationship",
        foreign_keys="DomainRelationship.source_entity_id",
        back_populates="source_entity",
        cascade="all, delete-orphan",
    )
    target_relationships: Mapped[list[DomainRelationship]] = relationship(
        "DomainRelationship",
        foreign_keys="DomainRelationship.target_entity_id",
        back_populates="target_entity",
        cascade="all, delete-orphan",
    )
    bounded_context_memberships: Mapped[list[BoundedContextMembership]] = relationship(
        "BoundedContextMembership",
        back_populates="domain_entity",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("idx_domain_entity_name", "name"),
        Index("idx_domain_entity_type", "entity_type"),
        # GIN indexes are PostgreSQL-specific, skip for SQLite
    )


class DomainRelationship(Base):
    """Semantic relationships between domain entities."""

    __tablename__ = "domain_relationships"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_entity_id: Mapped[int] = mapped_column(
        ForeignKey("domain_entities.id"), nullable=False
    )
    target_entity_id: Mapped[int] = mapped_column(
        ForeignKey("domain_entities.id"), nullable=False
    )

    relationship_type: Mapped[str] = mapped_column(
        Enum(
            "uses",
            "creates",
            "modifies",
            "deletes",
            "queries",
            "validates",
            "orchestrates",
            "implements",
            "extends",
            "aggregates",
            "references",
            "publishes",
            "subscribes_to",
            "depends_on",
            "composed_of",
            name="domain_relationship_type",
        ),
        nullable=False,
    )

    description: Mapped[str | None] = mapped_column(Text)
    strength: Mapped[float] = mapped_column(
        Float, default=1.0
    )  # Relationship strength (0-1)
    confidence_score: Mapped[float] = mapped_column(
        Float, default=1.0
    )  # LLM confidence

    # Evidence from code supporting this relationship
    evidence: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON,
        default=list,
    )  # List of {file_path, line_number, code_snippet}

    # Additional semantic information
    interaction_patterns: Mapped[list[str]] = mapped_column(
        JSON, default=list
    )  # How they interact
    data_flow: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict
    )  # What data flows between them

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )

    # Relationships
    source_entity: Mapped[DomainEntity] = relationship(
        "DomainEntity",
        foreign_keys=[source_entity_id],
        back_populates="source_relationships",
    )
    target_entity: Mapped[DomainEntity] = relationship(
        "DomainEntity",
        foreign_keys=[target_entity_id],
        back_populates="target_relationships",
    )

    __table_args__ = (
        Index("idx_domain_rel_source", "source_entity_id"),
        Index("idx_domain_rel_target", "target_entity_id"),
        Index("idx_domain_rel_type", "relationship_type"),
        UniqueConstraint(
            "source_entity_id",
            "target_entity_id",
            "relationship_type",
            name="uq_domain_relationship",
        ),
    )


class BoundedContext(Base):
    """Bounded contexts discovered through semantic analysis."""

    __tablename__ = "bounded_contexts"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text)

    # Domain language and concepts
    ubiquitous_language: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict
    )  # Term -> Definition mapping
    core_concepts: Mapped[list[str]] = mapped_column(
        JSON, default=list
    )  # Main concepts in this context

    # Boundaries and interfaces
    published_language: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict
    )  # Public API/events
    anti_corruption_layer: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict
    )  # External term mappings

    # Context metadata
    context_type: Mapped[str] = mapped_column(
        Enum(
            "core",
            "supporting",
            "generic",
            "external",
            name="bounded_context_type",
        ),
        default="supporting",
    )

    # Summarization
    summary: Mapped[str | None] = mapped_column(Text)  # LLM-generated summary
    responsibilities: Mapped[list[str]] = mapped_column(
        JSON, default=list
    )  # What this context handles

    # Graph analysis metadata
    cohesion_score: Mapped[float | None] = mapped_column(
        Float
    )  # Internal cohesion (0-1)
    coupling_score: Mapped[float | None] = mapped_column(
        Float
    )  # External coupling (0-1)
    modularity_score: Mapped[float | None] = mapped_column(
        Float
    )  # From community detection

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )

    # Relationships
    memberships: Mapped[list[BoundedContextMembership]] = relationship(
        "BoundedContextMembership",
        back_populates="bounded_context",
        cascade="all, delete-orphan",
    )
    context_relationships: Mapped[list[ContextRelationship]] = relationship(
        "ContextRelationship",
        foreign_keys="ContextRelationship.source_context_id",
        back_populates="source_context",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("idx_bounded_context_name", "name"),
        Index("idx_bounded_context_type", "context_type"),
    )


class BoundedContextMembership(Base):
    """Many-to-many relationship between entities and contexts."""

    __tablename__ = "bounded_context_memberships"

    id: Mapped[int] = mapped_column(primary_key=True)
    domain_entity_id: Mapped[int] = mapped_column(
        ForeignKey("domain_entities.id"), nullable=False
    )
    bounded_context_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("bounded_contexts.id"),
        nullable=False,
    )

    # Role of entity within context
    role: Mapped[str | None] = mapped_column(
        String(100)
    )  # e.g., "aggregate_root", "service", etc.
    importance_score: Mapped[float] = mapped_column(
        Float, default=1.0
    )  # How central to the context

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    # Relationships
    domain_entity: Mapped[DomainEntity] = relationship(
        "DomainEntity",
        back_populates="bounded_context_memberships",
    )
    bounded_context: Mapped[BoundedContext] = relationship(
        "BoundedContext", back_populates="memberships"
    )

    __table_args__ = (
        UniqueConstraint(
            "domain_entity_id",
            "bounded_context_id",
            name="uq_context_membership",
        ),
        Index("idx_membership_entity", "domain_entity_id"),
        Index("idx_membership_context", "bounded_context_id"),
    )


class ContextRelationship(Base):
    """Relationships between bounded contexts."""

    __tablename__ = "context_relationships"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_context_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("bounded_contexts.id"),
        nullable=False,
    )
    target_context_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("bounded_contexts.id"),
        nullable=False,
    )

    relationship_type: Mapped[str] = mapped_column(
        Enum(
            "shared_kernel",
            "customer_supplier",
            "conformist",
            "anti_corruption_layer",
            "open_host_service",
            "published_language",
            "partnership",
            "big_ball_of_mud",
            name="context_relationship_type",
        ),
        nullable=False,
    )

    description: Mapped[str | None] = mapped_column(Text)
    interface_description: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict
    )  # How contexts integrate

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )

    # Relationships
    source_context: Mapped[BoundedContext] = relationship(
        "BoundedContext",
        foreign_keys=[source_context_id],
        back_populates="context_relationships",
    )
    target_context: Mapped[BoundedContext] = relationship(
        "BoundedContext",
        foreign_keys=[target_context_id],
    )

    __table_args__ = (
        Index("idx_context_rel_source", "source_context_id"),
        Index("idx_context_rel_target", "target_context_id"),
        UniqueConstraint(
            "source_context_id",
            "target_context_id",
            "relationship_type",
            name="uq_context_relationship",
        ),
    )


class DomainSummary(Base):
    """Hierarchical summaries of code at different levels."""

    __tablename__ = "domain_summaries"

    id: Mapped[int] = mapped_column(primary_key=True)
    level: Mapped[str] = mapped_column(
        Enum("function", "class", "module", "package", "context", name="summary_level"),
        nullable=False,
    )

    # Reference to original code entity
    entity_type: Mapped[str | None] = mapped_column(
        String(50)
    )  # e.g., "function", "class", etc.
    entity_id: Mapped[int | None] = mapped_column(Integer)  # ID in respective table

    # LLM-generated summaries
    business_summary: Mapped[str | None] = mapped_column(
        Text
    )  # What it does in business terms
    technical_summary: Mapped[str | None] = mapped_column(
        Text
    )  # Technical implementation details
    domain_concepts: Mapped[list[str]] = mapped_column(
        JSON, default=list
    )  # Extracted concepts

    # Hierarchical reference
    parent_summary_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("domain_summaries.id")
    )

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )

    # Relationships
    parent_summary: Mapped[DomainSummary | None] = relationship(
        "DomainSummary", remote_side=[id]
    )

    __table_args__ = (
        Index("idx_summary_level", "level"),
        Index("idx_summary_entity", "entity_type", "entity_id"),
        Index("idx_summary_parent", "parent_summary_id"),
    )
