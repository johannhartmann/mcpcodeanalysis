"""Domain-driven design models for semantic code analysis."""

from __future__ import annotations

from typing import Any

try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    # For SQLite tests, use JSON instead of Vector
    def Vector(dim: int) -> type[JSON]:  # noqa: ARG001, N802
        _ = dim  # Mark as intentionally unused
        return JSON


from sqlalchemy import (
    JSON,
    Column,
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
    def ARRAY(item_type: Any) -> Any:  # noqa: ARG001, N802
        _ = item_type  # Mark as intentionally unused
        return JSON


from sqlalchemy.orm import relationship

from src.database.models import Base


class DomainEntity(Base):
    """Domain entities extracted by LLM from code."""

    __tablename__ = "domain_entities"
    __allow_unmapped__ = True

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    entity_type = Column(
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
    description = Column(Text)
    business_rules = Column(JSON, default=[])  # List of business rules
    invariants = Column(JSON, default=[])  # List of invariants to maintain
    responsibilities = Column(JSON, default=[])  # What this entity is responsible for
    ubiquitous_language = Column(JSON, default={})  # Domain terms used

    # Tracking which code entities implement this domain entity
    # Use ARRAY for PostgreSQL, JSON for SQLite
    source_entities = Column(JSON, default=[])  # IDs from code_entities
    confidence_score = Column(Float, default=1.0)  # LLM confidence in extraction

    # Embeddings for semantic search
    concept_embedding = Column(Vector(1536))  # Semantic embedding of the concept

    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    extraction_metadata = Column(JSON, default={})  # LLM model, prompts used, etc.

    # Relationships
    source_relationships: Any = relationship(
        "DomainRelationship",
        foreign_keys="DomainRelationship.source_entity_id",
        back_populates="source_entity",
        cascade="all, delete-orphan",
    )
    target_relationships: Any = relationship(
        "DomainRelationship",
        foreign_keys="DomainRelationship.target_entity_id",
        back_populates="target_entity",
        cascade="all, delete-orphan",
    )
    bounded_context_memberships: Any = relationship(
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
    __allow_unmapped__ = True

    id = Column(Integer, primary_key=True)
    source_entity_id = Column(Integer, ForeignKey("domain_entities.id"), nullable=False)
    target_entity_id = Column(Integer, ForeignKey("domain_entities.id"), nullable=False)

    relationship_type = Column(
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

    description = Column(Text)
    strength = Column(Float, default=1.0)  # Relationship strength (0-1)
    confidence_score = Column(Float, default=1.0)  # LLM confidence

    # Evidence from code supporting this relationship
    evidence = Column(
        JSON,
        default=[],
    )  # List of {file_path, line_number, code_snippet}

    # Additional semantic information
    interaction_patterns = Column(JSON, default=[])  # How they interact
    data_flow = Column(JSON, default={})  # What data flows between them

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    source_entity: Any = relationship(
        "DomainEntity",
        foreign_keys=[source_entity_id],
        back_populates="source_relationships",
    )
    target_entity: Any = relationship(
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
    __allow_unmapped__ = True

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text)

    # Domain language and concepts
    ubiquitous_language = Column(JSON, default={})  # Term -> Definition mapping
    core_concepts = Column(JSON, default=[])  # Main concepts in this context

    # Boundaries and interfaces
    published_language = Column(JSON, default={})  # Public API/events
    anti_corruption_layer = Column(JSON, default={})  # External term mappings

    # Context metadata
    context_type = Column(
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
    summary = Column(Text)  # LLM-generated summary
    responsibilities = Column(JSON, default=[])  # What this context handles

    # Graph analysis metadata
    cohesion_score = Column(Float)  # Internal cohesion (0-1)
    coupling_score = Column(Float)  # External coupling (0-1)
    modularity_score = Column(Float)  # From community detection

    # Migration-specific fields
    migration_priority = Column(
        Enum("high", "medium", "low", name="migration_priority_type"),
        default="medium",
    )
    migration_complexity = Column(Float)  # Calculated complexity score
    migration_readiness = Column(Float)  # Readiness assessment (0-1)
    extraction_strategy = Column(String(100))  # Preferred extraction approach
    migration_notes = Column(Text)  # Additional migration considerations

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    memberships: Any = relationship(
        "BoundedContextMembership",
        back_populates="bounded_context",
        cascade="all, delete-orphan",
    )
    context_relationships: Any = relationship(
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
    __allow_unmapped__ = True

    id = Column(Integer, primary_key=True)
    domain_entity_id = Column(Integer, ForeignKey("domain_entities.id"), nullable=False)
    bounded_context_id = Column(
        Integer,
        ForeignKey("bounded_contexts.id"),
        nullable=False,
    )

    # Role of entity within context
    role = Column(String(100))  # e.g., "aggregate_root", "service", etc.
    importance_score = Column(Float, default=1.0)  # How central to the context

    created_at = Column(DateTime, default=func.now())

    # Relationships
    domain_entity: Any = relationship(
        "DomainEntity",
        back_populates="bounded_context_memberships",
    )
    bounded_context: Any = relationship("BoundedContext", back_populates="memberships")

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
    __allow_unmapped__ = True

    id = Column(Integer, primary_key=True)
    source_context_id = Column(
        Integer,
        ForeignKey("bounded_contexts.id"),
        nullable=False,
    )
    target_context_id = Column(
        Integer,
        ForeignKey("bounded_contexts.id"),
        nullable=False,
    )

    relationship_type = Column(
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

    description = Column(Text)
    interface_description = Column(JSON, default={})  # How contexts integrate

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    source_context: Any = relationship(
        "BoundedContext",
        foreign_keys=[source_context_id],
        back_populates="context_relationships",
    )
    target_context: Any = relationship(
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
    __allow_unmapped__ = True

    id = Column(Integer, primary_key=True)
    level = Column(
        Enum("function", "class", "module", "package", "context", name="summary_level"),
        nullable=False,
    )

    # Reference to original code entity
    entity_type = Column(String(50))  # e.g., "function", "class", etc.
    entity_id = Column(Integer)  # ID in respective table

    # LLM-generated summaries
    business_summary = Column(Text)  # What it does in business terms
    technical_summary = Column(Text)  # Technical implementation details
    domain_concepts = Column(JSON, default=[])  # Extracted concepts

    # Hierarchical reference
    parent_summary_id = Column(Integer, ForeignKey("domain_summaries.id"))

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    parent_summary: Any = relationship("DomainSummary", remote_side=[id])

    __table_args__ = (
        Index("idx_summary_level", "level"),
        Index("idx_summary_entity", "entity_type", "entity_id"),
        Index("idx_summary_parent", "parent_summary_id"),
    )
