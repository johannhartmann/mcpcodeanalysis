"""Tests for context relationship saving functionality."""

from unittest.mock import MagicMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.domain_models import (
    BoundedContext,
    BoundedContextMembership,
    ContextRelationship,
    DomainEntity,
    DomainRelationship,
)
from src.domain.graph_builder import SemanticGraphBuilder
from src.domain.indexer import DomainIndexer


@pytest.mark.asyncio
async def test_save_context_relationships(async_session: AsyncSession):
    """Test saving context relationships."""
    # Create test domain entities
    entity1 = DomainEntity(
        name="OrderService",
        entity_type="domain_service",
        description="Service for managing orders",
        confidence_score=0.9,
    )
    entity2 = DomainEntity(
        name="PaymentService",
        entity_type="domain_service",
        description="Service for processing payments",
        confidence_score=0.9,
    )
    entity3 = DomainEntity(
        name="InventoryService",
        entity_type="domain_service",
        description="Service for managing inventory",
        confidence_score=0.9,
    )
    entity4 = DomainEntity(
        name="ShippingService",
        entity_type="domain_service",
        description="Service for shipping orders",
        confidence_score=0.9,
    )

    async_session.add_all([entity1, entity2, entity3, entity4])
    await async_session.flush()

    # Create relationships between entities
    rel1 = DomainRelationship(
        source_entity_id=entity1.id,
        target_entity_id=entity2.id,
        relationship_type="orchestrates",
        strength=0.8,
        confidence_score=0.9,
    )
    rel2 = DomainRelationship(
        source_entity_id=entity3.id,
        target_entity_id=entity4.id,
        relationship_type="publishes",
        strength=0.7,
        confidence_score=0.8,
    )

    async_session.add_all([rel1, rel2])
    await async_session.flush()

    # Create bounded contexts
    context1 = BoundedContext(
        name="Order Management",
        description="Context for managing orders and payments",
        cohesion_score=0.8,
        coupling_score=0.3,
        modularity_score=0.7,
    )
    context2 = BoundedContext(
        name="Fulfillment",
        description="Context for inventory and shipping",
        cohesion_score=0.75,
        coupling_score=0.4,
        modularity_score=0.65,
    )

    async_session.add_all([context1, context2])
    await async_session.flush()

    # Create memberships
    membership1 = BoundedContextMembership(
        domain_entity_id=entity1.id,
        bounded_context_id=context1.id,
    )
    membership2 = BoundedContextMembership(
        domain_entity_id=entity2.id,
        bounded_context_id=context1.id,
    )
    membership3 = BoundedContextMembership(
        domain_entity_id=entity3.id,
        bounded_context_id=context2.id,
    )
    membership4 = BoundedContextMembership(
        domain_entity_id=entity4.id,
        bounded_context_id=context2.id,
    )

    async_session.add_all([membership1, membership2, membership3, membership4])
    await async_session.commit()

    # Initialize graph builder with mocked OpenAI dependencies
    mock_embeddings = MagicMock()
    mock_llm = MagicMock()

    graph_builder = SemanticGraphBuilder(async_session, embeddings=mock_embeddings, llm=mock_llm)

    # Prepare relationship data
    relationships = [
        {
            "source_context_id": 0,  # Will be mapped to context1.id
            "target_context_id": 1,  # Will be mapped to context2.id
            "relationship_type": "customer_supplier",
            "strength": 0.75,
            "interaction_count": 2,
            "interaction_types": ["orchestrates", "publishes"],
        }
    ]

    # Save context relationships
    saved_relationships = await graph_builder.save_context_relationships(
        relationships, [context1, context2]
    )

    # Verify relationships were saved
    assert len(saved_relationships) == 1

    # Query saved relationship
    result = await async_session.execute(
        select(ContextRelationship).where(
            ContextRelationship.source_context_id == context1.id,
            ContextRelationship.target_context_id == context2.id,
        )
    )
    saved_rel = result.scalar_one()

    assert saved_rel.relationship_type == "customer_supplier"
    assert (
        saved_rel.description
        == "Customer-supplier relationship where one context drives the other"
    )
    assert saved_rel.interface_description["strength"] == 0.75
    assert saved_rel.interface_description["interaction_count"] == 2
    assert "orchestrates" in saved_rel.interface_description["interaction_types"]
    assert "publishes" in saved_rel.interface_description["interaction_types"]


@pytest.mark.asyncio
async def test_analyze_and_save_context_relationships(async_session: AsyncSession):
    """Test full flow of analyzing and saving context relationships."""
    # Create test domain entities
    entities = []
    for i in range(6):
        entity = DomainEntity(
            name=f"Entity{i}",
            entity_type="entity",
            description=f"Test entity {i}",
            confidence_score=0.9,
        )
        entities.append(entity)
        async_session.add(entity)

    await async_session.flush()

    # Create relationships to form two contexts
    # Context 1: entities 0, 1, 2 (densely connected)
    # Context 2: entities 3, 4, 5 (densely connected)
    # Cross-context: 2 -> 3 (publishes)

    relationships = [
        # Context 1 internal
        (0, 1, "uses"),
        (1, 2, "modifies"),
        (0, 2, "references"),
        # Context 2 internal
        (3, 4, "uses"),
        (4, 5, "modifies"),
        (3, 5, "references"),
        # Cross-context
        (2, 3, "publishes"),
    ]

    for source_idx, target_idx, rel_type in relationships:
        rel = DomainRelationship(
            source_entity_id=entities[source_idx].id,
            target_entity_id=entities[target_idx].id,
            relationship_type=rel_type,
            strength=0.8,
            confidence_score=0.9,
        )
        async_session.add(rel)

    await async_session.commit()

    # Initialize indexer with mocked OpenAI dependencies
    mock_embeddings = MagicMock()
    mock_llm = MagicMock()

    indexer = DomainIndexer(async_session, embeddings=mock_embeddings, llm=mock_llm)

    # Build graph
    graph = await indexer.graph_builder.build_graph()

    # Detect contexts (should find 2)
    contexts = await indexer.graph_builder.detect_bounded_contexts(
        graph, use_embeddings=False
    )
    assert len(contexts) >= 2

    # Save contexts
    saved_contexts = await indexer.graph_builder.save_bounded_contexts(contexts)

    # Analyze relationships
    context_relationships = (
        await indexer.graph_builder.analyze_context_relationships(graph, contexts)
    )

    # Should find at least one cross-context relationship
    assert len(context_relationships) >= 1

    # Save relationships
    saved_rels = await indexer.graph_builder.save_context_relationships(
        context_relationships, saved_contexts
    )

    assert len(saved_rels) >= 1

    # Verify in database
    result = await async_session.execute(select(ContextRelationship))
    all_context_rels = result.scalars().all()

    assert len(all_context_rels) >= 1

    # Check that relationship type was determined correctly
    # Since we have a "publishes" relationship, it should be "published_language"
    published_lang_rels = [
        r for r in all_context_rels if r.relationship_type == "published_language"
    ]
    assert len(published_lang_rels) >= 1


@pytest.mark.asyncio
async def test_duplicate_context_relationships_not_saved(async_session: AsyncSession):
    """Test that duplicate context relationships are not saved."""
    # Create bounded contexts
    context1 = BoundedContext(
        name="Context A",
        description="First context",
    )
    context2 = BoundedContext(
        name="Context B",
        description="Second context",
    )

    async_session.add_all([context1, context2])
    await async_session.flush()

    # Create existing relationship
    existing_rel = ContextRelationship(
        source_context_id=context1.id,
        target_context_id=context2.id,
        relationship_type="partnership",
        description="Existing partnership",
    )
    async_session.add(existing_rel)
    await async_session.commit()

    # Initialize graph builder with mocked OpenAI dependencies
    mock_embeddings = MagicMock()
    mock_llm = MagicMock()

    graph_builder = SemanticGraphBuilder(async_session, embeddings=mock_embeddings, llm=mock_llm)

    # Try to save duplicate relationship
    relationships = [
        {
            "source_context_id": 0,
            "target_context_id": 1,
            "relationship_type": "partnership",
            "strength": 0.9,
            "interaction_count": 5,
            "interaction_types": ["uses", "modifies"],
        }
    ]

    # Should not save duplicate
    saved_relationships = await graph_builder.save_context_relationships(
        relationships, [context1, context2]
    )

    assert len(saved_relationships) == 0

    # Verify only one relationship exists
    result = await async_session.execute(
        select(ContextRelationship).where(
            ContextRelationship.source_context_id == context1.id,
            ContextRelationship.target_context_id == context2.id,
            ContextRelationship.relationship_type == "partnership",
        )
    )
    all_rels = result.scalars().all()
    assert len(all_rels) == 1
    assert (
        all_rels[0].description == "Existing partnership"
    )  # Original not overwritten
