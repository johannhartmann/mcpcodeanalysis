"""Tests for pattern analyzer functionality."""

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.domain_models import (
    BoundedContext,
    BoundedContextMembership,
    DomainEntity,
    DomainRelationship,
)
from src.database.models import File, Repository
from src.domain.pattern_analyzer import DomainPatternAnalyzer


@pytest.fixture
async def sample_domain_data(async_session: AsyncSession) -> dict[str, Any]:
    """Create sample domain data for testing."""
    # Create repository
    repo = Repository(
        github_url="https://github.com/test/repo",
        owner="test",
        name="repo",
    )
    async_session.add(repo)
    await async_session.flush()

    # Create files
    file1 = File(
        repository_id=repo.id,
        path="src/orders/order.py",
        content_hash="hash1",
        language="python",
    )
    file2 = File(
        repository_id=repo.id,
        path="src/inventory/product.py",
        content_hash="hash2",
        language="python",
    )
    async_session.add_all([file1, file2])
    await async_session.flush()

    # Create bounded contexts
    sales_ctx = BoundedContext(
        name="Sales",
        description="Sales bounded context",
        cohesion_score=0.8,
        coupling_score=0.3,
    )
    inventory_ctx = BoundedContext(
        name="Inventory",
        description="Inventory bounded context",
        cohesion_score=0.7,
        coupling_score=0.4,
    )
    async_session.add_all([sales_ctx, inventory_ctx])
    await async_session.flush()

    # Create entities
    order_entity = DomainEntity(
        name="Order",
        entity_type="aggregate_root",
        description="Order aggregate",
        source_entities=[file1.id],
        business_rules=["Order must have items", "Total must be positive"],
        invariants=["items.count > 0", "total > 0"],
        responsibilities=["Manage order lifecycle", "Calculate totals"],
    )

    order_item_entity = DomainEntity(
        name="OrderItem",
        entity_type="entity",
        description="Order line item",
        source_entities=[file1.id],
        business_rules=[],
        invariants=[],
        responsibilities=["Track item quantity"],
    )

    product_entity = DomainEntity(
        name="Product",
        entity_type="aggregate_root",
        description="Product aggregate",
        source_entities=[file2.id],
        business_rules=["Stock must be non-negative"],
        invariants=["stock >= 0"],
        responsibilities=["Manage product info", "Track inventory"],
    )

    # Anemic entity for testing
    customer_entity = DomainEntity(
        name="Customer",
        entity_type="entity",
        description="Customer entity",
        source_entities=[file1.id],
        business_rules=[],  # No business rules - anemic
        invariants=[],  # No invariants - anemic
        responsibilities=["Store customer data"],
    )

    # God object for testing
    service_entity = DomainEntity(
        name="OrderService",
        entity_type="domain_service",
        description="Order service",
        source_entities=[file1.id],
        responsibilities=[
            "Create orders",
            "Update orders",
            "Cancel orders",
            "Process payments",
            "Send notifications",
            "Generate reports",
            "Handle refunds",
            "Validate addresses",
        ],  # Too many responsibilities
    )

    async_session.add_all(
        [
            order_entity,
            order_item_entity,
            product_entity,
            customer_entity,
            service_entity,
        ],
    )
    await async_session.flush()

    # Create context memberships
    async_session.add_all(
        [
            BoundedContextMembership(
                bounded_context_id=sales_ctx.id,
                domain_entity_id=order_entity.id,
            ),
            BoundedContextMembership(
                bounded_context_id=sales_ctx.id,
                domain_entity_id=order_item_entity.id,
            ),
            BoundedContextMembership(
                bounded_context_id=sales_ctx.id,
                domain_entity_id=customer_entity.id,
            ),
            BoundedContextMembership(
                bounded_context_id=sales_ctx.id,
                domain_entity_id=service_entity.id,
            ),
            BoundedContextMembership(
                bounded_context_id=inventory_ctx.id,
                domain_entity_id=product_entity.id,
            ),
        ],
    )

    # Create relationships
    # Order aggregates OrderItem
    rel1 = DomainRelationship(
        source_entity_id=order_entity.id,
        target_entity_id=order_item_entity.id,
        relationship_type="aggregates",
        description="Order contains items",
    )

    # Order depends on Product (cross-context)
    rel2 = DomainRelationship(
        source_entity_id=order_entity.id,
        target_entity_id=product_entity.id,
        relationship_type="depends_on",
        description="Order references products",
    )

    async_session.add_all([rel1, rel2])

    # Multiple relationships for high coupling (skip depends_on as rel2 already has it)
    relationship_types = ["uses", "references", "queries", "modifies", "validates"]
    for i, rel_type in enumerate(relationship_types):
        async_session.add(
            DomainRelationship(
                source_entity_id=order_entity.id,
                target_entity_id=product_entity.id,
                relationship_type=rel_type,
                description=f"Dependency {i}",
            ),
        )
    await async_session.commit()

    return {
        "repository": repo,
        "sales_context": sales_ctx,
        "inventory_context": inventory_ctx,
        "entities": {
            "order": order_entity,
            "order_item": order_item_entity,
            "product": product_entity,
            "customer": customer_entity,
            "service": service_entity,
        },
    }


@pytest.mark.asyncio
async def test_analyze_cross_context_coupling(
    async_session: AsyncSession,
    sample_domain_data: dict[str, Any],
) -> None:
    """Test cross-context coupling analysis."""
    analyzer = DomainPatternAnalyzer(async_session)

    result = await analyzer.analyze_cross_context_coupling(
        sample_domain_data["repository"].id,
    )

    assert "contexts" in result
    assert "high_coupling_pairs" in result
    assert "recommendations" in result
    assert "metrics" in result

    # Should find high coupling between Sales and Inventory
    assert len(result["high_coupling_pairs"]) > 0
    pair = result["high_coupling_pairs"][0]
    assert pair["source"] == "Sales"
    assert pair["target"] == "Inventory"
    assert pair["relationship_count"] > 5


@pytest.mark.asyncio
async def test_detect_anti_patterns(
    async_session: AsyncSession,
    sample_domain_data: dict[str, Any],
) -> None:
    """Test anti-pattern detection."""
    analyzer = DomainPatternAnalyzer(async_session)

    result = await analyzer.detect_anti_patterns(sample_domain_data["repository"].id)

    assert "anemic_domain_models" in result
    assert "god_objects" in result
    assert "circular_dependencies" in result
    assert "missing_aggregate_roots" in result

    # Should find anemic Customer entity
    anemic = result["anemic_domain_models"]
    assert len(anemic) > 0
    assert any(p["entity"] == "Customer" for p in anemic)

    # Should find god object OrderService
    god_objects = result["god_objects"]
    assert len(god_objects) > 0
    assert any(p["entity"] == "OrderService" for p in god_objects)


@pytest.mark.asyncio
async def test_suggest_context_splits(
    async_session: AsyncSession,
    sample_domain_data: dict[str, Any],
) -> None:
    """Test context split suggestions."""
    # First, create a large context with low cohesion
    large_ctx = BoundedContext(
        name="LargeContext",
        description="Large context with many entities",
        cohesion_score=0.3,  # Low cohesion
    )
    async_session.add(large_ctx)
    await async_session.flush()

    # Add many entities to it
    for i in range(25):
        entity = DomainEntity(
            name=f"Entity{i}",
            entity_type="entity",
            description=f"Entity {i}",
            source_entities=[],
        )
        async_session.add(entity)
        await async_session.flush()

        async_session.add(
            BoundedContextMembership(
                bounded_context_id=large_ctx.id,
                domain_entity_id=entity.id,
            ),
        )

    await async_session.commit()

    analyzer = DomainPatternAnalyzer(async_session)

    result = await analyzer.suggest_context_splits(
        min_entities=20,
        max_cohesion_threshold=0.4,
    )

    assert len(result) > 0
    suggestion = result[0]
    assert suggestion["context"] == "LargeContext"
    assert suggestion["current_size"] >= 20
    assert "suggested_splits" in suggestion


@pytest.mark.asyncio
async def test_analyze_evolution(
    async_session: AsyncSession, sample_domain_data: dict[str, Any]
) -> None:
    """Test domain evolution analysis."""
    analyzer = DomainPatternAnalyzer(async_session)

    # Create some recent entities
    recent_entity = DomainEntity(
        name="RecentEntity",
        entity_type="entity",
        description="Recently added entity",
        source_entities=[],
        created_at=datetime.now(UTC) - timedelta(days=5),
    )
    async_session.add(recent_entity)
    await async_session.commit()

    result = await analyzer.analyze_evolution(
        sample_domain_data["repository"].id,
        days=30,
    )

    assert "entity_changes" in result
    assert "context_changes" in result
    assert "trends" in result
    assert "insights" in result

    # Should include the recent entity
    assert len(result["entity_changes"]["added"]) > 0
