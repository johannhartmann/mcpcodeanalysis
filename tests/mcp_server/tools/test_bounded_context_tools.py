"""Tests for bounded context analysis tools."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastmcp import FastMCP
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.domain_models import (
    BoundedContext,
    BoundedContextMembership,
    ContextRelationship,
    DomainEntity,
)
from src.mcp_server.tools.domain_tools import DomainTools


@pytest.fixture
def mock_db_session():
    """Create mock database session."""
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
def mock_mcp():
    """Create mock FastMCP instance."""
    mcp = MagicMock(spec=FastMCP)
    mcp.tool = MagicMock(side_effect=lambda **kwargs: lambda func: func)
    return mcp


@pytest.fixture
def domain_tools(mock_db_session, mock_mcp):
    """Create domain tools fixture for bounded context tests."""
    return DomainTools(mock_db_session, mock_mcp)


class TestBoundedContextTools:
    """Tests specifically for bounded context related tools."""

    @pytest.mark.asyncio
    async def test_analyze_bounded_context_with_relationships(
        self, domain_tools, mock_db_session
    ):
        """Test analyzing bounded context with complex relationships."""
        # Mock bounded context
        mock_context = MagicMock(spec=BoundedContext)
        mock_context.id = 1
        mock_context.name = "OrderContext"
        mock_context.description = "Order management bounded context"
        mock_context.ubiquitous_language = [
            "Order",
            "OrderItem",
            "Customer",
            "Payment",
            "Shipping",
        ]
        mock_context.core_concepts = [
            "Order fulfillment",
            "Payment processing",
            "Inventory management",
        ]
        mock_context.cohesion_score = 0.85
        mock_context.coupling_score = 0.3
        mock_context.modularity_score = 0.75

        # Mock memberships with various entity types
        memberships = []
        for i, (entity_type, count) in enumerate(
            [
                ("aggregate_root", 2),
                ("entity", 5),
                ("value_object", 3),
                ("domain_service", 1),
            ]
        ):
            for j in range(count):
                membership = MagicMock(spec=BoundedContextMembership)
                membership.domain_entity_id = i * 10 + j
                memberships.append(membership)

        mock_context.memberships = memberships

        context_result = MagicMock()
        context_result.scalar_one_or_none.return_value = mock_context

        # Mock entities with different types
        entities = []
        entity_types = [
            ("aggregate_root", "Order", 2),
            ("entity", "OrderItem", 5),
            ("value_object", "Money", 3),
            ("domain_service", "PricingService", 1),
        ]

        for entity_type, base_name, count in entity_types:
            for i in range(count):
                entity = MagicMock(spec=DomainEntity)
                entity.name = f"{base_name}{i}" if count > 1 else base_name
                entity.entity_type = entity_type
                entity.description = f"{entity_type} for {base_name}"
                entities.append(entity)

        entity_result = MagicMock()
        entity_result.scalars.return_value.all.return_value = entities

        # Mock internal relationships
        internal_rels = []
        for i in range(3):
            rel = MagicMock()
            rel.source_entity = entities[i]
            rel.target_entity = entities[i + 1]
            rel.relationship_type = ["aggregates", "uses", "depends_on"][i % 3]
            internal_rels.append(rel)

        rel_result = MagicMock()
        rel_result.scalars.return_value.all.return_value = internal_rels

        # Mock external relationships
        external_entity = MagicMock(spec=DomainEntity)
        external_entity.name = "ExternalPaymentGateway"

        external_rels = []
        for i in range(2):
            rel = MagicMock()
            rel.source_entity = entities[i]
            rel.target_entity = external_entity
            rel.relationship_type = "integrates_with"
            external_rels.append(rel)

        ext_rel_result = MagicMock()
        ext_rel_result.scalars.return_value.all.return_value = external_rels

        # No summary for this test
        summary_result = MagicMock()
        summary_result.scalar_one_or_none.return_value = None

        mock_db_session.execute.side_effect = [
            context_result,
            entity_result,
            rel_result,
            ext_rel_result,
            summary_result,
        ]

        result = await domain_tools.analyze_bounded_context("OrderContext")

        assert result["name"] == "OrderContext"
        assert result["total_entities"] == 11
        assert result["entities_by_type"]["aggregate_root"][0]["name"] == "Order"
        assert len(result["entities_by_type"]["entity"]) == 5
        assert len(result["entities_by_type"]["value_object"]) == 3
        assert result["internal_relationships"] == 3
        assert result["external_dependencies"] == 2
        assert result["cohesion_score"] == 0.85
        assert result["summary"] is None

    @pytest.mark.asyncio
    async def test_find_bounded_contexts_with_filtering(
        self, domain_tools, mock_db_session
    ):
        """Test finding bounded contexts with entity count filtering."""
        # Create contexts with varying entity counts
        contexts = []
        for i, (name, entity_count, cohesion, ctx_type) in enumerate(
            [
                ("UserContext", 10, 0.95, "core"),
                ("NotificationContext", 2, 0.7, "supporting"),  # Below threshold
                ("PaymentContext", 5, 0.85, "core"),
                ("ReportingContext", 1, 0.6, "generic"),  # Below threshold
                ("InventoryContext", 7, 0.8, "supporting"),
            ]
        ):
            context = MagicMock(spec=BoundedContext)
            context.name = name
            context.description = f"{name} description"
            context.core_concepts = [f"Concept{j}" for j in range(min(5, entity_count))]
            context.cohesion_score = cohesion
            context.context_type = ctx_type
            context.memberships = [MagicMock() for _ in range(entity_count)]
            contexts.append(context)

        context_result = MagicMock()
        context_result.scalars.return_value.all.return_value = contexts

        mock_db_session.execute.return_value = context_result

        # Test with default min_entities (3)
        result = await domain_tools.find_bounded_contexts()

        assert len(result) == 3  # Only contexts with >= 3 entities
        assert result[0]["name"] == "UserContext"  # Highest cohesion
        assert result[0]["entity_count"] == 10
        assert result[0]["type"] == "core"
        assert result[1]["name"] == "PaymentContext"
        assert result[2]["name"] == "InventoryContext"

        # Test with higher threshold
        result = await domain_tools.find_bounded_contexts(min_entities=6)

        assert len(result) == 2  # Only UserContext and InventoryContext

    @pytest.mark.asyncio
    async def test_generate_context_map_with_all_relationship_types(
        self, domain_tools, mock_db_session
    ):
        """Test generating context map with all relationship types."""
        # Create contexts
        contexts = []
        for i, (name, ctx_type) in enumerate(
            [
                ("OrderContext", "core"),
                ("InventoryContext", "supporting"),
                ("PaymentContext", "core"),
                ("ShippingContext", "supporting"),
                ("LegacyContext", "generic"),
            ]
        ):
            ctx = MagicMock(spec=BoundedContext)
            ctx.id = i
            ctx.name = name
            ctx.context_type = ctx_type
            ctx.description = f"{name} description"
            contexts.append(ctx)

        context_result = MagicMock()
        context_result.scalars.return_value.all.return_value = contexts

        # Create various relationship types
        relationship_types = [
            ("shared_kernel", contexts[0], contexts[1]),
            ("customer_supplier", contexts[0], contexts[2]),
            ("conformist", contexts[1], contexts[0]),
            ("anti_corruption_layer", contexts[0], contexts[4]),
            ("open_host_service", contexts[2], contexts[3]),
            ("published_language", contexts[2], contexts[1]),
            ("partnership", contexts[1], contexts[3]),
            ("big_ball_of_mud", contexts[4], contexts[3]),
        ]

        relationships = []
        for rel_type, source, target in relationship_types:
            rel = MagicMock(spec=ContextRelationship)
            rel.source_context = source
            rel.target_context = target
            rel.relationship_type = rel_type
            rel.description = f"{rel_type} between {source.name} and {target.name}"
            relationships.append(rel)

        rel_result = MagicMock()
        rel_result.scalars.return_value.all.return_value = relationships

        mock_db_session.execute.side_effect = [context_result, rel_result]

        # Test JSON format
        result = await domain_tools.generate_context_map("json")

        assert len(result["contexts"]) == 5
        assert len(result["relationships"]) == 8

        # Verify all relationship types are present
        rel_types = {rel["type"] for rel in result["relationships"]}
        assert rel_types == {
            "shared_kernel",
            "customer_supplier",
            "conformist",
            "anti_corruption_layer",
            "open_host_service",
            "published_language",
            "partnership",
            "big_ball_of_mud",
        }

    @pytest.mark.asyncio
    async def test_generate_context_map_plantuml_format(
        self, domain_tools, mock_db_session
    ):
        """Test generating context map in PlantUML format."""
        # Create contexts
        ctx1 = MagicMock(spec=BoundedContext)
        ctx1.name = "CoreDomain"
        ctx1.context_type = "core"

        ctx2 = MagicMock(spec=BoundedContext)
        ctx2.name = "SupportingDomain"
        ctx2.context_type = "supporting"

        contexts = [ctx1, ctx2]
        context_result = MagicMock()
        context_result.scalars.return_value.all.return_value = contexts

        # Create relationship
        rel = MagicMock(spec=ContextRelationship)
        rel.source_context = ctx1
        rel.target_context = ctx2
        rel.relationship_type = "customer_supplier"

        rel_result = MagicMock()
        rel_result.scalars.return_value.all.return_value = [rel]

        mock_db_session.execute.side_effect = [context_result, rel_result]

        result = await domain_tools.generate_context_map("plantuml")

        assert "diagram" in result
        diagram = result["diagram"]

        # Check PlantUML structure
        assert "@startuml" in diagram
        assert "@enduml" in diagram
        assert 'package "CoreDomain" <<Core>>' in diagram
        assert 'package "SupportingDomain"' in diagram
        assert '"CoreDomain" --> : <<Customer/Supplier>> "SupportingDomain"' in diagram

    @pytest.mark.asyncio
    async def test_suggest_ddd_refactoring_context_boundary_violation(
        self, domain_tools, mock_db_session
    ):
        """Test DDD refactoring suggestion for context boundary violations."""
        # Mock file
        mock_file = MagicMock()
        mock_file.id = 1

        file_result = MagicMock()
        file_result.scalar_one_or_none.return_value = mock_file

        # Mock entities from different contexts
        entities = []
        for i, (name, ctx_name) in enumerate(
            [
                ("Order", "OrderContext"),
                ("OrderItem", "OrderContext"),
                ("Customer", "CustomerContext"),
                ("Product", "ProductContext"),
            ]
        ):
            entity = MagicMock(spec=DomainEntity)
            entity.id = i
            entity.name = name
            entity.entity_type = "entity"
            entity.business_rules = ["Rule"]
            entity.invariants = ["Invariant"]
            entity.responsibilities = ["Responsibility"]
            entities.append(entity)

        entity_result = MagicMock()
        entity_result.scalars.return_value.all.return_value = entities

        # Mock context memberships
        contexts = {}
        memberships = []
        for i, (entity, ctx_name) in enumerate(
            zip(
                entities,
                ["OrderContext", "OrderContext", "CustomerContext", "ProductContext"],
            )
        ):
            if ctx_name not in contexts:
                ctx = MagicMock(spec=BoundedContext)
                ctx.name = ctx_name
                contexts[ctx_name] = ctx

            membership = MagicMock(spec=BoundedContextMembership)
            membership.domain_entity_id = entity.id
            membership.bounded_context = contexts[ctx_name]
            memberships.append(membership)

        membership_result = MagicMock()
        membership_result.scalars.return_value.all.return_value = memberships

        mock_db_session.execute.side_effect = [
            file_result,
            entity_result,
            membership_result,
        ]

        result = await domain_tools.suggest_ddd_refactoring("mixed_contexts.py")

        # Should have context boundary violation
        boundary_violations = [
            s for s in result if s["type"] == "context_boundary_violation"
        ]
        assert len(boundary_violations) == 1
        assert boundary_violations[0]["severity"] == "high"
        assert len(boundary_violations[0]["contexts"]) == 3
        assert set(boundary_violations[0]["contexts"]) == {
            "OrderContext",
            "CustomerContext",
            "ProductContext",
        }

    @pytest.mark.asyncio
    async def test_find_aggregate_roots_with_complex_aggregates(
        self, domain_tools, mock_db_session
    ):
        """Test finding aggregate roots with complex member relationships."""
        # Mock bounded context
        mock_context = MagicMock(spec=BoundedContext)
        mock_context.id = 1
        mock_context.name = "ComplexContext"

        context_result = MagicMock()
        context_result.scalar_one_or_none.return_value = mock_context

        # Mock membership IDs
        membership_result = MagicMock()
        membership_result.__iter__ = MagicMock(
            return_value=iter([(1,), (2,), (3,), (4,), (5,)])
        )

        # Mock aggregate roots
        agg1 = MagicMock(spec=DomainEntity)
        agg1.id = 1
        agg1.name = "CustomerAggregate"
        agg1.description = "Customer aggregate root"
        agg1.invariants = ["Email must be unique", "Age must be positive"]

        agg2 = MagicMock(spec=DomainEntity)
        agg2.id = 2
        agg2.name = "OrderAggregate"
        agg2.description = "Order aggregate root"
        agg2.invariants = ["Total must equal sum of items"]

        agg_result = MagicMock()
        agg_result.scalars.return_value.all.return_value = [agg1, agg2]

        # Mock aggregate members
        # Customer aggregate members
        customer_members = [
            ("Address", "value_object"),
            ("ContactInfo", "value_object"),
            ("CustomerProfile", "entity"),
        ]

        # Order aggregate members
        order_members = [
            ("OrderItem", "entity"),
            ("ShippingInfo", "value_object"),
            ("PaymentInfo", "value_object"),
            ("Discount", "value_object"),
        ]

        def create_members(member_list):
            members = []
            for name, entity_type in member_list:
                member = MagicMock(spec=DomainEntity)
                member.name = name
                member.entity_type = entity_type
                members.append(member)
            return members

        customer_member_result = MagicMock()
        customer_member_result.scalars.return_value.all.return_value = create_members(
            customer_members
        )

        order_member_result = MagicMock()
        order_member_result.scalars.return_value.all.return_value = create_members(
            order_members
        )

        # Mock file paths
        file_result1 = MagicMock()
        file_result1.__iter__ = MagicMock(
            return_value=iter([("/src/domain/customer.py",)])
        )

        file_result2 = MagicMock()
        file_result2.__iter__ = MagicMock(
            return_value=iter(
                [("/src/domain/order.py",), ("/src/domain/order_item.py",)]
            )
        )

        mock_db_session.execute.side_effect = [
            context_result,
            membership_result,
            agg_result,
            customer_member_result,
            file_result1,
            order_member_result,
            file_result2,
        ]

        result = await domain_tools.find_aggregate_roots("ComplexContext")

        assert len(result) == 2

        # Check Customer aggregate
        customer_agg = next(r for r in result if r["name"] == "CustomerAggregate")
        assert len(customer_agg["members"]) == 3
        assert len(customer_agg["invariants"]) == 2
        member_types = {m["type"] for m in customer_agg["members"]}
        assert member_types == {"value_object", "entity"}

        # Check Order aggregate
        order_agg = next(r for r in result if r["name"] == "OrderAggregate")
        assert len(order_agg["members"]) == 4
        assert len(order_agg["source_files"]) == 2

    @pytest.mark.asyncio
    async def test_generate_context_map_empty_relationships(
        self, domain_tools, mock_db_session
    ):
        """Test generating context map with no relationships."""
        # Create isolated contexts
        contexts = []
        for i, name in enumerate(
            ["IsolatedContext1", "IsolatedContext2", "IsolatedContext3"]
        ):
            ctx = MagicMock(spec=BoundedContext)
            ctx.id = i
            ctx.name = name
            ctx.context_type = "supporting"
            ctx.description = f"{name} with no relationships"
            contexts.append(ctx)

        context_result = MagicMock()
        context_result.scalars.return_value.all.return_value = contexts

        # No relationships
        rel_result = MagicMock()
        rel_result.scalars.return_value.all.return_value = []

        mock_db_session.execute.side_effect = [context_result, rel_result]

        # Test Mermaid format with isolated contexts
        result = await domain_tools.generate_context_map("mermaid")

        assert "diagram" in result
        diagram = result["diagram"]

        # Should have all contexts but no arrows
        assert "IsolatedContext1[IsolatedContext1]" in diagram
        assert "IsolatedContext2[IsolatedContext2]" in diagram
        assert "IsolatedContext3[IsolatedContext3]" in diagram
        assert "-->" not in diagram  # No relationships
