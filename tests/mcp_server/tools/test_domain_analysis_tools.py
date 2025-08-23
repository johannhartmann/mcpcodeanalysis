"""Tests for domain-driven design analysis tools."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import FastMCP
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.domain_models import (
    BoundedContext,
    BoundedContextMembership,
    DomainEntity,
    DomainRelationship,
    DomainSummary,
)
from src.mcp_server.tools.domain_tools import DomainTools


@pytest.fixture
def mock_db_session() -> Any:
    """Create mock database session."""
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
def mock_mcp() -> Any:
    """Create mock FastMCP instance."""
    mcp = MagicMock(spec=FastMCP)
    mcp.tool = MagicMock(side_effect=lambda **kwargs: lambda func: func)
    return mcp


@pytest.fixture
def domain_tools(mock_db_session: Any, mock_mcp: Any) -> DomainTools:
    """Create domain tools fixture."""
    return DomainTools(mock_db_session, mock_mcp)


class TestDomainTools:
    """Tests for DomainTools class."""

    @pytest.mark.asyncio
    async def test_register_tools(
        self: "TestDomainTools", domain_tools: DomainTools, mock_mcp: Any
    ) -> None:
        """Test tool registration."""
        await domain_tools.register_tools()

        # Should register 6 tools
        assert mock_mcp.tool.call_count == 6

        # Check tool names
        tool_names = [call[1]["name"] for call in mock_mcp.tool.call_args_list]
        expected_tools = [
            "extract_domain_model",
            "find_aggregate_roots",
            "analyze_bounded_context",
            "suggest_ddd_refactoring",
            "find_bounded_contexts",
            "generate_context_map",
        ]
        assert set(tool_names) == set(expected_tools)

    @pytest.mark.asyncio
    async def test_extract_domain_model_file_not_found(
        self: "TestDomainTools",
        domain_tools: DomainTools,
        mock_db_session: Any,
    ) -> None:
        """Test extracting domain model when file is not found."""
        # Mock no file found
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        result = await domain_tools.extract_domain_model("test.py")

        assert result["error"] == "File not found: test.py"
        assert result["entities"] == []
        assert result["relationships"] == []

    @pytest.mark.asyncio
    async def test_extract_domain_model_cached(
        self: Any, domain_tools: DomainTools, mock_db_session: Any
    ) -> None:
        """Test extracting domain model from cache."""
        # Mock file
        mock_file = MagicMock()
        mock_file.id = 1
        mock_file.path = "/test/file.py"

        file_result = MagicMock()
        file_result.scalar_one_or_none.return_value = mock_file

        # Mock existing entities
        mock_entity = MagicMock(spec=DomainEntity)
        mock_entity.name = "TestEntity"
        mock_entity.entity_type = "entity"
        mock_entity.description = "Test description"
        mock_entity.business_rules = ["Rule 1"]
        mock_entity.invariants = ["Invariant 1"]
        mock_entity.responsibilities = ["Responsibility 1"]
        mock_entity.id = 10

        entity_result = MagicMock()
        entity_result.scalars.return_value.all.return_value = [mock_entity]

        # Mock relationship
        mock_relationship = MagicMock(spec=DomainRelationship)
        mock_relationship.source_entity = mock_entity
        mock_relationship.target_entity = mock_entity
        mock_relationship.relationship_type = "uses"
        mock_relationship.description = "Test relationship"

        rel_result = MagicMock()
        rel_result.scalars.return_value.all.return_value = [mock_relationship]

        mock_db_session.execute.side_effect = [
            file_result,
            entity_result,
            rel_result,
        ]

        result = await domain_tools.extract_domain_model("test.py", True)

        assert result["file"] == "test.py"
        assert result["source"] == "cached"
        assert len(result["entities"]) == 1
        assert result["entities"][0]["name"] == "TestEntity"
        assert len(result["relationships"]) == 1
        assert result["relationships"][0]["type"] == "uses"

    @pytest.mark.asyncio
    async def test_extract_domain_model_new_extraction(
        self: "TestDomainTools",
        domain_tools: DomainTools,
        mock_db_session: Any,
    ) -> None:
        """Test extracting new domain model."""
        # Mock file
        mock_file = MagicMock()
        mock_file.id = 1
        mock_file.path = "/test/file.py"

        file_result = MagicMock()
        file_result.scalar_one_or_none.return_value = mock_file

        # Mock no existing entities
        entity_result = MagicMock()
        entity_result.scalars.return_value.all.return_value = []

        mock_db_session.execute.side_effect = [file_result, entity_result]

        # Mock domain indexer
        with patch("src.mcp_server.tools.domain_tools.DomainIndexer") as mock_indexer:
            mock_indexer_instance = MagicMock()
            mock_indexer_instance.index_file = AsyncMock(
                return_value={"status": "success"}
            )
            mock_indexer.return_value = mock_indexer_instance

            # Mock newly created entities
            new_entity = MagicMock(spec=DomainEntity)
            new_entity.name = "NewEntity"
            new_entity.entity_type = "aggregate_root"
            new_entity.description = "New description"
            new_entity.business_rules = []
            new_entity.invariants = []
            new_entity.responsibilities = []
            new_entity.id = 20

            new_entity_result = MagicMock()
            new_entity_result.scalars.return_value.all.return_value = [new_entity]

            # Add the new entity result to the mock sequence
            mock_db_session.execute.side_effect = [
                file_result,
                entity_result,
                new_entity_result,
            ]

            result = await domain_tools.extract_domain_model("test.py", False)

            assert result["file"] == "test.py"
            assert result["source"] == "extracted"
            assert len(result["entities"]) == 1
            assert result["entities"][0]["name"] == "NewEntity"
            assert result["entities"][0]["type"] == "aggregate_root"
            assert result["relationships"] == []

    @pytest.mark.asyncio
    async def test_find_aggregate_roots_no_context(
        self: Any, domain_tools: DomainTools, mock_db_session: Any
    ) -> None:
        """Test finding aggregate roots without context filter."""
        # Mock aggregate roots
        mock_agg1 = MagicMock(spec=DomainEntity)
        mock_agg1.id = 1
        mock_agg1.name = "OrderAggregate"
        mock_agg1.description = "Order aggregate root"
        mock_agg1.invariants = ["Total must be positive"]

        mock_agg2 = MagicMock(spec=DomainEntity)
        mock_agg2.id = 2
        mock_agg2.name = "CustomerAggregate"
        mock_agg2.description = "Customer aggregate root"
        mock_agg2.invariants = []

        agg_result = MagicMock()
        agg_result.scalars.return_value.all.return_value = [mock_agg1, mock_agg2]

        # Mock related entities for each aggregate
        mock_member = MagicMock(spec=DomainEntity)
        mock_member.name = "OrderItem"
        mock_member.entity_type = "entity"

        member_result = MagicMock()
        member_result.scalars.return_value.all.return_value = [mock_member]

        # Mock file paths
        file_result = MagicMock()
        file_result.__iter__ = MagicMock(
            return_value=iter([("/src/order.py",), ("/src/customer.py",)])
        )

        mock_db_session.execute.side_effect = [
            agg_result,
            member_result,  # Members for first aggregate
            file_result,  # Files for first aggregate
            member_result,  # Members for second aggregate
            file_result,  # Files for second aggregate
        ]

        result = await domain_tools.find_aggregate_roots()

        assert len(result) == 2
        assert result[0]["name"] == "OrderAggregate"
        assert result[0]["invariants"] == ["Total must be positive"]
        assert len(result[0]["members"]) == 1
        assert result[0]["members"][0]["name"] == "OrderItem"

    @pytest.mark.asyncio
    async def test_find_aggregate_roots_with_context(
        self: Any, domain_tools: DomainTools, mock_db_session: Any
    ) -> None:
        """Test finding aggregate roots within a specific context."""
        # Mock bounded context
        mock_context = MagicMock(spec=BoundedContext)
        mock_context.id = 10
        mock_context.name = "OrderContext"

        context_result = MagicMock()
        context_result.scalar_one_or_none.return_value = mock_context

        # Mock memberships
        membership_result = MagicMock()
        membership_result.__iter__ = MagicMock(return_value=iter([(1,), (2,)]))

        # Mock aggregates in context
        mock_agg = MagicMock(spec=DomainEntity)
        mock_agg.id = 1
        mock_agg.name = "OrderAggregate"
        mock_agg.description = "Order aggregate"
        mock_agg.invariants = []

        agg_result = MagicMock()
        agg_result.scalars.return_value.all.return_value = [mock_agg]

        mock_db_session.execute.side_effect = [
            context_result,
            membership_result,
            agg_result,
            MagicMock(scalars=MagicMock(return_value=MagicMock(all=list))),
            MagicMock(__iter__=MagicMock(return_value=iter([]))),
        ]

        result = await domain_tools.find_aggregate_roots("OrderContext")

        assert len(result) == 1
        assert result[0]["name"] == "OrderAggregate"

    @pytest.mark.asyncio
    async def test_analyze_bounded_context_not_found(
        self: Any, domain_tools: DomainTools, mock_db_session: Any
    ) -> None:
        """Test analyzing bounded context that doesn't exist."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        result = await domain_tools.analyze_bounded_context("NonExistentContext")

        assert result["error"] == "Bounded context not found: NonExistentContext"

    @pytest.mark.asyncio
    async def test_analyze_bounded_context_success(
        self: Any, domain_tools: DomainTools, mock_db_session: Any
    ) -> None:
        """Test successful bounded context analysis."""
        # Mock bounded context
        mock_context = MagicMock(spec=BoundedContext)
        mock_context.id = 1
        mock_context.name = "OrderContext"
        mock_context.description = "Order management context"
        mock_context.ubiquitous_language = ["Order", "Customer", "Product"]
        mock_context.core_concepts = ["Order fulfillment", "Payment processing"]
        mock_context.cohesion_score = 0.85
        mock_context.coupling_score = 0.25
        mock_context.modularity_score = 0.75

        # Mock memberships
        mock_membership1 = MagicMock(spec=BoundedContextMembership)
        mock_membership1.domain_entity_id = 10
        mock_membership2 = MagicMock(spec=BoundedContextMembership)
        mock_membership2.domain_entity_id = 20

        mock_context.memberships = [mock_membership1, mock_membership2]

        context_result = MagicMock()
        context_result.scalar_one_or_none.return_value = mock_context

        # Mock entities
        mock_entity1 = MagicMock(spec=DomainEntity)
        mock_entity1.name = "Order"
        mock_entity1.entity_type = "aggregate_root"
        mock_entity1.description = "Order aggregate"

        mock_entity2 = MagicMock(spec=DomainEntity)
        mock_entity2.name = "OrderItem"
        mock_entity2.entity_type = "entity"
        mock_entity2.description = "Order item entity"

        entity_result = MagicMock()
        entity_result.scalars.return_value.all.return_value = [
            mock_entity1,
            mock_entity2,
        ]

        # Mock relationships
        rel_result = MagicMock()
        rel_result.scalars.return_value.all.return_value = []

        ext_rel_result = MagicMock()
        ext_rel_result.scalars.return_value.all.return_value = []

        # Mock summary
        mock_summary = MagicMock(spec=DomainSummary)
        mock_summary.business_summary = "Context for managing orders"

        summary_result = MagicMock()
        summary_result.scalar_one_or_none.return_value = mock_summary

        mock_db_session.execute.side_effect = [
            context_result,
            entity_result,
            rel_result,
            ext_rel_result,
            summary_result,
        ]

        result = await domain_tools.analyze_bounded_context("OrderContext")

        assert result["name"] == "OrderContext"
        assert result["total_entities"] == 2
        assert "aggregate_root" in result["entities_by_type"]
        assert len(result["entities_by_type"]["aggregate_root"]) == 1
        assert result["cohesion_score"] == 0.85
        assert result["summary"] == "Context for managing orders"

    @pytest.mark.asyncio
    async def test_suggest_ddd_refactoring_file_not_found(
        self: Any, domain_tools: DomainTools, mock_db_session: Any
    ) -> None:
        """Test DDD refactoring suggestions when file not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        result = await domain_tools.suggest_ddd_refactoring("test.py")

        assert len(result) == 1
        assert result[0]["error"] == "File not found: test.py"

    @pytest.mark.asyncio
    async def test_suggest_ddd_refactoring_missing_aggregate(
        self: Any, domain_tools: DomainTools, mock_db_session: Any
    ) -> None:
        """Test DDD refactoring suggestions for missing aggregate root."""
        # Mock file
        mock_file = MagicMock()
        mock_file.id = 1

        file_result = MagicMock()
        file_result.scalar_one_or_none.return_value = mock_file

        # Mock entities without aggregate root
        mock_entity1 = MagicMock(spec=DomainEntity)
        mock_entity1.name = "Order"
        mock_entity1.entity_type = "entity"
        mock_entity1.business_rules = []
        mock_entity1.invariants = []
        mock_entity1.responsibilities = ["Process order"]

        mock_entity2 = MagicMock(spec=DomainEntity)
        mock_entity2.name = "OrderItem"
        mock_entity2.entity_type = "entity"
        mock_entity2.business_rules = ["Price must be positive"]
        mock_entity2.invariants = []
        mock_entity2.responsibilities = ["Track item quantity"]

        entity_result = MagicMock()
        entity_result.scalars.return_value.all.return_value = [
            mock_entity1,
            mock_entity2,
        ]

        mock_db_session.execute.side_effect = [file_result, entity_result]

        result = await domain_tools.suggest_ddd_refactoring("test.py")

        # Should suggest missing aggregate root
        assert any(s["type"] == "missing_aggregate" for s in result)
        assert any(s["type"] == "anemic_domain_model" for s in result)

    @pytest.mark.asyncio
    async def test_suggest_ddd_refactoring_bloated_entity(
        self: Any, domain_tools: DomainTools, mock_db_session: Any
    ) -> None:
        """Test DDD refactoring suggestions for bloated entities."""
        # Mock file
        mock_file = MagicMock()
        mock_file.id = 1

        file_result = MagicMock()
        file_result.scalar_one_or_none.return_value = mock_file

        # Mock bloated entity
        mock_entity = MagicMock(spec=DomainEntity)
        mock_entity.name = "Order"
        mock_entity.entity_type = "aggregate_root"
        mock_entity.business_rules = ["Rule 1", "Rule 2"]
        mock_entity.invariants = ["Invariant 1"]
        mock_entity.responsibilities = [
            "Process order",
            "Calculate tax",
            "Send notifications",
            "Update inventory",
            "Generate invoice",
            "Process payment",
            "Handle refunds",
        ]

        entity_result = MagicMock()
        entity_result.scalars.return_value.all.return_value = [mock_entity]

        mock_db_session.execute.side_effect = [file_result, entity_result]

        result = await domain_tools.suggest_ddd_refactoring("test.py")

        # Should suggest extracting responsibilities
        bloated_suggestions = [s for s in result if s["type"] == "bloated_entity"]
        assert len(bloated_suggestions) == 1
        assert bloated_suggestions[0]["entity"] == "Order"
        assert "too many responsibilities" in bloated_suggestions[0]["message"]

    @pytest.mark.asyncio
    async def test_find_bounded_contexts(
        self: Any, domain_tools: DomainTools, mock_db_session: Any
    ) -> None:
        """Test finding bounded contexts."""
        # Mock contexts
        mock_context1 = MagicMock(spec=BoundedContext)
        mock_context1.name = "OrderContext"
        mock_context1.description = "Order management"
        mock_context1.core_concepts = [
            "Order",
            "Payment",
            "Shipping",
            "Customer",
            "Product",
        ]
        mock_context1.cohesion_score = 0.9
        mock_context1.context_type = "core"
        mock_context1.memberships = [MagicMock() for _ in range(5)]

        mock_context2 = MagicMock(spec=BoundedContext)
        mock_context2.name = "InventoryContext"
        mock_context2.description = "Inventory management"
        mock_context2.core_concepts = ["Stock", "Warehouse"]
        mock_context2.cohesion_score = 0.7
        mock_context2.context_type = "supporting"
        mock_context2.memberships = [MagicMock() for _ in range(2)]  # Below threshold

        mock_context3 = MagicMock(spec=BoundedContext)
        mock_context3.name = "ShippingContext"
        mock_context3.description = "Shipping management"
        mock_context3.core_concepts = ["Shipment", "Carrier", "Tracking"]
        mock_context3.cohesion_score = 0.8
        mock_context3.context_type = "supporting"
        mock_context3.memberships = [MagicMock() for _ in range(4)]

        context_result = MagicMock()
        context_result.scalars.return_value.all.return_value = [
            mock_context1,
            mock_context2,
            mock_context3,
        ]

        mock_db_session.execute.return_value = context_result

        result = await domain_tools.find_bounded_contexts(min_entities=3)

        # Should only return contexts with >= 3 entities
        assert len(result) == 2
        assert result[0]["name"] == "OrderContext"  # Highest cohesion
        assert result[0]["cohesion_score"] == 0.9
        assert result[1]["name"] == "ShippingContext"

    @pytest.mark.asyncio
    async def test_generate_context_map_json(
        self: Any, domain_tools: DomainTools, mock_db_session: Any
    ) -> None:
        """Test generating context map in JSON format."""
        # Mock contexts
        mock_context1 = MagicMock(spec=BoundedContext)
        mock_context1.id = 1
        mock_context1.name = "OrderContext"
        mock_context1.context_type = "core"
        mock_context1.description = "Order management"

        mock_context2 = MagicMock(spec=BoundedContext)
        mock_context2.id = 2
        mock_context2.name = "InventoryContext"
        mock_context2.context_type = "supporting"
        mock_context2.description = "Inventory management"

        context_result = MagicMock()
        context_result.scalars.return_value.all.return_value = [
            mock_context1,
            mock_context2,
        ]

        # Mock relationships
        from src.database.domain_models import ContextRelationship

        mock_rel = MagicMock(spec=ContextRelationship)
        mock_rel.source_context = mock_context1
        mock_rel.target_context = mock_context2
        mock_rel.relationship_type = "customer_supplier"
        mock_rel.description = "Order requests inventory"

        rel_result = MagicMock()
        rel_result.scalars.return_value.all.return_value = [mock_rel]

        mock_db_session.execute.side_effect = [context_result, rel_result]

        result = await domain_tools.generate_context_map("json")

        assert "contexts" in result
        assert len(result["contexts"]) == 2
        assert result["contexts"][0]["name"] == "OrderContext"
        assert "relationships" in result
        assert len(result["relationships"]) == 1
        assert result["relationships"][0]["type"] == "customer_supplier"

    @pytest.mark.asyncio
    async def test_generate_context_map_mermaid(
        self: Any, domain_tools: DomainTools, mock_db_session: Any
    ) -> None:
        """Test generating context map in Mermaid format."""
        # Mock contexts
        mock_context1 = MagicMock(spec=BoundedContext)
        mock_context1.name = "OrderContext"
        mock_context1.context_type = "core"

        mock_context2 = MagicMock(spec=BoundedContext)
        mock_context2.name = "PaymentContext"
        mock_context2.context_type = "generic"

        context_result = MagicMock()
        context_result.scalars.return_value.all.return_value = [
            mock_context1,
            mock_context2,
        ]

        # Mock relationship
        from src.database.domain_models import ContextRelationship

        mock_rel = MagicMock(spec=ContextRelationship)
        mock_rel.source_context = mock_context1
        mock_rel.target_context = mock_context2
        mock_rel.relationship_type = "anti_corruption_layer"

        rel_result = MagicMock()
        rel_result.scalars.return_value.all.return_value = [mock_rel]

        mock_db_session.execute.side_effect = [context_result, rel_result]

        result = await domain_tools.generate_context_map("mermaid")

        assert "diagram" in result
        diagram = result["diagram"]
        assert "graph TD" in diagram
        assert "OrderContext[[OrderContext]]" in diagram  # Core context notation
        assert "PaymentContext[PaymentContext]" in diagram  # Generic context notation
        assert "OrderContext -->|ACL| PaymentContext" in diagram

    @pytest.mark.asyncio
    async def test_generate_context_map_unsupported_format(
        self: Any, domain_tools: DomainTools, mock_db_session: Any
    ) -> None:
        """Test generating context map with unsupported format."""
        result = await domain_tools.generate_context_map("invalid")

        assert result["error"] == "Unsupported format: invalid"

    @pytest.mark.asyncio
    async def test_extract_domain_model_exception_handling(
        self: Any, domain_tools: DomainTools, mock_db_session: Any
    ) -> None:
        """Test exception handling in extract_domain_model."""
        mock_db_session.execute.side_effect = Exception("Database error")

        result = await domain_tools.extract_domain_model("test.py")

        assert result["error"] == "Database error"
        assert result["entities"] == []
        assert result["relationships"] == []

    @pytest.mark.asyncio
    async def test_find_aggregate_roots_exception_handling(
        self: Any, domain_tools: DomainTools, mock_db_session: Any
    ) -> None:
        """Test exception handling in find_aggregate_roots."""
        mock_db_session.execute.side_effect = Exception("Database error")

        result = await domain_tools.find_aggregate_roots()

        assert result == []
