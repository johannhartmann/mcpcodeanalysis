"""Build and analyze semantic graphs of domain entities."""

import json
from typing import Any

import networkx as nx
import networkx.algorithms.community as nx_comm
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.domain_models import (
    BoundedContext,
    BoundedContextMembership,
    DomainEntity,
    DomainRelationship,
)
from src.embeddings.openai_client import OpenAIClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Constants for magic values
DEFAULT_MIN_CONFIDENCE = 0.5
MIN_COMMUNITY_SIZE = 2
MIN_EDGE_COUNT_FOR_SHARED_KERNEL = 5

# Import leidenalg if available, otherwise use fallback
try:
    import igraph
    import leidenalg

    LEIDEN_AVAILABLE = True
except ImportError:
    logger.warning("leidenalg not available, using NetworkX community detection")
    LEIDEN_AVAILABLE = False


class SemanticGraphBuilder:
    """Build and analyze semantic graphs from domain entities."""

    def __init__(
        self,
        db_session: AsyncSession,
        openai_client: OpenAIClient | None = None,
    ) -> None:
        """Initialize the graph builder.

        Args:
            db_session: Database session
            openai_client: OpenAI client for embeddings
        """
        self.db_session = db_session
        self.openai_client = openai_client or OpenAIClient()

    async def build_graph(
        self,
        *,
        include_weak_relationships: bool = False,
        min_confidence: float = 0.5,
    ) -> nx.Graph:
        """Build a semantic graph from domain entities and relationships.

        Args:
            include_weak_relationships: Include relationships with low strength
            min_confidence: Minimum confidence score for entities/relationships

        Returns:
            NetworkX graph with domain entities as nodes
        """
        # Load entities
        entity_query = select(DomainEntity).where(
            DomainEntity.confidence_score >= min_confidence,
        )
        result = await self.db_session.execute(entity_query)
        entities = result.scalars().all()

        # Create graph
        graph = nx.Graph()

        # Add nodes
        for entity in entities:
            graph.add_node(
                entity.id,
                name=entity.name,
                type=entity.entity_type,
                description=entity.description,
                confidence=entity.confidence_score,
                embedding=entity.concept_embedding,
            )

        # Load relationships
        rel_query = select(DomainRelationship).where(
            DomainRelationship.confidence_score >= min_confidence,
        )
        if not include_weak_relationships:
            rel_query = rel_query.where(
                DomainRelationship.strength >= DEFAULT_MIN_CONFIDENCE,
            )

        result = await self.db_session.execute(rel_query)
        relationships = result.scalars().all()

        # Add edges
        for rel in relationships:
            if rel.source_entity_id in graph and rel.target_entity_id in graph:
                # Calculate edge weight combining strength and confidence
                weight = rel.strength * rel.confidence_score

                graph.add_edge(
                    rel.source_entity_id,
                    rel.target_entity_id,
                    relationship_type=rel.relationship_type,
                    description=rel.description,
                    weight=weight,
                    strength=rel.strength,
                    confidence=rel.confidence_score,
                )

        logger.info(
            "Built graph with %s nodes and %s edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )

        return graph

    async def detect_bounded_contexts(
        self,
        graph: nx.Graph,
        *,
        resolution: float = 1.0,
        use_embeddings: bool = True,
    ) -> list[dict[str, Any]]:
        """Detect bounded contexts using community detection.

        Args:
            graph: Domain entity graph
            resolution: Resolution parameter for community detection
            use_embeddings: Whether to use semantic embeddings for edge weights

        Returns:
            List of detected bounded contexts
        """
        if graph.number_of_nodes() == 0:
            return []

        # Enhance edge weights with semantic similarity if requested
        if use_embeddings:
            await self._enhance_weights_with_embeddings(graph)

        # Detect communities
        if LEIDEN_AVAILABLE:
            communities = self._detect_leiden_communities(graph, resolution)
        else:
            communities = self._detect_louvain_communities(graph, resolution)

        # Convert to bounded contexts
        contexts = []
        for community_id, node_ids in enumerate(communities):
            if len(node_ids) < MIN_COMMUNITY_SIZE:  # Skip single-node communities
                continue

            context = await self._create_bounded_context(
                graph,
                node_ids,
                community_id,
            )
            contexts.append(context)

        logger.info("Detected %d bounded contexts", len(contexts))
        return contexts

    def _detect_leiden_communities(
        self,
        graph: nx.Graph,
        resolution: float,
    ) -> list[list[int]]:
        """Detect communities using Leiden algorithm."""
        # Convert to igraph
        ig = igraph.Graph()
        ig.add_vertices(list(graph.nodes()))

        # Map node IDs to indices
        node_to_idx = {node: i for i, node in enumerate(graph.nodes())}
        idx_to_node = {i: node for node, i in node_to_idx.items()}

        # Add edges with weights
        edges = []
        weights = []
        for u, v, data in graph.edges(data=True):
            edges.append((node_to_idx[u], node_to_idx[v]))
            weights.append(data.get("weight", 1.0))

        ig.add_edges(edges)
        ig.es["weight"] = weights

        # Run Leiden algorithm
        partition = leidenalg.find_partition(
            ig,
            leidenalg.ModularityVertexPartition,
            weights=weights,
            resolution_parameter=resolution,
        )

        # Convert back to node IDs
        communities = []
        for community in partition:
            node_ids = [idx_to_node[idx] for idx in community]
            communities.append(node_ids)

        return communities

    def _detect_louvain_communities(
        self,
        graph: nx.Graph,
        resolution: float,
    ) -> list[list[int]]:
        """Detect communities using Louvain algorithm (fallback)."""
        # Apply Louvain algorithm
        communities = nx_comm.louvain_communities(
            graph,
            weight="weight",
            resolution=resolution,
        )

        # Convert to list of lists
        return [list(community) for community in communities]

    async def _enhance_weights_with_embeddings(
        self,
        graph: nx.Graph,
    ) -> None:
        """Enhance edge weights using semantic embeddings."""
        for u, v, data in graph.edges(data=True):
            u_embedding = graph.nodes[u].get("embedding")
            v_embedding = graph.nodes[v].get("embedding")

            if u_embedding is not None and v_embedding is not None:
                # Calculate cosine similarity
                u_vec = np.array(u_embedding)
                v_vec = np.array(v_embedding)

                similarity = np.dot(u_vec, v_vec) / (
                    np.linalg.norm(u_vec) * np.linalg.norm(v_vec)
                )

                # Combine with existing weight
                current_weight = data.get("weight", 1.0)
                enhanced_weight = current_weight * (0.5 + 0.5 * similarity)

                graph[u][v]["weight"] = enhanced_weight
                graph[u][v]["semantic_similarity"] = similarity

    async def _create_bounded_context(
        self,
        graph: nx.Graph,
        node_ids: list[int],
        community_id: int,
    ) -> dict[str, Any]:
        """Create a bounded context from a community of nodes."""
        # Get entities
        entities = []
        for node_id in node_ids:
            node_data = graph.nodes[node_id]
            entities.append(
                {
                    "id": node_id,
                    "name": node_data["name"],
                    "type": node_data["type"],
                    "description": node_data.get("description", ""),
                },
            )

        # Calculate context metrics
        subgraph = graph.subgraph(node_ids)

        # Cohesion: internal edge density
        possible_edges = len(node_ids) * (len(node_ids) - 1) / 2
        actual_edges = subgraph.number_of_edges()
        cohesion = actual_edges / possible_edges if possible_edges > 0 else 0

        # Coupling: external connections
        external_edges = 0
        for node in node_ids:
            for neighbor in graph.neighbors(node):
                if neighbor not in node_ids:
                    external_edges += 1

        coupling = external_edges / (len(node_ids) * 2) if node_ids else 0

        # Modularity score
        modularity = cohesion - coupling

        # Generate context name and description
        context_info = await self._generate_context_description(entities)

        return {
            "id": community_id,
            "name": context_info["name"],
            "description": context_info["description"],
            "entities": entities,
            "entity_ids": node_ids,
            "cohesion_score": cohesion,
            "coupling_score": coupling,
            "modularity_score": modularity,
            "core_concepts": context_info.get("core_concepts", []),
            "ubiquitous_language": context_info.get("ubiquitous_language", {}),
        }

    async def _generate_context_description(
        self,
        entities: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Use LLM to generate context name and description."""
        # Create prompt
        entity_list = "\n".join(
            [
                f"- {e['name']} ({e['type']}): {e['description']}"
                for e in entities[:20]  # Limit to prevent token overflow
            ],
        )

        prompt = f"""Given these related domain entities that form a bounded context:

{entity_list}

Generate:
1. A concise name for this bounded context (2-3 words)
2. A description of what this context is responsible for
3. Core concepts (list of 3-5 main ideas)
4. Key terms in the ubiquitous language (5-10 domain terms with definitions)

Output as JSON:
{{
  "name": "Context Name",
  "description": "What this context handles...",
  "core_concepts": ["concept1", "concept2"],
  "ubiquitous_language": {{"term": "definition"}}
}}"""

        try:
            response = await self.openai_client.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Domain-Driven Design expert analyzing bounded contexts.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )

            return json.loads(response)

        except Exception:
            logger.exception("Error generating context description: %s")
            # Fallback to simple naming
            aggregate_roots = [e for e in entities if e["type"] == "aggregate_root"]
            name = (
                aggregate_roots[0]["name"] if aggregate_roots else entities[0]["name"]
            )

            return {
                "name": f"{name} Context",
                "description": f"Context containing {len(entities)} related entities",
                "core_concepts": [e["name"] for e in entities[:5]],
                "ubiquitous_language": {},
            }

    async def save_bounded_contexts(
        self,
        contexts: list[dict[str, Any]],
    ) -> list[BoundedContext]:
        """Save detected bounded contexts to database."""
        saved_contexts = []

        for context_data in contexts:
            # Create bounded context
            context = BoundedContext(
                name=context_data["name"],
                description=context_data["description"],
                ubiquitous_language=context_data.get("ubiquitous_language", {}),
                core_concepts=context_data.get("core_concepts", []),
                cohesion_score=context_data.get("cohesion_score"),
                coupling_score=context_data.get("coupling_score"),
                modularity_score=context_data.get("modularity_score"),
            )

            self.db_session.add(context)
            await self.db_session.flush()  # Get ID

            # Create memberships
            for entity_id in context_data["entity_ids"]:
                membership = BoundedContextMembership(
                    domain_entity_id=entity_id,
                    bounded_context_id=context.id,
                )
                self.db_session.add(membership)

            saved_contexts.append(context)

        await self.db_session.commit()
        logger.info("Saved %d bounded contexts", len(saved_contexts))

        return saved_contexts

    async def analyze_context_relationships(
        self,
        graph: nx.Graph,
        contexts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Analyze relationships between bounded contexts."""
        context_relationships = []

        # Create mapping of entity to context
        entity_to_context = {}
        for context in contexts:
            for entity_id in context["entity_ids"]:
                entity_to_context[entity_id] = context["id"]

        # Analyze cross-context edges
        context_edges = {}  # (context1, context2) -> edge data

        for u, v, data in graph.edges(data=True):
            u_context = entity_to_context.get(u)
            v_context = entity_to_context.get(v)

            if (
                u_context is not None
                and v_context is not None
                and u_context != v_context
            ):
                key = tuple(sorted([u_context, v_context]))

                if key not in context_edges:
                    context_edges[key] = {
                        "count": 0,
                        "total_weight": 0,
                        "relationship_types": set(),
                    }

                context_edges[key]["count"] += 1
                context_edges[key]["total_weight"] += data.get("weight", 1.0)
                context_edges[key]["relationship_types"].add(
                    data.get("relationship_type", "unknown"),
                )

        # Create context relationships
        for (ctx1, ctx2), edge_data in context_edges.items():
            # Determine relationship type based on patterns
            rel_type = self._determine_context_relationship_type(
                contexts[ctx1],
                contexts[ctx2],
                edge_data,
            )

            context_relationships.append(
                {
                    "source_context_id": ctx1,
                    "target_context_id": ctx2,
                    "relationship_type": rel_type,
                    "strength": edge_data["total_weight"] / edge_data["count"],
                    "interaction_count": edge_data["count"],
                    "interaction_types": list(edge_data["relationship_types"]),
                },
            )

        return context_relationships

    def _determine_context_relationship_type(
        self,
        context1: dict[str, Any],
        context2: dict[str, Any],
        edge_data: dict[str, Any],
    ) -> str:
        """Determine the type of relationship between contexts."""
        # This is a simplified heuristic - could be enhanced with LLM
        rel_types = edge_data["relationship_types"]

        if "publishes" in rel_types or "subscribes_to" in rel_types:
            return "published_language"
        if "orchestrates" in rel_types:
            return "customer_supplier"
        if edge_data["count"] > MIN_EDGE_COUNT_FOR_SHARED_KERNEL:
            return "shared_kernel"
        if "validates" in rel_types:
            return "anti_corruption_layer"
        return "partnership"
