"""Analyze domain patterns and anti-patterns in codebases."""

from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.domain_models import (
    BoundedContext,
    BoundedContextMembership,
    DomainEntity,
    DomainRelationship,
)
from src.database.models import File
from src.logger import get_logger

logger = get_logger(__name__)

# Constants for coupling thresholds
HIGH_COUPLING_THRESHOLD = 5
VERY_HIGH_COUPLING_THRESHOLD = 10
HIGH_COUPLING_SCORE = 3
ENTITY_GROWTH_RATE_THRESHOLD = 20
GOD_OBJECT_RESPONSIBILITIES_THRESHOLD = 7
RAPID_COUPLING_THRESHOLD = 20
CONTEXT_SHARE_THRESHOLD = 2

# Constants for coupling buckets
LOW_COUPLING_THRESHOLD = 1
MEDIUM_COUPLING_THRESHOLD = 3


class DomainPatternAnalyzer:
    """Analyze domain patterns, anti-patterns, and evolution."""

    def __init__(self, db_session: AsyncSession) -> None:
        """Initialize the pattern analyzer.

        Args:
            db_session: Database session
        """
        self.db_session = db_session

    async def analyze_cross_context_coupling(
        self,
        repository_id: int | None = None,
    ) -> dict[str, Any]:
        """Analyze coupling between bounded contexts.

        Args:
            repository_id: Optional repository filter

        Returns:
            Coupling analysis with metrics and recommendations
        """
        # Get all contexts
        query = select(BoundedContext)
        if repository_id:
            # Filter by repository through entities
            query = (
                query.join(
                    BoundedContextMembership,
                    BoundedContext.id == BoundedContextMembership.bounded_context_id,
                )
                .join(
                    DomainEntity,
                    BoundedContextMembership.domain_entity_id == DomainEntity.id,
                )
                .join(
                    File,
                    # SQLite compatibility: use JSON contains instead of array any
                    DomainEntity.source_entities.contains(File.id),
                )
                .where(File.repository_id == repository_id)
                .distinct()
            )

        result = await self.db_session.execute(query)
        contexts = result.scalars().all()

        coupling_analysis: dict[str, Any] = {
            "contexts": [],
            "high_coupling_pairs": [],
            "recommendations": [],
            "metrics": {
                "average_coupling": 0,
                "max_coupling": 0,
                "coupling_distribution": {},
            },
        }

        # Analyze each context
        context_metrics: dict[int, dict[str, Any]] = {}
        total_coupling = 0.0
        coupling_scores: list[float] = []

        for context in contexts:
            # Get entities in this context
            membership_result = await self.db_session.execute(
                select(DomainEntity)
                .join(BoundedContextMembership)
                .where(BoundedContextMembership.bounded_context_id == context.id),
            )
            entities = membership_result.scalars().all()
            entity_ids = [int(e.id) for e in entities]

            # Count outgoing relationships to other contexts
            outgoing_result = await self.db_session.execute(
                select(
                    DomainRelationship,
                    DomainEntity.id,
                    BoundedContextMembership.bounded_context_id,
                )
                .join(
                    DomainEntity,
                    DomainRelationship.target_entity_id == DomainEntity.id,
                )
                .join(
                    BoundedContextMembership,
                    DomainEntity.id == BoundedContextMembership.domain_entity_id,
                )
                .where(
                    DomainRelationship.source_entity_id.in_(entity_ids),
                    BoundedContextMembership.bounded_context_id != context.id,
                ),
            )

            # Group by target context
            coupling_by_context: dict[int, int] = defaultdict(int)
            relationship_types = defaultdict(set)

            for rel, _target_entity_id, target_context_id in outgoing_result:
                t_id = int(target_context_id)
                coupling_by_context[t_id] += 1
                relationship_types[t_id].add(rel.relationship_type)

            # Calculate metrics
            total_outgoing = sum(coupling_by_context.values())
            context_coupling_score = total_outgoing / max(len(entities), 1)

            context_metrics[int(context.id)] = {
                "name": context.name,
                "entity_count": len(entities),
                "outgoing_relationships": total_outgoing,
                "coupling_score": context_coupling_score,
                "coupled_contexts": len(coupling_by_context),
                "coupling_details": dict(coupling_by_context),
                "relationship_types": {
                    k: list(v) for k, v in relationship_types.items()
                },
            }

            total_coupling += context_coupling_score
            coupling_scores.append(context_coupling_score)

            cast("list[dict[str, Any]]", coupling_analysis["contexts"]).append(
                {
                    "name": context.name,
                    "coupling_score": round(context_coupling_score, 2),
                    "outgoing_dependencies": total_outgoing,
                    "coupled_with": len(coupling_by_context),
                },
            )

        # Find high coupling pairs
        for ctx1_metrics in context_metrics.values():
            for ctx2_id, count in ctx1_metrics["coupling_details"].items():
                if count > HIGH_COUPLING_THRESHOLD:
                    ctx2_metrics = context_metrics.get(ctx2_id, {})
                    cast(
                        "list[dict[str, Any]]",
                        coupling_analysis["high_coupling_pairs"],
                    ).append(
                        {
                            "source": ctx1_metrics["name"],
                            "target": ctx2_metrics.get("name", "Unknown"),
                            "relationship_count": count,
                            "relationship_types": ctx1_metrics[
                                "relationship_types"
                            ].get(ctx2_id, []),
                            "recommendation": self._get_coupling_recommendation(
                                count,
                                ctx1_metrics["relationship_types"].get(ctx2_id, []),
                            ),
                        },
                    )

        # Calculate overall metrics
        if coupling_scores:
            coupling_analysis["metrics"]["average_coupling"] = round(
                total_coupling / len(coupling_scores),
                2,
            )
            coupling_analysis["metrics"]["max_coupling"] = round(
                max(coupling_scores),
                2,
            )

            # Distribution
            for score in coupling_scores:
                if score < LOW_COUPLING_THRESHOLD:
                    bucket = "low"
                elif score < MEDIUM_COUPLING_THRESHOLD:
                    bucket = "medium"
                else:
                    bucket = "high"

                cast(
                    "dict[str, int]",
                    coupling_analysis["metrics"]["coupling_distribution"],
                )[bucket] = (
                    coupling_analysis["metrics"]["coupling_distribution"].get(bucket, 0)
                    + 1
                )

        # Generate recommendations
        coupling_analysis[
            "recommendations"
        ] = await self._generate_coupling_recommendations(
            context_metrics,
        )

        return coupling_analysis

    async def suggest_context_splits(
        self,
        min_entities: int = 20,
        max_cohesion_threshold: float = 0.4,
    ) -> list[dict[str, Any]]:
        """Suggest how to split large bounded contexts.

        Args:
            min_entities: Minimum entities for a context to be considered
            max_cohesion_threshold: Maximum cohesion score to suggest split

        Returns:
            List of split suggestions with details
        """
        # Find large contexts with low cohesion
        result = await self.db_session.execute(
            select(BoundedContext).where(
                BoundedContext.cohesion_score <= max_cohesion_threshold,
            ),
        )

        candidates = []
        for context in result.scalars().all():
            # Count memberships
            membership_count_result = await self.db_session.execute(
                select(func.count(BoundedContextMembership.id)).where(
                    BoundedContextMembership.bounded_context_id == context.id
                )
            )
            membership_count = membership_count_result.scalar() or 0

            if membership_count >= min_entities:
                candidates.append(context)

        suggestions: list[dict[str, Any]] = []

        for context in candidates:
            # Get entities and their relationships
            entity_result = await self.db_session.execute(
                select(DomainEntity)
                .join(BoundedContextMembership)
                .where(BoundedContextMembership.bounded_context_id == context.id),
            )
            entities = entity_result.scalars().all()

            # Build internal relationship graph
            entity_graph: dict[int, set[int]] = defaultdict(set)
            entity_map: dict[int, DomainEntity] = {int(e.id): e for e in entities}

            for entity in entities:
                # Get relationships where this entity is source
                rel_result = await self.db_session.execute(
                    select(DomainRelationship).where(
                        DomainRelationship.source_entity_id == entity.id,
                        DomainRelationship.target_entity_id.in_(entity_map.keys()),
                    ),
                )

                for rel in rel_result.scalars().all():
                    entity_graph[int(entity.id)].add(int(rel.target_entity_id))
                    entity_graph[int(rel.target_entity_id)].add(int(entity.id))

            # Find clusters using simple connected components
            clusters = self._find_entity_clusters(entity_graph, entity_map)

            if len(clusters) > 1:
                suggestion: dict[str, Any] = {
                    "context": context.name,
                    "current_size": len(entities),
                    "cohesion_score": context.cohesion_score,
                    "suggested_splits": [],
                    "reasoning": f"Low cohesion ({context.cohesion_score:.2f}) indicates weak relationships between entity groups",
                }

                for _i, cluster in enumerate(clusters):
                    cluster_entities = [entity_map[eid] for eid in cluster]

                    # Find potential aggregate roots
                    aggregate_roots = [
                        e for e in cluster_entities if e.entity_type == "aggregate_root"
                    ]

                    # Determine cluster theme
                    cluster_name = self._suggest_cluster_name(
                        cluster_entities,
                        aggregate_roots,
                    )

                    suggestion["suggested_splits"].append(
                        {
                            "suggested_name": cluster_name,
                            "entity_count": len(cluster),
                            "aggregate_roots": [a.name for a in aggregate_roots],
                            "key_entities": [e.name for e in cluster_entities[:5]],
                            "cohesion_estimate": self._estimate_cluster_cohesion(
                                cluster,
                                entity_graph,
                            ),
                        },
                    )

                suggestions.append(suggestion)

        return list(suggestions)

    async def detect_anti_patterns(
        self,
        _repository_id: int | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Detect DDD anti-patterns in the codebase.

        Args:
            repository_id: Optional repository filter

        Returns:
            Dictionary of anti-patterns found
        """
        anti_patterns: dict[str, list[dict[str, Any]]] = {
            "anemic_domain_models": [],
            "god_objects": [],
            "circular_dependencies": [],
            "missing_aggregate_roots": [],
            "chatty_contexts": [],
            "shared_kernel_abuse": [],
        }

        # 1. Detect anemic domain models (entities with no business rules)
        query = select(DomainEntity).where(
            DomainEntity.entity_type.in_(["entity", "aggregate_root"]),
            # SQLite compatibility: check JSON array length
            func.json_array_length(DomainEntity.business_rules) == 0,
            func.json_array_length(DomainEntity.invariants) == 0,
        )

        result = await self.db_session.execute(query)
        for entity in result.scalars().all():
            anti_patterns["anemic_domain_models"].append(
                {
                    "entity": entity.name,
                    "type": entity.entity_type,
                    "issue": "No business rules or invariants defined",
                    "recommendation": "Add business logic to make this a rich domain model",
                    "severity": "medium",
                },
            )

        # 2. Detect god objects (entities with too many responsibilities)
        query = select(DomainEntity).where(
            func.json_array_length(DomainEntity.responsibilities)
            > GOD_OBJECT_RESPONSIBILITIES_THRESHOLD,
        )

        result = await self.db_session.execute(query)
        for entity in result.scalars().all():
            anti_patterns["god_objects"].append(
                {
                    "entity": entity.name,
                    "responsibility_count": len(entity.responsibilities),
                    "responsibilities": [*entity.responsibilities[:5], "..."],
                    "issue": "Too many responsibilities",
                    "recommendation": "Split into multiple focused entities or extract domain services",
                    "severity": "high",
                },
            )

        # 3. Detect circular dependencies
        circular_deps = await self._find_circular_dependencies()
        anti_patterns["circular_dependencies"] = circular_deps

        # 4. Detect missing aggregate roots (contexts with only entities)
        contexts_result = await self.db_session.execute(
            select(BoundedContext),
        )

        for context in contexts_result.scalars().all():
            # Get entity types in context
            entity_result = await self.db_session.execute(
                select(DomainEntity.entity_type)
                .join(BoundedContextMembership)
                .where(BoundedContextMembership.bounded_context_id == context.id)
                .distinct(),
            )

            entity_types = [row[0] for row in entity_result]

            if "entity" in entity_types and "aggregate_root" not in entity_types:
                anti_patterns["missing_aggregate_roots"].append(
                    {
                        "context": context.name,
                        "issue": "Context has entities but no aggregate root",
                        "recommendation": "Identify the main entity that maintains consistency and make it an aggregate root",
                        "severity": "high",
                    },
                )

        # 5. Detect chatty contexts (too many inter-context calls)
        chatty = await self._detect_chatty_contexts()
        anti_patterns["chatty_contexts"] = chatty

        # 6. Detect shared kernel abuse
        shared_kernel = await self._detect_shared_kernel_abuse()
        anti_patterns["shared_kernel_abuse"] = shared_kernel

        return anti_patterns

    async def analyze_evolution(
        self,
        _repository_id: int,
        days: int = 30,
    ) -> dict[str, Any]:
        """Analyze how the domain model evolved over time.

        Args:
            repository_id: Repository to analyze
            days: Number of days to look back

        Returns:
            Evolution analysis with trends and changes
        """
        since_date = datetime.now(UTC) - timedelta(days=days)

        evolution: dict[str, Any] = {
            "time_period": f"Last {days} days",
            "entity_changes": {
                "added": [],
                "removed": [],
                "modified": [],
            },
            "context_changes": {
                "added": [],
                "removed": [],
                "resized": [],
            },
            "relationship_changes": {
                "new_dependencies": [],
                "removed_dependencies": [],
            },
            "trends": {
                "entity_growth_rate": 0,
                "context_stability": 0,
                "coupling_trend": "stable",
            },
            "insights": [],
        }

        # Get entities created in time period
        new_entities_result = await self.db_session.execute(
            select(DomainEntity).where(
                DomainEntity.created_at >= since_date,
            ),
        )

        for entity in new_entities_result.scalars().all():
            evolution["entity_changes"]["added"].append(
                {
                    "name": entity.name,
                    "type": entity.entity_type,
                    "created": entity.created_at.isoformat(),
                },
            )

        # Get contexts created in time period
        new_contexts_result = await self.db_session.execute(
            select(BoundedContext).where(BoundedContext.created_at >= since_date),
        )

        for context in new_contexts_result.scalars().all():
            # Count memberships for this context
            membership_count_result = await self.db_session.execute(
                select(func.count(BoundedContextMembership.id)).where(
                    BoundedContextMembership.bounded_context_id == context.id
                )
            )
            membership_count = membership_count_result.scalar() or 0

            evolution["context_changes"]["added"].append(
                {
                    "name": context.name,
                    "created": context.created_at.isoformat(),
                    "size": membership_count,
                },
            )

        # Calculate trends
        # For simplicity with SQLite, just count all entities
        # In production with PostgreSQL, you could filter by repository
        total_entities = await self.db_session.execute(
            select(func.count(DomainEntity.id))
        )
        entity_count = total_entities.scalar() or 0

        if entity_count > 0:
            growth_rate = len(evolution["entity_changes"]["added"]) / entity_count
            evolution["trends"]["entity_growth_rate"] = round(growth_rate * 100, 1)

        # Generate insights
        if evolution["trends"]["entity_growth_rate"] > ENTITY_GROWTH_RATE_THRESHOLD:
            cast("list[str]", evolution["insights"]).append(
                "Rapid growth in domain entities indicates active feature development",
            )

        if len(evolution["context_changes"]["added"]) > 0:
            cast("list[str]", evolution["insights"]).append(
                "New bounded contexts suggest evolving domain understanding",
            )

        return evolution

    def _get_coupling_recommendation(
        self,
        relationship_count: int,
        relationship_types: list[str],
    ) -> str:
        """Generate recommendation for coupling issues."""
        if relationship_count > VERY_HIGH_COUPLING_THRESHOLD:
            if "orchestrates" in relationship_types:
                return "Consider using events or a saga pattern to reduce orchestration coupling"
            if "depends_on" in relationship_types:
                return "High dependency coupling - consider introducing an anti-corruption layer"
            return "Very high coupling - evaluate if these contexts should be merged or use shared kernel pattern"
        if relationship_count > HIGH_COUPLING_THRESHOLD:
            return (
                "Moderate coupling - consider if all these relationships are necessary"
            )
        return "Acceptable coupling level"

    async def _generate_coupling_recommendations(
        self,
        context_metrics: dict[int, dict[str, Any]],
    ) -> list[str]:
        """Generate overall coupling recommendations."""
        recommendations = []

        # Find contexts with highest coupling
        high_coupling = [
            (ctx_id, metrics)
            for ctx_id, metrics in context_metrics.items()
            if metrics["coupling_score"] > HIGH_COUPLING_SCORE
        ]

        if high_coupling:
            recommendations.append(
                f"Consider reviewing {len(high_coupling)} contexts with high coupling scores",
            )

        # Check for asymmetric relationships
        for ctx_id, metrics in context_metrics.items():
            for coupled_ctx_id, count in metrics["coupling_details"].items():
                reverse_count = (
                    context_metrics.get(coupled_ctx_id, {})
                    .get(
                        "coupling_details",
                        {},
                    )
                    .get(ctx_id, 0)
                )

                if count > HIGH_COUPLING_THRESHOLD and reverse_count == 0:
                    recommendations.append(
                        f"{metrics['name']} has one-way dependency on another context - "
                        "consider if this is a customer-supplier relationship",
                    )
                    break

        return recommendations

    def _find_entity_clusters(
        self,
        entity_graph: dict[int, set[int]],
        entity_map: dict[int, DomainEntity],
    ) -> list[set[int]]:
        """Find connected components in entity graph."""
        visited = set()
        clusters = []

        def dfs(entity_id: int, cluster: set[int]) -> None:
            if entity_id in visited:
                return
            visited.add(entity_id)
            cluster.add(entity_id)

            for neighbor in entity_graph.get(entity_id, set()):
                dfs(neighbor, cluster)

        for entity_id in entity_map:
            if entity_id not in visited:
                cluster: set[int] = set()
                dfs(entity_id, cluster)
                if cluster:
                    clusters.append(cluster)

        return clusters

    def _suggest_cluster_name(
        self,
        entities: list[DomainEntity],
        aggregate_roots: list[DomainEntity],
    ) -> str:
        """Suggest a name for an entity cluster."""
        if aggregate_roots:
            return f"{aggregate_roots[0].name} Context"
        if entities:
            # Use most common word in entity names
            words: dict[str, int] = defaultdict(int)
            for entity in entities:
                for word in entity.name.split():
                    words[word.lower()] += 1

            if words:
                most_common = max(words.items(), key=lambda x: x[1])[0]
                return f"{most_common.title()} Context"

        return "Unnamed Context"

    def _estimate_cluster_cohesion(
        self,
        cluster: set[int],
        entity_graph: dict[int, set[int]],
    ) -> float:
        """Estimate cohesion of an entity cluster."""
        if len(cluster) <= 1:
            return 1.0

        internal_edges = 0
        for entity_id in cluster:
            for neighbor in entity_graph.get(entity_id, set()):
                if neighbor in cluster:
                    internal_edges += 1

        # Divide by 2 since we count each edge twice
        internal_edges //= 2

        # Maximum possible edges
        max_edges = len(cluster) * (len(cluster) - 1) / 2

        return internal_edges / max_edges if max_edges > 0 else 0

    async def _find_circular_dependencies(self) -> list[dict[str, Any]]:
        """Find circular dependencies between entities."""
        # This is a simplified version - could be enhanced with cycle detection
        result = await self.db_session.execute(
            select(
                DomainRelationship.source_entity_id,
                DomainRelationship.target_entity_id,
            ),
        )

        # Build adjacency list
        graph = defaultdict(set)
        for source_id, target_id in result:
            graph[source_id].add(target_id)

        cycles = []
        visited = set()
        rec_stack = set()

        def has_cycle(node: int, path: list[int]) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    if has_cycle(neighbor, path.copy()):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle_path = [*path[cycle_start:], neighbor]
                    cycles.append(cycle_path)

            rec_stack.remove(node)
            return False

        for node in graph:
            if node not in visited:
                has_cycle(node, [])

        # Convert to readable format
        circular_deps = []
        for cycle in cycles[:5]:  # Limit to 5
            # Get entity names
            entities = []
            for entity_id in cycle:
                entity_result = await self.db_session.execute(
                    select(DomainEntity).where(DomainEntity.id == entity_id),
                )
                entity = entity_result.scalar_one_or_none()
                if entity:
                    entities.append(entity.name)

            if len(entities) > 1:
                # Coerce to strings for join to satisfy typing
                cycle_names: list[str] = [str(name) for name in entities]
                circular_deps.append(
                    {
                        "cycle": " -> ".join(cycle_names),
                        "length": len(cycle_names) - 1,
                        "issue": "Circular dependency creates tight coupling",
                        "recommendation": "Break the cycle by introducing events or inverting dependencies",
                        "severity": "high",
                    },
                )

        return circular_deps

    async def _detect_chatty_contexts(self) -> list[dict[str, Any]]:
        """Detect contexts with excessive inter-context communication."""
        # Count relationships between contexts
        result = await self.db_session.execute(
            select(
                BoundedContextMembership.bounded_context_id,
                func.count(DomainRelationship.id).label("relationship_count"),
            )
            .join(
                DomainEntity,
                BoundedContextMembership.domain_entity_id == DomainEntity.id,
            )
            .join(
                DomainRelationship,
                DomainEntity.id == DomainRelationship.source_entity_id,
            )
            .group_by(BoundedContextMembership.bounded_context_id)
            .having(func.count(DomainRelationship.id) > RAPID_COUPLING_THRESHOLD),
        )

        chatty = []
        for context_id, count in result:
            context_result = await self.db_session.execute(
                select(BoundedContext).where(BoundedContext.id == context_id),
            )
            context = context_result.scalar_one_or_none()

            if context:
                chatty.append(
                    {
                        "context": context.name,
                        "external_relationships": count,
                        "issue": "Excessive communication with other contexts",
                        "recommendation": "Consider if this context has too many responsibilities or needs better boundaries",
                        "severity": "medium",
                    },
                )

        return chatty

    async def _detect_shared_kernel_abuse(self) -> list[dict[str, Any]]:
        """Detect overuse of shared kernel pattern."""
        # Find contexts that share many entities (simplified check)
        _shared_entities: dict[int, set[int]] = defaultdict(set)

        result = await self.db_session.execute(
            select(
                DomainEntity.id,
                BoundedContextMembership.bounded_context_id,
            ).join(BoundedContextMembership),
        )

        entity_contexts = defaultdict(set)
        for entity_id, context_id in result:
            entity_contexts[entity_id].add(context_id)

        # Find entities in multiple contexts
        shared_kernel_issues = []
        for entity_id, contexts in entity_contexts.items():
            if len(contexts) > CONTEXT_SHARE_THRESHOLD:
                entity_result = await self.db_session.execute(
                    select(DomainEntity).where(DomainEntity.id == entity_id),
                )
                entity = entity_result.scalar_one_or_none()

                if entity:
                    shared_kernel_issues.append(
                        {
                            "entity": entity.name,
                            "shared_by_contexts": len(contexts),
                            "issue": "Entity shared by too many contexts",
                            "recommendation": "Consider if this truly needs to be shared or if each context needs its own version",
                            "severity": "medium",
                        },
                    )

        return shared_kernel_issues
