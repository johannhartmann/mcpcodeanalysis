"""Domain-driven indexing orchestrator."""

import asyncio
from pathlib import Path
from typing import Any, cast

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.domain_models import BoundedContext, DomainEntity, DomainRelationship
from src.database.models import File, Module
from src.domain.entity_extractor import DomainEntityExtractor
from src.domain.graph_builder import SemanticGraphBuilder
from src.domain.summarizer import HierarchicalSummarizer
from src.logger import get_logger
from src.utils.exceptions import NotFoundError

logger = get_logger(__name__)


class DomainIndexer:
    """Orchestrate domain-driven indexing of codebases."""

    def __init__(
        self,
        db_session: AsyncSession,
        embeddings: Any = None,
        llm: Any = None,
    ) -> None:
        """Initialize the domain indexer.

        Args:
            db_session: Database session
            embeddings: Optional embeddings instance (for testing)
            llm: Optional LLM instance (for testing)
        """
        self.db_session = db_session

        # Initialize components
        self.entity_extractor = DomainEntityExtractor(llm=llm)
        self.graph_builder = SemanticGraphBuilder(
            db_session, embeddings=embeddings, llm=llm
        )
        self.summarizer = HierarchicalSummarizer(db_session, llm=llm)

    async def index_file(
        self,
        file_id: int,
        *,
        force_reindex: bool = False,
    ) -> dict[str, Any]:
        """Index a single file for domain entities.

        Args:
            file_id: File ID to index
            force_reindex: Whether to force reindexing

        Returns:
            Indexing results
        """
        # Load file
        result = await self.db_session.execute(select(File).where(File.id == file_id))
        file = result.scalar_one_or_none()

        if not file:
            msg = "File not found"
            raise NotFoundError(msg)

        # Check if already indexed
        if not force_reindex:
            result = await self.db_session.execute(
                select(DomainEntity)
                .where(DomainEntity.source_entities.contains([file_id]))
                .limit(1),
            )
            if result.scalar_one_or_none():
                logger.info("File %s already indexed, skipping", file.path)
                return {"status": "skipped", "file_id": file_id}

        # Read file content
        path_str = str(file.path)
        file_path = Path(path_str)
        if not file_path.exists():
            logger.warning("File %s not found on disk", path_str)
            return {"status": "error", "file_id": file_id, "error": "File not found"}

        # Read content without external async file libs to keep typing strict
        content = await asyncio.to_thread(file_path.read_text, encoding="utf-8")

        # Extract entities
        logger.info("Extracting domain entities from %s", path_str)
        extraction_result = await self.entity_extractor.extract_from_module(
            content,
            path_str,
        )

        # Save entities
        saved_entities = await self._save_domain_entities(
            extraction_result["entities"],
            file_id,
        )

        # Save relationships
        saved_relationships = await self._save_domain_relationships(
            extraction_result["relationships"],
            saved_entities,
        )

        # Generate embeddings for entities
        await self._generate_entity_embeddings(saved_entities)

        await self.db_session.commit()

        return {
            "status": "success",
            "file_id": file_id,
            "entities_extracted": len(saved_entities),
            "relationships_extracted": len(saved_relationships),
        }

    async def index_repository(
        self,
        repository_id: int,
        max_files: int | None = None,
    ) -> dict[str, Any]:
        """Index all files in a repository.

        Args:
            repository_id: Repository ID
            max_files: Maximum number of files to index

        Returns:
            Indexing results
        """
        # Get Python files
        query = (
            select(File)
            .where(
                File.repository_id == repository_id,
                File.path.endswith(".py"),
                File.is_deleted.is_(False),
            )
            .order_by(File.size)  # Start with smaller files
        )

        if max_files:
            query = query.limit(max_files)

        result = await self.db_session.execute(query)
        files = result.scalars().all()

        logger.info("Indexing %d files from repository %d", len(files), repository_id)

        # Index each file
        results = {
            "total_files": len(files),
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "total_entities": 0,
            "total_relationships": 0,
        }

        for i, file in enumerate(files, 1):
            logger.info("Processing file %d/%d: %s", i, len(files), file.path)

            try:
                file_result = await self.index_file(file.id)

                if file_result["status"] == "success":
                    results["successful"] += 1
                    results["total_entities"] += file_result["entities_extracted"]
                    results["total_relationships"] += file_result[
                        "relationships_extracted"
                    ]
                elif file_result["status"] == "skipped":
                    results["skipped"] += 1
                else:
                    results["failed"] += 1

            except Exception:
                logger.exception("Error indexing file %s", file.path)
                results["failed"] += 1

        # Detect bounded contexts
        logger.info("Detecting bounded contexts...")
        await self.detect_and_save_contexts()

        return results

    async def detect_and_save_contexts(
        self,
        resolution: float = 1.0,
    ) -> list[int]:
        """Detect and save bounded contexts.

        Args:
            resolution: Resolution parameter for community detection

        Returns:
            List of created context IDs
        """
        # Build graph
        logger.info("Building semantic graph...")
        graph = await self.graph_builder.build_graph()

        if graph.number_of_nodes() == 0:
            logger.warning("No domain entities found, skipping context detection")
            return []

        # Detect contexts
        logger.info("Detecting bounded contexts...")
        contexts = await self.graph_builder.detect_bounded_contexts(
            graph,
            resolution=resolution,
        )

        # Save contexts
        saved_contexts = await self.graph_builder.save_bounded_contexts(contexts)

        # Analyze context relationships
        if len(saved_contexts) > 1:
            logger.info("Analyzing context relationships...")
            relationships = await self.graph_builder.analyze_context_relationships(
                graph,
                contexts,
            )

            if relationships:
                logger.info("Saving %d context relationships...", len(relationships))
                await self.graph_builder.save_context_relationships(
                    relationships,
                    saved_contexts,
                )

        return [ctx.id for ctx in saved_contexts]

    async def generate_summaries(
        self,
        entity_type: str = "module",
        entity_ids: list[int] | None = None,
    ) -> int:
        """Generate hierarchical summaries.

        Args:
            entity_type: Type of entity to summarize
            entity_ids: Specific entity IDs, or None for all

        Returns:
            Number of summaries generated
        """
        count = 0

        if entity_type == "module":
            # Get modules to summarize
            query = select(Module)
            if entity_ids:
                query = query.where(Module.id.in_(entity_ids))

            result = await self.db_session.execute(query)
            modules = result.scalars().all()

            for module in modules:
                try:
                    await self.summarizer.summarize_module(module.id)
                    count += 1
                except Exception:
                    logger.exception("Error summarizing module %d", module.id)

        elif entity_type == "context":
            # Get contexts to summarize
            query = select(BoundedContext)
            if entity_ids:
                query = query.where(BoundedContext.id.in_(entity_ids))

            result = await self.db_session.execute(query)
            contexts = result.scalars().all()

            for context in contexts:
                try:
                    await self.summarizer.summarize_bounded_context(context.id)
                    count += 1
                except Exception:
                    logger.exception("Error summarizing context %d", context.id)

        logger.info("Generated %d summaries for %s", count, entity_type)
        return count

    async def _save_domain_entities(
        self,
        entities: list[dict[str, Any]],
        file_id: int,
    ) -> dict[str, DomainEntity]:
        """Save extracted domain entities."""
        saved_entities = {}

        for entity_data in entities:
            # Check if entity already exists
            result = await self.db_session.execute(
                select(DomainEntity).where(
                    DomainEntity.name == entity_data["name"],
                    DomainEntity.entity_type == entity_data["type"],
                ),
            )
            existing = result.scalar_one_or_none()

            if existing:
                ex = cast("Any", existing)
                # Update existing entity
                ex.source_entities = list(
                    {*ex.source_entities, file_id},
                )
                # Merge other fields
                ex.business_rules = list(
                    set(
                        ex.business_rules + entity_data.get("business_rules", []),
                    ),
                )
                ex.invariants = list(
                    set(ex.invariants + entity_data.get("invariants", [])),
                )
                saved_entities[entity_data["name"]] = cast("DomainEntity", ex)
            else:
                # Create new entity
                entity = DomainEntity(
                    name=entity_data["name"],
                    entity_type=entity_data["type"],
                    description=entity_data.get("description", ""),
                    business_rules=entity_data.get("business_rules", []),
                    invariants=entity_data.get("invariants", []),
                    responsibilities=entity_data.get("responsibilities", []),
                    ubiquitous_language=entity_data.get("ubiquitous_language", {}),
                    source_entities=[file_id],
                    confidence_score=entity_data.get("confidence_score", 1.0),
                )
                self.db_session.add(entity)
                saved_entities[entity_data["name"]] = entity

        await self.db_session.flush()  # Get IDs
        return saved_entities

    async def _save_domain_relationships(
        self,
        relationships: list[dict[str, Any]],
        entity_map: dict[str, DomainEntity],
    ) -> list[DomainRelationship]:
        """Save extracted domain relationships."""
        saved_relationships = []

        for rel_data in relationships:
            source_entity = entity_map.get(rel_data["source"])
            target_entity = entity_map.get(rel_data["target"])

            if not source_entity or not target_entity:
                logger.warning(
                    "Skipping relationship %s -> %s: entity not found",
                    rel_data["source"],
                    rel_data["target"],
                )
                continue

            # Check if relationship exists
            result = await self.db_session.execute(
                select(DomainRelationship).where(
                    DomainRelationship.source_entity_id == source_entity.id,
                    DomainRelationship.target_entity_id == target_entity.id,
                    DomainRelationship.relationship_type == rel_data["type"],
                ),
            )
            existing = result.scalar_one_or_none()

            if not existing:
                relationship = DomainRelationship(
                    source_entity_id=source_entity.id,
                    target_entity_id=target_entity.id,
                    relationship_type=rel_data["type"],
                    description=rel_data.get("description", ""),
                    strength=rel_data.get("strength", 1.0),
                    evidence=rel_data.get("evidence", []),
                    interaction_patterns=rel_data.get("interaction_patterns", []),
                    data_flow=rel_data.get("data_flow", {}),
                )
                self.db_session.add(relationship)
                saved_relationships.append(relationship)

        return saved_relationships

    async def _generate_entity_embeddings(
        self,
        entities: dict[str, DomainEntity],
    ) -> None:
        """Generate embeddings for domain entities."""
        for entity in entities.values():
            if entity.concept_embedding is not None:
                continue  # Already has embedding

            # Create text representation
            text_parts = [
                f"Domain Entity: {entity.name}",
                f"Type: {entity.entity_type}",
                f"Description: {entity.description}",
            ]

            if entity.business_rules:
                text_parts.append(f"Business Rules: {', '.join(entity.business_rules)}")
            if entity.invariants:
                text_parts.append(f"Invariants: {', '.join(entity.invariants)}")
            if entity.responsibilities:
                text_parts.append(
                    f"Responsibilities: {', '.join(entity.responsibilities)}",
                )

            # Text would be used for embedding generation
            _ = "\n".join(
                text_parts
            )  # Currently unused, kept for future embedding integration

            # Generate embedding using embedding service
            try:
                from src.embeddings.embedding_service import EmbeddingService

                embedding_service = EmbeddingService(self.db_session)
                await embedding_service.create_domain_entity_embedding(entity.id)
                logger.info("Generated embedding for domain entity: %s", entity.name)
            except Exception:
                logger.exception(
                    "Error generating embedding for %s",
                    entity.name,
                )
