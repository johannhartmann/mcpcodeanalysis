"""Search engine for code queries."""

import time
from typing import Any

from src.database import get_repositories, get_session
from src.database.models import Class, File, Function, Module
from src.indexer.embeddings import EmbeddingGenerator
from src.mcp_server.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SearchEngine:
    """Search engine for code entities using embeddings."""

    def __init__(self) -> None:
        self.embedding_generator = EmbeddingGenerator()
        self.query_config = config.query

    async def search(
        self,
        query: str,
        limit: int | None = None,
        entity_types: list[str] | None = None,
        repository_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for code entities matching a query."""
        start_time = time.time()

        # Set default limit
        if limit is None:
            limit = self.query_config.default_limit
        limit = min(limit, self.query_config.max_limit)

        # Generate query embedding
        query_embedding = await self.embedding_generator.generate_embedding(query)

        async with get_session() as session, get_repositories(session) as repos:
            # Search both raw and interpreted embeddings
            raw_results = await repos["embedding"].search_similar(
                query_embedding,
                limit=limit * 2,  # Get more results for merging
                threshold=self.query_config.similarity_threshold,
                embedding_type="raw",
                entity_types=entity_types,
            )

            interpreted_results = await repos["embedding"].search_similar(
                query_embedding,
                limit=limit * 2,
                threshold=self.query_config.similarity_threshold,
                embedding_type="interpreted",
                entity_types=entity_types,
            )

            # Merge and rank results
            results = await self._merge_and_rank_results(
                raw_results,
                interpreted_results,
                query,
                repos,
                limit,
                repository_filter,
            )

            # Log query for analytics
            execution_time = (time.time() - start_time) * 1000
            await repos["search_query"].log_query(
                query=query,
                tool_name="search_code",
                results_count=len(results),
                execution_time_ms=execution_time,
                metadata={
                    "entity_types": entity_types,
                    "repository_filter": repository_filter,
                },
            )

            return results

    async def _merge_and_rank_results(
        self,
        raw_results: list[tuple[Any, float]],
        interpreted_results: list[tuple[Any, float]],
        query: str,
        repos: dict[str, Any],
        limit: int,
        repository_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Merge and rank search results."""
        # Create a map of entity results
        entity_scores: dict[tuple[str, int], dict[str, Any]] = {}

        # Process raw results
        for embedding, similarity in raw_results:
            key = (embedding.entity_type, embedding.entity_id)
            if key not in entity_scores:
                entity_scores[key] = {
                    "entity_type": embedding.entity_type,
                    "entity_id": embedding.entity_id,
                    "raw_similarity": similarity,
                    "interpreted_similarity": 0.0,
                    "raw_content": embedding.content,
                    "metadata": embedding.metadata,
                }
            else:
                entity_scores[key]["raw_similarity"] = max(
                    entity_scores[key]["raw_similarity"],
                    similarity,
                )

        # Process interpreted results
        for embedding, similarity in interpreted_results:
            key = (embedding.entity_type, embedding.entity_id)
            if key not in entity_scores:
                entity_scores[key] = {
                    "entity_type": embedding.entity_type,
                    "entity_id": embedding.entity_id,
                    "raw_similarity": 0.0,
                    "interpreted_similarity": similarity,
                    "interpreted_content": embedding.content,
                    "metadata": embedding.metadata,
                }
            else:
                entity_scores[key]["interpreted_similarity"] = similarity
                entity_scores[key]["interpreted_content"] = embedding.content

        # Load entity details and calculate final scores
        results = []

        for (entity_type, entity_id), scores in entity_scores.items():
            # Load entity details
            entity_data = await self._load_entity_details(repos, entity_type, entity_id)

            if not entity_data:
                continue

            # Apply repository filter if specified
            if (
                repository_filter
                and entity_data.get("repository_name") != repository_filter
            ):
                continue

            # Calculate composite score
            weights = self.query_config.ranking_weights
            final_score = weights["semantic_similarity"] * max(
                scores["raw_similarity"],
                scores["interpreted_similarity"],
            ) + weights["keyword_match"] * self._calculate_keyword_match(
                query,
                entity_data.get("name", ""),
                entity_data.get("docstring", ""),
            )

            # Build result
            result = {
                "score": final_score,
                "entity_type": entity_type,
                "entity_data": entity_data,
                "similarity": {
                    "raw": scores["raw_similarity"],
                    "interpreted": scores["interpreted_similarity"],
                },
                "matched_content": scores.get("interpreted_content")
                or scores.get("raw_content"),
            }

            results.append(result)

        # Sort by score and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    async def _load_entity_details(
        self,
        repos: dict[str, Any],
        entity_type: str,
        entity_id: int,
    ) -> dict[str, Any] | None:
        """Load detailed information about an entity."""
        session = repos["repository"].session

        try:
            if entity_type == "function":
                # Load function with related data
                func = await session.get(Function, entity_id)
                if not func:
                    return None

                # Load related module and file
                module = await session.get(Module, func.module_id)
                file = await session.get(File, module.file_id) if module else None
                repo = (
                    await repos["repository"].get_by_id(file.repository_id)
                    if file
                    else None
                )

                return {
                    "id": func.id,
                    "name": func.name,
                    "type": "method" if func.class_id else "function",
                    "parameters": func.parameters,
                    "return_type": func.return_type,
                    "docstring": func.docstring,
                    "is_async": func.is_async,
                    "is_generator": func.is_generator,
                    "start_line": func.start_line,
                    "end_line": func.end_line,
                    "file_path": file.path if file else None,
                    "repository_name": repo.name if repo else None,
                    "repository_url": repo.github_url if repo else None,
                }

            if entity_type == "class":
                # Load class with related data
                cls = await session.get(Class, entity_id)
                if not cls:
                    return None

                module = await session.get(Module, cls.module_id)
                file = await session.get(File, module.file_id) if module else None
                repo = (
                    await repos["repository"].get_by_id(file.repository_id)
                    if file
                    else None
                )

                return {
                    "id": cls.id,
                    "name": cls.name,
                    "type": "class",
                    "base_classes": cls.base_classes,
                    "docstring": cls.docstring,
                    "is_abstract": cls.is_abstract,
                    "start_line": cls.start_line,
                    "end_line": cls.end_line,
                    "file_path": file.path if file else None,
                    "repository_name": repo.name if repo else None,
                    "repository_url": repo.github_url if repo else None,
                }

            if entity_type == "module":
                # Load module with related data
                module = await session.get(Module, entity_id)
                if not module:
                    return None

                file = await session.get(File, module.file_id)
                repo = (
                    await repos["repository"].get_by_id(file.repository_id)
                    if file
                    else None
                )

                return {
                    "id": module.id,
                    "name": module.name,
                    "type": "module",
                    "docstring": module.docstring,
                    "file_path": file.path if file else None,
                    "repository_name": repo.name if repo else None,
                    "repository_url": repo.github_url if repo else None,
                }

            return None

        except Exception as e:
            logger.exception(f"Error loading entity details: {e}")
            return None

    def _calculate_keyword_match(self, query: str, name: str, docstring: str) -> float:
        """Calculate keyword match score."""
        query_words = set(query.lower().split())

        # Check name match
        name_words = set(name.lower().split("_"))
        name_match = len(query_words & name_words) / len(query_words)

        # Check docstring match
        docstring_match = 0.0
        if docstring:
            docstring_words = set(docstring.lower().split())
            docstring_match = len(query_words & docstring_words) / len(query_words)

        # Weight name match higher
        return 0.7 * name_match + 0.3 * docstring_match
