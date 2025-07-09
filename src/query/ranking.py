"""Result ranking and scoring for search queries."""

from datetime import datetime, timezone
from typing import Any

from src.mcp_server.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ResultRanker:
    """Rank and score search results."""

    def __init__(self) -> None:
        self.weights = config.query.ranking_weights

    def rank_results(
        self,
        results: list[dict[str, Any]],
        query: str,
        user_context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Rank search results based on multiple factors."""
        # Calculate additional scores for each result
        for result in results:
            result["scores"] = {
                "semantic": result.get("score", 0.0),
                "recency": self._calculate_recency_score(result),
                "popularity": self._calculate_popularity_score(result),
                "relevance": self._calculate_relevance_score(result, query),
                "context": self._calculate_context_score(result, user_context),
            }

            # Calculate final weighted score
            result["final_score"] = self._calculate_final_score(result["scores"])

        # Sort by final score
        results.sort(key=lambda x: x["final_score"], reverse=True)

        return results

    def _calculate_recency_score(self, result: dict[str, Any]) -> float:
        """Calculate recency score based on last modification time."""
        entity_data = result.get("entity_data", {})

        # Get last modified date (this would come from git metadata)
        last_modified = entity_data.get("last_modified")
        if not last_modified:
            return 0.5  # Default score

        # Calculate age in days
        if isinstance(last_modified, str):
            last_modified = datetime.fromisoformat(last_modified)

        age_days = (datetime.now(tz=timezone.utc) - last_modified).days

        # Score decays over time
        if age_days < 7:
            return 1.0
        if age_days < 30:
            return 0.8
        if age_days < 90:
            return 0.6
        if age_days < 365:
            return 0.4
        return 0.2

    def _calculate_popularity_score(self, result: dict[str, Any]) -> float:
        """Calculate popularity score based on usage and references."""
        entity_data = result.get("entity_data", {})

        # Factors that could indicate popularity:
        # - Number of references/usages
        # - Number of imports
        # - Number of methods (for classes)
        # - File size/complexity

        score = 0.5  # Base score

        # Boost for commonly used names
        name = entity_data.get("name", "").lower()
        common_names = {
            "init",
            "main",
            "get",
            "set",
            "create",
            "update",
            "delete",
            "process",
            "handle",
        }
        if any(common in name for common in common_names):
            score += 0.2

        # Boost for documented code
        if entity_data.get("docstring"):
            score += 0.1

        # Boost for classes with many methods
        if entity_data.get("type") == "class":
            method_count = entity_data.get("method_count", 0)
            if method_count > 10:
                score += 0.2
            elif method_count > 5:
                score += 0.1

        return min(1.0, score)

    def _calculate_relevance_score(self, result: dict[str, Any], query: str) -> float:
        """Calculate relevance based on query terms and context."""
        entity_data = result.get("entity_data", {})

        # Exact name match
        if query.lower() == entity_data.get("name", "").lower():
            return 1.0

        # Partial name match
        name = entity_data.get("name", "").lower()
        if query.lower() in name:
            return 0.8

        # Check if query terms appear in important fields
        query_terms = set(query.lower().split())

        score = 0.0

        # Name matching
        name_terms = set(name.replace("_", " ").split())
        name_overlap = len(query_terms & name_terms) / len(query_terms)
        score += 0.5 * name_overlap

        # Docstring matching
        docstring = entity_data.get("docstring", "").lower()
        if docstring:
            docstring_terms = set(docstring.split())
            doc_overlap = len(query_terms & docstring_terms) / len(query_terms)
            score += 0.3 * doc_overlap

        # Type matching (if query includes type keywords)
        type_keywords = {"function", "class", "method", "module"}
        query_type = query_terms & type_keywords
        if query_type:
            entity_type = entity_data.get("type", "")
            if any(t in entity_type for t in query_type):
                score += 0.2

        return min(1.0, score)

    def _calculate_context_score(
        self,
        result: dict[str, Any],
        user_context: dict[str, Any] | None,
    ) -> float:
        """Calculate score based on user context (e.g., current file, recent searches)."""
        if not user_context:
            return 0.5

        entity_data = result.get("entity_data", {})
        score = 0.5

        # Boost results from the same repository
        current_repo = user_context.get("current_repository")
        if current_repo and entity_data.get("repository_name") == current_repo:
            score += 0.2

        # Boost results from the same file
        current_file = user_context.get("current_file")
        if current_file and entity_data.get("file_path") == current_file:
            score += 0.3

        # Boost based on recent searches
        recent_searches = user_context.get("recent_searches", [])
        for recent_query in recent_searches[-5:]:  # Last 5 searches
            if self._queries_related(recent_query, entity_data.get("name", "")):
                score += 0.1
                break

        return min(1.0, score)

    def _calculate_final_score(self, scores: dict[str, float]) -> float:
        """Calculate final weighted score."""
        # Use configured weights
        final = 0.0

        # Map score types to weight keys
        score_mapping = {
            "semantic": "semantic_similarity",
            "relevance": "keyword_match",
            "recency": "recency",
            "popularity": "popularity",
        }

        for score_type, score_value in scores.items():
            weight_key = score_mapping.get(score_type)
            if weight_key and weight_key in self.weights:
                final += self.weights[weight_key] * score_value
            elif score_type == "context":
                # Context score is a bonus
                final += 0.1 * score_value

        return final

    def _queries_related(self, query1: str, query2: str) -> bool:
        """Check if two queries/terms are related."""
        # Simple check - could be made more sophisticated
        q1_terms = set(query1.lower().split("_"))
        q2_terms = set(query2.lower().split("_"))

        return len(q1_terms & q2_terms) > 0

    def format_results(
        self,
        results: list[dict[str, Any]],
        include_scores: bool = False,
        include_context: bool = True,
    ) -> list[dict[str, Any]]:
        """Format results for presentation."""
        formatted = []

        for result in results:
            entity_data = result.get("entity_data", {})

            formatted_result = {
                "name": entity_data.get("name"),
                "type": entity_data.get("type"),
                "description": self._build_description(entity_data),
                "location": {
                    "repository": entity_data.get("repository_name"),
                    "file": entity_data.get("file_path"),
                    "line": entity_data.get("start_line"),
                },
                "match_type": (
                    "interpreted"
                    if result["similarity"]["interpreted"] > result["similarity"]["raw"]
                    else "raw"
                ),
            }

            if include_scores:
                formatted_result["scores"] = result.get("scores", {})
                formatted_result["final_score"] = result.get("final_score", 0.0)

            if include_context:
                formatted_result["context"] = {
                    "docstring": entity_data.get("docstring"),
                    "matched_content": result.get("matched_content"),
                }

                # Add type-specific context
                if entity_data.get("type") == "function":
                    formatted_result["context"]["parameters"] = entity_data.get(
                        "parameters",
                        [],
                    )
                    formatted_result["context"]["return_type"] = entity_data.get(
                        "return_type",
                    )
                elif entity_data.get("type") == "class":
                    formatted_result["context"]["base_classes"] = entity_data.get(
                        "base_classes",
                        [],
                    )
                    formatted_result["context"]["is_abstract"] = entity_data.get(
                        "is_abstract",
                        False,
                    )

            formatted.append(formatted_result)

        return formatted

    def _build_description(self, entity_data: dict[str, Any]) -> str:
        """Build a concise description of an entity."""
        entity_type = entity_data.get("type", "entity")
        name = entity_data.get("name", "unknown")

        if entity_type in {"function", "method"}:
            params = entity_data.get("parameters", [])
            param_str = f"({len(params)} parameters)" if params else "()"
            return f"{entity_type.capitalize()} {name}{param_str}"
        if entity_type == "class":
            base_classes = entity_data.get("base_classes", [])
            if base_classes:
                return f"Class {name} inheriting from {', '.join(base_classes)}"
            return f"Class {name}"
        if entity_type == "module":
            return f"Module {name}"
        return f"{entity_type.capitalize()} {name}"
