"""Result ranking and scoring for search queries."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.config import settings
from src.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)

# Age thresholds in days
AGE_VERY_RECENT = 7
AGE_RECENT = 30


class WeightValueError(ValueError):
    """Raised when ranking weight values are out of the allowed range [0, 1]."""

    def __init__(self) -> None:
        super().__init__("Weights must be between 0 and 1")


AGE_QUARTER = 90


class RankingCriteria:
    """Compatibility criteria to match tests expectations.

    This is a minimal data container with optional weights and flags used by
    RankingEngine. Defaults are permissive and 0-weighted where appropriate.
    """

    def __init__(
        self,
        *,
        similarity_weight: float = 0.0,
        complexity_weight: float = 0.0,
        recency_weight: float = 0.0,
        documentation_weight: float = 0.0,
        size_weight: float = 0.0,
        prefer_simple: bool = False,
        prefer_concise: bool = False,
        exclude_tests: bool = False,
        min_similarity: float | None = None,
        preferred_paths: list[str] | None = None,
        path_weight: float = 0.0,
        boost_imports: list[str] | None = None,
        import_boost_factor: float = 1.0,
        promote_diversity: bool = False,
        diversity_factor: float = 0.0,
        custom_scorer: Callable[[dict[str, Any]], float] | None = None,
    ) -> None:
        self.similarity_weight = similarity_weight
        self.complexity_weight = complexity_weight
        self.recency_weight = recency_weight
        self.documentation_weight = documentation_weight
        self.size_weight = size_weight
        self.prefer_simple = prefer_simple
        self.prefer_concise = prefer_concise
        self.exclude_tests = exclude_tests
        self.min_similarity = min_similarity
        self.preferred_paths = preferred_paths or []
        self.path_weight = path_weight
        self.boost_imports = boost_imports or []
        self.import_boost_factor = import_boost_factor
        self.promote_diversity = promote_diversity
        self.diversity_factor = diversity_factor
        self.custom_scorer = custom_scorer


class RankingEngine:
    """Compatibility engine to satisfy tests that exercise ranking behaviors.

    This is a self-contained ranker operating on plain dict rows, independent of
    the more complex ResultRanker above.
    """

    @staticmethod
    def validate_criteria(criteria: RankingCriteria) -> None:
        for attr in (
            "similarity_weight",
            "complexity_weight",
            "recency_weight",
            "documentation_weight",
            "size_weight",
            "path_weight",
            "diversity_factor",
        ):
            val = getattr(criteria, attr)
            if val < 0 or val > 1:
                raise WeightValueError

    def _normalize(self, values: list[float]) -> list[float]:
        if not values:
            return []
        vmin, vmax = min(values), max(values)
        if vmax == vmin:
            return [0.0 for _ in values]
        return [(v - vmin) / (vmax - vmin) for v in values]

    def _compute_scores(
        self, results: list[dict[str, Any]], criteria: RankingCriteria
    ) -> list[float]:
        """Compute ranking scores for results with modular helpers to reduce complexity."""

        def norm_metric(key: str) -> list[float]:
            return self._normalize([float(r.get(key, 0.0)) for r in results])

        def recency_norm() -> list[float]:
            now = datetime.now(tz=UTC)
            raw: list[float] = []
            for r in results:
                last = r.get("last_modified")
                if isinstance(last, datetime):
                    # Normalize naive datetimes by assuming UTC
                    if last.tzinfo is None:
                        last = last.replace(tzinfo=UTC)
                    raw.append(-float((now - last).total_seconds()))
                else:
                    raw.append(float("-inf"))
            return self._normalize(raw)

        sim_n = norm_metric("similarity")
        comp_n = norm_metric("complexity")
        if criteria.prefer_simple:
            comp_n = [1.0 - c for c in comp_n]
        size_n = norm_metric("lines")
        if criteria.prefer_concise:
            size_n = [1.0 - s for s in size_n]
        recency_n = recency_norm()

        doc_scores = [1.0 if r.get("has_docstring") else 0.0 for r in results]

        path_scores = [0.0 for _ in results]
        if criteria.preferred_paths:
            for i, r in enumerate(results):
                fp = r.get("file_path") or r.get("path", "")
                path_scores[i] = (
                    1.0 if any(str(p) in fp for p in criteria.preferred_paths) else 0.0
                )

        import_scores = [0.0 for _ in results]
        if criteria.boost_imports:
            for i, r in enumerate(results):
                imps = set(r.get("imports", []))
                import_scores[i] = (
                    criteria.import_boost_factor
                    if any(b in imps for b in criteria.boost_imports)
                    else 0.0
                )

        # Determine effective weights; if caller provided no weights at all and no custom/boosts
        # fall back to similarity as the default signal.
        weight_sum = (
            criteria.similarity_weight
            + criteria.complexity_weight
            + criteria.recency_weight
            + criteria.documentation_weight
            + criteria.size_weight
            + criteria.path_weight
        )
        use_similarity_default = (
            weight_sum == 0.0
            and not criteria.custom_scorer
            and not criteria.boost_imports
        )
        sim_w = 1.0 if use_similarity_default else criteria.similarity_weight

        base_scores: list[float] = []
        for i in range(len(results)):
            base = 0.0
            base += sim_w * sim_n[i]
            base += criteria.complexity_weight * comp_n[i]
            base += criteria.recency_weight * recency_n[i]
            base += criteria.documentation_weight * doc_scores[i]
            base += criteria.size_weight * size_n[i]
            base += criteria.path_weight * path_scores[i]
            base += import_scores[i]

            if criteria.custom_scorer:
                from contextlib import suppress

                with suppress(Exception):
                    base += float(criteria.custom_scorer(results[i]))

            base_scores.append(base)

        if not criteria.promote_diversity:
            return base_scores

        # Diversity-aware greedy adjustment: pick items one by one, penalizing repeated directories
        from math import inf

        remaining = list(range(len(results)))
        seen_dirs: dict[str, int] = {}
        adjusted: list[float] = [0.0 for _ in results]
        order: list[int] = []

        while remaining:
            best_idx = -1
            best_score = -inf
            for i in remaining:
                fp = results[i].get("file_path") or results[i].get("path", "")
                directory = "/".join(str(fp).split("/")[:2]) if fp else ""
                count = seen_dirs.get(directory, 0)
                # Apply stronger penalty only for repeats of an already seen directory
                penalty = 0.0
                if directory and count > 0:
                    penalty = min((count + 1) * criteria.diversity_factor, 0.9)
                score = base_scores[i] * (1.0 - penalty)
                if score > best_score:
                    best_score = score
                    best_idx = i
            # select best_idx
            order.append(best_idx)
            adjusted[best_idx] = best_score
            fp_sel = results[best_idx].get("file_path") or results[best_idx].get(
                "path", ""
            )
            directory_sel = "/".join(str(fp_sel).split("/")[:2]) if fp_sel else ""
            seen_dirs[directory_sel] = seen_dirs.get(directory_sel, 0) + 1
            remaining.remove(best_idx)

        return adjusted

    def rank_results(
        self, results: list[dict[str, Any]], criteria: RankingCriteria
    ) -> list[dict[str, Any]]:
        # Validate input
        self.validate_criteria(criteria)

        # Filter
        filtered = [
            r
            for r in results
            if (
                not criteria.exclude_tests
                or not str(r.get("file_path", "")).startswith("tests/")
            )
            and (
                criteria.min_similarity is None
                or float(r.get("similarity", 0.0)) >= criteria.min_similarity
            )
        ]
        if not filtered:
            return []

        # Score
        scores = self._compute_scores(filtered, criteria)
        for r, s in zip(filtered, scores, strict=False):
            r["ranking_score"] = max(0.0, min(1.0, s))

        # Sort
        filtered.sort(key=lambda r: r["ranking_score"], reverse=True)
        return filtered


AGE_YEAR = 365

# Complexity thresholds
HIGH_COMPLEXITY = 10
MEDIUM_COMPLEXITY = 5


class ResultRanker:
    """Rank and score search results."""

    def __init__(self) -> None:
        self.weights = settings.query.ranking_weights

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

        age_days = (datetime.now(tz=UTC) - last_modified).days

        # Score decays over time
        if age_days < AGE_VERY_RECENT:
            return 1.0
        if age_days < AGE_RECENT:
            return 0.8
        if age_days < AGE_QUARTER:
            return 0.6
        if age_days < AGE_YEAR:
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
            if method_count > HIGH_COMPLEXITY:
                score += 0.2
            elif method_count > MEDIUM_COMPLEXITY:
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
        *,
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
