"""Tests for the ranking module."""

# These tests target a high-level ranking API (RankingCriteria, RankingEngine)
# Provide a thin compatibility layer if needed to satisfy these tests.

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from src.query.ranking import RankingCriteria, RankingEngine


@pytest.fixture
def ranking_engine() -> RankingEngine:
    """Create a ranking engine instance."""
    return RankingEngine()


@pytest.fixture
def sample_search_results() -> list[dict[str, Any]]:
    """Create sample search results for testing."""
    return [
        {
            "file_id": 1,
            "file_path": "src/core/database.py",
            "function_name": "connect_to_database",
            "similarity": 0.95,
            "complexity": 10,
            "lines": 50,
            "last_modified": datetime.now(UTC) - timedelta(days=1),
            "imports": ["psycopg2", "sqlalchemy"],
            "has_docstring": True,
            "is_test": False,
        },
        {
            "file_id": 2,
            "file_path": "tests/test_database.py",
            "function_name": "test_connection",
            "similarity": 0.90,
            "complexity": 3,
            "lines": 20,
            "last_modified": datetime.now(UTC) - timedelta(days=30),
            "imports": ["pytest", "unittest"],
            "has_docstring": False,
            "is_test": True,
        },
        {
            "file_id": 3,
            "file_path": "src/utils/db_helper.py",
            "function_name": "get_connection",
            "similarity": 0.85,
            "complexity": 15,
            "lines": 100,
            "last_modified": datetime.now(UTC),
            "imports": ["psycopg2", "logging"],
            "has_docstring": True,
            "is_test": False,
        },
    ]


class TestRankingEngine:
    """Test cases for RankingEngine class."""

    def test_rank_by_similarity(
        self, ranking_engine: RankingEngine, sample_search_results: list[dict[str, Any]]
    ) -> None:
        """Test ranking by similarity score."""
        # Arrange
        criteria = RankingCriteria(
            similarity_weight=1.0,
            complexity_weight=0.0,
            recency_weight=0.0,
            documentation_weight=0.0,
        )

        # Act
        ranked = ranking_engine.rank_results(sample_search_results, criteria)

        # Assert
        assert len(ranked) == 3
        assert ranked[0]["file_id"] == 1  # Highest similarity
        assert ranked[1]["file_id"] == 2
        assert ranked[2]["file_id"] == 3
        assert ranked[0]["ranking_score"] > ranked[1]["ranking_score"]
        assert ranked[1]["ranking_score"] > ranked[2]["ranking_score"]

    def test_rank_by_complexity(
        self, ranking_engine: RankingEngine, sample_search_results: list[dict[str, Any]]
    ) -> None:
        """Test ranking by code complexity (prefer simpler code)."""
        # Arrange
        criteria = RankingCriteria(
            similarity_weight=0.0,
            complexity_weight=1.0,
            recency_weight=0.0,
            documentation_weight=0.0,
            prefer_simple=True,
        )

        # Act
        ranked = ranking_engine.rank_results(sample_search_results, criteria)

        # Assert
        assert ranked[0]["file_id"] == 2  # Lowest complexity (3)
        assert ranked[1]["file_id"] == 1  # Medium complexity (10)
        assert ranked[2]["file_id"] == 3  # Highest complexity (15)

    def test_rank_by_recency(
        self, ranking_engine: RankingEngine, sample_search_results: list[dict[str, Any]]
    ) -> None:
        """Test ranking by recency (prefer newer code)."""
        # Arrange
        criteria = RankingCriteria(
            similarity_weight=0.0,
            complexity_weight=0.0,
            recency_weight=1.0,
            documentation_weight=0.0,
        )

        # Act
        ranked = ranking_engine.rank_results(sample_search_results, criteria)

        # Assert
        assert ranked[0]["file_id"] == 3  # Most recent (today)
        assert ranked[1]["file_id"] == 1  # Yesterday
        assert ranked[2]["file_id"] == 2  # 30 days ago

    def test_rank_by_documentation(
        self, ranking_engine: RankingEngine, sample_search_results: list[dict[str, Any]]
    ) -> None:
        """Test ranking by documentation quality."""
        # Arrange
        criteria = RankingCriteria(
            similarity_weight=0.0,
            complexity_weight=0.0,
            recency_weight=0.0,
            documentation_weight=1.0,
        )

        # Act
        ranked = ranking_engine.rank_results(sample_search_results, criteria)

        # Assert
        # Files with docstrings should rank higher
        assert ranked[0]["has_docstring"] is True
        assert ranked[1]["has_docstring"] is True
        assert ranked[2]["has_docstring"] is False

    def test_combined_ranking(
        self, ranking_engine: RankingEngine, sample_search_results: list[dict[str, Any]]
    ) -> None:
        """Test ranking with multiple criteria."""
        # Arrange
        criteria = RankingCriteria(
            similarity_weight=0.4,
            complexity_weight=0.2,
            recency_weight=0.2,
            documentation_weight=0.2,
        )

        # Act
        ranked = ranking_engine.rank_results(sample_search_results, criteria)

        # Assert
        assert len(ranked) == 3
        # Each result should have a ranking score
        for result in ranked:
            assert "ranking_score" in result
            assert 0 <= result["ranking_score"] <= 1

    def test_filter_test_files(
        self, ranking_engine: RankingEngine, sample_search_results: list[dict[str, Any]]
    ) -> None:
        """Test filtering out test files."""
        # Arrange
        criteria = RankingCriteria(exclude_tests=True)

        # Act
        ranked = ranking_engine.rank_results(sample_search_results, criteria)

        # Assert
        assert len(ranked) == 2
        assert all(not r["is_test"] for r in ranked)

    def test_minimum_similarity_threshold(
        self, ranking_engine: RankingEngine, sample_search_results: list[dict[str, Any]]
    ) -> None:
        """Test filtering by minimum similarity threshold."""
        # Arrange
        criteria = RankingCriteria(min_similarity=0.9)

        # Act
        ranked = ranking_engine.rank_results(sample_search_results, criteria)

        # Assert
        assert len(ranked) == 2
        assert all(r["similarity"] >= 0.9 for r in ranked)

    def test_file_type_preference(
        self, ranking_engine: RankingEngine, sample_search_results: list[dict[str, Any]]
    ) -> None:
        """Test preferring certain file types."""
        # Arrange
        criteria = RankingCriteria(
            preferred_paths=["src/core/"], path_weight=0.5, similarity_weight=0.5
        )

        # Act
        ranked = ranking_engine.rank_results(sample_search_results, criteria)

        # Assert
        # src/core/database.py should rank higher due to path preference
        assert ranked[0]["file_path"] == "src/core/database.py"

    def test_boost_by_imports(
        self, ranking_engine: RankingEngine, sample_search_results: list[dict[str, Any]]
    ) -> None:
        """Test boosting results that use specific imports."""
        # Arrange
        criteria = RankingCriteria(
            boost_imports=["sqlalchemy"], import_boost_factor=2.0
        )

        # Act
        ranked = ranking_engine.rank_results(sample_search_results, criteria)

        # Assert
        # Result with sqlalchemy import should rank higher
        assert ranked[0]["file_id"] == 1
        assert "sqlalchemy" in ranked[0]["imports"]

    def test_empty_results(self, ranking_engine: RankingEngine) -> None:
        """Test ranking empty results."""
        # Act
        ranked = ranking_engine.rank_results([], RankingCriteria())

        # Assert
        assert ranked == []

    def test_single_result(self, ranking_engine: RankingEngine) -> None:
        """Test ranking a single result."""
        # Arrange
        single_result = [{"file_id": 1, "similarity": 0.9, "complexity": 5}]

        # Act
        ranked = ranking_engine.rank_results(single_result, RankingCriteria())

        # Assert
        assert len(ranked) == 1
        assert ranked[0]["file_id"] == 1
        assert "ranking_score" in ranked[0]

    def test_normalize_scores(self, ranking_engine: RankingEngine) -> None:
        """Test score normalization."""
        # Arrange
        results = [
            {"file_id": 1, "similarity": 0.5, "lines": 1000},
            {"file_id": 2, "similarity": 0.7, "lines": 100},
            {"file_id": 3, "similarity": 0.9, "lines": 10},
        ]
        criteria = RankingCriteria(
            similarity_weight=0.5, size_weight=0.5, prefer_concise=True
        )

        # Act
        ranked = ranking_engine.rank_results(results, criteria)

        # Assert
        # All scores should be between 0 and 1
        for result in ranked:
            assert 0 <= result["ranking_score"] <= 1

    def test_custom_scoring_function(
        self, ranking_engine: RankingEngine, sample_search_results: list[dict[str, Any]]
    ) -> None:
        """Test using a custom scoring function."""

        # Arrange
        def custom_scorer(result: dict[str, Any]) -> float:
            # Prefer files with exactly 50 lines
            lines_val = float(result.get("lines", 0) or 0)
            distance_from_50 = abs(lines_val - 50.0)
            return 1.0 / (1.0 + distance_from_50)

        criteria = RankingCriteria(custom_scorer=custom_scorer)

        # Act
        ranked = ranking_engine.rank_results(sample_search_results, criteria)

        # Assert
        # File with 50 lines should rank first
        assert ranked[0]["lines"] == 50

    def test_diversity_ranking(self, ranking_engine: RankingEngine) -> None:
        """Test promoting diversity in results."""
        # Arrange
        results = [
            {"file_id": 1, "file_path": "src/db/conn.py", "similarity": 0.95},
            {"file_id": 2, "file_path": "src/db/pool.py", "similarity": 0.94},
            {"file_id": 3, "file_path": "src/api/handler.py", "similarity": 0.93},
            {"file_id": 4, "file_path": "src/db/manager.py", "similarity": 0.92},
        ]

        criteria = RankingCriteria(promote_diversity=True, diversity_factor=0.3)

        # Act
        ranked = ranking_engine.rank_results(results, criteria)

        # Assert
        # Results from different directories should be interleaved
        paths = [r["file_path"] for r in ranked[:2]]
        assert not all("src/db/" in p for p in paths)

    def test_invalid_weights(self, ranking_engine: RankingEngine) -> None:
        """Test handling of invalid weight values."""
        # Arrange
        criteria = RankingCriteria(
            similarity_weight=-0.5,  # Invalid negative weight
            complexity_weight=2.0,  # Weight > 1
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Weights must be between 0 and 1"):
            ranking_engine.validate_criteria(criteria)

    def test_performance_with_large_dataset(
        self, ranking_engine: RankingEngine
    ) -> None:
        """Test ranking performance with many results."""
        # Arrange
        large_results = [
            {
                "file_id": i,
                "similarity": 0.9 - (i * 0.0001),
                "complexity": i % 20,
                "lines": (i * 10) % 500,
                "last_modified": datetime.now(UTC) - timedelta(days=i % 365),
            }
            for i in range(1000)
        ]

        criteria = RankingCriteria(
            similarity_weight=0.4, complexity_weight=0.3, recency_weight=0.3
        )

        # Act
        import time

        start_time = time.time()
        ranked = ranking_engine.rank_results(large_results, criteria)
        end_time = time.time()

        # Assert
        assert len(ranked) == 1000
        assert end_time - start_time < 1.0  # Should complete within 1 second
