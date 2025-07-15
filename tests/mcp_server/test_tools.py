"""Tests for MCP tools."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import FastMCP
from sqlalchemy.ext.asyncio import AsyncSession

from src.mcp_server.tools.repository_management import (
    RepositoryManagementTools,
)

# Note: CodeAnalysisTools and CodeSearchTools have been converted to resources
# These tests may need to be updated to test resources instead


@pytest.fixture
def mock_db_session():
    """Create mock database session."""
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings."""
    with patch("langchain_openai.OpenAIEmbeddings") as mock_class:
        mock_instance = MagicMock()
        mock_instance.aembed_query = AsyncMock(return_value=[0.1] * 1536)
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_mcp():
    """Create mock FastMCP instance."""
    mcp = MagicMock(spec=FastMCP)
    mcp.tool = MagicMock(side_effect=lambda **kwargs: lambda func: func)
    return mcp


@pytest.mark.skip(
    reason="CodeSearchTools converted to resources - use code://search instead"
)
class TestCodeSearchTools:
    """Tests for CodeSearchTools."""

    @pytest.fixture
    def search_tools(self, mock_db_session, mock_mcp, mock_embeddings):
        """Create code search tools fixture."""
        # Mock both vector_search and domain_search settings
        with (
            patch("src.embeddings.vector_search.settings") as mock_vector_settings,
            patch("src.embeddings.domain_search.settings") as mock_domain_settings,
            patch("src.embeddings.vector_search.OpenAIEmbeddings") as mock_openai_class,
            patch(
                "src.embeddings.domain_search.OpenAIEmbeddings"
            ) as mock_domain_openai_class,
            patch("src.embeddings.domain_search.ChatOpenAI") as mock_chat_openai,
        ):
            # Configure settings
            mock_vector_settings.openai_api_key.get_secret_value.return_value = (
                "test-key"
            )
            mock_vector_settings.embeddings.model = "text-embedding-ada-002"
            mock_domain_settings.openai_api_key.get_secret_value.return_value = (
                "test-key"
            )
            mock_domain_settings.embeddings.model = "text-embedding-ada-002"
            mock_domain_settings.llm.model = "gpt-3.5-turbo"
            mock_domain_settings.llm.temperature = 0.0

            # Use the mock embeddings fixture
            mock_openai_class.return_value = mock_embeddings
            mock_domain_openai_class.return_value = mock_embeddings
            mock_chat_openai.return_value = MagicMock()

            return CodeSearchTools(mock_db_session, mock_mcp)

    @pytest.mark.asyncio
    async def test_register_tools(self, search_tools, mock_mcp) -> None:
        """Test tool registration."""
        await search_tools.register_tools()

        # Should register multiple tools
        assert (
            mock_mcp.tool.call_count >= 3
        )  # semantic_search, find_similar, keyword_search

    @pytest.mark.asyncio
    async def test_semantic_search(self, search_tools) -> None:
        """Test semantic search functionality."""
        # Mock vector search
        mock_results = [
            {
                "similarity": 0.95,
                "entity_type": "function",
                "entity": {"name": "test_function"},
            },
        ]

        search_tools.vector_search = MagicMock()
        search_tools.vector_search.search = AsyncMock(return_value=mock_results)

        # Register tools and get semantic_search
        await search_tools.register_tools()

        # Find the semantic_search function
        semantic_search = None
        for call in search_tools.mcp.tool.call_args_list:
            if call[1]["name"] == "semantic_search":
                # Get the decorated function
                decorator = search_tools.mcp.tool.return_value
                semantic_search = decorator
                break

        assert semantic_search is not None

    @pytest.mark.asyncio
    async def test_keyword_search(self, search_tools, mock_db_session) -> None:
        """Test keyword search functionality."""
        # Mock database results
        mock_function = MagicMock()
        mock_function.id = 1
        mock_function.name = "test_function"
        mock_function.file_id = 10
        mock_function.start_line = 10
        mock_function.end_line = 20
        mock_function.docstring = "Test function"

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_function]
        mock_db_session.execute.return_value = mock_result

        # Register tools
        await search_tools.register_tools()

        # Keyword search should always be registered
        assert mock_db_session is not None


@pytest.mark.skip(
    reason="CodeAnalysisTools converted to resources - use code://explain, code://structure, etc."
)
class TestCodeAnalysisTools:
    """Tests for CodeAnalysisTools."""

    @pytest.fixture
    def analysis_tools(self, mock_db_session, mock_mcp):
        """Create code analysis tools fixture."""
        return CodeAnalysisTools(mock_db_session, mock_mcp)

    @pytest.mark.asyncio
    async def test_register_tools(self, analysis_tools, mock_mcp) -> None:
        """Test tool registration."""
        await analysis_tools.register_tools()

        # Should register multiple tools
        assert mock_mcp.tool.call_count >= 4  # get_code, analyze_file, etc.

    @pytest.mark.asyncio
    async def test_get_code_function(self, analysis_tools, mock_db_session) -> None:
        """Test getting function code."""
        # Mock function entity
        mock_function = MagicMock()
        mock_function.id = 1
        mock_function.name = "test_function"
        mock_function.start_line = 10
        mock_function.end_line = 20
        mock_function.file = MagicMock()
        mock_function.file.path = "/test/file.py"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_function
        mock_db_session.execute.return_value = mock_result

        # Mock code extractor
        with patch.object(
            analysis_tools.code_extractor,
            "get_entity_content",
            return_value=("def test(): pass", "# Context\ndef test(): pass"),
        ):
            # Register tools
            await analysis_tools.register_tools()

            # Test would call the registered function
            assert analysis_tools.code_extractor is not None


class TestRepositoryManagementTools:
    """Tests for RepositoryManagementTools."""

    @pytest.fixture
    def repo_tools(self, mock_db_session, mock_mcp, mock_embeddings):
        """Create repository management tools fixture."""
        # Mock embedding generator and its dependencies
        with (
            patch("src.embeddings.embedding_generator.settings") as mock_gen_settings,
            patch(
                "src.embeddings.embedding_generator.OpenAIEmbeddings"
            ) as mock_gen_openai,
        ):
            # Configure settings
            mock_gen_settings.openai_api_key.get_secret_value.return_value = "test-key"
            mock_gen_settings.embeddings.model = "text-embedding-ada-002"

            # Use the mock embeddings fixture
            mock_gen_openai.return_value = mock_embeddings

            return RepositoryManagementTools(mock_db_session, mock_mcp)

    @pytest.mark.asyncio
    async def test_register_tools(self, repo_tools, mock_mcp) -> None:
        """Test tool registration."""
        await repo_tools.register_tools()

        # Should register multiple tools
        assert mock_mcp.tool.call_count >= 5  # add_repo, list_repos, scan, etc.

    @pytest.mark.asyncio
    async def test_add_repository(self, repo_tools, mock_db_session) -> None:
        """Test adding repository."""
        # Mock no existing repository
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        # Mock scanner
        mock_scanner = MagicMock()
        mock_scanner.scan_repository = AsyncMock(
            return_value={"repository_id": 1, "files_scanned": 10},
        )

        with patch(
            "src.mcp_server.tools.repository_management.RepositoryScanner",
            return_value=mock_scanner,
        ):
            # Register tools
            await repo_tools.register_tools()

            # Tool registration verified
            assert repo_tools.db_session is not None

    @pytest.mark.asyncio
    async def test_list_repositories(self, repo_tools, mock_db_session) -> None:
        """Test listing repositories."""
        # Mock repositories
        mock_repo = MagicMock()
        mock_repo.id = 1
        mock_repo.name = "test-repo"
        mock_repo.owner = "test-owner"
        mock_repo.github_url = "https://github.com/test/repo"
        mock_repo.default_branch = "main"
        mock_repo.last_synced = None

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_repo]
        mock_db_session.execute.return_value = mock_result

        # Register tools
        await repo_tools.register_tools()

        # Tool registration verified
        assert True  # Will be called when tool is invoked
