"""Tests for MCP server."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import FastMCP
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from src.mcp_server.server import MCPCodeAnalysisServer, create_server


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = MagicMock()
    settings.openai_api_key.get_secret_value.return_value = "test-key"
    settings.mcp.host = "0.0.0.0"
    settings.mcp.port = 8080
    settings.database.url = "postgresql+asyncpg://test@localhost/test"
    return settings


@pytest.fixture
def mock_engine():
    """Create mock database engine."""
    engine = AsyncMock(spec=AsyncEngine)
    engine.dispose = AsyncMock()
    return engine


@pytest.fixture
def mock_session():
    """Create mock database session."""
    session = AsyncMock(spec=AsyncSession)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock()
    return session


@pytest.fixture
def mcp_server(mock_settings):
    """Create MCP server fixture."""
    with patch("src.mcp_server.server.get_settings", return_value=mock_settings):
        return MCPCodeAnalysisServer()


class TestMCPCodeAnalysisServer:
    """Tests for MCPCodeAnalysisServer class."""
    
    def test_init(self, mcp_server, mock_settings):
        """Test server initialization."""
        assert mcp_server.settings == mock_settings
        assert isinstance(mcp_server.mcp, FastMCP)
        assert mcp_server.engine is None
        assert mcp_server.session_factory is None
    
    @pytest.mark.asyncio
    async def test_startup_success(self, mcp_server, mock_engine, mock_session):
        """Test successful server startup."""
        with patch("src.mcp_server.server.init_database", return_value=mock_engine):
            with patch("src.mcp_server.server.get_session_factory") as mock_factory:
                mock_factory.return_value = AsyncMock(return_value=mock_session)
                
                with patch.object(mcp_server, "_initialize_tools") as mock_init_tools:
                    with patch.object(
                        mcp_server.openai_client,
                        "test_connection",
                        return_value=True
                    ):
                        await mcp_server._startup()
                        
                        assert mcp_server.engine == mock_engine
                        assert mcp_server.session_factory is not None
                        mock_init_tools.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_startup_openai_failure(self, mcp_server, mock_engine, mock_session):
        """Test server startup with OpenAI connection failure."""
        with patch("src.mcp_server.server.init_database", return_value=mock_engine):
            with patch("src.mcp_server.server.get_session_factory") as mock_factory:
                mock_factory.return_value = AsyncMock(return_value=mock_session)
                
                with patch.object(mcp_server, "_initialize_tools"):
                    # Create OpenAI client first
                    mcp_server.openai_client = MagicMock()
                    mcp_server.openai_client.test_connection = AsyncMock(return_value=False)
                    
                    await mcp_server._startup()
                    
                    # Should still start but log warning
                    assert mcp_server.engine == mock_engine
    
    @pytest.mark.asyncio
    async def test_shutdown(self, mcp_server, mock_engine):
        """Test server shutdown."""
        mcp_server.engine = mock_engine
        
        await mcp_server._shutdown()
        
        mock_engine.dispose.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_session(self, mcp_server, mock_session):
        """Test getting database session."""
        mcp_server.session_factory = AsyncMock(return_value=mock_session)
        
        async with mcp_server.get_session() as session:
            assert session == mock_session
    
    @pytest.mark.asyncio
    async def test_get_session_not_initialized(self, mcp_server):
        """Test getting session when not initialized."""
        with pytest.raises(RuntimeError, match="Database not initialized"):
            async with mcp_server.get_session():
                pass
    
    def test_create_app(self, mcp_server):
        """Test creating FastMCP app."""
        app = mcp_server.create_app()
        
        assert isinstance(app, FastMCP)
        assert app.name == "Code Analysis Server"
        assert app.version == "0.1.0"
        assert app.capabilities["tools"] is True
    
    @pytest.mark.asyncio
    async def test_scan_repository(self, mcp_server, mock_session):
        """Test repository scanning."""
        mcp_server.session_factory = AsyncMock(return_value=mock_session)
        
        mock_scanner = MagicMock()
        mock_scanner.scan_repository = AsyncMock(
            return_value={
                "repository_id": 1,
                "files_scanned": 10,
                "files_parsed": 8,
            }
        )
        
        with patch("src.mcp_server.server.RepositoryScanner", return_value=mock_scanner):
            result = await mcp_server.scan_repository("https://github.com/test/repo")
            
            assert result["repository_id"] == 1
            assert result["files_scanned"] == 10
            mock_scanner.scan_repository.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_scan_repository_with_embeddings(self, mcp_server, mock_session):
        """Test repository scanning with embedding generation."""
        mcp_server.session_factory = AsyncMock(return_value=mock_session)
        mcp_server.openai_client = MagicMock()
        
        mock_scanner = MagicMock()
        mock_scanner.scan_repository = AsyncMock(
            return_value={
                "repository_id": 1,
                "files_parsed": 5,
            }
        )
        
        mock_embedding_service = MagicMock()
        mock_embedding_service.create_repository_embeddings = AsyncMock(
            return_value={"total_embeddings": 20}
        )
        
        with patch("src.mcp_server.server.RepositoryScanner", return_value=mock_scanner):
            with patch(
                "src.mcp_server.server.EmbeddingService",
                return_value=mock_embedding_service
            ):
                result = await mcp_server.scan_repository("https://github.com/test/repo")
                
                assert "embeddings" in result
                assert result["embeddings"]["total_embeddings"] == 20
    
    @pytest.mark.asyncio
    async def test_search_with_vector_search(self, mcp_server, mock_session):
        """Test search with vector search available."""
        mcp_server.session_factory = AsyncMock(return_value=mock_session)
        mcp_server.openai_client = MagicMock()
        
        mock_vector_search = MagicMock()
        mock_vector_search.search = AsyncMock(
            return_value=[
                {"entity": {"name": "test_function"}, "similarity": 0.95}
            ]
        )
        
        with patch("src.mcp_server.server.VectorSearch", return_value=mock_vector_search):
            results = await mcp_server.search("find test function")
            
            assert len(results) == 1
            assert results[0]["entity"]["name"] == "test_function"
            mock_vector_search.search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_without_vector_search(self, mcp_server, mock_session):
        """Test search when vector search is not available."""
        mcp_server.session_factory = AsyncMock(return_value=mock_session)
        mcp_server.openai_client = None
        
        results = await mcp_server.search("test query")
        
        # Should return empty results (keyword search not implemented)
        assert results == []


def test_create_server():
    """Test server creation."""
    with patch("src.mcp_server.server.get_settings"):
        server = create_server()
        assert isinstance(server, MCPCodeAnalysisServer)