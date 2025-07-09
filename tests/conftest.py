"""Shared pytest fixtures and configuration."""

import asyncio
import tempfile
from collections.abc import AsyncGenerator, Generator
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import Base
from src.mcp_server.config import Settings
from src.utils.logger import setup_logging


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_config_file() -> Generator[Path, None, None]:
    """Create a temporary test configuration file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(
            """
repositories:
  - url: https://github.com/test/repo1
    branch: main
  - url: https://github.com/test/repo2
    branch: develop

scanner:
  sync_interval: 60
  storage_path: /tmp/test_repos

embeddings:
  model: text-embedding-ada-002
  batch_size: 10
  use_cache: true
  cache_dir: /tmp/test_cache

database:
  host: localhost
  port: 5432
  database: test_code_analysis
  user: test_user

logging:
  level: DEBUG
  format: json
  file_enabled: false
  console_enabled: true
""",
        )
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def test_settings(test_config_file, monkeypatch) -> Settings:
    """Create test settings."""
    # Set required environment variables
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://test_user:test_pass@localhost:5432/test_code_analysis",
    )
    monkeypatch.setenv("CONFIG_PATH", str(test_config_file))

    # Clear singleton
    from src.mcp_server import config

    config._settings = None

    return Settings.from_yaml(test_config_file)


@pytest.fixture
def test_db_url() -> str:
    """Get test database URL."""
    return "sqlite:///:memory:"


@pytest.fixture
def sync_engine(test_db_url):
    """Create a synchronous SQLAlchemy engine for testing."""
    engine = create_engine(test_db_url)
    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest_asyncio.fixture
async def async_engine(test_db_url):
    """Create an asynchronous SQLAlchemy engine for testing."""
    # Convert to async URL
    if test_db_url.startswith("sqlite"):
        async_url = test_db_url.replace("sqlite://", "sqlite+aiosqlite://")
    else:
        async_url = test_db_url.replace("postgresql://", "postgresql+asyncpg://")

    engine = create_async_engine(async_url)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()


@pytest_asyncio.fixture
async def async_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create an async database session for testing."""
    async_session_maker = sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session_maker() as session:
        yield session
        await session.rollback()


@pytest.fixture
def mock_github_response():
    """Mock GitHub API response."""
    return {
        "rate": {
            "limit": 5000,
            "remaining": 4999,
            "reset": 1234567890,
        },
    }


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return {
        "data": [
            {"id": "text-embedding-ada-002"},
            {"id": "text-embedding-3-small"},
            {"id": "text-embedding-3-large"},
        ],
    }


@pytest.fixture
def sample_python_code() -> str:
    """Sample Python code for testing."""
    return '''
"""Sample module for testing."""

import os
from typing import List, Optional


class SampleClass:
    """A sample class for testing."""

    def __init__(self, name: str):
        self.name = name

    def greet(self, greeting: str = "Hello") -> str:
        """Return a greeting message."""
        return f"{greeting}, {self.name}!"

    @property
    def uppercase_name(self) -> str:
        """Get name in uppercase."""
        return self.name.upper()


def sample_function(items: List[str]) -> Optional[str]:
    """Process a list of items."""
    if not items:
        return None
    return ", ".join(items)


async def async_sample(value: int) -> int:
    """An async function example."""
    return value * 2
'''


@pytest.fixture(autouse=True)
def setup_test_logging() -> None:
    """Set up logging for tests."""
    setup_logging()


@pytest.fixture
def temp_repo_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for repository testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_path = Path(tmp_dir) / "test_repo"
        repo_path.mkdir()
        yield repo_path
