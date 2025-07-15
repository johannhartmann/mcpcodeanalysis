"""MCP Code Analysis Server entry point."""

import asyncio
import logging
import sys
from pathlib import Path

import click

from src.config import settings
from src.logger import setup_logging

# Removed create_server import - using initialize_server instead

# Default server configuration
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080


@click.group()
def cli() -> None:
    """MCP Code Analysis Server CLI."""


@cli.command()
@click.option(
    "--host",
    default="127.0.0.1",
    help="Host to bind to",
)
@click.option(
    "--port",
    type=int,
    default=8080,
    help="Port to bind to",
)
@click.option(
    "--reload",
    is_flag=True,
    help="Enable auto-reload for development",
)
@click.option(
    "--log-level",
    type=click.Choice(["debug", "info", "warning", "error"]),
    default="info",
    help="Logging level",
)
def serve(host: str, port: int, reload: bool, log_level: str) -> None:  # noqa: ARG001
    """Start the MCP server."""
    # Setup logging
    setup_logging()
    # Override log level if specified
    if log_level:
        logging.getLogger().setLevel(log_level.upper())

    # Import MCP server here to avoid circular imports
    from src.mcp_server.server import initialize_server, mcp

    # Override settings with CLI options if provided
    if host != DEFAULT_HOST:
        settings.mcp.host = host
    if port != DEFAULT_PORT:
        settings.mcp.port = port

    # Initialize server with resources before running
    asyncio.run(initialize_server())

    # Run the MCP server with streamable HTTP transport (standard MCP protocol)
    mcp.run(transport="streamable-http", host=host, port=port)


@cli.command()
@click.argument("repository_url")
@click.option(
    "--branch",
    help="Branch to scan",
)
@click.option(
    "--embeddings/--no-embeddings",
    default=True,
    help="Generate embeddings after scanning",
)
def scan(repository_url: str, branch: str, embeddings: bool) -> None:
    """Scan a repository."""

    async def _scan() -> None:
        # Setup logging
        setup_logging()
        logging.getLogger().setLevel("INFO")

        # Initialize server
        from src.mcp_server.server import initialize_server, mcp

        await initialize_server()

        try:
            # Use the add_repository tool directly
            from src.mcp_server.server import add_repository

            result = await add_repository(
                url=repository_url,
                branch=branch,
                scan_immediately=True,
                generate_embeddings=embeddings,
            )

            # Print results
            click.echo("Repository scanned successfully!")
            click.echo(f"Repository ID: {result['repository_id']}")
            click.echo(f"Files scanned: {result['files_scanned']}")
            click.echo(f"Files parsed: {result.get('files_parsed', 0)}")

            if "embeddings" in result:
                click.echo(
                    f"Embeddings created: {result['embeddings']['total_embeddings']}",
                )

        finally:
            # Cleanup is handled by context managers
            pass

    asyncio.run(_scan())


@cli.command()
@click.argument("query")
@click.option(
    "--repository-id",
    type=int,
    help="Repository ID to search in",
)
@click.option(
    "--limit",
    type=int,
    default=10,
    help="Maximum number of results",
)
def search(query: str, repository_id: int, limit: int) -> None:
    """Search for code."""

    async def _search() -> None:
        # Setup logging
        setup_logging()
        logging.getLogger().setLevel("INFO")

        # Initialize server
        from src.mcp_server.server import initialize_server, mcp

        await initialize_server()

        try:
            # Search using code search resource
            from src.mcp_server.resources.code_analysis import CodeAnalysisResources
            from src.mcp_server.server import _session_factory

            code_resources = CodeAnalysisResources(mcp, _session_factory)
            results = await code_resources._search_code_entities(
                query, repository_id, limit
            )

            # Print results
            click.echo(f"Found {len(results)} results for '{query}':\n")

            for i, result in enumerate(results, 1):
                entity = result["entity"]  # type: ignore[index]
                click.echo(f"{i}. {entity['type']}: {entity['name']}")  # type: ignore[index]
                click.echo(f"   File: {entity['file_path']}")  # type: ignore[index]
                click.echo(f"   Lines: {entity['start_line']}-{entity['end_line']}")  # type: ignore[index]
                click.echo(f"   Similarity: {result['similarity']:.3f}")  # type: ignore[index]
                click.echo()

        finally:
            # Cleanup is handled by context managers
            pass

    asyncio.run(_search())


@cli.command()
def init_db() -> None:
    """Initialize the database."""

    async def _init() -> None:
        from src.database.init_db import init_database, verify_database_setup

        # Setup logging
        logging.basicConfig(level=logging.INFO)

        try:
            # Initialize database
            engine = await init_database()

            # Verify setup
            if await verify_database_setup(engine):
                click.echo("✅ Database initialized successfully")
            else:
                click.echo("❌ Database initialization incomplete")
                sys.exit(1)

            await engine.dispose()

        except (ImportError, ConnectionError, OSError, ValueError) as e:
            click.echo(f"❌ Database initialization failed: {e}")
            sys.exit(1)

    asyncio.run(_init())


@cli.command()
def create_config() -> None:
    """Create a sample configuration file."""
    sample_config = """# MCP Code Analysis Server Configuration

# OpenAI API key (required for embeddings)
openai_api_key: "your-api-key-here"

# Repositories to track
repositories:
  - url: https://github.com/example/repo1
    branch: main
  - url: https://github.com/example/repo2
    # branch: develop  # Optional, uses default branch if not specified
    # access_token: github_pat_...  # For private repos

# Database configuration
database:
  host: localhost
  port: 5432
  database: code_analysis
  user: codeanalyzer
  password: your-password

# MCP server configuration
mcp:
  host: 0.0.0.0
  port: 8080

# Scanner configuration
scanner:
  storage_path: ./repositories
  exclude_patterns:
    - __pycache__
    - "*.pyc"
    - .git
    - node_modules
    - venv
    - .env

# Embeddings configuration
embeddings:
  model: text-embedding-3-small
  batch_size: 100
  max_tokens: 8000

# LLM configuration
llm:
  model: gpt-4o-mini
  temperature: 0.2
  max_tokens: 4096

# Logging configuration
logging:
  level: INFO
  file_enabled: true
  file_path: logs/mcp-server.log
"""

    config_path = Path("config.yaml")
    if config_path.exists():
        click.echo(f"Configuration file already exists: {config_path}")
        if not click.confirm("Overwrite?"):
            return

    config_path.write_text(sample_config)
    click.echo(f"Created configuration file: {config_path}")
    click.echo("\nNext steps:")
    click.echo("1. Edit config.yaml and add your OpenAI API key")
    click.echo("2. Update database credentials")
    click.echo("3. Add repositories to track")
    click.echo("4. Run 'python -m src.mcp_server init-db' to initialize the database")
    click.echo("5. Run 'python -m src.mcp_server serve' to start the server")


if __name__ == "__main__":
    cli()
