"""MCP Code Analysis Server entry point."""

import asyncio
import logging
import sys
from pathlib import Path

import click

from src.mcp_server.config import get_settings
from src.mcp_server.server import create_server
from src.utils.logger import setup_logging


@click.group()
def cli() -> None:
    """MCP Code Analysis Server CLI."""


@cli.command()
@click.option(
    "--host",
    default="0.0.0.0",
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
def serve(host: str, port: int, reload: bool, log_level: str) -> None:
    """Start the MCP server."""
    # Setup logging
    setup_logging()
    # Override log level if specified
    if log_level:
        logging.getLogger().setLevel(log_level.upper())

    # Get settings
    settings = get_settings()

    # Override with CLI options
    if host != "0.0.0.0":
        settings.mcp.host = host
    if port != 8080:
        settings.mcp.port = port

    # Import the mcp instance
    import os

    # FastMCP handles running the server
    from src.mcp_server.server import mcp

    # Set host and port via environment variables for FastMCP
    os.environ["FASTMCP_HOST"] = host
    os.environ["FASTMCP_PORT"] = str(port)

    # Run using FastMCP with HTTP transport
    mcp.run(transport="http", host=host, port=port)


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

        # Create server
        server = create_server()

        # Initialize server
        await server.initialize()

        try:
            # Scan repository
            result = await server.scan_repository(
                repository_url, branch=branch, generate_embeddings=embeddings,
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
            # Cleanup
            await server._shutdown()

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

        # Create server
        server = create_server()

        # Initialize server
        await server.initialize()

        try:
            # Search
            results = await server.search(query, repository_id, limit)

            # Print results
            click.echo(f"Found {len(results)} results for '{query}':\n")

            for i, result in enumerate(results, 1):
                entity = result["entity"]
                click.echo(f"{i}. {entity['type']}: {entity['name']}")
                click.echo(f"   File: {entity['file_path']}")
                click.echo(f"   Lines: {entity['start_line']}-{entity['end_line']}")
                click.echo(f"   Similarity: {result['similarity']:.3f}")
                click.echo()

        finally:
            # Cleanup
            await server._shutdown()

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

        except Exception as e:
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
  model: text-embedding-ada-002
  batch_size: 100
  max_tokens: 8000

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
