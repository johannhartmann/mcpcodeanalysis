"""Database initialization utilities."""

import asyncio
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker


# Import domain models to ensure they're registered with metadata
from src.database import domain_models  # noqa: F401
from src.database.models import Base
from src.logger import get_logger

logger = get_logger(__name__)


async def create_database_if_not_exists(database_url: str) -> None:
    """Create database if it doesn't exist.

    Args:
        database_url: PostgreSQL database URL
    """
    # Extract database name and create URL without it
    parts = database_url.split("/")
    db_name = parts[-1].split("?")[0]
    server_url = "/".join(parts[:-1]) + "/postgres"

    # Connect to postgres database
    engine = create_async_engine(server_url, isolation_level="AUTOCOMMIT")

    async with engine.connect() as conn:
        # Check if database exists
        result = await conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :db_name"),
            {"db_name": db_name},
        )
        exists = result.scalar()

        if not exists:
            logger.info("Creating database: %s", db_name)
            # CREATE DATABASE doesn't support parameters, but db_name is from our URL
            await conn.execute(text(f"CREATE DATABASE {db_name}"))
        else:
            logger.info("Database already exists: %s", db_name)

    await engine.dispose()


async def init_pgvector(engine: AsyncEngine) -> None:
    """Initialize pgvector extension.

    Args:
        engine: Database engine
    """
    async with engine.connect() as conn:
        try:
            # Create pgvector extension
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.commit()
            logger.info("pgvector extension initialized")
        except Exception:
            logger.exception("Failed to initialize pgvector")
            raise


async def create_tables(engine: AsyncEngine) -> None:
    """Create database tables.

    Args:
        engine: Database engine
    """
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")
    except Exception:
        logger.exception("Failed to create tables")
        raise


async def init_database(database_url: str | None = None) -> AsyncEngine:
    """Initialize database with all required setup.

    Args:
        database_url: Optional database URL, uses settings if not provided

    Returns:
        Configured database engine
    """
    if not database_url:
        # Build database URL from settings
        from src.config import get_database_url

        database_url = get_database_url()

    logger.info("Initializing database")

    # Check if using SQLite (for tests)
    is_sqlite = database_url.startswith("sqlite")

    if not is_sqlite:
        # Create database if needed (PostgreSQL only)
        await create_database_if_not_exists(database_url)

    # Create engine
    if is_sqlite:
        # SQLite configuration
        engine = create_async_engine(
            database_url,
            echo=False,
            future=True,
        )
    else:
        # PostgreSQL configuration
        engine = create_async_engine(
            database_url,
            echo=False,  # Set to True for SQL debugging
            future=True,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )

        # Initialize pgvector (PostgreSQL only)
        await init_pgvector(engine)

    # Create tables
    await create_tables(engine)

    logger.info("Database initialization complete")

    return engine


def get_session_factory(engine: AsyncEngine) -> sessionmaker:
    """Create session factory for the engine.

    Args:
        engine: Database engine

    Returns:
        Session factory
    """
    return sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


async def verify_database_setup(engine: AsyncEngine) -> bool:
    """Verify database is properly set up.

    Args:
        engine: Database engine

    Returns:
        True if setup is valid
    """
    try:
        async with engine.connect() as conn:
            # Check pgvector
            result = await conn.execute(
                text("SELECT 1 FROM pg_extension WHERE extname = 'vector'"),
            )
            if not result.scalar():
                logger.error("pgvector extension not found")
                return False

            # Check tables exist
            result = await conn.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name IN (
                        'repositories', 'files', 'modules', 'classes',
                        'functions', 'imports', 'commits', 'code_embeddings'
                    )
                """,
                ),
            )
            table_count = result.scalar()

            expected_tables = 8
            if table_count < expected_tables:
                logger.error(
                    "Missing tables, found only %d/%d",
                    table_count,
                    expected_tables,
                )
                return False

            logger.info("Database setup verified successfully")
            return True

    except Exception:
        logger.exception("Database verification failed")
        return False


if __name__ == "__main__":
    # Run database initialization
    async def main() -> None:
        """Initialize database from command line."""
        logging.basicConfig(level=logging.INFO)

        try:
            engine = await init_database()

            # Verify setup
            if await verify_database_setup(engine):
                print("✅ Database initialized successfully")
            else:
                print("❌ Database initialization incomplete")

            await engine.dispose()

        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
            raise

    asyncio.run(main())
