#!/bin/bash
set -e

# Default to 'postgres' if POSTGRES_HOST is not set
POSTGRES_HOST="${POSTGRES_HOST:-postgres}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_USER="${POSTGRES_USER:-codeanalyzer}"

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL..."
while ! pg_isready -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER"; do
    sleep 1
done
echo "PostgreSQL is ready!"

# Run database migrations if needed
if [ "$1" = "server" ]; then
    echo "Initializing database..."
    python -m src.mcp_server init-db || echo "Database already initialized"
    
    echo "Starting MCP server..."
    exec python -m src.mcp_server serve
elif [ "$1" = "scanner" ]; then
    echo "Starting scanner service..."
    exec python -m src.scanner.main
elif [ "$1" = "indexer" ]; then
    echo "Starting indexer service..."
    exec python -m src.indexer.main
elif [ "$1" = "init-db" ]; then
    echo "Initializing database..."
    exec python -m src.mcp_server init-db
else
    # Run custom command
    exec "$@"
fi