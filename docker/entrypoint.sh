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

# Function to start background services
start_background_services() {
    if [ "${ENABLE_SCANNER:-true}" = "true" ]; then
        echo "Starting scanner service in background..."
        python -m src.scanner.main &
        SCANNER_PID=$!
        echo "Scanner started with PID: $SCANNER_PID"
    fi

    if [ "${ENABLE_INDEXER:-true}" = "true" ]; then
        echo "Starting indexer service in background..."
        python -m src.indexer.main &
        INDEXER_PID=$!
        echo "Indexer started with PID: $INDEXER_PID"
    fi
}

# Trap to ensure background processes are killed on exit
cleanup() {
    echo "Shutting down services..."
    if [ ! -z "$SCANNER_PID" ]; then
        kill $SCANNER_PID 2>/dev/null || true
    fi
    if [ ! -z "$INDEXER_PID" ]; then
        kill $INDEXER_PID 2>/dev/null || true
    fi
    exit 0
}
trap cleanup EXIT INT TERM

# Run database migrations if needed
if [ "$1" = "server" ]; then
    echo "Initializing database..."
    python -m src.mcp_server init-db || echo "Database already initialized"

    echo "Starting MCP server..."
    exec python -m src.mcp_server serve --host 0.0.0.0
elif [ "$1" = "all" ]; then
    echo "Initializing database..."
    python -m src.mcp_server init-db || echo "Database already initialized"

    # Start background services
    start_background_services

    echo "Starting MCP server..."
    # Use exec to replace the shell with the MCP server process
    exec python -m src.mcp_server serve --host 0.0.0.0
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
