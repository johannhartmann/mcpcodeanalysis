# MCP Code Analysis Server - Development Guide

## Project Overview
This is an MCP (Model Context Protocol) server that provides intelligent code analysis and search capabilities for large codebases. It uses TreeSitter for parsing, PostgreSQL with pgvector for storage, and OpenAI embeddings for semantic search.

## Development Environment
- Uses Nix flakes with uv for Python dependency management
- Python 3.11 environment
- PostgreSQL with pgvector extension
- Docker Compose for deployment
- **Use the nix develop environment to execute python**

## Key Commands
- `nix develop` - Enter development shell
- `uv sync` - Sync Python dependencies
- `docker-compose up` - Start PostgreSQL and run server
- `pytest` - Run tests
- `ruff check .` - Run linter
- `mypy .` - Type checking
- `black .` - Format code

## Project Structure
```
├── src/
│   ├── scanner/          # Code scanning module
│   ├── parser/           # TreeSitter parsing
│   ├── embeddings/       # OpenAI embedding generation
│   ├── mcp_server/       # FastMCP server implementation
│   ├── query/            # Query processing
│   └── database/         # PostgreSQL models and migrations
├── tests/               # Test suite
├── config.yaml          # Configuration file
├── docker-compose.yml   # Docker services
├── pyproject.toml       # Python project config
└── flake.nix           # Nix development environment
```

## Core Components

### 1. Scanner Module (`src/scanner/`)
- Scans directories for Python files
- Uses Git for change tracking
- Implements incremental indexing
- Respects exclude patterns from config.yaml

### 2. Parser Module (`src/parser/`)
- Uses TreeSitter to extract code structure
- Extracts classes, functions, imports, docstrings
- Stores parsed data in PostgreSQL

### 3. Embeddings Module (`src/embeddings/`)
- Generates both raw code and interpreted embeddings
- Uses OpenAI text-embedding-ada-002
- Implements chunking strategies for large files
- Stores embeddings in pgvector

### 4. MCP Server (`src/mcp_server/`)
- FastMCP HTTP-based implementation
- Implements all required tools:
  - `search_code` - Natural language search
  - `explain_code` - Hierarchical code explanations
  - `find_definition` - Locate definitions
  - `find_usage` - Find usage locations
  - `analyze_dependencies` - Dependency analysis
  - `suggest_refactoring` - Refactoring suggestions
  - `find_similar_code` - Pattern matching
  - `get_code_structure` - Module structure

### 5. Query Module (`src/query/`)
- Processes natural language queries
- Implements ranking algorithms
- Handles aggregation for hierarchical explanations

## Database Schema
Main tables:
- `files` - File metadata and Git tracking
- `modules` - Python modules
- `classes` - Class definitions
- `functions` - Function/method definitions
- `imports` - Import statements
- `code_embeddings` - Vector embeddings

## Testing Strategy
- Use the MCP server's own codebase as test data
- Test incremental updates with Git changes
- Verify all MCP tools work correctly
- Performance benchmarks for large codebases

## Implementation Phases
1. **Core Infrastructure** - Scanner, database, Docker setup
2. **Parser Integration** - TreeSitter, AST mapping
3. **Embedding System** - OpenAI integration, pgvector storage
4. **MCP Server** - FastMCP, all tools implementation
5. **Advanced Features** - Refactoring, similarity, dependencies

## Performance Requirements
- Initial indexing: <1000 files/minute
- Incremental updates: <10s for 100 files
- Query response: <2s
- Support up to 10M LOC

## Environment Variables
```bash
OPENAI_API_KEY=your_api_key
POSTGRES_PASSWORD=secure_password
DATABASE_URL=postgresql://codeanalyzer:password@postgres:5432/code_analysis
```

## Quick Start
1. `nix develop` to enter shell
2. `cp config.example.yaml config.yaml` and edit
3. `docker-compose up -d postgres` to start database
4. `uv sync` to install dependencies
5. `python -m src.scanner` for initial scan
6. `python -m src.mcp_server` to start server

## Notes for Development
- Follow existing code patterns in TreeSitter parsing
- Use batch operations for embedding generation
- Implement proper error handling for external APIs
- Add logging for debugging large codebase scans
- Consider memory usage for large file processing
- Use database transactions for consistency

## Development Memories
- Do not try hacky workarounds. Fix it.
- For every change of the software you have to rebuild the docker container to see the result. 