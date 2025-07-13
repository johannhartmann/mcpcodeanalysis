# MCP Code Analysis Server - Complete Guide

## Project Overview
This is a production-ready MCP (Model Context Protocol) server that provides intelligent code analysis and search capabilities for large codebases. It uses TreeSitter for parsing, PostgreSQL with pgvector for storage, and OpenAI embeddings for semantic search.

The server implements the Model Context Protocol, making it compatible with Claude Desktop, custom clients, and any MCP-compatible application. It provides comprehensive code analysis including semantic search, dependency analysis, domain-driven design extraction, and advanced refactoring suggestions.

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
- `viberdash monitor --interval 300` - Monitor code quality metrics (updates every 5 minutes)

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
- FastMCP HTTP-based implementation serving on `http://localhost:8080/mcp/v1/messages`
- Full MCP protocol compliance with session management
- Comprehensive tool suite (30+ tools):

#### Core Analysis Tools:
  - `search_code` - Natural language semantic search
  - `explain_code` - Hierarchical code explanations
  - `find_definition` - Locate symbol definitions
  - `find_usage` - Find usage locations
  - `analyze_dependencies` - Dependency analysis
  - `suggest_refactoring` - AI-powered refactoring suggestions
  - `find_similar_code` - Pattern matching and code similarity
  - `get_code_structure` - Module/file structure analysis

#### Domain-Driven Design Tools:
  - `extract_domain_model` - Extract domain entities and relationships
  - `find_context_relationships` - Context mapping analysis
  - `analyze_bounded_contexts` - Bounded context identification
  - `suggest_aggregate_roots` - Aggregate root suggestions
  - `analyze_repository_patterns` - Repository pattern analysis

#### Package & Architecture Tools:
  - `analyze_packages` - Package structure analysis
  - `get_package_dependencies` - Package dependency graphs
  - `find_circular_dependencies` - Circular dependency detection
  - `get_package_coupling` - Coupling metrics analysis
  - `get_package_tree` - Package hierarchy visualization

#### Repository Management:
  - `list_repositories` - Repository management with stats
  - `sync_repository` - Manual repository synchronization
  - `health_check` - Server health and status

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

### Development Setup
1. `nix develop` - Enter development shell
2. `cp config.example.yaml config.yaml` - Create config file
3. Edit `config.yaml` with your OpenAI API key and repository URLs
4. `uv sync` - Install Python dependencies

### Production Deployment (Recommended)
```bash
# Start all services with Docker Compose
docker-compose up -d

# Check service status
docker ps

# View logs
docker logs mcp-server
docker logs mcp-scanner
docker logs mcp-postgres
```

The Docker setup includes:
- PostgreSQL with pgvector extension
- Automatic database initialization
- Code scanner service (incremental updates)
- MCP server on port 8080
- Persistent data volumes

### Manual Development
For development without Docker:
1. `docker-compose up -d postgres` - Start only database
2. `python -m src.mcp_server init-db` - Initialize database
3. `python -m src.scanner` - Run initial scan
4. `python -m src.mcp_server serve` - Start MCP server

## Using the MCP Server

### With Claude Desktop
Add this configuration to Claude Desktop's MCP settings:

```json
{
  "mcp": {
    "servers": {
      "code-analysis": {
        "command": "node",
        "args": ["-e", "require('http').request('http://localhost:8080/mcp/v1/messages', {method:'POST', headers:{'Content-Type':'application/json'}}, res => res.pipe(process.stdout)).end()"],
        "env": {}
      }
    }
  }
}
```

Or use HTTP transport directly at `http://localhost:8080/mcp/v1/messages`

### Available Tools & Usage Examples

#### Code Search & Analysis
```javascript
// Natural language code search
search_code({
  query: "TreeSitter parser that extracts function definitions",
  limit: 10
})

// Find specific definitions
find_definition({
  name: "CodeProcessor",
  entity_type: "class"
})

// Get file structure
get_code_structure({
  file_path: "src/parser/treesitter_parser.py"
})

// Analyze dependencies
analyze_dependencies({
  file_path: "src/mcp_server/server.py"
})
```

#### Domain-Driven Design Analysis
```javascript
// Extract domain model
extract_domain_model({
  code_path: "src/database/models.py",
  include_relationships: true
})

// Find bounded contexts
analyze_bounded_contexts({
  search_paths: ["src/"],
  min_entities: 3
})

// Suggest aggregate roots
suggest_aggregate_roots({
  domain_path: "src/domain/",
  include_reasoning: true
})
```

#### Architecture & Package Analysis
```javascript
// Analyze package structure
analyze_packages({
  root_path: "src/",
  include_metrics: true
})

// Find circular dependencies
find_circular_dependencies({
  root_path: "src/",
  max_depth: 5
})

// Get coupling metrics
get_package_coupling({
  package_path: "src/mcp_server/",
  include_details: true
})
```

#### Repository Management
```javascript
// List all repositories with stats
list_repositories({
  include_stats: true
})

// Manual sync
sync_repository({
  repository_url: "https://github.com/user/repo",
  force_full_scan: false
})

// Health check
health_check()
```

### Configuration

#### config.yaml Structure
```yaml
# OpenAI API configuration
openai_api_key: "your-api-key-here"

# Repositories to track
repositories:
  - url: https://github.com/your-org/your-repo
    branch: main  # optional, uses default branch if not specified
    access_token: github_pat_...  # for private repos

# Database configuration
database:
  host: localhost
  port: 5432
  database: code_analysis
  user: codeanalyzer
  password: your-password

# MCP server settings
mcp:
  host: 0.0.0.0
  port: 8080

# Scanner settings
scanner:
  storage_path: ./repositories
  exclude_patterns:
    - __pycache__
    - "*.pyc"
    - .git
    - node_modules
    - venv
    - .env

# Embedding configuration
embeddings:
  model: text-embedding-3-small
  batch_size: 100
  max_tokens: 8000

# LLM configuration (for analysis tools)
llm:
  model: gpt-4o-mini
  temperature: 0.2
  max_tokens: 4096
```

### Incremental Scanning

The scanner automatically handles incremental updates:

1. **Initial Scan**: Full repository scan on first run
2. **Incremental Updates**: Only processes files changed since last sync
3. **Git Integration**: Uses Git commit history to identify changes
4. **Periodic Sync**: Runs every 5 minutes by default
5. **Efficient Processing**: Smaller batch sizes for changed files

Monitor scanner activity:
```bash
# View scanner logs
docker logs mcp-scanner

# Check for changed files
docker logs mcp-scanner | grep "changed files"
```

### Performance Characteristics

- **Initial indexing**: ~1000 files/minute
- **Incremental updates**: <10s for 100 files
- **Query response**: <2s for semantic search
- **Capacity**: Supports up to 10M lines of code
- **Memory usage**: ~500MB for typical projects
- **Database size**: ~100MB per 1M LOC

### Monitoring & Debugging

#### Health Checks
```bash
# Container health
docker ps

# Service logs
docker logs mcp-server
docker logs mcp-scanner
docker logs mcp-postgres

# Database connectivity
docker exec mcp-postgres psql -U codeanalyzer -d code_analysis -c "SELECT COUNT(*) FROM files;"
```

#### Common Issues

1. **"Could not resolve target entity" warnings**: Expected for external dependencies
2. **Health check failures**: Docker health checks expect `/health` endpoint (cosmetic only)
3. **Scanner errors**: Check Git repository access and API keys
4. **Embedding failures**: Verify OpenAI API key and rate limits

#### Performance Tuning

```yaml
# Adjust in config.yaml
scanner:
  batch_size: 5  # Reduce for memory constraints

embeddings:
  batch_size: 50  # Reduce for API rate limits

database:
  pool_size: 10  # Adjust connection pool
```

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
- Always run python, uv etc in nix develop
- Never mention claude or anthropic in commit messages.
