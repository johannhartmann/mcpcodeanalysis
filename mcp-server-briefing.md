# MCP Code Analysis Server - Technical Briefing

## Project Overview

Build an MCP (Model Context Protocol) server that provides intelligent code analysis and search capabilities across multiple GitHub repositories. The server will monitor and sync GitHub repositories, analyze their structure using TreeSitter, and provide semantic search capabilities through PostgreSQL with pgvector.

## Technical Stack

- **Language**: Python
- **Framework**: LangGraph/LangChain
- **Database**: PostgreSQL with pgvector extension
- **MCP Implementation**: FastMCP (HTTP-based)
- **Parser**: TreeSitter
- **Embeddings**: OpenAI API
- **Deployment**: Docker Compose
- **Version Control Integration**: GitHub API + Git

## Core Components

### 1. Code Scanner Module

**Responsibilities:**
- Monitor multiple GitHub repositories for changes
- Clone and sync repositories locally
- Use GitHub API to track commits and file changes
- Implement incremental indexing (only process changed files)
- Store repository metadata and change tracking in PostgreSQL

**Key Features:**
- Watch multiple GitHub repositories (public and private with tokens)
- Webhook support for real-time updates
- Periodic polling for changes (configurable interval)
- Branch tracking (main/master by default, configurable per repo)
- Initial full clone and scan of repositories
- Incremental updates based on GitHub commits
- Support for large repositories with millions of lines
- Configurable exclude patterns (e.g., `__pycache__`, `.git`, `node_modules`)

### 2. Code Parser Module

**Using TreeSitter to extract:**
- Module/file structure
- Classes (name, docstring, methods, inheritance)
- Functions (name, parameters, return types, docstring)
- Methods (including decorators)
- Import statements
- Global variables and constants
- Type hints and annotations

**Storage Schema:**
```sql
-- Example tables structure
repositories (id, github_url, owner, name, default_branch, last_synced, access_token_id)
files (id, repository_id, path, last_modified, git_hash, branch)
modules (id, file_id, name, docstring)
classes (id, module_id, name, docstring, base_classes)
functions (id, module_id, class_id, name, parameters, return_type, docstring)
imports (id, file_id, import_statement, imported_from)
commits (id, repository_id, sha, message, author, timestamp, files_changed)
```

### 3. Embedding Generation Module

**Dual Embedding Strategy:**

1. **Raw Source Embeddings**
   - Embed actual code snippets
   - Chunk strategy: logical units (functions, classes)
   - Include surrounding context (imports, class definition for methods)

2. **Interpreted Code Embeddings**
   - Generate natural language descriptions of code functionality
   - Combine: function name + docstring + parameter info + inferred purpose
   - Use LLM to create searchable descriptions
   - **For classes**: Create aggregated embeddings combining all methods
   - **For packages/modules**: Create high-level purpose embeddings

**Implementation:**
- Use OpenAI embeddings API
- Store in pgvector with appropriate indexes
- Implement batch processing for efficiency

### 4. MCP Server Implementation

**Required MCP Tools:**

```python
@mcp.tool
async def search_code(query: str, limit: int = 10) -> list:
    """Search for code by natural language query"""

@mcp.tool
async def explain_code(path: str) -> str:
    """Explain what a code element does (function, class, module, or package).
    For classes: aggregates explanations of all methods and attributes.
    For packages: provides overview of all modules and main components."""

@mcp.tool
async def find_definition(name: str, type: str = "any") -> list:
    """Find where a function/class/module is defined.
    Type can be: 'function', 'class', 'module', or 'any'"""

@mcp.tool
async def find_usage(function_or_class: str, repository: str = None) -> list:
    """Find all places where a function/class is used.
    Can search across all repositories or filter by specific repository."""

@mcp.tool
async def analyze_dependencies(module_path: str) -> dict:
    """Analyze dependencies of a module"""

@mcp.tool
async def suggest_refactoring(code_path: str) -> list:
    """Suggest refactoring improvements"""

@mcp.tool
async def find_similar_code(code_snippet: str) -> list:
    """Find similar code patterns in the codebase"""

@mcp.tool
async def get_code_structure(path: str) -> dict:
    """Get hierarchical structure of a module/package"""

@mcp.tool
async def list_repositories() -> list:
    """List all monitored GitHub repositories with their sync status"""

@mcp.tool
async def sync_repository(repository_url: str) -> dict:
    """Manually trigger sync for a specific repository"""
```

### 5. Query Processing Module

**Search Pipeline:**
1. Accept natural language query
2. Generate embedding for query
3. Search both raw and interpreted embeddings
4. Rank results using pgvector similarity + additional heuristics
5. Enhance results with context (file path, surrounding code)
6. Format response appropriately

**Explanation Aggregation Logic:**

For the `explain_code` tool, implement hierarchical aggregation:

- **Function**: Return docstring + parameter analysis + inferred purpose
- **Class**:
  - Class-level docstring and purpose
  - Aggregate all method explanations
  - List attributes and their types
  - Inheritance information
  - Overall class responsibility summary
- **Module**:
  - Module docstring
  - List of main classes and their purposes
  - List of main functions and their purposes
  - Module's role in the system
- **Package**:
  - Package structure overview
  - Main modules and their responsibilities
  - Public API summary
  - Package's domain/purpose in the system

**Advanced Features:**
- Semantic search across docstrings and code
- Code pattern matching
- Import graph analysis
- Dead code detection

## Database Design

### PostgreSQL + pgvector Schema

```sql
-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Main tables (see section 2 for base tables)

-- Embedding tables
code_embeddings (
    id SERIAL PRIMARY KEY,
    entity_type VARCHAR(50), -- 'function', 'class', 'module'
    entity_id INTEGER,
    embedding_type VARCHAR(20), -- 'raw' or 'interpreted'
    embedding vector(1536), -- OpenAI dimension
    content TEXT,
    metadata JSONB
);

-- Indexes
CREATE INDEX ON code_embeddings USING ivfflat (embedding vector_cosine_ops);
```

## Configuration

### config.yaml
```yaml
repositories:
  - url: "https://github.com/owner/repo1"
    branch: "main"
    access_token: "${GITHUB_TOKEN_REPO1}"  # For private repos
  - url: "https://github.com/owner/repo2"
    branch: "develop"
  - url: "https://github.com/owner/repo3"
    # Uses default branch if not specified

scanner:
  sync_interval: 300  # Check for updates every 5 minutes
  webhook_secret: "${GITHUB_WEBHOOK_SECRET}"
  storage_path: "./repositories"  # Where to clone repos
  exclude_patterns:
    - "__pycache__"
    - "*.pyc"
    - ".git"
    - "venv"
    - ".env"
    - "node_modules"

parser:
  languages: ["python"]  # Initially, add more later
  chunk_size: 100  # Lines per chunk for large files

embeddings:
  model: "text-embedding-ada-002"
  batch_size: 100

database:
  host: "postgres"
  port: 5432
  database: "code_analysis"

mcp:
  host: "0.0.0.0"
  port: 8080

github:
  api_rate_limit: 5000  # Requests per hour
  webhook_endpoint: "/webhooks/github"
  use_webhooks: true  # Enable webhook support
  poll_interval: 300  # Fallback polling in seconds
```

## Docker Compose Setup

```yaml
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: code_analysis
      POSTGRES_USER: codeanalyzer
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data

  mcp-server:
    build: .
    ports:
      - "8080:8080"
    environment:
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      DATABASE_URL: postgresql://codeanalyzer:${POSTGRES_PASSWORD}@postgres:5432/code_analysis
    volumes:
      - ./repositories:/repositories  # Cloned GitHub repos
      - ./config.yaml:/app/config.yaml
    depends_on:
      - postgres

volumes:
  postgres_data:
```

## Implementation Guidelines

### Phase 1: Core Infrastructure
1. Set up Docker environment
2. Implement GitHub repository watcher
3. Create database schema with repository tracking
4. Implement GitHub API integration and cloning
5. Test with sample GitHub repositories

### Phase 2: Parser Integration
1. Integrate TreeSitter for Python
2. Build AST to database mapping
3. Implement incremental update logic
4. Test on the project's own codebase

### Phase 3: Embedding System
1. Implement code chunking strategies
2. Create embedding generation pipeline
3. Build both raw and interpreted embeddings
4. Store in pgvector with proper indexes

### Phase 4: MCP Server
1. Implement FastMCP server
2. Create all specified tools
3. Build query processing pipeline
4. Implement aggregation logic for explain_code
5. Add result ranking and formatting

### Phase 5: Advanced Features
1. Implement refactoring suggestions
2. Add code similarity detection
3. Build dependency analysis
4. Create import graph visualization data

## Testing Strategy

- Monitor multiple test repositories on GitHub
- Verify webhook and polling updates work correctly
- Test with queries like:
  - "Where is the FastAPI application defined in repo1?" (using find_definition)
  - "Explain the DatabaseConnection class across all repositories" (should show from multiple repos)
  - "Find all authentication functions in the organization's repos"
  - "Show me similar implementations of user validation across repos"
  - "What repositories use SQLAlchemy?"
  - "Compare error handling patterns between repo1 and repo2"

## Deliverables

1. **Source Code**: Complete Python implementation
2. **Docker Setup**: `docker-compose.yml` and `Dockerfile`
3. **Documentation**:
   - `README.md`: User guide for running and using the server
   - `DEVELOPER.md`: Technical documentation for extending the system
   - `ARCHITECTURE.md`: System design and component interaction
4. **Configuration**: Sample `config.yaml` with sensible defaults
5. **Tests**: Basic test suite for core functionality

## Performance Requirements

- Initial indexing: < 1000 files/minute
- Incremental updates: < 10 seconds for 100 changed files
- Query response time: < 2 seconds for typical searches
- Support codebases up to 10M lines of code

## Extension Points

Design the system to easily add:
- New programming languages (JavaScript, TypeScript, Java, PHP)
- Additional embedding models
- Custom analysis tools
- Alternative storage backends

## Questions or Clarifications

If you need any clarification during development, particularly regarding:
- Specific MCP tool behaviors
- Edge cases in code parsing
- Performance optimization strategies
- Integration patterns

Please don't hesitate to ask!
