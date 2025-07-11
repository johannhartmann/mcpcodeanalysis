# MCP Code Analysis Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

An intelligent MCP (Model Context Protocol) server that provides advanced code analysis and search capabilities for large codebases. Built with pure FastMCP implementation, it uses TreeSitter for parsing, PostgreSQL with pgvector for vector storage, and OpenAI embeddings for semantic search.

## Features

- 🔍 **Semantic Code Search**: Natural language queries to find relevant code
- 🏛️ **Domain-Driven Analysis**: Extract business entities and bounded contexts using LLM
- 📊 **Code Structure Analysis**: Hierarchical understanding of modules, classes, and functions
- 🔄 **Incremental Updates**: Git-based change tracking for efficient re-indexing
- 🎯 **Smart Code Explanations**: AI-powered explanations with context aggregation
- 🔗 **Dependency Analysis**: Understand code relationships and dependencies
- 🌐 **Knowledge Graph**: Build semantic graphs with community detection (Leiden algorithm)
- 💡 **DDD Refactoring**: Domain-Driven Design suggestions and improvements
- 🚀 **High Performance**: Handles codebases with millions of lines of code
- 🐍 **Python Support**: Full support for Python with more languages coming

## MCP Tools Available

### Core Search Tools
- `search_code` - Search for code using natural language queries with semantic understanding
- `find_definition` - Find where symbols (functions, classes, modules) are defined
- `find_similar_code` - Find code patterns similar to a given snippet using vector similarity
- `get_code_structure` - Get the hierarchical structure of a code file

### Code Analysis Tools
- `explain_code` - Get hierarchical explanations of code elements (modules, classes, functions)
- `suggest_refactoring` - Get AI-powered refactoring suggestions for code improvements
- `analyze_dependencies` - Analyze dependencies and relationships between code entities

### Repository Management Tools
- `sync_repository` - Manually trigger synchronization for a specific repository

## Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenAI API key (for semantic search capabilities)
- Nix with flakes (recommended for development)

### Docker Deployment (Recommended)

The easiest way to get started is using Docker Compose, which provides a complete isolated environment with PostgreSQL and pgvector.

1. Clone the repository:
```bash
git clone https://github.com/johannhartmann/mcp-code-analysis-server.git
cd mcp-code-analysis-server
```

2. Set up environment variables:
```bash
export OPENAI_API_KEY="your-api-key-here"
# Or add to .env file
```

3. Configure repositories:
Create a `config.yaml` file to specify which repositories to track:
```yaml
repositories:
  - url: https://github.com/owner/repo1
    branch: main
  - url: https://github.com/owner/repo2
    branch: develop
  - url: https://github.com/owner/private-repo
    access_token: "github_pat_..."  # For private repos

# Scanner configuration
scanner:
  storage_path: ./repositories
  exclude_patterns:
    - "__pycache__"
    - "*.pyc"
    - ".git"
    - "venv"
    - "node_modules"
```

4. Start the services with Docker Compose:
```bash
docker-compose up -d
```

This will:
- Start PostgreSQL with pgvector extension
- Build and start the MCP Code Analysis Server
- Initialize the database with required schemas
- Begin scanning configured repositories automatically

The server runs as a pure MCP implementation and can be accessed via any MCP-compatible client.

### Development Environment (Local)

For development work, use the Nix development environment which provides all necessary tools and dependencies:

```bash
# Enter the Nix development environment
nix develop

# Install Python dependencies
uv sync

# Start PostgreSQL (if not using Docker Compose)
docker-compose up -d postgres

# Run the scanner to populate the database
python -m src.scanner

# Start the MCP server
python -m src.mcp_server

# Or run tests
pytest

# Check code quality
ruff check .
black --check .
mypy .
vulture src vulture_whitelist.py
```

The Nix environment includes:
- Python 3.11 with all dependencies
- Code formatting tools (black, isort)
- Linters (ruff, pylint, bandit)
- Type checker (mypy)
- Dead code detection (vulture)
- Test runner (pytest)
- Pre-commit hooks

## Configuration

Edit `config.yaml` to customize:

```yaml
# OpenAI API key (can also use OPENAI_API_KEY env var)
openai_api_key: "sk-..."

# Repositories to track
repositories:
  - url: https://github.com/owner/repo
    branch: main  # Optional, uses default branch if not specified
  - url: https://github.com/owner/private-repo
    access_token: "github_pat_..."  # For private repos

# Scanner configuration
scanner:
  storage_path: ./repositories
  exclude_patterns:
    - "__pycache__"
    - "*.pyc"
    - ".git"
    - "venv"
    - "node_modules"

# Embeddings configuration
embeddings:
  model: "text-embedding-ada-002"
  batch_size: 100
  max_tokens: 8000

# MCP server configuration
mcp:
  host: "0.0.0.0"
  port: 8080

# Database configuration
database:
  host: localhost
  port: 5432
  database: code_analysis
  user: codeanalyzer
  password: your-secure-password
```

## Usage Examples

### Using the MCP Tools

Once the server is running, you can use the tools via any MCP client:

```python
# Search for code using natural language
await mcp.call_tool("search_code", {
    "query": "functions that handle user authentication",
    "limit": 10
})

# Find where a symbol is defined
await mcp.call_tool("find_definition", {
    "name": "UserService",
    "entity_type": "class"
})

# Get hierarchical code explanation
await mcp.call_tool("explain_code", {
    "path": "src.auth.user_service.UserService"
})

# Find similar code patterns
await mcp.call_tool("find_similar_code", {
    "code_snippet": "def authenticate_user(username, password):",
    "limit": 5,
    "threshold": 0.7
})

# Get code structure
await mcp.call_tool("get_code_structure", {
    "file_path": "src/auth/user_service.py"
})

# Get refactoring suggestions
await mcp.call_tool("suggest_refactoring", {
    "file_path": "src/auth/user_service.py",
    "focus_area": "performance"
})
```

### With Claude Desktop

Configure the MCP server in your Claude Desktop settings:

For stdio mode (when running locally):
```json
{
  "mcpServers": {
    "code-analysis": {
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "/path/to/mcp-code-analysis-server",
      "env": {
        "OPENAI_API_KEY": "your-api-key"
      }
    }
  }
}
```

For HTTP mode (when using Docker):
```json
{
  "mcpServers": {
    "code-analysis": {
      "url": "http://localhost:8000"
    }
  }
}
```

Then in Claude Desktop:
- "Search for functions that handle authentication"
- "Show me the implementation of the UserService class"
- "Find all usages of the database connection pool"
- "What files import the utils module?"

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test types
pytest tests/unit/
pytest tests/integration/
```

### Code Quality

The project uses comprehensive code quality tools integrated into the Nix development environment:

```bash
# Run all linters
ruff check .

# Format code
black .
isort .

# Type checking
mypy .

# Find dead code
vulture src vulture_whitelist.py

# Run pre-commit hooks
nix-pre-commit
```

### Pre-commit Hooks

Install the pre-commit hooks for automatic code quality checks:

```bash
echo '#!/bin/sh' > .git/hooks/pre-commit
echo 'nix-pre-commit' >> .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

## Architecture

The server consists of several key components:

- **Scanner Module**: Monitors and synchronizes Git repositories with incremental updates
- **Parser Module**: Extracts code structure using TreeSitter for accurate AST parsing
- **Embeddings Module**: Generates semantic embeddings via OpenAI for vector search
- **Database Module**: PostgreSQL with pgvector extension for efficient vector storage
- **Query Module**: Processes natural language queries and symbol lookup
- **MCP Server**: Pure FastMCP implementation exposing code analysis tools
- **Domain Module**: Extracts domain entities and relationships for DDD analysis

## Performance

- **Initial indexing**: ~1000 files/minute with parallel processing
- **Incremental updates**: <10 seconds for 100 changed files using Git tracking
- **Query response**: <2 seconds for semantic search with pgvector
- **Scalability**: Supports codebases up to 10M+ lines of code
- **Memory efficiency**: Optimized database sessions and batch processing

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Johann-Peter Hartmann**
Email: johann-peter.hartmann@mayflower.de
GitHub: [@johannhartmann](https://github.com/johannhartmann)

## Key Technologies

- **[FastMCP](https://github.com/fastmcp/fastmcp)**: Pure MCP protocol implementation
- **[TreeSitter](https://tree-sitter.github.io/tree-sitter/)**: Robust code parsing and AST generation
- **[pgvector](https://github.com/pgvector/pgvector)**: High-performance vector similarity search
- **[OpenAI Embeddings](https://openai.com/api/)**: Semantic understanding of code
- **[PostgreSQL](https://postgresql.org/)**: Reliable data persistence and complex queries
- **[Nix](https://nixos.org/)**: Reproducible development environment
- **[Docker](https://docker.com/)**: Containerized deployment and isolation
