# MCP Code Analysis Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

An intelligent MCP (Model Context Protocol) server that provides advanced code analysis and search capabilities for large codebases. It uses TreeSitter for parsing, PostgreSQL with pgvector for storage, and OpenAI embeddings for semantic search.

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

### Code Search Tools
- `semantic_search` - Search for code using natural language queries (with optional domain enhancement)
- `find_similar_code` - Find code similar to a given entity
- `search_by_code_snippet` - Search for code similar to a snippet
- `search_by_business_capability` - Find code implementing business capabilities
- `keyword_search` - Search for code using keywords

### Code Analysis Tools
- `get_code` - Get code content for a specific entity
- `analyze_file` - Analyze file structure and metrics
- `get_dependencies` - Get dependencies for a code entity
- `find_usages` - Find where a function or class is used

### Domain-Driven Design Tools
- `extract_domain_model` - Extract domain entities and relationships using LLM
- `find_aggregate_roots` - Find aggregate roots in the codebase
- `analyze_bounded_context` - Analyze bounded contexts and their relationships
- `suggest_ddd_refactoring` - Get DDD-based refactoring suggestions
- `find_bounded_contexts` - Discover all bounded contexts
- `generate_context_map` - Generate context maps (JSON, Mermaid, PlantUML)

### Advanced Analysis Tools
- `analyze_coupling` - Analyze coupling between bounded contexts with metrics
- `detect_anti_patterns` - Detect DDD anti-patterns (anemic models, god objects, etc.)
- `suggest_context_splits` - Suggest how to split large bounded contexts
- `analyze_domain_evolution` - Track domain model changes over time
- `get_domain_metrics` - Get comprehensive domain health metrics

### Repository Management Tools
- `add_repository` - Add a new GitHub repository to track
- `list_repositories` - List all tracked repositories
- `scan_repository` - Scan or rescan a repository
- `update_embeddings` - Update embeddings for a repository
- `get_repository_stats` - Get detailed statistics
- `delete_repository` - Delete a repository and its data

## Quick Start

### Prerequisites

- Nix with flakes enabled (recommended) OR Python 3.11+
- Docker and Docker Compose
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/johannhartmann/mcp-code-analysis-server.git
cd mcp-code-analysis-server
```

2. Enter the development environment:
```bash
nix develop  # Recommended
# OR
python -m venv venv && source venv/bin/activate
```

3. Install dependencies:
```bash
uv sync  # If using nix
# OR
pip install -e ".[dev]"  # If using regular Python
```

4. Create configuration file:
```bash
python -m src.mcp_server create-config
# Edit config.yaml with your settings
```

5. Set up environment variables:
```bash
export OPENAI_API_KEY="your-api-key-here"
# Or add to .env file
```

6. Start PostgreSQL with pgvector:
```bash
docker-compose up -d
```

7. Initialize the database:
```bash
python -m src.mcp_server init-db
```

8. Start the MCP server:
```bash
python -m src.mcp_server serve
# Or with options:
# python -m src.mcp_server serve --port 8080 --reload
```

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

### Command Line

```bash
# Add and scan a repository
python -m src.mcp_server scan https://github.com/owner/repo

# Search for code
python -m src.mcp_server search "authentication handler"

# Start the server
python -m src.mcp_server serve
```

### Using the MCP Tools

Once the server is running, you can use the tools via any MCP client:

```python
# Semantic search
await mcp.call_tool("semantic_search", {
    "query": "functions that handle user authentication",
    "scope": "functions",
    "limit": 10
})

# Get code content
await mcp.call_tool("get_code", {
    "entity_type": "function",
    "entity_id": 123,
    "include_context": True
})

# Add a repository
await mcp.call_tool("add_repository", {
    "url": "https://github.com/owner/repo",
    "scan_immediately": True,
    "generate_embeddings": True
})
```

### With Claude Desktop

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "code-analysis": {
      "command": "python",
      "args": ["-m", "src.mcp_server", "serve"],
      "cwd": "/path/to/mcp-code-analysis-server"
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
make test-all  # Run all tests with coverage
make test-unit  # Run unit tests only
make test-integration  # Run integration tests
```

### Code Quality

```bash
make qa  # Run all quality checks
make format  # Format code
make lint  # Run linters
make type-check  # Type checking
```

### Building Documentation

```bash
make docs  # Build docs
make docs-serve  # Serve docs locally
```

## Architecture

The server consists of several key components:

- **Scanner Module**: Monitors filesystem changes using Git
- **Parser Module**: Extracts code structure using TreeSitter
- **Embeddings Module**: Generates semantic embeddings via OpenAI
- **Query Module**: Processes natural language queries
- **MCP Server**: Exposes tools via FastMCP

## Performance

- Initial indexing: <1000 files/minute
- Incremental updates: <10 seconds for 100 changed files
- Query response time: <2 seconds
- Supports codebases up to 10M lines of code

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Johann-Peter Hartmann**  
Email: johann-peter.hartmann@mayflower.de  
GitHub: [@johannhartmann](https://github.com/johannhartmann)

## Acknowledgments

- Built with [FastMCP](https://github.com/fastmcp/fastmcp) for MCP protocol support
- Uses [TreeSitter](https://tree-sitter.github.io/tree-sitter/) for code parsing
- Powered by [LangChain](https://www.langchain.com/) and [LangGraph](https://github.com/langchain-ai/langgraph)
- Vector search via [pgvector](https://github.com/pgvector/pgvector)