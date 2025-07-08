# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure and configuration
- Core MCP server implementation with FastMCP
- TreeSitter-based Python code parser
- PostgreSQL database schema with pgvector support
- OpenAI embeddings integration for semantic search
- Git-based incremental indexing system
- Docker and Docker Compose configuration
- Comprehensive test suite with pytest
- Documentation with MkDocs
- CI/CD pipeline with GitHub Actions
- Pre-commit hooks for code quality

### MCP Tools
- `search_code` - Natural language code search
- `explain_code` - Hierarchical code explanations
- `find_definition` - Locate code definitions
- `find_usage` - Find usage locations
- `analyze_dependencies` - Module dependency analysis
- `suggest_refactoring` - AI-powered refactoring suggestions
- `find_similar_code` - Pattern-based code similarity search
- `get_code_structure` - Module/package structure visualization

## [0.1.0] - TBD

### Added
- First public release
- Basic functionality for Python code analysis
- Support for codebases up to 10M lines of code

[Unreleased]: https://github.com/johannhartmann/mcp-code-analysis-server/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/johannhartmann/mcp-code-analysis-server/releases/tag/v0.1.0