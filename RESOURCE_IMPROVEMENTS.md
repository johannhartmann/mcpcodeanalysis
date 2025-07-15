# MCP Resource Improvements Guide

## Critical Issues Found

### 1. **No Explicit Resource Descriptions**
Resources rely on docstrings instead of FastMCP's `description` parameter, limiting discoverability.

**Current:**
```python
@self.mcp.resource("code://explain/{entity_type}/{entity_id}")
async def explain_code(entity_type: str, entity_id: int) -> str:
    """Get explanation of code entities."""
```

**Improved:**
```python
@self.mcp.resource(
    "code://explain/{entity_type}/{entity_id}",
    description="""Get detailed explanation of a code entity with usage examples and metrics.

    Parameters:
    - entity_type: 'function', 'class', or 'module'
    - entity_id: Numeric ID from search results or other queries

    Returns: Markdown document with:
    - Full source code with syntax highlighting
    - Documentation and docstrings
    - Usage examples from the codebase
    - Metrics (complexity, lines, dependencies)

    Example: code://explain/function/12345""",
    mime_type="text/markdown"
)
```

### 2. **Undocumented URI Parameters**

**Problem Areas:**
- `{entity_type}` - What values are valid?
- `{entity_id}` - How do I get these IDs?
- `{repository_url}` - What format? Encoded?
- `{package_path}` - Relative to what?

**Solutions:**

```python
# Add parameter documentation
ENTITY_TYPES = Literal["function", "class", "module", "file"]
REPOSITORY_URL_PATTERN = r"https://github\.com/[\w-]+/[\w-]+"

@self.mcp.resource(
    "code://definitions/{name}",
    description="""Find all definitions of a symbol across the codebase.

    Parameters:
    - name: Symbol name to search for (e.g., 'UserService', 'process_order')

    Returns: List of all definitions with file paths and line numbers.

    Example: code://definitions/AuthenticationManager""",
    mime_type="text/markdown"
)
```

### 3. **Inconsistent URI Patterns**

**Current Issues:**
- Some use IDs: `code://explain/function/123`
- Others use names: `code://definitions/UserService`
- Some mix both: `packages://repo-url/path`

**Standardization Proposal:**
```
# Use names for discovery/search
code://search?q=authentication
code://definitions/UserService

# Use IDs for specific entities (from search results)
code://explain/function/123
code://similar/class/456

# Use URLs for repository context
packages://github.com/owner/repo/src/services
migration://readiness/github.com/owner/repo
```

### 4. **Missing Return Format Documentation**

**Improved Resource with Response Schema:**
```python
@self.mcp.resource(
    "code://search",
    description="""Search for code using natural language queries.

    Query Parameters:
    - q: Natural language search query
    - repository_url: (optional) Limit to specific repository
    - limit: (optional) Max results, default 10

    Returns: Markdown with sections:
    1. Summary (X results found)
    2. Results list with:
       - Entity name and type
       - File path and lines
       - Relevance score
       - Code preview
       - Link to full explanation

    Example: code://search?q=payment+processing&limit=5""",
    mime_type="text/markdown"
)
```

### 5. **Resource Categories Need Better Organization**

**Current:** Flat list of resources
**Improved:** Grouped by purpose

```python
# Code Analysis Resources
code://search - Natural language code search
code://explain/{type}/{id} - Detailed entity explanation
code://structure/{file_path} - File structure overview
code://definitions/{name} - Find symbol definitions
code://usages/{type}/{id} - Find where entity is used
code://similar - Find similar code patterns

# Package Analysis Resources
packages://{repo}/tree - Package hierarchy
packages://{repo}/dependencies - Dependency graph
packages://{repo}/circular - Circular dependencies
packages://{repo}/{path}/coupling - Coupling metrics

# Migration Resources
migration://readiness/{repo} - Migration assessment
migration://patterns/search - Reusable patterns
migration://dashboard/{repo} - Progress tracking

# System Resources
system://health - Server health status
system://stats - Usage statistics
system://config - Configuration info
```

## Implementation Example

Here's how to improve a resource definition:

```python
@self.mcp.resource(
    "packages://{repository_url}/circular-dependencies",
    description="""Detect circular dependencies in a repository's package structure.

    Parameters:
    - repository_url: GitHub repository URL (e.g., github.com/owner/repo)

    Returns: Markdown report containing:
    - Summary of circular dependency count
    - Detailed cycles with file paths
    - Suggested refactoring approaches
    - Visualization of dependency cycles

    Use when: Preparing for modularization or investigating build issues.

    Example: packages://github.com/acme/monolith/circular-dependencies""",
    mime_type="text/markdown"
)
async def get_circular_dependencies(repository_url: str) -> str:
    # Validate URL format
    if not re.match(REPOSITORY_URL_PATTERN, repository_url):
        raise ResourceError(f"Invalid repository URL format: {repository_url}")
    # ... implementation
```

## Resource Templates

Add templates for common parameter patterns:

```python
# In resource registration
self.mcp.add_resource_template(
    "code://explain/{entity_type}/{entity_id}",
    parameters={
        "entity_type": {
            "description": "Type of code entity",
            "enum": ["function", "class", "module"],
            "required": True
        },
        "entity_id": {
            "description": "Numeric ID from search results",
            "type": "integer",
            "required": True
        }
    }
)
```

## Error Response Standards

```python
# Consistent error responses
{
    "error": "Entity not found",
    "details": {
        "entity_type": "function",
        "entity_id": 12345,
        "suggestion": "Use code://search to find valid entities"
    }
}
```

## Testing Resource Usability

Create test scripts that simulate MCP client usage:

```python
# Test resource discovery
resources = await mcp.list_resources()
for resource in resources:
    assert resource.description is not None
    assert len(resource.description) > 50
    assert resource.mime_type is not None
```
