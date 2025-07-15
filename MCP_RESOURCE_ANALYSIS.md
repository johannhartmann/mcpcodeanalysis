# MCP Resource Quality Analysis

## Executive Summary

After analyzing the MCP resource definitions in the codebase, I found that while the URI patterns are generally well-structured and consistent, **the resources lack explicit descriptions in their decorators**, relying instead on function docstrings. This approach may limit discoverability and understanding for MCP hosts/clients.

## Current State Analysis

### 1. Resource Definitions Structure

All resources use the pattern:
```python
@self.mcp.resource("scheme://path/{parameters}")
async def resource_function(...) -> str:
    """Function docstring that serves as implicit description."""
```

**Key Issue**: None of the resources use the `description` parameter available in FastMCP's resource decorator:
```python
# Current approach (implicit description via docstring)
@self.mcp.resource("code://search")
async def search_code() -> str:
    """Search for code using natural language queries."""

# Better approach (explicit description)
@self.mcp.resource("code://search",
                   description="Search for code using natural language queries. Returns guidance on using the search_code tool.")
async def search_code() -> str:
```

### 2. URI Pattern Analysis

#### Strengths:
- **Consistent scheme usage**: `code://`, `migration://`, `packages://`, `system://`
- **Clear hierarchical patterns**: e.g., `packages://{repository_url}/{package_path}/dependencies`
- **RESTful-style paths**: Resources follow intuitive patterns

#### Good Examples:
1. **Clear parameter naming**: `code://definitions/{name}` - obvious what "name" refers to
2. **Hierarchical structure**: `packages://{repository_url}/{package_path}/dependencies` - shows clear parent-child relationship
3. **Action-oriented paths**: `migration://readiness/{repository_url}` - clear what analysis is performed

#### Poor Examples:
1. **Ambiguous parameters**:
   - `code://explain/{entity_type}/{entity_id}` - What are valid entity_types? What's an entity_id?
   - `code://refactoring/{entity_type}/{entity_id}` - Same issues

2. **Inconsistent patterns**:
   - Some use IDs: `code://usages/{entity_type}/{entity_id}`
   - Others use names/paths: `code://structure/{file_path}`
   - This inconsistency makes it harder to predict URI patterns

3. **Missing parameter constraints**:
   - No indication of valid values for `entity_type` (function, class, module?)
   - No format specification for `repository_url` (full URL? just owner/repo?)

### 3. Resource Return Value Documentation

**Major Gap**: Resources return string content but don't specify the format. An LLM consuming these resources would need to parse the response to understand:
- Is it Markdown? Plain text? JSON?
- What sections/headers to expect?
- What information is guaranteed vs optional?

Example issue:
```python
async def get_code_structure(file_path: str) -> str:
    """Get the structure of a code file."""
    # Returns Markdown with sections like "## Classes", "## Functions"
    # But this format isn't documented anywhere
```

### 4. Parameter Documentation

**Critical Issue**: URI template parameters lack documentation. An MCP client cannot know:
- Required vs optional parameters
- Parameter formats/constraints
- Valid parameter values

Example:
```python
@self.mcp.resource("packages://{repository_url}/circular-dependencies")
async def find_circular_dependencies(
    repository_url: str, max_depth: int | None = 10
) -> str:
```
Issues:
- What format for `repository_url`? Full GitHub URL? Just "owner/repo"?
- The `max_depth` parameter in the function isn't part of the URI - confusing!

## Specific Improvements Needed

### 1. Add Explicit Resource Descriptions

```python
@self.mcp.resource(
    "code://definitions/{name}",
    description="Find where symbols (functions, classes, modules) are defined in the codebase. "
                "Returns a Markdown document listing all definitions with file paths and signatures.",
    name="Find Symbol Definitions"
)
async def find_definitions(name: str) -> str:
```

### 2. Document URI Parameters

Create a resource manifest or use FastMCP's template features to document:
```python
@self.mcp.resource(
    "code://explain/{entity_type}/{entity_id}",
    description="Get detailed explanation of a code entity including documentation, signature, and metrics.",
    name="Explain Code Entity"
)
async def explain_code(
    entity_type: str = Field(description="Type of entity: 'function', 'class', or 'module'"),
    entity_id: int = Field(description="Database ID of the entity")
) -> str:
```

### 3. Standardize Return Formats

Add return format documentation:
```python
@self.mcp.resource(
    "system://health",
    description="Get comprehensive health status of the MCP server. "
                "Returns a Markdown document with sections: Overall Status, Component Health, "
                "Repository Statistics, and System Information.",
    mime_type="text/markdown"
)
```

### 4. Create Resource Guidelines

Document patterns for consistency:
- Use entity IDs for precise lookups: `resource://entity/{id}`
- Use paths/names for search operations: `resource://search/{name}`
- Always prefer full URLs for repository identification
- Use kebab-case for multi-word paths: `circular-dependencies` not `circularDependencies`

### 5. Add Resource Discovery

Some resources return instructions to use tools instead of data:
```python
# Current approach - returns instructions
@self.mcp.resource("code://search")
async def search_code() -> str:
    return """To search for code, use the search_code tool instead..."""

# Better approach - make it a proper resource or remove it
@self.mcp.resource(
    "code://search/help",
    description="Get help on using code search functionality"
)
```

## Recommendations

### High Priority:
1. **Add description parameter** to all resource decorators
2. **Document parameter constraints** in descriptions
3. **Standardize URI patterns** (use IDs vs names consistently)
4. **Specify return formats** (add mime_type where appropriate)

### Medium Priority:
1. **Create resource documentation** showing example URIs and responses
2. **Add parameter validation** with clear error messages
3. **Consider using JSON responses** for structured data instead of Markdown

### Low Priority:
1. **Add resource versioning** for future compatibility
2. **Implement resource metadata** endpoints
3. **Add resource pagination** for large result sets

## Example of Well-Documented Resource

```python
@self.mcp.resource(
    "code://definitions/{symbol_name}",
    description="Find all definitions of a symbol (function, class, or module) in the codebase. "
                "Symbol names are case-sensitive and support partial matching. "
                "Returns a Markdown document with sections for each definition found, "
                "including file path, line number, signature, and documentation.",
    name="Find Symbol Definitions",
    mime_type="text/markdown",
    tags={"code-analysis", "search"}
)
async def find_definitions(
    symbol_name: str = Field(
        description="Name of the symbol to search for. Supports exact and partial matching. "
                    "Examples: 'UserService', 'process_', 'utils.helper'"
    )
) -> str:
    """
    Find where symbols are defined in the codebase.

    Returns a Markdown document with:
    - Total number of definitions found
    - For each definition:
      - Symbol type (function/class/module)
      - File path and line number
      - Full signature
      - Docstring (if available)
    """
```

## Conclusion

While the MCP server has a good foundation with consistent URI schemes and comprehensive functionality, it needs better resource documentation to be truly usable by LLM-based MCP hosts. The main issues are:

1. **Missing explicit descriptions** in resource decorators
2. **Undocumented parameter constraints** and formats
3. **Unclear return value structures**
4. **Inconsistent URI patterns** for similar operations

Implementing the recommended improvements would significantly enhance the discoverability and usability of these resources for any MCP client, especially LLM-based ones that rely on clear descriptions to understand how to use the resources effectively.
