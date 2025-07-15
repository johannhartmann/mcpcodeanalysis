# MCP Usability Improvements Summary

## Overview
The current MCP implementation has functional tools and resources but lacks the documentation and structure needed for reliable usage by MCP hosts like Claude Desktop or other LLM-based clients.

## Key Issues

### 1. **Documentation Mismatch**
- README claims 30+ tools including `search_code`, `explain_code`, etc.
- These are actually implemented as resources, not tools
- This confusion makes the API hard to understand

### 2. **Poor Descriptions**
Most tools and resources have minimal descriptions that don't explain:
- When to use them
- What parameters mean
- What output format to expect
- How to get required IDs or values

### 3. **Inconsistent Patterns**
- Some resources use IDs, others use names
- No clear convention for URI structure
- Mix of query parameters and path parameters

## Specific Improvements Needed

### For Tools

1. **Repository Management Tools**
   ```python
   # Current
   description="Scan or rescan a repository"

   # Improved
   description="""Scan repository for code changes and update analysis database.
   Use after: git pull, code updates, or to fix sync issues.
   Process: Detects new/modified files, parses code structure, updates search indexes.
   Returns: Summary of files processed, entities found, and any errors."""
   ```

2. **Add Parameter Context**
   ```python
   # Current
   repository_id: int = Field(..., description="Repository ID")

   # Improved
   repository_id: int = Field(
       ...,
       description="Repository ID from list_repositories or add_repository response. Use list_repositories to see all available repositories with their IDs."
   )
   ```

### For Resources

1. **Add Explicit Descriptions**
   ```python
   # Current
   @self.mcp.resource("code://search")
   async def search_code(q: str) -> str:
       """Search for code."""

   # Improved
   @self.mcp.resource(
       "code://search",
       description="""Natural language code search across your codebase.

       Query params:
       - q: Search query (e.g. "user authentication", "calculate pricing")
       - repository_url: Optional GitHub URL to limit search
       - limit: Max results (default 10, max 50)

       Returns: Ranked list with code previews and links to details.
       Example: code://search?q=payment+processing&limit=5""",
       mime_type="text/markdown"
   )
   ```

2. **Document URI Parameters**
   ```python
   # Explain what entity_type and entity_id are
   "code://explain/{entity_type}/{entity_id}"

   # In description:
   entity_type: One of 'function', 'class', 'module'
   entity_id: Numeric ID from search results or other queries
   ```

3. **Standardize Response Formats**
   ```python
   # Document the response structure
   Returns: Markdown with sections:
   - Summary
   - Results (with specific fields)
   - Related resources
   ```

## Implementation Priority

### Phase 1: Critical Fixes (High Priority)
1. Add descriptions to all resource decorators
2. Document valid parameter values
3. Fix README to accurately describe tools vs resources
4. Add examples to all descriptions

### Phase 2: Usability Enhancements (Medium Priority)
1. Standardize URI patterns
2. Add mime_type to all resources
3. Create parameter validation with helpful errors
4. Add cross-references between related resources

### Phase 3: Advanced Features (Low Priority)
1. Add resource templates for parameter documentation
2. Create structured JSON responses for data-heavy resources
3. Add OpenAPI/AsyncAPI documentation generation
4. Create interactive documentation

## Testing Improvements

Create test suite for MCP usability:
```python
def test_all_resources_have_descriptions():
    """Ensure all resources have meaningful descriptions."""
    for resource in mcp.resources:
        assert resource.description is not None
        assert len(resource.description) > 100
        assert "example" in resource.description.lower()

def test_tools_have_parameter_documentation():
    """Ensure all tool parameters are documented."""
    for tool in mcp.tools:
        for param in tool.parameters:
            assert param.description is not None
            assert len(param.description) > 20
```

## Success Metrics

An MCP client should be able to:
1. Understand what each tool/resource does from its description alone
2. Know what parameters to provide without reading code
3. Predict the output format before calling
4. Find related tools/resources through cross-references
5. Recover from errors with helpful guidance

## Example: Improved Tool

```python
@mcp.tool(
    name="analyze_repository_coupling",
    description="""Analyze coupling between packages/modules in a repository to identify refactoring opportunities.

What it does:
- Calculates afferent coupling (dependencies on this module)
- Calculates efferent coupling (dependencies from this module)
- Identifies circular dependencies
- Suggests refactoring strategies

When to use:
- Preparing for modularization
- Investigating "spaghetti code"
- Planning service extraction
- Reducing build times

Parameters:
- repository_id: Get from list_repositories (shows all repos with IDs)
- package_path: Optional path like 'src/services' to focus analysis
- include_tests: Include test dependencies (default: false)

Returns: JSON with coupling metrics, circular dependencies, and specific refactoring suggestions.

Example response:
{
  "package": "src/services/auth",
  "afferent_coupling": 12,
  "efferent_coupling": 8,
  "instability": 0.4,
  "suggestions": [
    "Extract UserValidator to reduce coupling",
    "Consider facade pattern for auth subsystem"
  ]
}"""
)
async def analyze_coupling(
    repository_id: int,
    package_path: str = None,
    include_tests: bool = False
) -> dict:
    # Implementation
```

This level of documentation ensures any MCP client can use the tools effectively without additional context.
