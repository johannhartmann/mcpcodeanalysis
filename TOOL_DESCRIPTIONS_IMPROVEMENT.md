# MCP Tool Description Improvements

## Current Issues
1. Core analysis functions are resources, not tools - this is confusing
2. Tool descriptions lack context and purpose
3. No examples of usage
4. Technical terms without explanation

## Suggested Improvements

### Repository Management Tools

**Current:** "Scan or rescan a repository"
**Improved:** "Scan a repository to update the code analysis database with latest changes. Use after code updates or to fix sync issues. Processes new/modified files and updates search indexes."

**Current:** "Update embeddings for a repository"
**Improved:** "Regenerate the semantic search index for a repository. This improves code search accuracy. Use after major code changes or if search results seem outdated."

### Domain Analysis Tools

**Current:** "Extract domain entities and relationships from code"
**Improved:** "Analyze code to identify business domain concepts like Customer, Order, Payment. Uses AI to understand business logic beyond just class names. Returns entities with their business rules, relationships, and responsibilities."

### Migration Tools

**Current:** "Start execution of a migration step"
**Improved:** "Begin working on a specific step from a migration plan. Prerequisites: Must have created a migration plan first. Changes step status to 'in_progress' and tracks execution."

## Better Parameter Descriptions

**Current:**
```python
repository_id: int = Field(..., description="Repository ID")
```

**Improved:**
```python
repository_id: int = Field(
    ...,
    description="Repository ID (get from list_repositories or add_repository response)"
)
```

## Tool Naming Suggestions

Group tools by prefix for better organization:
- `repo_add`, `repo_list`, `repo_scan`
- `ddd_extract_entities`, `ddd_find_contexts`
- `migration_create_plan`, `migration_start_step`

## Add Usage Examples in Descriptions

```python
@mcp.tool(
    name="analyze_coupling",
    description="""Analyze coupling between code components to identify refactoring opportunities.

    Example: analyze_coupling(class_name="OrderService") returns:
    - Afferent coupling: 5 classes depend on OrderService
    - Efferent coupling: OrderService depends on 8 classes
    - Suggestions: Extract PaymentProcessor to reduce coupling

    Use when: Preparing for refactoring or investigating tight coupling."""
)
```

## Critical: Document Tool vs Resource Distinction

Add clear documentation explaining:
- **Tools**: Perform actions, modify state (add repo, create plan, start migration)
- **Resources**: Read-only data access (search, view, analyze existing data)

Many "tools" mentioned in docs are actually resources:
- `search_code` → `code://search`
- `explain_code` → `code://explain/{entity_type}/{entity_id}`
- `find_definition` → `code://definitions/{name}`
