# Tool to Resource Mapping

## Summary of Changes

### Removed Files (9 total)
1. ✅ `src/mcp_server/tools/code_analysis.py` - All functionality in resources
2. ✅ `src/mcp_server/tools/code_search.py` - All functionality in resources
3. ✅ `src/mcp_server/tools/package_analysis.py` - All functionality in resources
4. ✅ `src/mcp_server/tools/search.py` - Redundant with code://search
5. ✅ `src/mcp_server/tools/find.py` - Redundant with code://definitions
6. ✅ `src/mcp_server/tools/explain.py` - Redundant with code://explain
7. ✅ `src/mcp_server/tools/analyze.py` - Redundant with resources
8. ✅ `src/mcp_server/tools/migration_tools.py` - Functions in server.py
9. ✅ `src/mcp_server/tools/knowledge_tools.py` - Functions in server.py
10. ✅ `src/mcp_server/tools/execution_tools.py` - Functions in server.py
11. ✅ `src/mcp_server/tools/repository.py` - Redundant with RepositoryManagementTools
12. ✅ `src/mcp_server/tools/utils.py` - Unused utility functions

### Remaining Tool Classes (3 total)
1. **analysis_tools.py** - Advanced domain analysis (no resource equivalent)
2. **domain_tools.py** - DDD analysis tools (no resource equivalent)
3. **repository_management.py** - State-modifying repository operations

## Converted Tools (Now Resources)

### Code Analysis
- `get_code` → Resource: `code://explain/{entity_type}/{entity_id}`
- `find_usages` → Resource: `code://usages/{entity_type}/{entity_id}`
- `analyze_file` → Resource: `code://structure/{file_path}`
- `find_definition` → Resource: `code://definitions/{name}`
- `suggest_refactoring` → Resource: `code://refactoring/{entity_type}/{entity_id}`
- `find_similar_code` → Resource: `code://similar`

### Code Search
- `semantic_search` → Resource: `code://search`
- `search_code` → Resource: `code://search`

### Package Analysis
- `get_package_tree` → Resource: `packages://{repository_url}/tree`
- `get_package_dependencies` → Resource: `packages://{repository_url}/{package_path}/dependencies`
- `find_circular_dependencies` → Resource: `packages://{repository_url}/circular-dependencies`
- `get_package_coupling_metrics` → Resource: `packages://{repository_url}/{package_path}/coupling`
- `get_package_details` → Resource: `packages://{repository_url}/{package_path}/details`

### Migration Analysis
- `get_migration_readiness` → Resource: `migration://readiness/{repository_url}`
- `search_migration_patterns` → Resource: `migration://patterns/search`
- `get_pattern_library_stats` → Resource: `migration://patterns/stats`
- `get_migration_dashboard` → Resource: `migration://dashboard/{repository_url}`

### System/Repository Info
- `health_check` → Resource: `system://health`
- `get_system_stats` → Resource: `system://stats`
- `get_configuration` → Resource: `system://config`

## Tools to Remove (Redundant)

### Files to Delete
1. `src/mcp_server/tools/code_analysis.py` - All functionality in resources
2. `src/mcp_server/tools/code_search.py` - All functionality in resources
3. `src/mcp_server/tools/package_analysis.py` - All functionality in resources
4. `src/mcp_server/tools/search.py` - Redundant with code://search
5. `src/mcp_server/tools/find.py` - Redundant with code://definitions
6. `src/mcp_server/tools/explain.py` - Redundant with code://explain

## Tools to Keep (No Resource Equivalent)

### Domain Analysis (Keep)
- `extract_domain_model` - No resource equivalent
- `find_aggregate_roots` - No resource equivalent
- `analyze_bounded_context` - No resource equivalent
- `suggest_ddd_refactoring` - No resource equivalent
- `find_bounded_contexts` - No resource equivalent
- `generate_context_map` - No resource equivalent

### Advanced Analysis (Keep)
- `analyze_coupling` - No resource equivalent
- `suggest_context_splits` - No resource equivalent
- `detect_anti_patterns` - No resource equivalent
- `analyze_domain_evolution` - No resource equivalent

### Repository Management (Partial)
- `list_repositories` - Could be a resource
- `delete_repository` - Must remain a tool (modifies state)
- `update_embeddings` - Must remain a tool (modifies state)
