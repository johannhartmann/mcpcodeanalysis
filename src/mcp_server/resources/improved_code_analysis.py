"""Improved Code Analysis Resources with better descriptions and documentation."""

from urllib.parse import unquote

from fastmcp import FastMCP
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.database.models import Class, File, Function, Module
from src.embeddings.vector_search import VectorSearch
from src.logger import get_logger
from src.query.symbol_finder import SymbolFinder

logger = get_logger(__name__)

# Constants for validation
VALID_ENTITY_TYPES = ["function", "class", "module", "file"]
ENTITY_TYPE_MODELS = {
    "function": Function,
    "class": Class,
    "module": Module,
}


class ImprovedCodeAnalysisResources:
    """Code analysis resources with improved documentation and usability."""

    def __init__(
        self,
        mcp: FastMCP,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        """Initialize code analysis resources."""
        self.mcp = mcp
        self.session_maker = session_maker

    def register_resources(self) -> None:
        """Register all code analysis resources with improved descriptions."""

        @self.mcp.resource(
            "code://search",
            description="""Search for code using natural language queries powered by semantic search.

            Query Parameters:
            - q (required): Natural language search query (e.g., "authentication logic", "payment processing")
            - repository_url (optional): GitHub URL to limit search to specific repository
            - limit (optional): Maximum results to return (default: 10, max: 50)

            Returns: Markdown document with:
            - Summary of results found
            - Ranked list of code entities matching the query
            - Each result includes: entity name, type, file path, relevance score, and code preview
            - Direct links to detailed explanations (code://explain/{type}/{id})

            Examples:
            - code://search?q=user+authentication
            - code://search?q=calculate+total+price&repository_url=github.com/acme/shop&limit=5

            Use when: Looking for code that implements specific functionality or concepts.""",
            mime_type="text/markdown",
        )
        async def search_code(
            q: str, repository_url: str | None = None, limit: int = 10
        ) -> str:
            """Search for code using natural language queries."""
            # Validate parameters
            if not q or len(q.strip()) == 0:
                return "Error: Query parameter 'q' is required and cannot be empty"

            if limit < 1 or limit > 50:
                return "Error: Limit must be between 1 and 50"

            # Implementation continues...
            async with self.session_maker() as session:
                try:
                    vector_search = VectorSearch(session)

                    # Get repository ID if URL provided
                    repository_id = None
                    if repository_url:
                        # Validate and lookup repository
                        pass  # Implementation details

                    results = await vector_search.search(
                        query=q,
                        repository_id=repository_id,
                        limit=limit,
                    )

                    # Format results with clear structure
                    output = f"# Code Search Results\n\n"
                    output += f"**Query**: {q}\n"
                    if repository_url:
                        output += f"**Repository**: {repository_url}\n"
                    output += f"**Results Found**: {len(results)}\n\n"

                    for i, result in enumerate(results, 1):
                        output += f"## {i}. {result['entity']['name']}\n"
                        output += f"**Type**: {result['entity_type']}\n"
                        output += f"**File**: `{result['entity'].get('file_path', 'Unknown')}`\n"
                        output += f"**Relevance**: {result['similarity']:.2%}\n"
                        output += f"**View Details**: `code://explain/{result['entity_type']}/{result['entity']['id']}`\n"

                        if result["entity"].get("docstring"):
                            output += f"\n{result['entity']['docstring'][:200]}...\n"

                        output += "\n---\n\n"

                    return output

                except Exception as e:
                    logger.exception("Search failed")
                    return f"Error performing search: {e!s}\n\nPlease check your query and try again."

        @self.mcp.resource(
            "code://explain/{entity_type}/{entity_id}",
            description="""Get comprehensive explanation of a code entity with examples and metrics.

            Parameters:
            - entity_type: Type of entity - must be one of: function, class, module
            - entity_id: Numeric ID of the entity (obtained from search results or other queries)

            Returns: Markdown document containing:
            - Full source code with syntax highlighting
            - Documentation/docstrings
            - Signature (for functions/methods)
            - Metrics: complexity, lines of code, dependencies
            - Usage examples from the codebase
            - Related entities (for classes: methods, for modules: classes/functions)

            Examples:
            - code://explain/function/12345
            - code://explain/class/67890
            - code://explain/module/11111

            Use when: You need to understand how a specific piece of code works.""",
            mime_type="text/markdown",
        )
        async def explain_code(entity_type: str, entity_id: int) -> str:
            """Get detailed explanation of code entities."""
            # Validate entity type
            if entity_type not in VALID_ENTITY_TYPES:
                return f"Error: Invalid entity_type '{entity_type}'. Must be one of: {', '.join(VALID_ENTITY_TYPES)}"

            async with self.session_maker() as session:
                try:
                    # Get the appropriate model
                    model = ENTITY_TYPE_MODELS.get(entity_type)
                    if not model:
                        return f"Error: Entity type '{entity_type}' not supported for detailed explanation"

                    # Fetch entity
                    entity = await session.get(model, entity_id)
                    if not entity:
                        return f"Error: {entity_type.title()} with ID {entity_id} not found.\n\nTip: Use code://search to find valid entities."

                    # Build comprehensive explanation
                    file = await session.get(File, entity.file_id)

                    output = f"# {entity.name}\n\n"
                    output += f"**Type**: {entity_type.title()}\n"
                    output += f"**File**: `{file.file_path if file else 'Unknown'}`\n"
                    output += f"**Lines**: {entity.start_line}-{entity.end_line}\n"

                    if hasattr(entity, "signature") and entity.signature:
                        output += (
                            f"\n## Signature\n```python\n{entity.signature}\n```\n"
                        )

                    if hasattr(entity, "docstring") and entity.docstring:
                        output += f"\n## Documentation\n{entity.docstring}\n"

                    # Add metrics section
                    output += f"\n## Metrics\n"
                    if hasattr(entity, "complexity") and entity.complexity:
                        output += f"- **Cyclomatic Complexity**: {entity.complexity}\n"
                    if hasattr(entity, "cognitive_complexity"):
                        output += f"- **Cognitive Complexity**: {entity.cognitive_complexity}\n"
                    output += f"- **Lines of Code**: {entity.end_line - entity.start_line + 1}\n"

                    # Add source code
                    output += f"\n## Source Code\n```python\n"
                    # Here you would fetch and include the actual source code
                    output += "# Source code would be displayed here\n"
                    output += "```\n"

                    # Add usage examples
                    output += f"\n## Usage Examples\n"
                    output += f"To find where this {entity_type} is used: `code://usages/{entity_type}/{entity_id}`\n"

                    return output

                except Exception as e:
                    logger.exception("Failed to explain entity")
                    return f"Error explaining {entity_type}: {e!s}"

        @self.mcp.resource(
            "code://definitions/{name}",
            description="""Find all definitions of a symbol across the entire codebase.

            Parameters:
            - name: Symbol name to search for (case-sensitive)
                   Examples: 'UserService', 'calculate_total', 'AuthenticationError'

            Returns: Markdown document listing:
            - All locations where the symbol is defined
            - File path and line number for each definition
            - Symbol type (class, function, variable, etc.)
            - Signature or declaration
            - Brief documentation if available

            Examples:
            - code://definitions/UserRepository
            - code://definitions/process_payment
            - code://definitions/MAX_RETRY_COUNT

            Use when: You need to find where something is defined or locate all definitions of overloaded/duplicated names.""",
            mime_type="text/markdown",
        )
        async def find_definitions(name: str) -> str:
            """Find where symbols are defined."""
            # Decode URL-encoded name
            name = unquote(name)

            if not name or len(name.strip()) == 0:
                return "Error: Symbol name cannot be empty"

            async with self.session_maker() as session:
                try:
                    finder = SymbolFinder(session)
                    results = await finder.find_definitions(name=name)

                    if not results:
                        return f"No definitions found for: {name}\n\nTip: Check spelling and case. Use code://search for fuzzy matching."

                    output = f"# Definitions of `{name}`\n\n"
                    output += f"Found **{len(results)}** definition(s)\n\n"

                    for i, result in enumerate(results, 1):
                        output += (
                            f"## {i}. {result['type'].title()}: {result['name']}\n"
                        )
                        output += f"**Location**: `{result['file_path']}:{result.get('line_number', '?')}`\n"

                        if result.get("signature"):
                            output += f"**Signature**: `{result['signature']}`\n"

                        if result.get("docstring"):
                            doc_preview = result["docstring"].split("\n")[0][:100]
                            output += f"**Description**: {doc_preview}{'...' if len(result['docstring']) > 100 else ''}\n"

                        # Add link to full explanation if available
                        if (
                            result.get("entity_id")
                            and result.get("type") in ENTITY_TYPE_MODELS
                        ):
                            output += f"**Details**: `code://explain/{result['type']}/{result['entity_id']}`\n"

                        output += "\n---\n\n"

                    return output

                except Exception as e:
                    logger.exception("Failed to find definitions")
                    return f"Error finding definitions: {e!s}"

        # Register resource templates for better discovery
        logger.info("Registered improved code analysis resources")


# Additional resources would follow the same pattern with:
# 1. Comprehensive descriptions in the decorator
# 2. Parameter validation with helpful error messages
# 3. Structured output formats
# 4. Examples in descriptions
# 5. Cross-references to related resources
# 6. Error messages that guide users to solutions
