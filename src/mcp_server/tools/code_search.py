"""Code search tools for MCP server."""

from typing import Any

from fastmcp import FastMCP
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.embeddings.domain_search import DomainAwareSearch, DomainSearchScope
from src.embeddings.vector_search import SearchScope, VectorSearch
from src.logger import get_logger

logger = get_logger(__name__)


class SearchRequest(BaseModel):
    """Search request parameters."""

    query: str = Field(..., description="Search query text")
    scope: str = Field(
        default="all",
        description="Search scope: all, functions, classes, modules, repository, file",
    )
    repository_id: int | None = Field(
        None,
        description="Repository ID to limit search",
    )
    file_id: int | None = Field(None, description="File ID to limit search")
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results",
    )
    threshold: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold",
    )
    use_domain_knowledge: bool = Field(
        default=False,
        description="Use domain knowledge to enhance search",
    )
    bounded_context: str | None = Field(
        None,
        description="Limit search to specific bounded context",
    )


class SimilarCodeRequest(BaseModel):
    """Find similar code request."""

    embedding_id: int = Field(..., description="ID of embedding to find similar to")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of results")
    exclude_same_file: bool = Field(
        default=False,
        description="Exclude results from the same file",
    )


class BusinessCapabilitySearchRequest(BaseModel):
    """Search by business capability request."""

    capability: str = Field(..., description="Business capability description")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of results")


class CodeSnippetSearchRequest(BaseModel):
    """Search by code snippet request."""

    code_snippet: str = Field(..., description="Code snippet to search for")
    scope: str = Field(
        default="all",
        description="Search scope: all, functions, classes, modules",
    )
    repository_id: int | None = Field(
        None,
        description="Repository ID to limit search",
    )
    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of results")


class CodeSearchTools:
    """Code search tools for MCP."""

    def __init__(
        self,
        db_session: AsyncSession,
        mcp: FastMCP,
    ) -> None:
        """Initialize code search tools.

        Args:
            db_session: Database session
            mcp: FastMCP instance
        """
        self.db_session = db_session
        self.mcp = mcp
        self.vector_search = VectorSearch(db_session)
        self.domain_search = DomainAwareSearch(db_session)

    async def register_tools(self) -> None:
        """Register all code search tools."""
        if self.vector_search:
            # Semantic search tools
            @self.mcp.tool(
                name="semantic_search",
                description="Search for code using natural language queries",
            )
            async def semantic_search(request: SearchRequest) -> dict[str, Any]:
                """Search for code entities using semantic search.

                Args:
                    request: Search parameters

                Returns:
                    Search results with similarity scores
                """
                try:
                    # Use domain-aware search if requested
                    if request.use_domain_knowledge and self.domain_search:
                        # Convert to domain scope
                        domain_scope_map = {
                            "all": DomainSearchScope.ALL,
                            "functions": DomainSearchScope.DOMAIN_SERVICE,
                            "classes": DomainSearchScope.ENTITY,
                            "modules": DomainSearchScope.BOUNDED_CONTEXT,
                        }
                        domain_scope = domain_scope_map.get(
                            request.scope.lower(),
                            DomainSearchScope.ALL,
                        )

                        results = await self.domain_search.search_with_domain_context(
                            query=request.query,
                            scope=domain_scope,
                            bounded_context=request.bounded_context,
                            limit=request.limit,
                            include_related=True,
                        )
                    else:
                        # Standard vector search
                        scope = SearchScope[request.scope.upper()]
                        results = await self.vector_search.search(
                            query=request.query,
                            scope=scope,
                            repository_id=request.repository_id,
                            file_id=request.file_id,
                            limit=request.limit,
                            threshold=request.threshold,
                        )

                    return {
                        "success": True,
                        "query": request.query,
                        "results": results,
                        "count": len(results),
                        "search_type": (
                            "domain-aware"
                            if request.use_domain_knowledge
                            else "standard"
                        ),
                    }

                except (AttributeError, KeyError, ValueError, TypeError) as e:
                    logger.exception("Semantic search failed: %s")
                    return {
                        "success": False,
                        "error": str(e),
                        "results": [],
                    }

            @self.mcp.tool(
                name="find_similar_code",
                description="Find code similar to a given entity",
            )
            async def find_similar_code(request: SimilarCodeRequest) -> dict[str, Any]:
                """Find code entities similar to a given embedding.

                Args:
                    request: Similar code search parameters

                Returns:
                    Similar code entities
                """
                try:
                    results = await self.vector_search.search_similar(
                        embedding_id=request.embedding_id,
                        limit=request.limit,
                        exclude_same_file=request.exclude_same_file,
                    )

                    return {
                        "success": True,
                        "embedding_id": request.embedding_id,
                        "results": results,
                        "count": len(results),
                    }

                except (AttributeError, KeyError, ValueError, TypeError) as e:
                    logger.exception("Similar code search failed: %s")
                    return {
                        "success": False,
                        "error": str(e),
                        "results": [],
                    }

            @self.mcp.tool(
                name="search_by_code_snippet",
                description="Search for code similar to a given snippet",
            )
            async def search_by_code_snippet(
                request: CodeSnippetSearchRequest,
            ) -> dict[str, Any]:
                """Search for code similar to a provided snippet.

                Args:
                    request: Code snippet search parameters

                Returns:
                    Similar code entities
                """
                try:
                    # Convert scope string to enum
                    scope = SearchScope[request.scope.upper()]

                    results = await self.vector_search.search_by_code(
                        code_snippet=request.code_snippet,
                        scope=scope,
                        repository_id=request.repository_id,
                        limit=request.limit,
                    )

                    return {
                        "success": True,
                        "code_snippet": request.code_snippet[:100] + "...",
                        "results": results,
                        "count": len(results),
                    }

                except (AttributeError, KeyError, ValueError, TypeError) as e:
                    logger.exception("Code snippet search failed: %s")
                    return {
                        "success": False,
                        "error": str(e),
                        "results": [],
                    }

            # Business capability search
            if self.domain_search:

                @self.mcp.tool(
                    name="search_by_business_capability",
                    description="Search for code that implements a business capability",
                )
                async def search_by_business_capability(
                    request: BusinessCapabilitySearchRequest,
                ) -> dict[str, Any]:
                    """Search for code by business capability.

                    Args:
                        request: Business capability search parameters

                    Returns:
                        Code implementing the capability
                    """
                    try:
                        results = (
                            await self.domain_search.search_by_business_capability(
                                capability=request.capability,
                                limit=request.limit,
                            )
                        )

                        return {
                            "success": True,
                            "capability": request.capability,
                            "results": results,
                            "count": len(results),
                        }

                    except (AttributeError, KeyError, ValueError, TypeError) as e:
                        logger.exception("Business capability search failed: %s")
                        return {
                            "success": False,
                            "error": str(e),
                            "results": [],
                        }

        # Keyword search tools (always available)
        @self.mcp.tool(
            name="keyword_search",
            description="Search for code using keywords",
        )
        async def keyword_search(
            query: str = Field(..., description="Keyword search query"),
            entity_type: str | None = Field(
                None,
                description="Entity type: function, class, module",
            ),
            repository_id: int | None = Field(
                None,
                description="Repository ID to limit search",
            ),
            limit: int = Field(default=20, ge=1, le=100),
        ) -> dict[str, Any]:
            """Search for code entities using keywords.

            Args:
                query: Keyword query
                entity_type: Optional entity type filter
                repository_id: Optional repository filter
                limit: Maximum results

            Returns:
                Matching code entities
            """
            try:
                from sqlalchemy import or_, select

                from src.database.models import Class, Function, Module

                results = []

                # Search functions
                if not entity_type or entity_type == "function":
                    func_query = select(Function).where(
                        or_(
                            Function.name.ilike(f"%{query}%"),
                            Function.docstring.ilike(f"%{query}%"),
                        ),
                    )
                    if repository_id:
                        func_query = func_query.join(Function.file).where(
                            Function.file.has(repository_id=repository_id),
                        )
                    func_query = func_query.limit(limit)

                    func_result = await self.db_session.execute(func_query)
                    functions = func_result.scalars().all()

                    results.extend(
                        {
                            "type": "function",
                            "id": func.id,
                            "name": func.name,
                            "file_id": func.file_id,
                            "start_line": func.start_line,
                            "end_line": func.end_line,
                            "docstring": func.docstring,
                        }
                        for func in functions
                    )

                # Search classes
                if not entity_type or entity_type == "class":
                    class_query = select(Class).where(
                        or_(
                            Class.name.ilike(f"%{query}%"),
                            Class.docstring.ilike(f"%{query}%"),
                        ),
                    )
                    if repository_id:
                        class_query = class_query.join(Class.file).where(
                            Class.file.has(repository_id=repository_id),
                        )
                    class_query = class_query.limit(limit)

                    class_result = await self.db_session.execute(class_query)
                    classes = class_result.scalars().all()

                    results.extend(
                        {
                            "type": "class",
                            "id": cls.id,
                            "name": cls.name,
                            "file_id": cls.file_id,
                            "start_line": cls.start_line,
                            "end_line": cls.end_line,
                            "docstring": cls.docstring,
                        }
                        for cls in classes
                    )

                # Search modules
                if not entity_type or entity_type == "module":
                    module_query = select(Module).where(
                        or_(
                            Module.name.ilike(f"%{query}%"),
                            Module.docstring.ilike(f"%{query}%"),
                        ),
                    )
                    if repository_id:
                        module_query = module_query.join(Module.file).where(
                            Module.file.has(repository_id=repository_id),
                        )
                    module_query = module_query.limit(limit)

                    module_result = await self.db_session.execute(module_query)
                    modules = module_result.scalars().all()

                    results.extend(
                        {
                            "type": "module",
                            "id": module.id,
                            "name": module.name,
                            "file_id": module.file_id,
                            "start_line": module.start_line,
                            "end_line": module.end_line,
                            "docstring": module.docstring,
                        }
                        for module in modules
                    )

                # Sort by relevance (name matches first)
                results.sort(
                    key=lambda x: (
                        0 if query.lower() in x["name"].lower() else 1,
                        x["name"],
                    ),
                )

                return {
                    "success": True,
                    "query": query,
                    "results": results[:limit],
                    "count": len(results),
                }

            except (AttributeError, KeyError, ValueError, TypeError) as e:
                logger.exception("Keyword search failed: %s")
                return {
                    "success": False,
                    "error": str(e),
                    "results": [],
                }

        logger.info("Code search tools registered")
