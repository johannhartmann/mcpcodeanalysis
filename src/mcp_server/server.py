"""MCP Code Analysis Server implementation - Fixed version."""

from collections.abc import AsyncGenerator
from typing import Any, cast

from fastmcp import FastMCP
from pydantic import Field
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from starlette.requests import Request
from starlette.responses import JSONResponse

from src.database.init_db import get_session_factory, init_database
from src.database.models import Repository
from src.logger import get_logger
from src.mcp_server.tools.code_analysis import CodeAnalysisTools
from src.mcp_server.tools.code_search import CodeSearchTools
from src.mcp_server.tools.domain_tools import DomainTools
from src.mcp_server.tools.package_analysis import (
    AnalyzePackagesRequest,
    FindCircularDependenciesRequest,
    GetPackageCouplingRequest,
    GetPackageDependenciesRequest,
    GetPackageDetailsRequest,
    GetPackageTreeRequest,
    analyze_packages,
    find_circular_dependencies,
    get_package_coupling_metrics,
    get_package_dependencies,
    get_package_details,
    get_package_tree,
)
from src.models import RepositoryConfig

logger = get_logger(__name__)

# Create the global FastMCP instance
mcp: FastMCP = FastMCP("Code Analysis Server")

# Global variables for shared resources
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None
_settings: Any | None = None


async def initialize_server() -> None:
    """Initialize server resources."""
    global _engine, _session_factory

    if _engine is not None:
        return  # Already initialized

    logger.info("Starting MCP Code Analysis Server")

    # Load settings
    # settings imported globally from src.config

    # Initialize database
    logger.info("Initializing database connection")
    _engine = await init_database()
    _session_factory = get_session_factory(_engine)

    # OpenAI initialization is now handled by individual components using LangChain

    logger.info("Server initialized successfully")


async def get_db_session() -> AsyncGenerator[Any, None]:
    """Get a database session."""
    if _session_factory is None:
        await initialize_server()

    if _session_factory is None:
        raise RuntimeError

    async with _session_factory() as session:
        yield session


# Register repository management tools
@mcp.tool(name="add_repository", description="Add a new repository to track")
async def add_repository(
    url: str = Field(description="Repository URL (GitHub or file://)"),
    branch: str | None = Field(default=None, description="Branch to track (optional)"),
    scan_immediately: bool = Field(
        default=True,
        description="Scan repository immediately",
    ),
    generate_embeddings: bool = Field(
        default=True,
        description="Generate embeddings for code",
    ),
) -> dict[str, Any]:
    """Add a new repository to track."""
    await initialize_server()

    async for session in get_db_session():
        try:
            # Check for existing repository
            result = await session.execute(
                select(Repository).where(Repository.github_url == url),
            )
            existing_repo = result.scalar_one_or_none()

            if existing_repo:
                return {
                    "success": False,
                    "error": f"Repository already exists with ID {existing_repo.id}",
                    "repository_id": existing_repo.id,
                }

            # Add repository to database
            repo = Repository(
                github_url=url,
                owner=url.split("/")[-2] if "/" in url else "local",
                name=url.split("/")[-1].replace(".git", "") if "/" in url else url,
                default_branch=branch or "main",
            )
            session.add(repo)
            await session.commit()
            await session.refresh(repo)

            scan_result: dict[str, Any] = {"repository_id": int(repo.id)}

            # Scan if requested
            if scan_immediately:
                from src.scanner.repository_scanner import RepositoryScanner

                repo_config = RepositoryConfig(
                    url=url,
                    branch=branch,
                )

                scanner = RepositoryScanner(session)
                scan_result = await scanner.scan_repository(repo_config)

                # Generate embeddings if requested
                if generate_embeddings:
                    from src.embeddings.embedding_service import EmbeddingService

                    embedding_service = EmbeddingService(session)
                    embedding_result = (
                        await embedding_service.create_repository_embeddings(
                            scan_result["repository_id"],
                        )
                    )
                    scan_result["embeddings"] = embedding_result

            return {
                "success": True,
                "repository": {
                    "id": repo.id,
                    "url": url,
                    "branch": branch or "default",
                },
                "scan_result": scan_result,
            }
        except Exception as e:
            logger.exception("Failed to add repository: %s")
            return {
                "success": False,
                "error": str(e),
            }
    return {"success": False, "error": "No session"}


@mcp.tool(name="list_repositories", description="List all tracked repositories")
async def list_repositories(
    include_stats: bool = Field(
        default=False,
        description="Include repository statistics",
    ),
) -> dict[str, Any]:
    """List all tracked repositories."""
    await initialize_server()

    async for session in get_db_session():
        try:
            # Get repositories
            result = await session.execute(select(Repository).order_by(Repository.name))
            repositories = result.scalars().all()

            repo_list = []
            for repo in repositories:
                repo_data = {
                    "id": repo.id,
                    "name": repo.name,
                    "owner": repo.owner,
                    "url": repo.github_url,
                    "default_branch": repo.default_branch,
                    "last_synced": (
                        repo.last_synced.isoformat() if repo.last_synced else None
                    ),
                }

                if include_stats:
                    # Get file count
                    from sqlalchemy import func

                    from src.database.models import File
                    from src.embeddings.vector_search import VectorSearch

                    file_count_result = await session.execute(
                        select(func.count(File.id)).where(
                            File.repository_id == repo.id,
                        ),
                    )
                    file_count = file_count_result.scalar() or 0

                    # Get embedding stats via vector search (includes totals and breakdown)
                    vector_search = VectorSearch(session)
                    embedding_stats = await vector_search.get_repository_stats(
                        cast("int", repo.id)
                    )

                    repo_data["stats"] = {
                        "file_count": file_count,
                        "total_embeddings": embedding_stats.get("total_embeddings", 0),
                        "embeddings_by_type": embedding_stats.get(
                            "embeddings_by_type", {}
                        ),
                    }

                repo_list.append(repo_data)

            return {
                "success": True,
                "repositories": repo_list,
                "count": len(repo_list),
            }
        except Exception as e:
            logger.exception("Failed to list repositories: %s")
            return {
                "success": False,
                "error": str(e),
                "repositories": [],
            }
    return {"success": False, "repositories": [], "error": "No session"}


@mcp.tool(name="scan_repository", description="Scan or rescan a repository")
async def scan_repository(
    repository_id: int = Field(description="Repository ID to scan"),
    force_full_scan: bool = Field(default=False, description="Force full rescan"),
    generate_embeddings: bool = Field(default=True, description="Generate embeddings"),
) -> dict[str, Any]:
    """Scan or rescan a repository."""
    await initialize_server()

    async for session in get_db_session():
        # Lookup repository
        result = await session.execute(
            select(Repository).where(Repository.id == repository_id)
        )
        repo = result.scalar_one_or_none()
        if not repo:
            return {"success": False, "error": f"Repository {repository_id} not found"}

        # Build RepositoryConfig
        repo_config = RepositoryConfig(
            url=cast("str", repo.github_url),
            branch=cast("str | None", repo.default_branch),
        )

        # Perform scan
        from src.scanner.repository_scanner import RepositoryScanner

        scanner = RepositoryScanner(session)
        scan_result = await scanner.scan_repository(
            repo_config, force_full_scan=force_full_scan
        )

        # Optionally generate embeddings
        if generate_embeddings:
            from src.embeddings.embedding_service import EmbeddingService

            embedding_service = EmbeddingService(session)
            embeddings_info = await embedding_service.create_repository_embeddings(
                int(repository_id)
            )
            scan_result["embeddings"] = embeddings_info

        return {
            "success": True,
            "repository": {
                "id": repo.id,
                "name": repo.name,
                "url": repo.github_url,
            },
            "scan_result": scan_result,
        }
    return {"success": False, "error": "No session"}


@mcp.tool(name="remove_repository", description="Remove a repository from tracking")
async def remove_repository(
    repository_id: int = Field(description="Repository ID to remove"),
) -> dict[str, Any]:
    """Remove a repository from tracking."""
    await initialize_server()

    async for session in get_db_session():
        # Find repository
        result = await session.execute(
            select(Repository).where(Repository.id == repository_id)
        )
        repo = result.scalar_one_or_none()
        if not repo:
            return {"success": False, "error": f"Repository {repository_id} not found"}
        try:
            await session.delete(repo)
            await session.commit()
            return {
                "success": True,
                "repository": {
                    "id": repository_id,
                    "url": getattr(repo, "github_url", None),
                    "name": getattr(repo, "name", None),
                },
                "message": "Repository deleted successfully",
            }
        except SQLAlchemyError as e:
            await session.rollback()
            return {"success": False, "error": str(e)}
    return {"success": False, "error": "No session"}


@mcp.tool(name="update_repository_settings", description="Update repository settings")
async def update_repository_settings(
    repository_id: int = Field(description="Repository ID"),
    branch: str | None = Field(default=None, description="New branch to track"),
    auto_scan: bool | None = Field(
        default=None, description="Enable automatic scanning"
    ),
) -> dict[str, Any]:
    """Update repository settings."""
    await initialize_server()

    async for session in get_db_session():
        # Find repository
        result = await session.execute(
            select(Repository).where(Repository.id == repository_id)
        )
        _ = auto_scan  # Unused but accepted for compatibility

        repo = result.scalar_one_or_none()
        if not repo:
            return {"success": False, "error": f"Repository {repository_id} not found"}

        updated_fields: dict[str, Any] = {}
        if branch is not None:
            repo.default_branch = branch
            updated_fields["branch"] = branch

        # Note: auto_scan setting is not stored in Repository model; ignored if provided
        try:
            await session.commit()
            return {
                "success": True,
                "repository": {
                    "id": repository_id,
                    "url": getattr(repo, "github_url", None),
                    "name": getattr(repo, "name", None),
                    "branch": getattr(repo, "default_branch", None),
                },
                "updated": updated_fields,
            }
        except SQLAlchemyError as e:
            await session.rollback()
            return {"success": False, "error": str(e)}
    return {"success": False, "error": "No session"}


# Register code search tools
@mcp.tool(name="semantic_search", description="Search code using natural language")
async def semantic_search(
    query: str = Field(description="Natural language search query"),
    repository_id: int | None = Field(
        default=None,
        description="Limit to specific repository",
    ),
    limit: int = Field(default=10, description="Maximum results to return"),
) -> dict[str, Any]:
    """Search code using natural language."""
    await initialize_server()

    async for session in get_db_session():
        search_tools = CodeSearchTools(session, mcp)
        from src.embeddings.vector_search import SearchScope

        try:
            results = await search_tools.vector_search.search(
                query=query,
                scope=SearchScope.ALL,
                repository_id=repository_id,
                file_id=None,
                limit=limit,
                threshold=None,
            )
            return {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results),
            }
        except Exception as e:
            logger.exception("Semantic search failed")
            return {"success": False, "error": str(e), "results": []}
    return {"success": False, "error": "No session", "results": []}


@mcp.tool(name="keyword_search", description="Search code using keywords")
async def keyword_search(
    keywords: list[str] = Field(description="Keywords to search for"),
    scope: str = Field(
        default="all",
        description="Search scope: all, functions, classes, modules",
    ),
    repository_id: int | None = Field(
        default=None,
        description="Limit to specific repository",
    ),
    limit: int = Field(default=20, description="Maximum results"),
) -> dict[str, Any]:
    """Search code using keywords."""
    await initialize_server()

    async for session in get_db_session():
        try:
            # Build keyword conditions
            from typing import Any as _Any

            from sqlalchemy import or_, select

            from src.database.models import Class, Function, Module

            def ilike_any(field: _Any) -> _Any:
                return (
                    or_(*[field.ilike(f"%{kw}%") for kw in keywords])
                    if keywords
                    else field.ilike("%%")
                )

            results: list[dict[str, Any]] = []
            query_text = " ".join(keywords)

            # Functions
            if scope in ("all", "functions"):
                func_query = select(Function).where(
                    or_(ilike_any(Function.name), ilike_any(Function.docstring))
                )
                if repository_id is not None:
                    func_query = func_query.join(Function.file).where(
                        Function.file.has(repository_id=repository_id)
                    )
                func_query = func_query.limit(limit)
                func_result = await session.execute(func_query)
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

            # Classes
            if scope in ("all", "classes"):
                class_query = select(Class).where(
                    or_(ilike_any(Class.name), ilike_any(Class.docstring))
                )
                if repository_id is not None:
                    class_query = class_query.join(Class.file).where(
                        Class.file.has(repository_id=repository_id)
                    )
                class_query = class_query.limit(limit)
                class_result = await session.execute(class_query)
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

            # Modules
            if scope in ("all", "modules"):
                module_query = select(Module).where(
                    or_(ilike_any(Module.name), ilike_any(Module.docstring))
                )
                if repository_id is not None:
                    module_query = module_query.join(Module.file).where(
                        Module.file.has(repository_id=repository_id)
                    )
                module_query = module_query.limit(limit)
                module_result = await session.execute(module_query)
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

            # Sort results: direct name hits first
            results.sort(
                key=lambda x: (
                    0 if any(kw.lower() in x["name"].lower() for kw in keywords) else 1,
                    x["name"],
                )
            )

            return {
                "success": True,
                "query": query_text,
                "results": results[:limit],
                "count": len(results),
            }
        except Exception as e:
            logger.exception("Keyword search failed")
            return {"success": False, "error": str(e), "results": []}
    return {"success": False, "error": "No session", "results": []}


@mcp.tool(name="find_similar_code", description="Find code similar to a given snippet")
async def find_similar_code(
    code_snippet: str = Field(description="Code snippet to find similar code for"),
    repository_id: int | None = Field(
        default=None,
        description="Limit to specific repository",
    ),
    threshold: float = Field(default=0.7, description="Similarity threshold (0-1)"),
    limit: int = Field(default=10, description="Maximum results"),
) -> dict[str, Any]:
    """Find code similar to a given snippet."""
    await initialize_server()

    async for session in get_db_session():
        search_tools = CodeSearchTools(session, mcp)
        # Register the tools first
        await search_tools.register_tools()

        # Use search_by_code_snippet which accepts a code snippet
        from src.embeddings.vector_search import SearchScope

        try:
            results = await search_tools.vector_search.search_by_code(
                code_snippet=code_snippet,
                scope=SearchScope.ALL,
                repository_id=repository_id,
                limit=limit,
            )

            return {
                "success": True,
                "code_snippet": (
                    code_snippet[:100] + "..."
                    if len(code_snippet) > 100
                    else code_snippet
                ),
                "results": results,
                "count": len(results),
                "threshold": threshold,  # Note: threshold is not used in search_by_code
            }
        except Exception as e:
            logger.exception("Find similar code failed")
            return {
                "success": False,
                "error": str(e),
                "results": [],
            }
    return {"success": False, "error": "No session", "results": []}


# Register code analysis tools
@mcp.tool(name="get_code", description="Get code for a specific entity")
async def get_code(
    entity_type: str = Field(description="Entity type: function, class, or module"),
    entity_id: int = Field(description="Entity ID"),
    include_context: bool = Field(
        default=False,
        description="Include surrounding context",
    ),
) -> dict[str, Any]:
    """Get code for a specific entity."""
    await initialize_server()

    async for session in get_db_session():
        try:
            from pathlib import Path

            from sqlalchemy.orm import selectinload

            from src.database.models import Class, Function, Module

            # Fetch entity and its file
            entity = None
            file_path: str | None = None
            start_line: int | None = None
            end_line: int | None = None
            name: str | None = None

            if entity_type == "function":
                result = await session.execute(
                    select(Function)
                    .where(Function.id == entity_id)
                    .options(selectinload(Function.file))
                )
                entity = result.scalar_one_or_none()
            elif entity_type == "class":
                result = await session.execute(
                    select(Class)
                    .where(Class.id == entity_id)
                    .options(selectinload(Class.file))
                )
                entity = result.scalar_one_or_none()
            elif entity_type == "module":
                result = await session.execute(
                    select(Module)
                    .where(Module.id == entity_id)
                    .options(selectinload(Module.file))
                )
                entity = result.scalar_one_or_none()
            else:
                return {
                    "success": False,
                    "error": f"Unknown entity_type: {entity_type}",
                }

            if not entity:
                return {
                    "success": False,
                    "error": f"{entity_type} {entity_id} not found",
                }

            file_path = cast("str", entity.file.path)
            start_line = cast("int", entity.start_line)
            end_line = cast("int", entity.end_line)
            name = cast("str", entity.name)

            analysis_tools = CodeAnalysisTools(session, mcp)
            raw_content, contextual_content = (
                analysis_tools.code_extractor.get_entity_content(
                    Path(file_path),
                    entity_type,
                    start_line,
                    end_line,
                    include_context=include_context,
                )
            )

            return {
                "success": True,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "name": name,
                "file_path": file_path,
                "start_line": start_line,
                "end_line": end_line,
                "code": contextual_content if include_context else raw_content,
                "raw_code": raw_content,
            }
        except Exception as e:
            logger.exception("Error in get_code")
            return {"success": False, "error": str(e)}
    return {"success": False, "error": "No session"}


@mcp.tool(name="analyze_file", description="Analyze a specific file")
async def analyze_file(
    file_path: str = Field(description="File path within repository"),
    repository_id: int = Field(description="Repository ID"),
) -> dict[str, Any]:
    """Analyze a specific file."""
    await initialize_server()

    async for session in get_db_session():
        try:
            from sqlalchemy import select

            from src.database.models import File

            # Locate file record
            file_result = await session.execute(
                select(File).where(
                    (File.path == file_path) & (File.repository_id == repository_id)
                )
            )
            file = file_result.scalar_one_or_none()
            if not file:
                return {
                    "success": False,
                    "error": f"File not found: {file_path} in repository {repository_id}",
                }

            # Use CodeProcessor to get structure
            from src.scanner.code_processor import CodeProcessor

            processor = CodeProcessor(session)
            structure = await processor.get_file_structure(file)

            # Derive simple metrics
            modules = structure.get("modules", [])
            classes = structure.get("classes", [])
            functions = structure.get("functions", [])
            imports = structure.get("imports", [])

            metrics = {
                "modules": len(modules),
                "classes": len(classes),
                "functions": sum(1 for f in functions if f.get("class_id") is None),
                "methods": sum(1 for f in functions if f.get("class_id") is not None),
                "imports": len(imports),
            }

            return {
                "success": True,
                "file": structure.get("file", {}),
                "structure": {
                    "modules": modules,
                    "classes": classes,
                    "functions": functions,
                },
                "imports": imports,
                "metrics": metrics,
            }
        except Exception as e:
            logger.exception("Error in analyze_file")
            return {"success": False, "error": str(e)}
    return {"success": False, "error": "No session"}


@mcp.tool(name="get_file_structure", description="Get the structure of a file")
async def get_file_structure(
    file_path: str = Field(description="File path within repository"),
    repository_id: int = Field(description="Repository ID"),
    include_imports: bool = Field(
        default=True,
        description="Include import statements",
    ),
) -> dict[str, Any]:
    """Get the structure of a file."""
    await initialize_server()

    async for session in get_db_session():
        try:
            from sqlalchemy import select

            from src.database.models import File
            from src.scanner.code_processor import CodeProcessor

            file_result = await session.execute(
                select(File).where(
                    (File.path == file_path) & (File.repository_id == repository_id)
                )
            )
            file = file_result.scalar_one_or_none()
            if not file:
                return {
                    "success": False,
                    "error": f"File not found: {file_path} in repository {repository_id}",
                }

            processor = CodeProcessor(session)
            structure = await processor.get_file_structure(file)
            if not include_imports:
                structure = {**structure, "imports": []}
            return {"success": True, **structure}
        except Exception as e:
            logger.exception("Error in get_file_structure")
            return {"success": False, "error": str(e)}
    return {"success": False, "error": "No session"}


@mcp.tool(
    name="analyze_dependencies",
    description="Analyze dependencies of a module or file",
)
async def analyze_dependencies(
    file_path: str = Field(description="File path to analyze"),
    repository_id: int = Field(description="Repository ID"),
    depth: int = Field(default=1, description="Depth of dependency analysis"),
) -> dict[str, Any]:
    """Analyze dependencies of a module or file."""
    await initialize_server()

    async for session in get_db_session():
        try:
            from sqlalchemy import select

            from src.database.models import File
            from src.scanner.code_processor import CodeProcessor

            file_result = await session.execute(
                select(File).where(
                    (File.path == file_path) & (File.repository_id == repository_id)
                )
            )
            file = file_result.scalar_one_or_none()
            if not file:
                return {
                    "success": False,
                    "error": f"File not found: {file_path} in repository {repository_id}",
                }

            processor = CodeProcessor(session)
            structure = await processor.get_file_structure(file)
            imports = structure.get("imports", [])
            depends_on = [imp.get("from") or imp.get("statement") for imp in imports]
            return {
                "success": True,
                "file": structure.get("file", {}),
                "imports": imports,
                "dependency_graph": {
                    "node": structure.get("file", {}).get("path"),
                    "depends_on": depends_on,
                },
                "depth": depth,
            }
        except Exception as e:
            logger.exception("Error in analyze_dependencies")
            return {"success": False, "error": str(e)}
    return {"success": False, "error": "No session"}


@mcp.tool(
    name="analyze_package_structure",
    description="Analyze the package structure of a repository",
)
async def analyze_package_structure_tool(
    repository_id: int = Field(description="Repository ID to analyze"),
    force_refresh: bool = Field(
        default=False, description="Force re-analysis even if data exists"
    ),
) -> dict[str, Any]:
    """Analyze the package structure of a repository."""
    await initialize_server()

    async for session in get_db_session():
        return await analyze_packages(
            AnalyzePackagesRequest(
                repository_id=repository_id, force_refresh=force_refresh
            ),
            session,
        )
    return {"success": False, "error": "No session"}


@mcp.tool(
    name="get_package_tree",
    description="Get the hierarchical package structure",
)
async def get_package_tree_tool(
    repository_id: int = Field(description="Repository ID"),
) -> dict[str, Any]:
    """Get the hierarchical package structure of a repository."""
    await initialize_server()

    async for session in get_db_session():
        return await get_package_tree(
            GetPackageTreeRequest(repository_id=repository_id), session
        )
    return {"success": False, "error": "No session"}


@mcp.tool(
    name="get_package_details",
    description="Get detailed information about a specific package",
)
async def get_package_details_tool(
    repository_id: int = Field(description="Repository ID"),
    package_path: str = Field(description="Package path (e.g., 'src/utils')"),
) -> dict[str, Any]:
    """Get detailed information about a specific package."""
    await initialize_server()

    async for session in get_db_session():
        return await get_package_details(
            GetPackageDetailsRequest(
                repository_id=repository_id, package_path=package_path
            ),
            session,
        )

    return {"success": False, "error": "No session"}


@mcp.tool(
    name="get_package_dependencies",
    description="Get dependencies for a specific package",
)
async def get_package_dependencies_tool(
    repository_id: int = Field(description="Repository ID"),
    package_path: str = Field(description="Package path"),
    direction: str = Field(
        default="both", description="Direction: 'imports', 'imported_by', or 'both'"
    ),
) -> dict[str, Any]:
    """Get dependencies for a specific package."""
    await initialize_server()

    async for session in get_db_session():
        return await get_package_dependencies(
            GetPackageDependenciesRequest(
                repository_id=repository_id,
                package_path=package_path,
                direction=direction,
            ),
            session,
        )
    return {"success": False, "error": "No session"}


@mcp.tool(
    name="find_circular_dependencies",
    description="Find circular dependencies between packages",
)
async def find_circular_dependencies_tool(
    repository_id: int = Field(description="Repository ID"),
) -> dict[str, Any]:
    """Find circular dependencies between packages."""
    await initialize_server()

    async for session in get_db_session():
        return await find_circular_dependencies(
            FindCircularDependenciesRequest(repository_id=repository_id), session
        )

    return {"success": False, "error": "No session"}


@mcp.tool(
    name="get_package_coupling_metrics",
    description="Get coupling metrics for all packages",
)
async def get_package_coupling_metrics_tool(
    repository_id: int = Field(description="Repository ID"),
) -> dict[str, Any]:
    """Get coupling metrics for all packages in a repository."""
    await initialize_server()

    async for session in get_db_session():
        return await get_package_coupling_metrics(
            GetPackageCouplingRequest(repository_id=repository_id), session
        )
    return {"success": False, "error": "No session"}


# Register domain-driven design analysis tools
@mcp.tool(
    name="extract_domain_model",
    description="Extract domain entities and relationships from code using LLM analysis",
)
async def extract_domain_model(
    code_path: str = Field(description="Path to file or module to analyze"),
    include_relationships: bool = Field(
        default=True,
        description="Whether to extract relationships",
    ),
) -> dict[str, Any]:
    """Extract domain model from code."""
    await initialize_server()

    async for session in get_db_session():
        domain_tools = DomainTools(session, mcp)
        return await domain_tools.extract_domain_model(code_path, include_relationships)
    return {"success": False, "error": "No session"}


@mcp.tool(
    name="find_aggregate_roots",
    description="Find aggregate roots in the codebase using domain analysis",
)
async def find_aggregate_roots(
    context_name: str | None = Field(
        default=None,
        description="Optional bounded context to search within",
    ),
) -> list[dict[str, Any]]:
    """Find aggregate roots."""
    await initialize_server()

    async for session in get_db_session():
        domain_tools = DomainTools(session, mcp)
        return await domain_tools.find_aggregate_roots(context_name)
    return []


@mcp.tool(
    name="analyze_bounded_context",
    description="Analyze a bounded context and its relationships",
)
async def analyze_bounded_context(
    context_name: str = Field(description="Name of the bounded context"),
) -> dict[str, Any]:
    """Analyze bounded context."""
    await initialize_server()

    async for session in get_db_session():
        domain_tools = DomainTools(session, mcp)
        return await domain_tools.analyze_bounded_context(context_name)
    return {}


@mcp.tool(
    name="suggest_ddd_refactoring",
    description="Suggest Domain-Driven Design refactoring improvements",
)
async def suggest_ddd_refactoring(
    code_path: str = Field(description="Path to analyze"),
) -> list[dict[str, Any]]:
    """Suggest DDD refactoring."""
    await initialize_server()

    async for session in get_db_session():
        domain_tools = DomainTools(session, mcp)
        return await domain_tools.suggest_ddd_refactoring(code_path)
    return []


@mcp.tool(
    name="find_bounded_contexts",
    description="Find all bounded contexts in the codebase",
)
async def find_bounded_contexts(
    min_entities: int = Field(
        default=3,
        description="Minimum number of entities for a context",
    ),
) -> list[dict[str, Any]]:
    """Find bounded contexts."""
    await initialize_server()

    async for session in get_db_session():
        domain_tools = DomainTools(session, mcp)
        return await domain_tools.find_bounded_contexts(min_entities)
    return []


@mcp.tool(
    name="generate_context_map",
    description="Generate a context map showing relationships between bounded contexts",
)
async def generate_context_map(
    output_format: str = Field(
        default="json",
        description="Output format: json, mermaid, or plantuml",
    ),
) -> dict[str, Any]:
    """Generate context map."""
    await initialize_server()

    async for session in get_db_session():
        domain_tools = DomainTools(session, mcp)
        return await domain_tools.generate_context_map(output_format)
    return {}


# Register advanced analysis tools
@mcp.tool(
    name="analyze_coupling",
    description="Analyze coupling between bounded contexts with metrics and recommendations",
)
async def analyze_coupling(
    repository_id: int | None = Field(
        default=None,
        description="Optional repository ID to filter analysis",
    ),
) -> dict[str, Any]:
    """Analyze cross-context coupling."""
    await initialize_server()

    async for session in get_db_session():
        from src.domain.pattern_analyzer import DomainPatternAnalyzer

        analyzer = DomainPatternAnalyzer(session)
        return await analyzer.analyze_cross_context_coupling(repository_id)
    return {}


@mcp.tool(
    name="suggest_context_splits",
    description="Suggest how to split large bounded contexts based on cohesion analysis",
)
async def suggest_context_splits(
    min_entities: int = Field(
        default=20,
        description="Minimum entities for a context to be considered",
    ),
    max_cohesion_threshold: float = Field(
        default=0.4,
        description="Maximum cohesion score to suggest split",
    ),
) -> list[dict[str, Any]]:
    """Suggest context splits."""
    await initialize_server()

    async for session in get_db_session():
        from src.domain.pattern_analyzer import DomainPatternAnalyzer

        analyzer = DomainPatternAnalyzer(session)
        return await analyzer.suggest_context_splits(
            min_entities, max_cohesion_threshold
        )
    return []


@mcp.tool(
    name="detect_anti_patterns",
    description="Detect DDD anti-patterns like anemic models, god objects, and circular dependencies",
)
async def detect_anti_patterns(
    repository_id: int | None = Field(
        default=None,
        description="Optional repository ID to filter analysis",
    ),
) -> dict[str, list[dict[str, Any]]]:
    """Detect anti-patterns."""
    await initialize_server()

    async for session in get_db_session():
        from src.domain.pattern_analyzer import DomainPatternAnalyzer

        analyzer = DomainPatternAnalyzer(session)
        return await analyzer.detect_anti_patterns(repository_id)
    return {}


@mcp.tool(
    name="analyze_domain_evolution",
    description="Analyze how the domain model has evolved over time",
)
async def analyze_domain_evolution(
    repository_id: int = Field(description="Repository ID to analyze"),
    days: int = Field(default=30, description="Number of days to look back"),
) -> dict[str, Any]:
    """Analyze domain evolution."""
    await initialize_server()

    async for session in get_db_session():
        from src.domain.pattern_analyzer import DomainPatternAnalyzer

        analyzer = DomainPatternAnalyzer(session)
        return await analyzer.analyze_evolution(repository_id, days)
    return {}


@mcp.tool(
    name="get_domain_metrics",
    description="Get comprehensive domain health metrics and insights",
)
async def get_domain_metrics(
    repository_id: int | None = Field(
        default=None,
        description="Optional repository ID to filter analysis",
    ),
) -> dict[str, Any]:
    """Get comprehensive domain metrics."""
    await initialize_server()

    async for session in get_db_session():
        from src.domain.pattern_analyzer import DomainPatternAnalyzer

        analyzer = DomainPatternAnalyzer(session)

        # Combine multiple analyses
        coupling = await analyzer.analyze_cross_context_coupling(repository_id)
        anti_patterns = await analyzer.detect_anti_patterns(repository_id)

        # Count issues by severity
        severity_counts = {"high": 0, "medium": 0, "low": 0}

        for issues in anti_patterns.values():
            if isinstance(issues, list):
                for issue in issues:
                    severity = issue.get("severity", "medium")
                    severity_counts[severity] += 1

        return {
            "coupling_metrics": coupling["metrics"],
            "anti_pattern_summary": severity_counts,
            "health_score": 100
            - (severity_counts["high"] * 10 + severity_counts["medium"] * 5),
            "recommendations": coupling.get("recommendations", []),
        }
    return {}


# Tools from main.py for compatibility
@mcp.tool(name="search_code", description="Search for code by natural language query")
async def search_code(
    query: str = Field(description="Natural language search query"),
    limit: int = Field(default=10, description="Maximum results to return"),
) -> list[dict[str, Any]]:
    """Search for code by natural language query."""
    await initialize_server()

    async for session in get_db_session():
        search_tools = CodeSearchTools(session, mcp)
        from src.embeddings.vector_search import SearchScope

        try:
            return await search_tools.vector_search.search(
                query=query,
                scope=SearchScope.ALL,
                repository_id=None,
                file_id=None,
                limit=limit,
                threshold=None,
            )
        except Exception as e:
            logger.exception("Error in search_code")
            return [{"error": str(e)}]
    return [{"error": "No session"}]


@mcp.tool(name="explain_code", description="Explain what a code element does")
async def explain_code(
    path: str = Field(
        description="Path to code element (e.g., 'src.utils.helpers.parse_json')"
    ),
) -> str:
    """
    Explain what a code element does (function, class, module, or package).
    Returns a hierarchical explanation of the code element.
    """
    await initialize_server()

    try:
        from src.mcp_server.tools.explain import ExplainTool

        tool = ExplainTool()
        return await tool.explain_code(path)
    except Exception as e:
        logger.exception("Error in explain_code")
        return f"Error explaining code: {e!s}"


@mcp.tool(name="find_definition", description="Find where a symbol is defined")
async def find_definition(
    name: str = Field(description="Name of the symbol to find"),
    file_path: str | None = Field(
        default=None, description="Optional file path to search in"
    ),
    entity_type: str | None = Field(
        default=None, description="Optional entity type filter"
    ),
) -> list[dict[str, Any]]:
    """Find where a symbol is defined."""
    await initialize_server()

    try:
        from src.query.symbol_finder import SymbolFinder

        async for session in get_db_session():
            finder = SymbolFinder(session)
            return await finder.find_definitions(
                name=name,
                file_path=file_path,
                entity_type=entity_type,
            )
    except Exception as e:
        logger.exception("Error in find_definition")
        return [{"error": str(e), "name": name}]
    return []


@mcp.tool(
    name="find_usage", description="Find all places where a function/class is used"
)
async def find_usage(
    function_or_class: str = Field(description="Name of the function or class"),
    repository: str | None = Field(
        default=None, description="Optional repository name filter"
    ),
) -> list[dict[str, Any]]:
    """Find all places where a function/class is used."""
    await initialize_server()

    try:
        from src.mcp_server.tools.find import FindTool

        async for session in get_db_session():
            tool = FindTool(session)
            return await tool.find_usage(function_or_class, repository)
    except Exception as e:
        logger.exception("Error in find_usage")
        return [{"error": str(e)}]
    return []


@mcp.tool(name="get_code_structure", description="Get the structure of a code file")
async def get_code_structure(
    file_path: str = Field(description="Path to the file relative to repository root"),
) -> dict[str, Any]:
    """Get the structure of a code file."""
    await initialize_server()

    try:
        from sqlalchemy import select

        from src.database.models import File

        async for session in get_db_session():
            # Get file from database
            result = await session.execute(select(File).where(File.path == file_path))
            file = result.scalar_one_or_none()

            if not file:
                return {"error": f"File not found: {file_path}"}

            # Get code structure using CodeProcessor
            from src.scanner.code_processor import CodeProcessor

            processor = CodeProcessor(session)
            return await processor.get_file_structure(file)

    except Exception as e:
        logger.exception("Error in get_code_structure")
        return {"error": str(e)}
    return {}


@mcp.tool(name="suggest_refactoring", description="Suggest refactoring opportunities")
async def suggest_refactoring(
    file_path: str = Field(description="Path to the file to analyze"),
    focus_area: str | None = Field(
        default=None, description="Optional focus area for suggestions"
    ),
) -> list[dict[str, Any]]:
    """Suggest refactoring opportunities."""
    await initialize_server()

    try:
        from src.mcp_server.tools.analyze import AnalyzeTool

        async for session in get_db_session():
            tool = AnalyzeTool(session)
            suggestions = await tool.suggest_refactoring(file_path)

            # If focus_area is provided, filter suggestions
            if focus_area and suggestions:
                return [
                    {
                        "focus_area": focus_area,
                        "suggestions": suggestions,
                        "note": f"Focused on {focus_area} improvements",
                    }
                ]

            return suggestions
    except Exception as e:
        logger.exception("Error in suggest_refactoring")
        return [{"error": str(e)}]
    return []


@mcp.tool(
    name="sync_repository",
    description="Manually trigger sync for a specific repository",
)
async def sync_repository(
    repository_url: str = Field(description="Repository URL to sync"),
) -> dict[str, Any]:
    """Manually trigger sync for a specific repository."""
    await initialize_server()

    try:
        from src.mcp_server.tools.repository import RepositoryTool

        async for session in get_db_session():
            tool = RepositoryTool(session)
            return await tool.sync_repository(repository_url)
    except Exception as e:
        logger.exception("Error in sync_repository")
        return {"error": str(e)}
    return {}


@mcp.custom_route("/health", methods=["GET"])
async def health_check(_request: Request) -> JSONResponse:
    """Basic health check that the server is running."""
    return JSONResponse({"status": "alive"}, status_code=200)


# Aliases for compatibility
server = mcp
app = mcp


class MockServer:
    """Mock server for compatibility."""

    async def initialize(self) -> None:
        await initialize_server()

    async def shutdown(self) -> None:
        if _engine:
            await _engine.dispose()

    async def scan_repository(
        self, url: str, branch: str | None = None, generate_embeddings: bool = True
    ) -> dict[str, Any] | None:
        await initialize_server()
        async for session in get_db_session():
            # Create repository entry
            from sqlalchemy import select

            from src.database.models import Repository
            from src.embeddings.embedding_service import EmbeddingService
            from src.models import RepositoryConfig
            from src.scanner.repository_scanner import RepositoryScanner

            # Check if exists
            existing = await session.execute(
                select(Repository).where(Repository.github_url == url)
            )
            repo = existing.scalar_one_or_none()
            if not repo:
                repo = Repository(
                    github_url=url,
                    owner=url.split("/")[-2] if "/" in url else "local",
                    name=url.split("/")[-1].replace(".git", "") if "/" in url else url,
                    default_branch=branch or "main",
                )
                session.add(repo)
                await session.commit()
                await session.refresh(repo)

            # Scan repository
            repo_config = RepositoryConfig(url=url, branch=branch)
            scanner = RepositoryScanner(session)
            scan_result = await scanner.scan_repository(repo_config)

            # Optionally generate embeddings
            if generate_embeddings:
                embedding_service = EmbeddingService(session)
                embedding_result = await embedding_service.create_repository_embeddings(
                    int(repo.id)
                )
                scan_result["embeddings"] = embedding_result

            return {
                "success": True,
                "repository": {
                    "id": repo.id,
                    "url": url,
                    "branch": branch or "default",
                },
                "scan_result": scan_result,
            }
        return None

    async def search(
        self, query: str, repository_id: int | None = None, limit: int = 10
    ) -> dict[str, Any] | None:
        await initialize_server()
        async for session in get_db_session():
            search_tools = CodeSearchTools(session, mcp)
            from src.embeddings.vector_search import SearchScope

            try:
                results = await search_tools.vector_search.search(
                    query=query,
                    scope=SearchScope.ALL,
                    repository_id=repository_id,
                    file_id=None,
                    limit=limit,
                    threshold=None,
                )
                return {
                    "success": True,
                    "query": query,
                    "results": results,
                    "count": len(results),
                }
            except Exception as e:
                logger.exception("Search failed")
                return {"success": False, "error": str(e), "results": []}
        return None


def create_server() -> MockServer:
    """Create server instance for compatibility."""
    return MockServer()


if __name__ == "__main__":
    # Run using FastMCP's built-in runner with HTTP transport
    mcp.run(transport="http")
