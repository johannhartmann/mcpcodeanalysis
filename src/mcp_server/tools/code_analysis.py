"""Code analysis tools for MCP server."""

from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from src.database.models import Class, File, Function, Import, Module
from src.logger import get_logger
from src.parser.code_extractor import CodeExtractor

logger = get_logger(__name__)


class GetCodeRequest(BaseModel):
    """Get code content request."""

    entity_type: str = Field(..., description="Entity type: function, class, module")
    entity_id: int = Field(..., description="Entity ID")
    include_context: bool = Field(
        default=True,
        description="Include surrounding context",
    )
    context_lines: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Number of context lines",
    )


class AnalyzeFileRequest(BaseModel):
    """Analyze file structure request."""

    file_id: int = Field(..., description="File ID to analyze")
    include_imports: bool = Field(default=True, description="Include import analysis")
    include_metrics: bool = Field(default=True, description="Include code metrics")


class GetDependenciesRequest(BaseModel):
    """Get dependencies request."""

    entity_type: str = Field(
        ...,
        description="Entity type: file, module, class, function",
    )
    entity_id: int = Field(..., description="Entity ID")
    depth: int = Field(default=1, ge=1, le=5, description="Dependency depth to explore")


class GetCallersRequest(BaseModel):
    """Get function/method callers request."""

    function_id: int = Field(..., description="Function ID")
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of callers",
    )


class CodeAnalysisTools:
    """Code analysis tools for MCP."""

    def __init__(self, db_session: AsyncSession, mcp: FastMCP) -> None:
        """Initialize code analysis tools.

        Args:
            db_session: Database session
            mcp: FastMCP instance
        """
        self.db_session = db_session
        self.mcp = mcp
        self.code_extractor = CodeExtractor()

    async def register_tools(self) -> None:
        """Register all code analysis tools."""

        @self.mcp.tool(name="get_code", description="Get code content for an entity")
        async def get_code(request: GetCodeRequest) -> dict[str, Any]:
            """Get code content for a specific entity.

            Args:
                request: Get code request parameters

            Returns:
                Code content and metadata
            """
            try:
                entity = None
                file_path = None

                # Get entity based on type
                if request.entity_type == "function":
                    result = await self.db_session.execute(
                        select(Function)
                        .where(Function.id == request.entity_id)
                        .options(selectinload(Function.file)),
                    )
                    entity = result.scalar_one_or_none()
                    if entity:
                        file_path = Path(entity.file.path)

                elif request.entity_type == "class":
                    result = await self.db_session.execute(
                        select(Class)
                        .where(Class.id == request.entity_id)
                        .options(selectinload(Class.file)),
                    )
                    entity = result.scalar_one_or_none()
                    if entity:
                        file_path = Path(entity.file.path)

                elif request.entity_type == "module":
                    result = await self.db_session.execute(
                        select(Module)
                        .where(Module.id == request.entity_id)
                        .options(selectinload(Module.file)),
                    )
                    entity = result.scalar_one_or_none()
                    if entity:
                        file_path = Path(entity.file.path)

                if not entity:
                    return {
                        "success": False,
                        "error": f"{request.entity_type} {request.entity_id} not found",
                    }

                # Get code content
                raw_content, contextual_content = (
                    self.code_extractor.get_entity_content(
                        file_path,
                        request.entity_type,
                        entity.start_line,
                        entity.end_line,
                        include_context=request.include_context,
                    )
                )

                return {
                    "success": True,
                    "entity_type": request.entity_type,
                    "entity_id": request.entity_id,
                    "name": entity.name,
                    "file_path": str(file_path),
                    "start_line": entity.start_line,
                    "end_line": entity.end_line,
                    "code": (
                        contextual_content if request.include_context else raw_content
                    ),
                    "raw_code": raw_content,
                }

            except Exception as e:
                logger.exception("Failed to get code: %s")
                return {
                    "success": False,
                    "error": str(e),
                }

        @self.mcp.tool(
            name="analyze_file",
            description="Analyze file structure and metrics",
        )
        async def analyze_file(request: AnalyzeFileRequest) -> dict[str, Any]:
            """Analyze a file's structure and metrics.

            Args:
                request: Analyze file request parameters

            Returns:
                File analysis results
            """
            try:
                # Get file
                result = await self.db_session.execute(
                    select(File)
                    .where(File.id == request.file_id)
                    .options(selectinload(File.repository)),
                )
                file = result.scalar_one_or_none()

                if not file:
                    return {
                        "success": False,
                        "error": f"File {request.file_id} not found",
                    }

                analysis = {
                    "success": True,
                    "file": {
                        "id": file.id,
                        "path": file.path,
                        "language": file.language,
                        "size": file.size,
                        "repository": file.repository.name if file.repository else None,
                    },
                }

                # Get modules
                modules_result = await self.db_session.execute(
                    select(Module).where(Module.file_id == request.file_id),
                )
                modules = modules_result.scalars().all()

                # Get classes
                classes_result = await self.db_session.execute(
                    select(Class).where(Class.file_id == request.file_id),
                )
                classes = classes_result.scalars().all()

                # Get functions
                functions_result = await self.db_session.execute(
                    select(Function).where(Function.file_id == request.file_id),
                )
                functions = functions_result.scalars().all()

                analysis["structure"] = {
                    "modules": [
                        {
                            "id": m.id,
                            "name": m.name,
                            "lines": f"{m.start_line}-{m.end_line}",
                        }
                        for m in modules
                    ],
                    "classes": [
                        {
                            "id": c.id,
                            "name": c.name,
                            "base_classes": c.base_classes,
                            "is_abstract": c.is_abstract,
                            "lines": f"{c.start_line}-{c.end_line}",
                        }
                        for c in classes
                    ],
                    "functions": [
                        {
                            "id": f.id,
                            "name": f.name,
                            "class_id": f.class_id,
                            "is_async": f.is_async,
                            "lines": f"{f.start_line}-{f.end_line}",
                        }
                        for f in functions
                    ],
                }

                # Get imports if requested
                if request.include_imports:
                    imports_result = await self.db_session.execute(
                        select(Import).where(Import.file_id == request.file_id),
                    )
                    imports = imports_result.scalars().all()

                    analysis["imports"] = [
                        {
                            "statement": i.import_statement,
                            "from": i.imported_from,
                            "names": i.imported_names,
                            "line": i.line_number,
                        }
                        for i in imports
                    ]

                # Calculate metrics if requested
                if request.include_metrics:
                    # Calculate lines of code
                    total_lines = max(
                        [m.end_line for m in modules]
                        + [c.end_line for c in classes]
                        + [f.end_line for f in functions]
                        + [0],
                    )

                    # Count methods vs functions
                    methods = sum(1 for f in functions if f.class_id is not None)
                    standalone_functions = len(functions) - methods

                    # Calculate complexity
                    total_complexity = sum(f.complexity or 1 for f in functions)
                    avg_complexity = (
                        total_complexity / len(functions) if functions else 0
                    )

                    analysis["metrics"] = {
                        "total_lines": total_lines,
                        "modules": len(modules),
                        "classes": len(classes),
                        "functions": standalone_functions,
                        "methods": methods,
                        "total_complexity": total_complexity,
                        "average_complexity": round(avg_complexity, 2),
                    }

                return analysis

            except Exception as e:
                logger.exception("Failed to analyze file: %s")
                return {
                    "success": False,
                    "error": str(e),
                }

        @self.mcp.tool(
            name="get_dependencies",
            description="Get dependencies for a code entity",
        )
        async def get_dependencies(request: GetDependenciesRequest) -> dict[str, Any]:
            """Get dependencies for a code entity.

            Args:
                request: Get dependencies request parameters

            Returns:
                Dependency information
            """
            try:
                dependencies = {
                    "success": True,
                    "entity_type": request.entity_type,
                    "entity_id": request.entity_id,
                    "imports": [],
                    "references": [],
                }

                # Get imports based on entity type
                if request.entity_type == "file":
                    imports_result = await self.db_session.execute(
                        select(Import).where(Import.file_id == request.entity_id),
                    )
                    imports = imports_result.scalars().all()

                elif request.entity_type == "module":
                    # Get file ID for module
                    module_result = await self.db_session.execute(
                        select(Module).where(Module.id == request.entity_id),
                    )
                    module = module_result.scalar_one_or_none()
                    if module:
                        imports_result = await self.db_session.execute(
                            select(Import).where(Import.file_id == module.file_id),
                        )
                        imports = imports_result.scalars().all()
                    else:
                        imports = []
                else:
                    # For classes and functions, get imports from their file
                    if request.entity_type == "class":
                        entity_result = await self.db_session.execute(
                            select(Class).where(Class.id == request.entity_id),
                        )
                    else:  # function
                        entity_result = await self.db_session.execute(
                            select(Function).where(Function.id == request.entity_id),
                        )

                    entity = entity_result.scalar_one_or_none()
                    if entity:
                        imports_result = await self.db_session.execute(
                            select(Import).where(Import.file_id == entity.file_id),
                        )
                        imports = imports_result.scalars().all()
                    else:
                        imports = []

                # Format imports
                dependencies["imports"] = [
                    {
                        "id": i.id,
                        "statement": i.import_statement,
                        "from": i.imported_from,
                        "names": i.imported_names,
                        "is_relative": i.is_relative,
                    }
                    for i in imports
                ]

                # TODO(@dev): Add reference analysis (which entities use this one)

                return dependencies

            except Exception as e:
                logger.exception("Failed to get dependencies: %s")
                return {
                    "success": False,
                    "error": str(e),
                }

        @self.mcp.tool(
            name="find_usages",
            description="Find where a function or class is used",
        )
        async def find_usages(
            entity_type: str = Field(..., description="Entity type: function or class"),
            entity_name: str = Field(..., description="Entity name to find"),
            repository_id: int | None = Field(
                None,
                description="Repository ID to limit search",
            ),
            limit: int = Field(default=50, ge=1, le=200),
        ) -> dict[str, Any]:
            """Find usages of a function or class.

            Args:
                entity_type: Type of entity to find
                entity_name: Name of entity
                repository_id: Optional repository filter
                limit: Maximum results

            Returns:
                Usage locations
            """
            try:
                # This is a simplified implementation
                # In a real system, we'd parse code to find actual usage

                # Search for the entity name in file contents
                # For now, return files that might contain references

                from sqlalchemy import text

                query = text(
                    """
                    SELECT DISTINCT f.id, f.path, f.repository_id
                    FROM files f
                    WHERE f.is_deleted = false
                    AND f.content IS NOT NULL
                    AND f.content LIKE :pattern
                """,
                )

                if repository_id:
                    query = text(query.text + " AND f.repository_id = :repo_id")

                query = text(query.text + " LIMIT :limit")

                params = {
                    "pattern": f"%{entity_name}%",
                    "limit": limit,
                }
                if repository_id:
                    params["repo_id"] = repository_id

                result = await self.db_session.execute(query, params)
                files = result.fetchall()

                usages = []
                for file_id, file_path, repo_id in files:
                    usages.append(
                        {
                            "file_id": file_id,
                            "file_path": file_path,
                            "repository_id": repo_id,
                        },
                    )

                return {
                    "success": True,
                    "entity_type": entity_type,
                    "entity_name": entity_name,
                    "usages": usages,
                    "count": len(usages),
                    "note": "This is a text-based search. Actual usage analysis requires parsing.",
                }

            except Exception as e:
                logger.exception("Failed to find usages: %s")
                return {
                    "success": False,
                    "error": str(e),
                    "usages": [],
                }

        logger.info("Code analysis tools registered")
