"""Find tool for MCP server."""

from typing import Any

from src.database import get_repositories, get_session
from src.database.models import File, Function, Module
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FindTool:
    """MCP tools for finding code definitions and usages."""

    async def find_definition(
        self,
        name: str,
        type: str = "any",
    ) -> list[dict[str, Any]]:
        """
        Find where a function/class/module is defined.

        Args:
            name: Name of the entity to find
            type: Type to search for - 'function', 'class', 'module', or 'any'

        Returns:
            List of locations where the entity is defined
        """
        try:
            async with get_session() as session:
                async with get_repositories(session) as repos:
                    # Search for entities by name
                    entity_type = None if type == "any" else type
                    results = await repos["code_entity"].find_by_name(name, entity_type)

                    # Format results
                    definitions = []
                    for result in results:
                        entity = result["entity"]
                        module = result["module"]

                        # Get file and repository info
                        file = (
                            await repos["file"].get_by_id(module.file_id)
                            if module
                            else None
                        )
                        repo = (
                            await repos["repository"].get_by_id(file.repository_id)
                            if file
                            else None
                        )

                        definition = {
                            "name": entity.name,
                            "type": result["type"],
                            "location": {
                                "repository": repo.name if repo else None,
                                "repository_url": repo.github_url if repo else None,
                                "file": file.path if file else None,
                                "line": (
                                    entity.start_line
                                    if hasattr(entity, "start_line")
                                    else None
                                ),
                            },
                            "docstring": (
                                entity.docstring
                                if hasattr(entity, "docstring")
                                else None
                            ),
                        }

                        # Add type-specific info
                        if result["type"] == "function":
                            definition["signature"] = self._build_function_signature(
                                entity,
                            )
                            definition["is_method"] = bool(entity.class_id)
                            if result.get("class"):
                                definition["class_name"] = result["class"].name
                        elif result["type"] == "class":
                            definition["base_classes"] = entity.base_classes
                            definition["is_abstract"] = entity.is_abstract

                        definitions.append(definition)

                    return definitions

        except Exception as e:
            logger.exception("Error in find_definition: %s")
            return [{"error": str(e), "name": name, "type": type}]

    async def find_usage(
        self,
        function_or_class: str,
        repository: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Find all places where a function/class is used.

        Args:
            function_or_class: Name of the function or class to find usages for
            repository: Optional repository name to filter by

        Returns:
            List of locations where the entity is used
        """
        try:
            async with get_session() as session:
                async with get_repositories(session) as repos:
                    # Find the entity definition first
                    definitions = await repos["code_entity"].find_by_name(
                        function_or_class,
                    )

                    if not definitions:
                        return [
                            {
                                "error": f"Entity '{function_or_class}' not found",
                                "suggestions": [],
                            },
                        ]

                    usages = []

                    # For each definition, find usages
                    for definition in definitions:
                        definition["entity"]

                        # Search in imports
                        import_usages = await self._find_import_usages(
                            repos,
                            function_or_class,
                            repository,
                        )
                        usages.extend(import_usages)

                        # Search in code (this would require more sophisticated AST analysis)
                        # For now, we do a simple text search in embeddings
                        code_usages = await self._find_code_usages(
                            repos,
                            function_or_class,
                            repository,
                        )
                        usages.extend(code_usages)

                    # Deduplicate and sort
                    seen = set()
                    unique_usages = []
                    for usage in usages:
                        key = (usage["location"]["file"], usage["location"]["line"])
                        if key not in seen:
                            seen.add(key)
                            unique_usages.append(usage)

                    return unique_usages

        except Exception as e:
            logger.exception("Error in find_usage: %s")
            return [
                {
                    "error": str(e),
                    "function_or_class": function_or_class,
                    "repository": repository,
                },
            ]

    async def _find_import_usages(
        self,
        repos: dict[str, Any],
        name: str,
        repository: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find usages in import statements."""
        session = repos["repository"].session
        usages = []

        # Query imports that contain the name
        query = (
            "SELECT i.*, f.path, f.repository_id FROM imports i "
            "JOIN files f ON i.file_id = f.id "
            "WHERE i.import_statement LIKE :pattern "
            "OR :name = ANY(i.imported_names)"
        )

        params = {"pattern": f"%{name}%", "name": name}

        results = await session.execute(query, params)

        for import_record in results:
            # Get repository info
            repo = await repos["repository"].get_by_id(import_record.repository_id)

            if repository and repo.name != repository:
                continue

            usages.append(
                {
                    "type": "import",
                    "statement": import_record.import_statement,
                    "location": {
                        "repository": repo.name if repo else None,
                        "repository_url": repo.github_url if repo else None,
                        "file": import_record.path,
                        "line": import_record.line_number,
                    },
                    "context": f"Imported from {import_record.imported_from or 'module'}",
                },
            )

        return usages

    async def _find_code_usages(
        self,
        repos: dict[str, Any],
        name: str,
        repository: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find usages in code (simplified version)."""
        # This is a simplified implementation
        # A full implementation would use AST analysis to find actual usages

        session = repos["repository"].session
        usages = []

        # Search in code embeddings content
        query = (
            "SELECT DISTINCT e.entity_type, e.entity_id, e.metadata "
            "FROM code_embeddings e "
            "WHERE e.content LIKE :pattern "
            "AND e.embedding_type = 'raw' "
            "LIMIT 50"
        )

        params = {"pattern": f"%{name}%"}

        results = await session.execute(query, params)

        for embedding in results:
            # This is a simplified check - would need proper AST analysis
            metadata = embedding.metadata or {}

            # Skip if it's the definition itself
            if metadata.get("entity_name") == name:
                continue

            # Try to get location info
            if embedding.entity_type == "function":
                func = await session.get(Function, embedding.entity_id)
                if func:
                    module = await session.get(Module, func.module_id)
                    file = await session.get(File, module.file_id) if module else None
                    repo = (
                        await repos["repository"].get_by_id(file.repository_id)
                        if file
                        else None
                    )

                    if repository and repo and repo.name != repository:
                        continue

                    usages.append(
                        {
                            "type": "code_reference",
                            "in_entity": func.name,
                            "location": {
                                "repository": repo.name if repo else None,
                                "repository_url": repo.github_url if repo else None,
                                "file": file.path if file else None,
                                "line": func.start_line,
                            },
                            "context": f"Used in {embedding.entity_type} '{func.name}'",
                        },
                    )

        return usages

    def _build_function_signature(self, func: Any) -> str:
        """Build function signature string."""
        params = []

        for param in func.parameters or []:
            param_str = param["name"]
            if param.get("type"):
                param_str += f": {param['type']}"
            params.append(param_str)

        signature = f"{func.name}({', '.join(params)})"

        if func.return_type:
            signature += f" -> {func.return_type}"

        return signature
