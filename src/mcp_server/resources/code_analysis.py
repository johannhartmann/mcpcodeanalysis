"""Code analysis resources for read-only access to code search and analysis data."""

from urllib.parse import unquote

from fastmcp import FastMCP
from sqlalchemy import func, select

from src.database.models import Class, File, Function, Module
from src.query.symbol_finder import SymbolFinder


class CodeAnalysisResources:
    """Resources for code analysis data access."""

    def __init__(self, mcp: FastMCP, session_maker):
        """Initialize code analysis resources."""
        self.mcp = mcp
        self.session_maker = session_maker

    def register_resources(self):
        """Register all code analysis resources."""

        @self.mcp.resource("code://search")
        async def search_code() -> str:
            """Search for code using natural language queries."""
            return """# Code Search

To search for code, use the search_code tool instead, which accepts parameters:
- query: Natural language search query
- repository_id: Optional repository filter
- limit: Maximum results to return

This resource provides information about code search capabilities."""

        @self.mcp.resource("code://explain/{entity_type}/{entity_id}")
        async def explain_code(entity_type: str, entity_id: int) -> str:
            """Get explanation of code entities."""
            async with self.session_maker() as session:
                try:
                    # Get entity details based on type
                    if entity_type == "function":
                        entity = await session.get(Function, entity_id)
                        if not entity:
                            return f"Function with ID {entity_id} not found"
                    elif entity_type == "class":
                        entity = await session.get(Class, entity_id)
                        if not entity:
                            return f"Class with ID {entity_id} not found"
                    elif entity_type == "module":
                        entity = await session.get(Module, entity_id)
                        if not entity:
                            return f"Module with ID {entity_id} not found"
                    else:
                        return f"Unknown entity type: {entity_type}"

                    # Get file information
                    file = await session.get(File, entity.file_id)

                    output = "# Code Explanation\n\n"
                    output += f"## {entity.name}\n"
                    output += f"**Type**: {entity_type}\n"
                    output += f"**File**: `{file.file_path if file else 'Unknown'}`\n"

                    if hasattr(entity, "docstring") and entity.docstring:
                        output += f"\n### Documentation\n{entity.docstring}\n"

                    if hasattr(entity, "signature") and entity.signature:
                        output += f"\n### Signature\n`{entity.signature}`\n"

                    if hasattr(entity, "complexity") and entity.complexity:
                        output += f"\n### Complexity\nCyclomatic Complexity: {entity.complexity}\n"

                    return output

                except (AttributeError, KeyError, ValueError) as e:
                    return f"Error explaining code: {e!s}"

        @self.mcp.resource("code://definitions/{name}")
        async def find_definitions(name: str) -> str:
            """Find where symbols are defined."""
            # Decode URL-encoded name
            name = unquote(name)

            async with self.session_maker() as session:
                try:
                    finder = SymbolFinder(session)
                    results = await finder.find_definitions(name=name)

                    if not results:
                        return f"No definitions found for: {name}"

                    output = f"# Symbol Definitions\n\n**Symbol**: `{name}`\n"
                    output += f"**Found**: {len(results)} definitions\n\n"

                    for result in results:
                        output += f"## {result['type'].title()}: {result['name']}\n"
                        output += f"**File**: `{result['file_path']}:{result.get('line_number', 0)}`\n"

                        if result.get("signature"):
                            output += f"**Signature**: `{result['signature']}`\n"

                        if result.get("docstring"):
                            output += f"\n{result['docstring']}\n"

                        output += "\n---\n\n"

                    return output

                except (AttributeError, KeyError, ValueError) as e:
                    return f"Error finding definitions: {e!s}"

        @self.mcp.resource("code://usages/{entity_type}/{entity_id}")
        async def find_usages(entity_type: str, entity_id: int) -> str:
            """Find all usages of a code entity."""
            async with self.session_maker() as session:
                try:
                    finder = SymbolFinder(session)
                    results = await finder.find_usages(
                        entity_type=entity_type,
                        entity_id=entity_id,
                        include_tests=True,  # Default to including tests
                    )

                    if not results:
                        return f"No usages found for {entity_type} with ID {entity_id}"

                    output = "# Usage Analysis\n\n"
                    output += f"**Entity Type**: {entity_type}\n"
                    output += f"**Total Usages**: {len(results)}\n\n"

                    # Group usages by file
                    usages_by_file = {}
                    for usage in results:
                        file_path = usage.get("file_path", "Unknown")
                        if file_path not in usages_by_file:
                            usages_by_file[file_path] = []
                        usages_by_file[file_path].append(usage)

                    for file_path, file_usages in usages_by_file.items():
                        output += f"## {file_path} ({len(file_usages)} usages)\n"

                        for usage in file_usages:
                            output += f"- Line {usage.get('line_number', '?')}: "
                            output += f"{usage.get('usage_type', 'reference')}\n"

                        output += "\n"

                    return output

                except (AttributeError, KeyError, ValueError) as e:
                    return f"Error finding usages: {e!s}"

        @self.mcp.resource("code://structure/{file_path}")
        async def get_code_structure(file_path: str) -> str:
            """Get the structure of a code file."""
            # Decode URL-encoded file path
            file_path = unquote(file_path)

            async with self.session_maker() as session:
                try:
                    # Get file
                    result = await session.execute(
                        select(File).where(File.file_path == file_path)
                    )
                    file = result.scalar_one_or_none()

                    if not file:
                        return f"File not found: {file_path}"

                    output = f"# Code Structure\n\n**File**: `{file_path}`\n\n"

                    # File metrics
                    output += "## File Metrics\n"
                    # Estimate lines from file size (assuming average 50 bytes per line)
                    estimated_lines = (file.size or 0) // 50
                    output += f"- **Lines of Code**: {estimated_lines} (estimated)\n"
                    output += f"- **Language**: {file.language or 'Unknown'}\n\n"

                    # Get classes
                    classes = await session.execute(
                        select(Class).where(Class.file_id == file.id)
                    )
                    class_list = classes.scalars().all()

                    if class_list:
                        output += "## Classes\n"
                        for cls in class_list:
                            output += f"### {cls.name}\n"
                            if cls.docstring:
                                output += f"{cls.docstring}\n"

                            # Get methods
                            methods = await session.execute(
                                select(Function).where(
                                    Function.file_id == file.id,
                                    Function.parent_class_id == cls.id,
                                )
                            )
                            method_list = methods.scalars().all()

                            if method_list:
                                output += "**Methods**:\n"
                                for method in method_list:
                                    output += f"- `{method.name}`\n"
                            output += "\n"

                    # Get standalone functions
                    functions = await session.execute(
                        select(Function).where(
                            Function.file_id == file.id,
                            Function.parent_class_id.is_(None),
                        )
                    )
                    function_list = functions.scalars().all()

                    if function_list:
                        output += "## Functions\n"
                        for func in function_list:
                            output += f"- `{func.name}`"
                            if func.docstring:
                                docstring_first_line = func.docstring.split("\n")[0]
                                output += f" - {docstring_first_line}"
                            output += "\n"

                    return output

                except (AttributeError, KeyError, ValueError) as e:
                    return f"Error getting code structure: {e!s}"

        @self.mcp.resource("code://refactoring/{entity_type}/{entity_id}")
        async def suggest_refactoring(entity_type: str, entity_id: int) -> str:
            """Get refactoring suggestions for code entities."""
            async with self.session_maker() as session:
                try:
                    # Get entity
                    if entity_type == "function":
                        entity = await session.get(Function, entity_id)
                        entity_name = entity.name if entity else "Unknown"
                    elif entity_type == "class":
                        entity = await session.get(Class, entity_id)
                        entity_name = entity.name if entity else "Unknown"
                    else:
                        return f"Unsupported entity type for refactoring: {entity_type}"

                    if not entity:
                        return f"{entity_type.title()} with ID {entity_id} not found"

                    output = "# Refactoring Analysis\n\n"
                    output += f"## {entity_name}\n"
                    output += f"**Type**: {entity_type}\n"

                    # Basic refactoring suggestions based on metrics
                    suggestions = []

                    if hasattr(entity, "complexity") and entity.complexity:
                        output += f"**Complexity**: {entity.complexity}\n"
                        if entity.complexity > 10:
                            suggestions.append(
                                {
                                    "title": "Reduce Complexity",
                                    "description": "This code has high cyclomatic complexity. Consider breaking it down into smaller functions.",
                                    "priority": (
                                        "high" if entity.complexity > 20 else "medium"
                                    ),
                                }
                            )

                    if (
                        hasattr(entity, "lines_of_code")
                        and entity.lines_of_code
                        and entity.lines_of_code > 50
                    ):
                        suggestions.append(
                            {
                                "title": "Split Large Function",
                                "description": f"This {entity_type} is {entity.lines_of_code} lines long. Consider splitting it into smaller, more focused units.",
                                "priority": "medium",
                            }
                        )

                    if entity_type == "class":
                        # Count methods
                        method_count = await session.scalar(
                            select(func.count())
                            .select_from(Function)
                            .where(Function.parent_class_id == entity.id)
                        )
                        if method_count and method_count > 20:
                            suggestions.append(
                                {
                                    "title": "Consider Single Responsibility Principle",
                                    "description": f"This class has {method_count} methods. It might be doing too much. Consider splitting responsibilities.",
                                    "priority": "high",
                                }
                            )

                    output += "\n## Refactoring Suggestions\n"

                    if suggestions:
                        for i, suggestion in enumerate(suggestions, 1):
                            priority_icon = {
                                "high": "🔴",
                                "medium": "🟡",
                                "low": "🟢",
                            }.get(suggestion.get("priority", "medium"), "⚪")

                            output += (
                                f"\n### {i}. {suggestion['title']} {priority_icon}\n"
                            )
                            output += f"{suggestion['description']}\n"
                    else:
                        output += "No specific refactoring suggestions. The code appears to be well-structured.\n"

                    return output

                except (AttributeError, KeyError, ValueError) as e:
                    return f"Error analyzing code for refactoring: {e!s}"

        @self.mcp.resource("code://similar")
        async def find_similar_code() -> str:
            """Find code similar to the provided snippet."""
            return """# Similar Code Search

To find similar code, use the find_similar_code tool instead, which accepts parameters:
- reference_path: Path to reference file
- reference_name: Name of reference entity
- entity_type: Type of entity (function, class, etc.)
- limit: Maximum results to return

This resource provides information about code similarity search capabilities."""
