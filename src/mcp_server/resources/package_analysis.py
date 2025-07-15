"""Package analysis resources for read-only access to package structure data."""

from fastmcp import FastMCP

from src.database.package_repository import PackageRepository


class PackageAnalysisResources:
    """Resources for package analysis data access."""

    def __init__(self, mcp: FastMCP, session_maker):
        """Initialize package analysis resources."""
        self.mcp = mcp
        self.session_maker = session_maker

    def register_resources(self):
        """Register all package analysis resources."""

        @self.mcp.resource(
            "packages://{repository_url}/tree",
            description="""Get the hierarchical package/module structure of a repository.

            Parameters:
            - repository_url: GitHub repository URL (e.g., 'github.com/owner/repo')
                             Note: Do not include https:// prefix

            Returns: Markdown document containing:
            - Visual tree representation of package hierarchy
            - Package statistics (total count, max depth)
            - Identification of root packages
            - Module organization insights

            Examples:
            - packages://github.com/django/django/tree
            - packages://github.com/fastapi/fastapi/tree

            Use when: Understanding code organization, planning refactoring,
                     identifying module boundaries, or analyzing architecture.""",
        )
        async def get_package_tree(repository_url: str) -> str:
            """Get the hierarchical package structure of a repository."""
            async with self.session_maker() as session:
                repo = PackageRepository(session)

                try:
                    tree = await repo.get_package_tree(repository_url)

                    if not tree:
                        return f"No package structure found for repository: {repository_url}\n\nTip: Ensure the repository has been added and scanned. Use list_repositories tool to check."

                    return f"""# Package Structure

**Repository**: {repository_url}

## Package Tree
```
{self._format_tree(tree)}
```

## Package Statistics
- **Total Packages**: {self._count_packages(tree)}
- **Max Depth**: {self._get_max_depth(tree)}
- **Root Packages**: {len(tree.get('children', []))}
"""
                except (AttributeError, KeyError, ValueError, TypeError) as e:
                    return f"Error getting package tree: {e!s}"

        @self.mcp.resource(
            "packages://{repository_url}/{package_path}/dependencies",
            description="""Analyze dependencies for a specific package or module.

            Parameters:
            - repository_url: GitHub repository URL (e.g., 'github.com/owner/repo')
            - package_path: Path to package relative to repository root
                           Examples: 'src/auth', 'app/models', 'lib/utils'

            Returns: Markdown document containing:
            - Direct dependencies (what this package imports)
            - Reverse dependencies (what imports this package)
            - Coupling metrics (Ca, Ce, Instability, Abstractness)
            - Dependency graph visualization

            Metrics explained:
            - Afferent Coupling (Ca): Number of packages that depend on this package
            - Efferent Coupling (Ce): Number of packages this package depends on
            - Instability (I): Ce/(Ca+Ce) - ranges from 0 (stable) to 1 (unstable)
            - Abstractness (A): Ratio of interfaces/abstract classes to total classes

            Examples:
            - packages://github.com/myapp/api/src/auth/dependencies
            - packages://github.com/django/django/django/core/dependencies

            Use when: Analyzing coupling, planning module extraction,
                     understanding dependencies before refactoring.""",
        )
        async def get_package_dependencies(
            repository_url: str, package_path: str
        ) -> str:
            """Get dependencies for a specific package."""
            async with self.session_maker() as session:
                repo = PackageRepository(session)

                try:
                    deps = await repo.get_package_dependencies(
                        repository_url, package_path
                    )

                    if not deps:
                        return f"No dependency information found for package: {package_path}\n\nTip: Verify the package path exists. Use packages://{repository_url}/tree to see available packages."

                    return f"""# Package Dependencies

**Repository**: {repository_url}
**Package**: {package_path}

## Direct Dependencies ({len(deps.get('imports', []))})
{self._format_dependencies(deps.get('imports', []))}

## Imported By ({len(deps.get('imported_by', []))})
{self._format_dependencies(deps.get('imported_by', []))}

## Dependency Metrics
- **Afferent Coupling (Ca)**: {deps.get('metrics', {}).get('afferent_coupling', 0)}
- **Efferent Coupling (Ce)**: {deps.get('metrics', {}).get('efferent_coupling', 0)}
- **Instability (I)**: {deps.get('metrics', {}).get('instability', 0):.2f}
- **Abstractness (A)**: {deps.get('metrics', {}).get('abstractness', 0):.2f}
"""
                except (AttributeError, KeyError, ValueError, TypeError) as e:
                    return f"Error getting package dependencies: {e!s}"

        @self.mcp.resource(
            "packages://{repository_url}/circular-dependencies",
            description="""Detect circular dependencies in a repository's package structure.

            Parameters:
            - repository_url: GitHub repository URL (e.g., 'github.com/owner/repo')
            - max_depth: Maximum depth to search for cycles (default: 10)

            Returns: Markdown document containing:
            - List of circular dependency chains
            - Visual representation of cycles
            - Impact analysis
            - Refactoring recommendations to break cycles

            Examples:
            - packages://github.com/myapp/backend/circular-dependencies
            - packages://github.com/legacy/monolith/circular-dependencies

            Use when: Build times are slow, preparing for modularization,
                     investigating coupling issues, or planning service extraction.

            Note: Circular dependencies prevent proper modularization and
                  increase build complexity.""",
        )
        async def find_circular_dependencies(
            repository_url: str, max_depth: int | None = 10
        ) -> str:
            """Find circular dependencies in the repository."""
            async with self.session_maker() as session:
                repo = PackageRepository(session)

                try:
                    circles = await repo.find_circular_dependencies(
                        repository_url, max_depth=max_depth
                    )

                    if not circles:
                        return f"""# Circular Dependencies Analysis

**Repository**: {repository_url}

✅ **No circular dependencies found!**

This is excellent - your package structure maintains a clean dependency hierarchy."""

                    return f"""# Circular Dependencies Analysis

**Repository**: {repository_url}

⚠️ **Found {len(circles)} circular dependency chains**

## Circular Dependencies
{self._format_circular_dependencies(circles)}

## Recommendations
{self._generate_circular_dep_recommendations(circles)}
"""
                except (AttributeError, KeyError, ValueError, TypeError) as e:
                    return f"Error finding circular dependencies: {e!s}"

        @self.mcp.resource(
            "packages://{repository_url}/{package_path}/coupling",
            description="""Get detailed coupling analysis for a specific package.

            Parameters:
            - repository_url: GitHub repository URL (e.g., 'github.com/owner/repo')
            - package_path: Path to package (e.g., 'src/services/auth')

            Returns: Markdown document containing:
            - Coupling metrics (Ca, Ce, I, A, D)
            - List of coupled packages with strength indicators
            - Coupling heatmap visualization
            - Specific recommendations for reducing coupling

            Metrics explained:
            - Ca (Afferent): Packages depending on this one (incoming)
            - Ce (Efferent): Packages this one depends on (outgoing)
            - I (Instability): Ce/(Ca+Ce) - 0=stable, 1=unstable
            - A (Abstractness): Abstract types / Total types
            - D (Distance): |A+I-1| - Distance from main sequence

            Examples:
            - packages://github.com/app/api/src/auth/coupling
            - packages://github.com/lib/core/utils/coupling

            Use when: Evaluating package stability, planning refactoring,
                     identifying highly coupled code, or improving architecture.""",
        )
        async def get_package_coupling_metrics(
            repository_url: str, package_path: str
        ) -> str:
            """Get detailed coupling metrics for a package."""
            async with self.session_maker() as session:
                repo = PackageRepository(session)

                try:
                    metrics = await repo.get_coupling_metrics(
                        repository_url, package_path
                    )

                    if not metrics:
                        return f"No coupling metrics found for package: {package_path}"

                    return f"""# Package Coupling Analysis

**Repository**: {repository_url}
**Package**: {package_path}

## Coupling Metrics
- **Afferent Coupling (Ca)**: {metrics.get('afferent_coupling', 0)}
  - Packages that depend on this package
- **Efferent Coupling (Ce)**: {metrics.get('efferent_coupling', 0)}
  - Packages that this package depends on

## Stability Metrics
- **Instability (I = Ce / (Ca + Ce))**: {metrics.get('instability', 0):.3f}
  - 0 = maximally stable, 1 = maximally unstable
- **Abstractness (A)**: {metrics.get('abstractness', 0):.3f}
  - Ratio of abstract types to total types

## Main Sequence Distance
- **Distance from Main Sequence**: {metrics.get('distance_from_main_sequence', 0):.3f}
  - Ideal packages have A + I = 1

## Coupling Details

### Afferent Couplings (Depended upon by)
{self._format_coupling_list(metrics.get('afferent_details', []))}

### Efferent Couplings (Depends on)
{self._format_coupling_list(metrics.get('efferent_details', []))}

## Analysis
{self._analyze_coupling_metrics(metrics)}
"""
                except (AttributeError, KeyError, ValueError, TypeError) as e:
                    return f"Error getting coupling metrics: {e!s}"

        @self.mcp.resource("packages://{repository_url}/{package_path}/details")
        async def get_package_details(repository_url: str, package_path: str) -> str:
            """Get comprehensive details about a specific package."""
            async with self.session_maker() as session:
                repo = PackageRepository(session)

                try:
                    details = await repo.get_package_details(
                        repository_url, package_path
                    )

                    if not details:
                        return f"Package not found: {package_path}"

                    return f"""# Package Details

**Repository**: {repository_url}
**Package**: {package_path}

## Overview
- **Type**: {details.get('type', 'Unknown')}
- **Size**: {details.get('metrics', {}).get('loc', 0)} lines of code
- **Files**: {len(details.get('files', []))}
- **Classes**: {details.get('metrics', {}).get('classes', 0)}
- **Functions**: {details.get('metrics', {}).get('functions', 0)}

## Contents

### Files
{self._format_file_list(details.get('files', []))}

### Main Classes
{self._format_class_list(details.get('classes', []))}

### Complexity Metrics
- **Average Cyclomatic Complexity**: {details.get('metrics', {}).get('avg_complexity', 0):.1f}
- **Max Cyclomatic Complexity**: {details.get('metrics', {}).get('max_complexity', 0)}
- **Maintainability Index**: {details.get('metrics', {}).get('maintainability_index', 0):.1f}

## Dependencies Summary
- **Internal Dependencies**: {len(details.get('internal_deps', []))}
- **External Dependencies**: {len(details.get('external_deps', []))}
- **Total Imports**: {details.get('metrics', {}).get('total_imports', 0)}

## Package Health
{self._assess_package_health(details)}
"""
                except (AttributeError, KeyError, ValueError, TypeError) as e:
                    return f"Error getting package details: {e!s}"

    def _format_tree(self, tree: dict, prefix: str = "", is_last: bool = True) -> str:
        """Format package tree structure."""
        if not tree:
            return ""

        result = ""
        connector = "└── " if is_last else "├── "
        result += f"{prefix}{connector}{tree['name']}\n"

        children = tree.get("children", [])
        if children:
            extension = "    " if is_last else "│   "
            for i, child in enumerate(children):
                is_last_child = i == len(children) - 1
                result += self._format_tree(child, prefix + extension, is_last_child)

        return result

    def _count_packages(self, tree: dict) -> int:
        """Count total packages in tree."""
        if not tree:
            return 0

        count = 1  # Count this package
        for child in tree.get("children", []):
            count += self._count_packages(child)

        return count

    def _get_max_depth(self, tree: dict, current_depth: int = 0) -> int:
        """Get maximum depth of package tree."""
        if not tree or not tree.get("children"):
            return current_depth

        max_child_depth = current_depth
        for child in tree.get("children", []):
            child_depth = self._get_max_depth(child, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)

        return max_child_depth

    def _format_dependencies(self, deps: list) -> str:
        """Format dependency list."""
        if not deps:
            return "None"

        result = ""
        for dep in sorted(deps):
            result += f"- `{dep}`\n"

        return result

    def _format_circular_dependencies(self, circles: list) -> str:
        """Format circular dependency chains."""
        if not circles:
            return "None found."

        result = ""
        for i, circle in enumerate(circles, 1):
            result += f"### Chain {i}\n"
            result += " → ".join(f"`{pkg}`" for pkg in circle)
            result += f" → `{circle[0]}`\n\n"

        return result

    def _generate_circular_dep_recommendations(self, circles: list) -> str:
        """Generate recommendations for fixing circular dependencies."""
        if not circles:
            return ""

        recommendations = [
            "1. **Extract interfaces**: Create interface packages to break direct dependencies",
            "2. **Introduce abstraction layer**: Add an intermediate package to mediate dependencies",
            "3. **Merge packages**: If packages are tightly coupled, consider combining them",
            "4. **Refactor shared code**: Extract common functionality to a separate package",
            "5. **Use dependency injection**: Inject dependencies rather than importing directly",
        ]

        return "\n".join(recommendations)

    def _format_coupling_list(self, couplings: list) -> str:
        """Format list of coupled packages."""
        if not couplings:
            return "None"

        result = ""
        for coupling in couplings:
            result += f"- `{coupling['package']}` "
            if coupling.get("strength"):
                result += f"(strength: {coupling['strength']})"
            result += "\n"

        return result

    def _analyze_coupling_metrics(self, metrics: dict) -> str:
        """Analyze coupling metrics and provide insights."""
        instability = metrics.get("instability", 0)
        abstractness = metrics.get("abstractness", 0)
        distance = metrics.get("distance_from_main_sequence", 0)

        analysis = []

        if instability > 0.8:
            analysis.append(
                "⚠️ **High instability**: This package depends on many others but few depend on it. Consider reducing dependencies."
            )
        elif instability < 0.2:
            analysis.append(
                "📌 **High stability**: Many packages depend on this. Changes here could have wide impact."
            )

        if distance > 0.3:
            analysis.append(
                "📊 **Far from main sequence**: This package might be poorly designed. Consider refactoring."
            )

        if abstractness < 0.1 and instability < 0.5:
            analysis.append(
                "🔧 **Concrete and stable**: This might be a 'painful' package - hard to change but needs to change."
            )

        if not analysis:
            analysis.append(
                "✅ **Well-balanced package**: Coupling metrics are within healthy ranges."
            )

        return "\n".join(analysis)

    def _format_file_list(self, files: list) -> str:
        """Format list of files in package."""
        if not files:
            return "No files found."

        result = ""
        for file in files[:10]:  # Limit to first 10
            result += f"- `{file['name']}` ({file.get('loc', 0)} lines)\n"

        if len(files) > 10:
            result += f"- ... and {len(files) - 10} more files\n"

        return result

    def _format_class_list(self, classes: list) -> str:
        """Format list of main classes."""
        if not classes:
            return "No classes found."

        result = ""
        for cls in classes[:5]:  # Limit to first 5
            result += f"- **{cls['name']}**"
            if cls.get("methods"):
                result += f" ({len(cls['methods'])} methods)"
            result += "\n"

        if len(classes) > 5:
            result += f"- ... and {len(classes) - 5} more classes\n"

        return result

    def _assess_package_health(self, details: dict) -> str:
        """Assess overall package health."""
        metrics = details.get("metrics", {})
        issues = []

        if metrics.get("avg_complexity", 0) > 10:
            issues.append("⚠️ High average complexity")

        if metrics.get("loc", 0) > 1000:
            issues.append("📏 Large package size")

        if len(details.get("external_deps", [])) > 20:
            issues.append("🔗 Many external dependencies")

        if metrics.get("maintainability_index", 100) < 50:
            issues.append("🔧 Low maintainability index")

        if not issues:
            return "✅ **Healthy Package**: All metrics are within acceptable ranges."
        return "**Issues Found**:\n" + "\n".join(f"- {issue}" for issue in issues)
