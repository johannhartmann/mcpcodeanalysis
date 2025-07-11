"""Version management utilities for dynamic version retrieval."""

import importlib.metadata
from functools import lru_cache
from typing import NamedTuple

from src.logger import get_logger

logger = get_logger(__name__)


class VersionInfo(NamedTuple):
    """Version information container."""

    version: str
    major: int
    minor: int
    patch: int
    pre_release: str | None = None
    build: str | None = None


@lru_cache(maxsize=1)
def get_package_version(package_name: str = "mcp-code-analysis-server") -> str:
    """
    Get the current package version from installed metadata.

    Args:
        package_name: The name of the package to get version for

    Returns:
        Version string (e.g., "0.1.0")

    Raises:
        ImportError: If package metadata cannot be found
    """
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        logger.warning(
            "Package %s not found in metadata, falling back to default version",
            package_name,
        )
        # Fallback to reading from __init__.py
        try:
            import src

            return src.__version__
        except (ImportError, AttributeError):
            logger.error("Cannot determine package version, using fallback")
            return "0.1.0"


@lru_cache(maxsize=1)
def get_version_info(package_name: str = "mcp-code-analysis-server") -> VersionInfo:
    """
    Get detailed version information including parsed components.

    Args:
        package_name: The name of the package to get version for

    Returns:
        VersionInfo object with parsed version components
    """
    version_str = get_package_version(package_name)

    try:
        # Parse semantic version (e.g., "1.2.3", "1.2.3-alpha.1", "1.2.3+build.1")
        version_parts = version_str.split("+")
        build = version_parts[1] if len(version_parts) > 1 else None

        version_and_prerelease = version_parts[0].split("-")
        pre_release = (
            version_and_prerelease[1] if len(version_and_prerelease) > 1 else None
        )

        version_components = version_and_prerelease[0].split(".")
        major = int(version_components[0])
        minor = int(version_components[1]) if len(version_components) > 1 else 0
        patch = int(version_components[2]) if len(version_components) > 2 else 0

        return VersionInfo(
            version=version_str,
            major=major,
            minor=minor,
            patch=patch,
            pre_release=pre_release,
            build=build,
        )
    except (ValueError, IndexError) as e:
        logger.warning("Error parsing version %s: %s", version_str, e)
        return VersionInfo(version=version_str, major=0, minor=1, patch=0)


def compare_versions(version1: str, version2: str) -> int:
    """
    Compare two version strings.

    Args:
        version1: First version string
        version2: Second version string

    Returns:
        -1 if version1 < version2
         0 if version1 == version2
         1 if version1 > version2
    """
    v1_info = get_version_info()
    v1_info = VersionInfo(version=version1, major=0, minor=0, patch=0)

    v2_info = VersionInfo(version=version2, major=0, minor=0, patch=0)

    # Parse versions
    try:
        v1_parts = version1.split(".")
        v1_major = int(v1_parts[0])
        v1_minor = int(v1_parts[1]) if len(v1_parts) > 1 else 0
        v1_patch = int(v1_parts[2]) if len(v1_parts) > 2 else 0

        v2_parts = version2.split(".")
        v2_major = int(v2_parts[0])
        v2_minor = int(v2_parts[1]) if len(v2_parts) > 1 else 0
        v2_patch = int(v2_parts[2]) if len(v2_parts) > 2 else 0

        # Compare major.minor.patch
        if v1_major != v2_major:
            return 1 if v1_major > v2_major else -1
        if v1_minor != v2_minor:
            return 1 if v1_minor > v2_minor else -1
        if v1_patch != v2_patch:
            return 1 if v1_patch > v2_patch else -1

        return 0

    except (ValueError, IndexError) as e:
        logger.warning("Error comparing versions %s and %s: %s", version1, version2, e)
        return 0  # Consider equal if parsing fails


def is_compatible_version(
    required_version: str, current_version: str | None = None
) -> bool:
    """
    Check if current version is compatible with required version.

    Args:
        required_version: Required version string
        current_version: Current version string (defaults to package version)

    Returns:
        True if current version is compatible with required version
    """
    if current_version is None:
        current_version = get_package_version()

    # For now, implement simple comparison
    # In the future, this could support more complex compatibility rules
    return compare_versions(current_version, required_version) >= 0


def get_build_info() -> dict[str, str]:
    """
    Get build information including version, git commit, and timestamp.

    Returns:
        Dictionary with build information
    """
    version_info = get_version_info()

    build_info = {
        "version": version_info.version,
        "major": str(version_info.major),
        "minor": str(version_info.minor),
        "patch": str(version_info.patch),
    }

    if version_info.pre_release:
        build_info["pre_release"] = version_info.pre_release

    if version_info.build:
        build_info["build"] = version_info.build

    # Try to get git information
    try:
        import subprocess  # nosec B404

        # Get git commit hash
        result = subprocess.run(  # nosec B603, B607
            ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            build_info["git_commit"] = result.stdout.strip()[:8]  # Short hash

        # Get git branch
        result = subprocess.run(  # nosec B603, B607
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=False, capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            build_info["git_branch"] = result.stdout.strip()

        # Get git status (dirty/clean)
        result = subprocess.run(  # nosec B603, B607
            ["git", "status", "--porcelain"], check=False, capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            build_info["git_dirty"] = "true" if result.stdout.strip() else "false"

    except (
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        logger.debug("Git information not available")

    return build_info


def get_api_version() -> str:
    """
    Get the API version for MCP server compatibility.

    Returns:
        API version string compatible with MCP protocol
    """
    # MCP protocol version - this should be updated when MCP protocol changes
    return "2024-11-05"


def get_server_info() -> dict[str, str]:
    """
    Get server information for health checks and API responses.

    Returns:
        Dictionary with server information
    """
    version_info = get_version_info()
    build_info = get_build_info()

    return {
        "name": "MCP Code Analysis Server",
        "version": version_info.version,
        "api_version": get_api_version(),
        "build_info": build_info,
        "status": "healthy",
    }
