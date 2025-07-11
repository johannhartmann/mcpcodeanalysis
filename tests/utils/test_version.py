"""Tests for version management utilities."""

import subprocess
from unittest.mock import Mock, patch

from src.utils.version import (
    VersionInfo,
    compare_versions,
    get_api_version,
    get_build_info,
    get_package_version,
    get_server_info,
    get_version_info,
    is_compatible_version,
)


class TestVersionUtils:
    """Tests for version utilities."""

    def test_version_info_creation(self):
        """Test VersionInfo creation."""
        version_info = VersionInfo(
            version="1.2.3",
            major=1,
            minor=2,
            patch=3,
            pre_release="alpha.1",
            build="build.123",
        )

        assert version_info.version == "1.2.3"
        assert version_info.major == 1
        assert version_info.minor == 2
        assert version_info.patch == 3
        assert version_info.pre_release == "alpha.1"
        assert version_info.build == "build.123"

    @patch("importlib.metadata.version")
    def test_get_package_version_success(self, mock_version):
        """Test successful package version retrieval."""
        mock_version.return_value = "1.2.3"

        version = get_package_version("test-package")

        assert version == "1.2.3"
        mock_version.assert_called_once_with("test-package")

    def test_get_package_version_default(self):
        """Test default package version retrieval."""
        version = get_package_version()

        # Should return the actual version (0.1.0 by default)
        assert version == "0.1.0"

    def test_get_version_info_simple(self):
        """Test version info parsing for simple version."""
        # Test with actual version
        version_info = get_version_info()

        assert version_info.version == "0.1.0"
        assert version_info.major == 0
        assert version_info.minor == 1
        assert version_info.patch == 0
        assert version_info.pre_release is None
        assert version_info.build is None

    def test_compare_versions_equal(self):
        """Test version comparison for equal versions."""
        assert compare_versions("1.2.3", "1.2.3") == 0

    def test_compare_versions_greater(self):
        """Test version comparison for greater version."""
        assert compare_versions("1.2.4", "1.2.3") == 1
        assert compare_versions("1.3.0", "1.2.9") == 1
        assert compare_versions("2.0.0", "1.9.9") == 1

    def test_compare_versions_lesser(self):
        """Test version comparison for lesser version."""
        assert compare_versions("1.2.2", "1.2.3") == -1
        assert compare_versions("1.1.9", "1.2.0") == -1
        assert compare_versions("0.9.9", "1.0.0") == -1

    def test_compare_versions_missing_components(self):
        """Test version comparison with missing components."""
        assert compare_versions("1.2", "1.2.0") == 0
        assert compare_versions("1", "1.0.0") == 0
        assert compare_versions("1.2.1", "1.2") == 1

    def test_compare_versions_invalid(self):
        """Test version comparison with invalid versions."""
        assert compare_versions("invalid", "1.2.3") == 0
        assert compare_versions("1.2.3", "invalid") == 0

    @patch("src.utils.version.get_package_version")
    def test_is_compatible_version_compatible(self, mock_get_version):
        """Test version compatibility check for compatible versions."""
        mock_get_version.return_value = "1.2.3"

        assert is_compatible_version("1.2.3") is True
        assert is_compatible_version("1.2.0") is True
        assert is_compatible_version("1.1.0") is True

    @patch("src.utils.version.get_package_version")
    def test_is_compatible_version_incompatible(self, mock_get_version):
        """Test version compatibility check for incompatible versions."""
        mock_get_version.return_value = "1.2.3"

        assert is_compatible_version("1.2.4") is False
        assert is_compatible_version("1.3.0") is False
        assert is_compatible_version("2.0.0") is False

    def test_is_compatible_version_with_current(self):
        """Test version compatibility check with explicit current version."""
        assert is_compatible_version("1.2.0", "1.2.3") is True
        assert is_compatible_version("1.2.4", "1.2.3") is False

    @patch("src.utils.version.get_version_info")
    def test_get_build_info_basic(self, mock_get_version_info):
        """Test basic build info retrieval."""
        mock_get_version_info.return_value = VersionInfo(
            version="1.2.3", major=1, minor=2, patch=3
        )

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "git")

            build_info = get_build_info()

            assert build_info["version"] == "1.2.3"
            assert build_info["major"] == "1"
            assert build_info["minor"] == "2"
            assert build_info["patch"] == "3"

    @patch("src.utils.version.get_version_info")
    def test_get_build_info_with_git(self, mock_get_version_info):
        """Test build info retrieval with git information."""
        mock_get_version_info.return_value = VersionInfo(
            version="1.2.3",
            major=1,
            minor=2,
            patch=3,
            pre_release="alpha.1",
            build="build.123",
        )

        with patch("subprocess.run") as mock_run:
            # Mock git commands
            mock_run.side_effect = [
                Mock(returncode=0, stdout="abcd1234\n"),  # git rev-parse HEAD
                Mock(returncode=0, stdout="main\n"),  # git rev-parse --abbrev-ref HEAD
                Mock(returncode=0, stdout=""),  # git status --porcelain
            ]

            build_info = get_build_info()

            assert build_info["version"] == "1.2.3"
            assert build_info["pre_release"] == "alpha.1"
            assert build_info["build"] == "build.123"
            assert build_info["git_commit"] == "abcd1234"
            assert build_info["git_branch"] == "main"
            assert build_info["git_dirty"] == "false"

    @patch("src.utils.version.get_version_info")
    def test_get_build_info_with_dirty_git(self, mock_get_version_info):
        """Test build info retrieval with dirty git repository."""
        mock_get_version_info.return_value = VersionInfo(
            version="1.2.3", major=1, minor=2, patch=3
        )

        with patch("subprocess.run") as mock_run:
            # Mock git commands with dirty status
            mock_run.side_effect = [
                Mock(returncode=0, stdout="abcd1234\n"),  # git rev-parse HEAD
                Mock(returncode=0, stdout="main\n"),  # git rev-parse --abbrev-ref HEAD
                Mock(
                    returncode=0, stdout="M file.py\n"
                ),  # git status --porcelain (dirty)
            ]

            build_info = get_build_info()

            assert build_info["git_dirty"] == "true"

    def test_get_api_version(self):
        """Test API version retrieval."""
        api_version = get_api_version()

        assert api_version == "2024-11-05"

    @patch("src.utils.version.get_version_info")
    @patch("src.utils.version.get_build_info")
    def test_get_server_info(self, mock_get_build_info, mock_get_version_info):
        """Test server info retrieval."""
        mock_get_version_info.return_value = VersionInfo(
            version="1.2.3", major=1, minor=2, patch=3
        )

        mock_get_build_info.return_value = {
            "version": "1.2.3",
            "git_commit": "abcd1234",
            "git_branch": "main",
        }

        server_info = get_server_info()

        assert server_info["name"] == "MCP Code Analysis Server"
        assert server_info["version"] == "1.2.3"
        assert server_info["api_version"] == "2024-11-05"
        assert server_info["status"] == "healthy"
        assert "build_info" in server_info
        assert server_info["build_info"]["git_commit"] == "abcd1234"

    def test_version_info_defaults(self):
        """Test VersionInfo with default values."""
        version_info = VersionInfo(version="1.2.3", major=1, minor=2, patch=3)

        assert version_info.pre_release is None
        assert version_info.build is None
