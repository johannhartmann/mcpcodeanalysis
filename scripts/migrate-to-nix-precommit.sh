#!/usr/bin/env bash
# Script to migrate from pip-based pre-commit to Nix-based pre-commit

set -euo pipefail

echo "Migrating to Nix-based pre-commit hooks..."

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Error: Not in a git repository"
    exit 1
fi

# Uninstall existing pre-commit hooks
if [ -f .git/hooks/pre-commit ]; then
    echo "Removing existing pre-commit hooks..."
    pre-commit uninstall 2>/dev/null || true
fi

# Remove pip-based pre-commit cache
if [ -d ~/.cache/pre-commit ]; then
    echo "Cleaning pre-commit cache..."
    rm -rf ~/.cache/pre-commit
fi

# Backup existing .pre-commit-config.yaml
if [ -f .pre-commit-config.yaml ]; then
    echo "Backing up existing .pre-commit-config.yaml to .pre-commit-config.yaml.bak"
    mv .pre-commit-config.yaml .pre-commit-config.yaml.bak
fi

echo ""
echo "Migration complete! To use the Nix-based pre-commit hooks:"
echo ""
echo "1. Enter the nix development shell:"
echo "   nix develop"
echo ""
echo "2. Install the git hooks:"
echo "   pre-commit install"
echo ""
echo "3. Test the hooks:"
echo "   pre-commit run --all-files"
echo ""
echo "All pre-commit tools (including Ruff) are now provided by Nix and will work correctly on NixOS."
echo ""
echo "Your old .pre-commit-config.yaml has been backed up to .pre-commit-config.yaml.bak"
