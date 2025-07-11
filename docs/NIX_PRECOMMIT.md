# Nix-based Pre-commit Hooks

This project now uses a Nix-based pre-commit solution that works correctly on NixOS, including full support for Ruff and all other Python tools.

## Why Nix-based Pre-commit?

Traditional pre-commit hooks install tools via pip in isolated environments. On NixOS, this causes issues because:
- Binary tools like Ruff expect system libraries in standard locations (/usr/lib)
- NixOS stores libraries in /nix/store with unique paths
- The dynamic linker can't find required libraries without Nix's environment setup

Our Nix-based solution ensures all tools are properly linked and work correctly on NixOS.

## Features

The `nix-pre-commit` script runs the following checks:
- **Black** - Python code formatting
- **isort** - Import sorting
- **Ruff** - Fast Python linting (now works on NixOS!)
- **Bandit** - Security vulnerability scanning

## Usage

### Enter the Nix development shell
```bash
nix develop
```

### Run checks manually
```bash
# Check staged files (default)
nix-pre-commit

# Check all files
nix-pre-commit --all-files
```

### Install as a Git hook
```bash
# Create the pre-commit hook
echo '#!/bin/sh' > .git/hooks/pre-commit
echo 'exec nix develop --command nix-pre-commit' >> .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

## Migrating from pip-based pre-commit

If you were using the traditional pre-commit setup:

1. Uninstall old hooks:
   ```bash
   pre-commit uninstall
   ```

2. Clean up:
   ```bash
   rm -rf ~/.cache/pre-commit
   ```

3. Use the new Nix-based hooks as described above

## Benefits

1. **Full NixOS compatibility** - All tools work correctly, including Ruff
2. **Reproducible** - Same tool versions for everyone using the flake
3. **Fast** - No need to create isolated virtualenvs for each tool
4. **Simple** - Just one script that uses the project's development environment

## Customization

To modify the checks, edit the `nixPreCommit` definition in `flake.nix`. The script is a simple bash script that runs each tool and reports results.
