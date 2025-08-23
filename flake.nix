{
  description = "MCP Code Analysis Server";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    viberdash = {
      url = "github:johannhartmann/viberdash";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
      viberdash,
    }:
    let
      inherit (nixpkgs) lib;

      # Load the uv workspace from uv.lock
      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

      # Create package overlay from workspace
      overlay = workspace.mkPyprojectOverlay {
        sourcePreference = "wheel";  # Prefer pre-built wheels
      };

      # Override sets
      pyprojectOverrides = _final: _prev: {
        # Add any specific package overrides here if needed
      };

      # Generate outputs for each system
      eachSystem = nixpkgs.lib.genAttrs [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

    in
    {
      # Expose the overlay
      overlays.default = nixpkgs.lib.composeManyExtensions [
        pyproject-nix.overlays.default
        pyproject-build-systems.overlays.default
        overlay
        pyprojectOverrides
      ];

      # Development shells
      devShells = eachSystem (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};

          # Python version
          python = pkgs.python313;

          # Construct Python package set
          pythonSet = (pkgs.callPackage pyproject-nix.build.packages {
            inherit python;
          }).overrideScope (
            nixpkgs.lib.composeManyExtensions [
              pyproject-build-systems.overlays.default
              overlay
              pyprojectOverrides
            ]
          );

          # Create a virtual environment with all dependencies
          virtualenv = pythonSet.mkVirtualEnv "mcp-code-analysis-env" {
            mcp-code-analysis-server = [ "dev" ];
          };

          # PostgreSQL with pgvector
          postgresqlWithExtensions = pkgs.postgresql_16.withPackages (p: [
            p.pgvector
          ]);

          # Custom pre-commit script that works on NixOS
          nixPreCommit = pkgs.writeShellScriptBin "nix-pre-commit" ''
            #!/usr/bin/env bash
            set -euo pipefail

            # Colors
            RED='\033[0;31m'
            GREEN='\033[0;32m'
            YELLOW='\033[0;33m'
            NC='\033[0m' # No Color

            echo -e "''${GREEN}Running Nix-based pre-commit checks...''${NC}"

            # Get list of Python files
            if [ "$1" = "--all-files" ]; then
              FILES=$(find src tests -name "*.py" -type f 2>/dev/null || true)
            else
              FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$' || true)
            fi

            if [ -z "$FILES" ]; then
              echo "No Python files to check"
              exit 0
            fi

            # Track if any check fails
            FAILED=0

            # Run Black
            echo -e "\n''${YELLOW}Running Black...''${NC}"
            if ${virtualenv}/bin/black --check $FILES; then
              echo -e "''${GREEN}✓ Black passed''${NC}"
            else
              echo -e "''${RED}✗ Black failed - run 'black .' to fix''${NC}"
              FAILED=1
            fi

            # Run isort
            echo -e "\n''${YELLOW}Running isort...''${NC}"
            if ${virtualenv}/bin/isort --check-only $FILES; then
              echo -e "''${GREEN}✓ isort passed''${NC}"
            else
              echo -e "''${RED}✗ isort failed - run 'isort .' to fix''${NC}"
              FAILED=1
            fi

            # Run Ruff (this now works on NixOS!)
            echo -e "\n''${YELLOW}Running Ruff...''${NC}"
            if ${virtualenv}/bin/ruff check $FILES; then
              echo -e "''${GREEN}✓ Ruff passed''${NC}"
            else
              echo -e "''${RED}✗ Ruff failed''${NC}"
              FAILED=1
            fi

            # Run Bandit
            echo -e "\n''${YELLOW}Running Bandit...''${NC}"
            if ${virtualenv}/bin/bandit -c pyproject.toml $FILES; then
              echo -e "''${GREEN}✓ Bandit passed''${NC}"
            else
              echo -e "''${RED}✗ Bandit failed''${NC}"
              FAILED=1
            fi

            # Run Vulture
            echo -e "\n''${YELLOW}Running Vulture...''${NC}"
            if ${virtualenv}/bin/vulture $FILES vulture_whitelist.py --min-confidence 80; then
              echo -e "''${GREEN}✓ Vulture passed''${NC}"
            else
              echo -e "''${RED}✗ Vulture found dead code''${NC}"
              echo -e "''${YELLOW}Note: Review findings - some might be false positives''${NC}"
              # Don't fail on vulture - it often has false positives
            fi

            # Summary
            echo ""
            if [ $FAILED -eq 0 ]; then
              echo -e "''${GREEN}All checks passed!''${NC}"
            else
              echo -e "''${RED}Some checks failed!''${NC}"
              exit 1
            fi
          '';

        in
        {
          # Pure development shell with dependencies from uv.lock
          default = pkgs.mkShell {
            packages = [
              # Virtual environment with all workspace dependencies
              virtualenv

              # System libraries
              pkgs.stdenv.cc.cc.lib
              pkgs.zlib

              # Database
              postgresqlWithExtensions

              # Development tools
              pkgs.git
              pkgs.docker
              pkgs.docker-compose
              pkgs.tree-sitter
              pkgs.just

              # Utilities
              pkgs.jq
              pkgs.curl
              pkgs.httpie
              pkgs.pgcli
              pkgs.mdbook

              # ViberDash for code quality monitoring
              viberdash.packages.${system}.default

              # Nix-based pre-commit script
              nixPreCommit
            ];

            shellHook = ''

              echo "────────────────────────────────────────────"
              echo "    MCP Code Analysis Server Development"
              echo "────────────────────────────────────────────"
              echo ""
              echo "Python environment is ready!"
              echo "Python: ${virtualenv}/bin/python"
              echo ""
              echo "Available commands:"
              echo "  pytest               - Run tests"
              echo "  ruff check .         - Run linter"
              echo "  mypy .               - Type checking"
              echo "  black .              - Format code"
              echo "  pyupgrade --py311-plus <files> - Upgrade Python syntax"
              echo "  vulture src vulture_whitelist.py - Find dead code"
              echo "  viberdash monitor --interval 300 - Monitor code quality metrics (5 min updates)"
              echo "  vdash                - Start ViberDash monitor (alias)"
              echo ""
              echo "Server commands:"
              echo "  mcp-code-server      - Start MCP server"
              echo "  mcp-scanner          - Run code scanner"
              echo "  mcp-indexer          - Run indexer"
              echo ""
              echo "Pre-commit hooks (Nix-based - works with Ruff!):"
              echo "  nix-pre-commit       - Run all checks on staged files"
              echo "  nix-pre-commit --all-files - Run all checks on all files"
              echo ""
              echo "To install as git hook:"
              echo "  echo '#!/bin/sh' > .git/hooks/pre-commit"
              echo "  echo 'nix-pre-commit' >> .git/hooks/pre-commit"
              echo "  chmod +x .git/hooks/pre-commit"
              echo ""

              # Set up environment
              export PATH="${virtualenv}/bin:$PATH"
              export PYTHONPATH="$PWD/src:$PYTHONPATH"
              export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"

              # Simple alias for viberdash
              alias vdash='viberdash monitor --interval 300'

              # Check configuration files
              if [ ! -f .env ]; then
                echo "⚠️  No .env file found. Create one with:"
                echo "   cp .env.example .env"
              fi

              if [ ! -f config.yaml ]; then
                echo "⚠️  No config.yaml found. Create one with:"
                echo "   cp config.example.yaml config.yaml"
              fi
            '';

            # Environment variables
            POSTGRES_HOST = "localhost";
            POSTGRES_PORT = "5432";
            POSTGRES_DB = "code_analysis";
            POSTGRES_USER = "codeanalyzer";
          };

          # Impure shell for using uv directly
          impure = pkgs.mkShell {
            packages = with pkgs; [
              python313
              uv

              # System libraries
              stdenv.cc.cc.lib
              zlib

              # Database
              postgresqlWithExtensions

              # Development tools
              git
              docker
              docker-compose
              tree-sitter
              just

              # Utilities
              jq
              curl
              httpie
              pgcli
              mdbook

              # ViberDash for code quality monitoring
              viberdash.packages.${system}.default
            ];

            shellHook = ''
              echo "────────────────────────────────────────────"
              echo "    MCP Code Analysis Server Development"
              echo "      (Impure shell - using uv directly)"
              echo "────────────────────────────────────────────"
              echo ""
              echo "Available commands:"
              echo "  uv sync              - Install Python dependencies"
              echo "  uv run pytest        - Run tests"
              echo "  uv run ruff check .  - Run linter"
              echo "  uv run mypy .        - Type checking"
              echo "  uv run black .       - Format code"
              echo "  uv run pyupgrade --py311-plus <files> - Upgrade Python syntax"
              echo "  uv run vulture src vulture_whitelist.py - Find dead code"
              echo "  uv run viberdash monitor --interval 300 - Monitor code quality metrics (5 min updates)"
              echo "  vdash                - Start ViberDash monitor (alias)"
              echo ""

              export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"

              # Simple alias for viberdash
              alias vdash='viberdash monitor --interval 300'
            '';
          };
        }
      );
    };
}
