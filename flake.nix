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
              
              # Utilities
              pkgs.jq
              pkgs.curl
              pkgs.httpie
              pkgs.pgcli
              pkgs.mdbook
              
              # ViberDash for code quality monitoring
              viberdash.packages.${system}.default
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
              echo "  viberdash monitor --interval 300 - Monitor code quality metrics (5 min updates)"
              echo ""
              echo "Server commands:"
              echo "  mcp-code-server      - Start MCP server"
              echo "  mcp-scanner          - Run code scanner"
              echo "  mcp-indexer          - Run indexer"
              echo ""
              
              # Set up environment
              export PATH="${virtualenv}/bin:$PATH"
              export PYTHONPATH="$PWD/src:$PYTHONPATH"
              export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
              
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
              echo "  uv run viberdash monitor --interval 300 - Monitor code quality metrics (5 min updates)"
              echo ""
              
              export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
            '';
          };
        }
      );
    };
}