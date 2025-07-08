{
  description = "MCP Code Analysis Server - Intelligent code analysis and search";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, uv2nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python311;
        
        # PostgreSQL with pgvector extension
        postgresqlWithExtensions = pkgs.postgresql_16.withPackages (p: [
          p.pgvector
        ]);

        pythonEnv = python.withPackages (ps: with ps; [
          # Core dependencies (minimal for nix shell)
          pip
          setuptools
          wheel
          
          # Development tools
          black
          ruff
          mypy
          pytest
          pytest-asyncio
          ipython
          
          # Type stubs
          types-requests
          types-pyyaml
          # Note: pandas-stubs removed as it pulls in PyTorch as dependency in nixpkgs
        ]);
        
        pyprojectToml = builtins.fromTOML (builtins.readFile ./pyproject.toml);
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Python environment
            pythonEnv
            uv
            
            # Database
            postgresqlWithExtensions
            
            # Development tools
            git
            docker
            docker-compose
            
            # Code analysis tools
            tree-sitter
            
            # Utilities
            jq
            curl
            httpie
            pgcli
            
            # Documentation
            mdbook
          ];

          shellHook = ''
            echo "────────────────────────────────────────────"
            echo "    MCP Code Analysis Server Development"
            echo "────────────────────────────────────────────"
            echo ""
            echo "Available commands:"
            echo "  uv sync              - Install Python dependencies"
            echo "  uv run pytest        - Run tests"
            echo "  uv run ruff check .  - Run linter"
            echo "  uv run mypy .        - Type checking"
            echo "  uv run black .       - Format code"
            echo ""
            echo "Server commands:"
            echo "  uv run mcp-code-server  - Start MCP server"
            echo "  uv run mcp-scanner      - Run code scanner"
            echo "  uv run mcp-indexer      - Run indexer"
            echo ""
            echo "Database:"
            echo "  docker-compose up -d postgres  - Start PostgreSQL"
            echo "  pgcli -h localhost -U codeanalyzer -d code_analysis"
            echo ""
            echo "Environment setup:"
            echo "  cp .env.example .env  - Create environment file"
            echo "  cp config.example.yaml config.yaml"
            echo ""
            
            # Create project directories if they don't exist
            mkdir -p src/{scanner,parser,embeddings,mcp_server,query,database}
            mkdir -p tests
            mkdir -p docs
            
            # Set Python path
            export PYTHONPATH="$PWD/src:$PYTHONPATH"
            
            # Check if .env exists
            if [ ! -f .env ]; then
              echo "⚠️  No .env file found. Create one with:"
              echo "   cp .env.example .env"
            fi
            
            # Check if config.yaml exists
            if [ ! -f config.yaml ]; then
              echo "⚠️  No config.yaml found. Create one with:"
              echo "   cp config.example.yaml config.yaml"
            fi
          '';

          PYTHONPATH = "./src";
          
          # Development environment variables
          POSTGRES_HOST = "localhost";
          POSTGRES_PORT = "5432";
          POSTGRES_DB = "code_analysis";
          POSTGRES_USER = "codeanalyzer";
        };
        
        packages.default = python.pkgs.buildPythonPackage {
          pname = "mcp-code-analysis-server";
          version = pyprojectToml.project.version;
          
          src = ./.;
          
          format = "pyproject";
          
          nativeBuildInputs = with python.pkgs; [
            hatchling
          ];
          
          propagatedBuildInputs = with python.pkgs; [
            # Core dependencies from pyproject.toml
            fastapi
            uvicorn
            httpx
            pydantic
            pydantic-settings
            sqlalchemy
            alembic
            asyncpg
            psycopg2
            gitpython
            pyyaml
            click
            python-dotenv
            rich
            tenacity
            openai
            tiktoken
            numpy
            pandas
            
            # Note: Some packages like fastmcp, langchain, langgraph, pgvector, 
            # and tree-sitter-python may need to be installed via pip/uv
          ];
          
          checkInputs = with python.pkgs; [
            pytest
            pytest-asyncio
            pytest-cov
            pytest-mock
          ];
          
          meta = with pkgs.lib; {
            description = pyprojectToml.project.description;
            homepage = pyprojectToml.project.urls.Homepage;
            license = licenses.mit;
            maintainers = [];
          };
        };
        
        # Docker image
        dockerImage = pkgs.dockerTools.buildImage {
          name = "mcp-code-analysis-server";
          tag = "latest";
          
          contents = [
            self.packages.${system}.default
            pkgs.bash
            pkgs.coreutils
          ];
          
          config = {
            Cmd = [ "${self.packages.${system}.default}/bin/mcp-code-server" ];
            Env = [
              "PYTHONUNBUFFERED=1"
            ];
            ExposedPorts = {
              "8080/tcp" = {};
            };
          };
        };
      });
}