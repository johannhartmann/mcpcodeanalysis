.PHONY: help install dev-install format lint type-check test test-unit test-integration test-all test-integration-docker security clean pre-commit qa docker-up-integration docker-down-integration docker-logs-integration

help:  ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install:  ## Install production dependencies
	uv sync --no-dev

dev-install:  ## Install all dependencies including dev
	uv sync
	uv run pre-commit install

format:  ## Format code with black and isort
	uv run black src tests
	uv run isort src tests

lint:  ## Run all linters
	uv run ruff check src tests
	uv run pylint src
	uv run bandit -r src

type-check:  ## Run type checking with mypy
	uv run mypy src tests

test:  ## Run unit tests
	uv run pytest -m "unit" -v

test-unit:  ## Run unit tests with coverage
	uv run pytest -m "unit" --cov=src --cov-report=term-missing

test-integration:  ## Run integration tests
	uv run pytest -m "integration" -v

test-all:  ## Run all tests with coverage
	uv run pytest --cov=src --cov-report=term-missing --cov-report=html

security:  ## Run security checks
	uv run bandit -r src
	uv run safety check

clean:  ## Clean up generated files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/

pre-commit:  ## Run pre-commit on all files
	uv run pre-commit run --all-files

qa:  ## Run all quality assurance checks
	@echo "Running format..."
	@make format
	@echo "\nRunning linters..."
	@make lint
	@echo "\nRunning type checks..."
	@make type-check
	@echo "\nRunning tests..."
	@make test-all
	@echo "\nRunning security checks..."
	@make security
	@echo "\n✅ All QA checks passed!"

# ------------------------------
# Integration via Docker Compose
# ------------------------------

# Choose your compose command. Override if you use `docker compose`:
#   make test-integration-docker DOCKER_COMPOSE="docker compose"
DOCKER_COMPOSE ?= docker-compose

docker-up-integration: ## Start Postgres + MCP server containers (detached)
	$(DOCKER_COMPOSE) up -d --build postgres mcp-server

docker-down-integration: ## Stop and remove integration containers + volumes
	$(DOCKER_COMPOSE) down -v

docker-logs-integration: ## Tail MCP server logs
	$(DOCKER_COMPOSE) logs -f mcp-server

MCP_EXTERNAL_PORT ?= 8080

test-integration-docker: ## Build, start stack, wait for server, run integration tests, teardown
	@set -e; \
	status=0; \
	$(DOCKER_COMPOSE) up -d --build postgres mcp-server; \
	echo "Waiting for MCP server on http://localhost:$(MCP_EXTERNAL_PORT)/mcp/ ..."; \
	for i in $$(seq 1 60); do \
		if curl -s -o /dev/null http://localhost:$(MCP_EXTERNAL_PORT)/mcp/; then \
			echo "MCP server is up"; \
			break; \
		fi; \
		sleep 2; \
		if [ $$i -eq 60 ]; then \
			echo "Timeout waiting for MCP server. Recent logs:"; \
			$(DOCKER_COMPOSE) logs --tail=200 mcp-server || true; \
			exit 1; \
		fi; \
	done; \
	MCP_EXTERNAL_PORT=$(MCP_EXTERNAL_PORT) uv run pytest -m "integration" -v || status=$$?; \
	if [ "$$KEEP_CONTAINERS" != "1" ]; then \
		$(DOCKER_COMPOSE) down -v; \
	else \
		echo "KEEP_CONTAINERS=1 set; leaving containers running"; \
	fi; \
	exit $$status

watch-test:  ## Watch for changes and run tests
	uv run ptw --runner "pytest -x -vs"

coverage-html:  ## Generate HTML coverage report and open it
	uv run pytest --cov=src --cov-report=html
	@echo "Opening coverage report..."
	@python -m webbrowser htmlcov/index.html

docs:  ## Build documentation
	uv run mkdocs build

docs-serve:  ## Serve documentation locally
	uv run mkdocs serve

version:  ## Show current version
	@grep version pyproject.toml | head -1 | cut -d'"' -f2

bump-patch:  ## Bump patch version
	uv run semantic-release version patch

bump-minor:  ## Bump minor version
	uv run semantic-release version minor

bump-major:  ## Bump major version
	uv run semantic-release version major
