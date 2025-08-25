.PHONY: help install dev-install format lint type-check test test-unit test-integration test-all security clean pre-commit qa

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
