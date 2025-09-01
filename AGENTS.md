# Repository Guidelines

## Project Structure & Module Organization
- `src/`: Python package with core modules — `mcp_server/` (MCP server), `scanner/`, `parser/`, `query/`, `embeddings/`, `database/`, `domain/`, `indexer/`, `utils/`.
- `tests/`: Mirrors `src/` by area with `test_*.py` files; includes `unit/`, `integration/`, and feature folders.
- `docs/`, `alembic.ini`, `docker-compose.yml`, `config.example.yaml`, `settings.toml` support docs, DB migrations, and config.
- Entry points (installed via `uv`): `mcp-code-server`, `mcp-scanner`, `mcp-indexer`.

## Build, Test, and Development Commands
- Install deps: `make dev-install` (dev + pre-commit) or `make install` (prod only).
- Lint/format/type-check: `make lint`, `make format`, `make type-check`, full QA: `make qa`.
- Run tests: `make test` (unit), `make test-integration`, `make test-all`, coverage HTML: `make coverage-html`.
- Quick feedback loop (Just): `just unveil-next-error`, `just see-what-you-did-there`.
- Run locally: `uv run mcp-code-server` or `uv run python -m src.mcp_server`.

## Coding Style & Naming Conventions
- Python 3.11+, Black (88 cols), Ruff, isort; 4-space indents.
- Type hints required in code paths (mypy strict settings); prefer explicit return types.
- Imports: prefer absolute; relative imports are lint-banned.
- Naming: `snake_case` for functions/vars, `PascalCase` classes, `lower_snake_case.py` modules, package dirs lowercase.

## Testing Guidelines
- Framework: `pytest` with markers: `unit`, `integration`, `slow`, `requires_db`, `requires_openai`.
- Structure: `tests/<area>/test_*.py`; keep tests close to the feature area.
- Run examples: `uv run pytest -m "unit" -q`, `uv run pytest -m "integration" -v`.
- Coverage is preconfigured (`--cov=src`); open report via `make coverage-html`.

## Commit & Pull Request Guidelines
- Use Conventional Commits: e.g., `feat: …`, `fix: …`, `docs: …`, `refactor: …`, `ci: …`, `chore: …`.
- Scope when helpful: `fix(scanner): handle rate limit`.
- PRs: follow `.github/pull_request_template.md` — clear description, linked issue (`Fixes #123`), tests, and any screenshots where relevant.
- Keep changes focused and ensure `make qa` passes locally.

## Security & Configuration Tips
- Never commit secrets. Use env vars (e.g., `OPENAI_API_KEY`) or `.env` locally.
- Start from `config.example.yaml` → create `config.yaml` (untracked) for repos, DB, embeddings, and server settings.
- Optional containers: `docker-compose up -d` to bring up services; prefer local `uv` for dev.
