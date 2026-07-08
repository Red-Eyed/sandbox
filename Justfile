set dotenv-load := false

# List available recipes
default:
    @just --list

# Install all dependencies (including dev)
sync:
    uv sync --group dev

# Format source and tests
fmt:
    uv run ruff format torchquant/ tests/

# Check formatting without modifying files
fmt-check:
    uv run ruff format --check torchquant/ tests/

# Lint source and tests
lint:
    uv run ruff check torchquant/ tests/

# Lint and auto-fix what's safe
lint-fix:
    uv run ruff check --fix torchquant/ tests/

# Type-check with pyrefly
types:
    uv run pyrefly check torchquant/ tests/

# Run tests
test:
    uv run pytest tests/ -q

# Install git hooks
hooks:
    uvx prek install

# Run all git hooks against every file
hooks-run:
    uvx prek run --all-files

# Run fmt-check + lint + types + test (CI gate)
check: fmt-check lint types test

# Format, lint-fix, then run all checks
fix: fmt lint-fix
    just check

# Remove build artifacts and caches
clean:
    rm -rf dist/ .venv/ .ruff_cache/ .pytest_cache/
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type d -name "*.egg-info" -exec rm -rf {} +
