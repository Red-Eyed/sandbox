set dotenv-load := false

# List available recipes
default:
    @just --list

# Install all dependencies (including dev)
sync:
    uv sync --group dev

# Format source
fmt:
    uv run ruff format src/

# Check formatting without modifying files
fmt-check:
    uv run ruff format --check src/

# Lint source
lint:
    uv run ruff check src/

# Lint and auto-fix what's safe
lint-fix:
    uv run ruff check --fix src/

# Type-check with pyrefly
types:
    uv run pyrefly check src/

# Run fmt-check + lint + types (CI gate)
check: fmt-check lint types

# Format, lint-fix, then run all checks
fix: fmt lint-fix
    just check

# Remove build artifacts and caches
clean:
    rm -rf dist/ .ruff_cache/ .pytest_cache/ __pycache__
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type d -name "*.egg-info" -exec rm -rf {} +
