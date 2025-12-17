# graph-layout Makefile
# Build system for graph layout algorithms in Python

.PHONY: all help install install-dev clean test test-watch test-coverage \
		lint format check typecheck all dev sync build publish publish-test \
		wheel-check

# Source and test directories
SRC_DIR := src/graph_layout
TEST_DIR := tests
ALL_DIRS := $(SRC_DIR) $(TEST_DIR)

all: build

# Default target
help:
	@echo "graph-layout Build Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make sync         - Sync all dependencies from lockfile (recommended)"
	@echo "  make install      - Sync runtime dependencies only"
	@echo "  make dev          - Sync dev dependencies (alias for sync)"
	@echo ""
	@echo "Testing:"
	@echo "  make test         - Run all tests"
	@echo "  make test-watch   - Run tests in watch mode"
	@echo "  make test-coverage - Run tests with coverage report"
	@echo "  make test-html    - Run tests with HTML coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make format       - Format code with ruff"
	@echo "  make lint         - Lint code with ruff"
	@echo "  make check        - Run all checks (format check + lint)"
	@echo "  make typecheck    - Run mypy type checking"
	@echo "  make verify       - Run all verification (check + typecheck + test)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        - Remove build artifacts and cache files"
	@echo "  make distclean    - Remove all generated files including .venv"
	@echo ""
	@echo "Development:"
	@echo "  make all          - Run full CI pipeline (sync + verify)"
	@echo "  make fix          - Auto-fix formatting and linting issues"
	@echo ""
	@echo "Publishing:"
	@echo "  make build        - Build sdist and wheel"
	@echo "  make publish-test - Upload to TestPyPI"
	@echo "  make publish      - Upload to PyPI"

# Sync all dependencies from lockfile (runtime + dev)
sync:
	@uv sync

# Install runtime dependencies only
install:
	@uv sync --no-dev

# Install dev dependencies (alias for sync)
dev: sync

# Run tests
test:
	@uv run pytest

# Run tests in watch mode
test-watch:
	@uv run pytest-watch -- -v

# Run tests with coverage
test-coverage:
	@uv run pytest --cov=$(SRC_DIR) --cov-report=term-missing

# Run tests with HTML coverage report
test-html:
	@uv run pytest --cov=$(SRC_DIR) --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

# Format code with ruff
format:
	@uv run ruff format $(ALL_DIRS)

# Lint code with ruff
lint:
	@uv run ruff check --fix $(ALL_DIRS)

# Check formatting and linting without fixing
check:
	@echo "Checking code formatting..."
	@uv run ruff format --check $(ALL_DIRS)
	@echo "Running linter..."
	@uv run ruff check $(ALL_DIRS)

# Run mypy type checking
typecheck:
	@uv run mypy $(SRC_DIR)

# Fix formatting and linting issues automatically
fix:
	@uv run ruff format $(ALL_DIRS)
	@uv run ruff check --fix $(ALL_DIRS)

# Run all verification steps
verify: check typecheck test

wheel-check:
	@uv run twine check dist/*.whl

# Clean build artifacts and cache
clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name "*.pyd" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".*_cache" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf htmlcov/
	@rm -rf dist/
	@rm -rf build/
	@rm -f .coverage

# Deep clean including virtualenv
distclean: clean
	@rm -rf venv/
	@rm -rf .venv/

# Full CI pipeline
all: sync verify

# Quick development check before commit
pre-commit: fix verify

# Show Python and package versions
version:
	@echo "uv version:"
	@uv --version
	@echo ""
	@echo "Python version:"
	@uv run python --version
	@echo ""
	@echo "Installed packages:"
	@uv pip list | grep -E "(numpy|sortedcontainers|pytest|mypy|ruff)" || echo "No packages installed yet"

# Build sdist and wheel
build: clean
	@uv build
	@uv run twine check dist/*.whl

# Upload to TestPyPI
publish-test: build
	@uv run twine upload --repository testpypi dist/*

# Upload to PyPI
publish: build
	@uv run twine upload dist/*
