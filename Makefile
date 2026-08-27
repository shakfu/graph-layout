# graph-layout Makefile
# Build system for graph layout algorithms in Python

.PHONY: all help install install-dev clean test test-watch test-coverage \
		lint format check typecheck typecheck-legacy all dev sync build publish publish-test \
		docs docs-serve docs-strict docs-deploy docs-clean \
		wheel-check rebuild-cython qa showcase showcase-improvements demos \
		oracle-install bench-ogdf

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
	@echo "  make demos        - Generate all tests/demos/ visual demos to build/"
	@echo "  make showcase     - Generate showcase HTML with visual demos"
	@echo ""
	@echo "Code Quality:"
	@echo "  make format       - Format code with ruff"
	@echo "  make lint         - Lint code with ruff"
	@echo "  make check        - Run all checks (format check + lint)"
	@echo "  make typecheck    - Run mypy type checking (incl. the legacy ratchet)"
	@echo "  make typecheck-legacy - Check the modules pyproject.toml silences"
	@echo "  make qa           - Run all QA checks (check + typecheck + test)"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs         - Build the MkDocs site into site/"
	@echo "  make docs-serve   - Serve the docs with live reload"
	@echo "  make docs-strict  - Build with warnings as errors"
	@echo "  make docs-deploy  - Publish the site to the gh-pages branch"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        - Remove build artifacts and cache files"
	@echo "  make distclean    - Remove all generated files including .venv"
	@echo ""
	@echo "Development:"
	@echo "  make all          - Run full CI pipeline (sync + qa)"
	@echo "  make fix          - Auto-fix formatting and linting issues"
	@echo ""
	@echo "Publishing:"
	@echo "  make build          - Build sdist and wheel"
	@echo "  make rebuild-cython - Rebuild with fresh Cython compilation"
	@echo "  make publish-test   - Upload to TestPyPI"
	@echo "  make publish        - Upload to PyPI"

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

# The OGDF differential-testing oracle (ogdf-py) is pinned in the dev group and
# installed from PyPI by `uv sync`. Use this target only to test graph-layout
# against a LOCAL, unreleased build of the sibling ogdf-py checkout: it builds a
# wheel from OGDF_PY and installs it over the pinned release in the dev venv.
OGDF_PY ?= ../ogdf-py
oracle-install:
	@cd $(OGDF_PY) && uv build --wheel
	@uv pip install --reinstall $(OGDF_PY)/dist/ogdf_py-*.whl
	@uv run python -c "import ogdf; print('local ogdf oracle installed:', ogdf.__version__)"

# Run tests in watch mode
test-watch:
	@uv run pytest-watch -- -v

# Run tests with coverage
test-coverage:
	@uv run pytest --cov=$(SRC_DIR) --cov-report=term-missing

# Generate showcase HTML with visual demos
showcase:
	@uv run python tests/demos/showcase.py
	@if [ "$$(uname)" = "Darwin" ]; then open build/showcase.html; fi

# Generate the review-improvements showcase (bend-optimal GIOTTO, Cola constraints, cyclic fixes)
showcase-improvements:
	@uv run python tests/demos/improvements_showcase.py
	@if [ "$$(uname)" = "Darwin" ]; then open build/improvements_showcase.html; fi

# Generate every demo in tests/demos/ to build/ (and open them on macOS)
demos:
	@mkdir -p build
	@for demo in tests/demos/*.py; do \
		echo "Generating $$demo ..."; \
		uv run python "$$demo" >/dev/null; \
	done
	@echo "All demos written to build/"
	@if [ "$$(uname)" = "Darwin" ]; then open build/*showcase.html; fi

# Compare graph-layout vs OGDF (ogdf-py) on layout quality and speed.
# Needs the ogdf-py oracle (dev group / `make oracle-install`); prints a message
# and exits cleanly if it is absent. Pass args via ARGS, e.g. ARGS="--all".
bench-ogdf:
	@uv run python tests/benchmarks/compare_ogdf.py $(ARGS)

# Documentation. MkDocs lives in the `docs` dependency group, so `uv run --group
# docs` installs it on demand rather than pulling it into the test environment.
# Figures are rendered by executing the ```graph-layout blocks in docs/ against
# the installed library (scripts/mkdocs_hooks.py), so a stale example fails here.
DOCS := uv run --group docs mkdocs

docs:
	@$(DOCS) build

docs-serve:
	@$(DOCS) serve

docs-strict:
	@$(DOCS) build --strict

# Builds and force-pushes the site to the gh-pages branch. This is the only way
# the site is published: docs.yml is gated on workflow_dispatch and carries no
# publish job. Set the repository's Pages source to the gh-pages branch. Strict,
# so what gets published is held to the same warnings-as-errors bar as a local
# `make docs-strict`.
docs-deploy:
	@$(DOCS) gh-deploy --strict --force

docs-clean:
	@rm -rf site/

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
typecheck: typecheck-legacy
	@uv run mypy --strict $(SRC_DIR)

# Ratchet on the cola modules that pyproject.toml silences with
# `ignore_errors = true`. `make typecheck` reports success over those files
# without checking them, and the suppressed count grew unnoticed from a
# documented 146 to 158. This re-checks them under the project's own settings
# (see mypy-legacy.ini) and fails if the count climbs above the baseline.
#
# Lower this number when you fix something; never raise it. At zero, delete
# mypy-legacy.ini and the ignore_errors block in pyproject.toml.
MYPY_LEGACY_BASELINE := 158
typecheck-legacy:
	@count=$$(uv run mypy --config-file=mypy-legacy.ini $(SRC_DIR) 2>&1 \
		| grep -cE '^src/.*error:' || true); \
	if [ "$$count" -gt "$(MYPY_LEGACY_BASELINE)" ]; then \
		echo "FAIL: suppressed type errors rose to $$count (baseline $(MYPY_LEGACY_BASELINE))."; \
		echo "      Fix them, or see mypy-legacy.ini for what this checks."; \
		uv run mypy --config-file=mypy-legacy.ini $(SRC_DIR) 2>&1 | grep -E '^src/.*error:' || true; \
		exit 1; \
	elif [ "$$count" -lt "$(MYPY_LEGACY_BASELINE)" ]; then \
		echo "Suppressed type errors down to $$count (baseline $(MYPY_LEGACY_BASELINE))."; \
		echo "Lower MYPY_LEGACY_BASELINE in the Makefile to lock the improvement in."; \
	else \
		echo "Suppressed type errors: $$count (baseline $(MYPY_LEGACY_BASELINE))."; \
	fi

# Fix formatting and linting issues automatically
fix:
	@uv run ruff format $(ALL_DIRS)
	@uv run ruff check --fix $(ALL_DIRS)

# Run all QA checks
qa: test lint typecheck format

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
	@rm -rf site/
	@rm -rf dist/
	@rm -rf build/
	@rm -f .coverage

# Deep clean including virtualenv
distclean: clean
	@rm -rf venv/
	@rm -rf .venv/

# Full CI pipeline
all: sync qa

# Quick development check before commit
pre-commit: fix qa

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

# Rebuild with fresh Cython compilation (removes old generated .c and .so files)
rebuild-cython:
	@echo "Removing generated Cython files (.c and .so)..."
	@rm -f $(SRC_DIR)/_speedups.c
	@rm -f $(SRC_DIR)/_speedups*.so
	@echo "Cleaning build artifacts..."
	@rm -rf build/
	@echo "Installing build dependencies..."
	@uv pip install scikit-build-core cython
	@echo "Rebuilding Cython extension in place..."
	@uv pip install --no-build-isolation -e .
	@echo "Done. Verifying Cython module..."
	@uv run python -c "from graph_layout import _speedups; print('Cython module loaded:', _speedups.__file__); print('FA2 functions:', hasattr(_speedups, '_compute_fa2_repulsive_forces'))"

# Upload to TestPyPI
publish-test:
	@uv run twine upload --repository testpypi dist/*

# Upload to PyPI
publish:
	@uv run twine upload dist/*
