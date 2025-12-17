# CHANGELOG

All notable project-wide changes will be documented in this file. Note that each subproject has its own CHANGELOG.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and [Commons Changelog](https://common-changelog.org). This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Types of Changes

- Added: for new features.
- Changed: for changes in existing functionality.
- Deprecated: for soon-to-be removed features.
- Removed: for now removed features.
- Fixed: for any bug fixes.
- Security: in case of vulnerabilities.

---

## [0.1.2] - Cython Shortest Paths Optimization

### Added
- **Cython-compiled shortest paths (Dijkstra's algorithm)** - 5x additional speedup
- Optional scipy integration for even better performance (`pip install graph-layout[fast]`)
- Priority cascade implementation: Cython → scipy → pure Python
- Pre-built wheels for Linux, macOS (x86_64, arm64), and Windows
- GitHub Actions workflow for multi-platform wheel building with cibuildwheel

### Changed
- **MAJOR PERFORMANCE IMPROVEMENT**: Cython-compiled Dijkstra's algorithm
  - **5x faster** for large graphs on top of vectorization gains
  - **100x total speedup** compared to original implementation (v0.1.0)
  - Medium graphs (100 nodes): 4.1s → 0.05s (80x faster overall)
  - Large graphs (500 nodes): 115.8s → 1.1s (105x faster overall)
- Shortest paths now uses Cython extensions by default (no runtime dependencies)
- Build system changed from `uv_build` to `setuptools` for Cython support
- Added optional `[fast]` extra for scipy integration

### Performance (Combined: Vectorization + Cython)
- **Small graphs (20 nodes)**: ~0.02s (was ~1.7s) - **85x faster**
- **Medium graphs (100 nodes)**: ~0.05s (was ~4.1s) - **82x faster**
- **Large graphs (500 nodes)**: ~1.1s (was ~115.8s) - **105x faster**

### Installation
- **With Cython extensions** (recommended): `pip install graph-layout` or `uv pip install graph-layout`
- **With scipy** (fastest): `pip install graph-layout[fast]`
- **From source** (for development): `pip install -e .` (requires C compiler)

### Testing
- All 312 tests pass with Cython implementation
- Fallback to pure Python when Cython extensions unavailable
- Numerical correctness maintained across all implementations

## [0.1.1] - Performance Optimization Release

### Added
- Comprehensive performance profiling system (`scripts/profile_layout.py`)
- Performance analysis documentation (`docs/OPTIMIZATION_ANALYSIS.md`)
- Performance comparison documentation (`docs/PERFORMANCE_COMPARISON.md`)
- Performance benchmarks in README.md and CLAUDE.md

### Changed
- **MAJOR PERFORMANCE IMPROVEMENT**: Vectorized `compute_derivatives()` in `descent.py` using NumPy broadcasting
  - **20-65x overall speedup** depending on graph size
  - **110-170x faster** for gradient descent computation specifically
  - Medium graphs (100 nodes): 4.1s → 0.2s (20x faster)
  - Large graphs (500 nodes): 115.8s → 5.6s (21x faster)
- Replaced nested Python loops with NumPy array operations in gradient descent
- All edge cases properly handled (diagonal elements, division by zero, P-stress filtering)
- Updated Makefile to use `uv` for dependency management
- Added `from __future__ import annotations` to all modules for forward type references
- Updated CLAUDE.md with current performance metrics and optimization roadmap

### Fixed
- Forward reference type hints in `vpsc.py`, `powergraph.py`, `descent.py`, and `layout.py`
- Import paths in profiling scripts
- Source directory path corrections in Makefile (`src/pycola` vs `pycola`)

### Performance
- **Small graphs (20 nodes)**: ~0.03s (was ~1.7s) - **65x faster**
- **Medium graphs (100 nodes)**: ~0.2s (was ~4.1s) - **20x faster**
- **Large graphs (500 nodes)**: ~5.6s (was ~115.8s) - **21x faster**
- New bottleneck identified: Shortest path calculation (Dijkstra) - 75-82% of runtime
- Next optimization target: Replace Dijkstra with scipy for potential 3-5x additional improvement

### Testing
- All 312 tests pass with vectorized implementation
- Numerical correctness maintained (floating-point accuracy within tolerance)
- Test suite completes in 0.41s

## [0.1.0] - Initial Release

### Added
- Complete Python port of WebCola graph layout library
- 2D force-directed layout with gradient descent
- 3D layout support
- VPSC (Variable Placement with Separation Constraints) solver
- Constraint-based layout (separation, alignment)
- Overlap avoidance with rectangle projection
- Hierarchical group layouts with containment
- Power graph automatic clustering
- Grid router for orthogonal edge routing
- Event system (start/tick/end events)
- Fluent API with method chaining
- Disconnected component handling
- Link length calculators (symmetric difference, Jaccard)
- Flow layouts (directed graph layouts)
- Interactive drag support
- Comprehensive test suite (312 tests, 100% pass rate)
- Priority queue (pairing heap) implementation
- Red-black tree implementation
- Shortest paths (Dijkstra) implementation
- Computational geometry utilities
- Batch layout operations
- Complete documentation with examples
- CLAUDE.md with architecture overview
- TypeScript to Python translation guide

### Dependencies
- numpy>=1.20.0
- sortedcontainers>=2.4.0

### Development
- pytest>=8.3.5
- pytest-cov>=5.0.0
- mypy>=1.14.1
- ruff>=0.13.3
- Uses `uv` for dependency management
- Python 3.9+ required
- MIT License
