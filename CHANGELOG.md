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

## [unreleased]

## [0.1.7]

### Fixed

- **Type annotations** for clean `make typecheck`:
  - Added `cast()` for Cython function returns in `planarization.py` (segments_intersect, find_edge_crossings)
  - Added `cast()` for link attribute access in `preprocessing.py` (_default_get_source, _default_get_target)
  - Added generic type parameters to `yifan_hu.py` (dict -> dict[str, Any] for coarsening hierarchy)
  - Extended mypy `ignore_errors` for legacy cola modules (linklengths, handledisconnected, geom, rectangle, powergraph, layout3d, adapter)
  - Added ruff N806 exception for `orthogonal/*.py` to allow mathematical variable names in ILP formulation

### Added

- **Port Constraints for Kandinsky** (`orthogonal/kandinsky.py`):
  - User-specified edge exit/entry sides via `source_side` and `target_side` link attributes
  - Accepts `Side` enum values or strings ('north', 'south', 'east', 'west')
  - Partial constraints supported (constrain one side, heuristic for other)
  - New property: `port_constraints` to access parsed constraints

- **ILP-based Optimal Compaction** (`orthogonal/compaction_ilp.py`):
  - Linear programming formulation to minimize layout area optimally
  - Uses `scipy.optimize.milp` when scipy is available
  - Graceful fallback to greedy compaction if scipy unavailable
  - New KandinskyLayout parameter: `compaction_method` ("auto", "greedy", "ilp")
  - New exports: `ILPCompactionResult`, `compact_layout_ilp`, `is_scipy_available`
  - Optional `[ilp]` dependency: `pip install graph-layout[ilp]`

- **GIOTTO Algorithm** (`orthogonal/giotto.py`):
  - Bend-optimal orthogonal layout for degree-4 planar graphs
  - Based on Tamassia's algorithm for minimum-bend orthogonal drawings
  - Validates degree <= 4 and planarity (Euler's formula + K5 check)
  - `strict` mode (raise ValueError) or fallback mode (Kandinsky-like)
  - Properties: `is_valid_input`, `total_bends`, `orthogonal_rep`
  - Exported from package root: `from graph_layout import GIOTTOLayout`

- **Test suite additions** (`tests/test_kandinsky.py`):
  - 7 tests for port constraints (source, target, both, partial, string values)
  - 8 tests for ILP compaction (valid layout, separation, empty, single, method property)
  - 14 tests for GIOTTO (degree-4, rejects degree-5, rejects non-planar, strict mode, fallback)

- **Showcase updates** (`scripts/showcase.py`):
  - Port Constraints Demo graph with user-specified edge directions
  - 3x3 Grid and Ladder Graph examples for GIOTTO algorithm
  - Kandinsky (ILP) layout variant
  - Updated legend with new features section

- **Graph showcase script** (`scripts/showcase.py`):
  - Creates a html 'showcase' page of svg graphs before and after the application of the various layout algorithms in the package.

- **Random layout algorithm** (`basic/random.py`):
  - Places nodes at uniformly random positions within canvas bounds
  - Configurable margin parameter to keep nodes away from edges
  - Respects fixed nodes (preserves their positions)
  - Reproducible layouts with random_seed parameter
  - Useful as baseline for comparing layout quality metrics
  - Useful as starting point for iterative algorithms (force-directed, etc.)
  - O(n) complexity

- **Test suite for Random layout** (`tests/test_random_layout.py`):
  - 21 tests covering basic functionality, configuration, fixed nodes, events, reproducibility

- **Bipartite layout algorithm** (`bipartite/bipartite.py`):
  - Places nodes in two parallel rows/columns for bipartite graphs
  - Automatic bipartite detection using BFS coloring
  - User-specified sets support (top_set, bottom_set parameters)
  - Edge crossing minimization using barycenter heuristic
  - Horizontal (top/bottom) or vertical (left/right) orientation
  - Configurable layer_separation and node_separation
  - Utility functions: `is_bipartite()`, `count_crossings()`
  - Common use cases: user-item networks, author-paper, gene-disease
  - O(n + m) complexity for layout, O(iterations × n) for crossing minimization

- **Test suite for Bipartite layout** (`tests/test_bipartite.py`):
  - 29 tests covering detection, user sets, crossing minimization, events

- **Kandinsky orthogonal layout algorithm** (`orthogonal/kandinsky.py`):
  - Produces orthogonal drawings where edges use only horizontal/vertical segments
  - Supports vertices of arbitrary degree (unlike simpler orthogonal models)
  - Three-phase approach: layer assignment, node placement, edge routing
  - Configurable: node_width, node_height, node_separation, edge_separation, layer_separation
  - Outputs OrthogonalEdge objects with port and bend information
  - Ideal for UML diagrams, ER diagrams, flowcharts, circuit schematics

- **Orthogonal layout types** (`orthogonal/types.py`):
  - `Side` enum: NORTH, SOUTH, EAST, WEST
  - `Port`: Connection point on node side
  - `NodeBox`: Node as rectangle with edge access methods
  - `OrthogonalEdge`: Edge with source/target ports and bend points
  - `RoutingGrid`: Grid for edge routing

- **Test suite for Kandinsky** (`tests/test_kandinsky.py`):
  - 65 tests covering basic functionality, configuration, layering, edge routing, events
  - Tests for planarization, orthogonalization, and compaction phases

- **Kandinsky Planarization** (`orthogonal/planarization.py`):
  - `segments_intersect()`: Detect intersection point of two line segments
  - `find_edge_crossings()`: Find all edge crossings in a positioned graph
  - `planarize_graph()`: Insert crossing vertices at edge intersections
  - `CrossingVertex`: Data class for dummy vertices at crossings
  - `PlanarizedGraph`: Result structure with augmented edges and crossing info
  - `is_planar_quick_check()`: Quick Euler formula check for planarity
  - KandinskyLayout now handles non-planar graphs automatically
  - New properties: `handle_crossings`, `crossing_vertices`, `num_crossings`

- **Kandinsky Orthogonalization** (`orthogonal/orthogonalization.py`):
  - Bend minimization using min-cost flow formulation (Tamassia's algorithm)
  - `AngleType` enum: 90°, 180°, 270°, 0° angle types
  - `OrthogonalRepresentation`: Stores vertex-face angles and edge bends
  - `Face`: Represents faces in planar embedding with vertex/edge lists
  - `FlowNetwork`: Min-cost flow network with supplies, demands, and arcs
  - `compute_faces()`: Compute faces from planar embedding with angular ordering
  - `build_flow_network()`: Build flow network for orthogonalization
  - `solve_min_cost_flow_simple()`: Successive shortest path algorithm
  - `flow_to_orthogonal_rep()`: Convert flow solution to orthogonal representation
  - `compute_orthogonal_representation()`: Main entry point for orthogonalization
  - New KandinskyLayout property: `optimize_bends` to enable bend minimization
  - New property: `orthogonal_rep` to access the computed representation

- **Kandinsky Compaction** (`orthogonal/compaction.py`):
  - Constraint-based layout compaction to minimize drawing area
  - `CompactionConstraint`: Separation constraints between elements
  - `CompactionSolver`: Iterative relaxation solver for constraint satisfaction
  - `CompactionResult`: Result with new positions and final dimensions
  - `compact_horizontal()`: Horizontal compaction pass
  - `compact_vertical()`: Vertical compaction pass
  - `compact_layout()`: Full two-pass compaction (horizontal then vertical)
  - New KandinskyLayout property: `compact` to enable area minimization

- **Cython-optimized Kandinsky functions** in `_speedups.pyx`:
  - `_segments_intersect()`: Fast line segment intersection detection
  - `_find_edge_crossings()`: O(m²) edge crossing detection with Cython speedup
  - Automatic fallback to pure Python when Cython unavailable

- **Yifan Hu Multilevel layout algorithm** (`force/yifan_hu.py`):
  - Based on "Efficient and High Quality Force-Directed Graph Drawing" by Yifan Hu (2005)
  - Spring-electrical force model: repulsion C·K²/d, attraction d²/K
  - Multilevel coarsening using edge collapsing with maximal matching
  - Adaptive step length control (increases after 5 progress iterations)
  - Barnes-Hut O(n log n) approximation enabled by default for graphs >50 nodes
  - Configurable parameters: optimal_distance, relative_strength, step_ratio, convergence_tolerance
  - Multilevel parameters: coarsening_threshold (ρ=0.75), min_coarsest_size, level_iterations
  - Cython-accelerated force calculations with pure Python fallback
  - Ideal for medium-large graphs (1K-100K nodes)

- **Test suite for Yifan Hu** (`tests/test_yifan_hu.py`):
  - 21 tests covering basic functionality, configuration, fixed nodes, events, reproducibility
  - Algorithm-specific tests: multilevel coarsening, adaptive step, Barnes-Hut, cluster separation

- **ForceAtlas2 layout algorithm** (`force/force_atlas2.py`):
  - Based on the Gephi ForceAtlas2 paper by Jacomy et al. (2014)
  - Degree-weighted repulsion: hubs repel more strongly
  - Adaptive per-node speed based on swing/traction (no temperature cooling)
  - LinLog mode for tighter community clusters
  - Strong gravity mode to prevent component drift
  - Edge weight influence control
  - Overlap prevention option
  - Barnes-Hut O(n log n) approximation enabled by default for graphs >50 nodes

- **Cython-optimized ForceAtlas2 functions** in `_speedups.pyx`:
  - `_compute_fa2_repulsive_forces()` - O(n^2) degree-weighted repulsion
  - `_compute_fa2_repulsive_forces_overlap()` - with overlap prevention
  - `_compute_fa2_repulsive_forces_barnes_hut()` - O(n log n) approximation
  - `_compute_fa2_attractive_forces()` - linear/linlog attraction with edge weights
  - `_compute_fa2_gravity()` - degree-weighted gravity (normal and strong modes)
  - `_compute_fa2_swing_traction()` - adaptive speed calculation
  - `_apply_fa2_displacements()` - per-node speed application
  - Graceful fallback to pure Python when Cython not available

- **New Makefile target**: `make rebuild-cython` - removes old .c/.so files and rebuilds Cython extensions fresh

- **Test suite for ForceAtlas2** (`tests/test_force_atlas2.py`):
  - 21 tests covering basic functionality, configuration, fixed nodes, events, reproducibility
  - Algorithm-specific tests: LinLog mode, strong gravity, degree-weighted repulsion, Barnes-Hut

### Changed

- **Cython function naming convention**: All Cython speedup functions now use underscore prefix (e.g., `_compute_repulsive_forces`) to indicate internal implementation. Pure Python fallback methods have no underscore prefix (e.g., `compute_repulsive_naive`). This clarifies that Cython functions are internal optimizations while Python methods are the public fallback API.

- **CI wheel build time reduced from 2h 50m to 16m** by replacing QEMU emulation with native GitHub ARM runners (`ubuntu-24.04-arm`) for Linux aarch64 builds

### Performance

Cython speedups over pure Python (200 nodes, 50 iterations):

| Algorithm | Cython Speedup |
|-----------|----------------|
| Fruchterman-Reingold | **50-60x faster** |
| ForceAtlas2 | **15-20x faster** |
| Yifan Hu | **5-7x faster** |

ForceAtlas2 and Yifan Hu use Barnes-Hut O(n log n) by default for graphs >50 nodes. Yifan Hu is fastest for large graphs due to multilevel coarsening reducing the problem size before force calculation.

Kandinsky orthogonal layout performance (with Cython optimization):

| Graph Size | Time |
|------------|------|
| 100 nodes, 224 edges | 0.045s |
| 500 nodes, 1230 edges | 0.78s |
| 1000 nodes, 2495 edges | 3.63s |

Initial implementation was 42x slower (151s for 1000 nodes). Optimizations: (1) cached box bounds to avoid repeated property access, (2) removed O(n×m×bends) redundant loop, (3) Cython-optimized edge crossing detection.

Bipartite `count_crossings()` utility optimized from O(m²) to O(m log m):

| Edges | Before | After | Speedup |
|-------|--------|-------|---------|
| 1,000 | 0.27s | 0.0015s | **180x** |
| 5,000 | 6.6s | 0.0086s | **767x** |
| 10,000 | 26s | 0.017s | **1529x** |

Uses merge sort inversion counting instead of pairwise comparison. BipartiteLayout itself remains fast at 0.14s for 1000×1000 nodes (100k edges).

## [0.1.6] - Unified Cython Speedups and PyPI Publishing

### Added

- **Unified Cython `_speedups` module** (`src/graph_layout/_speedups.pyx`):
  - Consolidated all Cython code into a single extension module at package root
  - Priority queue (pairing heap) for Dijkstra's algorithm
  - Shortest paths calculator using Dijkstra's algorithm
  - Force-directed layout calculations:
    - `compute_repulsive_forces()` - O(n^2) pairwise repulsion
    - `compute_attractive_forces()` - O(m) edge attraction
    - `apply_displacements()` - O(n) position updates with bounds clamping
  - Barnes-Hut QuadTree implementation:
    - `FastQuadTree` class with O(n log n) force approximation
    - `compute_repulsive_forces_barnes_hut()` function
    - Configurable theta parameter for accuracy/speed tradeoff
    - Depth-limited insertion to prevent stack overflow from coincident points

- **Cython-accelerated Fruchterman-Reingold layout**:
  - Automatic Cython acceleration when `_speedups` module available
  - Falls back to pure Python implementation seamlessly
  - Both naive O(n^2) and Barnes-Hut O(n log n) modes accelerated

- **Graph preprocessing utilities** (`preprocessing.py`):
  - `detect_cycle()` / `has_cycle()` - Detect cycles in directed graphs
  - `remove_cycles()` - Make graph acyclic by reversing back edges
  - `topological_sort()` - Kahn's algorithm for DAG ordering
  - `connected_components()` / `is_connected()` - Find graph connectivity
  - `assign_layers_longest_path()` - Layer assignment for hierarchical layouts
  - `minimize_crossings_barycenter()` - Reduce edge crossings between layers
  - `count_crossings()` - Count edge crossings in layered layouts

- **Python 3.14 support** in wheel builds (cibuildwheel v3.3.0)

- **QEMU support** for aarch64 Linux wheel builds in CI

- **PyPI publishing configuration**:
  - `py.typed` marker for PEP 561 type checking support
  - `MANIFEST.in` for source distribution
  - Trusted Publishing workflow for GitHub Actions
  - Wheel collection job in CI workflow

- **Documentation**:
  - `docs/algorithms-guide.md` - Comprehensive guide to all layout algorithms with images, parameters, and decision guide
  - `docs/preprocessing-guide.md` - Guide to graph preprocessing utilities with examples and complete pipeline

### Changed

- Moved Cython extensions from `cola/` subdirectory to package root for use by all algorithms
- Updated `shortestpaths.py` to import from unified `_speedups` module
- License clarified as MIT (SPDX format in pyproject.toml)
- Excluded `.c`, `.pyx`, `.pxd` files from wheel distributions
- Build system simplified: `uv build` replaces `python -m build` (removed `build` package from dev dependencies)

### Removed

- **`[fast]` optional dependency**: scipy fallback removed since Cython extensions are pre-built in PyPI wheels and faster than scipy
- **scipy fallback code** in `shortestpaths.py`: Simplified from 257 to 157 lines, now just Cython > pure Python

### Fixed

- **Segfault in Barnes-Hut implementation**: Added depth limit (50 levels) to QuadTree insertion to prevent stack overflow when nodes have coincident or near-coincident positions
- License file now correctly contains MIT license text (was GPL v3)

### Performance

With Cython `_speedups` enabled:

| Algorithm | Graph Size | Time |
|-----------|-----------|------|
| Fruchterman-Reingold | 500 nodes, 1000 edges | 0.046s |
| FR + Barnes-Hut | 500 nodes, 1000 edges | 0.089s |
| Cola (constraint-based) | 500 nodes, 1000 edges | 1.167s |
| Kamada-Kawai | 100 nodes, 200 edges | 0.674s |
| Spring | 100 nodes, 200 edges | 0.456s |
| Circular | 100 nodes | 0.001s |
| Spectral | 100 nodes | 0.011s |

Note: Barnes-Hut has higher overhead than naive O(n^2) at 500 nodes; becomes beneficial at ~2000+ nodes.

---

## [0.1.5] - Pythonic API

### Changed

- **BREAKING: New Pythonic API** - Complete API redesign from JavaScript-style fluent methods to Pythonic constructor parameters and properties.

  **Before (fluent API):**

  ```python
  layout = FruchtermanReingoldLayout()
  layout.nodes(nodes).links(links).size([500, 500])
  layout.start(iterations=100)
  result = layout.nodes()
  ```

  **After (Pythonic API):**

  ```python
  layout = FruchtermanReingoldLayout(
      nodes=nodes,
      links=links,
      size=(500, 500),
      iterations=100,
  )
  layout.run()
  result = layout.nodes  # Property, not method
  ```

- **Renamed `start()` to `run()`** for all layout classes (except internal Cola `Layout` class)
- **Properties replace getter/setter methods**: `layout.nodes`, `layout.links`, `layout.size`, etc.
- **Constructor parameters for configuration**: All algorithm-specific settings configurable via constructor
- **Event callbacks in constructor**: `on_start`, `on_tick`, `on_end` parameters

### Added

- **Type aliases** in `types.py`: `NodeLike`, `LinkLike`, `GroupLike`, `SizeType` for flexible input types
- **RadialTreeLayout** added to hierarchical layouts module

### Migration Guide

| Old API | New API |
|---------|---------|
| `layout.nodes(data)` | `layout = Layout(nodes=data)` or `layout.nodes = data` |
| `layout.nodes()` | `layout.nodes` |
| `layout.size([w, h])` | `layout = Layout(size=(w, h))` or `layout.size = (w, h)` |
| `layout.start()` | `layout.run()` |
| `layout.start(iterations=N)` | `layout = Layout(iterations=N); layout.run()` |
| `layout.temperature(T)` | `layout = Layout(temperature=T)` or `layout.temperature = T` |
| `layout.barnes_hut(True, theta=0.5)` | `layout = Layout(use_barnes_hut=True, barnes_hut_theta=0.5)` |

### Internal

- **Cola `Layout` class unchanged**: The internal `cola/layout.py` retains the JavaScript-style fluent API for compatibility with the WebCola port. Use `ColaLayoutAdapter` for the Pythonic API.
- Updated `scripts/visualize.py` for new API
- Updated all test files (530 tests passing)
- Updated README.md with new API examples

---

## [0.1.4] - Validation, Metrics, and Performance

### Added

- **Input Validation Module** (`validation.py`):
  - `validate_canvas_size()` - Rejects zero/negative canvas dimensions
  - `validate_link_indices()` - Bounds-checks link source/target against node count
  - `validate_group_indices()` - Validates group leaf/subgroup references
  - Custom exceptions: `ValidationError`, `InvalidCanvasSizeError`, `InvalidLinkError`, `InvalidGroupError`
  - Integrated into `base.py` size() method and `types.py` Link constructor

- **Layout Quality Metrics Module** (`metrics.py`):
  - `edge_crossings(nodes, links)` - Count intersecting edges
  - `stress(nodes, links, ideal_edge_length)` - Measure distance deviation from ideal
  - `edge_length_variance(nodes, links)` - Variance of edge lengths
  - `edge_length_uniformity(nodes, links)` - Normalized uniformity score (0-1)
  - `angular_resolution(nodes, links)` - Minimum angle between edges at nodes
  - `layout_quality_summary(nodes, links)` - All metrics in one dict

- **Cola Layout Adapter** (`cola/adapter.py`):
  - `ColaLayoutAdapter` class wrapping Cola's `Layout` with `BaseLayout`-compatible interface
  - Enables polymorphic usage with other layout algorithms
  - Preserves access to Cola-specific features (constraints, overlap avoidance, groups)
  - Consistent event forwarding (start, tick, end)

- **Barnes-Hut Optimization** (`spatial/quadtree.py`):
  - `QuadTree` class for spatial partitioning
  - `Body` dataclass for node representation with mass
  - O(n log n) approximate force calculation vs O(n^2) naive
  - Configurable theta parameter for accuracy/speed tradeoff
  - `QuadTree.from_nodes()` factory method for easy integration

- **Barnes-Hut Integration in Force Layouts**:
  - `FruchtermanReingoldLayout.barnes_hut(enabled, theta)` - Enable/configure approximation
  - `SpringLayout.barnes_hut(enabled, theta)` - Enable/configure approximation
  - Automatically activates for graphs with >50 nodes when enabled
  - SpringLayout uses proper Coulomb force law (1/d^2) in Barnes-Hut mode

- **Algorithm Assumption Warnings** (hierarchical layouts):
  - `GraphStructureWarning` - Issued when Sugiyama layout receives cyclic graph (not a DAG)
  - `TreeStructureWarning` - Issued when tree layouts receive non-tree graphs
  - Warns on disconnected nodes unreachable from root
  - Helps users identify when graph structure doesn't match algorithm assumptions

- **New Tests**:
  - `tests/test_validation.py` - Input validation tests
  - `tests/test_metrics.py` - Layout quality metrics tests
  - `tests/test_cola_adapter.py` - Cola adapter interface tests
  - `tests/test_quadtree.py` - QuadTree and Barnes-Hut accuracy tests

### Changed

- `base.py`: Added validation in `size()` method, added `validate()` method for explicit validation
- `types.py`: Link constructor now validates source/target are not None
- Test count increased from 409 to 529
- **Documented magic numbers**:
  - `fruchterman_reingold.py`: Explained `_cooling_factor` and `_min_temperature` constants
  - `descent.py`: Documented `ZERO_DISTANCE`, added `MIN_DIST_SQ` class constant with explanation

### Fixed

- `cola/handledisconnected.py`: Fixed TypeError when node width/height is None (now falls back to node_size)
- `cola/descent.py`: Added missing `-> None` return type annotation on `Locks.__init__`

---

## [0.1.3] - Multi-Algorithm Layout Library

### Added

- **New layout algorithm families** expanding beyond Cola:
  - **Force-Directed**: `FruchtermanReingoldLayout`, `KamadaKawaiLayout`, `SpringLayout`
  - **Hierarchical**: `SugiyamaLayout`, `ReingoldTilfordLayout`
  - **Circular**: `CircularLayout`, `ShellLayout`
  - **Spectral**: `SpectralLayout`
- **Shared infrastructure** (`base.py`, `types.py`):
  - `BaseLayout` - Abstract base for all layout algorithms
  - `IterativeLayout` - Base for iterative/animated layouts (force-directed)
  - `StaticLayout` - Base for single-pass layouts (circular, hierarchical)
  - Common `Node`, `Link`, `Group`, `EventType` types
- **Visualization script** (`scripts/visualize.py`):
  - Generates images for all algorithms to `./build/`
  - Individual layout images and comparison images
- **Comprehensive test suite** for all new algorithms (409 tests total)

### Changed

- Reorganized package structure with algorithm families as subpackages
- Renamed package from `pycola` to `graph_layout`
- All layouts now use consistent fluent API pattern
- Updated README with documentation for all algorithms
- Updated pyproject.toml with new package structure

### Algorithm Details

| Algorithm | Module | Description |
|-----------|--------|-------------|
| `FruchtermanReingoldLayout` | `force/` | Classic force-directed with temperature cooling |
| `KamadaKawaiLayout` | `force/` | Stress minimization using graph-theoretic distances |
| `SpringLayout` | `force/` | Simple Hooke's law spring forces |
| `SugiyamaLayout` | `hierarchical/` | Layered DAG drawing with crossing minimization |
| `ReingoldTilfordLayout` | `hierarchical/` | Compact tree layout |
| `CircularLayout` | `circular/` | Nodes on a single circle |
| `ShellLayout` | `circular/` | Concentric circles by degree/grouping |
| `SpectralLayout` | `spectral/` | Laplacian eigenvector embedding |

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
