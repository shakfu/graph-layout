# TODO

ref: Open Graph Drawing Framework <https://github.com/ogdf/ogdf>

## High Priority

### Orthogonal Layout Improvements

Remaining items for closing the gap with [OGDF](https://github.com/ogdf/ogdf)'s orthogonal drawing capabilities.

#### ~~Planarity: Kuratowski Subgraph Extraction~~ (Done)

~~Extract Kuratowski subgraph (K5 or K3,3 minor) as proof of non-planarity.~~ Implemented: `check_planarity()` now returns `kuratowski_edges` and `kuratowski_type` ("K5" or "K3,3") for non-planar graphs (up to 50 vertices). Uses edge-deletion to find minimal non-planar subgraph, then identifies the K5/K3,3 subdivision structure.

#### ~~EmbedderOptimalFlexDraw~~ (Done)

~~Optimal flexibility embedding for orthogonal drawing.~~ Implemented as `OptimalFlexEmbedder`: LP-based bend minimization over candidate outer faces using `scipy.optimize.linprog`. Falls back to `MaxFaceEmbedder` when scipy is unavailable.

#### Advanced Edge Routing (Partial)

Visibility graph routing is implemented. Segment nudging needs rework.

- [x] Visibility graph routing: builds orthogonal visibility graph from obstacle corners, finds shortest path via BFS
- [x] Segment nudging: `nudge_overlapping_segments()` post-processing separates coincident parallel edge segments
- [ ] **Obstacle-aware segment nudging**: Current nudging blindly offsets segments without checking node-box collisions, causing edges to route through nodes. Fix requires: (1) checking nudged positions against node boxes, (2) re-routing segments that collide with obstacles after nudging, or (3) integrating nudging into the routing phase so obstacle avoidance is preserved. See `edge_routing.py:nudge_overlapping_segments()`.

#### Topology-Shape-Metrics bend-optimal drawing (Partial)

The orthogonalization's bend-minimal representation now drives the GIOTTO drawing (previously it was computed and discarded). `orthogonal/metrics.py` implements the shape stage (compass direction per segment) and coordinate assignment, wired into `GIOTTOLayout(bend_optimal=True)` with a `used_bend_optimal` signal and safe fallback to the heuristic router.

- [x] Flow model emits valid orthogonal representations (every face turns +/-4) for biconnected max-degree-4 planar graphs
- [x] Shape + integer-coordinate assignment (two-tier compact/spread); conflict detection so nothing broken is emitted
- [x] `GIOTTOLayout(bend_optimal=...)` opt-in, `used_bend_optimal` signal
- [ ] **Turn-regularization for compaction**: ~11% of in-scope graphs still fall back (non-planar coordinate assignment / crossings). Add kitty-corner-based separation constraints so every face is rectangular, taking coverage to ~100% and enabling `bend_optimal` by default. Full design, reference algorithm (Bridgeman et al. 2000), and verification oracle in [`docs/rectangularization-plan.md`](docs/rectangularization-plan.md).
- [ ] **Kandinsky degree > 4** (H5): 0-degree-angle flow model for vertices of degree > 4 (out of the current TSM domain)
- [ ] **Non-biconnected graphs** (H6a): per-corner angles for bridges / cut vertices

---

## Known Correctness Issues

Verified open defects in shipped, documented capabilities (distinct from the new-algorithm wish list below). Each was confirmed by direct code inspection. Ordered by severity. IDs (C_/H_) are retained from the algorithm review for traceability.

### High severity

- [x] **C1 - Cola group containment** (Done): `Projection` now generates the full non-overlap + containment constraint set over the whole group hierarchy via `_generate_group_constraints` (faithful port of WebCola's recursive `generateGroupConstraints`, incl. the border-variable and constraint-redirection tail); the group `min_var`/`max_var` border variables enter the VPSC solve, so group boxes contain their members and sibling groups stay disjoint. Also fixed the latent `AttributeError` on the missing optional `stiffness` attribute that crashed the avoid_overlaps + groups path. Regression tests in `tests/test_layout.py` (`TestLayoutWithGroups`) assert group-box disjointness under inter-group attraction and confirm the recursive nested-group path. This completes the WebCola projection port (node-variable projection, separation/alignment/non-overlap were completed in 0.2.0).
- [x] **H7 - Planarization is geometric, not topological** (Done): `planarize_graph` (`orthogonal/planarization.py`) now performs topological planarization -- a greedy maximal planar subgraph is embedded (via `check_planarity`), then the remaining edges are reinserted one at a time along a minimum-crossing path through the embedding's faces (dual-graph BFS), each crossing becoming a degree-four dummy vertex. Crossings depend only on topology, not on any drawing: a genuinely planar graph gains no crossings regardless of node positions (positions are used only to give dummies an approximate coordinate), and the augmented graph is always planar. Recovers the known crossing numbers (K5=1, K3,3=1, Petersen=2). The old geometric `find_edge_crossings`/`segments_intersect` helpers are retained (used elsewhere). Regression tests in `tests/test_planarization.py` (planar->0 crossings, augmented-always-planar over 300 random graphs, degree-4 dummies, segment-path connectivity); two Kandinsky tests that asserted the old position-dependent behavior were corrected.
- [x] **H8 - Greedy compaction does not compact** (Done): `CompactionSolver.solve` now performs longest-path compaction -- each element is pulled to its leftmost/topmost feasible position (max over incoming constraints, else the base), closing interior slack instead of only pushing right. `compact_horizontal`/`compact_vertical` now constrain every overlapping pair (not just consecutive ones) so the tighter packing stays overlap-free. This is the default orthogonal path when scipy is unavailable. Direct tests in `tests/test_compaction.py` (width reduced, no collapsed overlaps).
- [x] **Sugiyama coordinate assignment is index-based** (Done): replaced the independent per-layer centering with Brandes-Köpf horizontal coordinate assignment (`hierarchical/_brandes_koepf.py`) -- four vertical-alignment runs (align up/down x median left/right) with type-1 conflict marking, packed left/right respectively and balanced per vertex (left runs aligned by min, right runs by max, onto the smallest-width run). Each vertex aligns with the median of its neighbours, so edges (especially long-edge dummy chains) are straightened, and the layout is symmetric (a parent is centred over its children). The paper's O(n) shift-class compaction is replaced with a longest-path block compaction that computes the identical tightest packing but guarantees the within-layer ordering / min-separation invariant by construction (verified 0 pre-clamp violations over thousands of random graphs; the final clamp is a defensive safety net that does not fire in practice). Regression tests: `tests/test_brandes_koepf.py` (invariant, straightening, symmetry) and a Sugiyama-level assertion that a long edge's bends are vertically collinear.
- [x] **`preprocessing.count_crossings` wrong for long edges** (Done): replaced the `(l1, l2)` layer-pair bucketing with a geometric proper-crossing test over edge segments in `(position, layer)` space, so a long edge is now counted against the shorter edges in every layer gap it passes through (edges sharing a node are excluded). Regression test covers a long-edge crossing that the old bucketing missed.

### Medium severity

- [x] **FA2 regular vs strong gravity swapped** (Done): regular gravity is now distance-independent and strong gravity distance-scaled, matching Gephi/Jacomy et al. (verified numerically: regular net pull constant across distances, strong scales linearly). Fixed in both `force/force_atlas2.py` and the Cython kernel `_speedups.pyx` (rebuilt). Note: this exposed a pre-existing flaky test (`test_linlog_mode_tighter_clusters`) that asserted LinLog produces *smaller* intra-cluster diameters -- not actually a LinLog property (its weaker log attraction spreads small clusters; the assertion only held because the buggy distance-scaled gravity compressed the layout). Replaced it with a robust, spec-derived assertion (LinLog's weaker attraction => longer mean edge length).
- [x] **FA2 global-speed update lacks max-rise / jitter damping** (Done): the global speed now tracks `tolerance * traction / swing` but rises by at most 50% per iteration (falls freely), matching Jacomy et al.; jumping straight to the target caused jitter (`force/force_atlas2.py`). Regression asserts the per-tick speed rise is damped.
- [x] **Yifan Hu ignores fixed nodes during optimization** (Done): `_layout_level` now takes a `fixed_mask` and skips displacing pinned vertices (they still exert forces on others); the finest-level refinement pins fixed nodes at their true positions so they no longer drift and perturb the free nodes (`force/yifan_hu.py`). Coarser levels are unchanged (super-vertices don't map 1:1). Unit regression asserts a masked vertex stays put while a free one moves.
- [x] **Pure-Python quadtree infinite recursion on coincident points** (Done): added a `MAX_DEPTH = 50` cap to `_insert_into`/`_insert_into_child` (`spatial/quadtree.py`) mirroring the Cython kernel -- at max depth bodies merge in place (weighted-average position, summed mass) instead of recursing forever. Regression test inserts 200 coincident bodies.
- [x] **Cola grid-snap modulo sign mismatch** (Done): `cola/descent.py` now uses `math.fmod` (sign of the dividend) instead of Python `%` (sign of the divisor), matching WebCola's JS `%`, so negative coordinates snap toward the correct grid line. Regression asserts a node at a negative coordinate receives a snap force (previously none).
- [x] **Cola groups initial-layout KeyError** (Done): the grouped unconstrained warm-up now reads positions from `flat_layout.nodes()` (the laid-out Node objects) instead of the never-populated input dicts (`cola/layout.py`). Regression test runs a grouped layout with `initial_unconstrained_iterations > 0`.
- [x] **Spectral disconnected graphs pick zero-eigenvalue eigenvectors** (Done): `_compute` now skips *all* near-zero eigenvalues (start at the first strictly-positive one) rather than only index 0, so disconnected components no longer collapse to a point. Regression test asserts within-component spread on two disjoint triangles.
- [x] **Spectral normalized Laplacian missing `D^-1/2` back-transform** (Done): when `normalized`, the selected eigenvectors are now scaled by `D^-1/2` (Koren degree-weighted layout); `_compute_laplacian` returns the raw degree vector for this. Both fixes landed together in `spectral/spectral.py`.
- [x] **Edge-crossing metric misses collinear / T-junction cases** (Done): `metrics._segments_intersect` now uses the canonical orientation + on-segment test (CLRS), counting collinear overlaps and endpoint-on-interior (T-junction) touches in addition to proper crossings. Regression tests cover collinear-overlap, T-junction, and collinear-disjoint (non-crossing).
- [x] **`compact_flow_1d` can be looser than longest-path** (Done): the flow result now falls back to the longest-path positions whenever it would widen the span (longest-path is the provably minimum 1D span), so it is never looser; the misleading "tighter layouts" docstring was corrected (`orthogonal/compaction_flow.py`). Regression asserts the flow span never exceeds the longest-path span.
- [x] **Shell layout IndexError on out-of-range link indices** (Done): `_compute_degrees` (`circular/shell.py`) now guards `0 <= src < n and 0 <= tgt < n` before incrementing, mirroring `base._build_adjacency`. Regression test runs auto-shells with an out-of-range link.

### Low severity

- [x] **Kandinsky mode setter rejects its own documented modes** (Done): `compaction_method` setter (`orthogonal/kandinsky.py`) now accepts all five dispatched methods (`auto`/`greedy`/`ilp`/`flow`/`longest_path`), not just three.
- [x] **Spring docstring/behavior mismatch** (Done): `force/spring.py` docstring now describes the inverse-square Coulomb repulsion it actually implements.
- [x] **RandomLayout seeds the global `random` module** (Done): `basic/random.py` now uses a local `random.Random(seed)` instance, so seeding no longer mutates process-wide RNG state. Regression asserts the global RNG is untouched.
- [x] **Recursion-depth limits** (Done): `preprocessing.detect_cycle` is now an iterative DFS (explicit stack), so it handles arbitrarily deep graphs. The recursive tree walks in `reingold_tilford.py` and `radial_tree.py` run under a `raised_recursion_limit` context manager (`base.py`) that lifts the interpreter limit to the tree depth (ceilinged to avoid a native-stack overflow). Regression tests lay out 4000-5000 node chains without `RecursionError`.
- [x] **Inconsistent `node.fixed` semantics** (Done): Circular, Shell, and Bipartite layouts now skip repositioning fixed nodes (guard mirroring `RandomLayout`), so pinned nodes keep their positions. Regression tests per layout.
- [x] **Unpositioned nodes left unvalidated** (Done): nodes omitted from user-supplied bipartite sets go to the bottom row (`bipartite.py`) and nodes omitted from explicit shells go into an extra outer shell (`circular/shell.py`), so every node is positioned. Regression tests per layout.
- [x] **Radial tree wedge allocation by node count, not leaf count** (Done): angular wedges are now sized by subtree *leaf* count, so a deep narrow subtree no longer hogs the same span as a bushy one (`hierarchical/radial_tree.py`). Regression asserts leaf-count-driven weights.
- [x] **Cola `_alpha is None` fragility in `tick()`** (Done): `tick()` guarded against `_alpha` being None (ticking before `start()` raised `TypeError`); it now returns converged. Regression added. (`offset_dir` is retained -- it is exercised by a test and represents the still-omitted coincident-node perturbation feature, not removable dead code; the `cola/shortestpaths` docstrings are accurate.)
- [x] **Metrics reporting nits** (Done): `metrics._compute_ideal_distances` now uses weighted shortest paths honoring each link's length (Dijkstra) instead of an unweighted hop count; `angular_resolution` excludes self-loops and parallel edges so they no longer register spurious 0-degree angles. Regression tests for both.
- [x] **Dead / unused code** (Done): removed the no-op `directed` parameter from `preprocessing.connected_components` (it always computed undirected / weakly-connected components). (`nudge_overlapping_segments` is tracked as the Advanced Edge Routing feature above; the finite bend capacity is intrinsic to the min-cost-flow bend model.)

---

## Medium Priority

### Stress Majorization (SMACOF)

Iterative stress minimization that converges faster than Kamada-Kawai with similar quality. Different from KK -- uses majorization, not gradient descent, and scales better.

- Better optimization than gradient descent used in KK
- Handles weighted graphs well
- Reference: [Gansner et al. 2004](https://graphviz.org/Documentation/GKN04.pdf)

### Pivot MDS

Fast MDS approximation using pivot nodes. `O(k*n) where k << n`. Very fast initial layout for large graphs, useful when spectral layout is too slow.

### Planar Straight-Line Drawing Algorithms (New Category)

Entire category missing from graph-layout. Foundational in graph drawing theory.

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **SchnyderLayout** | Realizer-based drawing on (n-2) x (n-2) grid -- most compact | Schnyder 1990 |
| **FPPLayout** | de Fraysseix-Pach-Pollack convex grid drawing on (2n-4) x (n-2) grid | de Fraysseix et al. 1990 |
| **MixedModelLayout** | Combines visibility representation with barycentric refinement | Kant 1996 |
| **PlanarizationLayout** | Layout for non-planar graphs via planarization (insert dummy crossings) | Batini et al. 1986 |

### Upward Drawing Algorithms (New Category)

Entire category missing. Important for DAG visualization where edge direction matters.

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| **VisibilityLayout** | Nodes as horizontal segments, edges as vertical segments | DAGs, st-planar digraphs |
| **DominanceLayout** | x/y coordinates encode dominance relationships | st-planar digraphs |
| **UpwardPlanarizationLayout** | All edges point upward via upward planarization | General DAGs |

### Additional Force-Directed Algorithms

Algorithms representing fundamentally different optimization paradigms from current implementations.

| Algorithm | Description | Distinct Value |
|-----------|-------------|----------------|
| **GEMLayout** | Per-node adaptive temperature | Different convergence behavior |
| **TutteLayout** | Barycentric embedding for 3-connected planar graphs | Provably convex planar drawings |
| **DavidsonHarelLayout** | Simulated annealing with composite energy | Different optimization paradigm entirely |
| **NodeRespecterLayout** | Force-directed with built-in non-overlap | Pure force approach (vs Cola's constraint approach) |
| **BertaultLayout** | Force-directed preserving graph topology | Crossings never change |

### Sugiyama Framework Enhancements

Current Sugiyama implementation works but lacks pluggable alternatives OGDF provides.

| Stage | Current | Missing Alternatives |
|-------|---------|---------------------|
| Cycle removal | DFS-based | Greedy cycle removal heuristic |
| Layer assignment | Longest path | Coffman-Graham ranking, ILP-based optimal ranking |
| Crossing minimization | Barycenter | Median heuristic, Sifting, Greedy insert/switch, Grid sifting |
| Coordinate assignment | Basic | Fast hierarchy layout, ILP-optimal coordinate assignment |

### Complete Type Annotations

Finish typing the internal Cola module files that currently have `ignore_errors = true`:

- `cola/layout.py` (1304 lines) - Core layout class with fluent API
- `cola/gridrouter.py` (800 lines) - Grid-based edge routing
- `cola/batch.py` (207 lines) - Batch layout operations

**Known issues**: 146 mypy errors when strict checking is enabled. Main causes: Optional handling, Union types from getter/setter pattern, dynamic attribute access on ported JavaScript objects.

### API Documentation

Generate API reference documentation using Sphinx or MkDocs.

### Property-Based Tests

Add Hypothesis tests for robustness:

- Fuzz testing for input validation
- Property-based tests for layout invariants
- Edge case testing (very large/small inputs)

---

## Low Priority

### Balloon Tree Layout

Radial subtree packing -- each subtree occupies a "balloon" wedge. Visually distinct from RadialTree (which uses concentric rings). Practical for dense, unbalanced trees.

### Cluster Graph Layouts (New Category)

Important for grouped/hierarchical data. Architecturally complex.

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| **ClusterPlanarizationLayout** | Planarization-based layout respecting nested cluster boundaries | Grouped network data |
| **ClusterOrthoLayout** | Orthogonal layout keeping clustered nodes within bounding rectangles | UML packages, org charts |

### Simultaneous Graph Drawing

Drawing multiple graphs simultaneously on shared vertex positions. Useful for visualizing graph evolution over time.

### Component Packing

Automatic arrangement of disconnected components. Current library lacks dedicated packing.

- `TileToRowsCCPacker`: Tile components into rows
- `SimpleCCPacker`: Simple greedy packing

### Graph Augmentation

Add minimum edges to achieve structural properties while preserving planarity.

- Biconnectivity augmentation
- Planar augmentation

### Edge Label Placement

Automated edge label positioning to avoid overlaps with nodes and other edges.

### SPQR-Tree Decomposition

Triconnected component decomposition. Foundational data structure for many planar graph algorithms (embedding, drawing, planarity).

### Incremental Layout

Support graph modifications without full re-layout:

- Add/remove nodes dynamically
- Preserve existing positions where possible

### GPU Acceleration

Consider CuPy backend for large graphs:

- Parallel force calculations
- CUDA/Metal compute shaders

---

## Potential Future Algorithms

### Specialized Layouts

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| **Arc Diagram** | Nodes on line, arcs above/below | Sequence data, timelines |
| **Grid** | Nodes on regular grid | When uniform spacing needed |

### Tree Variants

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| **Dendrogram** | Hierarchical clustering tree | Clustering results |
| **Icicle/Sunburst** | Space-filling tree | File systems, hierarchies with size |
| **Balloon Tree** | Circular subtree packing | Dense trees |

### Modern/Research

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| **UMAP layout** | Graph from UMAP embedding | High-dimensional data |
| **t-SNE layout** | Graph from t-SNE embedding | Cluster visualization |
| **CoSE** | Compound Spring Embedder | Nested/grouped graphs |

### Geometric Algorithms (CGAL-equivalent)

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| **CrossingMinimalPosition** | Position vertices to minimize crossings geometrically | Planar graph optimization |
| **GeometricEdgeInsertion** | Insert edges into straight-line drawings with minimal crossings | Incremental graph drawing |

---

## Missing Features

Other features common in graph layout libraries that are not yet implemented:

| Feature | Description |
|---------|-------------|
| Hierarchical Edge Bundling | Reduce visual clutter in dense graphs |
| Layout Blending/Morphing | Animation between layout states |
| Node Label Placement | Automatic label positioning to avoid overlaps |

---

## Test Coverage Gaps

### Under-Tested Modules (<80%)

| Module | Coverage | Notes |
|--------|----------|-------|
| `cola/batch.py` | 43% | Batch processing functionality |
| `cola/shortestpaths.py` | 47% | Wrapper module (fallback paths untested) |
| `cola/_shortestpaths_py.py` | 67% | Pure Python fallback |
| `base.py` | 71% | Some inherited methods untested |
| `cola/layout.py` | 75% | Many advanced features untested |
| `types.py` | 77% | Some utility methods untested |

### Missing Test Types

- Property-based testing (Hypothesis)
- Performance regression tests
- Fuzz testing

---

## Documentation Gaps

| Document | Description |
|----------|-------------|
| API Reference | Generated docs (Sphinx/MkDocs) |
| Algorithm Guide | When to use which algorithm |
| Performance Guide | Algorithm selection for graph sizes |
| Tutorial | Step-by-step for common use cases |

---

## Performance Improvements

### Potential Optimizations

1. **Parallel Processing**: Force calculations are embarrassingly parallel
2. **GPU Acceleration**: Large graphs could benefit from CUDA/Metal
3. **Lazy Evaluation**: Don't recompute unchanged portions of layout
