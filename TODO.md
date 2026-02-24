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
