# TODO

ref: Open Graph Drawing Framework <https://github.com/ogdf/ogdf>

## High Priority

### Orthogonal Layout Improvements

Items for closing the gap with [OGDF](https://github.com/ogdf/ogdf)'s
orthogonal drawing capabilities. The previously-listed high-priority items are
done:

- [x] **Obstacle-aware segment nudging**: `nudge_overlapping_segments()` now
  checks every nudged segment position against the node boxes and picks a
  clear offset (planned, then mirrored, then unmoved), so separating parallel
  segments can no longer push an edge through a node
  (`edge_routing.py`, tests in `tests/test_edge_routing.py`).
- [x] **Non-biconnected graphs** (H6a): angles are stored per corner (keyed by
  the incoming dart) so face walks that visit a cut vertex or bridge more than
  once no longer collide; the flow network scales angle-arc capacities by
  corner multiplicity, and rectangularization splits the 360-degree corners of
  degree-1 vertices with zero-min-length virtual darts. The bend-optimal
  drawing now covers trees, pendant edges, bridges, and cut vertices
  (`tests/test_tsm_nonbiconnected.py`).
- [x] **Degree > 4** (H5): vertices of degree > 4 are expanded into cage
  cycles in rotation order (the classical GIOTTO / OGDF approach, chosen over
  the Kandinsky 0-degree-angle flow model whose exact form is NP-hard); the
  cage face is constrained to a rectangle (corner angles <= 180 degrees, no
  bends on cycle edges) and drawn as the node box, with edges attaching along
  the box sides at distinct ports (`orthogonal/expansion.py`,
  `tests/test_tsm_expansion.py`).

Together with rectangularization (`orthogonal/metrics.py:_rectangularize`),
`GIOTTOLayout` draws from the bend-minimal orthogonal representation by
default for **all connected planar graphs**; non-planar or disconnected inputs
fall back to the heuristic router (`used_bend_optimal` reports which path
ran). The realization stage is shared (`orthogonal/realization.py`) and also
drives `KandinskyLayout(bend_optimal=True)` (opt-in; default stays the layered
hierarchical layout). Kandinsky's bend-optimal path additionally covers
**non-planar** graphs of maximum degree 4 by realizing the *planarized* graph
(`realize_planarized_drawing`): each crossing is drawn as a clean orthogonal
crossing point, straight-through by construction.

Possible follow-ups (not planned):

- [ ] **Crossings + degree > 4 together**: the crossing realizer requires
  original max-degree <= 4 (cage expansion and crossing gadgets are not yet
  combined), so a non-planar graph with a degree > 4 vertex still falls back.
- [ ] **Compact cages (area vs bends)**: an expanded degree > 4 vertex becomes a
  cage box whose width/height is the grid span of its cage vertices. That span is
  *bend-forced* -- each spoke leaves its cage vertex straight, so the cage must be
  wide enough to align with spread-out neighbours; verified that Kandinsky's
  `bend_optimal` output is exactly bend-minimal (matches GIOTTO) over 137 random
  degree > 4 planar graphs, so shrinking a cage necessarily adds a bend. A
  smaller-cage look would need either a separate mode that trades bends for area,
  or embedding-level neighbour-ordering optimisation (group nearby neighbours on
  the same cage side). Not a bug in the current bend-minimal drawing.
- [ ] True Kandinsky 0-degree-angle flow model (bend-minimal in the Kandinsky
  metric proper, rather than over the expanded graph)
- [x] **Disconnected graphs: per-component TSM drawing + component packing**:
  `GIOTTOLayout` and `KandinskyLayout(bend_optimal=True)` split a disconnected
  graph into connected components, draw each bend-optimally in its own frame
  (recursively, reusing the full pipeline), and reassemble them with a shared
  shelf-packer (`realization.pack_component_drawings`) whose gaps keep component
  bounding boxes -- over boxes and bends -- non-overlapping. Any component
  outside the bend-optimal domain falls the whole graph back to the heuristic
  router (`used_bend_optimal` stays truthful); Kandinsky additionally covers
  disconnected graphs with non-planar components
  (`tests/test_orthogonal_disconnected.py`).

---

## Medium Priority

### Stress Majorization (SMACOF) -- DONE

Implemented as `SMACOFLayout` (`force/smacof.py`). Iterative stress
minimization by majorization (the Guttman transform) rather than the per-node
Newton-Raphson of Kamada-Kawai, so the stress decreases monotonically and it
converges more reliably. Reference:
[Gansner et al. 2004](https://graphviz.org/Documentation/GKN04.pdf).

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
