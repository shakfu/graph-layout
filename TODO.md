# TODO

## High Priority

### 1. ForceAtlas2 Layout

Implement Gephi's ForceAtlas2 algorithm. This is the most commonly used algorithm for social network visualization and handles large clustered graphs better than Fruchterman-Reingold.

- LinLog mode for better cluster separation
- Gravity to prevent disconnected components drifting
- Barnes-Hut optimization for O(n log n) performance
- Reference: [Jacomy et al. 2014](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0098679)

### 2. Yifan Hu Multilevel Layout

Fast multilevel force-directed algorithm for medium-large graphs (1K-100K nodes).

- Coarsening phase: collapse graph iteratively
- Layout phase: layout coarsest graph
- Refinement phase: uncoarsen and refine positions
- Reference: [Yifan Hu 2005](https://yifanhu.net/PUB/graph_draw_small.pdf)

### 3. Random Layout

Trivial but essential baseline. Random positions within canvas bounds.

- Useful as starting point for iterative algorithms
- Baseline for comparing layout quality metrics
- Simple to implement (high value/effort ratio)

### 4. Bipartite Layout

Two parallel rows/columns for bipartite graphs.

- Automatic bipartite detection or user-specified sets
- Minimize edge crossings between rows
- Common need: user-item, author-paper, gene-disease networks

---

## Medium Priority

### 5. Orthogonal Layout

Edges restricted to horizontal/vertical segments. Complex but valuable for specific domains.

- Essential for: UML diagrams, flowcharts, circuit schematics, ER diagrams
- Algorithms: Kandinsky, GIOTTO, Topology-Shape-Metrics approach
- Edge routing with bend minimization

### 6. Stress Majorization

Iterative stress minimization that converges faster than Kamada-Kawai with similar quality.

- Better optimization than gradient descent used in KK
- Handles weighted graphs well
- Reference: [Gansner et al. 2004](https://graphviz.org/Documentation/GKN04.pdf)

### 7. Complete Type Annotations

Finish typing the internal Cola module files that currently have `ignore_errors = true`:

- `cola/layout.py` (1304 lines) - Core layout class with fluent API
- `cola/gridrouter.py` (800 lines) - Grid-based edge routing
- `cola/batch.py` (207 lines) - Batch layout operations

**Known issues**: 146 mypy errors when strict checking is enabled. Main causes: Optional handling, Union types from getter/setter pattern, dynamic attribute access on ported JavaScript objects.

### 8. API Documentation

Generate API reference documentation using Sphinx or MkDocs.

### 9. Property-Based Tests

Add Hypothesis tests for robustness:

- Fuzz testing for input validation
- Property-based tests for layout invariants
- Edge case testing (very large/small inputs)

---

## Low Priority

### 10. Export Formats

Add export functionality:

- SVG export
- DOT (Graphviz) export
- GraphML export

### 11. Incremental Layout

Support graph modifications without full re-layout:

- Add/remove nodes dynamically
- Preserve existing positions where possible

### 12. GPU Acceleration

Consider CuPy backend for large graphs:

- Parallel force calculations
- CUDA/Metal compute shaders

### 13. Pivot MDS

Fast MDS approximation using pivot nodes. Useful when spectral layout is too slow for large graphs.

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
