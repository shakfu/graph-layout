# TODO

## High Priority

### Bipartite Layout

Two parallel rows/columns for bipartite graphs.

- Automatic bipartite detection or user-specified sets
- Minimize edge crossings between rows
- Common need: user-item, author-paper, gene-disease networks

---

## Medium Priority

### Orthogonal Layout

Edges restricted to horizontal/vertical segments. Complex but valuable for specific domains.

- Essential for: UML diagrams, flowcharts, circuit schematics, ER diagrams
- Edge routing with bend minimization

**Implementation Order:**

1. **Kandinsky** ✅ MVP implemented
   - Works on **any graph** (not just planar) - most practical
   - Allows multiple edges per node side (handles high-degree nodes)
   - Well-documented, used in commercial tools (yEd)
   - Good for UML/ER diagrams where nodes often have many connections
   - **Future improvements**: min-cost flow bend minimization, better compaction

2. **Topology-Shape-Metrics (TSM)** (implement second)
   - Three decoupled phases: planarization → orthogonalization → compaction
   - Each phase can be improved independently
   - Industry standard (used in OGDF library)
   - Most flexible but higher complexity

3. **GIOTTO** (defer or skip)
   - Restricted to planar graphs with max degree 4
   - Less practical for real-world use cases
   - Academic interest mostly

4. **Simple Visibility Representation** (optional foundation)
   - Nodes as horizontal segments, edges as vertical
   - O(n) for planar graphs
   - Good learning stepping stone

### Stress Majorization

Iterative stress minimization that converges faster than Kamada-Kawai with similar quality.

- Better optimization than gradient descent used in KK
- Handles weighted graphs well
- Reference: [Gansner et al. 2004](https://graphviz.org/Documentation/GKN04.pdf)

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

### Export Formats

Add export functionality:

- SVG export
- DOT (Graphviz) export
- GraphML export

### Incremental Layout

Support graph modifications without full re-layout:

- Add/remove nodes dynamically
- Preserve existing positions where possible

### GPU Acceleration

Consider CuPy backend for large graphs:

- Parallel force calculations
- CUDA/Metal compute shaders

### Pivot MDS

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
