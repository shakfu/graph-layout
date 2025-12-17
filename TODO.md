# TODO

## Medium Priority

### 1. Complete Type Annotations
Finish typing the internal Cola module files that currently have `ignore_errors = true`:
- `cola/layout.py` (1304 lines) - Core layout class with fluent API
- `cola/gridrouter.py` (800 lines) - Grid-based edge routing
- `cola/batch.py` (207 lines) - Batch layout operations

**Known issues**: 146 mypy errors when strict checking is enabled. Main causes: Optional handling, Union types from getter/setter pattern, dynamic attribute access on ported JavaScript objects.

### 2. API Documentation
Generate API reference documentation using Sphinx or MkDocs.

### 3. Property-Based Tests
Add Hypothesis tests for robustness:
- Fuzz testing for input validation
- Property-based tests for layout invariants
- Edge case testing (very large/small inputs)

---

## Low Priority

### 4. Export Formats
Add export functionality:
- SVG export
- DOT (Graphviz) export
- GraphML export

### 5. Incremental Layout
Support graph modifications without full re-layout:
- Add/remove nodes dynamically
- Preserve existing positions where possible

### 6. GPU Acceleration
Consider CuPy backend for large graphs:
- Parallel force calculations
- CUDA/Metal compute shaders

---

## Missing Features

Features common in other graph layout libraries that are not yet implemented:

| Feature | Description |
|---------|-------------|
| Hierarchical Edge Bundling | Reduce visual clutter in dense graphs |
| Multi-level Layout | Coarsening/refinement for very large graphs |
| Layout Blending/Morphing | Animation between layout states |
| Bipartite Layout | Specialized layout for bipartite graphs |
| Force Atlas 2 | Popular algorithm from Gephi |
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
