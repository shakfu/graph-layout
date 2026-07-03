# graph-layout: Algorithm Correctness Review

> Archived snapshot. This is the point-in-time correctness review that drove the
> fixes now recorded in `CHANGELOG.md`. The forward-looking orthogonal-compaction
> work (turn-regularization) has its own living design doc at
> `docs/rectangularization-plan.md`; the broader wish list lives in `TODO.md`.
> Kept here for the findings narrative, the "what is verified correct" record,
> and the prioritized missing-algorithms assessment (sections 6-8).

Date: 2026-07-03 Scope: correctness of the implemented layout algorithms and supporting data structures, plus an assessment of worthwhile missing algorithms. Baseline: full test suite passes (1044 tests) before this review.

This review was produced by reading each algorithm family against its canonical published description. Findings marked **[verified]** were additionally reproduced or confirmed by direct code inspection or a runnable repro during the review; the rest are read-based and should be treated as high-confidence but unconfirmed by execution.

---

## 0. Fixes Applied (post-review)

The following findings have since been fixed and locked behind regression tests
that were each confirmed to fail on the original bug. The suite grew from 1044
to 1060 passing tests.

| ID | Fix | Files | Tests |
|----|-----|-------|-------|
| C3 | LR planarity nesting-depth off-by-one (`2*height[w]+1` -> `2*height[w]`) | `planarity/_lr_planarity.py` | `tests/test_planarity_regression.py` (order-independence, Euler-validity, networkx oracle) |
| H9 | Euler `3n-6` bound now counts distinct pairs, not multi-edges | `planarity/__init__.py` | `tests/test_planarity_regression.py` (multigraph cases) |
| H2 | Spring repulsion no longer divides by `dist_sq==0` for coincident nodes | `force/spring.py` | `tests/test_force_layouts.py` (coincident naive + Barnes-Hut) |
| H1 | ForceAtlas2 Barnes-Hut now applies the acting node's `(deg_i+1)` factor (Python + Cython) | `spatial/quadtree.py`, `_speedups.pyx` | `tests/test_force_atlas2.py` (kernel-vs-naive, Python quadtree mass) |
| C1 | Cola VPSC projection implemented: separation + alignment + non-overlap constraints now enforced in `Layout` | `cola/rectangle.py` | `tests/test_layout.py` (strengthened overlap + separation, new alignment) |
| H3/H4 | Sugiyama cycle removal now invoked; dummy nodes inserted for edges spanning >1 layer (fixes barycenter contamination and crossing counting); best-ordering retained across sweeps; bend points exposed via `edge_bends` | `hierarchical/sugiyama.py` | `tests/test_hierarchical_layouts.py` (cycle removal, dummy/bends, crossing minimization) |
| H6b | Orthogonalization bends modeled per edge (unique intermediate node) instead of per face-pair, so faces sharing multiple edges no longer duplicate bends; solvers include arc-only nodes | `orthogonal/orthogonalization.py`, `orthogonal/_min_cost_flow.py` | `tests/test_min_cost_flow.py` (per-edge bends, no-duplication invariant) |
| C2 flow-model | Bend-to-face sign now aligned with the embedding (bends attributed to the dart bordering each face), so the orthogonalization emits **valid** orthogonal representations -- every face turns +/-4 -- across its domain (biconnected, max degree 4) | `orthogonal/orthogonalization.py` | `tests/test_orthogonal_metrics.py` (all-faces-+/-4 for K4/wheel/prism + grids) |
| C2 shape stage | New `orthogonal/metrics.py`: assigns compass directions to every edge segment from the representation; detects unrealizable reps for safe fallback | `orthogonal/metrics.py` | `tests/test_orthogonal_metrics.py` (shape validity, reverse-dart opposition, detection) |
| C2 coordinates | `compute_coordinates()`: shape -> integer coordinates via H/V constraint-graph longest-path; flags coincident-vertex degeneracies | `orthogonal/metrics.py` | `tests/test_orthogonal_metrics.py` (square/grids/K4 drawings, axis-alignment, distinctness) |
| C2 wiring | GIOTTO `bend_optimal` opt-in drives the drawing from the bend-minimal representation (fallback to heuristic out of domain) | `orthogonal/giotto.py` | `tests/test_giotto_bend_optimal.py` (valid orthogonal drawings, fallback, default off) |
| GIOTTO recursion | `_assign_layers` no longer infinite-recurses on cyclic input (back-edge guard) | `orthogonal/giotto.py` | `tests/test_giotto_bend_optimal.py` (cyclic graph runs) |

Notes / scoped limitations:

- **C3** regression net additionally validates every planar embedding via
  `V - E + F == 2` face-tracing, closing the test blind spot (all prior
  planarity tests used sorted-order graphs that dodged the bug).
- **H9** does not change the documented "3+ parallel edges" convention (see
  section 5 observation); mathematically 3+ parallel edges are planar, but that
  is a separate design decision, left as-is.
- **C1** solves node variables only. Nested-**group** containment (the
  `min_var`/`max_var` machinery) is not yet enforced; grouped layouts run
  without error but group bounding rectangles are not constrained. This is the
  remaining piece of a complete WebCola projection port.
- **H3/H4** fixes the barycenter contamination, crossing miscount, and edge
  routing consequences of missing dummy nodes, and wires in cycle removal. The
  separate MEDIUM item -- coordinate assignment is still index-based rather than
  Brandes-Kopf/priority (section 4) -- is unchanged; dummy nodes now occupy real
  horizontal slots, which straightens long edges but does not add barycentric
  x-alignment. The standalone `preprocessing.count_crossings` utility (its own
  MEDIUM item) is likewise untouched; Sugiyama now counts crossings internally
  over the dummy-expanded graph.

Still open in the orthogonal stack:

- **C2 stage 1 (shape computation) is done** (`orthogonal/metrics.py`,
  `tests/test_orthogonal_metrics.py`): given a valid orthogonal representation
  and the faces, it assigns a compass direction to every edge segment by
  propagating turns around faces, and it *detects* representations that are not
  realizable so callers can fall back safely.

- **Flow model now emits valid representations (fixed).** Building the shape
  stage surfaced that the orthogonalization did not produce valid orthogonal
  representations: a valid representation must have every bounded face turn by
  +4 quarter-turns (the outer face by -4), where a face's turn is
  `sum(2 - angle over corners) + sum(bends)`. For K4 the representation had face
  turn-sums of `[0, 0, 4, 4]` (verified to predate this session). The cause was
  that the bend-to-face sign in `flow_to_orthogonal_rep` was not tied to the
  directed-edge/face incidence. The fix records, per edge, the dart bordering
  the first face and attributes each bend as +1 on that dart's side and -1 on
  the reverse -- so every face now turns +/-4.
  - **Verified scope:** biconnected planar graphs of max degree 4 (the standard
    Tamassia domain) -- all faces turn +/-4 across grids, K4, cube, wheel,
    prism, theta, and 62/62 random biconnected max-degree-4 graphs.
  - **Still out of scope (separate known items), where the flow is infeasible
    or the rep inconsistent, and `ShapeResult.valid` is False so callers fall
    back to the heuristic router:**
    - **H5 (degree > 4)**: a vertex of degree > 4 has negative supply but only
      outgoing angle arcs, so the flow is infeasible. Needs the Kandinsky
      0-degree-angle model (a public `OrthogonalRepresentation` change).
    - **H6a (non-biconnected)**: bridges and cut vertices give faces with
      repeated vertices; the `(vertex, face)`-keyed angle map cannot store a cut
      vertex's two corners on one face. Needs per-corner angles (or biconnected
      decomposition before orthogonalization).
- **C2 metrics stages 2-3: done for the in-scope domain.** Coordinate
  assignment (`compute_coordinates`) and GIOTTO wiring (`bend_optimal` opt-in)
  are implemented and tested: for biconnected max-degree-4 graphs the drawing is
  now produced from the bend-minimal representation (grids, K4, cube, wheel,
  prism -- non-overlapping boxes, orthogonal edges). Out-of-domain inputs fall
  back to the heuristic router.
  - Kept opt-in (default off) so the existing heuristic-based tests -- e.g. the
    "<= 2 bends per edge" assertion -- are unaffected.
  - **Safe by construction:** `compute_coordinates` detects drawings that are
    not clean (overlaps, crossings, edges through a vertex) and reports them
    invalid, so `bend_optimal` falls back to the heuristic rather than emit a
    broken drawing. Verified: no drawing reported valid has a conflict over
    random biconnected max-degree-4 graphs.
  - **Two-tier coordinate assignment (done):** compact longest-path first, then
    a "spread" assignment (distinct coordinate per class) that separates the
    independent features longest-path collapses. This raised clean bend-optimal
    coverage from ~83% to **~89%** of in-scope graphs (all common structured
    ones -- grids, K4, cube, wheel, prism -- clean); the rest fall back.
  - **Remaining -- turn-regularization for compaction:** the ~11% that still
    fall back have genuine *crossings* (the coordinate assignment is non-planar,
    verified by conflict-type classification). Closing this needs the separation
    constraints that keep the two sides of a non-rectangular face apart, added
    only between the specific reflex-corner ("kitty corner") pairs that conflict
    -- not a naive all-pairs sweep, which was tried and *regressed* coverage to
    86% by creating constraint-graph cycles. The full design, named reference
    algorithm (Bridgeman et al., "Turn-Regularity and Optimal Area Drawings of
    Orthogonal Representations," 2000), integration points, and the ready-made
    verification oracle are written up in
    [`docs/rectangularization-plan.md`](rectangularization-plan.md). Completing
    it takes coverage toward 100% and lets `bend_optimal` be the default. Plus
    **H5** (Kandinsky 0-degree angles for degree > 4) and **H6a** (per-corner
    angles for non-biconnected graphs) for the out-of-domain cases.

---

## 1. Executive Summary

The library is broad and, in its numerically-oriented core, largely correct. The force-directed math (Fruchterman-Reingold, Kamada-Kawai), the tidy-tree implementation (Reingold-Tilford / Buchheim), the VPSC constraint solver, the min-cost-flow solver, and the low-level planarity machinery (conflict pairs, biconnected components, block-cut tree) are all faithful to their sources.

However, three of the library's headline "advanced" capabilities do not actually deliver what their documentation claims, and each was confirmed by direct inspection:

1. **Constraint-based layout (Cola) does not enforce constraints.** The VPSC projection used by the `Layout` descent loop is an empty stub, so user separation/alignment constraints and `avoid_overlaps=True` are silently no-ops. **[verified]**

2. **Orthogonal bend minimization is decorative.** Both `GIOTTOLayout` and `KandinskyLayout` compute a bend-minimizing orthogonal representation and then discard it; the drawn bends come from a local geometric heuristic. GIOTTO's advertised "bend-optimal" property is not realized. **[verified]**

3. **The planarity test returns wrong answers.** A one-character off-by-one in the Left-Right nesting-depth computation makes `check_planarity` report some planar graphs as non-planar (order-dependently) and produce non-planar embeddings. **[verified with repro]**

These three are the priority items. Everything else is a narrower bug, a missing canonical sub-step, or a robustness/edge-case issue.

### Severity tally

| Severity | Count | Examples |
|----------|-------|----------|
| Critical | 3 | Cola projection stub; orthogonal rep discarded; LR planarity off-by-one |
| High | 7 | FA2 Barnes-Hut degree factor; Spring divide-by-zero; Sugiyama dummy nodes; Kandinsky flow model absent; greedy compaction; planarization model; planar multigraph rejection |
| Medium | ~12 | FA2 gravity swap; Yifan Hu fixed nodes; quadtree recursion; grid-snap sign; spectral disconnected/normalization; edge-crossing collinearity; Sugiyama coordinates |
| Low | ~20 | recursion depth, fixed-node inconsistency, dead code, docstring/behavior mismatches |

---

## 2. Critical Findings

### C1. Cola constraint projection is an unimplemented stub **[verified]**

`cola/rectangle.py:618-626`

```python
def x_project(self, x0, y0, x) -> None:
    # Implementation simplified for brevity
    pass
def y_project(self, x0, y0, y) -> None:
    # Implementation simplified for brevity
    pass
```

`Projection.project_functions()` returns lambdas wrapping these two methods, and `cola/layout.py:863-865` and `877-879` assign that result to `self._descent.project`. During `Descent.step_and_project` the projection is invoked but returns immediately, so:

- User separation and alignment constraints have no effect in `Layout`.

- `avoid_overlaps=True` performs no overlap removal inside the descent loop.

The full IPSep-CoLa projection (build variables, generate axis constraints, run the VPSC `Solver`, write positions back, plus group-containment via min/max variables) is missing. Note that the standalone `remove_overlaps()` in the same file is correctly implemented, but `Layout` never calls it. The test suite passes because the `Layout` tests assert convergence and rough positions, not constraint satisfaction.

Impact: the primary differentiator of the WebCola port -- constraint-based layout with overlap avoidance -- is inert. This should be treated as unimplemented rather than buggy.

### C2. Orthogonal representation is computed then discarded **[verified]**

`orthogonal/giotto.py` and `orthogonal/kandinsky.py`

Both layouts call `_compute_orthogonal_rep()` (giotto.py:353, kandinsky.py:448), which stores a bend-minimizing `OrthogonalRepresentation` in `self._orthogonal_rep` (giotto.py:420/424, kandinsky.py:491/495). A grep confirms this attribute is only ever assigned and returned by a read-only property; no routing or export code reads it. The bends actually drawn come from the local 5-case L/Z heuristic in `edge_routing.py:route_edge` (max two bends per edge), and the port sides come from a position heuristic rather than from the computed angle assignments.

Consequences:

- GIOTTO's docstring claim that "no other orthogonal drawing can have fewer bends" (giotto.py:8-9) is unsupported by the implementation.

- `total_bends` reflects heuristic bends, not the flow-optimal bends.

- The min-cost-flow orthogonalization (which is itself correct in isolation, see section 4) is dead-ended.

To realize the advertised behavior, the compaction/routing stage must consume `self._orthogonal_rep` (the angle and bend sequences) instead of re-deriving bends geometrically.

### C3. Left-Right planarity nesting-depth off-by-one **[verified with repro]**

`planarity/_lr_planarity.py:166`

```python
else:
    self.nesting_depth[e] = 2 * self.height[w] + 1
```

For a back edge `e = (v, w)` (w an ancestor), the canonical Brandes value is `2 * lowpt[e]` with a `+1` only when the edge is chordal (`lowpt2[e] < height[v]`). For a pure back edge `lowpt[e] = height[w]` and `lowpt2[e] = height[v]`, so the correct value is `2 * height[w]` with **no** `+1`. The tree-edge branch (lines 162-164) is correct; only this branch is wrong.

The spurious `+1` corrupts the nesting-depth sort that both the planarity decision and the embedding construction depend on. Reproduced during review:

```python
from graph_layout import check_planarity
edges = [(0,2),(2,4),(0,4),(1,2),(3,4),(1,3),(2,3),(1,4),(0,3)]  # maximal planar
check_planarity(5, edges).is_planar          # -> False  (WRONG)
check_planarity(5, sorted(edges)).is_planar  # -> True   (order-dependent)
```

The order-dependence is the signature of the bug: the same graph is judged differently depending on adjacency order. The reviewing agent's differential test against NetworkX found dozens of false negatives (planar reported non-planar) and invalid, non-planar embeddings over thousands of random graphs, and confirmed that changing the line to `2 * self.height[w]` drives mismatches to zero. No false positives were observed. Because the Kuratowski extractor reuses this phase, its witness rate is also degraded.

Impact: `is_planar` / `check_planarity` can return incorrect results, and `PlanarEmbedding` results can be non-planar (Euler's formula violated). This propagates to every embedder (`MaxFaceEmbedder`, `MinDepthEmbedder`, etc.), which assume a valid planar embedding. This is a one-line fix.

---

## 3. High-Severity Findings

### H1. ForceAtlas2 Barnes-Hut repulsion drops the acting node's degree factor **[verified]**

`force/force_atlas2.py:583-585`, `spatial/quadtree.py:248`

FA2 repulsion should be `scaling * (deg_i + 1) * (deg_j + 1) / d`. The quadtree stores the source mass `(deg_j + 1)` and computes `force = k_sq * node.total_mass / dist`, but the acting body's mass `(deg_i + 1)` (`Body(..., mass=deg_i)`) is never consumed. The result is `scaling * (deg_j + 1) / d`, missing the `(deg_i + 1)` factor. This is the default path for graphs above 50 nodes (Barnes-Hut on by default) and is present in both the pure-Python and Cython implementations. Hubs under-repel, materially changing layouts. Fine for FR / Yifan Hu, where mass is uniformly 1.

### H2. Spring layout divides by zero on coincident nodes **[verified]**

`force/spring.py:343-348`

```python
dist = math.sqrt(dist_sq) if dist_sq > 0 else 0.0001
if dist > 0:
    force = self._repulsion / dist_sq   # dist_sq is still 0.0 here
```

The guard patches `dist` but the denominator is `dist_sq`, which remains `0.0` for coincident nodes. `dist > 0` is true (0.0001), so this raises `ZeroDivisionError`. Triggered by duplicate or coincident fixed coordinates. FR avoids this by dividing by `dist`.

### H3. Sugiyama inserts no dummy nodes for long edges

`hierarchical/sugiyama.py:171-297`

Edges spanning non-adjacent layers are never subdivided with virtual nodes. This is the defining step of the Sugiyama method and its absence causes genuine correctness defects, not just aesthetics:

- Barycenter ordering averages per-layer position indices of neighbors that may be several layers away, so the ordering signal is corrupted (sugiyama.py:287-297).

- Intermediate layers exert no ordering force on long edges, so their crossings are neither minimized nor counted.

- `preprocessing.count_crossings` (preprocessing.py:576-631) buckets edges by layer-pair and is therefore incorrect for graphs with long edges.

- Long edges are drawn as straight lines that can pass through unrelated nodes.

### H4. Sugiyama never runs cycle removal

`hierarchical/sugiyama.py:369-382`

The pipeline calls layer assignment, crossing minimization, and coordinate assignment, but never `remove_cycles` (which exists and is correct in `preprocessing.py:132-209`). Cyclic input is handled only by a warning and a fallback, so back edges receive an arbitrary layering that violates the "edges point downward" invariant. `remove_cycles` is effectively dead code with respect to the layout.

### H5. Kandinsky flow model is not actually implemented

`orthogonal/orthogonalization.py`

`build_flow_network` is identical for GIOTTO and Kandinsky. Angle arcs have capacity 3, so an angle is confined to [1,4] units and a 0-degree angle (two edges leaving one side of a vertex box) is impossible -- yet 0-degree angles are exactly what the Kandinsky model adds to support degree > 4. There are no bend-or-not constraints. For a vertex of degree > 4 the supply `4 - deg` becomes negative while the vertex has only outgoing angle arcs, so the flow is infeasible and all angles fall back to 90 degrees. `AngleType.DEGREE_0` is declared but never used. Kandinsky's degree > 4 handling comes solely from geometric port spreading in `edge_routing.py`, not from the flow model. (Largely moot today because of C2, but wrong on its own terms.)

### H6. Orthogonalization network mismodels cut vertices and multi-edge faces

`orthogonal/orthogonalization.py:375-393`, `586-595`

- Angle arcs are deduplicated by `(vertex, face)`, so a cut vertex that appears multiple times on one face collapses into a single capacity-3 arc, while the face demand still counts the repeated occurrences. Non-biconnected instances become infeasible or wrongly solved.

- Bend arcs are keyed per face-pair, not per edge, and deduplicated. Multiple edges shared between the same two faces read the same flow value, so bends are mis-attributed; bridges (same face on both sides) are dropped entirely.

Again mostly moot under C2, but these would be defects if the representation were consumed.

### H7. "Planarization" is geometric crossing detection, not topological planarization

`orthogonal/planarization.py:190-343`

Canonical TSM planarization computes a planar embedding of a maximal planar subgraph and reinserts edges to minimize crossings, adding dummy vertices topologically. This implementation instead takes straight-line coordinates from the layering phase and inserts dummy vertices at geometric segment intersections. Consequences: crossing count depends on arbitrary initial coordinates rather than a crossing-minimization heuristic; a genuinely planar graph laid out poorly gets spurious crossing vertices; and because edges are later rerouted orthogonally (not along the straight segments used for detection), planarity of the final drawing is not guaranteed. It is also disconnected from the correct `check_planarity`/embedder machinery.

### H8. Greedy compaction does not compact

`orthogonal/compaction.py:95-97`, `169-177`

The greedy `CompactionSolver.solve` only ever pushes an element rightward when a minimum-gap constraint is violated; it never pulls elements left to close interior slack. Starting from the already-spread coordinates, any gap larger than the minimum is preserved, and the only size reduction is a final whole-layout translation that trims outer margins. The docstring's "minimize the total area" is not achieved. This is the default GIOTTO path and the "auto" path when scipy is unavailable. (`compaction_flow.compact_longest_path_1d` does compact correctly and should be preferred.)

### H9. Planar multigraphs wrongly rejected **[verified with repro]**

`planarity/__init__.py:82-83`, also `_lr_planarity.py:30-31`, `:514`

The Euler edge bound `m > 3n - 6` is applied to the multigraph edge count, which retains up to two parallel edges per pair. `3n - 6` is a simple-graph bound, so a planar multigraph legitimately exceeds it:

```python
from graph_layout import check_planarity
check_planarity(3, [(0,1),(0,1),(1,2),(2,0)]).is_planar   # -> False (WRONG)
```

Fix: compute the bound on the deduplicated simple-edge count, or add slack for parallel edges.

---

## 4. Medium-Severity Findings

- **FA2 regular vs strong gravity swapped** (`force_atlas2.py:658-668`, `_speedups.pyx:1065-1071`). Regular gravity should be distance-independent and strong gravity distance-scaled; the code inverts the two. Both still pull toward center.

- **FA2 global-speed update lacks max-rise/jitter damping** (`force_atlas2.py:479-482`). Displacement stays bounded, so it does not explode, but convergence dynamics differ from Jacomy et al.

- **Yifan Hu ignores fixed nodes during optimization** (`yifan_hu.py:533-536`, `791-794`). Fixed nodes move freely during the simulation and are only restored at copy-back, so their intermediate (wrong) positions perturb other nodes throughout.

- **Pure-Python quadtree can infinitely recurse on coincident points** (`spatial/quadtree.py:129-147`). No depth cap; the Cython version guards with a depth-50 cap. Reachable via Spring Barnes-Hut and Python fallbacks.

- **Cola grid-snap uses Python `%` where WebCola relies on JS sign semantics** (`descent.py:279-285`). JS `%` takes the dividend's sign; Python's takes the divisor's. For negative coordinates the snap direction flips, breaking `grid_snap_iterations > 0` for roughly half the nodes.

- **Cola groups initial-layout raises KeyError** (`layout.py:937-950`). The port reads `vs[i]["x"]` from dicts that the Python `nodes()` setter never mutates (it builds fresh `Node` objects). Fires when `groups` are present and `initial_unconstrained_iterations > 0`.

- **Spectral: disconnected graphs pick zero-eigenvalue eigenvectors** (`spectral.py:193-196`). The code blindly takes eigenvector indices 1 and 2; for a graph with k components, eigenvalue 0 has multiplicity k, so these are component indicators, not Fiedler vectors, collapsing each component to a point. Selection is correct for connected graphs.

- **Spectral: normalized Laplacian used without the `D^{-1/2}` back-transform** (`spectral.py:145-153`, default on). The canonical degree-weighted layout (Koren) scales symmetric-normalized eigenvectors by `D^{-1/2}`; omitting it mis-weights high-degree nodes.

- **Edge-crossing metric misses collinear / T-junction cases** (`metrics.py:89-92`). The boolean CCW straddle test is correct for general position but silently drops collinear overlaps and endpoint-on-segment touches, contrary to the docstring.

- **Sugiyama coordinate assignment is index-based** (`sugiyama.py:335-363`). No Brandes-Kopf or priority alignment; nodes are placed by integer position and each layer centered independently, discarding the barycenter values and producing avoidable bends.

- **`compact_flow_1d` can be looser than longest-path** (`compaction_flow.py:360-376`), despite its "tighter layouts" docstring.

- **Shell layout raises IndexError on out-of-range link indices** (`circular/shell.py:159-163`). `_compute_degrees` indexes without the bounds guard used everywhere else in the codebase.

---

## 5. Low-Severity Findings and Nits

- Recursion-depth limits on deep trees/graphs in the recursive walks of `reingold_tilford.py`, `radial_tree.py`, and `preprocessing.detect_cycle`.

- `node.fixed` is honored by `RandomLayout` but ignored by Bipartite, Circular, and Shell layouts (inconsistent pinning semantics).

- Nodes absent from user-supplied bipartite sets / shells are left unpositioned and unvalidated (`bipartite.py:235-236`, `shell.py:194-196`).

- Radial tree allocates angular wedges by subtree node count rather than leaf count (`radial_tree.py:285-294`); a spacing-quality variant, not a bug (no overlap results).

- `kandinsky.py:333` setter rejects `"flow"` and `"longest_path"` even though the constructor documents and implements them **[verified]**.

- Cola: coincident-node perturbation from WebCola omitted (`descent.py`, `offset_dir` is dead code); grid-snap boundary strictness; `_alpha is None` fragility in `tick()`.

- Spring class docstring says "constant force" repulsion but implements inverse-square (`spring.py:34`).

- `metrics.stress` returns normalized (not raw) stress and BFS ignores link weights; `angular_resolution` reports spurious 0 degrees for self-loops and parallel edges.

- `RandomLayout` seeds the global `random` module instead of a local `random.Random` instance (`random.py:147`).

- `nudge_overlapping_segments` (`edge_routing.py`) is exported but never called; finite bend capacity 4 in the flow network; edge segments ignored during compaction.

- `preprocessing.connected_components` has an unused `directed` parameter.

- Several misleading docstrings in `cola/shortestpaths.py` ("Johnson's", path inclusion), harmless.

---

## 6. What Is Correct (Verified Faithful)

For balance, the following were checked closely and found faithful to their canonical sources, with no correctness bugs:

- **Fruchterman-Reingold**: `k^2/d` repulsion, `d^2/k` attraction, temperature capping and cooling, Barnes-Hut integration -- all correct.

- **Kamada-Kawai**: gradient, full 2x2 Hessian, Newton-Raphson Cramer solve, spring constants `K/d^2`, ideal lengths `L*d` -- verified term-by-term.

- **Reingold-Tilford**: a line-accurate linear-time Buchheim/Walker implementation (apportion, threads/contours, `move_subtree`, `execute_shifts`, second walk `x = prelim + mod`-sum). No bugs.

- **Radial tree**: correct depth-to-radius mapping and disjoint nested angular wedges (overlap-free).

- **VPSC solver** (`cola/vpsc.py`): faithful port of the block-based active-set algorithm -- merging, Lagrange multipliers, split-on-negative-multiplier, comparator returns proper ints. No bugs.

- **Cola descent** math (gradient, Hessian, step size, Runge-Kutta, P-stress masking, LCG PRNG) matches WebCola.

- **Min-cost flow** (`orthogonal/_min_cost_flow.py`): correct successive shortest paths with Johnson potentials on reduced costs.

- **Planarity conflict-pair machinery**: `add_constraints`, `trim_back_edges`, sign resolution, per-edge lowpt reconstruction -- all match Brandes / NetworkX (the C3 bug is isolated to nesting depth).

- **Biconnected components and block-cut tree**: verified against NetworkX, zero mismatches over random graphs.

- **Convex hull / tangents** (`cola/geom.py`), **pairing heap** (`cola/pqueue.py`), **Dijkstra shortest paths** -- faithful ports.

---

## 7. Missing Algorithms Worth Adding

The project's `TODO.md` already enumerates a large wish list. Rather than repeat it, this section gives an independent priority assessment. The most important recommendation is a caveat:

**Fixing the existing critical gaps (sections 2-3) has higher value than any new algorithm.** Completing the VPSC projection, consuming the orthogonal representation, adding Sugiyama dummy nodes, and fixing the LR off-by-one each make an already-shipped, documented capability actually work. New algorithms should come after.

With that said, the genuine coverage gaps, prioritized:

### High value

- **SMACOF stress majorization.** The library's only stress-based method is Kamada-Kawai's gradient descent. SMACOF (Gansner et al. 2004) is the modern standard: better convergence, robust weighting, and it scales better. It reuses the existing graph-distance and stress code. This is the single most worthwhile addition.

- **Pivot MDS / classical MDS.** `O(k*n)` approximate layout using k pivot nodes. Fills the "fast initial layout for large graphs" gap that neither spectral (eigensolve cost, and currently buggy on disconnected graphs) nor the force methods cover well. Pairs naturally with SMACOF as an initializer.

- **Component packing** (tile-to-rows / greedy). Every layout here handles a single connected component implicitly; disconnected graphs are laid out with ad-hoc separation. A dedicated packer is low-complexity and broadly useful.

### Medium value

- **Tutte barycentric embedding.** Provably crossing-free convex drawing for 3-connected planar graphs. Small, foundational, and directly leverages the (once fixed) planarity/embedding module. Good building block and teaching example.

- **Planar straight-line drawing** (Schnyder realizers, or de Fraysseix-Pach-Pollack). An entire missing category with strong theoretical standing, and the planarity embedders provide the needed substrate. Higher implementation cost.

- **Upward drawing for DAGs** (dominance drawing, visibility representation). Complements Sugiyama for direction-encoding layouts of st-planar digraphs.

### Lower value / niche

- Alternative force paradigms (GEM per-node temperature, Davidson-Harel simulated annealing) -- mainly useful as different optimizers, not new capability.

- Balloon tree, dendrogram, arc diagram, grid -- simple specialized layouts, easy but low marginal value given existing tree/circular coverage.

- SPQR-tree decomposition -- foundational for advanced planar algorithms but only worth it if the planar-drawing category is pursued seriously.

---

## 8. Recommended Order of Work

1. **C3** LR planarity off-by-one -- one line, restores correct planarity decisions and valid embeddings (and improves Kuratowski extraction).

2. **H9** planar multigraph edge-bound -- small, removes false rejections.

3. **H1 / H2** ForceAtlas2 degree factor and Spring divide-by-zero -- localized fixes to the default numerical paths.

4. **C1** Cola VPSC projection -- larger, but restores the port's core feature.

5. **C2 / H5 / H6** orthogonal representation consumption and Kandinsky flow model -- the orthogonal stack needs the rep wired through routing before its optimality claims hold.

6. **H3 / H4 / medium Sugiyama items** -- dummy nodes, cycle-removal wiring, and Brandes-Kopf coordinates to make hierarchical layouts correct on real DAGs.

7. Test-suite hardening: Euler-formula validation for planar embeddings, adjacency-order fuzzing (would have caught C3), and constraint-satisfaction assertions for Cola (would have caught C1).

8. New algorithms per section 7, starting with SMACOF.

Note the recurring theme in items 1, 4, and 7: several critical defects are invisible to the current tests because the tests assert coarse outcomes (convergence, a boolean, node counts) rather than the defining property of the algorithm (constraints satisfied, embedding planar, bends minimal). Tightening the tests toward those properties is as important as the fixes themselves.
