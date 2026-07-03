# Design plan: turn-regularization for orthogonal compaction

## Goal

Make `GIOTTOLayout(bend_optimal=True)` produce a clean, planar orthogonal drawing for **100%** of its domain (biconnected planar graphs of maximum degree 4), so it can become the default rather than an opt-in with fallback.

Today the Topology-Shape-Metrics pipeline is:

    embedding -> faces -> min-cost-flow bends -> orthogonal representation (shape)
              -> compute_coordinates() -> drawing

`compute_coordinates()` (in `orthogonal/metrics.py`) assigns integer coordinates by longest-path in two constraint graphs, trying a compact then a "spread" assignment. This is clean for ~89% of in-scope graphs. The remaining ~11% come out **non-planar** (edges cross), because the coordinate assignment lacks the *separation constraints* that keep the two sides of a non-rectangular face apart. Adding those constraints correctly is **rectangularization**.

## Current status of the orthogonal pipeline (already implemented)

This work builds on a completed Topology-Shape-Metrics pipeline. What exists:

- **Flow model emits valid representations.** The min-cost-flow orthogonalization now produces representations where every bounded face turns +4 quarter-turns (outer face -4). The prior bug was that `flow_to_orthogonal_rep` attributed bends by raw edge-tuple order, not by which dart borders which face; the fix records the dart bordering the first face and signs each bend +1 on that side / -1 on the reverse. Verified across grids, K4, cube, wheel, prism, theta and random biconnected max-degree-4 graphs. (`orthogonal/orthogonalization.py`)

- **Shape stage** (`compute_orthogonal_shape`): assigns a compass direction to every edge segment by propagating turns around faces; detects unrealizable representations (`ShapeResult.valid = False`) for safe fallback. `face_turn_sum` checks the +/-4 invariant.

- **Coordinate stage** (`compute_coordinates`): two-tier (compact longest-path, then "spread" = distinct coordinate per class). Raised clean coverage from ~83% to ~89%. Includes `_drawing_conflict` (overlaps / crossings / edge through vertex) so nothing broken is returned.

- **GIOTTO wiring** (`bend_optimal`, default off): draws directly from the bend-minimal representation when the shape is realizable, else falls back to the heuristic router. `_assign_layers` cyclic-input recursion also fixed.

- **Domain**: biconnected planar graphs of maximum degree 4. Out of domain and correctly falling back: **H5** degree > 4 (needs the Kandinsky 0-degree-angle model -- an `OrthogonalRepresentation` change), and **H6a** non-biconnected / bridges / cut vertices (needs per-corner angles; the `(vertex, face)`-keyed angle map cannot store a cut vertex's two corners on one face). Both are separate from this compaction task.

The only remaining gap for the in-scope domain is the ~11% of non-planar coordinate assignments this document addresses.

## Why the naive attempt failed (important)

A first attempt added, within each face, an ordering constraint between every "west wall" and "east wall" whose y-ranges overlap (ordered by their current x). This **lowered** coverage (89% -> 86%): it over-constrains and creates cycles in the constraint graph, because the correct order of two walls is not determined by their coordinates in a (possibly already-crossing) trial drawing.

**The fix is to add separation constraints only between the specific reflex corners that actually conflict -- the "kitty corners" -- not between all wall pairs.** That is exactly what turn-regularization computes.

## Reference algorithm

Primary reference:

> S. Bridgeman, G. Di Battista, W. Didimo, G. Liotta, R. Tamassia, L. Vismara, > "Turn-Regularity and Optimal Area Drawings of Orthogonal Representations," > Computational Geometry: Theory and Applications, 16(1):53-93, 2000.

Textbook background: Di Battista, Eades, Tamassia, Tollis, *Graph Drawing: Algorithms for the Visualization of Graphs* (1999), Ch. 5 (orthogonal drawings, compaction). Working code reference: OGDF's `CompactionModule` / `LongestPathCompaction` / `FlowCompaction`.

### Definitions

For each face, walk its boundary (interior on the left; bounded faces turn +4, outer face -4). Every corner -- at a vertex *and* at a bend -- has a turn:

- `+1` convex (interior angle 90 degrees, a left turn),

- `0`  flat (180 degrees),

- `-1` reflex (270 degrees, a right turn).

For two corners `c`, `d` on the same face, `rot(c, d)` is the sum of turns walking the boundary from `c` to `d`. Two reflex corners `c`, `d` are **kitty corners** iff `rot(c, d) == 2` (equivalently `rot(d, c) == 2`). A face is **turn-regular** iff it has no kitty-corner pair. A turn-regular representation admits a planar, area-optimal drawing directly from two acyclic constraint graphs.

### Algorithm

1. **Detect kitty corners** per face in linear time. The standard method sweeps the face boundary twice maintaining two stacks keyed by a running rotation value; a corner pops matching partners off the stack, and reflex-reflex matches with `rot == 2` are the kitty corners. (See the paper's Fig. for the two "increasing"/"decreasing" passes.)

2. **Saturate**: for each kitty-corner pair, add one **saturating edge** -- an axis-parallel artificial edge between the two reflex corners -- choosing its direction (horizontal or vertical) so the face is split and no new kitty corner is introduced. The saturating edges carry no ink; they are pure separation constraints. After saturation every face is turn-regular (rectangular in effect).

3. **Compaction**: build the horizontal constraint graph `Gh` and vertical constraint graph `Gv`. Nodes are the maximal segments (or vertices/bends); arcs are the real edges' direction constraints **plus the saturating edges**. For a turn-regular representation both graphs are acyclic; longest path (minimum unit length) gives integer coordinates and a planar drawing. Optional: replace longest-path with a min-cost flow for minimum total edge length (area-optimal), per the paper.

The current `_assign_axis` longest-path solver is already the compaction step; the missing piece is generating the correct extra arcs (step 2) from kitty-corner detection (step 1) instead of the naive all-pairs rule.

## Integration points (this codebase)

- `orthogonal/metrics.py`

  - Have `compute_orthogonal_shape` (or a new helper) expose, per face, the ordered corner sequence with turn values (it already computes segment directions; corners/turns are derivable).

  - New `_kitty_corners(faces, shape) -> list[pair]` implementing step 1.

  - New saturation producing extra ordering constraints per axis (step 2).

  - `_assign_axis` already accepts the constraint set; feed it the real + saturating arcs (extend it to take extra `(class_a, class_b)` arcs -- the prototype signature is in the git history of the failed attempt).

  - Keep the existing `_drawing_conflict` gate as a belt-and-suspenders check.

- `orthogonal/giotto.py`: once coverage is ~100% and verified, flip `bend_optimal` default to `True` and update the tests that assume the heuristic (e.g. `test_bend_optimal_defaults_off`, the `<= 2 bends per edge` assertion in `test_kandinsky.py`).

## Verification oracle (already built)

- `tests/test_orthogonal_metrics.py::face_turn_sum` -- representation validity (`+/-4` per face).

- `_drawing_conflict` in `metrics.py` -- detects overlaps / crossings / edges through a vertex; a correct rectangularization must yield **zero** conflicts.

- Random generator harness (see the prototype scripts / `test_orthogonal_metrics` grid helpers): sample biconnected max-degree-4 planar graphs (grid subgraphs, Apollonian-style, etc.), draw each, assert `_drawing_conflict is None`. Target: 100% clean. This is the exact harness used to measure the 83% -> 89% gain and to catch the naive attempt's regression.

## Definition of done

1. `compute_coordinates` (or a `bend_optimal` path) draws every biconnected max-degree-4 planar graph in the random harness with **zero** conflicts.

2. Drawings remain axis-aligned with distinct vertices (existing assertions).

3. `bend_optimal` can be turned on by default; heuristic-specific tests updated.

4. No regression in the ~1080-test suite.

## Scope / effort

This is a self-contained but non-trivial algorithm (kitty-corner detection + saturation are the subtle parts; compaction reuses existing machinery). Estimate: a dedicated session. Out of scope here (separate items): **H5** (Kandinsky 0-degree angles for degree > 4) and **H6a** (per-corner angles for non-biconnected graphs), which extend the *domain* rather than the compaction.
