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

### Added

- **MkDocs documentation site** (`mkdocs.yml`, `scripts/mkdocs_hooks.py`, `docs/`): a Material-themed site built with `make docs`, served with `make docs-serve`, and published to the `gh-pages` branch by `make docs-deploy`. Publishing is manual; `.github/workflows/docs.yml` holds a strict build gated on `workflow_dispatch`, since GitHub Pages serves one source and an Actions-artifact publish would fight the branch. Figures are not checked-in images. A fenced block tagged ```` ```graph-layout ```` is executed against the installed library at build time and replaced with the resulting inline SVG, so an example that stops matching the API fails the build instead of leaving a stale picture in place. The fence takes `title=`, `class=`, `seed=`, `source=`, and any keyword the renderer accepts; the block binds `layout`, `svg`, or `boxes`/`edges`. Every figure carries the code that drew it -- a block beneath it, or a collapsed `<details>` inside it for `inline` figures in a comparison row -- and a test fails if any page suppresses it. Blocks quoted inside a longer fence are left as text rather than executed, so the page documenting the syntax can show it. Labels default to `fill="currentColor"` so they follow the light and dark themes. A block may bind `caption` to report a measured quantity -- crossings, stress, bend count -- that only exists after the layout runs, and `class=inline` flows figures side by side for comparisons. The gallery covers all 22 layouts across 36 figures, including the bend-count difference between the default Kandinsky router and `bend_optimal=True` (12 bends against 1) and Kuratowski witnesses drawn over non-planar graphs. `MixedModelLayout` and `PlanarizationLayout` are drawn from `vertex_bars`/`edge_routes` rather than `to_svg()`, which joins node centres with straight lines and so invents crossings neither layout has. MkDocs is in a separate `docs` dependency group, keeping it out of the test environment (`tests/test_mkdocs_hooks.py`).

````markdown
```graph-layout title="8-cycle" source node_radius=14
layout = CircularLayout(
    nodes=[{"index": i} for i in range(8)],
    links=[{"source": i, "target": (i + 1) % 8} for i in range(8)],
    size=(300, 300),
).run()
```
````

## [0.5.0]

Two changes alter behaviour for existing callers: `metrics.stress` now takes `links` as its second positional parameter, and `KamadaKawaiLayout` starts from a circle instead of a random placement, so its output differs. Both are described under Changed, and both have an escape hatch -- pass `ideal_distances=` by keyword, and `initial_layout="random"`.

### Fixed

- **ForceAtlas2 computed a different force model with and without the Cython extension** (`spatial/quadtree.py`, `force/force_atlas2.py`): `QuadTree.from_nodes` hard-coded `mass=1.0` for every body it inserted. That is correct for the uniform-repulsion layouts (Fruchterman-Reingold, Spring, Yifan Hu), whose Cython kernel also inserts `1.0`, but ForceAtlas2's repulsion is degree-weighted -- `scaling * (deg_i + 1) * (deg_j + 1) / d`. The Cython kernel inserted `degrees[i] + 1.0` as the tree mass and so applied both factors; the pure-Python path went through `from_nodes` and silently dropped the source-side `(deg_j + 1)`, keeping only the acting body's factor. Degree-weighted repulsion is what distinguishes ForceAtlas2, so the fallback degenerated toward uniform repulsion. Barnes-Hut is on by default above 50 nodes, so this was the normal path for any graph the algorithm targets: a user on a platform without a wheel, or hitting the `ImportError` fallback, got a quantitatively different and worse drawing with no warning (stress 0.818 against the compiled path's 0.558 on a 10x10 grid). `from_nodes` now takes an optional `masses` sequence and ForceAtlas2 passes `degree + 1`; the two paths agree to floating-point noise (5.7e-14 at iteration 1, against a 7.9-unit divergence before). The in-code comment asserting parity was corrected -- it claimed the acting body's mass was the only missing factor. Nothing exercised the pure-Python path, which is how this survived; parity tests now force `_HAS_CYTHON = False` for all three accelerated layouts (`tests/test_force_atlas2.py`, `tests/test_force_layouts.py`, `tests/test_yifan_hu.py`).

- **Yifan Hu's convergence test stopped the finest refinement levels after a single iteration** (`force/yifan_hu.py`): the check compared `movement`, the L2 norm of the whole displacement vector, against `convergence_tolerance * k * n`. The norm grows like `sqrt(n)` while the threshold grows like `n`, so the test became progressively easier to satisfy as a level refined -- exactly backwards. On a 30x30 grid the last four levels each ran 1 of their 50 iterations, making the multilevel refinement a no-op above a few hundred nodes and leaving drawings barely better than random (stress 0.68 against SMACOF's 0.015 on a 12x12 grid). The criterion is now intensive: the per-vertex root-mean-square displacement (`movement / sqrt(n)`) against `tol * k`, both lengths and so directly comparable. Every level now runs a comparable number of iterations, and grid stress fell from 0.507-0.594 to 0.275-0.330 over 20 seeds at 10x10, 0.850-0.871 to 0.670-0.691 at 20x20 (`tests/test_yifan_hu.py::TestYifanHuConvergence`).

- **Yifan Hu's multilevel refinement made drawings worse than not coarsening at all** (`force/yifan_hu.py`): exposed once the convergence fix let the levels actually run. Two causes in `_run_multilevel`. It shrank `k` at every refinement (`base_k / sqrt(level_n / coarser_n)`), so even the finest level -- the one producing the drawing the caller sees -- was laid out at roughly 0.85 of the graph's true optimal distance rather than at `base_k`. And refinement used plain geometric cooling, which caps a level's *total* displacement at `step_0 / (1 - t) == k`: a vertex could never travel further than one optimal distance while unfolding a level, however large that level was. Hu's paper uses adaptive step control at every level. A third, smaller issue: the prolongation jitter was a fixed `gauss(0, 1)` in absolute coordinate units, meaningless at an arbitrary graph scale -- with `k ~= 60` it left vertices sharing a coarse parent effectively coincident -- and is now proportional to `k`. Stress improved on every family measured (median of 5 seeds): 12x12 grid 0.680 to 0.090, 30x30 grid 0.930 to 0.240, binary tree 0.660 to 0.198, path 0.955 to 0.548, sparse 0.871 to 0.198, clustered 0.778 to 0.099. A variant that also expands the prolongated positions by `sqrt(level_n / coarser_n)` was measured and rejected: markedly better on large meshes (25x25 grid 0.153 to 0.057) but worse on trees (0.372 to 0.451) and clustered graphs (0.099 to 0.393). On a 12x12 grid Yifan Hu now draws zero crossings at stress 0.078, from 346 crossings at 0.684 (`tests/test_yifan_hu.py::TestYifanHuMultilevel`).

- **`YifanHuLayout` was not reproducible despite `random_seed`** (`force/yifan_hu.py`): the multilevel path drew from the process-global `random` module -- `random.shuffle` for the coarsening order, `random.uniform` for the coarsest-level placement, `random.gauss` for the prolongation jitter. `random_seed` seeded that module once via `_initialize_positions`, but the multilevel path performs its own initialization after an unbounded, graph-dependent number of intervening draws, so two identical runs in one process produced different coordinates and results depended on the caller's global RNG state. It was the only layout of the 17 tested that failed a repeatability check. The existing `test_random_seed` missed it by using a three-node graph, which takes the single-level branch and never enters `_run_multilevel` (`tests/test_yifan_hu.py::TestYifanHuMultilevelReproducibility`).

- **`BaseLayout` seeded the process-global RNG** (`base.py`): `_initialize_positions` called `random.seed(self._random_seed)` and drew from the `random` module, so running a layout clobbered the caller's global random stream as a side effect, and two layouts running concurrently interleaved draws from one shared generator -- neither reproducible. `BaseLayout` now owns a `random.Random` instance, exposed as the `rng` property and restarted from `random_seed` at the start of every run (in `_initialize_indices`, the one point every `run()` already passes through). All layouts are now both repeatable and isolated in each direction. `cola/layout3d.py` still calls `random.random()` for initial positions; `Layout3D` exposes no seed parameter, so giving it one is an API addition rather than a fix.

- **`PriorityQueue.is_heap` rejected any heap containing equal keys** (`cola/pqueue.py`): it asserted `less_than(parent, child)`, demanding a *strict* ordering between a node and its children. With the natural `a < b` comparator that reports any heap holding a repeated key as invalid; the smallest failing case is the two-element heap `[3, 3]`, and roughly 8% of random 100-element datasets drawn from 1..1000 contain enough duplicates to trip it. The invariant is that no child is strictly less than its parent, so the test is now `less_than(child, parent)`. Scope was limited and is worth stating precisely: `is_heap` is a diagnostic called nowhere in the library, and the heap structure itself was never affected -- pop order and count were correct across 3000 duplicate-heavy datasets -- so this was a wrong answer from a public method and a source of spurious test failures, not a defect in Dijkstra or any layout (`tests/test_pqueue.py::TestDuplicateKeys`).

- **`BaseLayout.validate()` raised `TypeError` on links holding unindexed `Node` objects** (`base.py`, `validation.py`): `validation._get_index` called `int(val.index)` without checking for `None`, and `Node.index` is `None` until assigned. Links may reference `Node` objects and `validate()` is documented as the early fail-fast call, so those two documented behaviours combined into an unhandled exception. `validate()` now assigns indices first (via a new `_assign_indices`, split out of `_initialize_indices` so validation does not also restart the RNG), `_get_index` and `_get_index_simple` no longer call `int()` on `None`, and the resulting diagnostic reads "source is missing or not a resolvable node index" rather than the misleading "source is None" (`tests/test_validation.py::TestValidateOnLayouts`).

- **`Node`, `Link` and `Group` silently dropped keyword arguments naming a pre-set attribute** (`types.py`): each copied "additional" kwargs only `if not hasattr(self, key)`. Because `px`, `py` and `parent` on `Node` (and `index` and `parent` on `Group`) were assigned unconditionally beforehand, passing them was a no-op: `Node(px=5.0)` produced `px=None` with no error. Each class now tracks the keyword names it consumes instead of probing `hasattr` (`tests/test_validation.py::TestConstructorKeywordPassthrough`).

- **The Cython extension was built with fused-multiply-add contraction, so the compiled and pure-Python force paths no longer matched bit for bit** (`CMakeLists.txt`): clang on arm64 macOS contracts `dx * dx + dy * dy` into a single FMA by default, which rounds once where CPython's float arithmetic rounds twice. The resulting one-ulp difference is invisible in a single force, but force-directed dynamics are chaotic and amplify it: through Yifan Hu's five-level refinement on a 10x10 grid it grew from 1e-15 to 4e-11 at the coarsest level, 1e-6, 0.3, 1 and finally 19 units at the finest -- a visibly different drawing. x86-64 Linux and Windows do not contract by default, so `TestYifanHuFallbackParity` failed only on the macOS runner. The extension is now compiled with `-ffp-contract=off` (`/fp:precise` on MSVC); the kernels are bit-identical again, at no measurable cost (the inner loop is bound by `sqrt` and division, not the two multiplies). `tests/test_cython_parity.py::TestBuildHygiene` asserts the bit-exactness directly and names the flag when it fails.

- **The pure-Python force fallbacks disagreed with the compiled kernels whenever two nodes nearly coincided** (`force/yifan_hu.py`, `force/fruchterman_reingold.py`, `force/force_atlas2.py`): the two implementations guarded degenerate separations differently, so they only agreed on input that no ordinary graph produces but a randomly initialised or multilevel-prolongated one occasionally does. The compiled kernels floor the *squared* distance at `1e-10`, capping the repulsive force; the fallbacks tested `dist_sq > 0` and substituted a distance of `0.0001` only when it was exactly zero, so a pair separated by less than `1e-5` divided by its true distance and took a step large enough to throw a node clear out of the drawing. Attraction differed the other way: the kernels skip an edge below the floor, the fallbacks computed a force. ForceAtlas2 diverged furthest -- its fallback gave exactly coincident nodes an arbitrary `+x` kick where the kernel gives zero, and floored the overlap-adjusted distance at `0` rather than the kernel's `0.01`, so deeply overlapping nodes repelled with an unbounded force. All three fallbacks now use the kernels' guards, and the floors live in one place (`force/_kernel_constants.py`) that `_speedups.pyx` cross-references. Only the fallback path changes; the compiled path, which is what a wheel install runs, is untouched.

### Changed

- **`KamadaKawaiLayout` starts from a circle rather than uniformly at random** (`force/kamada_kawai.py`): Kamada and Kawai (1989) place the vertices on a circle, and the minimisation is local -- where it starts decides which optimum it reaches. From uniformly random starts the layout folded on roughly one run in ten: on the three-node path `0 -- 1 -- 2` it put node 2 between 0 and 1, giving a 0.5 distance ratio instead of 2.0 (samples of 200 runs measured 13 and 21 failures). A circular start reached the correct drawing on 200 of 200 runs, is deterministic, and had the better *worst case* on every family measured -- path n=60: 0.130 stress against 0.438; cycle n=60: 0.059 against 0.435; and better maxima on grids and trees too. Random starts do score a slightly better *median* on grids, so the old behaviour remains available through a new `initial_layout` parameter (`"circular"` default, `"random"` to restore it). The trade taken is predictability: a better worst case everywhere, and no 10% failure mode (`tests/test_force_layouts.py::TestKamadaKawaiInitialLayout`).

- **`metrics.stress` takes `links` as its second positional parameter** (`metrics.py`): every other metric in the module is `f(nodes, links)`, but `stress` took `ideal_distances` there, so the natural call `stress(nodes, links)` treated the links as a distance matrix and failed deep in the loop with `'Link' object is not subscriptable`. The signature is now `stress(nodes, links=None, ideal_distances=None, edge_length=100.0)`. Every caller in the repository already passed `links=` by keyword, so nothing in-tree changed behaviour; an external caller passing a matrix positionally now gets a `TypeError` naming `ideal_distances=` instead of silently misbehaving (`tests/test_metrics.py::TestStressSignature`).

- **Layout warnings are attributed to the caller** (new `_warnings.py`, `hierarchical/radial_tree.py`, `hierarchical/reingold_tilford.py`, `hierarchical/sugiyama.py`): the tree layouts warn when handed a graph that is not a tree, but the literal `stacklevel` values counted library frames by hand, differed per call path (2, 3 and 4 were all in use), and had drifted -- warnings were landing on `base.py:56` and `base.py:732` rather than the line that called `run()`, so `warnings` filters keyed on the caller's module did not match. A new `warn_at_caller` helper counts the frames at runtime instead. One limitation is documented in the helper: when a tree layout recurses on a worker thread via `run_deep_recursive`, the caller's frame is not on that stack at all, so no `stacklevel` can reach it. Nothing previously asserted on these warnings at all (`tests/test_warnings.py`).

- **`BaseLayout._center_graph` no longer reaches into orthogonal-layout internals** (`base.py`, `orthogonal/giotto.py`, `orthogonal/kandinsky.py`): it probed for `_node_boxes` and `_orthogonal_edges` with `getattr` and imported `NodeBox` from the `orthogonal` subpackage inside the method body to dodge a circular import -- the abstract base knowing one concrete subpackage's private attributes, with the deferred import as the tell. It now calls a `_translate_extra(dx, dy)` hook that the two orthogonal layouts override.

- **`validate()` documents that it is not called by `run()`** (`base.py`): the docstring claimed "Called automatically by run()", which was never true -- no `run()` implementation called it, and `_build_adjacency` silently skips out-of-range indices, so a link naming a nonexistent node vanished from the drawing with no diagnostic. Wiring validation into `run()` was tried and reverted: `tests/test_circular_layouts.py::test_out_of_range_link_indices` is an explicit regression test asserting that such a link must *not* raise, recording the earlier `IndexError` that motivated the leniency. Leniency is a deliberate, pinned design decision, so the docstring was the thing that was wrong. It now states plainly that invalid links are skipped and shows how to opt in (`layout.validate().run()`).

- **README install claim narrowed to what the wheel matrix delivers** (`README.md`): it said the library "installs anywhere with `pip` -- no native compiler or build toolchain required", but the build backend unconditionally compiles the Cython extension and the wheel matrix skips `*-win32`, `*-manylinux_i686` and `*-musllinux_*`. Where no wheel matches, pip falls back to the sdist and the build *does* require a C compiler and CMake. The runtime fallbacks are genuine and unchanged; the sentence now says which platforms get wheels and that an sdist build needs a toolchain.

- **The Cython/pure-Python parity tests compare a bounded run instead of a full one** (`tests/test_cython_parity.py`, `tests/test_force_atlas2.py`, `tests/test_force_layouts.py`, `tests/test_yifan_hu.py`): comparing the output of a full-length layout measures how fast the dynamics amplify rounding, not whether the two implementations agree, and the answer depends on the compiler and the CPU. The amplification is steep -- Fruchterman-Reingold multiplies the Barnes-Hut starting gap by roughly 100 per iteration on a 10x10 grid (1e-13 at one iteration, 6e-6 at ten, 0.7 at twenty), so the existing test at ten iterations sat within a factor of 16 of its own threshold. Each layout-level parity test now runs a few iterations, which still catches a wrong force model immediately (the ForceAtlas2 Barnes-Hut bug fixed in 0.5.0 moved nodes ~8 units on the first iteration). The unamplified comparison moved to a new `tests/test_cython_parity.py`, which checks the force fields themselves -- every shared, ForceAtlas2 and Barnes-Hut kernel against its fallback, on ordinary, coincident and near-coincident positions -- and so gives the same answer on every platform.

### Added

- **CI that runs the test suite** (`.github/workflows/test.yml`): nothing previously ran on a push or pull request -- the only workflow was `build-wheels.yml`, gated on `workflow_dispatch`. For a library whose correctness argument rests on its test suite, that was the largest single gap. The new workflow runs `pytest` across Python 3.9/3.10/3.13/3.14 on Linux with macOS and Windows spot-checks, a quality job (`ruff format --check`, `ruff check`, `mypy --strict`), and an `oracles` job. The wheel job's in-container tests are also weaker than they look: `CIBW_TEST_REQUIRES` lists neither `networkx` nor `ogdf-py`, so both differential-oracle suites skip there silently.

- **The differential oracles are now required to run somewhere** (`.github/workflows/test.yml`): 445 of the 446 skipped tests were `tests/test_ogdf_oracle.py`. The dev pin is `python_version < '3.14'` while the project's own venv is 3.14, so the strongest correctness check in the repository -- an independent C++ implementation cross-checking a quarter of the collected tests -- was dark both locally and in CI. The `oracles` job pins Python 3.13 and *asserts both oracles are importable* before running them, so the suites fail rather than silently skip.

- **A ratchet on the type errors `pyproject.toml` suppresses** (`mypy-legacy.ini`, `Makefile`): `ignore_errors = true` hides eleven ported-from-JavaScript `cola` modules completely, so `make typecheck` reported success over code it never checked and the suppressed count grew unwatched from a documented 146 to 158. `mypy-legacy.ini` mirrors the project's own settings minus the suppression, and `make typecheck-legacy` -- now part of `make typecheck`, and a CI step -- fails if the count rises above `MYPY_LEGACY_BASELINE`. The suppression can shrink but not grow. An error-code allowlist was considered first and rejected: the errors span 17 distinct codes, so an allowlist of all 17 would constrain nothing. Separately, `graph_layout.cola.adapter` appeared in *both* the strict override list and the `ignore_errors` list; the later section wins in mypy, so it was silently unchecked despite being declared strict. It is removed from the suppression, and the one error that surfaced is fixed properly by annotating `Layout.__init__` rather than suppressed.

- **`scipy` in the dev dependency group** (`pyproject.toml`): it backed the optional `[ilp]` extra but was absent from the dev environment, so the ILP compaction path -- advertised as the route to optimal Kandinsky area minimization -- sat at 15% coverage and the scipy-dependent embedder tests skipped. The tests already existed; they were merely never running. Coverage of `orthogonal/compaction_ilp.py` went from 15% to 88% and `planarity/embedders.py` from 55% to 95%.

- **Cython / pure-Python parity tests** (`tests/test_force_atlas2.py`, `tests/test_force_layouts.py`, `tests/test_yifan_hu.py`): nothing forced `_HAS_CYTHON = False`, so every local and CI run exercised only the compiled path -- which is how the ForceAtlas2 divergence above went unnoticed. All three accelerated layouts now assert the two paths agree, ForceAtlas2 across both `use_barnes_hut` settings. Without Barnes-Hut they agree exactly; with it they differ only by float accumulation order (~6e-6 on the test graph).

- **Tests for warning emission and attribution** (`tests/test_warnings.py`): no test anywhere asserted on the tree-structure warnings -- running the hierarchical suite under `-W error::UserWarning` passed clean -- so neither the messages nor their attribution were covered.

- **`REVIEW.md`**: a full review of the codebase recording all nineteen findings, the evidence for each, and how each was resolved or explicitly closed.

## [0.4.1]

### Added

- **OGDF differential-testing oracle** (`tests/test_ogdf_oracle.py`, `tests/_ogdf_oracle.py`): a second, independent reference-implementation check complementing the existing `networkx` planarity oracle, backed by `ogdf-py` (Python bindings to the C++ Open Graph Drawing Framework, whose Boyer-Myrvold planarity test is algorithmically unrelated to graph-layout's Left-Right implementation, so a blind spot shared between graph-layout and networkx would still be caught). It cross-checks three things over random and adversarial graphs: (1) `is_planar` agreement; (2) planar-layout flag fidelity -- the `used_schnyder` / `used_fpp` / `used_tutte` / `used_mixed_model` flags never claim a crossing-free drawing of a graph OGDF reports non-planar, and for the exact integer-grid methods (Schnyder, FPP) the drawing has zero crossings (the check is scoped away from Tutte and the mixed model, where `edge_crossings` on node centres is not a valid oracle -- near-degenerate collinearity and port-attached edges respectively); (3) the connectivity ladder -- `is_connected`, connected-component count, biconnected-component count, and derived biconnectivity against the released bindings, plus cut-vertex set equality (validating `planarity/_block_cut_tree.py` against an independent articulation-point implementation) and SPQR / separation-pair consistency anchoring the triconnected tier against builds that expose them. `ogdf-py` is a dev-only dependency, pinned behind an environment marker matching its prebuilt-wheel coverage (CPython 3.10-3.13 on Linux/macOS); every check skips cleanly where it, or a given function, is absent. A `make oracle-install` target installs a local `ogdf-py` build for the finer checks.

- **Exhaustive-enumeration oracle for connectivity and planarity** (`tests/test_exhaustive_oracle.py`): where the differential oracle samples random graphs, this enumerates the entire small-graph universe -- every labeled simple graph on up to 6 vertices (`2^15 = 32,768` at n=6, all labelings, so order-dependent bugs are exercised) plus all 1,253 non-isomorphic graphs up to 7 vertices (the networkx Read-Wilson atlas). graph-layout's `is_connected`, connected-component count, and block-cut-tree articulation points are checked against a self-contained brute-force reference (components by flood fill; a cut vertex defined directly as "removing it raises the component count"), and `is_planar` against `networkx.check_planarity` plus, when installed, OGDF's independent `is_planar`. This turns "agrees on a random sample" into an exhaustive guarantee at those sizes for the combinatorial properties; the sweep is non-vacuous (at n=6 it exercises 697 non-planar graphs, 19,506 with articulation points, and 6,064 disconnected graphs).

- **OGDF layout benchmark harness** (`tests/benchmarks/compare_ogdf.py`, `make bench-ogdf`): compares graph-layout's layouts against the equivalent OGDF C++ layouts on the shared benchmark graphs, grouped by algorithm family (stress/MDS, force-directed) so the numbers line up directly. Reports wall-clock time and scale-invariant normalized stress -- each drawing is optimally rescaled before scoring, so quality is compared independent of each engine's coordinate scale (sampled over 200k node pairs above that size, seeded so every engine is scored on identical pairs), computed in two passes for O(1) memory in the pair count -- plus edge crossings on small graphs. Both engines' outputs are scored by the same functions, so comparisons are apples-to-apples; non-finite output (e.g. OGDF PivotMDS / Kamada-Kawai emitting NaN on disconnected graphs) and prohibitively slow pure-Python layouts (Kamada-Kawai, capped at n<=200) are reported explicitly rather than silently dropped. A persisted results table across 100-5000 node scale-free graphs is documented in `tests/benchmarks/README.md`: on the identical stress-majorization algorithm OGDF's C++ runs ~15-20x faster at matched quality, while graph-layout's multilevel `YifanHu` is faster than OGDF's `FMMM` at every size.

### Changed

- **Moved `benchmarks/` to `tests/benchmarks/`**: the benchmark graphs, the `compare_ogdf.py` harness, and its README now live under `tests/`. The `scripts/benchmark_layouts.py` and `scripts/generate_benchmark_graphs.py` graph-directory paths, the `make bench-ogdf` target, and the README loading examples were updated accordingly; the harness is not a `test_*.py` file, so pytest does not collect it.

## [0.4.0]

### Added

- **`MixedModelLayout`: visibility-representation (mixed-model) layout** (`planar/mixed_model.py`): draws a connected planar graph with each vertex a horizontal bar and each edge a bendless vertical segment attaching at a distinct port, so high-degree vertices spread their edges out for good angular resolution. It builds a Tamassia-Tollis visibility representation from two topological numberings: the canonical ordering supplies each vertex's st-number, giving a `y`-coordinate as the longest path from the source, and a longest path in the dual of the st-oriented embedded graph gives every edge (and the span of every bar) an `x`-coordinate. A barycentric refinement then places each node's point at the mean of its ports, centring the marker over its edges without moving any bar or edge (so the drawing stays crossing-free). This is the "visibility representation + refinement" design noted in `TODO.md` -- the visibility half of Kant's mixed model (Kant 1996; Tamassia-Tollis 1986). After `run()` the layout exposes `vertex_bars` (each node's `(x_left, x_right, y)` bar) and `edge_routes` (each edge's two-point vertical polyline), with `used_mixed_model` reporting whether the method ran (non-planar/disconnected input falls back to a circle). Exposed as `graph_layout.MixedModelLayout`; validated by a geometric oracle (edges stay within their bars, no edge pierces a non-incident bar, independent edges never overlap, same-row bars never overlap) over grids, wheels, trees, and random triangulations, plus Hypothesis fuzzing, with a showcase renderer drawing the bars and vertical edges (`tests/test_planar_straightline.py`).

- **`PlanarizationLayout`: straight-line drawing of non-planar graphs** (`planar/planarization.py`): the three planar straight-line layouts fall back to a circle for non-planar input; this layout draws it. It reuses the existing topological planarizer (`orthogonal/planarization.py`) to embed a maximal planar subgraph and reinsert the remaining edges along minimum-crossing paths, turning each crossing into a degree-four dummy vertex; the resulting planar graph is drawn straight-line (FPP by default, or Schnyder via `method="schnyder"`), placing every real and dummy vertex on a grid point; and each original edge is rendered as the polyline through the dummy vertices it was routed across. Because the planarized graph itself is drawn crossing-free, original edges meet only at the explicit crossing points. After `run()` the layout exposes `crossings` (crossing-point coordinates), `crossing_count`, and `edge_routes` (per-link polyline points), with `used_planarization` reporting whether the method ran. A genuinely planar graph gains no dummy vertices and is drawn straight-line. Exposed as `graph_layout.PlanarizationLayout`; validated on K5/K6/K7/K3,3/K4,4/Petersen (crossing counts match the planarizer) and confirmed to draw the augmented graph crossing-free (`tests/test_planar_straightline.py`), with a showcase renderer marking crossings as dots.

- **Planar straight-line drawing layouts: `SchnyderLayout`, `FPPLayout`, `TutteLayout`** (`planar/`): a new algorithm category that draws a connected planar graph with straight-line edges and no crossings. All three share one substrate (`planar/_shared.py`): a combinatorial planar embedding from the LR-planarity test, ear-clipping triangulation to a maximal planar graph (tracking the added chords, which are used only for positioning and never drawn), and a de Fraysseix-Pach-Pollack canonical (shelling) ordering. `SchnyderLayout` builds the realizer -- a decomposition of the triangulation into three edge-disjoint trees -- and places each interior vertex by counting the vertices in the three regions its tree-paths cut the drawing into, giving barycentric coordinates `(r1, r2, r3)` with `r1 + r2 + r3 = n - 1` on the `(n-1) x (n-1)` integer grid (Schnyder 1990). (Vertex counting is about twice as compact as counting faces; Schnyder's classical `(n-2)` optimum is one unit tighter but requires a boundary tie-breaking that permits controlled collinearity at the outer edges, so the strictly non-degenerate `n-1` placement is used.) `FPPLayout` installs vertices in canonical order, each as the apex of a slope-(+/-1) "tent" over a contiguous run of the current outer contour after shifting the contour apart, landing on the `(2n-4) x (n-2)` grid (de Fraysseix-Pach-Pollack 1990). `TutteLayout` nails the largest face to a convex polygon and solves the barycentric equilibrium (a Dirichlet-Laplacian linear system) for the remaining vertices, giving convex faces for 3-connected planar graphs (Tutte 1963). Each layout draws any connected planar simple graph of at least three vertices and scales the result onto the canvas; non-planar, disconnected, or trivially small inputs fall back to a deterministic circular placement, with `used_schnyder` / `used_fpp` / `used_tutte` reporting which path ran. Exposed as `graph_layout.SchnyderLayout`, `graph_layout.FPPLayout`, and `graph_layout.TutteLayout`. Correctness is pinned by a brute-force crossing-free oracle (every pair of non-adjacent edges checked for intersection) over triangles, grids, wheels, trees, and random triangulations, plus grid-bound, distinctness, determinism, and Hypothesis fuzzing tests (`tests/test_planar_straightline.py`).

- **`SMACOFLayout`: stress-majorization force-directed layout** (`force/smacof.py`): a new layout that minimizes the same graph-drawing stress as Kamada-Kawai -- Euclidean distances matching graph-theoretic (shortest-path) distances -- but optimizes it by majorization rather than per-node Newton-Raphson. Each iteration replaces the stress with a quadratic upper bound touching it at the current layout and moves every node at once to that bound's global minimum (the Guttman transform `X <- V^+ B(X) X`, with standard `w_ij = d_ij^-2` weights and a pseudo-inverse of the weighted Laplacian). Because each step is the exact minimizer of a majorizing quadratic, the stress decreases monotonically, so it converges more reliably and in fewer iterations than gradient descent (Gansner, Koren & North 2004). Exposed as `graph_layout.SMACOFLayout`, with the same `edge_length` / `epsilon` / `disconnected_distance` knobs as `KamadaKawaiLayout`; fixed nodes are held in place after each transform (`tests/test_smacof.py`, including a monotonic-stress-decrease property test).

- **Disconnected-graph bend-optimal drawing via per-component Topology-Shape-Metrics + component packing** (`orthogonal/realization.py`, `orthogonal/giotto.py`, `orthogonal/kandinsky.py`): a planar embedding (and therefore the shared TSM realizer) is only defined per connected component, so the bend-optimal path previously fell back to the heuristic router for any disconnected input -- the last remaining "silently fall back" case noted in 0.3.1/0.3.2 for connected planar graphs. `GIOTTOLayout` and `KandinskyLayout(bend_optimal=True)` now split a disconnected graph into components, draw each one bend-optimally in its own coordinate frame (recursively, so each component reuses the full pipeline -- cage expansion, per-corner angles, and, for Kandinsky, planarization of non-planar components), and reassemble the drawings with a new shared shelf-packer (`pack_component_drawings`) that lays the components out roughly square with separation gaps so their bounding boxes -- taken over node boxes *and* edge bends -- never overlap, then centers the result on the canvas. Isolated vertices are placed directly (zero bends). If any non-trivial component is outside the bend-optimal domain the whole graph falls back to the heuristic router, keeping `used_bend_optimal` truthful. Disconnected planar graphs (and, for Kandinsky, disconnected graphs with non-planar components such as a K5 island) now draw bend-optimally with no cross-component overlaps (`tests/test_orthogonal_disconnected.py`).

- **Property-based tests for the orthogonal layouts** (`tests/test_orthogonal_properties.py`, Hypothesis added to the dev dependencies): the bend-optimal path fans out across many branches (biconnected vs. not, degree <= 4 vs. cage expansion, planar vs. planarized, connected vs. packed components) that example-based tests can only sample. New Hypothesis strategies generate structurally-varied planar graphs (random trees, grids, forests) and assert the invariants that must hold for every orthogonal drawing -- every original vertex has a box, every edge is drawn with axis-aligned segments only, and whenever the bend-minimal path actually drove the drawing the node boxes do not overlap and the result is deterministic. A fuzz test feeds arbitrary small graphs through both `GIOTTOLayout` and `KandinskyLayout` to confirm they never raise and always emit orthogonal edges.

## [0.3.2]

### Added

- **`KandinskyLayout(bend_optimal=True)` draws non-planar graphs bend-optimally through their crossings** (`orthogonal/realization.py`, `orthogonal/kandinsky.py`): the opt-in bend-optimal path previously fell back to the heuristic router whenever the graph was non-planar (crossing dummies present) -- exactly Kandinsky's distinguishing case. It now realizes the *planarized* graph directly: the augmented graph (crossings replaced by degree-4 dummy vertices) is planar, so the Topology-Shape-Metrics pipeline draws it, and each original edge's polyline is reassembled by walking its augmented segments through the crossing dummies, whose grid points become the edge's bend points. Both edges of a crossing therefore pass through one shared point -- a clean orthogonal crossing. The crossing is straight-through for free: a degree-4 dummy has flow supply `4-4=0`, forcing all four corners to exactly 90 degrees, so an alternating rotation (which the embedder produces, verified on K5/K6/K7/K4,4/Petersen) makes each edge pass straight; a per-dummy alternation check guards the rare exception. New `realize_planarized_drawing` returns `None` (safe fallback) outside scope -- a non-planar graph that *also* has an original vertex of degree > 4 (would additionally need cage expansion). Non-planar graphs like K5, K3,3 and the Petersen graph now draw with the minimum bends and each crossing rendered as an orthogonal X (`tests/test_kandinsky_bend_optimal.py::TestBendOptimalThroughCrossings`).

### Fixed

- **Showcase cross-cut renderer detached edges from expanded cage boxes** (`tests/demos/showcase.py`): `orthogonal_layout_to_svg` (the renderer for the Kandinsky OptimalFlex / nudging / bend-optimal cards) drew every node box at a single uniform size, but a bend-optimal drawing with a degree > 4 vertex has a large cage box. Edges attach to ports on the true (large) cage boundary, so drawing the box small left the edges floating in empty space -- a valid, connected drawing looked broken. It now draws each box at its actual width/height (matching the earlier fix to the main `layout_to_svg`), so cage-bearing bend-optimal cards render as coherent connected drawings. Not a layout change: the wide cages themselves are the bend-minimal result (each spoke leaves its cage vertex straight, so the cage must span its spread-out neighbours; verified bend-count-identical to GIOTTO over 137 random degree > 4 planar graphs) -- see the area-vs-bends note in `TODO.md`.

## [0.3.1]

### Added

- **Bend-optimal drawing of non-biconnected graphs (H6a)** (`orthogonal/orthogonalization.py`, `orthogonal/metrics.py`): the Topology-Shape-Metrics pipeline previously required biconnected inputs because angles were keyed by (vertex, face) -- a face walk visiting a cut vertex or bridge endpoint more than once collided in that dict and the shape stage rejected the representation. Angles are now stored per corner, keyed by the incoming dart (`OrthogonalRepresentation.corner_angles`); the flow network scales angle-arc capacities by corner multiplicity and the extraction distributes each vertex-face flow over that vertex's corners. Degree-1 vertices put 360-degree (U-turn) corners on face walks; rectangularization splits each into two reflex corners with a zero-min-length virtual dart so the dissection applies unchanged, and the coordinate assignment gained per-segment minimum lengths. The shape stage also validates that angles around every vertex sum to 360 degrees, catching representations extracted from infeasible flows. The bend-minimal drawing now covers trees, pendant edges, bridges, and cut vertices, verified over hundreds of random non-biconnected planar graphs (`tests/test_tsm_nonbiconnected.py`).

- **Bend-optimal drawing of degree > 4 vertices via cage expansion (H5)** (`orthogonal/expansion.py`, `orthogonal/giotto.py`): vertices of degree > 4 are outside the Tamassia flow model (a grid point has only four compass directions). `GIOTTOLayout` now expands each such vertex into a cycle of degree-3 cage vertices in rotation order (the classical GIOTTO / OGDF approach), runs the pipeline on the expanded graph with the cage face constrained to a rectangle -- corner angles capped at 180 degrees and no bends on cycle edges, via new `build_flow_network(cage_faces=..., rigid_edges=...)` parameters -- and maps the cage rectangle back to the vertex's node box, with its edges attaching along the box sides at distinct ports (the Kandinsky look). Combined with H6a, the bend-minimal drawing now covers **all connected planar graphs**; strict mode still raises for degree > 4, and `used_bend_optimal` reports True for planar degree > 4 inputs in non-strict mode (`tests/test_tsm_expansion.py`).

- **`KandinskyLayout(bend_optimal=True)` draws from the bend-minimal representation** (`orthogonal/kandinsky.py`, new `orthogonal/realization.py`): Kandinsky computed the bend-minimal orthogonal representation (`optimize_bends`) but never drew from it -- the representation was stored on `orthogonal_rep` and discarded, and every drawing came from the hierarchical heuristic router. The Topology-Shape-Metrics realization (shape -> coordinates -> rectangularization, plus H5 cages and H6a per-corner angles) that drives `GIOTTOLayout` was extracted into a shared `realization` module (`bend_optimal_representation`, `realize_bend_optimal_drawing`) and wired into Kandinsky behind a new opt-in `bend_optimal` flag. With `bend_optimal=True` Kandinsky draws the compact, provably bend-minimal orthogonal layout for connected planar graphs (including degree > 4 and bridges / cut vertices), matching GIOTTO's bend count on shared inputs; `used_bend_optimal` reports whether it applied. Default `False` preserves the layered hierarchical layout. Non-planar input (crossing dummies present) falls back to the heuristic router -- realizing through crossing gadgets remains a follow-up (`tests/test_kandinsky_bend_optimal.py`).

### Changed

- **`KandinskyLayout(optimize_bends=...)` docstring corrected** (`orthogonal/kandinsky.py`): it claimed to "minimize the number of bends in edge routing," but the heuristic router never consumed the computed representation. It now documents accurately -- `optimize_bends` populates the `orthogonal_rep` property and is what `bend_optimal` draws from.

### Fixed

- **Obstacle-aware segment nudging** (`orthogonal/edge_routing.py`): `nudge_overlapping_segments()` separated coincident parallel segments by blind offsets, which could push a segment straight through a node box. Every candidate offset is now checked against the node boxes (excluding the segment's own edge endpoints); a blocked segment tries the mirrored offset and otherwise stays in place, so nudging never introduces an edge-through-node overlap.

- **Rectangularization (turn-regularization) for orthogonal compaction** (`orthogonal/metrics.py`): the bend-optimal coordinate assignment previously produced crossing drawings for ~5-11% of in-scope graphs, because its per-edge constraint graphs lack the separation constraints that keep the two sides of a non-rectangular face apart. `compute_coordinates` now rectangularizes first (classical Tamassia refinement, Di Battista et al. ch. 5): every reflex corner of every bounded face is projected onto the wall it faces (dummy point + dummy axis-parallel separation edge) until all faces are rectangles, and the outer face is enclosed in a dummy rectangle via four connector rays so the region outside the boundary is refined the same way. For a fully rectangular subdivision the per-edge constraint graphs are provably sufficient, so the drawing is planar by construction. Verified 100% clean over 1500 random biconnected max-degree-4 planar graphs across 5 seeds (previously 94.5% on the same harness); the `_drawing_conflict` oracle remains as a belt-and-suspenders gate with fallback.

### Changed

- **`GIOTTOLayout` draws bend-optimally by default** (`orthogonal/giotto.py`): with rectangularization covering the whole in-scope domain, `bend_optimal` now defaults to True -- drawings come from the bend-minimal Topology-Shape-Metrics representation instead of the geometric routing heuristic. Out-of-domain inputs (non-planar, disconnected; see the H5 / H6a entries above for degree > 4 and non-biconnected support) silently fall back to the heuristic router; `used_bend_optimal` reports which path ran, and `bend_optimal=False` forces the heuristic.

## [0.3.0]

### Fixed

- **Cola: group containment now enforced in the constraint projection** (`cola/rectangle.py`):

  - `Projection` previously built the VPSC solve from node variables only, so the `min_var`/`max_var` group border variables were created but never entered the solver -- nested-group bounding rectangles were left unconstrained. Grouped layouts ran but groups could freely overlap and interleave.

  - Ported WebCola's recursive `generateGroupConstraints` (`_generate_group_constraints`), including the border-variable representation of contained groups and the constraint-redirection tail, so the non-overlap and containment constraints are generated together over the whole group hierarchy. Group boxes now contain their members and sibling groups stay disjoint (verified: two groups pulled together by inter-group links stay as separated blocks; without containment their boxes overlap substantially).

  - Also fixed a latent `AttributeError` that crashed the `avoid_overlaps` + groups path: group `stiffness` is optional (matching WebCola's `typeof g.stiffness !== "undefined" ? g.stiffness : 0.01`) but was accessed unconditionally.

  - Regression tests (`tests/test_layout.py::TestLayoutWithGroups`) assert group-box disjointness under inter-group attraction and exercise the recursive nested-group path. This completes the WebCola projection port begun in 0.2.0 (node-variable separation/alignment/non-overlap).

- **Cola: grouped layout with unconstrained warm-up no longer crashes** (`cola/layout.py`): the grouped `initial_unconstrained_iterations` path laid out a flat graph, then read positions back from the input dicts that the `nodes()` setter never populated, raising `KeyError`. It now reads the laid-out coordinates from the flat layout's `Node` objects.

- **Spectral: disconnected graphs no longer collapse each component to a point** (`spectral/spectral.py`): the layout took eigenvectors at indices 1 and 2, but for a graph with k connected components the Laplacian's eigenvalue 0 has multiplicity k -- those eigenvectors are per-component indicators, so every node of a component landed on one point. It now skips all near-zero eigenvalues and starts at the first strictly-positive one.

- **Spectral: normalized-Laplacian layout now applies the `D^-1/2` back-transform** (`spectral/spectral.py`): the symmetric-normalized Laplacian's eigenvectors were used directly; the canonical degree-weighted (Koren) layout requires scaling them by `D^-1/2`, so high-degree nodes were mis-weighted. The scaling is now applied on the default `normalized=True` path.

- **Shell layout: out-of-range link indices no longer raise `IndexError`** (`circular/shell.py`): `_compute_degrees` now skips links whose endpoints are outside the node range, matching the guard in `base._build_adjacency`.

- **Quadtree (pure-Python): coincident points no longer infinitely recurse** (`spatial/quadtree.py`): coincident bodies always fall in the same quadrant, so subdivision never separated them (`RecursionError`). A `MAX_DEPTH = 50` cap now merges bodies in place once the cell is effectively zero-sized, mirroring the Cython kernel.

- **Yifan Hu now respects fixed nodes during optimization** (`force/yifan_hu.py`): fixed nodes were only restored at copy-back but moved freely during the simulation, so their wrong intermediate positions perturbed the other nodes. `_layout_level` now accepts a fixed mask and keeps pinned vertices in place (they still exert forces on others); the finest refinement level pins fixed nodes at their true positions.

- **Cola grid-snap now works for negative coordinates** (`cola/descent.py`): the snap offset used Python's `%` (sign of the divisor); WebCola relies on JS `%` (sign of the dividend), so the snap direction flipped for negative coordinates and `grid_snap_iterations > 0` snapped roughly half the nodes the wrong way (or not at all). It now uses `math.fmod`.

- **Edge-crossing metric now counts collinear overlaps and T-junctions** (`metrics.py`): the strict CCW straddle test silently dropped collinear-overlapping edges and endpoint-on-segment touches. `_segments_intersect` now uses the canonical orientation + on-segment test (CLRS), so those degenerate crossings are counted while proper crossings and disjoint collinear edges are handled correctly.

- **ForceAtlas2: regular vs strong gravity were swapped** (`force/force_atlas2.py`, `_speedups.pyx`): regular gravity was distance-scaled and strong gravity distance-independent -- the reverse of the Gephi/Jacomy et al. definition. Regular gravity is now a distance-independent pull toward the center and strong gravity scales with distance, in both the pure-Python and Cython paths. (The fix corrected an existing FA2 test that asserted LinLog produces smaller intra-cluster diameters -- an artifact of the old distance-scaled gravity, not a real LinLog property; it now asserts LinLog's weaker log attraction yields longer edges.)

- **`node.fixed` is now honored by the Circular, Shell, and Bipartite layouts** (`circular/circular.py`, `circular/shell.py`, `bipartite/bipartite.py`): these geometric layouts repositioned every node, ignoring the pin that `RandomLayout` already respected. They now skip fixed nodes, so a pinned node keeps its position.

- **RandomLayout no longer mutates the global RNG** (`basic/random.py`): it called `random.seed()`/`random.uniform()` on the global `random` module, so a seeded run reseeded the process-wide generator. It now uses a local `random.Random` instance.

- **Kandinsky `compaction_method` setter accepts all supported methods** (`orthogonal/kandinsky.py`): the setter rejected `"flow"` and `"longest_path"` even though the constructor documents them and the compaction dispatch implements them; all five methods (`auto`/`greedy`/`ilp`/`flow`/`longest_path`) are now accepted.

- **Spring layout docstring corrected** (`force/spring.py`): it described "constant force" repulsion but implements an inverse-square Coulomb force; the docstring now matches.

- **Metrics: `stress` honors per-link lengths and `angular_resolution` ignores self-loops/parallel edges** (`metrics.py`): the ideal-distance computation used an unweighted hop count (ignoring `link.length`); it now uses weighted shortest paths. `angular_resolution` counted self-loops and parallel edges as spurious 0-degree angles; both are now excluded.

- **Radial tree: angular wedges sized by leaf count** (`hierarchical/radial_tree.py`): wedges were proportional to subtree node count, so a deep narrow subtree hogged as much angular space as a bushy one; they are now proportional to subtree leaf count.

- **Unpositioned nodes are now placed** (`bipartite/bipartite.py`, `circular/shell.py`): nodes omitted from user-supplied bipartite sets or explicit shells were left unpositioned; they are now assigned to the bottom row / an extra outer shell respectively.

- **FA2 global speed is damped** (`force/force_atlas2.py`): the adaptive global speed jumped straight to `tolerance * traction / swing` each iteration, causing jitter; it now rises by at most 50% per iteration (max-rise damping, Jacomy et al.).

- **`compact_flow_1d` never looser than longest-path** (`orthogonal/compaction_flow.py`): the flow compaction could widen the span beyond the longest-path minimum; it now falls back to longest-path whenever the flow solution would loosen it, and the "tighter layouts" docstring was corrected.

- **Deep recursive walks no longer overflow** (`preprocessing.py`, `base.py`, `hierarchical/`): `detect_cycle` is now an iterative DFS, and the Reingold-Tilford and radial-tree walks run in a worker thread with a large stack (`base.run_deep_recursive`), so deep (chain-like) trees are laid out safely on every platform, including Windows (whose ~1 MB native stack would otherwise crash on a raised recursion limit).

- **Cola `tick()` before `start()` no longer crashes** (`cola/layout.py`): it compared `alpha < threshold` with `alpha` still None; it now returns converged.

- **Removed the no-op `directed` parameter from `connected_components`** (`preprocessing.py`): it always computed undirected (weakly connected) components.

- **`count_crossings` now counts crossings involving long edges** (`preprocessing.py`): edges were bucketed by their exact `(layer_src, layer_tgt)` pair, so an edge spanning more than one layer was never compared against the shorter edges in the layer gaps it passes through. It now tests every edge as a straight segment in `(position, layer)` space, counting proper crossings (edges sharing a node excluded).

- **Greedy orthogonal compaction now actually compacts** (`orthogonal/compaction.py`): `CompactionSolver.solve` only pushed elements right to satisfy minimum gaps and never pulled them left, so interior slack survived and the only size reduction was a final margin translate. It now performs longest-path compaction (each element pulled to its leftmost/topmost feasible position), and `compact_horizontal`/`compact_vertical` constrain every overlapping pair (not just consecutive ones) so the tighter packing stays overlap-free. This is the default orthogonal compaction path when scipy is unavailable.

### Added

- **Topological planarization for orthogonal layout** (`orthogonal/planarization.py`): `planarize_graph` inserted dummy vertices at the *geometric* intersections of straight-line edges, so the crossings depended on the (arbitrary) node positions -- a genuinely planar graph laid out poorly gained spurious crossing vertices, and the result was not guaranteed planar. It now performs proper topological planarization: a greedy maximal planar subgraph is embedded via `check_planarity`, then the remaining edges are reinserted one at a time along a minimum-crossing path through the embedding's faces (dual-graph BFS), each crossing becoming a degree-four dummy vertex. Crossings now depend only on the graph's topology (positions are used only to give dummies an approximate coordinate), the augmented graph is always planar, and planar graphs gain no crossings. The reinsertion recovers the known crossing numbers of small graphs (K5=1, K3,3=1, Petersen=2).

- **Sugiyama: Brandes-Köpf horizontal coordinate assignment** (`hierarchical/_brandes_koepf.py`, `hierarchical/sugiyama.py`): the layered layout placed nodes at evenly-spaced integer slots and centered each layer independently, discarding the crossing-minimization ordering signal and producing avoidable bends. It now assigns within-layer x-coordinates with Brandes-Köpf ("Fast and Simple Horizontal Coordinate Assignment", 2002) -- four vertical-alignment runs with type-1 conflict marking, packed left/right and balanced per vertex -- so each vertex aligns with the median of its neighbours, long-edge dummy chains are drawn as straight vertical segments, and the drawing is symmetric (parents are centred over their children). The block compaction uses a longest-path formulation that computes the same tightest packing as the paper's shift-class method while guaranteeing the within-layer ordering and minimum-separation invariant by construction.

### Changed

- **Showcase demos updated for the orthogonal and group work** (`tests/demos/`):

  - `showcase.py`: the GIOTTO catalog entry now actually enables `bend_optimal`, so it renders the bend-minimal Topology-Shape-Metrics drawing instead of the routing heuristic (the entry was labelled "bend-optimal" but never passed the flag). On the degree-4 planar demo graphs this drops the 3x3 grid from 24 to 0 bends and the ladder from 26 to 2 bends.

  - `improvements_showcase.py`: added a Cola nested-group containment (C1) panel -- a before/after (`avoid_overlaps` off vs on) of two groups pulled together by inter-group links, drawing the group bounding boxes and member-colored node boxes. Group-box overlap area goes from 788 (interleaved) to 0 (separated blocks).

## [0.2.0]

### Changed

- **Build system migrated from setuptools to scikit-build-core** (`pyproject.toml`, `CMakeLists.txt`): the `graph_layout._speedups` Cython extension is now built via CMake through the `scikit-build-core` backend. A new `CMakeLists.txt` drives Cython compilation (out-of-source) and the module install; `[tool.scikit-build]` governs wheel/sdist packaging. The legacy `setup.py` and `MANIFEST.in` are removed.

  - Note: `_speedups` is now a hard build requirement -- a C compiler is required to install from source. The previous setuptools build treated the extension as optional and silently fell back to pure Python if compilation failed; the pure-Python fallbacks still exist at runtime, but a source install can no longer skip compilation.

### Added

- **Orthogonal Topology-Shape-Metrics pipeline** (`orthogonal/metrics.py`):

  - `compute_orthogonal_shape()` assigns a compass direction (E/N/W/S) to every edge segment of an orthogonal representation by propagating turns around faces; `face_turn_sum()` checks the per-face turn invariant; unrealizable representations are detected (`valid=False`) for safe fallback

  - `compute_coordinates()` turns a shape into integer coordinates via constraint-graph assignment, producing an orthogonal drawing (axis-aligned segments, minimum length 1). It tries a compact longest-path assignment first, then a "spread" assignment that gives every coordinate class a distinct value (separating independent features that longest-path collapses), keeping the first clean result

  - Detects drawings that are not clean (coincident vertices, overlapping or crossing edges, edges through a vertex) and reports them invalid so callers fall back rather than emit a broken drawing (verified: no valid drawing has a conflict over random biconnected max-degree-4 graphs)

  - Coverage: ~89% of in-scope (biconnected, max-degree-4) graphs draw cleanly with the bend-minimal representation; the rest (genuine crossings that only full face rectangularization resolves) fall back to the heuristic

- **GIOTTO bend-optimal drawing** (`orthogonal/giotto.py`): new `bend_optimal` option (default off). When on and the representation is a realizable shape (biconnected, max degree 4), the drawing is produced directly from the bend-minimal representation instead of the geometric routing heuristic; otherwise it falls back. Verified on grids, K4, cube, wheel, prism (non-overlapping boxes, orthogonal edges). The `used_bend_optimal` property reports whether a run actually drew from the bend-minimal representation or silently fell back to the heuristic.

- **Review-improvements showcase** (`tests/demos/improvements_showcase.py`, `make showcase-improvements`): generates `build/improvements_showcase.html` with side-by-side visual evidence of the changes -- GIOTTO `bend_optimal` vs the heuristic router (e.g. a 3x3 grid drops from 24 bends to 0; K4 from 14 to 4), Cola overlap-avoidance and separation constraints now taking effect, and Sugiyama / GIOTTO handling cyclic input.

### Fixed

- **GIOTTO infinite recursion on cyclic graphs** (`orthogonal/giotto.py`):

  - `_assign_layers` had no back-edge guard, so any cycle drove the DFS depth upward without bound (`RecursionError`) -- affecting the default path, since orthogonal layouts are almost always cyclic. Added an on-path stack guard so back edges are ignored and the layering stays acyclic.


- **Planarity: Left-Right nesting-depth off-by-one** (`planarity/_lr_planarity.py`):

  - Back-edge nesting depth was `2*height[w] + 1` instead of the canonical `2*height[w]`

  - The spurious `+1` corrupted the nesting-depth sort, so `check_planarity()`/`is_planar()` returned order-dependent wrong verdicts (planar graphs reported non-planar) and could produce non-planar (genus > 0) embeddings, degrading every embedder built on top

  - Regression tests validate order-independence, Euler's formula (`V - E + F == 2`) on every returned embedding, and agreement with `networkx.check_planarity` over random graphs

- **Planarity: planar multigraphs falsely rejected** (`planarity/__init__.py`):

  - The Euler `3n - 6` edge bound was applied to the raw multi-edge count (up to two parallel edges per pair are kept), rejecting valid planar multigraphs such as a triangle with one doubled edge

  - The bound now counts distinct vertex pairs

- **ForceAtlas2: Barnes-Hut repulsion missing a degree factor** (`spatial/quadtree.py`, `_speedups.pyx`):

  - The Barnes-Hut path applied only the source degree factor `(deg_j + 1)`, omitting the acting node's `(deg_i + 1)`, so hubs under-repelled for graphs above 50 nodes (the default path)

  - Both the pure-Python quadtree and the Cython kernel now apply the acting node's factor; with `theta=0` the Barnes-Hut kernel matches the naive O(n^2) kernel exactly

- **Spring: divide-by-zero on coincident nodes** (`force/spring.py`):

  - The naive Coulomb repulsion divided by squared distance, which is zero for coincident nodes, raising `ZeroDivisionError`; exactly-coincident pairs are now skipped (matching the Barnes-Hut / Cython paths)

- **Cola: constraint projection was an unimplemented stub** (`cola/rectangle.py`):

  - `Projection.x_project`/`y_project` were empty, so user separation/alignment constraints and `avoid_overlaps=True` had no effect in `Layout`

  - Implemented the VPSC projection: each axis solves separation, alignment, and (optionally) generated non-overlap constraints, keeping nodes close to their stepped positions

  - Note: nested-group containment (`min_var`/`max_var`) is not yet enforced

- **Orthogonalization: bends modeled per face-pair instead of per edge** (`orthogonal/orthogonalization.py`, `orthogonal/_min_cost_flow.py`):

  - Two faces sharing more than one edge (e.g. the length-2 paths of a theta graph) shared a single bend variable, so `flow_to_orthogonal_rep` assigned the same flow to every shared edge and inflated the bend count (a 2-bend drawing was reported as 4)

  - Each edge now has an independent bend variable routed through a unique intermediate node; the number of bends in the representation matches the min-cost-flow cost

  - The min-cost-flow solvers now include arc-only auxiliary nodes when mapping the network

- **Orthogonalization: representation was not a valid orthogonal shape** (`orthogonal/orthogonalization.py`):

  - Bends were attributed to edges by raw tuple order rather than by which directed edge (dart) borders which face, so faces did not turn by the required +/-4 quarter-turns (e.g. K4 came out `[0, 0, 4, 4]`) and no realizable shape existed -- the reason the representation was computed but discarded

  - Bends are now attributed to the dart bordering each face with the correct sign (+1 convex on one side, -1 reflex on the reverse), so every face turns +/-4 across the model's domain (biconnected planar graphs of max degree 4): verified on grids, K4, cube, wheel, prism, theta, and random biconnected max-degree-4 graphs

  - Out of scope and detected as invalid (callers fall back to the heuristic router): degree > 4 (needs the Kandinsky 0-degree-angle model) and non-biconnected graphs with bridges / cut vertices (need per-corner angles)

- **Sugiyama: cycle removal never invoked and no dummy nodes** (`hierarchical/sugiyama.py`):

  - Cycle removal is now run before layer assignment, so cyclic input is reversed into a DAG instead of warning and falling back

  - Edges spanning more than one layer are split with dummy nodes, fixing barycenter contamination (positions of non-adjacent-layer neighbors were being averaged) and enabling correct crossing counting over the expanded graph

  - Crossing minimization now retains the best ordering seen across sweeps rather than the last one

  - Bend points for long edges are exposed via the new `edge_bends` property

## [0.1.8]

### Added

- **Within-layer node ordering for Kandinsky** (`orthogonal/kandinsky.py`):

  - Calls `minimize_crossings_barycenter()` between layer assignment and node positioning

  - Tree and DAG layouts now have logical left-to-right ordering within layers instead of arbitrary insertion order

- **Parent-centering in ILP compaction** (`orthogonal/compaction_ilp.py`):

  - Added auxiliary variables for `|x_parent - x_child|` per cross-layer edge to the LP objective (alpha=0.5)

  - Parents are now centered over their children instead of pushed to the far left

- **Edge-based vertical separation constraints** (`orthogonal/compaction.py`, `orthogonal/compaction_ilp.py`):

  - Both greedy and ILP compaction now enforce minimum vertical separation between connected nodes in different layers

  - Prevents compaction from collapsing distinct layers into the same y-level

- **Orthogonal edge repair and simplification** (`orthogonal/edge_routing.py`):

  - `_ensure_orthogonal()`: post-processing function that fixes diagonal segments, removes zero-length micro-segments (duplicate consecutive points), and removes redundant collinear bends (three consecutive points on same axis)

  - Applied in `route_edge()`, `_route_planarized_edges()`, `_route_single_edge()`, and after segment nudging

- **Fit-to-canvas scaling in showcase** (`tests/demos/showcase.py`):

  - `_fit_transform()`: computes SVG scale+translate to fit layout content within the card area with padding

  - Prevents nodes and edges from being clipped at SVG boundaries

  - Applied to both general and orthogonal layout rendering

- **Makefile `showcase` target**:

  - `make showcase` generates the showcase HTML and opens it on macOS

- **Kandinsky node ordering tests** (`tests/test_kandinsky.py`):

  - `test_tree_parent_between_children`: verifies parent x-position is between children

  - `test_tree_layers_separated`: verifies distinct layers have strictly increasing y

- **Constraint-Aware Edge Routing** (`orthogonal/edge_routing.py`):

  - New module providing global edge routing shared by KandinskyLayout and GIOTTOLayout

  - `assign_ports()`: Even port distribution along node sides using `(i+1)/(k+1)` formula for k edges on the same side

  - `route_self_loop()`: Self-loop routing with 3-bend path around a node corner

  - `route_edge()`: 5-case orthogonal bend logic with basic obstacle-aware segment detouring

  - `route_all_edges()`: Global routing pipeline -- classifies edges (normal/self-loop/parallel), determines sides, distributes ports, routes with obstacle awareness

  - Supports port constraints and custom side-determination functions

  - Self-loops now rendered (previously silently dropped)

  - Parallel edges get distinct port positions (previously overlapped)

  - New package exports: `assign_ports`, `determine_port_sides`, `route_all_edges`, `route_edge`, `route_self_loop`

- **Test suite for edge routing** (`tests/test_edge_routing.py`):

  - 16 tests covering port distribution, self-loop routing, parallel edge separation, obstacle avoidance, port constraints, and layout integration (Kandinsky with self-loops, parallel edges; GIOTTO fallback with self-loops)

- **Test suite for face computation** (`tests/test_face_computation.py`):

  - 15 tests covering edge sanitization, self-loop filtering, multi-edge deduplication, disconnected graphs, embedding verification fallback, Euler formula validation, and orthogonal representation edge cases

- **Flow-Based and Longest-Path Compaction** (`orthogonal/compaction_flow.py`):

  - Two new compaction strategies based on constraint DAGs (ref: Eiglsperger et al. 2001)

  - `compact_layout_longest_path()`: Assigns coordinates via longest-path distances in the constraint DAG. O(n^2) for DAG construction, O(n+m) for longest path. Guaranteed correct in one pass (unlike greedy which can miss constraints between non-consecutive pairs)

  - `compact_layout_flow()`: Min-cost flow compaction that redistributes slack for tighter layouts. Reuses the existing `_min_cost_flow` solver

  - `_build_constraint_dag()`: Checks all pairs for perpendicular overlap, fixing a correctness gap in the greedy approach which only checks consecutive sorted pairs

  - New `compaction_method` options for `KandinskyLayout`: `"flow"`, `"longest_path"` (in addition to existing `"greedy"`, `"ilp"`, `"auto"`)

  - New `compaction_method` parameter for `GIOTTOLayout`: `"greedy"` (default), `"flow"`, `"longest_path"`

  - New package exports: `compact_layout_flow`, `compact_layout_longest_path`

- **Test suite for flow/longest-path compaction** (`tests/test_compaction_flow.py`):

  - 25 tests covering constraint DAG construction, longest-path compaction, flow compaction, Kandinsky/GIOTTO integration, and performance benchmarks (100-node, 500-node)

- **LR-Planarity Testing Module** (`planarity/`):

  - Linear-time O(n+m) planarity testing using the Left-Right planarity algorithm (de Fraysseix & Rosenstiehl, Brandes 2009 implementation)

  - Public API: `is_planar(num_nodes, edges)` and `check_planarity(num_nodes, edges)`

  - `PlanarityResult` dataclass with `is_planar` flag and optional combinatorial embedding

  - `PlanarEmbedding` class wrapping the rotation system with face enumeration, outer face detection, and Euler formula verification

  - Preprocessing: self-loop removal, parallel edge handling (2 ok, 3+ non-planar), disconnected component splitting, biconnected component decomposition (Tarjan's algorithm)

  - Correctly detects K5, K3,3, Petersen graph, subdivisions, and all non-planar minors

  - New package exports: `is_planar`, `check_planarity`, `PlanarityResult`, `PlanarEmbedding`

- **Test suite for planarity** (`tests/test_planarity.py`):

  - 60 tests covering planar graphs (K4, W5, W6, grids, trees, cycles, maximal planar), non-planar graphs (K5, K3,3, Petersen, K6, K7, K4,4, subdivisions), embedding verification (bidirectional check, Euler formula), edge cases (self-loops, multi-edges, disconnected graphs, isolated vertices), GIOTTO integration, and performance

- **Planarity showcase demos** (`tests/demos/showcase.py`):

  - 6 planarity demo graphs (K4, W5, 3x3 grid, K5, K3,3, Petersen) rendered as SVG cards with planar/non-planar status and embedding statistics

### Changed

- **Showcase renders orthogonal nodes as rectangles** (`tests/demos/showcase.py`):

  - Both `layout_to_svg` and `orthogonal_layout_to_svg` now draw rounded rectangles matching the layout's configured `node_width`/`node_height` for Kandinsky/GIOTTO, instead of small circles that left visual gaps between edges and nodes

- **Showcase layout area excludes title/stats** (`tests/demos/showcase.py`):

  - Layout algorithms receive a reduced canvas height (minus 50px top margin, 25px bottom margin) so nodes don't overlap title text

  - Rendered content shifted down via SVG `<g transform>` group

- **`_center_graph()` now shifts orthogonal data** (`base.py`):

  - Centering also shifts `_node_boxes` and `_orthogonal_edges` bend points, fixing edge disconnection in SVG rendering for both KandinskyLayout and GIOTTOLayout

- **Showcase planarity section moved to bottom** (`tests/demos/showcase.py`)

- **Segment nudging showcase cards removed** (`tests/demos/showcase.py`):

  - Both Kandinsky and GIOTTO segment nudging cards disabled until obstacle-aware nudging is implemented (see TODO.md)

- **KandinskyLayout edge routing** (`orthogonal/kandinsky.py`):

  - `_route_original_edges()` now delegates to `route_all_edges()` from the new edge routing module

  - Self-loops and parallel edges now handled correctly with proper port distribution

  - Port constraints still respected via the `port_constraints` parameter

- **GIOTTOLayout edge routing** (`orthogonal/giotto.py`):

  - `_route_edges()` now delegates to `route_all_edges()` from the new edge routing module

  - Same self-loop, parallel edge, and port distribution improvements as Kandinsky

- **GIOTTO planarity validation** (`orthogonal/giotto.py`):

  - Replaced Euler formula heuristic + O(n^5) K5 brute-force check with single call to `check_planarity()`

  - K3,3-based non-planarity now correctly detected (was never detected before)

  - Graphs with n>20 now tested correctly (K5 check was disabled above this threshold)

  - Removed `_is_k5_subgraph()` method

### Fixed

- **Edges disconnected from nodes after centering**: `_center_graph()` shifted node positions but not `node_boxes` or edge bend coordinates, causing SVG edges to render at pre-centering positions while nodes were at post-centering positions

- **Empty leading layers for cyclic graphs**: `longest_path` layering on graphs with cycles (e.g., Petersen) produced empty layers; now filtered with `layers = [layer for layer in layers if layer]`

- **Diagonal edge segments**: Visibility-graph routing and planarized edge routing could produce non-orthogonal paths; `_ensure_orthogonal()` post-processing now guarantees all segments are axis-aligned

- **Segment nudging creating diagonal paths** (`orthogonal/edge_routing.py`):

  - `nudge_overlapping_segments()` moved bend points without updating port positions, creating diagonal segments from port to first/last bend

  - Now applies `_ensure_orthogonal()` after nudging to reconnect with L-shaped bends

- **Redundant edge bend points** (`orthogonal/edge_routing.py`):

  - Edge routing produced zero-length micro-segments (duplicate consecutive points) and redundant collinear bends (unnecessary direction changes in empty space)

  - `_ensure_orthogonal()` now includes path simplification: deduplicates consecutive points and removes collinear middle points

- **Robust face computation** (`orthogonal/orthogonalization.py`):

  - Self-loops no longer corrupt vertex degrees in flow network (were inflating supply values)

  - Multi-edges no longer produce duplicate face entries (deduplicated before face tracing)

  - `compute_faces()` now validates `PlanarEmbedding.verify()` before using embedding path; falls back to legacy path with warning on failure

  - Legacy face tracing has safety bound to prevent infinite loops on malformed input

  - Legacy `.index()` lookup uses `try/except` instead of bare call (handles missing neighbors gracefully)

  - Neighbor lists deduplicated before angular sorting in legacy path

- **Embedder edge-case handling** (`planarity/embedders.py`):

  - All three embedders (`FixedEmbedder`, `MaxFaceEmbedder`, `MinDepthEmbedder`) now filter self-loops from edges before processing

  - Disconnected planar graphs handled correctly through existing `check_planarity` component decomposition

  - Isolated vertices no longer cause failures in block-cut tree traversal

- **Embedder edge-case tests** (`tests/test_embedders.py`):

  - 11 new tests for disconnected graphs (two disjoint triangles), isolated vertices, self-loops across all three embedders

## [0.1.7]

### Fixed

- **Type annotations** for clean `make typecheck`:

  - Added `cast()` for Cython function returns in `planarization.py` (segments_intersect, find_edge_crossings)

  - Added `cast()` for link attribute access in `preprocessing.py` (_default_get_source, _default_get_target)

  - Added generic type parameters to `yifan_hu.py` (dict -> dict[str, Any] for coarsening hierarchy)

  - Extended mypy `ignore_errors` for legacy cola modules (linklengths, handledisconnected, geom, rectangle, powergraph, layout3d, adapter)

  - Added ruff N806 exception for `orthogonal/*.py` to allow mathematical variable names in ILP formulation

### Added

- **Export Formats** (`export/`):

  - Method-based API on all layout classes:

    - `layout.to_svg()` - Export to SVG format

    - `layout.to_dot()` - Export to DOT (Graphviz) format

    - `layout.to_graphml()` - Export to GraphML format

  - SVG export (`to_svg`, `to_svg_orthogonal`):

    - Generates SVG representations of graph layouts

    - Customizable node shapes (circle, rect), colors, stroke widths

    - Support for node labels with font customization

    - Background color support

    - Special handling for orthogonal layouts with bends

  - DOT/Graphviz export (`to_dot`, `to_dot_orthogonal`):

    - Generates DOT format for Graphviz tools

    - Directed and undirected graph support

    - Position embedding for exact layout reproduction

    - Custom node/edge attribute callbacks

    - Orthogonal variant with `splines=ortho` setting

  - GraphML export (`to_graphml`, `to_graphml_orthogonal`):

    - XML-based format for graph data interchange

    - Node positions, sizes, and labels as data keys

    - Edge weights and lengths

    - Orthogonal variant includes bend points and port sides

  - Standalone functions also available: `from graph_layout import to_svg, to_dot, to_graphml`

  - KandinskyLayout and GIOTTOLayout override export methods to use orthogonal-specific output

- **Test suite for export** (`tests/test_export.py`):

  - 56 tests covering SVG, DOT, and GraphML export

  - Tests for both function and method-based APIs

  - Edge cases: empty graphs, single nodes, self-loops, special characters

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

  - Validates degree <= 4 and planarity

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
