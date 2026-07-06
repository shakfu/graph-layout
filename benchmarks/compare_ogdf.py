"""Compare graph-layout's layouts against OGDF (``ogdf-py``) on quality and speed.

graph-layout reimplements, in pure Python/Cython, many algorithms that OGDF
implements in C++. This harness runs *comparable* layouts from both engines on
the shared benchmark graphs and reports, per layout:

* **time** -- wall-clock seconds for the layout call (OGDF sets the C++ ceiling;
  the gap shows where pure Python pays).
* **stress** -- *scale-invariant normalized stress* (lower is better): the
  layout is optimally rescaled before scoring, so the number measures drawing
  *shape* independent of each engine's coordinate scale. This is the fair
  cross-implementation quality metric (Gansner et al. 2004).
* **crossings** -- edge crossings (small graphs only; the check is O(m^2)).

Both engines' outputs are scored by the *same* functions here, so the comparison
is apples-to-apples. This is a dev/benchmark tool, not part of the package; it
exits cleanly with a message if ``ogdf-py`` is not installed.

Usage:
    uv run python benchmarks/compare_ogdf.py                 # default: small+medium
    uv run python benchmarks/compare_ogdf.py --graphs small_random grid_small
    uv run python benchmarks/compare_ogdf.py --all           # include large/xlarge
    uv run python benchmarks/compare_ogdf.py --max-nodes 600
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Callable, Optional

GRAPHS_DIR = Path(__file__).parent / "graphs"

# Default graph set: small + medium only. Pure-Python layouts on 1000+ nodes can
# take minutes; opt into those with --all or an explicit --graphs list.
DEFAULT_GRAPHS = [
    "small_random",
    "small_scalefree",
    "grid_small",
    "medium_random",
    "medium_scalefree",
    "medium_smallworld",
]

# Only count crossings for graphs at or below this edge count (O(m^2)).
CROSSINGS_MAX_EDGES = 500


# ---------------------------------------------------------------------------
# Graph loading and fair quality metrics
# ---------------------------------------------------------------------------


def load_graph(name: str) -> tuple[int, list[tuple[int, int]]]:
    data = json.loads((GRAPHS_DIR / f"{name}.json").read_text())
    n = len(data["nodes"])
    edges = [(int(e["source"]), int(e["target"])) for e in data["links"]]
    return n, edges


def _bfs_distances(n: int, adj: list[list[int]], source: int) -> list[float]:
    dist = [math.inf] * n
    dist[source] = 0.0
    q = deque([source])
    while q:
        u = q.popleft()
        for w in adj[u]:
            if dist[w] == math.inf:
                dist[w] = dist[u] + 1.0
                q.append(w)
    return dist


def graph_distances(n: int, edges: list[tuple[int, int]]) -> list[list[float]]:
    """All-pairs shortest-path (hop) distances; ``inf`` across components."""
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return [_bfs_distances(n, adj, s) for s in range(n)]


def component_count(dist: list[list[float]]) -> int:
    """Number of connected components, read off the all-pairs distance matrix."""
    n = len(dist)
    seen = [False] * n
    count = 0
    for i in range(n):
        if seen[i]:
            continue
        count += 1
        for j in range(n):
            if math.isfinite(dist[i][j]):
                seen[j] = True
    return count


def all_finite(positions: list[tuple[float, float]]) -> bool:
    """True if every coordinate is finite.

    Some MDS-family layouts (e.g. OGDF's PivotMDS, SpringEmbedderKK) emit NaN
    coordinates on disconnected graphs, where their all-pairs-distance machinery
    breaks down. The harness reports that explicitly rather than scoring nan.
    """
    return all(math.isfinite(x) and math.isfinite(y) for x, y in positions)


def normalized_stress(
    positions: list[tuple[float, float]], dist: list[list[float]]
) -> float:
    """Scale-invariant normalized stress (lower is better).

    Stress is ``sum_{i<j} w_ij (s * ||p_i - p_j|| - d_ij)^2`` with weights
    ``w_ij = 1 / d_ij^2``. The optimal global scale ``s`` has a closed form and is
    applied before scoring, so the result is invariant to each engine's
    coordinate units -- the only fair way to compare two implementations whose
    canvases differ. Pairs in different components (``d = inf``) are skipped. The
    sum is normalized by the pair count so graphs of different sizes are roughly
    comparable.
    """
    n = len(positions)
    num = 0.0
    den = 0.0
    geo: list[tuple[float, float]] = []  # (euclidean, d_ij) for finite pairs
    for i in range(n):
        xi, yi = positions[i]
        di = dist[i]
        for j in range(i + 1, n):
            d = di[j]
            if d == math.inf or d == 0.0:
                continue
            xj, yj = positions[j]
            eucl = math.hypot(xi - xj, yi - yj)
            geo.append((eucl, d))
            w = 1.0 / (d * d)
            num += w * d * eucl
            den += w * eucl * eucl
    if not geo or den == 0.0:
        return math.nan
    s = num / den
    total = 0.0
    for eucl, d in geo:
        r = s * eucl - d
        total += (r * r) / (d * d)
    return total / len(geo)


# ---------------------------------------------------------------------------
# Engine adapters: run a layout, return (positions, seconds)
# ---------------------------------------------------------------------------


def run_graph_layout(
    layout_cls, n: int, edges: list[tuple[int, int]]
) -> tuple[list[tuple[float, float]], float]:
    nodes: list[dict] = [{} for _ in range(n)]
    links = [{"source": u, "target": v} for u, v in edges]
    layout = layout_cls(nodes=nodes, links=links, size=(1000.0, 1000.0))
    t0 = time.perf_counter()
    layout.run()
    dt = time.perf_counter() - t0
    return [(float(nd.x), float(nd.y)) for nd in layout.nodes], dt


def run_ogdf(
    layout_factory: Callable[[], object], n: int, edges: list[tuple[int, int]]
) -> tuple[list[tuple[float, float]], float]:
    import ogdf

    g = ogdf.Graph()
    ns = [g.new_node() for _ in range(n)]
    for u, v in edges:
        g.new_edge(ns[u], ns[v])
    ga = ogdf.GraphAttributes(g)
    layout = layout_factory()
    t0 = time.perf_counter()
    layout.call(ga)  # type: ignore[attr-defined]
    dt = time.perf_counter() - t0
    return [(float(ga.x(node)), float(ga.y(node))) for node in g.nodes()], dt


# ---------------------------------------------------------------------------
# Comparable layout registry, grouped by algorithm family
# ---------------------------------------------------------------------------


def build_registry() -> list[tuple[str, str, Callable, Optional[int]]]:
    """Return ``(family, label, runner, max_nodes)`` for every comparable layout.

    ``runner(n, edges) -> (positions, seconds)``. graph-layout and OGDF entries
    in the same family solve the same problem (stress/MDS or force-directed), so
    their stress/time are directly comparable. ``max_nodes`` caps a layout that is
    prohibitively slow beyond a size (it is reported as skipped, never silently
    omitted) -- notably the pure-Python O(n^2)-per-iteration Kamada-Kawai.
    """
    import ogdf

    import graph_layout as gl

    reg: list[tuple[str, str, Callable, Optional[int]]] = []

    def gl_runner(cls):
        return lambda n, edges: run_graph_layout(cls, n, edges)

    def ogdf_runner(factory):
        return lambda n, edges: run_ogdf(factory, n, edges)

    # Stress / MDS family
    reg.append(("stress", "gl.SMACOF", gl_runner(gl.SMACOFLayout), None))
    reg.append(("stress", "gl.KamadaKawai", gl_runner(gl.KamadaKawaiLayout), 200))
    reg.append(("stress", "ogdf.StressMinimization", ogdf_runner(ogdf.StressMinimization), None))
    reg.append(("stress", "ogdf.PivotMDS", ogdf_runner(ogdf.PivotMDS), None))
    reg.append(("stress", "ogdf.SpringEmbedderKK", ogdf_runner(ogdf.SpringEmbedderKK), None))

    # Force-directed family
    reg.append(("force", "gl.FruchtermanReingold", gl_runner(gl.FruchtermanReingoldLayout), None))
    reg.append(("force", "gl.YifanHu", gl_runner(gl.YifanHuLayout), None))
    reg.append(("force", "ogdf.FMMM", ogdf_runner(ogdf.FMMMLayout), None))
    reg.append(("force", "ogdf.GEM", ogdf_runner(ogdf.GEMLayout), None))

    return reg


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def benchmark_graph(
    name: str,
    n: int,
    edges: list[tuple[int, int]],
    registry: list[tuple[str, str, Callable, Optional[int]]],
) -> None:
    from graph_layout import Link, Node
    from graph_layout.metrics import edge_crossings

    dist = graph_distances(n, edges)
    n_components = component_count(dist)
    disc = f", {n_components} components" if n_components > 1 else ""
    print(f"\n=== {name}: {n} nodes, {len(edges)} edges{disc} ===")
    do_crossings = len(edges) <= CROSSINGS_MAX_EDGES
    links = [Link(u, v) for u, v in edges] if do_crossings else []

    header = f"{'family':7}  {'layout':24}  {'time(s)':>9}  {'stress':>9}  {'crossings':>9}"
    print(header)
    print("-" * len(header))

    last_family: Optional[str] = None
    for family, label, runner, max_nodes in registry:
        if family != last_family and last_family is not None:
            print()
        last_family = family
        if max_nodes is not None and n > max_nodes:
            print(f"{family:7}  {label:24}  {'skip':>9}  (n>{max_nodes})")
            continue
        try:
            positions, dt = runner(n, edges)
        except Exception as exc:  # keep going; report which layout failed
            print(f"{family:7}  {label:24}  {'FAIL':>9}  {str(exc)[:40]}")
            continue
        if not all_finite(positions):
            # e.g. PivotMDS / KK on a disconnected graph -> NaN coordinates.
            print(f"{family:7}  {label:24}  {dt:9.3f}  {'non-finite':>9}  {'-':>9}")
            continue
        ns = normalized_stress(positions, dist)
        if do_crossings:
            scored_nodes = [Node(x=p[0], y=p[1]) for p in positions]
            xings = str(edge_crossings(scored_nodes, links))
        else:
            xings = "-"
        print(f"{family:7}  {label:24}  {dt:9.3f}  {ns:9.4f}  {xings:>9}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="+", help="graph names (default: small+medium)")
    parser.add_argument("--all", action="store_true", help="include large/xlarge graphs")
    parser.add_argument(
        "--max-nodes", type=int, default=None, help="skip graphs above this node count"
    )
    args = parser.parse_args(argv)

    try:
        import ogdf  # noqa: F401
    except ImportError:
        print(
            "ogdf-py is not installed; nothing to compare against.\n"
            "Install it with `uv sync` (pinned in the dev group on supported "
            "platforms) or `make oracle-install` for a local build."
        )
        return 0

    if args.graphs:
        names = args.graphs
    elif args.all:
        index = json.loads((GRAPHS_DIR / "index.json").read_text())
        names = [entry["name"] for entry in index["graphs"]]
    else:
        names = DEFAULT_GRAPHS

    registry = build_registry()
    print(
        "Scale-invariant normalized stress (lower = better); "
        f"crossings shown for <= {CROSSINGS_MAX_EDGES} edges."
    )
    for name in names:
        path = GRAPHS_DIR / f"{name}.json"
        if not path.exists():
            print(f"\n=== {name}: NOT FOUND ({path}) ===")
            continue
        n, edges = load_graph(name)
        if args.max_nodes is not None and n > args.max_nodes:
            print(f"\n=== {name}: {n} nodes > --max-nodes {args.max_nodes}, skipped ===")
            continue
        benchmark_graph(name, n, edges, registry)
    return 0


if __name__ == "__main__":
    sys.exit(main())
