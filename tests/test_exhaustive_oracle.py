"""Exhaustive-enumeration oracle for the connectivity and planarity checks.

Where ``test_ogdf_oracle.py`` samples *random* graphs, this module enumerates the
*entire* small-graph universe and checks graph-layout against a reference on every
one. That converts "agrees on a random sample" into "provably agrees on all graphs
up to ``n`` vertices" -- and for graph algorithms almost every structural bug
manifests at tiny sizes, so this is close to a correctness guarantee on the
combinatorial properties.

Two enumerations are used:

* **All labeled graphs on ``n <= 6`` vertices** (``2^(n(n-1)/2)`` graphs, so 32768
  at ``n = 6``). Enumerating *labeled* rather than isomorphism-class
  representatives means every vertex/edge ordering is tested, which is what
  catches order-dependent bugs (the historical Left-Right nesting-depth bug was
  exactly of that shape).
* **The Read-Wilson graph atlas** (networkx), i.e. all 1253 non-isomorphic graphs
  up to ``n = 7``, for one extra size of structural coverage at low cost.

References (chosen to be either self-evidently correct or independent of
graph-layout):

* **Connectivity** -- a brute-force reference computed here: components by flood
  fill, and a cut vertex defined directly as "removing it increases the number of
  connected components." These are obviously correct and depend on nothing.
* **Planarity** -- ``networkx.check_planarity`` (the project's established
  planarity oracle), plus OGDF's independent Boyer-Myrvold ``is_planar`` as a
  second reference when ``ogdf-py`` is installed. Two unrelated planarity
  implementations agreeing on every small graph rules out a shared blind spot.
"""

from __future__ import annotations

from typing import Iterator

import pytest

from graph_layout.planarity import is_planar
from graph_layout.planarity._block_cut_tree import build_block_cut_tree
from graph_layout.preprocessing import connected_components, is_connected

from ._ogdf_oracle import HAS_OGDF, build_ogdf_graph

# Largest vertex count for the exhaustive *labeled* enumeration. n=6 is 32768
# graphs (~a few seconds); n=7 would be ~2.1M and is covered instead, at one
# labeling per isomorphism class, by the graph atlas below.
MAX_LABELED_N = 6

# Cap the (more expensive) OGDF cross-check to keep the labeled sweep quick; K5
# (n=5) is already covered, and networkx carries the n=6 non-planar witness K3,3.
MAX_OGDF_LABELED_N = 5


# ---------------------------------------------------------------------------
# Graph enumeration
# ---------------------------------------------------------------------------


def all_labeled_graphs(n: int) -> Iterator[list[tuple[int, int]]]:
    """Yield the edge list of every labeled simple graph on ``n`` vertices.

    Iterates all subsets of the ``n(n-1)/2`` possible edges via a bitmask, so the
    full ``2^(n(n-1)/2)`` graphs are produced with no isomorphism reduction.
    """
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(pairs)
    for mask in range(1 << m):
        yield [pairs[k] for k in range(m) if (mask >> k) & 1]


def atlas_graphs() -> Iterator[tuple[int, list[tuple[int, int]]]]:
    """Yield ``(n, edges)`` for every non-isomorphic graph up to 7 vertices.

    Uses the Read-Wilson atlas shipped with networkx. Node labels are normalized
    to a contiguous ``0..n-1`` range. Skips the trivial empty and single-vertex
    graphs, which carry no connectivity or planarity content.
    """
    from networkx.generators.atlas import graph_atlas_g

    for g in graph_atlas_g():
        nodes = sorted(g.nodes())
        n = len(nodes)
        if n < 2:
            continue
        relabel = {old: i for i, old in enumerate(nodes)}
        edges = sorted(
            (min(relabel[u], relabel[v]), max(relabel[u], relabel[v])) for u, v in g.edges()
        )
        yield n, edges


# ---------------------------------------------------------------------------
# Brute-force connectivity reference (self-evidently correct)
# ---------------------------------------------------------------------------


def _adj(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _links(edges: list[tuple[int, int]]) -> list[dict]:
    return [{"source": u, "target": v} for u, v in edges]


def _num_components(vertices: list[int], adj: list[list[int]]) -> int:
    """Connected-component count over exactly ``vertices`` (flood fill)."""
    present = set(vertices)
    seen: set[int] = set()
    count = 0
    for start in vertices:
        if start in seen:
            continue
        count += 1
        stack = [start]
        seen.add(start)
        while stack:
            u = stack.pop()
            for w in adj[u]:
                if w in present and w not in seen:
                    seen.add(w)
                    stack.append(w)
    return count


def _brute_num_components(n: int, adj: list[list[int]]) -> int:
    return _num_components(list(range(n)), adj)


def _brute_connected(n: int, adj: list[list[int]]) -> bool:
    return n <= 1 or _num_components(list(range(n)), adj) == 1


def _brute_cut_vertices(n: int, adj: list[list[int]]) -> set[int]:
    """Articulation points by definition: removing ``v`` adds a component.

    Deleting a leaf leaves the component count unchanged; deleting an isolated
    vertex lowers it; deleting a vertex whose removal splits its component raises
    it. So ``components(G - v) > components(G)`` iff ``v`` is a cut vertex.
    """
    before = _num_components(list(range(n)), adj)
    cuts: set[int] = set()
    for v in range(n):
        rest = [u for u in range(n) if u != v]
        if _num_components(rest, adj) > before:
            cuts.add(v)
    return cuts


# ---------------------------------------------------------------------------
# Exhaustive connectivity: all labeled graphs on n <= MAX_LABELED_N vertices
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", range(1, MAX_LABELED_N + 1))
def test_exhaustive_connectivity_labeled(n):
    """Every labeled graph: is_connected, component count, and cut-vertex set.

    The cut-vertex check is the sharpest: it validates
    ``planarity/_block_cut_tree.py`` against the brute-force definition over every
    labeled graph, including disconnected ones (not covered by the random oracle).
    """
    for edges in all_labeled_graphs(n):
        adj = _adj(n, edges)
        links = _links(edges)

        assert is_connected(n, links) == _brute_connected(n, adj), (n, edges)
        assert len(connected_components(n, links)) == _brute_num_components(n, adj), (
            n,
            edges,
        )
        bct = build_block_cut_tree(n, adj)
        assert bct.cut_vertices == _brute_cut_vertices(n, adj), (n, edges)


def test_exhaustive_connectivity_atlas():
    """Same connectivity checks over all non-isomorphic graphs up to 7 vertices."""
    pytest.importorskip("networkx")
    for n, edges in atlas_graphs():
        adj = _adj(n, edges)
        links = _links(edges)
        assert is_connected(n, links) == _brute_connected(n, adj), (n, edges)
        assert len(connected_components(n, links)) == _brute_num_components(n, adj), (
            n,
            edges,
        )
        assert build_block_cut_tree(n, adj).cut_vertices == _brute_cut_vertices(n, adj), (n, edges)


# ---------------------------------------------------------------------------
# Exhaustive planarity: all labeled graphs on n <= MAX_LABELED_N vertices
# ---------------------------------------------------------------------------


def _ogdf_is_planar(n: int, edges: list[tuple[int, int]]) -> bool:
    import ogdf

    g, _ = build_ogdf_graph(n, edges)
    return bool(ogdf.is_planar(g))


@pytest.mark.parametrize("n", range(1, MAX_LABELED_N + 1))
def test_exhaustive_planarity_labeled(n):
    """Every labeled graph: graph-layout planarity vs networkx (and OGDF)."""
    nx = pytest.importorskip("networkx")
    check_ogdf = HAS_OGDF and n <= MAX_OGDF_LABELED_N
    for edges in all_labeled_graphs(n):
        gl = is_planar(n, edges)

        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from(edges)
        assert gl == nx.check_planarity(g)[0], ("networkx", n, edges)

        if check_ogdf:
            assert gl == _ogdf_is_planar(n, edges), ("ogdf", n, edges)


def test_exhaustive_planarity_atlas():
    """graph-layout planarity vs networkx over all graphs up to 7 vertices."""
    nx = pytest.importorskip("networkx")
    for n, edges in atlas_graphs():
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from(edges)
        assert is_planar(n, edges) == nx.check_planarity(g)[0], (n, edges)
