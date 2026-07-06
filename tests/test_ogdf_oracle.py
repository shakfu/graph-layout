"""Differential-testing oracle backed by OGDF (``ogdf-py``).

This complements ``test_planarity_regression.py`` (which uses ``networkx``) with a
*second, independent* reference implementation. Two things are checked:

1. **Planarity agreement.** ``graph_layout.planarity.is_planar`` must agree with
   ``ogdf.is_planar`` on random and adversarial graphs. OGDF's Boyer-Myrvold test
   is unrelated to graph-layout's Left-Right implementation, so a shared blind
   spot between graph-layout and networkx (both LR-family) would still be caught
   here.

2. **Planar-layout flag fidelity.** The straight-line planar layouts
   (``SchnyderLayout``/``FPPLayout``/``TutteLayout``/``MixedModelLayout``) expose a
   ``used_*`` flag claiming "I produced a crossing-free drawing." Using
   ``ogdf.is_planar`` as ground truth for whether the input *can* be drawn
   crossing-free with straight lines, this asserts the flag is truthful and, when
   set, that the actual drawing has zero crossings (graph-layout's own
   ``edge_crossings`` metric). This closes the loop between "is the input planar"
   and "did the layout take its crossing-free path" -- a check neither the
   brute-force geometry oracle nor the boolean networkx oracle makes on its own.

3. **Connectivity ladder.** graph-layout's connectivity code -- ``is_connected``,
   ``connected_components``, and the block-cut tree (``cut_vertices`` /
   biconnected blocks, which drive the planar embedders) -- is cross-checked
   against OGDF at every tier of the connected -> biconnected -> triconnected
   ladder. The block-cut-tree checks are the highest-value target: an independent
   implementation of articulation points directly validates
   ``planarity/_block_cut_tree.py``. graph-layout has no triconnectivity/SPQR of
   its own, so ``is_triconnected`` and (where available) ``separation_pair`` /
   ``spqr_tree_summary`` are used one-directionally to anchor the top of the
   ladder and to confirm graph-layout's decomposition is consistent with it.

The whole module skips when ``ogdf-py`` is not installed; see
``tests/_ogdf_oracle.py`` for how to install the optional oracle. Some
connectivity checks additionally skip on the pinned PyPI build, which does not
expose ``cut_vertices`` / ``separation_pair`` / ``spqr_tree_summary``.
"""

from __future__ import annotations

import random

import pytest

from graph_layout import (
    FPPLayout,
    MixedModelLayout,
    SchnyderLayout,
    TutteLayout,
)
from graph_layout.metrics import edge_crossings
from graph_layout.planarity import is_planar
from graph_layout.planarity._block_cut_tree import build_block_cut_tree
from graph_layout.preprocessing import connected_components, is_connected

from ._ogdf_oracle import build_ogdf_graph, require_ogdf_attr, requires_ogdf

pytestmark = requires_ogdf


# ---------------------------------------------------------------------------
# Graph generators (self-contained; no external oracle needed to build them)
# ---------------------------------------------------------------------------


def _nodes(n: int) -> list[dict]:
    return [{} for _ in range(n)]


def _links(edges: list[tuple[int, int]]) -> list[dict]:
    return [{"source": u, "target": v} for u, v in edges]


def _complete(n: int) -> tuple[int, list[tuple[int, int]]]:
    return n, [(i, j) for i in range(n) for j in range(i + 1, n)]


def _complete_bipartite(a: int, b: int) -> tuple[int, list[tuple[int, int]]]:
    return a + b, [(i, a + j) for i in range(a) for j in range(b)]


def _random_connected_simple(
    rng: random.Random, n: int
) -> tuple[int, list[tuple[int, int]]]:
    """A random connected simple graph on ``n`` vertices.

    A random spanning tree guarantees connectivity; a random number of extra
    non-parallel, non-loop edges is layered on top. Connectivity and simplicity
    are exactly the domain preconditions of the planar layouts, so the resulting
    ``used_*`` flag is governed purely by planarity -- which the oracle decides.
    """
    edges: set[tuple[int, int]] = set()
    perm = list(range(n))
    rng.shuffle(perm)
    for i in range(1, n):
        j = rng.randrange(i)
        a, b = perm[i], perm[j]
        edges.add((min(a, b), max(a, b)))
    extra = rng.randint(0, n)
    attempts = 0
    while extra > 0 and attempts < n * 6:
        attempts += 1
        a, b = rng.randrange(n), rng.randrange(n)
        if a == b:
            continue
        e = (min(a, b), max(a, b))
        if e in edges:
            continue
        edges.add(e)
        extra -= 1
    return n, sorted(edges)


def _apollonian(rng: random.Random, n_extra: int) -> tuple[int, list[tuple[int, int]]]:
    """A random planar triangulation (connected, simple, guaranteed planar)."""
    edges: set[tuple[int, int]] = {(0, 1), (1, 2), (0, 2)}
    faces: list[tuple[int, int, int]] = [(0, 1, 2)]
    nid = 3
    for _ in range(n_extra):
        a, b, c = faces.pop(rng.randrange(len(faces)))
        for x in (a, b, c):
            edges.add((min(x, nid), max(x, nid)))
        faces += [(a, b, nid), (b, c, nid), (a, c, nid)]
        nid += 1
    return nid, sorted(edges)


def _random_simple(rng: random.Random, n: int) -> tuple[int, list[tuple[int, int]]]:
    """A random simple graph on ``n`` vertices, possibly disconnected.

    Edge count ranges from 0 up to ~2n, so the sample spans disconnected graphs,
    trees, and dense graphs -- exercising every tier of the connectivity ladder.
    """
    target = rng.randint(0, 2 * n)
    edges: set[tuple[int, int]] = set()
    attempts = 0
    while len(edges) < target and attempts < n * 10:
        attempts += 1
        a, b = rng.randrange(n), rng.randrange(n)
        if a == b:
            continue
        edges.add((min(a, b), max(a, b)))
    return n, sorted(edges)


def _adj(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _links_of(edges: list[tuple[int, int]]) -> list[dict]:
    return [{"source": u, "target": v} for u, v in edges]


def _gl_biconnected_blocks(n: int, edges: list[tuple[int, int]]) -> int:
    """Number of edge-bearing biconnected blocks graph-layout finds.

    The block-cut tree adds singleton (edgeless) blocks for isolated vertices;
    OGDF's ``biconnected_components`` partitions *edges*, so edgeless blocks are
    filtered out to make the two counts directly comparable.
    """
    bct = build_block_cut_tree(n, _adj(n, edges))
    return sum(1 for b in bct.blocks if b.edges)


PLANAR_LAYOUTS = [
    (SchnyderLayout, "used_schnyder"),
    (FPPLayout, "used_fpp"),
    (TutteLayout, "used_tutte"),
    (MixedModelLayout, "used_mixed_model"),
]

# ``metrics.edge_crossings`` treats every edge as an exact straight segment
# between node centers. That is a *valid* crossing oracle only for the
# integer-grid straight-line methods, where the orientation determinant is an
# exact integer. It is NOT valid for:
#   * ``TutteLayout`` -- barycentric embeddings crowd interior vertices into
#     machine-epsilon collinearity, so the exact test reports false-positive
#     crossings (this is exactly why ``test_planar_straightline.py`` uses a
#     tolerance-based orientation oracle for Tutte, not ``edge_crossings``).
#   * ``MixedModelLayout`` -- edges are port-attached polylines, so node-center
#     segments are not the drawn edges at all (validated there by a separate
#     visibility oracle).
# For those two the flag-truthfulness (planarity) implications below are still
# asserted; only the exact crossing count is restricted to the grid methods.
EXACT_CROSSING_FLAGS = {"used_schnyder", "used_fpp"}


# ---------------------------------------------------------------------------
# 1. Planarity boolean agreement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, edges",
    [
        _complete(3),
        _complete(4),
        _complete(5),  # non-planar
        _complete(6),  # non-planar
        _complete_bipartite(2, 3),
        _complete_bipartite(3, 3),  # K3,3, non-planar
        _complete_bipartite(2, 5),
    ],
    ids=["K3", "K4", "K5", "K6", "K2_3", "K3_3", "K2_5"],
)
def test_planarity_agreement_known_families(n, edges):
    import ogdf

    g, _ = build_ogdf_graph(n, edges)
    assert is_planar(n, edges) == ogdf.is_planar(g)


@pytest.mark.parametrize("seed", range(40))
def test_planarity_agreement_random(seed):
    import ogdf

    rng = random.Random(seed)
    n = rng.randint(3, 14)
    n, edges = _random_connected_simple(rng, n)
    g, _ = build_ogdf_graph(n, edges)
    assert is_planar(n, edges) == ogdf.is_planar(g), (
        f"planarity disagreement: n={n} edges={edges} "
        f"graph_layout={is_planar(n, edges)} ogdf={ogdf.is_planar(g)}"
    )


# ---------------------------------------------------------------------------
# 2. Planar-layout flag fidelity vs OGDF ground truth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout_cls, flag", PLANAR_LAYOUTS)
@pytest.mark.parametrize("seed", range(20))
def test_planar_layout_flag_is_truthful(layout_cls, flag, seed):
    """The ``used_*`` flag must never over-claim, and must draw crossing-free.

    Two implications, both sound for every straight-line planar layout regardless
    of any extra domain restrictions (e.g. biconnectivity) a given method has:

    * ``flag`` set  => input is planar (OGDF confirms) AND drawing has 0 crossings.
    * input non-planar (OGDF) => ``flag`` is unset (a non-planar graph cannot be
      drawn crossing-free with straight lines).
    """
    import ogdf

    rng = random.Random(1000 + seed)
    n = rng.randint(4, 12)
    n, edges = _random_connected_simple(rng, n)
    g, _ = build_ogdf_graph(n, edges)
    ogdf_planar = ogdf.is_planar(g)

    layout = layout_cls(nodes=_nodes(n), links=_links(edges), size=(800.0, 600.0))
    layout.run()
    used = getattr(layout, flag)

    if used:
        assert ogdf_planar, (
            f"{layout_cls.__name__} claimed a crossing-free drawing of a graph "
            f"OGDF reports non-planar: n={n} edges={edges}"
        )
        if flag in EXACT_CROSSING_FLAGS:
            assert edge_crossings(layout.nodes, layout.links) == 0, (
                f"{layout_cls.__name__} set {flag} but the drawing has crossings: "
                f"n={n} edges={edges}"
            )
    if not ogdf_planar:
        assert not used, (
            f"{layout_cls.__name__} set {flag} on a non-planar graph: "
            f"n={n} edges={edges}"
        )


@pytest.mark.parametrize("layout_cls, flag", PLANAR_LAYOUTS)
@pytest.mark.parametrize("seed", range(12))
def test_planar_layout_draws_confirmed_planar_graph(layout_cls, flag, seed):
    """On a graph OGDF confirms planar, the layout should take its planar path.

    Apollonian networks are connected, simple, planar triangulations -- inside
    the shared domain of all four planar layouts -- so the ``used_*`` flag must be
    set and the drawing crossing-free. OGDF confirms the planarity precondition
    independently of graph-layout's own embedding code.
    """
    import ogdf

    rng = random.Random(2000 + seed)
    n, edges = _apollonian(rng, rng.randint(1, 8))
    g, _ = build_ogdf_graph(n, edges)
    assert ogdf.is_planar(g), "generator invariant: apollonian networks are planar"

    layout = layout_cls(nodes=_nodes(n), links=_links(edges), size=(800.0, 600.0))
    layout.run()
    assert getattr(layout, flag) is True, (
        f"{layout_cls.__name__} did not take its planar path on an OGDF-confirmed "
        f"planar triangulation: n={n} edges={edges}"
    )
    if flag in EXACT_CROSSING_FLAGS:
        assert edge_crossings(layout.nodes, layout.links) == 0


# ---------------------------------------------------------------------------
# 3. Connectivity ladder: connected -> biconnected -> triconnected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(40))
def test_is_connected_agreement(seed):
    import ogdf

    rng = random.Random(3000 + seed)
    n = rng.randint(1, 16)
    n, edges = _random_simple(rng, n)
    g, _ = build_ogdf_graph(n, edges)
    assert is_connected(n, _links_of(edges)) == ogdf.is_connected(g), (
        f"is_connected disagreement: n={n} edges={edges}"
    )


@pytest.mark.parametrize("seed", range(40))
def test_connected_component_count_agreement(seed):
    import ogdf

    rng = random.Random(4000 + seed)
    n = rng.randint(1, 16)
    n, edges = _random_simple(rng, n)
    g, _ = build_ogdf_graph(n, edges)
    comp = ogdf.NodeArrayInt(g)
    ogdf_count = ogdf.connected_components(g, comp)
    gl_count = len(connected_components(n, _links_of(edges)))
    assert gl_count == ogdf_count, (
        f"component-count disagreement: graph_layout={gl_count} ogdf={ogdf_count} "
        f"n={n} edges={edges}"
    )


@pytest.mark.parametrize("seed", range(40))
def test_cut_vertices_agreement(seed):
    """The block-cut tree's articulation points must match OGDF's exactly.

    This is the sharpest connectivity oracle: OGDF's ``cut_vertices`` is an
    independent implementation, and equality of the *sets* (not just counts)
    directly validates ``planarity/_block_cut_tree.py``. Requires a build that
    exposes ``cut_vertices`` (skipped on the pinned PyPI wheel).
    """
    require_ogdf_attr("cut_vertices")
    import ogdf

    rng = random.Random(5000 + seed)
    n = rng.randint(3, 16)
    n, edges = _random_connected_simple(rng, n)
    g, _ = build_ogdf_graph(n, edges)

    bct = build_block_cut_tree(n, _adj(n, edges))
    # Guarded by require_ogdf_attr above; absent from the PyPI 0.1.1 stubs.
    ogdf_cut = {node.index for node in ogdf.cut_vertices(g)}  # type: ignore[attr-defined]
    assert bct.cut_vertices == ogdf_cut, (
        f"cut-vertex disagreement: graph_layout={sorted(bct.cut_vertices)} "
        f"ogdf={sorted(ogdf_cut)} n={n} edges={edges}"
    )


@pytest.mark.parametrize("seed", range(40))
def test_biconnected_component_count_agreement(seed):
    import ogdf

    rng = random.Random(6000 + seed)
    n = rng.randint(3, 16)
    n, edges = _random_connected_simple(rng, n)
    if not edges:
        pytest.skip("needs at least one edge")
    g, _ = build_ogdf_graph(n, edges)

    ea = ogdf.EdgeArrayInt(g)
    ogdf_count = ogdf.biconnected_components(g, ea)
    gl_count = _gl_biconnected_blocks(n, edges)
    assert gl_count == ogdf_count, (
        f"biconnected-block-count disagreement: graph_layout={gl_count} "
        f"ogdf={ogdf_count} n={n} edges={edges}"
    )


@pytest.mark.parametrize("seed", range(40))
def test_derived_biconnectivity_agreement(seed):
    """graph-layout has no ``is_biconnected``; derive it from the block-cut tree.

    For a connected graph on >= 3 vertices, biconnected <=> no cut vertices <=>
    exactly one biconnected block. That derived verdict must match OGDF's
    first-class ``is_biconnected``.
    """
    import ogdf

    rng = random.Random(7000 + seed)
    n = rng.randint(3, 16)
    n, edges = _random_connected_simple(rng, n)
    g, _ = build_ogdf_graph(n, edges)

    bct = build_block_cut_tree(n, _adj(n, edges))
    blocks_with_edges = sum(1 for b in bct.blocks if b.edges)
    gl_biconnected = (not bct.cut_vertices) and blocks_with_edges == 1
    assert gl_biconnected == ogdf.is_biconnected(g), (
        f"derived-biconnectivity disagreement: graph_layout={gl_biconnected} "
        f"ogdf={ogdf.is_biconnected(g)} n={n} edges={edges}"
    )


@pytest.mark.parametrize("seed", range(40))
def test_triconnected_implies_single_block(seed):
    """Anchor the top of the ladder: OGDF-triconnected => one block, no cut vertex.

    graph-layout cannot detect triconnectivity, but a triconnected graph (>= 3
    vertices) is necessarily biconnected, so its block-cut tree must collapse to a
    single block with no articulation points. Using OGDF's ``is_triconnected`` as
    ground truth turns that implication into a check on graph-layout's
    decomposition.
    """
    import ogdf

    rng = random.Random(8000 + seed)
    n = rng.randint(4, 14)
    n, edges = _random_connected_simple(rng, n)
    g, _ = build_ogdf_graph(n, edges)
    if not ogdf.is_triconnected(g):
        pytest.skip("not triconnected; nothing to anchor for this sample")

    bct = build_block_cut_tree(n, _adj(n, edges))
    blocks_with_edges = sum(1 for b in bct.blocks if b.edges)
    assert not bct.cut_vertices and blocks_with_edges == 1, (
        f"triconnected graph did not collapse to a single block: "
        f"cut_vertices={sorted(bct.cut_vertices)} blocks={blocks_with_edges} "
        f"n={n} edges={edges}"
    )


@pytest.mark.parametrize("seed", range(30))
def test_spqr_consistent_with_triconnectivity(seed):
    """SPQR summary and separation pair are consistent with the biconnected tier.

    graph-layout has no SPQR, so this documents and guards the relationship the
    ladder rests on, using OGDF as reference: on a biconnected graph (which
    graph-layout confirms yields exactly one block), a separation pair exists iff
    the graph is not triconnected, and a triconnected graph is a single SPQR
    R-node. Requires a build exposing ``spqr_tree_summary`` / ``separation_pair``.
    """
    require_ogdf_attr("spqr_tree_summary")
    require_ogdf_attr("separation_pair")
    import ogdf

    rng = random.Random(9000 + seed)
    n = rng.randint(4, 14)
    n, edges = _random_connected_simple(rng, n)
    g, _ = build_ogdf_graph(n, edges)
    if not ogdf.is_biconnected(g):
        pytest.skip("SPQR summary requires a biconnected graph")

    # graph-layout must see this biconnected graph as a single block.
    assert _gl_biconnected_blocks(n, edges) == 1

    triconnected = ogdf.is_triconnected(g)
    # Guarded by require_ogdf_attr above; absent from the PyPI 0.1.1 stubs.
    has_sep_pair = ogdf.separation_pair(g) is not None  # type: ignore[attr-defined]
    assert has_sep_pair == (not triconnected), (
        f"separation-pair/triconnectivity inconsistency: sep_pair={has_sep_pair} "
        f"triconnected={triconnected} n={n} edges={edges}"
    )
    summary = ogdf.spqr_tree_summary(g)  # type: ignore[attr-defined]
    if triconnected:
        assert summary["R"] == 1 and summary["S"] == 0 and summary["P"] == 0, (
            f"triconnected graph is not a single SPQR R-node: {summary}"
        )
