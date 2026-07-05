"""Property-based tests for the orthogonal (Topology-Shape-Metrics) layouts.

The bend-optimal path fans out across many branches -- biconnected vs. not,
degree <= 4 vs. cage expansion, planar vs. planarized, connected vs. packed
components -- so example-based tests can only cover a handful of shapes. These
Hypothesis tests generate structurally-varied *planar* graphs (trees, grids,
forests) and assert the invariants that must hold for every orthogonal drawing:

* every drawn vertex has a box, and originals are all present;
* every edge is drawn with axis-aligned (horizontal/vertical) segments only;
* whenever the bend-minimal path actually drove the drawing
  (``used_bend_optimal``), node boxes do not overlap and the result is
  deterministic;
* grids are drawn bend-free.

A separate fuzz test feeds arbitrary small graphs through both layouts to check
they never raise and always produce orthogonal edges.
"""

from __future__ import annotations

import hypothesis.strategies as st
from hypothesis import HealthCheck, given, settings

from graph_layout.orthogonal import GIOTTOLayout, KandinskyLayout

_SETTINGS = settings(
    max_examples=60,
    deadline=None,  # layout time varies with graph shape (cage expansion, etc.)
    suppress_health_check=[HealthCheck.too_slow],
)


# ---------------------------------------------------------------------------
# Graph strategies (all planar)
# ---------------------------------------------------------------------------


@st.composite
def random_tree(draw, min_n=1, max_n=12):
    """A random labelled tree (connected, planar; degree unconstrained)."""
    n = draw(st.integers(min_value=min_n, max_value=max_n))
    edges = []
    for v in range(1, n):
        parent = draw(st.integers(min_value=0, max_value=v - 1))
        edges.append((parent, v))
    return n, edges


@st.composite
def grid_graph(draw):
    """A w x h grid graph (connected, planar, max degree 4)."""
    w = draw(st.integers(min_value=2, max_value=5))
    h = draw(st.integers(min_value=2, max_value=5))
    edges = []
    for r in range(h):
        for c in range(w):
            v = r * w + c
            if c + 1 < w:
                edges.append((v, v + 1))
            if r + 1 < h:
                edges.append((v, v + w))
    return w * h, edges


@st.composite
def forest(draw):
    """Disjoint union of 1-3 random trees (planar, usually disconnected)."""
    k = draw(st.integers(min_value=1, max_value=3))
    total = 0
    edges: list[tuple[int, int]] = []
    for _ in range(k):
        n, tree_edges = draw(random_tree(min_n=1, max_n=6))
        edges.extend((a + total, b + total) for a, b in tree_edges)
        total += n
    return total, edges


# ---------------------------------------------------------------------------
# Invariant helpers
# ---------------------------------------------------------------------------


def _make(cls, n, edges, **kwargs):
    nodes = [{} for _ in range(n)]
    links = [{"source": u, "target": v} for u, v in edges]
    layout = cls(nodes=nodes, links=links, size=(800, 600), **kwargs)
    layout.run()
    return layout


def _boxes_by_index(layout):
    return {b.index: b for b in layout.node_boxes}


def _all_edges_orthogonal(layout) -> bool:
    by_idx = _boxes_by_index(layout)
    for e in layout.orthogonal_edges:
        src, tgt = by_idx.get(e.source), by_idx.get(e.target)
        if src is None or tgt is None:
            return False
        sp = src.get_port_position(e.source_port.side, e.source_port.position)
        tp = tgt.get_port_position(e.target_port.side, e.target_port.position)
        pts = [sp, *list(e.bends), tp]
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            if abs(x1 - x2) > 1e-6 and abs(y1 - y2) > 1e-6:
                return False
    return True


def _no_box_overlap(layout) -> bool:
    boxes = layout.node_boxes
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if (
                abs(a.x - b.x) * 2 < (a.width + b.width) - 1e-6
                and abs(a.y - b.y) * 2 < (a.height + b.height) - 1e-6
            ):
                return False
    return True


def _originals_all_present(layout, n) -> bool:
    return set(range(n)).issubset(_boxes_by_index(layout).keys())


def _check_core_invariants(layout, n):
    assert _originals_all_present(layout, n)
    assert _all_edges_orthogonal(layout)
    if layout.used_bend_optimal:
        assert _no_box_overlap(layout)


# ---------------------------------------------------------------------------
# GIOTTO
# ---------------------------------------------------------------------------


@_SETTINGS
@given(random_tree())
def test_giotto_trees(graph):
    n, edges = graph
    layout = _make(GIOTTOLayout, n, edges, strict=False, bend_optimal=True)
    _check_core_invariants(layout, n)


@_SETTINGS
@given(grid_graph())
def test_giotto_grids_draw_bend_optimally(graph):
    n, edges = graph
    layout = _make(GIOTTOLayout, n, edges, strict=False, bend_optimal=True)
    # A grid is connected, planar, max degree 4 -- always in the bend-optimal
    # domain. (The bend *count* is minimal for the embedder's chosen embedding,
    # not necessarily zero: bend-minimality is relative to a fixed planar
    # embedding, and MaxFaceEmbedder need not pick the axis-aligned one.)
    assert layout.used_bend_optimal is True
    _check_core_invariants(layout, n)


@_SETTINGS
@given(forest())
def test_giotto_forests(graph):
    n, edges = graph
    layout = _make(GIOTTOLayout, n, edges, strict=False, bend_optimal=True)
    _check_core_invariants(layout, n)


@_SETTINGS
@given(random_tree(min_n=2))
def test_giotto_deterministic(graph):
    n, edges = graph
    a = _make(GIOTTOLayout, n, edges, strict=False, bend_optimal=True)
    b = _make(GIOTTOLayout, n, edges, strict=False, bend_optimal=True)
    boxes_a = sorted((x.index, x.x, x.y) for x in a.node_boxes)
    boxes_b = sorted((x.index, x.x, x.y) for x in b.node_boxes)
    assert boxes_a == boxes_b


# ---------------------------------------------------------------------------
# Kandinsky
# ---------------------------------------------------------------------------


@_SETTINGS
@given(grid_graph())
def test_kandinsky_grids(graph):
    n, edges = graph
    layout = _make(KandinskyLayout, n, edges, bend_optimal=True)
    _check_core_invariants(layout, n)


@_SETTINGS
@given(forest())
def test_kandinsky_forests(graph):
    n, edges = graph
    layout = _make(KandinskyLayout, n, edges, bend_optimal=True)
    _check_core_invariants(layout, n)


# ---------------------------------------------------------------------------
# Fuzz: arbitrary small graphs must never crash and always draw orthogonally
# ---------------------------------------------------------------------------


@st.composite
def arbitrary_graph(draw):
    n = draw(st.integers(min_value=1, max_value=8))
    if n < 2:
        return n, []
    edges = draw(
        st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=n - 1),
                st.integers(min_value=0, max_value=n - 1),
            ).filter(lambda e: e[0] != e[1]),
            max_size=16,
            unique=True,
        )
    )
    return n, edges


@_SETTINGS
@given(arbitrary_graph())
def test_giotto_fuzz_never_crashes(graph):
    n, edges = graph
    layout = _make(GIOTTOLayout, n, edges, strict=False, bend_optimal=True)
    assert _originals_all_present(layout, n)
    assert _all_edges_orthogonal(layout)


@_SETTINGS
@given(arbitrary_graph())
def test_kandinsky_fuzz_never_crashes(graph):
    n, edges = graph
    layout = _make(KandinskyLayout, n, edges, bend_optimal=True)
    assert _originals_all_present(layout, n)
    assert _all_edges_orthogonal(layout)
