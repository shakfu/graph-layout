"""Regression tests for planarity correctness.

These tests target two bugs that the coarse-grained assertions in
``test_planarity.py`` failed to catch:

* **C3** -- an off-by-one in the Left-Right nesting-depth computation
  (``_lr_planarity.py``) that produced *order-dependent* false negatives
  (planar graphs reported non-planar) and non-planar (genus > 0) embeddings.
* **H9** -- the Euler ``3n - 6`` edge bound was applied to the raw multi-edge
  count, so planar multigraphs (up to two parallel edges per pair are kept)
  were falsely rejected.

The net here is deliberately property-based rather than example-based, so it
keeps catching regressions of the same *shape*:

1. **Order independence** -- the verdict must not depend on edge/adjacency
   order (the direct signature of C3).
2. **Euler validity** -- every planar embedding must satisfy ``V - E + F == 2``
   per connected component, verified by tracing faces of the returned rotation
   system (C3 produced toroidal embeddings that violate this).
3. **Differential oracle** -- when ``networkx`` is installed, the verdict is
   compared against ``networkx.check_planarity`` over many random graphs.
"""

from __future__ import annotations

import random

import pytest

from graph_layout.planarity import check_planarity, is_planar

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _complete(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _complete_bipartite(a: int, b: int) -> tuple[int, list[tuple[int, int]]]:
    return a + b, [(i, a + j) for i in range(a) for j in range(b)]


def _apollonian(n_extra: int, rng: random.Random) -> tuple[int, list[tuple[int, int]]]:
    """Grow a random planar triangulation (Apollonian network).

    Repeatedly split a random triangular face by inserting a vertex joined to
    the face's three corners. The result is guaranteed planar and provides
    ground truth without an external oracle.
    """
    edges: set[tuple[int, int]] = {(0, 1), (1, 2), (0, 2)}
    faces: list[tuple[int, int, int]] = [(0, 1, 2)]
    nid = 3
    for _ in range(n_extra):
        a, b, c = faces.pop(rng.randrange(len(faces)))
        for x in (a, b, c):
            edges.add((min(x, nid), max(x, nid)))
        faces += [(a, b, nid), (b, c, nid), (a, c, nid)]
        nid += 1
    return nid, list(edges)


def _subdivide(
    n: int, edges: list[tuple[int, int]], rng: random.Random, k: int
) -> tuple[int, list[tuple[int, int]]]:
    """Subdivide ``k`` random edges (topology-preserving; keeps (non-)planarity)."""
    edges = list(edges)
    for _ in range(k):
        u, v = edges.pop(rng.randrange(len(edges)))
        w = n
        n += 1
        edges += [(u, w), (w, v)]
    return n, edges


def _euler_per_component(embedding: dict[int, list[int]]) -> list[tuple[int, int, int]]:
    """Return ``(V, E, F)`` for each edge-bearing component of a rotation system.

    Faces are the orbits of ``phi((u, v)) = (v, w)`` where ``w`` is the neighbor
    following ``u`` in ``v``'s cyclic rotation. For a valid planar embedding of a
    connected graph, ``V - E + F == 2``. Assumes a simple graph (each neighbor
    appears once per rotation), which is the case for every graph tested here.
    """
    adj = {v: list(nbrs) for v, nbrs in embedding.items()}
    seen: set[int] = set()
    out: list[tuple[int, int, int]] = []
    for start in [v for v in adj if adj[v]]:
        if start in seen:
            continue
        comp: list[int] = []
        stack = [start]
        seen.add(start)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)

        pos = {(v, u): i for v in comp for i, u in enumerate(adj[v])}
        darts = {(v, u) for v in comp for u in adj[v]}
        unvisited = set(darts)
        faces = 0
        while unvisited:
            faces += 1
            first = next(iter(unvisited))
            d = first
            while True:
                unvisited.discard(d)
                u, v = d
                i = pos[(v, u)]
                w = adj[v][(i + 1) % len(adj[v])]
                d = (v, w)
                if d == first:
                    break
        out.append((len(comp), len(darts) // 2, faces))
    return out


def _assert_embedding_planar(embedding: dict[int, list[int]]) -> None:
    for v, e, f in _euler_per_component(embedding):
        assert v - e + f == 2, f"non-planar embedding: V={v} E={e} F={f} (V-E+F={v - e + f})"


# ---------------------------------------------------------------------------
# C3: specific repros and order independence
# ---------------------------------------------------------------------------


def test_c3_maximal_planar_repro_both_orders():
    """The exact graph from the review: planar in every edge order, valid embedding."""
    edges = [(0, 2), (2, 4), (0, 4), (1, 2), (3, 4), (1, 3), (2, 3), (1, 4), (0, 3)]
    for order in (edges, sorted(edges), list(reversed(edges))):
        r = check_planarity(5, order)
        assert r.is_planar, f"maximal planar graph reported non-planar for order {order}"
        _assert_embedding_planar(r.embedding)


def test_planar_verdict_is_order_independent():
    """Verdict must not depend on edge order (the direct signature of C3)."""
    rng = random.Random(20260703)
    for _ in range(600):
        n = rng.randrange(3, 11)
        pairs = _complete(n)
        edges = rng.sample(pairs, rng.randrange(0, len(pairs) + 1))
        base = check_planarity(n, edges).is_planar
        for _ in range(4):
            shuffled = edges[:]
            rng.shuffle(shuffled)
            assert check_planarity(n, shuffled).is_planar == base, (
                f"order-dependent verdict on n={n} edges={edges}"
            )


def test_planar_embeddings_satisfy_euler():
    """Every planar result yields a genuinely planar (V-E+F==2) embedding."""
    rng = random.Random(4242)
    for _ in range(400):
        n, edges = _apollonian(rng.randrange(0, 14), rng)
        for order in (edges, list(reversed(edges))):
            r = check_planarity(n, order)
            assert r.is_planar  # Apollonian networks are planar by construction
            _assert_embedding_planar(r.embedding)


def test_nonplanar_subdivisions_stay_nonplanar():
    """K5 / K3,3 and their subdivisions are non-planar under any edge order."""
    rng = random.Random(77)
    k33_n, k33 = _complete_bipartite(3, 3)
    for base_n, base_e in ((5, _complete(5)), (k33_n, k33)):
        for _ in range(150):
            n, e = _subdivide(base_n, base_e, rng, rng.randrange(0, 4))
            for _ in range(3):
                shuffled = e[:]
                rng.shuffle(shuffled)
                assert not check_planarity(n, shuffled).is_planar


# ---------------------------------------------------------------------------
# H9: planar multigraphs must not be rejected by the Euler edge bound
# ---------------------------------------------------------------------------


def test_multigraph_triangle_with_doubled_edge_is_planar():
    # 4 edges, n=3: raw count 4 > 3n-6=3 would falsely reject; simple count 3 is fine.
    assert is_planar(3, [(0, 1), (0, 1), (1, 2), (2, 0)])


def test_planar_graph_with_parallel_edges_is_planar():
    # A 4-cycle with one doubled edge; still planar.
    assert is_planar(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 1)])


def test_parallel_edges_do_not_mask_nonplanarity():
    # K5 stays non-planar even if we double one of its edges.
    edges = _complete(5) + [(0, 1)]
    assert not is_planar(5, edges)


# ---------------------------------------------------------------------------
# Differential oracle (networkx), skipped if the oracle is unavailable
# ---------------------------------------------------------------------------


def test_matches_networkx_over_random_graphs():
    nx = pytest.importorskip("networkx")
    rng = random.Random(20260703)
    mismatches = []
    for _ in range(2000):
        n = rng.randrange(3, 12)
        pairs = _complete(n)
        edges = rng.sample(pairs, rng.randrange(0, len(pairs) + 1))
        ours = check_planarity(n, edges).is_planar
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from(edges)
        theirs, _ = nx.check_planarity(g, counterexample=False)
        if ours != theirs:
            mismatches.append((n, sorted(edges), ours, theirs))
    assert not mismatches, (
        f"{len(mismatches)} planarity mismatches vs networkx; first: {mismatches[0]}"
    )
