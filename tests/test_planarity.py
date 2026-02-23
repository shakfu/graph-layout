"""Comprehensive tests for the LR-planarity module."""

from __future__ import annotations

import time

import pytest

from graph_layout.planarity import (
    PlanarEmbedding,
    PlanarityResult,
    check_planarity,
    is_planar,
)
from graph_layout.planarity._lr_planarity import (
    biconnected_components,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_complete_graph(n: int) -> list[tuple[int, int]]:
    """Return edges for K_n."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _make_complete_bipartite(a: int, b: int) -> tuple[int, list[tuple[int, int]]]:
    """Return (num_nodes, edges) for K_{a,b}."""
    n = a + b
    edges = [(i, a + j) for i in range(a) for j in range(b)]
    return n, edges


def _make_cycle(n: int) -> list[tuple[int, int]]:
    """Return edges for C_n."""
    return [(i, (i + 1) % n) for i in range(n)]


def _make_path(n: int) -> list[tuple[int, int]]:
    """Return edges for P_n (path on n vertices)."""
    return [(i, i + 1) for i in range(n - 1)]


def _make_grid(rows: int, cols: int) -> tuple[int, list[tuple[int, int]]]:
    """Return (num_nodes, edges) for a rows x cols grid graph."""
    n = rows * cols
    edges: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            v = r * cols + c
            if c + 1 < cols:
                edges.append((v, v + 1))
            if r + 1 < rows:
                edges.append((v, v + cols))
    return n, edges


def _make_wheel(n: int) -> tuple[int, list[tuple[int, int]]]:
    """Return (num_nodes, edges) for wheel W_n (hub + n-cycle)."""
    # Hub is vertex 0, cycle is vertices 1..n
    num_nodes = n + 1
    edges: list[tuple[int, int]] = []
    for i in range(1, n + 1):
        edges.append((0, i))  # hub to cycle
        edges.append((i, 1 + (i % n)))  # cycle edge
    return num_nodes, edges


def _make_petersen() -> tuple[int, list[tuple[int, int]]]:
    """Return (num_nodes, edges) for the Petersen graph."""
    # Outer cycle: 0-1-2-3-4-0
    # Inner pentagram: 5-7-9-6-8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    edges = [
        # Outer cycle
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 0),
        # Inner pentagram
        (5, 7),
        (7, 9),
        (9, 6),
        (6, 8),
        (8, 5),
        # Spokes
        (0, 5),
        (1, 6),
        (2, 7),
        (3, 8),
        (4, 9),
    ]
    return 10, edges


def _make_tree_binary(depth: int) -> tuple[int, list[tuple[int, int]]]:
    """Return (num_nodes, edges) for a complete binary tree of given depth."""
    # BFS-style numbering
    n = (1 << (depth + 1)) - 1  # 2^(depth+1) - 1
    edges: list[tuple[int, int]] = []
    for v in range(n):
        left = 2 * v + 1
        right = 2 * v + 2
        if left < n:
            edges.append((v, left))
        if right < n:
            edges.append((v, right))
    return n, edges


def _verify_embedding(
    num_nodes: int,
    edges: list[tuple[int, int]],
    result: PlanarityResult,
) -> None:
    """Verify that a planar embedding is consistent."""
    assert result.is_planar
    assert result.embedding is not None

    emb = PlanarEmbedding(result.embedding)

    # Every edge appears in exactly one direction in each endpoint's rotation
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))

    # Check every edge is represented in the embedding
    emb_edges = set()
    for v, neighbors in result.embedding.items():
        for w in neighbors:
            emb_edges.add((min(v, w), max(v, w)))

    # All edges should be in embedding
    for e in edge_set:
        assert e in emb_edges, f"Edge {e} not found in embedding"

    # Check bidirectionality: if w in rotation[v], then v in rotation[w]
    assert emb.verify(), "Embedding failed bidirectional check"


# =========================================================================
# Planar graphs -- should return is_planar=True
# =========================================================================


class TestPlanarGraphs:
    """Test that known planar graphs are correctly identified."""

    def test_empty_graph(self) -> None:
        result = check_planarity(0, [])
        assert result.is_planar

    def test_single_vertex(self) -> None:
        result = check_planarity(1, [])
        assert result.is_planar
        assert result.embedding == {0: []}

    def test_single_edge(self) -> None:
        result = check_planarity(2, [(0, 1)])
        assert result.is_planar
        _verify_embedding(2, [(0, 1)], result)

    def test_path_3(self) -> None:
        edges = _make_path(3)
        result = check_planarity(3, edges)
        assert result.is_planar
        _verify_embedding(3, edges, result)

    def test_path_10(self) -> None:
        edges = _make_path(10)
        result = check_planarity(10, edges)
        assert result.is_planar
        _verify_embedding(10, edges, result)

    def test_cycle_3(self) -> None:
        """Triangle."""
        edges = _make_cycle(3)
        result = check_planarity(3, edges)
        assert result.is_planar
        _verify_embedding(3, edges, result)

    def test_cycle_5(self) -> None:
        edges = _make_cycle(5)
        result = check_planarity(5, edges)
        assert result.is_planar
        _verify_embedding(5, edges, result)

    def test_cycle_100(self) -> None:
        edges = _make_cycle(100)
        result = check_planarity(100, edges)
        assert result.is_planar

    def test_k4(self) -> None:
        """K4 is planar."""
        edges = _make_complete_graph(4)
        result = check_planarity(4, edges)
        assert result.is_planar
        _verify_embedding(4, edges, result)

    def test_k2_3(self) -> None:
        """K_{2,3} is planar."""
        n, edges = _make_complete_bipartite(2, 3)
        result = check_planarity(n, edges)
        assert result.is_planar
        _verify_embedding(n, edges, result)

    def test_wheel_5(self) -> None:
        """W5 (wheel with 5-cycle) is planar."""
        n, edges = _make_wheel(5)
        result = check_planarity(n, edges)
        assert result.is_planar
        _verify_embedding(n, edges, result)

    def test_wheel_6(self) -> None:
        """W6 is planar."""
        n, edges = _make_wheel(6)
        result = check_planarity(n, edges)
        assert result.is_planar

    def test_grid_3x3(self) -> None:
        n, edges = _make_grid(3, 3)
        result = check_planarity(n, edges)
        assert result.is_planar
        _verify_embedding(n, edges, result)

    def test_grid_5x5(self) -> None:
        n, edges = _make_grid(5, 5)
        result = check_planarity(n, edges)
        assert result.is_planar

    def test_binary_tree(self) -> None:
        """Trees are always planar."""
        n, edges = _make_tree_binary(5)
        result = check_planarity(n, edges)
        assert result.is_planar
        _verify_embedding(n, edges, result)

    def test_maximal_planar_k4(self) -> None:
        """K4 is a maximal planar graph (E = 3V - 6 = 6)."""
        edges = _make_complete_graph(4)
        assert len(edges) == 6  # 3*4 - 6
        result = check_planarity(4, edges)
        assert result.is_planar

    def test_two_triangles_sharing_edge(self) -> None:
        """Two triangles sharing an edge (diamond)."""
        edges = [(0, 1), (1, 2), (2, 0), (0, 3), (3, 2)]
        result = check_planarity(4, edges)
        assert result.is_planar
        _verify_embedding(4, edges, result)

    def test_outerplanar_fan(self) -> None:
        """Fan graph: a path with an apex connected to all vertices."""
        n = 6
        # Path: 1-2-3-4-5
        edges = [(i, i + 1) for i in range(1, 5)]
        # Apex 0 connected to all
        edges.extend((0, i) for i in range(1, n))
        result = check_planarity(n, edges)
        assert result.is_planar


# =========================================================================
# Non-planar graphs -- should return is_planar=False
# =========================================================================


class TestNonPlanarGraphs:
    """Test that known non-planar graphs are correctly identified."""

    def test_k5(self) -> None:
        """K5 is non-planar (Kuratowski)."""
        edges = _make_complete_graph(5)
        result = check_planarity(5, edges)
        assert not result.is_planar

    def test_k3_3(self) -> None:
        """K_{3,3} is non-planar (Kuratowski)."""
        n, edges = _make_complete_bipartite(3, 3)
        result = check_planarity(n, edges)
        assert not result.is_planar

    def test_k6(self) -> None:
        """K6 is non-planar (contains K5)."""
        edges = _make_complete_graph(6)
        result = check_planarity(6, edges)
        assert not result.is_planar

    def test_k7(self) -> None:
        edges = _make_complete_graph(7)
        result = check_planarity(7, edges)
        assert not result.is_planar

    def test_k4_4(self) -> None:
        """K_{4,4} is non-planar (contains K_{3,3})."""
        n, edges = _make_complete_bipartite(4, 4)
        result = check_planarity(n, edges)
        assert not result.is_planar

    def test_petersen(self) -> None:
        """Petersen graph is non-planar."""
        n, edges = _make_petersen()
        result = check_planarity(n, edges)
        assert not result.is_planar

    def test_k5_subdivided(self) -> None:
        """K5 with one edge subdivided is still non-planar."""
        # K5 on vertices 0-4, subdivide edge (0,1) with vertex 5
        edges = _make_complete_graph(5)
        # Remove (0,1), add (0,5) and (5,1)
        edges = [(u, v) for u, v in edges if not (u == 0 and v == 1)]
        edges.extend([(0, 5), (5, 1)])
        result = check_planarity(6, edges)
        assert not result.is_planar

    def test_k3_3_subdivided(self) -> None:
        """K_{3,3} with one edge subdivided is still non-planar."""
        n, edges = _make_complete_bipartite(3, 3)
        # Subdivide edge (0, 3) -> (0, 6), (6, 3)
        edges = [(u, v) for u, v in edges if not (u == 0 and v == 3)]
        edges.extend([(0, 6), (6, 3)])
        result = check_planarity(7, edges)
        assert not result.is_planar


# =========================================================================
# Boolean API
# =========================================================================


class TestIsPlanar:
    """Test the simple is_planar() boolean API."""

    def test_planar(self) -> None:
        assert is_planar(4, _make_complete_graph(4))

    def test_non_planar(self) -> None:
        assert not is_planar(5, _make_complete_graph(5))

    def test_empty(self) -> None:
        assert is_planar(0, [])

    def test_single(self) -> None:
        assert is_planar(1, [])


# =========================================================================
# Embedding verification
# =========================================================================


class TestEmbedding:
    """Test embedding correctness for planar graphs."""

    def test_triangle_embedding(self) -> None:
        edges = _make_cycle(3)
        result = check_planarity(3, edges)
        assert result.is_planar
        assert result.embedding is not None

        emb = PlanarEmbedding(result.embedding)
        assert emb.verify()

        # Triangle has 2 faces (interior + exterior)
        faces = emb.faces()
        assert len(faces) == 2

    def test_k4_embedding_euler(self) -> None:
        """K4: V=4, E=6. Euler: F = E - V + 2 = 4."""
        edges = _make_complete_graph(4)
        result = check_planarity(4, edges)
        assert result.is_planar
        assert result.embedding is not None

        emb = PlanarEmbedding(result.embedding)
        assert emb.verify()
        assert emb.num_faces() == 4

    def test_edge_appears_twice(self) -> None:
        """Each undirected edge appears exactly twice across all rotations
        (once as (u,v) and once as (v,u))."""
        edges = _make_complete_graph(4)
        result = check_planarity(4, edges)
        assert result.embedding is not None

        directed_count: dict[tuple[int, int], int] = {}
        for v, neighbors in result.embedding.items():
            for w in neighbors:
                directed_count[(v, w)] = directed_count.get((v, w), 0) + 1

        for v, neighbors in result.embedding.items():
            for w in neighbors:
                assert directed_count.get((v, w), 0) == 1
                assert directed_count.get((w, v), 0) == 1

    def test_cycle_embedding_faces(self) -> None:
        """C_n has exactly 2 faces."""
        for n in [3, 4, 5, 6, 10]:
            edges = _make_cycle(n)
            result = check_planarity(n, edges)
            assert result.is_planar
            assert result.embedding is not None
            emb = PlanarEmbedding(result.embedding)
            assert emb.num_faces() == 2, f"C_{n} should have 2 faces, got {emb.num_faces()}"

    def test_grid_embedding_euler(self) -> None:
        """3x3 grid: V=9, E=12. Euler: F = 12 - 9 + 2 = 5."""
        n, edges = _make_grid(3, 3)
        result = check_planarity(n, edges)
        assert result.is_planar
        assert result.embedding is not None
        emb = PlanarEmbedding(result.embedding)
        assert emb.num_faces() == 5

    def test_outer_face(self) -> None:
        """Outer face of a triangle has more edges than the inner face."""
        edges = _make_cycle(3)
        result = check_planarity(3, edges)
        assert result.embedding is not None
        emb = PlanarEmbedding(result.embedding)
        outer = emb.outer_face()
        assert outer is not None


# =========================================================================
# Edge cases
# =========================================================================


class TestEdgeCases:
    """Test edge cases: self-loops, multi-edges, disconnected graphs."""

    def test_self_loop_ignored(self) -> None:
        """Self-loops do not affect planarity."""
        edges = [(0, 0), (0, 1), (1, 2), (2, 0)]
        result = check_planarity(3, edges)
        assert result.is_planar

    def test_two_parallel_edges_planar(self) -> None:
        """Two parallel edges between same pair is still planar."""
        edges = [(0, 1), (0, 1), (1, 2)]
        result = check_planarity(3, edges)
        assert result.is_planar

    def test_three_parallel_edges_nonplanar(self) -> None:
        """Three parallel edges between same pair is non-planar."""
        edges = [(0, 1), (0, 1), (0, 1)]
        result = check_planarity(2, edges)
        assert not result.is_planar

    def test_disconnected_planar(self) -> None:
        """Two disjoint triangles are planar."""
        edges = [
            (0, 1),
            (1, 2),
            (2, 0),  # triangle 1
            (3, 4),
            (4, 5),
            (5, 3),  # triangle 2
        ]
        result = check_planarity(6, edges)
        assert result.is_planar

    def test_disconnected_one_nonplanar(self) -> None:
        """A triangle + K5 is non-planar."""
        edges = [(0, 1), (1, 2), (2, 0)]  # triangle
        # K5 on vertices 3-7
        edges.extend((i, j) for i in range(3, 8) for j in range(i + 1, 8))
        result = check_planarity(8, edges)
        assert not result.is_planar

    def test_isolated_vertices(self) -> None:
        """Graph with isolated vertices is planar."""
        result = check_planarity(10, [(0, 1)])
        assert result.is_planar

    def test_many_isolated_vertices(self) -> None:
        """100 isolated vertices."""
        result = check_planarity(100, [])
        assert result.is_planar

    def test_two_vertices_no_edge(self) -> None:
        result = check_planarity(2, [])
        assert result.is_planar


# =========================================================================
# Biconnected components
# =========================================================================


class TestBiconnectedComponents:
    """Test the biconnected component decomposition."""

    def test_triangle(self) -> None:
        adj: list[list[int]] = [[1, 2], [0, 2], [0, 1]]
        comps = biconnected_components(3, adj)
        # Triangle is already biconnected
        edge_comps = [c for c in comps if len(c[1]) > 0]
        assert len(edge_comps) == 1

    def test_two_triangles_bridge(self) -> None:
        """Two triangles connected by a bridge edge."""
        # Triangle 1: 0-1-2, Triangle 2: 3-4-5, Bridge: 2-3
        adj: list[list[int]] = [
            [1, 2],  # 0
            [0, 2],  # 1
            [0, 1, 3],  # 2
            [2, 4, 5],  # 3
            [3, 5],  # 4
            [3, 4],  # 5
        ]
        comps = biconnected_components(6, adj)
        edge_comps = [c for c in comps if len(c[1]) > 0]
        # 3 components: triangle1, bridge, triangle2
        assert len(edge_comps) == 3

    def test_single_edge(self) -> None:
        adj: list[list[int]] = [[1], [0]]
        comps = biconnected_components(2, adj)
        edge_comps = [c for c in comps if len(c[1]) > 0]
        assert len(edge_comps) == 1

    def test_isolated_vertex(self) -> None:
        adj: list[list[int]] = [[], [2], [1]]
        comps = biconnected_components(3, adj)
        # One isolated vertex component, one edge component
        assert len(comps) == 2


# =========================================================================
# Integration with GIOTTO
# =========================================================================


class TestGIOTTOIntegration:
    """Test that GIOTTO uses the new planarity module correctly."""

    def test_k5_raises_strict(self) -> None:
        """GIOTTO in strict mode should raise ValueError for K5."""
        from graph_layout import GIOTTOLayout

        nodes = [{"x": i * 10, "y": i * 10} for i in range(5)]
        links = [{"source": i, "target": j} for i in range(5) for j in range(i + 1, 5)]
        layout = GIOTTOLayout(nodes=nodes, links=links, strict=True)
        with pytest.raises(ValueError, match="planar"):
            layout.run()

    def test_k3_3_raises_strict(self) -> None:
        """GIOTTO in strict mode should raise ValueError for K_{3,3}."""
        from graph_layout import GIOTTOLayout

        nodes = [{"x": i * 10, "y": i * 10} for i in range(6)]
        links = [{"source": i, "target": 3 + j} for i in range(3) for j in range(3)]
        layout = GIOTTOLayout(nodes=nodes, links=links, strict=True)
        with pytest.raises(ValueError, match="planar"):
            layout.run()

    def test_k3_3_fallback_non_strict(self) -> None:
        """GIOTTO in non-strict mode should fall back for K_{3,3}."""
        from graph_layout import GIOTTOLayout

        nodes = [{"x": i * 10, "y": i * 10} for i in range(6)]
        links = [{"source": i, "target": 3 + j} for i in range(3) for j in range(3)]
        layout = GIOTTOLayout(nodes=nodes, links=links, strict=False)
        layout.run()
        assert not layout.is_valid_input

    def test_planar_graph_works(self) -> None:
        """GIOTTO should accept a planar degree-4 graph."""
        from graph_layout import GIOTTOLayout

        # Simple path graph: degree <= 2
        nodes = [{"x": i * 10, "y": 0} for i in range(5)]
        links = [{"source": i, "target": i + 1} for i in range(4)]
        layout = GIOTTOLayout(nodes=nodes, links=links, strict=True)
        layout.run()
        assert layout.is_valid_input

    def test_large_nonplanar_detected(self) -> None:
        """K3,3 embedded in a larger graph (n>20) should be detected.

        Uses K3,3 (max degree 3) instead of K5 (max degree 4) to avoid
        hitting the degree-4 check before the planarity check.
        """
        from graph_layout import GIOTTOLayout

        n = 25
        nodes = [{"x": i * 10, "y": i * 10} for i in range(n)]
        # Path on nodes 6..24 + K3,3 on nodes 0..5
        links = [{"source": i, "target": i + 1} for i in range(6, n - 1)]
        links.extend({"source": i, "target": 3 + j} for i in range(3) for j in range(3))
        # Connect K3,3 to the path
        links.append({"source": 5, "target": 6})

        layout = GIOTTOLayout(nodes=nodes, links=links, strict=True)
        with pytest.raises(ValueError, match="planar"):
            layout.run()


# =========================================================================
# Performance
# =========================================================================


class TestPerformance:
    """Performance regression tests."""

    def test_grid_1000_nodes(self) -> None:
        """1000-node grid should complete in < 1 second."""
        n, edges = _make_grid(32, 32)  # 1024 nodes
        start = time.monotonic()
        result = check_planarity(n, edges)
        elapsed = time.monotonic() - start
        assert result.is_planar
        assert elapsed < 1.0, f"Took {elapsed:.2f}s, expected < 1s"

    def test_large_cycle(self) -> None:
        """Large cycle should be fast."""
        n = 5000
        edges = _make_cycle(n)
        start = time.monotonic()
        result = check_planarity(n, edges)
        elapsed = time.monotonic() - start
        assert result.is_planar
        assert elapsed < 2.0, f"Took {elapsed:.2f}s"

    def test_large_nonplanar_fast_reject(self) -> None:
        """K100 should be rejected quickly by edge count."""
        edges = _make_complete_graph(100)
        start = time.monotonic()
        result = check_planarity(100, edges)
        elapsed = time.monotonic() - start
        assert not result.is_planar
        assert elapsed < 0.5, f"Took {elapsed:.2f}s"


# =========================================================================
# Regression / edge case stress
# =========================================================================


class TestRegression:
    """Additional correctness checks."""

    def test_all_small_complete_graphs(self) -> None:
        """K1..K4 planar, K5..K8 non-planar."""
        for n in range(1, 5):
            assert is_planar(n, _make_complete_graph(n)), f"K{n} should be planar"
        for n in range(5, 9):
            assert not is_planar(n, _make_complete_graph(n)), f"K{n} should be non-planar"

    def test_k2_n_bipartite(self) -> None:
        """K_{2,n} is always planar."""
        for n in [1, 2, 3, 5, 10]:
            num, edges = _make_complete_bipartite(2, n)
            assert is_planar(num, edges), f"K_{{2,{n}}} should be planar"

    def test_k3_n_bipartite(self) -> None:
        """K_{3,n} is planar only for n <= 2 (since K_{3,3} is non-planar)."""
        for n in [1, 2]:
            num, edges = _make_complete_bipartite(3, n)
            assert is_planar(num, edges), f"K_{{3,{n}}} should be planar"
        for n in [3, 4, 5]:
            num, edges = _make_complete_bipartite(3, n)
            assert not is_planar(num, edges), f"K_{{3,{n}}} should be non-planar"

    def test_prism_graph(self) -> None:
        """Prism graph (two triangles connected by edges) is planar."""
        edges = [
            (0, 1),
            (1, 2),
            (2, 0),  # triangle 1
            (3, 4),
            (4, 5),
            (5, 3),  # triangle 2
            (0, 3),
            (1, 4),
            (2, 5),  # connecting edges
        ]
        result = check_planarity(6, edges)
        assert result.is_planar
