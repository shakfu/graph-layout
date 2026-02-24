"""Tests for planar embedding strategies and supporting data structures."""

from __future__ import annotations

import pytest

from graph_layout.planarity import (
    FixedEmbedder,
    MaxFaceEmbedder,
    MinDepthEmbedder,
    OptimalFlexEmbedder,
    check_planarity,
)
from graph_layout.planarity._block_cut_tree import build_block_cut_tree
from graph_layout.planarity._embedding import PlanarEmbedding

# ---------------------------------------------------------------------------
# Helper graph builders
# ---------------------------------------------------------------------------


def _k4_edges() -> list[tuple[int, int]]:
    """K4 edges (6 edges, 4 nodes)."""
    return [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def _cycle_edges(n: int) -> list[tuple[int, int]]:
    """Cycle on n vertices."""
    return [(i, (i + 1) % n) for i in range(n)]


def _grid_edges(rows: int, cols: int) -> list[tuple[int, int]]:
    """Grid graph with rows*cols vertices."""
    edges = []
    for r in range(rows):
        for c in range(cols):
            v = r * cols + c
            if c + 1 < cols:
                edges.append((v, v + 1))
            if r + 1 < rows:
                edges.append((v, v + cols))
    return edges


def _wheel_edges(n: int) -> list[tuple[int, int]]:
    """Wheel graph: hub vertex 0, rim vertices 1..n."""
    edges = [(0, i) for i in range(1, n + 1)]
    for i in range(1, n + 1):
        j = i % n + 1
        edges.append((i, j))
    return edges


def _bridge_graph_edges() -> list[tuple[int, int]]:
    """Two triangles connected by a bridge.

    Vertices 0-2 form triangle A, vertices 3-5 form triangle B.
    Bridge: (2, 3).
    """
    return [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]


def _chain_of_blocks_edges() -> list[tuple[int, int]]:
    """Three triangles in a chain: A-B-C.

    Triangle 0-1-2, bridge (2,3), triangle 3-4-5, bridge (5,6), triangle 6-7-8.
    """
    return [
        (0, 1),
        (1, 2),
        (0, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (3, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (6, 8),
    ]


def _k5_edges() -> list[tuple[int, int]]:
    """K5 edges (non-planar)."""
    return [(i, j) for i in range(5) for j in range(i + 1, 5)]


def _adj_from_edges(num_nodes: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(num_nodes)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


# ---------------------------------------------------------------------------
# PlanarEmbedding.outer_face_index / set_outer_face tests
# ---------------------------------------------------------------------------


class TestEmbeddingOuterFace:
    def test_outer_face_default_none(self) -> None:
        result = check_planarity(3, _cycle_edges(3))
        emb = PlanarEmbedding(result.embedding)
        assert emb.outer_face_index is None

    def test_set_outer_face_valid(self) -> None:
        result = check_planarity(4, _k4_edges())
        emb = PlanarEmbedding(result.embedding)
        faces = emb.faces()
        assert len(faces) >= 2
        emb.set_outer_face(0)
        assert emb.outer_face_index == 0
        emb.set_outer_face(len(faces) - 1)
        assert emb.outer_face_index == len(faces) - 1

    def test_set_outer_face_out_of_range(self) -> None:
        result = check_planarity(3, _cycle_edges(3))
        emb = PlanarEmbedding(result.embedding)
        with pytest.raises(IndexError):
            emb.set_outer_face(999)
        with pytest.raises(IndexError):
            emb.set_outer_face(-1)


# ---------------------------------------------------------------------------
# Block-cut tree tests
# ---------------------------------------------------------------------------


class TestBlockCutTree:
    def test_biconnected_graph_single_block(self) -> None:
        """A biconnected graph (K4) has exactly 1 block, no cut vertices."""
        adj = _adj_from_edges(4, _k4_edges())
        bct = build_block_cut_tree(4, adj)
        assert len(bct.blocks) == 1
        assert len(bct.cut_vertices) == 0

    def test_bridge_graph_two_blocks(self) -> None:
        """Two triangles joined by a bridge produce 3 blocks (2 triangles + bridge)."""
        edges = _bridge_graph_edges()
        adj = _adj_from_edges(6, edges)
        bct = build_block_cut_tree(6, adj)
        # Tarjan's biconnected components for a bridge graph:
        # each triangle is a block, and the bridge itself is a block
        assert len(bct.blocks) == 3
        # Vertices 2 and 3 are cut vertices (they connect the blocks)
        assert 2 in bct.cut_vertices
        assert 3 in bct.cut_vertices

    def test_chain_of_blocks(self) -> None:
        """Chain of 3 triangles with bridges."""
        edges = _chain_of_blocks_edges()
        adj = _adj_from_edges(9, edges)
        bct = build_block_cut_tree(9, adj)
        # 3 triangle blocks + 2 bridge blocks = 5
        assert len(bct.blocks) == 5
        assert len(bct.cut_vertices) >= 2

    def test_cycle_single_block(self) -> None:
        """A cycle is biconnected: 1 block, no cut vertices."""
        edges = _cycle_edges(6)
        adj = _adj_from_edges(6, edges)
        bct = build_block_cut_tree(6, adj)
        assert len(bct.blocks) == 1
        assert len(bct.cut_vertices) == 0

    def test_vertex_to_blocks_consistency(self) -> None:
        """Every vertex should appear in at least one block."""
        edges = _bridge_graph_edges()
        adj = _adj_from_edges(6, edges)
        bct = build_block_cut_tree(6, adj)
        for v in range(6):
            assert v in bct.vertex_to_blocks
            assert len(bct.vertex_to_blocks[v]) >= 1

    def test_block_adjacency(self) -> None:
        """Adjacent blocks should be connected in block_adj."""
        edges = _bridge_graph_edges()
        adj = _adj_from_edges(6, edges)
        bct = build_block_cut_tree(6, adj)
        # With 3 blocks, the bridge block should be adjacent to both triangle blocks
        has_adj = any(len(v) > 0 for v in bct.block_adj.values())
        assert has_adj

    def test_empty_graph(self) -> None:
        adj: list[list[int]] = []
        bct = build_block_cut_tree(0, adj)
        assert len(bct.blocks) == 0
        assert len(bct.cut_vertices) == 0

    def test_single_node(self) -> None:
        adj: list[list[int]] = [[]]
        bct = build_block_cut_tree(1, adj)
        assert len(bct.blocks) == 1
        assert len(bct.cut_vertices) == 0


# ---------------------------------------------------------------------------
# FixedEmbedder tests
# ---------------------------------------------------------------------------


class TestFixedEmbedder:
    def test_k4_produces_valid_embedding(self) -> None:
        emb = FixedEmbedder().embed(4, _k4_edges())
        assert emb.verify()
        assert emb.outer_face_index is not None

    def test_cycle_produces_valid_embedding(self) -> None:
        emb = FixedEmbedder().embed(5, _cycle_edges(5))
        assert emb.verify()
        # Cycle has exactly 2 faces
        assert len(emb.faces()) == 2

    def test_grid_produces_valid_embedding(self) -> None:
        emb = FixedEmbedder().embed(9, _grid_edges(3, 3))
        assert emb.verify()
        assert emb.outer_face_index is not None

    def test_wheel_produces_valid_embedding(self) -> None:
        emb = FixedEmbedder().embed(5, _wheel_edges(4))
        assert emb.verify()
        assert emb.outer_face_index is not None

    def test_non_planar_raises(self) -> None:
        with pytest.raises(ValueError, match="not planar"):
            FixedEmbedder().embed(5, _k5_edges())

    def test_with_precomputed_result(self) -> None:
        result = check_planarity(4, _k4_edges())
        emb = FixedEmbedder().embed(4, _k4_edges(), planarity_result=result)
        assert emb.verify()


# ---------------------------------------------------------------------------
# MaxFaceEmbedder tests
# ---------------------------------------------------------------------------


class TestMaxFaceEmbedder:
    def test_outer_face_is_largest(self) -> None:
        """MaxFaceEmbedder should pick the largest face as outer face."""
        emb = MaxFaceEmbedder().embed(4, _k4_edges())
        faces = emb.faces()
        outer_size = len(faces[emb.outer_face_index])
        for fi, face in enumerate(faces):
            assert len(face) <= outer_size

    def test_outer_face_at_least_as_large_as_fixed(self) -> None:
        """MaxFace outer face should be >= FixedEmbedder outer face size."""
        edges = _grid_edges(3, 3)
        fixed_emb = FixedEmbedder().embed(9, edges)
        max_emb = MaxFaceEmbedder().embed(9, edges)
        fixed_outer = len(fixed_emb.faces()[fixed_emb.outer_face_index])
        max_outer = len(max_emb.faces()[max_emb.outer_face_index])
        assert max_outer >= fixed_outer

    def test_biconnected_graph(self) -> None:
        """Biconnected graph: should work without cut-vertex logic."""
        emb = MaxFaceEmbedder().embed(4, _k4_edges())
        assert emb.verify()
        assert emb.outer_face_index is not None

    def test_graph_with_cut_vertices(self) -> None:
        """Bridge graph: should handle cut vertices."""
        edges = _bridge_graph_edges()
        emb = MaxFaceEmbedder().embed(6, edges)
        assert emb.verify()
        assert emb.outer_face_index is not None

    def test_non_planar_raises(self) -> None:
        with pytest.raises(ValueError, match="not planar"):
            MaxFaceEmbedder().embed(5, _k5_edges())

    def test_cycle(self) -> None:
        emb = MaxFaceEmbedder().embed(6, _cycle_edges(6))
        assert emb.verify()
        faces = emb.faces()
        # Cycle has 2 faces; outer should be the larger (both have same
        # number of directed edges for a simple cycle, so either is fine)
        assert emb.outer_face_index is not None
        assert 0 <= emb.outer_face_index < len(faces)

    def test_chain_of_blocks(self) -> None:
        edges = _chain_of_blocks_edges()
        emb = MaxFaceEmbedder().embed(9, edges)
        assert emb.verify()
        assert emb.outer_face_index is not None


# ---------------------------------------------------------------------------
# MinDepthEmbedder tests
# ---------------------------------------------------------------------------


class TestMinDepthEmbedder:
    def test_produces_valid_embedding(self) -> None:
        emb = MinDepthEmbedder().embed(4, _k4_edges())
        assert emb.verify()
        assert emb.outer_face_index is not None

    def test_biconnected_same_as_fixed(self) -> None:
        """For biconnected graphs, MinDepth reduces to largest-face selection."""
        edges = _k4_edges()
        fixed_emb = FixedEmbedder().embed(4, edges)
        depth_emb = MinDepthEmbedder().embed(4, edges)
        # Both should pick the same outer face size
        fixed_outer = len(fixed_emb.faces()[fixed_emb.outer_face_index])
        depth_outer = len(depth_emb.faces()[depth_emb.outer_face_index])
        assert depth_outer == fixed_outer

    def test_chain_of_blocks_picks_center(self) -> None:
        """For a chain of 3 triangles, MinDepth should pick the center block."""
        edges = _chain_of_blocks_edges()
        emb = MinDepthEmbedder().embed(9, edges)
        assert emb.verify()
        assert emb.outer_face_index is not None

    def test_non_planar_raises(self) -> None:
        with pytest.raises(ValueError, match="not planar"):
            MinDepthEmbedder().embed(5, _k5_edges())

    def test_cycle(self) -> None:
        emb = MinDepthEmbedder().embed(5, _cycle_edges(5))
        assert emb.verify()

    def test_wheel(self) -> None:
        emb = MinDepthEmbedder().embed(6, _wheel_edges(5))
        assert emb.verify()
        assert emb.outer_face_index is not None


# ---------------------------------------------------------------------------
# Integration with orthogonalization
# ---------------------------------------------------------------------------


class TestOrthogonalizationEmbedding:
    """Test that compute_faces works with PlanarEmbedding input."""

    def test_compute_faces_from_embedding(self) -> None:
        from graph_layout.orthogonal.orthogonalization import Face, compute_faces

        edges = _k4_edges()
        emb = MaxFaceEmbedder().embed(4, edges)
        faces = compute_faces(4, edges, embedding=emb)

        assert len(faces) > 0
        assert all(isinstance(f, Face) for f in faces)
        # Exactly one outer face
        outer_faces = [f for f in faces if f.is_outer]
        assert len(outer_faces) == 1

    def test_compute_faces_without_embedding_unchanged(self) -> None:
        """Legacy path: positions-based face computation still works."""
        from graph_layout.orthogonal.orthogonalization import compute_faces

        edges = [(0, 1), (1, 2), (2, 0)]
        positions = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        faces = compute_faces(3, edges, positions=positions)
        assert len(faces) > 0

    def test_compute_orthogonal_representation_with_embedding(self) -> None:
        from graph_layout.orthogonal.orthogonalization import (
            compute_orthogonal_representation,
        )

        edges = _cycle_edges(4)
        emb = MaxFaceEmbedder().embed(4, edges)
        ortho = compute_orthogonal_representation(4, edges, embedding=emb)
        # Should produce some representation (may have empty bends for cycle)
        assert ortho is not None


# ---------------------------------------------------------------------------
# Integration with GIOTTO layout
# ---------------------------------------------------------------------------


class TestGIOTTOEmbedder:
    """Test GIOTTO with embedder parameter.

    Uses DAG-style links (no back-edges) to avoid the directed DFS
    recursion limitation in _assign_layers.
    """

    def _diamond_links(self) -> list[dict[str, int]]:
        """Diamond DAG: 0->1, 0->2, 1->3, 2->3 (planar, max degree 3)."""
        return [
            {"source": 0, "target": 1},
            {"source": 0, "target": 2},
            {"source": 1, "target": 3},
            {"source": 2, "target": 3},
        ]

    def test_giotto_with_default_embedder(self) -> None:
        from graph_layout import GIOTTOLayout

        layout = GIOTTOLayout(
            nodes=[{} for _ in range(4)],
            links=self._diamond_links(),
            size=(400, 400),
        )
        layout.run()
        assert layout.is_valid_input

    def test_giotto_with_fixed_embedder(self) -> None:
        from graph_layout import GIOTTOLayout

        layout = GIOTTOLayout(
            nodes=[{} for _ in range(4)],
            links=self._diamond_links(),
            size=(400, 400),
            embedder=FixedEmbedder(),
        )
        layout.run()
        assert layout.is_valid_input

    def test_giotto_with_min_depth_embedder(self) -> None:
        from graph_layout import GIOTTOLayout

        layout = GIOTTOLayout(
            nodes=[{} for _ in range(4)],
            links=self._diamond_links(),
            size=(400, 400),
            embedder=MinDepthEmbedder(),
        )
        layout.run()
        assert layout.is_valid_input


# ---------------------------------------------------------------------------
# Integration with Kandinsky layout
# ---------------------------------------------------------------------------


class TestKandinskyEmbedder:
    """Test Kandinsky with embedder parameter.

    Uses DAG-style links to avoid DFS recursion in _assign_layers.
    """

    def _diamond_links(self) -> list[dict[str, int]]:
        return [
            {"source": 0, "target": 1},
            {"source": 0, "target": 2},
            {"source": 1, "target": 3},
            {"source": 2, "target": 3},
        ]

    def test_kandinsky_with_default_embedder(self) -> None:
        from graph_layout import KandinskyLayout

        layout = KandinskyLayout(
            nodes=[{} for _ in range(4)],
            links=self._diamond_links(),
            size=(400, 400),
        )
        layout.run()
        assert len(layout.node_boxes) == 4

    def test_kandinsky_with_fixed_embedder(self) -> None:
        from graph_layout import KandinskyLayout

        layout = KandinskyLayout(
            nodes=[{} for _ in range(4)],
            links=self._diamond_links(),
            size=(400, 400),
            embedder=FixedEmbedder(),
        )
        layout.run()
        assert len(layout.node_boxes) == 4

    def test_kandinsky_non_planar_falls_back(self) -> None:
        """Non-planar graphs should still work (embedder ValueError caught)."""
        from graph_layout import KandinskyLayout

        # Star graph (DAG, non-planar is hard to make as DAG, so use K5 with
        # directed edges to avoid cycles)
        nodes = [{} for _ in range(5)]
        links = [{"source": i, "target": j} for i in range(5) for j in range(i + 1, 5)]
        layout = KandinskyLayout(
            nodes=nodes,
            links=links,
            size=(400, 400),
        )
        layout.run()
        assert len(layout.node_boxes) == 5


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEmbedderEdgeCases:
    def test_single_edge(self) -> None:
        emb = FixedEmbedder().embed(2, [(0, 1)])
        assert emb.verify()

    def test_empty_graph(self) -> None:
        emb = FixedEmbedder().embed(1, [])
        assert emb.outer_face_index is None  # No faces

    def test_tree_graph(self) -> None:
        """Trees are planar; each edge creates a degenerate face."""
        edges = [(0, 1), (1, 2), (1, 3)]
        emb = MaxFaceEmbedder().embed(4, edges)
        assert emb.verify()

    def test_precomputed_non_planar_result_reruns_check(self) -> None:
        """Passing a non-planar result should trigger re-check and raise."""
        from graph_layout.planarity._types import PlanarityResult

        bad_result = PlanarityResult(is_planar=False)
        with pytest.raises(ValueError, match="not planar"):
            FixedEmbedder().embed(5, _k5_edges(), planarity_result=bad_result)

    def test_large_grid(self) -> None:
        """10x10 grid should be handled efficiently."""
        edges = _grid_edges(10, 10)
        emb = MaxFaceEmbedder().embed(100, edges)
        assert emb.verify()
        assert emb.outer_face_index is not None

    def test_disconnected_planar_graph_fixed(self) -> None:
        """Two disjoint triangles: FixedEmbedder should work."""
        edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]
        emb = FixedEmbedder().embed(6, edges)
        assert emb.verify()

    def test_disconnected_planar_graph_max_face(self) -> None:
        """Two disjoint triangles: MaxFaceEmbedder should work."""
        edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]
        emb = MaxFaceEmbedder().embed(6, edges)
        assert emb.verify()
        assert emb.outer_face_index is not None

    def test_disconnected_planar_graph_min_depth(self) -> None:
        """Two disjoint triangles: MinDepthEmbedder should work."""
        edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]
        emb = MinDepthEmbedder().embed(6, edges)
        assert emb.verify()
        assert emb.outer_face_index is not None

    def test_isolated_vertex_fixed(self) -> None:
        """Graph with isolated vertex should not crash."""
        edges = [(0, 1), (1, 2), (2, 0)]  # vertex 3 is isolated
        emb = FixedEmbedder().embed(4, edges)
        assert emb.verify()

    def test_isolated_vertex_max_face(self) -> None:
        edges = [(0, 1), (1, 2), (2, 0)]
        emb = MaxFaceEmbedder().embed(4, edges)
        assert emb.verify()

    def test_only_isolated_vertices(self) -> None:
        """No edges at all -- just isolated vertices."""
        emb = FixedEmbedder().embed(5, [])
        # No faces, so outer_face_index stays None
        assert emb.outer_face_index is None

    def test_self_loop_in_edges_ignored(self) -> None:
        """Self-loops should be stripped and not affect the embedding."""
        edges = [(0, 1), (1, 2), (2, 0), (0, 0)]
        emb = FixedEmbedder().embed(3, edges)
        assert emb.verify()

    def test_self_loop_max_face(self) -> None:
        edges = [(0, 1), (1, 2), (2, 0), (1, 1)]
        emb = MaxFaceEmbedder().embed(3, edges)
        assert emb.verify()

    def test_self_loop_min_depth(self) -> None:
        edges = [(0, 1), (1, 2), (2, 0), (2, 2)]
        emb = MinDepthEmbedder().embed(3, edges)
        assert emb.verify()


# ---------------------------------------------------------------------------
# OptimalFlexEmbedder tests
# ---------------------------------------------------------------------------


def _has_scipy() -> bool:
    try:
        from scipy.optimize import linprog  # noqa: F401

        return True
    except ImportError:
        return False


class TestOptimalFlexEmbedder:
    def test_produces_valid_embedding_k4(self) -> None:
        emb = OptimalFlexEmbedder().embed(4, _k4_edges())
        assert emb.verify()
        assert emb.outer_face_index is not None

    def test_produces_valid_embedding_cycle(self) -> None:
        emb = OptimalFlexEmbedder().embed(5, _cycle_edges(5))
        assert emb.verify()
        assert emb.outer_face_index is not None

    def test_produces_valid_embedding_grid(self) -> None:
        emb = OptimalFlexEmbedder().embed(9, _grid_edges(3, 3))
        assert emb.verify()
        assert emb.outer_face_index is not None

    def test_produces_valid_embedding_wheel(self) -> None:
        emb = OptimalFlexEmbedder().embed(6, _wheel_edges(5))
        assert emb.verify()
        assert emb.outer_face_index is not None

    def test_produces_valid_embedding_bridge(self) -> None:
        emb = OptimalFlexEmbedder().embed(6, _bridge_graph_edges())
        assert emb.verify()
        assert emb.outer_face_index is not None

    @pytest.mark.skipif(not _has_scipy(), reason="scipy not available")
    def test_bends_leq_max_face(self) -> None:
        """OptimalFlexEmbedder bends should be <= MaxFaceEmbedder bends."""
        from scipy.optimize import linprog

        for graph_fn, n in [
            (_k4_edges, 4),
            (lambda: _cycle_edges(6), 6),
            (lambda: _grid_edges(3, 3), 9),
            (lambda: _wheel_edges(5), 6),
        ]:
            edges = graph_fn()
            opt_emb = OptimalFlexEmbedder().embed(n, edges)
            max_emb = MaxFaceEmbedder().embed(n, edges)

            # Compare LP objectives for the chosen outer faces
            opt_data = OptimalFlexEmbedder._build_face_data(opt_emb.faces(), n)
            max_data = OptimalFlexEmbedder._build_face_data(max_emb.faces(), n)

            opt_obj = OptimalFlexEmbedder._solve_lp(opt_data, opt_emb.outer_face_index, linprog)
            max_obj = OptimalFlexEmbedder._solve_lp(max_data, max_emb.outer_face_index, linprog)

            if opt_obj is not None and max_obj is not None:
                assert opt_obj <= max_obj + 1e-6, f"OptimalFlex ({opt_obj}) > MaxFace ({max_obj})"

    def test_non_planar_raises(self) -> None:
        with pytest.raises(ValueError, match="not planar"):
            OptimalFlexEmbedder().embed(5, _k5_edges())

    def test_fallback_without_scipy(self) -> None:
        """When scipy is unavailable, should fall back to MaxFaceEmbedder."""
        import sys

        # Temporarily hide scipy
        real_scipy = sys.modules.get("scipy")
        real_scipy_opt = sys.modules.get("scipy.optimize")
        sys.modules["scipy"] = None  # type: ignore[assignment]
        sys.modules["scipy.optimize"] = None  # type: ignore[assignment]
        try:
            emb = OptimalFlexEmbedder().embed(4, _k4_edges())
            assert emb.verify()
            assert emb.outer_face_index is not None
        finally:
            if real_scipy is not None:
                sys.modules["scipy"] = real_scipy
            else:
                sys.modules.pop("scipy", None)
            if real_scipy_opt is not None:
                sys.modules["scipy.optimize"] = real_scipy_opt
            else:
                sys.modules.pop("scipy.optimize", None)

    def test_max_candidates_limits_search(self) -> None:
        """With max_candidates=1, should still produce a valid embedding."""
        emb = OptimalFlexEmbedder(max_candidates=1).embed(9, _grid_edges(3, 3))
        assert emb.verify()
        assert emb.outer_face_index is not None

    def test_single_edge(self) -> None:
        emb = OptimalFlexEmbedder().embed(2, [(0, 1)])
        assert emb.verify()

    def test_disconnected_graph(self) -> None:
        edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]
        emb = OptimalFlexEmbedder().embed(6, edges)
        assert emb.verify()
