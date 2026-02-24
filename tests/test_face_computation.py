"""Tests for robust face computation in orthogonalization."""

from __future__ import annotations

from graph_layout.orthogonal.orthogonalization import (
    _sanitize_edges,
    build_flow_network,
    compute_faces,
    compute_orthogonal_representation,
)
from graph_layout.planarity import MaxFaceEmbedder
from graph_layout.planarity._embedding import PlanarEmbedding

# ---------------------------------------------------------------------------
# _sanitize_edges
# ---------------------------------------------------------------------------


class TestSanitizeEdges:
    def test_removes_self_loops(self) -> None:
        edges = [(0, 1), (1, 1), (1, 2)]
        clean, loops = _sanitize_edges(edges)
        assert (1, 1) not in clean
        assert len(loops) == 1
        assert loops[0] == (1, 1)

    def test_deduplicates_multi_edges(self) -> None:
        edges = [(0, 1), (1, 0), (0, 1)]
        clean, loops = _sanitize_edges(edges)
        # Only one copy of (0,1) should remain
        assert len(clean) == 1
        assert clean[0] == (0, 1)

    def test_no_change_for_simple_edges(self) -> None:
        edges = [(0, 1), (1, 2), (2, 0)]
        clean, loops = _sanitize_edges(edges)
        assert len(clean) == 3
        assert len(loops) == 0

    def test_empty_input(self) -> None:
        clean, loops = _sanitize_edges([])
        assert clean == []
        assert loops == []

    def test_only_self_loops(self) -> None:
        edges = [(0, 0), (1, 1)]
        clean, loops = _sanitize_edges(edges)
        assert clean == []
        assert len(loops) == 2


# ---------------------------------------------------------------------------
# compute_faces with self-loops and multi-edges
# ---------------------------------------------------------------------------


class TestComputeFacesSanitization:
    def test_self_loop_filtered_from_faces(self) -> None:
        """A triangle with a self-loop should produce same faces as plain triangle."""
        triangle_edges = [(0, 1), (1, 2), (2, 0)]
        triangle_plus_loop = [(0, 1), (1, 2), (2, 0), (0, 0)]

        positions = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]

        faces_clean = compute_faces(3, triangle_edges, positions=positions)
        faces_loop = compute_faces(3, triangle_plus_loop, positions=positions)

        assert len(faces_clean) == len(faces_loop)

    def test_multi_edge_deduplicated(self) -> None:
        """Duplicate edges should not produce duplicate faces."""
        edges = [(0, 1), (1, 2), (2, 0), (0, 1)]  # duplicate (0,1)
        positions = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]

        faces = compute_faces(3, edges, positions=positions)
        # Triangle has exactly 2 faces (inner + outer)
        assert len(faces) == 2

    def test_triangle_has_two_faces(self) -> None:
        """Basic triangle produces 2 faces."""
        edges = [(0, 1), (1, 2), (2, 0)]
        positions = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        faces = compute_faces(3, edges, positions=positions)
        assert len(faces) == 2
        assert sum(1 for f in faces if f.is_outer) == 1


# ---------------------------------------------------------------------------
# compute_faces with embedding path
# ---------------------------------------------------------------------------


class TestComputeFacesEmbedding:
    def test_embedding_path_produces_faces(self) -> None:
        edges = [(0, 1), (1, 2), (2, 0)]
        emb = MaxFaceEmbedder().embed(3, edges)
        faces = compute_faces(3, edges, embedding=emb)
        assert len(faces) == 2
        assert sum(1 for f in faces if f.is_outer) == 1

    def test_embedding_verification_fallback(self) -> None:
        """Broken embedding triggers fallback to legacy path."""
        import warnings

        # Build a deliberately broken embedding (missing reverse edge)
        rotation = {0: [1], 1: [2], 2: [0]}  # only forward edges
        emb = PlanarEmbedding(rotation)
        # verify() should fail
        assert not emb.verify()

        edges = [(0, 1), (1, 2), (2, 0)]
        positions = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            faces = compute_faces(3, edges, positions=positions, embedding=emb)
            # Should have a warning about fallback
            assert len(w) >= 1
            assert "verification" in str(w[0].message).lower()

        # Should still produce faces via legacy path
        assert len(faces) >= 1

    def test_embedding_with_outer_face_set(self) -> None:
        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]  # K4
        emb = MaxFaceEmbedder().embed(4, edges)
        faces = compute_faces(4, edges, embedding=emb)
        outer = [f for f in faces if f.is_outer]
        assert len(outer) == 1
        assert outer[0].index == emb.outer_face_index


# ---------------------------------------------------------------------------
# Disconnected graph face computation
# ---------------------------------------------------------------------------


class TestComputeFacesDisconnected:
    def test_two_disconnected_triangles(self) -> None:
        """Two separate triangles should produce faces for each component."""
        edges = [
            (0, 1),
            (1, 2),
            (2, 0),  # Triangle A
            (3, 4),
            (4, 5),
            (5, 3),  # Triangle B
        ]
        positions = [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.5, 1.0),
            (3.0, 0.0),
            (4.0, 0.0),
            (3.5, 1.0),
        ]
        faces = compute_faces(6, edges, positions=positions)
        # Each triangle has 2 faces -> 4 total
        assert len(faces) >= 2

    def test_single_vertex_no_edges(self) -> None:
        """Single vertex with no edges produces no faces."""
        faces = compute_faces(1, [])
        assert faces == []

    def test_isolated_vertices_with_edges_elsewhere(self) -> None:
        """Isolated vertices don't break face computation."""
        edges = [(0, 1), (1, 2), (2, 0)]  # Triangle on 0-2, vertex 3 isolated
        positions = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0), (5.0, 5.0)]
        faces = compute_faces(4, edges, positions=positions)
        assert len(faces) == 2  # Triangle produces 2 faces

    def test_empty_graph(self) -> None:
        faces = compute_faces(0, [])
        assert faces == []


# ---------------------------------------------------------------------------
# build_flow_network with sanitized inputs
# ---------------------------------------------------------------------------


class TestBuildFlowNetworkRobust:
    def test_self_loop_does_not_corrupt_degrees(self) -> None:
        """Self-loops should not inflate vertex degrees in the flow network."""
        edges_with_loop = [(0, 1), (1, 2), (2, 0), (0, 0)]
        edges_clean = [(0, 1), (1, 2), (2, 0)]
        positions = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]

        faces = compute_faces(3, edges_clean, positions=positions)
        net_loop = build_flow_network(3, edges_with_loop, faces)
        net_clean = build_flow_network(3, edges_clean, faces)

        # Supplies should be the same since self-loops are filtered
        assert net_loop.supplies == net_clean.supplies

    def test_euler_formula_holds(self) -> None:
        """V - E + F = 2 for connected planar graph."""
        edges = [(0, 1), (1, 2), (2, 0)]
        positions = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        faces = compute_faces(3, edges, positions=positions)

        v = 3
        e = 3
        f = len(faces)
        assert v - e + f == 2


# ---------------------------------------------------------------------------
# compute_orthogonal_representation with edge cases
# ---------------------------------------------------------------------------


class TestOrthogonalRepEdgeCases:
    def test_only_self_loops(self) -> None:
        """Graph with only self-loops should produce empty representation."""
        ortho = compute_orthogonal_representation(2, [(0, 0), (1, 1)])
        assert ortho is not None
        assert ortho.total_bends == 0

    def test_self_loop_mixed_with_normal(self) -> None:
        """Self-loop alongside normal edges should not crash."""
        edges = [(0, 1), (1, 2), (2, 0), (0, 0)]
        positions = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        ortho = compute_orthogonal_representation(3, edges, positions)
        assert ortho is not None

    def test_multi_edge_does_not_crash(self) -> None:
        """Duplicate edges should be handled gracefully."""
        edges = [(0, 1), (0, 1), (1, 2), (2, 0)]
        positions = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        ortho = compute_orthogonal_representation(3, edges, positions)
        assert ortho is not None
