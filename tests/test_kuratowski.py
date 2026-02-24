"""Tests for Kuratowski subgraph extraction."""

from __future__ import annotations

import pytest

from graph_layout.planarity import check_planarity


def _k5_edges() -> list[tuple[int, int]]:
    """Complete graph K5 (non-planar)."""
    return [(i, j) for i in range(5) for j in range(i + 1, 5)]


def _k33_edges() -> list[tuple[int, int]]:
    """Complete bipartite graph K3,3 (non-planar)."""
    edges = []
    for i in range(3):
        for j in range(3, 6):
            edges.append((i, j))
    return edges


def _k4_edges() -> list[tuple[int, int]]:
    return [(i, j) for i in range(4) for j in range(i + 1, 4)]


def _cycle_edges(n: int) -> list[tuple[int, int]]:
    return [(i, (i + 1) % n) for i in range(n)]


def _grid_edges(rows: int, cols: int) -> list[tuple[int, int]]:
    edges = []
    for r in range(rows):
        for c in range(cols):
            v = r * cols + c
            if c + 1 < cols:
                edges.append((v, v + 1))
            if r + 1 < rows:
                edges.append((v, v + cols))
    return edges


def _k6_edges() -> list[tuple[int, int]]:
    """Complete graph K6 (non-planar)."""
    return [(i, j) for i in range(6) for j in range(i + 1, 6)]


def _petersen_edges() -> list[tuple[int, int]]:
    """Petersen graph (non-planar, 10 vertices, 15 edges)."""
    outer = [(i, (i + 1) % 5) for i in range(5)]
    inner = [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
    spokes = [(i, i + 5) for i in range(5)]
    return outer + inner + spokes


def _k44_edges() -> list[tuple[int, int]]:
    """Complete bipartite graph K4,4 (non-planar)."""
    edges = []
    for i in range(4):
        for j in range(4, 8):
            edges.append((i, j))
    return edges


def _validate_witness(
    num_nodes: int,
    graph_edges: list[tuple[int, int]],
    witness_edges: list[tuple[int, int]],
    expected_type: str,
) -> None:
    """Validate that witness edges form a valid Kuratowski witness."""
    # All witness edges should be a subset of the input graph edges
    graph_edge_set = set()
    for u, v in graph_edges:
        graph_edge_set.add((min(u, v), max(u, v)))
    for u, v in witness_edges:
        canon = (min(u, v), max(u, v))
        assert canon in graph_edge_set, f"Witness edge {canon} not in graph"

    # Witness edges should form a connected subgraph
    w_adj: dict[int, set[int]] = {}
    for u, v in witness_edges:
        w_adj.setdefault(u, set()).add(v)
        w_adj.setdefault(v, set()).add(u)

    if not w_adj:
        pytest.fail("Witness has no edges")

    # BFS connectivity check
    start = next(iter(w_adj))
    visited = {start}
    queue = [start]
    while queue:
        v = queue.pop()
        for w in w_adj.get(v, set()):
            if w not in visited:
                visited.add(w)
                queue.append(w)
    assert visited == set(w_adj.keys()), "Witness subgraph is not connected"

    # Check minimum edge counts for subdivision
    if expected_type == "K5":
        # K5 subdivision needs at least 10 edges (paths between 5 branch vertices)
        assert len(witness_edges) >= 10, (
            f"K5 witness should have >= 10 edges, got {len(witness_edges)}"
        )
    elif expected_type == "K3,3":
        # K3,3 subdivision needs at least 9 edges (paths between 6 branch vertices)
        assert len(witness_edges) >= 9, (
            f"K3,3 witness should have >= 9 edges, got {len(witness_edges)}"
        )


class TestKuratowskiK5:
    def test_k5_returns_witness(self) -> None:
        result = check_planarity(5, _k5_edges())
        assert not result.is_planar
        assert result.kuratowski_edges is not None
        assert result.kuratowski_type == "K5"
        _validate_witness(5, _k5_edges(), result.kuratowski_edges, "K5")

    def test_k5_witness_has_10_edges(self) -> None:
        result = check_planarity(5, _k5_edges())
        assert result.kuratowski_edges is not None
        assert len(result.kuratowski_edges) == 10


class TestKuratowskiK33:
    def test_k33_returns_witness(self) -> None:
        result = check_planarity(6, _k33_edges())
        assert not result.is_planar
        assert result.kuratowski_edges is not None
        assert result.kuratowski_type == "K3,3"
        _validate_witness(6, _k33_edges(), result.kuratowski_edges, "K3,3")

    def test_k33_witness_has_9_edges(self) -> None:
        result = check_planarity(6, _k33_edges())
        assert result.kuratowski_edges is not None
        assert len(result.kuratowski_edges) == 9


class TestKuratowskiLargerGraphs:
    def test_k6_returns_witness(self) -> None:
        result = check_planarity(6, _k6_edges())
        assert not result.is_planar
        assert result.kuratowski_edges is not None
        assert result.kuratowski_type in ("K5", "K3,3")
        _validate_witness(6, _k6_edges(), result.kuratowski_edges, result.kuratowski_type)

    def test_petersen_returns_witness(self) -> None:
        result = check_planarity(10, _petersen_edges())
        assert not result.is_planar
        assert result.kuratowski_edges is not None
        assert result.kuratowski_type in ("K5", "K3,3")
        _validate_witness(
            10,
            _petersen_edges(),
            result.kuratowski_edges,
            result.kuratowski_type,
        )

    def test_k44_returns_witness(self) -> None:
        result = check_planarity(8, _k44_edges())
        assert not result.is_planar
        assert result.kuratowski_edges is not None
        assert result.kuratowski_type in ("K5", "K3,3")
        _validate_witness(8, _k44_edges(), result.kuratowski_edges, result.kuratowski_type)


class TestKuratowskiPlanarGraphs:
    def test_k4_no_witness(self) -> None:
        result = check_planarity(4, _k4_edges())
        assert result.is_planar
        assert result.kuratowski_edges is None
        assert result.kuratowski_type is None

    def test_cycle_no_witness(self) -> None:
        result = check_planarity(6, _cycle_edges(6))
        assert result.is_planar
        assert result.kuratowski_edges is None

    def test_grid_no_witness(self) -> None:
        result = check_planarity(9, _grid_edges(3, 3))
        assert result.is_planar
        assert result.kuratowski_edges is None


class TestKuratowskiWitnessProperties:
    def test_witness_edges_are_subset(self) -> None:
        """Witness edges must be a subset of the input graph edges."""
        edges = _k5_edges()
        result = check_planarity(5, edges)
        assert result.kuratowski_edges is not None
        edge_set = {(min(u, v), max(u, v)) for u, v in edges}
        for u, v in result.kuratowski_edges:
            assert (min(u, v), max(u, v)) in edge_set

    def test_witness_forms_connected_subgraph(self) -> None:
        """Witness edges should form a connected subgraph."""
        result = check_planarity(6, _k33_edges())
        assert result.kuratowski_edges is not None
        adj: dict[int, set[int]] = {}
        for u, v in result.kuratowski_edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        start = next(iter(adj))
        visited = {start}
        queue = [start]
        while queue:
            node = queue.pop()
            for nb in adj.get(node, set()):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        assert visited == set(adj.keys())
