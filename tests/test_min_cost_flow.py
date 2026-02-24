"""
Tests for the Dijkstra-based min-cost flow solver.
"""

from __future__ import annotations

import time

import pytest

from graph_layout.orthogonal._min_cost_flow import solve_min_cost_flow
from graph_layout.orthogonal.orthogonalization import (
    Face,
    FlowNetwork,
    _solve_bellman_ford,
    build_flow_network,
    compute_faces,
    solve_min_cost_flow_simple,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_network(
    num_vertices: int,
    faces: list[Face],
    supplies: dict[int, int],
    arcs: dict[tuple[int, int], tuple[int, int]],
) -> FlowNetwork:
    """Convenience constructor for FlowNetwork."""
    net = FlowNetwork(num_vertices=num_vertices, faces=faces)
    net.supplies = supplies
    net.arcs = arcs
    return net


def _total_cost(network: FlowNetwork) -> int:
    total = 0
    for arc, flow in network.flow.items():
        if flow > 0 and arc in network.arcs:
            _, c = network.arcs[arc]
            total += flow * c
    return total


# ---------------------------------------------------------------------------
# Unit tests -- small hand-crafted networks
# ---------------------------------------------------------------------------


class TestSmallNetworks:
    """Hand-crafted small network tests."""

    def test_trivial_two_nodes(self):
        """2 nodes, 1 arc: flow = min(supply, capacity)."""
        faces: list[Face] = []
        net = _make_network(
            num_vertices=2,
            faces=faces,
            supplies={0: 3, 1: -3},
            arcs={(0, 1): (5, 1)},
        )
        ok = solve_min_cost_flow(net)
        assert ok
        assert net.flow[(0, 1)] == 3

    def test_trivial_supply_exceeds_capacity(self):
        """Supply exceeds arc capacity -- only capacity worth of flow sent."""
        faces: list[Face] = []
        net = _make_network(
            num_vertices=2,
            faces=faces,
            supplies={0: 10, 1: -10},
            arcs={(0, 1): (3, 1)},
        )
        # infeasible: cannot satisfy all supply/demand
        ok = solve_min_cost_flow(net)
        assert not ok
        assert net.flow[(0, 1)] == 3

    def test_two_paths_picks_cheaper(self):
        """Two parallel paths, solver should prefer the cheaper one."""
        # 0 -> 1 (cap=5, cost=3)
        # 0 -> 2 -> 1 (cap=5+5, cost=1+1=2)
        faces: list[Face] = []
        net = _make_network(
            num_vertices=3,
            faces=faces,
            supplies={0: 4, 1: -4},
            arcs={
                (0, 1): (5, 3),
                (0, 2): (5, 1),
                (2, 1): (5, 1),
            },
        )
        ok = solve_min_cost_flow(net)
        assert ok
        # Cheaper path 0->2->1 (cost 2 per unit) should carry all 4 units
        assert net.flow[(0, 2)] == 4
        assert net.flow[(2, 1)] == 4
        assert net.flow[(0, 1)] == 0
        assert _total_cost(net) == 8

    def test_zero_cost_arcs(self):
        """All arcs cost 0 -- any feasible flow is optimal."""
        faces: list[Face] = []
        net = _make_network(
            num_vertices=3,
            faces=faces,
            supplies={0: 2, 1: -1, 2: -1},
            arcs={
                (0, 1): (2, 0),
                (0, 2): (2, 0),
            },
        )
        ok = solve_min_cost_flow(net)
        assert ok
        assert net.flow[(0, 1)] + net.flow[(0, 2)] == 2
        assert _total_cost(net) == 0

    def test_diamond_network(self):
        """Diamond: S -> A, S -> B, A -> T, B -> T with different costs."""
        # S=0, A=1, B=2, T=3
        faces: list[Face] = []
        net = _make_network(
            num_vertices=4,
            faces=faces,
            supplies={0: 3, 3: -3},
            arcs={
                (0, 1): (2, 1),  # S->A: cap=2, cost=1
                (0, 2): (2, 2),  # S->B: cap=2, cost=2
                (1, 3): (2, 1),  # A->T: cap=2, cost=1
                (2, 3): (2, 2),  # B->T: cap=2, cost=2
            },
        )
        ok = solve_min_cost_flow(net)
        assert ok
        # Cheap path S->A->T (cost 2/unit) takes 2 units,
        # expensive path S->B->T (cost 4/unit) takes 1 unit
        assert net.flow[(0, 1)] == 2
        assert net.flow[(1, 3)] == 2
        assert net.flow[(0, 2)] == 1
        assert net.flow[(2, 3)] == 1
        assert _total_cost(net) == 2 * 2 + 1 * 4  # 8

    def test_infeasible_disconnected(self):
        """Demand node unreachable from supply -- returns False."""
        faces: list[Face] = []
        net = _make_network(
            num_vertices=3,
            faces=faces,
            supplies={0: 2, 2: -2},
            arcs={(0, 1): (5, 0)},  # no path from 0 to 2
        )
        ok = solve_min_cost_flow(net)
        assert not ok


# ---------------------------------------------------------------------------
# Equivalence tests -- new solver matches old Bellman-Ford
# ---------------------------------------------------------------------------


def _solve_old(network: FlowNetwork) -> int:
    """Solve using old Bellman-Ford and return total cost."""
    _solve_bellman_ford(network)
    return _total_cost(network)


def _solve_new(network: FlowNetwork) -> int:
    """Solve using new Dijkstra solver and return total cost."""
    solve_min_cost_flow(network)
    return _total_cost(network)


def _build_graph_network(
    num_nodes: int, edges: list[tuple[int, int]], positions: list[tuple[float, float]]
) -> FlowNetwork:
    """Build a FlowNetwork from a graph specification."""
    faces = compute_faces(num_nodes, edges, positions)
    return build_flow_network(num_nodes, edges, faces)


class TestEquivalence:
    """New solver produces same total cost as old Bellman-Ford solver."""

    def test_triangle(self):
        """Triangle graph: 3 vertices, 3 edges."""
        edges = [(0, 1), (1, 2), (2, 0)]
        positions = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]

        net_old = _build_graph_network(3, edges, positions)
        net_new = _build_graph_network(3, edges, positions)

        cost_old = _solve_old(net_old)
        cost_new = _solve_new(net_new)
        assert cost_new == cost_old

    def test_square_cycle(self):
        """Square cycle: 4 vertices, 4 edges -- zero bends possible."""
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        positions = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

        net_old = _build_graph_network(4, edges, positions)
        net_new = _build_graph_network(4, edges, positions)

        cost_old = _solve_old(net_old)
        cost_new = _solve_new(net_new)
        assert cost_new == cost_old
        # Square cycle can be drawn with zero bends
        assert cost_new == 0

    def test_k4(self):
        """Complete graph K4."""
        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        positions = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]

        net_old = _build_graph_network(4, edges, positions)
        net_new = _build_graph_network(4, edges, positions)

        cost_old = _solve_old(net_old)
        cost_new = _solve_new(net_new)
        assert cost_new == cost_old

    def test_path_graph(self):
        """Path graph P5: 5 vertices in a line."""
        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        positions = [(float(i), 0.0) for i in range(5)]

        net_old = _build_graph_network(5, edges, positions)
        net_new = _build_graph_network(5, edges, positions)

        cost_old = _solve_old(net_old)
        cost_new = _solve_new(net_new)
        assert cost_new == cost_old

    def test_wheel_graph(self):
        """Wheel graph W5: central vertex connected to a cycle of 4."""
        import math

        n_outer = 5
        edges = []
        positions = [(0.0, 0.0)]  # center
        for i in range(n_outer):
            angle = 2 * math.pi * i / n_outer
            positions.append((math.cos(angle), math.sin(angle)))
            edges.append((0, i + 1))  # hub to rim
            edges.append((i + 1, ((i + 1) % n_outer) + 1))  # rim

        num_nodes = n_outer + 1
        net_old = _build_graph_network(num_nodes, edges, positions)
        net_new = _build_graph_network(num_nodes, edges, positions)

        cost_old = _solve_old(net_old)
        cost_new = _solve_new(net_new)
        assert cost_new == cost_old

    def test_grid_2x3(self):
        """2x3 grid graph."""
        # Nodes: (r, c) for r in 0..1, c in 0..2
        # Numbered row-major: node = r*3 + c
        positions = []
        for r in range(2):
            for c in range(3):
                positions.append((float(c), float(r)))

        edges = []
        for r in range(2):
            for c in range(3):
                v = r * 3 + c
                if c + 1 < 3:
                    edges.append((v, v + 1))
                if r + 1 < 2:
                    edges.append((v, v + 3))

        net_old = _build_graph_network(6, edges, positions)
        net_new = _build_graph_network(6, edges, positions)

        cost_old = _solve_old(net_old)
        cost_new = _solve_new(net_new)
        assert cost_new == cost_old


# ---------------------------------------------------------------------------
# Integration test via the public API
# ---------------------------------------------------------------------------


class TestIntegration:
    """Test that solve_min_cost_flow_simple (which now delegates) works."""

    def test_solve_min_cost_flow_simple_delegates(self):
        """solve_min_cost_flow_simple should use the new solver."""
        edges = [(0, 1), (1, 2), (2, 0)]
        positions = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        net = _build_graph_network(3, edges, positions)
        ok = solve_min_cost_flow_simple(net)
        assert ok
        # Verify flow dict is populated
        assert len(net.flow) > 0


# ---------------------------------------------------------------------------
# Performance tests
# ---------------------------------------------------------------------------


def _random_planar_graph(
    n: int, seed: int = 42
) -> tuple[int, list[tuple[int, int]], list[tuple[float, float]]]:
    """Generate a random planar graph by Delaunay-like triangulation of random points."""
    import random

    rng = random.Random(seed)

    # Random positions in a grid to avoid degeneracies
    positions = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)]

    # Simple planar graph: connect each point to nearest neighbors
    # Use a greedy approach that maintains planarity (approximate)
    from math import sqrt

    edges_set: set[tuple[int, int]] = set()

    # Sort by x for sweep-line-like connectivity
    sorted_indices = sorted(range(n), key=lambda i: (positions[i][0], positions[i][1]))

    # Connect consecutive in sorted order (guaranteed planar chain)
    for i in range(len(sorted_indices) - 1):
        u, v = sorted_indices[i], sorted_indices[i + 1]
        edges_set.add((min(u, v), max(u, v)))

    # Add some cross-connections using nearest neighbors (keep sparse)
    for i in range(n):
        dists = []
        for j in range(n):
            if i != j:
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dists.append((sqrt(dx * dx + dy * dy), j))
        dists.sort()
        # Connect to 2 nearest not-yet-connected neighbors
        added = 0
        for _, j in dists:
            if added >= 2:
                break
            e = (min(i, j), max(i, j))
            if e not in edges_set:
                edges_set.add(e)
                added += 1

    edges = list(edges_set)
    return n, edges, positions


class TestPerformance:
    """Performance regression tests."""

    def test_100_node_graph_under_1s(self):
        """100-node graph should solve in under 1 second."""
        n, edges, positions = _random_planar_graph(100)
        faces = compute_faces(n, edges, positions)
        if not faces:
            pytest.skip("Could not compute faces for random graph")

        network = build_flow_network(n, edges, faces)

        start = time.monotonic()
        solve_min_cost_flow(network)
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, f"100-node graph took {elapsed:.2f}s"

    def test_500_node_graph_under_1s(self):
        """500-node graph should solve in under 1 second."""
        n, edges, positions = _random_planar_graph(500)
        faces = compute_faces(n, edges, positions)
        if not faces:
            pytest.skip("Could not compute faces for random graph")

        network = build_flow_network(n, edges, faces)

        start = time.monotonic()
        solve_min_cost_flow(network)
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, f"500-node graph took {elapsed:.2f}s"
