"""
Tests for the SMACOF stress-majorization layout.

The defining property of SMACOF is that each majorization step (Guttman
transform) cannot increase the stress; the tests below check that monotone
decrease, convergence, determinism, and that the layout reproduces distances
for graphs whose graph metric embeds isometrically in the plane.
"""

import math

import pytest

from graph_layout import SMACOFLayout
from graph_layout.force import SMACOFLayout as SMACOFFromForce


def _run(nodes, links, **kwargs):
    layout = SMACOFLayout(nodes=nodes, links=links, size=(800, 600), random_seed=42, **kwargs)
    layout.run()
    return layout


def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def test_exported_from_top_level_and_force():
    assert SMACOFLayout is SMACOFFromForce


def test_empty_graph():
    layout = _run([], [])
    assert layout.nodes == []


def test_single_node():
    layout = _run([{}], [])
    assert len(layout.nodes) == 1


def test_path_graph_positions_finite():
    nodes = [{} for _ in range(5)]
    links = [{"source": i, "target": i + 1} for i in range(4)]
    layout = _run(nodes, links)
    for node in layout.nodes:
        assert math.isfinite(node.x)
        assert math.isfinite(node.y)


def test_stress_decreases_monotonically():
    """Each Guttman transform must not increase the stress (majorization)."""
    nodes = [{} for _ in range(8)]
    # A cycle: metric is well defined, not trivially isometric.
    links = [{"source": i, "target": (i + 1) % 8} for i in range(8)]
    layout = SMACOFLayout(nodes=nodes, links=links, size=(800, 600), random_seed=1)
    stresses = []
    layout.on("tick", lambda e: stresses.append(e["stress"]))
    layout.run()

    assert len(stresses) >= 2
    for prev, cur in zip(stresses, stresses[1:]):
        # Allow a tiny numerical slack.
        assert cur <= prev + 1e-6


def test_triangle_recovers_edge_lengths():
    """An equilateral triangle embeds isometrically; SMACOF should recover it."""
    nodes = [{} for _ in range(3)]
    links = [
        {"source": 0, "target": 1},
        {"source": 1, "target": 2},
        {"source": 2, "target": 0},
    ]
    layout = _run(nodes, links, edge_length=100.0, epsilon=1e-8, iterations=500)
    n = layout.nodes
    for a, b in [(0, 1), (1, 2), (2, 0)]:
        assert _dist(n[a], n[b]) == pytest.approx(100.0, rel=0.05)


def test_path_preserves_distance_ordering():
    """On a path, endpoints should end up farther apart than adjacent nodes."""
    nodes = [{} for _ in range(6)]
    links = [{"source": i, "target": i + 1} for i in range(5)]
    layout = _run(nodes, links, edge_length=100.0, epsilon=1e-8, iterations=500)
    n = layout.nodes
    adjacent = _dist(n[0], n[1])
    endpoints = _dist(n[0], n[5])
    assert endpoints > adjacent * 2


def test_deterministic_with_seed():
    nodes = [{} for _ in range(7)]
    links = [{"source": i, "target": (i + 1) % 7} for i in range(7)]
    a = SMACOFLayout(nodes=nodes, links=links, size=(800, 600), random_seed=99).run()
    b = SMACOFLayout(
        nodes=[{} for _ in range(7)],
        links=links,
        size=(800, 600),
        random_seed=99,
    ).run()
    for na, nb in zip(a.nodes, b.nodes):
        assert na.x == pytest.approx(nb.x)
        assert na.y == pytest.approx(nb.y)


def test_disconnected_components_do_not_blow_up():
    nodes = [{} for _ in range(4)]
    links = [
        {"source": 0, "target": 1},
        {"source": 2, "target": 3},
    ]
    layout = _run(nodes, links)
    for node in layout.nodes:
        assert math.isfinite(node.x)
        assert math.isfinite(node.y)


def test_fixed_node_stays_put():
    nodes = [{"x": 123.0, "y": 456.0, "fixed": 1}, {}, {}]
    links = [{"source": 0, "target": 1}, {"source": 1, "target": 2}]
    layout = SMACOFLayout(nodes=nodes, links=links, size=(800, 600), random_seed=3).run(
        center_graph=False
    )
    assert layout.nodes[0].x == pytest.approx(123.0)
    assert layout.nodes[0].y == pytest.approx(456.0)


def test_converges_before_iteration_budget():
    """A graph that embeds cleanly should converge well within the budget."""
    nodes = [{} for _ in range(4)]
    links = [
        {"source": 0, "target": 1},
        {"source": 1, "target": 2},
        {"source": 2, "target": 3},
        {"source": 3, "target": 0},
    ]
    layout = SMACOFLayout(nodes=nodes, links=links, size=(800, 600), random_seed=7, iterations=1000)
    ticks = []
    layout.on("tick", lambda e: ticks.append(e))
    layout.run()
    assert len(ticks) < 1000
