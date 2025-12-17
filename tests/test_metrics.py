"""Tests for layout quality metrics."""

import pytest

from graph_layout import Link, Node
from graph_layout.metrics import (
    angular_resolution,
    edge_crossings,
    edge_length_uniformity,
    edge_length_variance,
    layout_quality_summary,
    stress,
)


class TestEdgeCrossings:
    """Tests for edge crossing detection."""

    def test_no_crossings_triangle(self):
        """Triangle has no crossings."""
        nodes = [Node(x=0, y=0), Node(x=100, y=0), Node(x=50, y=87)]
        links = [Link(0, 1), Link(1, 2), Link(2, 0)]

        assert edge_crossings(nodes, links) == 0

    def test_crossing_square_with_diagonals(self):
        """Square with diagonals has exactly 1 crossing."""
        nodes = [
            Node(x=0, y=0),
            Node(x=100, y=0),
            Node(x=100, y=100),
            Node(x=0, y=100),
        ]
        links = [
            Link(0, 1),
            Link(1, 2),
            Link(2, 3),
            Link(3, 0),  # Square edges
            Link(0, 2),
            Link(1, 3),  # Diagonals
        ]

        assert edge_crossings(nodes, links) == 1

    def test_empty_graph(self):
        """Empty graph has no crossings."""
        assert edge_crossings([], []) == 0

    def test_no_links(self):
        """Graph with no links has no crossings."""
        nodes = [Node(x=0, y=0), Node(x=100, y=0)]
        assert edge_crossings(nodes, []) == 0

    def test_single_edge(self):
        """Single edge has no crossings."""
        nodes = [Node(x=0, y=0), Node(x=100, y=0)]
        links = [Link(0, 1)]
        assert edge_crossings(nodes, links) == 0

    def test_parallel_edges_no_crossing(self):
        """Parallel edges don't cross."""
        nodes = [
            Node(x=0, y=0),
            Node(x=100, y=0),
            Node(x=0, y=50),
            Node(x=100, y=50),
        ]
        links = [Link(0, 1), Link(2, 3)]  # Two horizontal parallel lines
        assert edge_crossings(nodes, links) == 0

    def test_adjacent_edges_no_crossing(self):
        """Edges sharing a vertex don't count as crossing."""
        nodes = [Node(x=0, y=0), Node(x=100, y=0), Node(x=50, y=50)]
        links = [Link(0, 1), Link(0, 2)]  # V shape
        assert edge_crossings(nodes, links) == 0


class TestStress:
    """Tests for stress calculation."""

    def test_perfect_layout(self):
        """Layout with exact ideal distances has near-zero stress."""
        # Three nodes in a line with equal spacing
        nodes = [Node(x=0, y=0), Node(x=100, y=0), Node(x=200, y=0)]
        links = [Link(0, 1), Link(1, 2)]

        # Ideal: d(0,1)=100, d(1,2)=100, d(0,2)=200
        s = stress(nodes, links=links, edge_length=100)
        assert s < 0.01  # Near zero

    def test_stressed_layout(self):
        """Layout with wrong distances has positive stress."""
        # Nodes too close together
        nodes = [Node(x=0, y=0), Node(x=50, y=0), Node(x=100, y=0)]
        links = [Link(0, 1), Link(1, 2)]

        # With edge_length=100, ideal d(0,1)=100 but actual=50
        s = stress(nodes, links=links, edge_length=100)
        assert s > 0.1

    def test_single_node(self):
        """Single node has zero stress."""
        nodes = [Node(x=0, y=0)]
        assert stress(nodes, links=[], edge_length=100) == 0.0

    def test_two_nodes_exact_distance(self):
        """Two nodes at exact ideal distance have near-zero stress."""
        nodes = [Node(x=0, y=0), Node(x=100, y=0)]
        links = [Link(0, 1)]
        s = stress(nodes, links=links, edge_length=100)
        assert s < 0.01

    def test_requires_links_or_ideal_distances(self):
        """Stress raises if neither links nor ideal_distances provided."""
        nodes = [Node(x=0, y=0), Node(x=100, y=0)]
        with pytest.raises(ValueError, match="Must provide"):
            stress(nodes)


class TestEdgeLengthVariance:
    """Tests for edge length variance."""

    def test_uniform_lengths_triangle(self):
        """Equilateral triangle has near-zero variance."""
        # Approximately equilateral
        nodes = [Node(x=0, y=0), Node(x=100, y=0), Node(x=50, y=86.6)]
        links = [Link(0, 1), Link(1, 2), Link(2, 0)]

        var = edge_length_variance(nodes, links)
        assert var < 1  # Near zero

    def test_variable_lengths(self):
        """Different length edges have positive variance."""
        nodes = [Node(x=0, y=0), Node(x=100, y=0), Node(x=300, y=0)]
        links = [Link(0, 1), Link(1, 2)]  # Lengths: 100, 200

        var = edge_length_variance(nodes, links)
        assert var > 2000  # Significant variance

    def test_empty_links(self):
        """Empty links return zero variance."""
        nodes = [Node(x=0, y=0)]
        assert edge_length_variance(nodes, []) == 0.0

    def test_single_edge(self):
        """Single edge has zero variance."""
        nodes = [Node(x=0, y=0), Node(x=100, y=0)]
        links = [Link(0, 1)]
        assert edge_length_variance(nodes, links) == 0.0


class TestEdgeLengthUniformity:
    """Tests for edge length uniformity."""

    def test_uniform_returns_high_value(self):
        """Uniform edge lengths return high uniformity."""
        nodes = [Node(x=0, y=0), Node(x=100, y=0), Node(x=50, y=86.6)]
        links = [Link(0, 1), Link(1, 2), Link(2, 0)]

        uniformity = edge_length_uniformity(nodes, links)
        assert uniformity > 0.9

    def test_variable_returns_low_value(self):
        """Highly variable edges return low uniformity."""
        nodes = [Node(x=0, y=0), Node(x=10, y=0), Node(x=500, y=0)]
        links = [Link(0, 1), Link(1, 2)]  # Lengths: 10, 490

        uniformity = edge_length_uniformity(nodes, links)
        assert uniformity < 0.5

    def test_empty_links(self):
        """Empty links return 1.0 uniformity."""
        nodes = [Node(x=0, y=0)]
        assert edge_length_uniformity(nodes, []) == 1.0

    def test_single_edge(self):
        """Single edge returns 1.0 uniformity."""
        nodes = [Node(x=0, y=0), Node(x=100, y=0)]
        links = [Link(0, 1)]
        assert edge_length_uniformity(nodes, links) == 1.0


class TestAngularResolution:
    """Tests for angular resolution."""

    def test_optimal_star_graph(self):
        """Star graph with evenly distributed spokes has good angular resolution."""
        # Center + 4 nodes at 90-degree intervals
        nodes = [
            Node(x=0, y=0),  # Center
            Node(x=100, y=0),  # East
            Node(x=0, y=100),  # North
            Node(x=-100, y=0),  # West
            Node(x=0, y=-100),  # South
        ]
        links = [Link(0, i) for i in range(1, 5)]

        angle = angular_resolution(nodes, links)
        assert abs(angle - 90) < 1  # Should be ~90 degrees

    def test_poor_angular_resolution(self):
        """Bunched edges have small angular resolution."""
        nodes = [
            Node(x=0, y=0),
            Node(x=100, y=0),
            Node(x=100, y=10),
            Node(x=100, y=20),
        ]
        links = [Link(0, 1), Link(0, 2), Link(0, 3)]

        angle = angular_resolution(nodes, links)
        assert angle < 20  # Small angle between bunched edges

    def test_empty_graph(self):
        """Empty graph returns 360."""
        assert angular_resolution([], []) == 360.0

    def test_single_edge(self):
        """Single edge (no multi-edge nodes) returns 360."""
        nodes = [Node(x=0, y=0), Node(x=100, y=0)]
        links = [Link(0, 1)]
        assert angular_resolution(nodes, links) == 360.0

    def test_linear_chain(self):
        """Linear chain (degree 2 nodes) has 180 degree resolution."""
        nodes = [Node(x=0, y=0), Node(x=100, y=0), Node(x=200, y=0)]
        links = [Link(0, 1), Link(1, 2)]

        angle = angular_resolution(nodes, links)
        assert abs(angle - 180) < 1


class TestLayoutQualitySummary:
    """Tests for layout quality summary."""

    def test_returns_all_metrics(self):
        """Summary returns all expected metrics."""
        nodes = [Node(x=0, y=0), Node(x=100, y=0), Node(x=50, y=87)]
        links = [Link(0, 1), Link(1, 2), Link(2, 0)]

        summary = layout_quality_summary(nodes, links)

        assert "edge_crossings" in summary
        assert "stress" in summary
        assert "edge_length_variance" in summary
        assert "edge_length_uniformity" in summary
        assert "angular_resolution" in summary

    def test_triangle_good_metrics(self):
        """Equilateral triangle has good metrics."""
        nodes = [Node(x=0, y=0), Node(x=100, y=0), Node(x=50, y=86.6)]
        links = [Link(0, 1), Link(1, 2), Link(2, 0)]

        summary = layout_quality_summary(nodes, links, edge_length=100)

        assert summary["edge_crossings"] == 0
        assert summary["stress"] < 0.1
        assert summary["edge_length_uniformity"] > 0.9
        assert summary["angular_resolution"] > 50  # Should be ~60 degrees

    def test_empty_graph(self):
        """Empty graph has valid summary."""
        summary = layout_quality_summary([], [])

        assert summary["edge_crossings"] == 0
        assert summary["stress"] == 0.0
        assert summary["edge_length_variance"] == 0.0
        assert summary["edge_length_uniformity"] == 1.0
        assert summary["angular_resolution"] == 360.0


class TestMetricsWithRealLayout:
    """Integration tests using actual layout algorithms."""

    def test_fruchterman_reingold_quality(self):
        """FR layout produces reasonable quality metrics."""
        from graph_layout import FruchtermanReingoldLayout

        # Create a simple graph
        nodes = [{} for _ in range(6)]
        links = [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
            {"source": 2, "target": 3},
            {"source": 3, "target": 4},
            {"source": 4, "target": 5},
            {"source": 5, "target": 0},  # Cycle
        ]

        layout = FruchtermanReingoldLayout(
            nodes=nodes,
            links=links,
            size=(500, 500),
            random_seed=42,
            iterations=100,
        )
        layout.run()

        result_nodes = layout.nodes
        result_links = layout.links

        summary = layout_quality_summary(result_nodes, result_links)

        # Layout should produce no crossings for a cycle
        assert summary["edge_crossings"] == 0
        # Uniformity should be reasonable
        assert summary["edge_length_uniformity"] > 0.3
