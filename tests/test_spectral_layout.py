"""
Tests for spectral layout algorithm.
"""

import math

from graph_layout.spectral import SpectralLayout

# =============================================================================
# Test Fixtures
# =============================================================================


def create_path_graph(n=5):
    """Create a path graph: 0-1-2-...-n-1."""
    nodes = [{} for _ in range(n)]
    links = [{"source": i, "target": i + 1} for i in range(n - 1)]
    return nodes, links


def create_cycle_graph(n=5):
    """Create a cycle graph."""
    nodes = [{} for _ in range(n)]
    links = [{"source": i, "target": (i + 1) % n} for i in range(n)]
    return nodes, links


def create_two_cliques():
    """Create two connected cliques (community structure)."""
    # Clique 1: nodes 0-3
    # Clique 2: nodes 4-7
    # Bridge: 3-4
    nodes = [{} for _ in range(8)]
    links = []

    # Clique 1
    for i in range(4):
        for j in range(i + 1, 4):
            links.append({"source": i, "target": j})

    # Clique 2
    for i in range(4, 8):
        for j in range(i + 1, 8):
            links.append({"source": i, "target": j})

    # Bridge
    links.append({"source": 3, "target": 4})

    return nodes, links


# =============================================================================
# Spectral Layout Tests
# =============================================================================


class TestSpectralLayout:
    """Tests for Spectral layout."""

    def test_basic_layout(self):
        """Test basic layout runs without error."""
        nodes, links = create_path_graph()
        layout = SpectralLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
        )
        layout.run()

        assert len(layout.nodes) == 5
        for node in layout.nodes:
            assert hasattr(node, "x")
            assert hasattr(node, "y")

    def test_path_graph_ordering(self):
        """Test that path graph maintains approximate ordering."""
        nodes, links = create_path_graph(5)
        layout = SpectralLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
        )
        layout.run()

        # Adjacent nodes in path should be relatively close
        for i in range(len(layout.nodes) - 1):
            dist = math.sqrt(
                (layout.nodes[i].x - layout.nodes[i + 1].x) ** 2
                + (layout.nodes[i].y - layout.nodes[i + 1].y) ** 2
            )
            # Distance should not be too large
            assert dist < 500

    def test_configuration_properties(self):
        """Test configuration via constructor and properties."""
        layout = SpectralLayout(
            dimension=2,
            normalized=True,
        )

        assert layout.dimension == 2
        assert layout.normalized is True

        # Test property setters
        layout.dimension = 3
        layout.normalized = False
        assert layout.dimension == 3
        assert layout.normalized is False

    def test_normalized_vs_unnormalized(self):
        """Test that both normalized and unnormalized work."""
        nodes, links = create_path_graph()

        layout1 = SpectralLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
            normalized=True,
        )
        layout1.run()

        nodes2 = [{} for _ in range(5)]
        links2 = [{"source": i, "target": i + 1} for i in range(4)]

        layout2 = SpectralLayout(
            nodes=nodes2,
            links=links2,
            size=(800, 600),
            normalized=False,
        )
        layout2.run()

        # Both should complete
        assert len(layout1.nodes) == 5
        assert len(layout2.nodes) == 5

    def test_community_separation(self):
        """Test that spectral layout separates communities."""
        nodes, links = create_two_cliques()
        layout = SpectralLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
        )
        layout.run()

        # Compute centroid of each clique
        clique1_x = sum(layout.nodes[i].x for i in range(4)) / 4
        clique1_y = sum(layout.nodes[i].y for i in range(4)) / 4
        clique2_x = sum(layout.nodes[i].x for i in range(4, 8)) / 4
        clique2_y = sum(layout.nodes[i].y for i in range(4, 8)) / 4

        # Cliques should be somewhat separated
        separation = math.sqrt((clique1_x - clique2_x) ** 2 + (clique1_y - clique2_y) ** 2)
        # There should be some separation (not identical positions)
        assert separation > 10

    def test_cycle_graph(self):
        """Test layout of a cycle graph."""
        nodes, links = create_cycle_graph(6)
        layout = SpectralLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
        )
        layout.run()

        assert len(layout.nodes) == 6

    def test_single_node(self):
        """Test layout with single node."""
        layout = SpectralLayout(
            nodes=[{}],
            links=[],
            size=(800, 600),
        )
        layout.run()

        assert len(layout.nodes) == 1
        # Single node should be at center
        assert abs(layout.nodes[0].x - 400) < 1.0
        assert abs(layout.nodes[0].y - 300) < 1.0

    def test_two_nodes(self):
        """Test layout with two connected nodes."""
        nodes = [{}, {}]
        links = [{"source": 0, "target": 1}]

        layout = SpectralLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
        )
        layout.run()

        assert len(layout.nodes) == 2

    def test_disconnected_nodes(self):
        """Test layout with disconnected nodes."""
        nodes = [{} for _ in range(4)]
        links = [{"source": 0, "target": 1}]  # Only 0-1 connected

        layout = SpectralLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
        )
        layout.run()

        assert len(layout.nodes) == 4

    def test_empty_graph(self):
        """Test layout with no nodes."""
        layout = SpectralLayout(
            nodes=[],
            links=[],
            size=(800, 600),
        )
        layout.run()
        assert len(layout.nodes) == 0

    def test_weighted_edges(self):
        """Test layout with weighted edges."""
        nodes = [{} for _ in range(4)]
        links = [
            {"source": 0, "target": 1, "weight": 10.0},  # Strong
            {"source": 2, "target": 3, "weight": 10.0},  # Strong
            {"source": 1, "target": 2, "weight": 0.1},  # Weak bridge
        ]

        layout = SpectralLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
        )
        layout.run()

        assert len(layout.nodes) == 4

    def test_pythonic_api(self):
        """Test Pythonic API."""
        nodes, links = create_path_graph()
        layout = SpectralLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
            dimension=2,
            normalized=True,
        )

        # Properties should be accessible
        assert len(layout.nodes) == 5
        assert len(layout.links) == 4
        assert layout.size == (800, 600)
        assert layout.dimension == 2
        assert layout.normalized is True
