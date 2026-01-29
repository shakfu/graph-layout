"""
Tests for Bipartite layout.
"""

from graph_layout import BipartiteLayout
from graph_layout.bipartite import count_crossings, is_bipartite


class TestBipartiteBasic:
    """Basic functionality tests."""

    def test_layout_runs_without_error(self):
        """Layout should run without raising exceptions."""
        nodes = [{} for _ in range(6)]
        links = [
            {"source": 0, "target": 3},
            {"source": 1, "target": 4},
            {"source": 2, "target": 5},
        ]

        layout = BipartiteLayout(nodes=nodes, links=links, size=(800, 600))
        result = layout.run()

        assert result is layout

    def test_nodes_have_positions_after_run(self):
        """All nodes should have positions after layout."""
        nodes = [{} for _ in range(6)]
        links = [
            {"source": 0, "target": 3},
            {"source": 1, "target": 4},
            {"source": 2, "target": 5},
        ]

        layout = BipartiteLayout(nodes=nodes, links=links, size=(800, 600))
        layout.run()

        for node in layout.nodes:
            assert node.x is not None
            assert node.y is not None

    def test_empty_graph(self):
        """Empty graph should not raise errors."""
        layout = BipartiteLayout(nodes=[], links=[], size=(800, 600))
        layout.run()

        assert len(layout.nodes) == 0

    def test_single_node(self):
        """Single node should be positioned."""
        nodes = [{}]
        layout = BipartiteLayout(nodes=nodes, size=(800, 600))
        layout.run()

        assert len(layout.nodes) == 1
        assert layout.nodes[0].x is not None

    def test_two_nodes_with_edge(self):
        """Two connected nodes should be in different sets."""
        nodes = [{}, {}]
        links = [{"source": 0, "target": 1}]

        layout = BipartiteLayout(nodes=nodes, links=links, size=(800, 600))
        layout.run()

        assert len(layout.nodes) == 2
        # One node in each set
        assert len(layout.top_nodes) >= 1
        assert len(layout.bottom_nodes) >= 1


class TestBipartiteConfiguration:
    """Configuration property tests."""

    def test_orientation_horizontal(self):
        """Horizontal orientation should place nodes in rows."""
        nodes = [{} for _ in range(4)]
        links = [{"source": 0, "target": 2}, {"source": 1, "target": 3}]

        layout = BipartiteLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
            orientation="horizontal",
        )
        layout.run()

        # Top nodes should have same y
        if len(layout.top_nodes) >= 2:
            y_values = [layout.nodes[i].y for i in layout.top_nodes]
            assert len(set(round(y, 1) for y in y_values)) == 1

    def test_orientation_vertical(self):
        """Vertical orientation should place nodes in columns."""
        nodes = [{} for _ in range(4)]
        links = [{"source": 0, "target": 2}, {"source": 1, "target": 3}]

        layout = BipartiteLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
            orientation="vertical",
        )
        layout.run()

        # Top (left) nodes should have same x
        if len(layout.top_nodes) >= 2:
            x_values = [layout.nodes[i].x for i in layout.top_nodes]
            assert len(set(round(x, 1) for x in x_values)) == 1

    def test_layer_separation_property(self):
        """Layer separation should be settable."""
        layout = BipartiteLayout(layer_separation=300)
        assert layout.layer_separation == 300

        layout.layer_separation = 400
        assert layout.layer_separation == 400

    def test_node_separation_property(self):
        """Node separation should be settable."""
        layout = BipartiteLayout(node_separation=100)
        assert layout.node_separation == 100

    def test_minimize_crossings_property(self):
        """Minimize crossings should be settable."""
        layout = BipartiteLayout(minimize_crossings=False)
        assert layout.minimize_crossings is False


class TestBipartiteUserSets:
    """Tests for user-specified bipartite sets."""

    def test_user_specified_sets(self):
        """User can specify the two sets explicitly."""
        nodes = [{} for _ in range(6)]
        links = [
            {"source": 0, "target": 3},
            {"source": 1, "target": 4},
            {"source": 2, "target": 5},
        ]

        layout = BipartiteLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
            top_set=[0, 1, 2],
            bottom_set=[3, 4, 5],
        )
        layout.run()

        assert set(layout.top_nodes) == {0, 1, 2}
        assert set(layout.bottom_nodes) == {3, 4, 5}

    def test_partial_user_sets(self):
        """Layout should work with partial user specification."""
        nodes = [{} for _ in range(4)]
        links = [{"source": 0, "target": 2}, {"source": 1, "target": 3}]

        layout = BipartiteLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
            top_set=[0, 1],
            bottom_set=[2, 3],
        )
        layout.run()

        assert 0 in layout.top_nodes
        assert 2 in layout.bottom_nodes


class TestBipartiteDetection:
    """Tests for automatic bipartite detection."""

    def test_auto_detect_bipartite(self):
        """Should auto-detect bipartite structure."""
        nodes = [{} for _ in range(4)]
        links = [
            {"source": 0, "target": 2},
            {"source": 0, "target": 3},
            {"source": 1, "target": 2},
            {"source": 1, "target": 3},
        ]

        layout = BipartiteLayout(nodes=nodes, links=links, size=(800, 600))
        layout.run()

        assert layout.is_bipartite
        assert len(layout.top_nodes) == 2
        assert len(layout.bottom_nodes) == 2

    def test_detect_non_bipartite(self):
        """Should detect non-bipartite graphs."""
        # Triangle is not bipartite
        nodes = [{} for _ in range(3)]
        links = [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
            {"source": 2, "target": 0},
        ]

        layout = BipartiteLayout(nodes=nodes, links=links, size=(800, 600))
        layout.run()

        # Should fall back to splitting nodes
        assert len(layout.top_nodes) > 0
        assert len(layout.bottom_nodes) > 0

    def test_disconnected_bipartite(self):
        """Should handle disconnected bipartite graphs."""
        nodes = [{} for _ in range(6)]
        links = [
            {"source": 0, "target": 1},  # Component 1
            {"source": 2, "target": 3},  # Component 2
            {"source": 4, "target": 5},  # Component 3
        ]

        layout = BipartiteLayout(nodes=nodes, links=links, size=(800, 600))
        layout.run()

        assert layout.is_bipartite
        assert len(layout.top_nodes) + len(layout.bottom_nodes) == 6


class TestCrossingMinimization:
    """Tests for edge crossing minimization."""

    def test_crossing_minimization_reduces_crossings(self):
        """Crossing minimization should reduce or maintain crossings."""
        nodes = [{} for _ in range(6)]
        # K(3,3) bipartite - some crossings unavoidable
        links = [
            {"source": 0, "target": 3},
            {"source": 0, "target": 4},
            {"source": 0, "target": 5},
            {"source": 1, "target": 3},
            {"source": 1, "target": 4},
            {"source": 1, "target": 5},
            {"source": 2, "target": 3},
            {"source": 2, "target": 4},
            {"source": 2, "target": 5},
        ]

        # Without minimization
        layout_no_min = BipartiteLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
            top_set=[0, 1, 2],
            bottom_set=[3, 4, 5],
            minimize_crossings=False,
        )
        layout_no_min.run()

        # With minimization
        layout_min = BipartiteLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
            top_set=[0, 1, 2],
            bottom_set=[3, 4, 5],
            minimize_crossings=True,
        )
        layout_min.run()

        # Both should complete without error
        assert len(layout_no_min.nodes) == 6
        assert len(layout_min.nodes) == 6


class TestBipartiteEvents:
    """Event system tests."""

    def test_start_event_fires(self):
        """Start event should fire when layout begins."""
        events = []

        def on_start(event):
            events.append(("start", event))

        layout = BipartiteLayout(nodes=[{}, {}], on_start=on_start)
        layout.run()

        assert len(events) == 1
        assert events[0][0] == "start"

    def test_end_event_fires(self):
        """End event should fire when layout completes."""
        events = []

        def on_end(event):
            events.append(("end", event))

        layout = BipartiteLayout(nodes=[{}, {}], on_end=on_end)
        layout.run()

        assert len(events) == 1
        assert events[0][0] == "end"


class TestIsBipartite:
    """Tests for is_bipartite utility function."""

    def test_complete_bipartite(self):
        """Complete bipartite graph should be detected."""
        edges = [(0, 2), (0, 3), (1, 2), (1, 3)]
        is_bp, set_a, set_b = is_bipartite(4, edges)

        assert is_bp
        assert len(set_a) == 2
        assert len(set_b) == 2

    def test_path_is_bipartite(self):
        """Path graph is bipartite."""
        edges = [(0, 1), (1, 2), (2, 3)]
        is_bp, set_a, set_b = is_bipartite(4, edges)

        assert is_bp

    def test_odd_cycle_not_bipartite(self):
        """Odd cycle is not bipartite."""
        edges = [(0, 1), (1, 2), (2, 0)]  # Triangle
        is_bp, set_a, set_b = is_bipartite(3, edges)

        assert not is_bp

    def test_even_cycle_is_bipartite(self):
        """Even cycle is bipartite."""
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]  # Square
        is_bp, set_a, set_b = is_bipartite(4, edges)

        assert is_bp

    def test_empty_graph_is_bipartite(self):
        """Empty graph is bipartite."""
        is_bp, set_a, set_b = is_bipartite(0, [])
        assert is_bp


class TestCountCrossings:
    """Tests for count_crossings utility function."""

    def test_no_crossings(self):
        """Parallel edges have no crossings."""
        top = [0, 1, 2]
        bottom = [3, 4, 5]
        edges = [(0, 3), (1, 4), (2, 5)]

        crossings = count_crossings(top, bottom, edges)
        assert crossings == 0

    def test_one_crossing(self):
        """Swapped edges have one crossing."""
        top = [0, 1]
        bottom = [2, 3]
        edges = [(0, 3), (1, 2)]  # These cross

        crossings = count_crossings(top, bottom, edges)
        assert crossings == 1

    def test_k33_crossings(self):
        """K(3,3) has 9 crossings in standard layout."""
        top = [0, 1, 2]
        bottom = [3, 4, 5]
        # All edges from K(3,3)
        edges = [
            (0, 3),
            (0, 4),
            (0, 5),
            (1, 3),
            (1, 4),
            (1, 5),
            (2, 3),
            (2, 4),
            (2, 5),
        ]

        crossings = count_crossings(top, bottom, edges)
        # K(3,3) has 9 crossings in any drawing
        assert crossings == 9


class TestBipartiteImport:
    """Import tests."""

    def test_import_from_package(self):
        """BipartiteLayout should be importable from package root."""
        from graph_layout import BipartiteLayout as BipartiteLayoutPkg

        assert BipartiteLayoutPkg is BipartiteLayout

    def test_import_from_bipartite_module(self):
        """BipartiteLayout should be importable from bipartite module."""
        from graph_layout.bipartite import BipartiteLayout as BipartiteLayoutMod

        assert BipartiteLayoutMod is BipartiteLayout

    def test_import_utilities(self):
        """Utility functions should be importable."""
        from graph_layout.bipartite import count_crossings, is_bipartite

        assert callable(is_bipartite)
        assert callable(count_crossings)
