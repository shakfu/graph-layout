"""Tests for ColaLayoutAdapter."""

import pytest

from graph_layout.cola.adapter import ColaLayoutAdapter


class TestColaAdapterBaseInterface:
    """Test BaseLayout interface compatibility."""

    def test_fluent_api(self):
        """Test fluent API chaining."""
        nodes = [{"x": 0, "y": 0}, {"x": 100, "y": 0}]
        links = [{"source": 0, "target": 1}]

        layout = (
            ColaLayoutAdapter()
            .nodes(nodes)
            .links(links)
            .size([500, 500])
            .iterations(50)
        )

        assert layout is not None
        assert len(layout.nodes()) == 2
        assert len(layout.links()) == 1

    def test_nodes_getter_setter(self):
        """Test nodes() as both getter and setter."""
        layout = ColaLayoutAdapter()

        # Setter
        result = layout.nodes([{"x": 0, "y": 0}, {"x": 100, "y": 0}])
        assert result is layout  # Returns self

        # Getter
        nodes = layout.nodes()
        assert len(nodes) == 2

    def test_links_getter_setter(self):
        """Test links() as both getter and setter."""
        layout = ColaLayoutAdapter()
        layout.nodes([{}, {}])

        # Setter
        result = layout.links([{"source": 0, "target": 1}])
        assert result is layout

        # Getter
        links = layout.links()
        assert len(links) == 1

    def test_size_getter_setter(self):
        """Test size() as both getter and setter."""
        layout = ColaLayoutAdapter()

        # Setter
        result = layout.size([800, 600])
        assert result is layout

        # Getter
        size = layout.size()
        assert size == [800, 600]

    def test_size_validation(self):
        """Test that size validation is inherited."""
        from graph_layout.validation import InvalidCanvasSizeError

        layout = ColaLayoutAdapter()
        with pytest.raises(InvalidCanvasSizeError):
            layout.size([-100, 600])

    def test_iterations_getter_setter(self):
        """Test iterations() as both getter and setter."""
        layout = ColaLayoutAdapter()

        layout.iterations(200)
        assert layout.iterations() == 200


class TestColaAdapterLayout:
    """Test layout execution."""

    def test_start_runs_layout(self):
        """Test that start() runs the layout."""
        nodes = [{"x": 0, "y": 0}, {"x": 100, "y": 0}, {"x": 50, "y": 100}]
        links = [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
            {"source": 2, "target": 0},
        ]

        layout = ColaLayoutAdapter()
        layout.nodes(nodes).links(links).size([500, 500])
        layout.start(iterations=10, keep_running=False)

        result = layout.nodes()
        assert len(result) == 3
        # Nodes should have positions
        for node in result:
            assert hasattr(node, "x")
            assert hasattr(node, "y")

    def test_events_propagate(self):
        """Test that events are forwarded from Cola."""
        nodes = [{"x": 0, "y": 0}, {"x": 100, "y": 0}]
        links = [{"source": 0, "target": 1}]
        events = []

        layout = ColaLayoutAdapter()
        layout.nodes(nodes).links(links).size([500, 500])
        layout.on("start", lambda e: events.append("start"))
        layout.on("end", lambda e: events.append("end"))

        layout.start(iterations=10, keep_running=False)

        assert "start" in events
        assert "end" in events

    def test_tick_events(self):
        """Test that tick events are fired in keep_running mode."""
        nodes = [{"x": 0, "y": 0}, {"x": 100, "y": 0}]
        links = [{"source": 0, "target": 1}]
        tick_count = [0]

        layout = ColaLayoutAdapter()
        layout.nodes(nodes).links(links).size([500, 500])
        layout.on("tick", lambda e: tick_count.__setitem__(0, tick_count[0] + 1))

        # Use keep_running=True so tick events are fired
        # Set a low convergence threshold so it stops quickly
        layout.convergence_threshold(0.1)
        layout.start(iterations=10, keep_running=True)

        # Should have at least one tick
        assert tick_count[0] > 0


class TestColaAdapterSpecificFeatures:
    """Test Cola-specific features are accessible."""

    def test_avoid_overlaps(self):
        """Test avoid_overlaps() getter/setter."""
        layout = ColaLayoutAdapter()
        assert layout.avoid_overlaps() is False

        layout.avoid_overlaps(True)
        assert layout.avoid_overlaps() is True

    def test_handle_disconnected(self):
        """Test handle_disconnected() getter/setter."""
        layout = ColaLayoutAdapter()
        # Default is True
        assert layout.handle_disconnected() is True

        layout.handle_disconnected(False)
        assert layout.handle_disconnected() is False

    def test_convergence_threshold(self):
        """Test convergence_threshold() getter/setter."""
        layout = ColaLayoutAdapter()

        layout.convergence_threshold(0.001)
        assert layout.convergence_threshold() == 0.001

    def test_link_distance_fixed(self):
        """Test link_distance() with fixed value."""
        layout = ColaLayoutAdapter()

        layout.link_distance(100)
        assert layout.link_distance() == 100

    def test_link_distance_function(self):
        """Test link_distance() with function."""
        layout = ColaLayoutAdapter()

        def distance_fn(link):
            return 50

        layout.link_distance(distance_fn)
        # Should accept function
        result = layout.link_distance()
        assert callable(result)

    def test_constraints(self):
        """Test constraints() getter/setter."""
        layout = ColaLayoutAdapter()
        constraint = {"axis": "x", "left": 0, "right": 1, "gap": 50}

        layout.constraints([constraint])
        assert len(layout.constraints()) == 1

    def test_default_node_size(self):
        """Test default_node_size() getter/setter."""
        layout = ColaLayoutAdapter()

        layout.default_node_size(20)
        assert layout.default_node_size() == 20

    def test_group_compactness(self):
        """Test group_compactness() getter/setter."""
        layout = ColaLayoutAdapter()

        layout.group_compactness(0.001)
        assert layout.group_compactness() == 0.001

    def test_access_underlying_cola(self):
        """Test that underlying Cola is accessible."""
        layout = ColaLayoutAdapter()
        cola = layout.cola

        assert cola is not None
        # Should have Cola-specific methods
        assert hasattr(cola, "power_graph_groups")
        assert hasattr(cola, "prepare_edge_routing")

    def test_flow_layout(self):
        """Test flow_layout() configuration."""
        layout = ColaLayoutAdapter()

        # Should return self for chaining
        result = layout.flow_layout(axis="y", min_separation=50)
        assert result is layout

    def test_symmetric_diff_link_lengths(self):
        """Test symmetric_diff_link_lengths()."""
        nodes = [{"x": 0, "y": 0}, {"x": 100, "y": 0}, {"x": 200, "y": 0}]
        links = [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
        ]

        layout = ColaLayoutAdapter()
        layout.nodes(nodes).links(links)

        # Should return self for chaining
        result = layout.symmetric_diff_link_lengths(100, w=1.0)
        assert result is layout

    def test_jaccard_link_lengths(self):
        """Test jaccard_link_lengths()."""
        nodes = [{"x": 0, "y": 0}, {"x": 100, "y": 0}, {"x": 200, "y": 0}]
        links = [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
        ]

        layout = ColaLayoutAdapter()
        layout.nodes(nodes).links(links)

        result = layout.jaccard_link_lengths(100, w=1.0)
        assert result is layout


class TestColaAdapterWithGroups:
    """Test Cola adapter with group functionality."""

    def test_groups_getter_setter(self):
        """Test groups() as both getter and setter."""
        layout = ColaLayoutAdapter()
        layout.nodes([{}, {}, {}])

        groups = [{"leaves": [0, 1]}]
        result = layout.groups(groups)
        assert result is layout

        retrieved = layout.groups()
        assert len(retrieved) == 1

    def test_layout_with_groups(self):
        """Test layout execution with groups."""
        nodes = [{"x": 0, "y": 0}, {"x": 100, "y": 0}, {"x": 200, "y": 0}]
        links = [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
        ]
        groups = [{"leaves": [0, 1], "padding": 10}]

        layout = ColaLayoutAdapter()
        layout.nodes(nodes).links(links).groups(groups).size([500, 500])
        layout.start(iterations=10, keep_running=False)

        result = layout.nodes()
        assert len(result) == 3


class TestColaAdapterPolymorphism:
    """Test polymorphic usage with other layouts."""

    def test_same_interface_as_force_layouts(self):
        """Test that ColaLayoutAdapter has same interface as force layouts."""
        from graph_layout import FruchtermanReingoldLayout

        # Both should have these methods
        for LayoutClass in [ColaLayoutAdapter, FruchtermanReingoldLayout]:
            layout = LayoutClass()
            assert hasattr(layout, "nodes")
            assert hasattr(layout, "links")
            assert hasattr(layout, "size")
            assert hasattr(layout, "start")
            assert hasattr(layout, "stop")
            assert hasattr(layout, "on")
            assert hasattr(layout, "iterations")
            assert hasattr(layout, "alpha")

    def test_interchangeable_usage(self):
        """Test that layouts can be used interchangeably."""
        from graph_layout import FruchtermanReingoldLayout

        nodes = [{"x": 0, "y": 0}, {"x": 100, "y": 0}, {"x": 50, "y": 100}]
        links = [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
            {"source": 2, "target": 0},
        ]

        def run_layout(layout_class):
            layout = layout_class()
            layout.nodes(nodes).links(links).size([500, 500])
            if isinstance(layout, ColaLayoutAdapter):
                layout.start(iterations=10, keep_running=False)
            else:
                layout.start(iterations=10)
            return layout.nodes()

        # Both should produce valid results
        cola_result = run_layout(ColaLayoutAdapter)
        fr_result = run_layout(FruchtermanReingoldLayout)

        assert len(cola_result) == 3
        assert len(fr_result) == 3


class TestColaAdapterAlphaMapping:
    """Test alpha and alpha_min mapping."""

    def test_alpha_min_maps_to_convergence_threshold(self):
        """Test that alpha_min maps to convergence_threshold."""
        layout = ColaLayoutAdapter()

        layout.alpha_min(0.005)
        assert layout.alpha_min() == 0.005
        assert layout.convergence_threshold() == 0.005

    def test_alpha_getter(self):
        """Test alpha getter returns sensible value."""
        layout = ColaLayoutAdapter()

        # Before start, alpha should be 0
        alpha = layout.alpha()
        assert alpha == 0.0
