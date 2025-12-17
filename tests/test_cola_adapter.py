"""Tests for ColaLayoutAdapter."""

import pytest

from graph_layout.cola.adapter import ColaLayoutAdapter


class TestColaAdapterBaseInterface:
    """Test BaseLayout interface compatibility."""

    def test_pythonic_api(self):
        """Test Pythonic API with constructor parameters."""
        nodes = [{"x": 0, "y": 0}, {"x": 100, "y": 0}]
        links = [{"source": 0, "target": 1}]

        layout = ColaLayoutAdapter(
            nodes=nodes,
            links=links,
            size=(500, 500),
            iterations=50,
        )

        assert layout is not None
        assert len(layout.nodes) == 2
        assert len(layout.links) == 1

    def test_nodes_property(self):
        """Test nodes property getter and setter."""
        layout = ColaLayoutAdapter()

        # Setter
        layout.nodes = [{"x": 0, "y": 0}, {"x": 100, "y": 0}]

        # Getter
        assert len(layout.nodes) == 2

    def test_links_property(self):
        """Test links property getter and setter."""
        layout = ColaLayoutAdapter(nodes=[{}, {}])

        # Setter
        layout.links = [{"source": 0, "target": 1}]

        # Getter
        assert len(layout.links) == 1

    def test_size_property(self):
        """Test size property getter and setter."""
        layout = ColaLayoutAdapter()

        # Setter
        layout.size = (800, 600)

        # Getter
        assert layout.size == (800, 600)

    def test_size_validation(self):
        """Test that size validation is inherited."""
        from graph_layout.validation import InvalidCanvasSizeError

        with pytest.raises(InvalidCanvasSizeError):
            ColaLayoutAdapter(size=(-100, 600))

    def test_iterations_property(self):
        """Test iterations property getter and setter."""
        layout = ColaLayoutAdapter(iterations=200)
        assert layout.iterations == 200

        layout.iterations = 300
        assert layout.iterations == 300


class TestColaAdapterLayout:
    """Test layout execution."""

    def test_run_executes_layout(self):
        """Test that run() executes the layout."""
        nodes = [{"x": 0, "y": 0}, {"x": 100, "y": 0}, {"x": 50, "y": 100}]
        links = [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
            {"source": 2, "target": 0},
        ]

        layout = ColaLayoutAdapter(
            nodes=nodes,
            links=links,
            size=(500, 500),
        )
        layout.run(all_constraints_iterations=10, keep_running=False)

        assert len(layout.nodes) == 3
        # Nodes should have positions
        for node in layout.nodes:
            assert hasattr(node, "x")
            assert hasattr(node, "y")

    def test_events_propagate(self):
        """Test that events are forwarded from Cola."""
        nodes = [{"x": 0, "y": 0}, {"x": 100, "y": 0}]
        links = [{"source": 0, "target": 1}]
        events = []

        layout = ColaLayoutAdapter(
            nodes=nodes,
            links=links,
            size=(500, 500),
            on_start=lambda e: events.append("start"),
            on_end=lambda e: events.append("end"),
        )

        layout.run(all_constraints_iterations=10, keep_running=False)

        assert "start" in events
        assert "end" in events

    def test_tick_events(self):
        """Test that tick events are fired in keep_running mode."""
        nodes = [{"x": 0, "y": 0}, {"x": 100, "y": 0}]
        links = [{"source": 0, "target": 1}]
        tick_count = [0]

        layout = ColaLayoutAdapter(
            nodes=nodes,
            links=links,
            size=(500, 500),
            convergence_threshold=0.1,
            on_tick=lambda e: tick_count.__setitem__(0, tick_count[0] + 1),
        )

        # Use keep_running=True so tick events are fired
        layout.run(all_constraints_iterations=10, keep_running=True)

        # Should have at least one tick
        assert tick_count[0] > 0


class TestColaAdapterSpecificFeatures:
    """Test Cola-specific features are accessible."""

    def test_avoid_overlaps(self):
        """Test avoid_overlaps property getter/setter."""
        layout = ColaLayoutAdapter()
        assert layout.avoid_overlaps is False

        layout.avoid_overlaps = True
        assert layout.avoid_overlaps is True

    def test_handle_disconnected(self):
        """Test handle_disconnected property getter/setter."""
        layout = ColaLayoutAdapter()
        # Default is True
        assert layout.handle_disconnected is True

        layout.handle_disconnected = False
        assert layout.handle_disconnected is False

    def test_convergence_threshold(self):
        """Test convergence_threshold property getter/setter."""
        layout = ColaLayoutAdapter(convergence_threshold=0.001)
        assert layout.convergence_threshold == 0.001

        layout.convergence_threshold = 0.01
        assert layout.convergence_threshold == 0.01

    def test_link_distance_fixed(self):
        """Test link_distance property with fixed value."""
        layout = ColaLayoutAdapter(link_distance=100)
        assert layout.link_distance == 100

    def test_link_distance_function(self):
        """Test link_distance property with function."""

        def distance_fn(link):
            return 50

        layout = ColaLayoutAdapter(link_distance=distance_fn)
        # Should accept function
        assert callable(layout.link_distance)

    def test_constraints(self):
        """Test constraints property getter/setter."""
        constraint = {"axis": "x", "left": 0, "right": 1, "gap": 50}

        layout = ColaLayoutAdapter(constraints=[constraint])
        assert len(layout.constraints) == 1

    def test_default_node_size(self):
        """Test default_node_size property getter/setter."""
        layout = ColaLayoutAdapter(default_node_size=20)
        assert layout.default_node_size == 20

    def test_group_compactness(self):
        """Test group_compactness property getter/setter."""
        layout = ColaLayoutAdapter(group_compactness=0.001)
        assert layout.group_compactness == 0.001

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

        layout = ColaLayoutAdapter(nodes=nodes, links=links)

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

        layout = ColaLayoutAdapter(nodes=nodes, links=links)

        result = layout.jaccard_link_lengths(100, w=1.0)
        assert result is layout


class TestColaAdapterWithGroups:
    """Test Cola adapter with group functionality."""

    def test_groups_property(self):
        """Test groups property getter and setter."""
        layout = ColaLayoutAdapter(nodes=[{}, {}, {}])

        groups = [{"leaves": [0, 1]}]
        layout.groups = groups

        assert len(layout.groups) == 1

    def test_layout_with_groups(self):
        """Test layout execution with groups."""
        nodes = [{"x": 0, "y": 0}, {"x": 100, "y": 0}, {"x": 200, "y": 0}]
        links = [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
        ]
        groups = [{"leaves": [0, 1], "padding": 10}]

        layout = ColaLayoutAdapter(
            nodes=nodes,
            links=links,
            groups=groups,
            size=(500, 500),
        )
        layout.run(all_constraints_iterations=10, keep_running=False)

        assert len(layout.nodes) == 3


class TestColaAdapterPolymorphism:
    """Test polymorphic usage with other layouts."""

    def test_same_interface_as_force_layouts(self):
        """Test that ColaLayoutAdapter has same interface as force layouts."""
        from graph_layout import FruchtermanReingoldLayout

        # Both should have these properties/methods
        for LayoutClass in [ColaLayoutAdapter, FruchtermanReingoldLayout]:
            layout = LayoutClass()
            assert hasattr(layout, "nodes")
            assert hasattr(layout, "links")
            assert hasattr(layout, "size")
            assert hasattr(layout, "run")
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
            if layout_class == ColaLayoutAdapter:
                layout = layout_class(
                    nodes=nodes,
                    links=links,
                    size=(500, 500),
                )
                layout.run(all_constraints_iterations=10, keep_running=False)
            else:
                layout = layout_class(
                    nodes=nodes,
                    links=links,
                    size=(500, 500),
                    iterations=10,
                )
                layout.run()
            return layout.nodes

        # Both should produce valid results
        cola_result = run_layout(ColaLayoutAdapter)
        fr_result = run_layout(FruchtermanReingoldLayout)

        assert len(cola_result) == 3
        assert len(fr_result) == 3


class TestColaAdapterAlphaMapping:
    """Test alpha and alpha_min mapping."""

    def test_alpha_min_property(self):
        """Test alpha_min property."""
        layout = ColaLayoutAdapter(alpha_min=0.005)
        assert layout.alpha_min == 0.005

    def test_alpha_property(self):
        """Test alpha property returns sensible value."""
        layout = ColaLayoutAdapter()

        # Before run, alpha should be initial value
        alpha = layout.alpha
        assert alpha == 1.0
