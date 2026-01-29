"""
Tests for RandomLayout.
"""

from graph_layout import EventType, RandomLayout


class TestRandomLayoutBasic:
    """Basic functionality tests."""

    def test_layout_runs_without_error(self):
        """Layout should run without raising exceptions."""
        nodes = [{} for _ in range(5)]
        links = [{"source": 0, "target": 1}, {"source": 1, "target": 2}]

        layout = RandomLayout(nodes=nodes, links=links, size=(800, 600))
        result = layout.run()

        assert result is layout  # Returns self

    def test_nodes_have_positions_after_run(self):
        """All nodes should have positions within canvas bounds."""
        nodes = [{} for _ in range(10)]
        layout = RandomLayout(nodes=nodes, size=(800, 600))
        layout.run()

        for node in layout.nodes:
            assert 0 <= node.x <= 800
            assert 0 <= node.y <= 600

    def test_nodes_move_from_origin(self):
        """Nodes should not all remain at origin after layout."""
        nodes = [{"x": 0, "y": 0} for _ in range(5)]
        layout = RandomLayout(nodes=nodes, size=(800, 600))
        layout.run()

        # At least some nodes should have moved
        moved = sum(1 for n in layout.nodes if n.x != 0 or n.y != 0)
        assert moved > 0

    def test_empty_graph(self):
        """Empty graph should not raise errors."""
        layout = RandomLayout(nodes=[], links=[], size=(800, 600))
        layout.run()
        assert len(layout.nodes) == 0

    def test_single_node(self):
        """Single node should be positioned within bounds."""
        nodes = [{}]
        layout = RandomLayout(nodes=nodes, size=(800, 600))
        layout.run()

        assert 0 <= layout.nodes[0].x <= 800
        assert 0 <= layout.nodes[0].y <= 600


class TestRandomLayoutConfiguration:
    """Configuration property tests."""

    def test_size_property(self):
        """Canvas size should be settable."""
        layout = RandomLayout(size=(1000, 800))
        assert layout.size == (1000, 800)

        layout.size = (500, 400)
        assert layout.size == (500, 400)

    def test_margin_property(self):
        """Margin should be settable and non-negative."""
        layout = RandomLayout(margin=50)
        assert layout.margin == 50

        layout.margin = 100
        assert layout.margin == 100

        # Negative margin should be clamped to 0
        layout.margin = -10
        assert layout.margin == 0

    def test_margin_affects_placement(self):
        """Nodes should be placed within margin bounds."""
        nodes = [{} for _ in range(50)]
        layout = RandomLayout(nodes=nodes, size=(800, 600), margin=100, random_seed=42)
        layout.run()

        for node in layout.nodes:
            assert 100 <= node.x <= 700
            assert 100 <= node.y <= 500

    def test_margin_too_large_for_canvas(self):
        """When margin is too large, should fall back to full canvas."""
        nodes = [{} for _ in range(5)]
        layout = RandomLayout(nodes=nodes, size=(100, 100), margin=60, random_seed=42)
        layout.run()

        # Should still place nodes (fallback to full canvas)
        for node in layout.nodes:
            assert 0 <= node.x <= 100
            assert 0 <= node.y <= 100

    def test_random_seed_property(self):
        """Random seed should be settable."""
        layout = RandomLayout(random_seed=42)
        assert layout.random_seed == 42

        layout.random_seed = 123
        assert layout.random_seed == 123


class TestRandomLayoutReproducibility:
    """Reproducibility tests with random seeds."""

    def test_same_seed_produces_same_layout(self):
        """Same random seed should produce identical layouts."""
        nodes1 = [{} for _ in range(10)]
        nodes2 = [{} for _ in range(10)]

        layout1 = RandomLayout(nodes=nodes1, size=(800, 600), random_seed=42)
        layout1.run()

        layout2 = RandomLayout(nodes=nodes2, size=(800, 600), random_seed=42)
        layout2.run()

        for n1, n2 in zip(layout1.nodes, layout2.nodes):
            assert n1.x == n2.x
            assert n1.y == n2.y

    def test_different_seeds_produce_different_layouts(self):
        """Different seeds should produce different layouts."""
        nodes1 = [{} for _ in range(10)]
        nodes2 = [{} for _ in range(10)]

        layout1 = RandomLayout(nodes=nodes1, size=(800, 600), random_seed=42)
        layout1.run()

        layout2 = RandomLayout(nodes=nodes2, size=(800, 600), random_seed=123)
        layout2.run()

        # Positions should differ for at least one node
        different = any(
            n1.x != n2.x or n1.y != n2.y for n1, n2 in zip(layout1.nodes, layout2.nodes)
        )
        assert different


class TestRandomLayoutFixedNodes:
    """Fixed node behavior tests."""

    def test_fixed_nodes_preserve_position(self):
        """Nodes marked as fixed should not move."""
        nodes = [
            {"x": 100, "y": 200, "fixed": True},
            {"x": 0, "y": 0},  # Not fixed
            {"x": 300, "y": 400, "fixed": True},
        ]

        layout = RandomLayout(nodes=nodes, size=(800, 600), random_seed=42)
        layout.run()

        # Fixed nodes should stay in place
        assert layout.nodes[0].x == 100
        assert layout.nodes[0].y == 200
        assert layout.nodes[2].x == 300
        assert layout.nodes[2].y == 400

    def test_unfixed_nodes_can_be_outside_canvas(self):
        """Fixed nodes can be anywhere, even outside canvas."""
        nodes = [
            {"x": -100, "y": -100, "fixed": True},
            {"x": 1000, "y": 1000, "fixed": True},
        ]

        layout = RandomLayout(nodes=nodes, size=(800, 600))
        layout.run()

        # Fixed nodes should stay at their positions
        assert layout.nodes[0].x == -100
        assert layout.nodes[0].y == -100
        assert layout.nodes[1].x == 1000
        assert layout.nodes[1].y == 1000


class TestRandomLayoutEvents:
    """Event system tests."""

    def test_start_event_fires(self):
        """Start event should fire when layout begins."""
        events = []

        def on_start(event):
            events.append(("start", event))

        layout = RandomLayout(nodes=[{}], on_start=on_start)
        layout.run()

        assert len(events) == 1
        assert events[0][0] == "start"
        assert events[0][1]["type"] == EventType.start

    def test_end_event_fires(self):
        """End event should fire when layout completes."""
        events = []

        def on_end(event):
            events.append(("end", event))

        layout = RandomLayout(nodes=[{}], on_end=on_end)
        layout.run()

        assert len(events) == 1
        assert events[0][0] == "end"
        assert events[0][1]["type"] == EventType.end

    def test_event_registration_via_on_method(self):
        """Events can be registered using on() method."""
        events = []

        layout = RandomLayout(nodes=[{}])
        layout.on(EventType.start, lambda e: events.append("start"))
        layout.on(EventType.end, lambda e: events.append("end"))
        layout.run()

        assert "start" in events
        assert "end" in events


class TestRandomLayoutChaining:
    """Method chaining tests."""

    def test_run_returns_self(self):
        """run() should return self for chaining."""
        layout = RandomLayout(nodes=[{}])
        result = layout.run()
        assert result is layout

    def test_property_setters_work_after_construction(self):
        """Properties should be settable after construction."""
        layout = RandomLayout()
        layout.nodes = [{}, {}, {}]
        layout.size = (1000, 800)
        layout.margin = 50
        layout.random_seed = 42
        layout.run()

        assert len(layout.nodes) == 3
        assert layout.size == (1000, 800)
        assert layout.margin == 50

        for node in layout.nodes:
            assert 50 <= node.x <= 950
            assert 50 <= node.y <= 750


class TestRandomLayoutImport:
    """Import tests."""

    def test_import_from_package(self):
        """RandomLayout should be importable from package root."""
        from graph_layout import RandomLayout as RandomLayoutPkg

        assert RandomLayoutPkg is RandomLayout

    def test_import_from_basic_module(self):
        """RandomLayout should be importable from basic module."""
        from graph_layout.basic import RandomLayout as RandomLayoutMod

        assert RandomLayoutMod is RandomLayout
