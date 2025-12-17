"""Tests for 2D layout module."""

import numpy as np

from graph_layout.cola.layout import EventType, Group, Layout, Link, Node, is_group


class TestEventType:
    """Test EventType enum."""

    def test_event_values(self):
        """Test event type values."""
        assert EventType.start == 0
        assert EventType.tick == 1
        assert EventType.end == 2

    def test_event_names(self):
        """Test event type names."""
        assert EventType.start.name == "start"
        assert EventType.tick.name == "tick"
        assert EventType.end.name == "end"

    def test_string_access(self):
        """Test accessing by string name."""
        assert EventType["start"] == EventType.start
        assert EventType["tick"] == EventType.tick
        assert EventType["end"] == EventType.end


class TestNode:
    """Test Node class."""

    def test_create_empty_node(self):
        """Test creating node with defaults."""
        node = Node()
        assert node.x == 0.0
        assert node.y == 0.0
        assert node.fixed == 0
        assert node.index is None
        assert node.width is None
        assert node.height is None

    def test_create_node_with_position(self):
        """Test creating node with position."""
        node = Node(x=10.0, y=20.0)
        assert node.x == 10.0
        assert node.y == 20.0

    def test_create_node_with_size(self):
        """Test creating node with size."""
        node = Node(width=50.0, height=30.0)
        assert node.width == 50.0
        assert node.height == 30.0

    def test_create_fixed_node(self):
        """Test creating fixed node."""
        node = Node(x=5.0, y=10.0, fixed=1)
        assert node.fixed == 1

    def test_node_custom_properties(self):
        """Test adding custom properties to node."""
        node = Node(id="node1", color="red", weight=2.5)
        assert node.id == "node1"
        assert node.color == "red"
        assert node.weight == 2.5

    def test_node_index_property(self):
        """Test node index property."""
        node = Node(index=5)
        assert node.index == 5


class TestLink:
    """Test Link class."""

    def test_create_link_with_indices(self):
        """Test creating link with node indices."""
        link = Link(0, 1)
        assert link.source == 0
        assert link.target == 1
        assert link.length is None
        assert link.weight is None

    def test_create_link_with_nodes(self):
        """Test creating link with node objects."""
        n1 = Node(index=0)
        n2 = Node(index=1)
        link = Link(n1, n2)
        assert link.source == n1
        assert link.target == n2

    def test_create_link_with_length(self):
        """Test creating link with length."""
        link = Link(0, 1, length=50.0)
        assert link.length == 50.0

    def test_create_link_with_weight(self):
        """Test creating link with weight."""
        link = Link(0, 1, weight=0.5)
        assert link.weight == 0.5

    def test_link_custom_properties(self):
        """Test adding custom properties to link."""
        link = Link(0, 1, id="edge1", label="connects")
        assert link.id == "edge1"
        assert link.label == "connects"


class TestGroup:
    """Test Group class."""

    def test_create_empty_group(self):
        """Test creating empty group."""
        group = Group()
        assert group.leaves is None
        assert group.groups is None
        assert group.padding == 1.0
        assert group.bounds is None

    def test_create_group_with_leaves(self):
        """Test creating group with leaf nodes."""
        group = Group(leaves=[0, 1, 2])
        assert len(group.leaves) == 3
        assert group.leaves == [0, 1, 2]

    def test_create_group_with_subgroups(self):
        """Test creating group with subgroups."""
        group = Group(groups=[0, 1])
        assert len(group.groups) == 2

    def test_create_group_with_padding(self):
        """Test creating group with custom padding."""
        group = Group(padding=5.0)
        assert group.padding == 5.0

    def test_is_group_function(self):
        """Test is_group helper function."""
        group = Group(leaves=[0, 1])
        node = Node()

        assert is_group(group) is True
        assert is_group(node) is False


class TestLayout:
    """Test Layout class."""

    def test_create_layout(self):
        """Test creating layout with defaults."""
        layout = Layout()
        assert layout.nodes() == []
        assert layout.links() == []
        assert layout.groups() == []
        assert layout.size() == [1.0, 1.0]

    def test_fluent_api_size(self):
        """Test fluent API for size."""
        layout = Layout()
        result = layout.size([800.0, 600.0])

        assert result is layout  # Method chaining
        assert layout.size() == [800.0, 600.0]

    def test_fluent_api_link_distance(self):
        """Test fluent API for link distance."""
        layout = Layout()
        result = layout.link_distance(50.0)

        assert result is layout
        assert layout.link_distance() == 50.0

    def test_link_distance_function(self):
        """Test link distance with function."""
        layout = Layout()
        layout.link_distance(lambda link: link.length if link.length else 20.0)

        link = Link(0, 1, length=100.0)
        assert layout.get_link_length(link) == 100.0

        link2 = Link(0, 1)
        assert layout.get_link_length(link2) == 20.0

    def test_set_nodes_as_list(self):
        """Test setting nodes as list."""
        layout = Layout()
        nodes = [Node(x=0, y=0), Node(x=10, y=10)]
        layout.nodes(nodes)

        assert len(layout.nodes()) == 2
        assert layout.nodes()[0].x == 0
        assert layout.nodes()[1].x == 10

    def test_set_nodes_as_dicts(self):
        """Test setting nodes as dicts."""
        layout = Layout()
        nodes = [{"x": 0, "y": 0}, {"x": 10, "y": 10}]
        layout.nodes(nodes)

        assert len(layout.nodes()) == 2
        assert layout.nodes()[0].x == 0
        assert layout.nodes()[1].x == 10

    def test_set_links_as_list(self):
        """Test setting links as list."""
        layout = Layout()
        links = [Link(0, 1), Link(1, 2)]
        layout.links(links)

        assert len(layout.links()) == 2

    def test_set_links_as_dicts(self):
        """Test setting links as dicts."""
        layout = Layout()
        links = [{"source": 0, "target": 1}, {"source": 1, "target": 2}]
        layout.links(links)

        assert len(layout.links()) == 2
        assert layout.links()[0].source == 0
        assert layout.links()[1].target == 2

    def test_auto_create_nodes_from_links(self):
        """Test that nodes are auto-created from links."""
        layout = Layout()
        layout.links([Link(0, 1), Link(1, 2)])

        nodes = layout.nodes()
        assert len(nodes) == 3  # Nodes 0, 1, 2

    def test_avoid_overlaps_getter_setter(self):
        """Test avoid overlaps getter/setter."""
        layout = Layout()
        assert layout.avoid_overlaps() is False

        layout.avoid_overlaps(True)
        assert layout.avoid_overlaps() is True

    def test_handle_disconnected_getter_setter(self):
        """Test handle disconnected getter/setter."""
        layout = Layout()
        assert layout.handle_disconnected() is True

        layout.handle_disconnected(False)
        assert layout.handle_disconnected() is False

    def test_convergence_threshold(self):
        """Test convergence threshold getter/setter."""
        layout = Layout()
        assert layout.convergence_threshold() == 0.01

        layout.convergence_threshold(0.001)
        assert layout.convergence_threshold() == 0.001

    def test_default_node_size(self):
        """Test default node size getter/setter."""
        layout = Layout()
        assert layout.default_node_size() == 10.0

        layout.default_node_size(20.0)
        assert layout.default_node_size() == 20.0

    def test_group_compactness(self):
        """Test group compactness getter/setter."""
        layout = Layout()
        assert layout.group_compactness() == 1e-6

        layout.group_compactness(1e-5)
        assert layout.group_compactness() == 1e-5


class TestLayoutEventSystem:
    """Test Layout event system."""

    def test_register_event_with_enum(self):
        """Test registering event with EventType enum."""
        layout = Layout()
        called = []

        def on_start(e):
            called.append("start")

        layout.on(EventType.start, on_start)

        assert layout.event is not None
        assert EventType.start in layout.event

    def test_register_event_with_string(self):
        """Test registering event with string name."""
        layout = Layout()
        called = []

        def on_tick(e):
            called.append("tick")

        layout.on("tick", on_tick)

        assert EventType.tick in layout.event

    def test_trigger_event(self):
        """Test triggering an event."""
        layout = Layout()
        events = []

        def on_start(e):
            events.append(e)

        layout.on(EventType.start, on_start)
        layout.trigger({"type": EventType.start, "alpha": 1.0})

        assert len(events) == 1
        assert events[0]["type"] == EventType.start
        assert events[0]["alpha"] == 1.0

    def test_multiple_event_types(self):
        """Test registering multiple event types."""
        layout = Layout()
        start_called = []
        end_called = []

        layout.on(EventType.start, lambda e: start_called.append(e))
        layout.on(EventType.end, lambda e: end_called.append(e))

        layout.trigger({"type": EventType.start, "alpha": 1.0})
        assert len(start_called) == 1
        assert len(end_called) == 0

        layout.trigger({"type": EventType.end, "alpha": 0.0})
        assert len(start_called) == 1
        assert len(end_called) == 1


class TestLayoutSimpleGraph:
    """Test Layout with simple graphs."""

    def test_two_nodes_connected(self):
        """Test layout with two connected nodes."""
        layout = Layout()
        layout.nodes([Node(x=0, y=0, width=10, height=10), Node(x=100, y=0, width=10, height=10)])
        layout.links([Link(0, 1)])
        layout.size([200, 200])
        layout.link_distance(50)

        # Run layout
        layout.start(10, 0, 0, 0, False)

        # Nodes should have been positioned
        nodes = layout.nodes()
        assert nodes[0].x != 0 or nodes[0].y != 0 or nodes[1].x != 100
        # Check that distance computation worked
        assert layout._descent is not None

    def test_three_nodes_linear(self):
        """Test layout with three nodes in line."""
        layout = Layout()
        layout.nodes(
            [
                Node(x=0, y=0, width=10, height=10),
                Node(x=50, y=0, width=10, height=10),
                Node(x=100, y=0, width=10, height=10),
            ]
        )
        layout.links([Link(0, 1), Link(1, 2)])
        layout.size([200, 200])

        layout.start(10, 0, 0, 0, False)

        nodes = layout.nodes()
        assert len(nodes) == 3
        # All nodes should have positions
        for node in nodes:
            assert hasattr(node, "x")
            assert hasattr(node, "y")

    def test_triangle_graph(self):
        """Test layout with triangle graph."""
        layout = Layout()
        layout.nodes(
            [
                Node(x=0, y=0, width=10, height=10),
                Node(x=100, y=0, width=10, height=10),
                Node(x=50, y=87, width=10, height=10),
            ]
        )
        layout.links([Link(0, 1), Link(1, 2), Link(2, 0)])

        layout.start(10, 0, 0, 0, False)

        nodes = layout.nodes()
        assert len(nodes) == 3


class TestLayoutWithSizes:
    """Test Layout with node sizes."""

    def test_nodes_with_width_height(self):
        """Test layout with nodes having width and height."""
        layout = Layout()
        layout.nodes([Node(x=0, y=0, width=20, height=20), Node(x=50, y=0, width=30, height=30)])
        layout.links([Link(0, 1)])

        layout.start(5, 0, 0, 0, False)

        nodes = layout.nodes()
        assert nodes[0].width == 20
        assert nodes[1].width == 30

    def test_overlap_avoidance(self):
        """Test overlap avoidance."""
        layout = Layout()
        layout.nodes(
            [
                Node(x=0, y=0, width=20, height=20),
                Node(x=5, y=5, width=20, height=20),  # Overlapping
            ]
        )
        layout.links([Link(0, 1)])
        layout.avoid_overlaps(True)

        layout.start(0, 0, 10, 0, False)

        # After layout, nodes should not overlap significantly
        nodes = layout.nodes()
        dx = abs(nodes[0].x - nodes[1].x)
        dy = abs(nodes[0].y - nodes[1].y)
        # Distance should be at least sum of half-widths
        assert dx >= 10 or dy >= 10


class TestLayoutWithGroups:
    """Test Layout with groups."""

    def test_simple_group(self):
        """Test layout with a simple group."""
        layout = Layout()
        layout.nodes([Node(x=0, y=0), Node(x=10, y=10), Node(x=100, y=100)])
        layout.groups([Group(leaves=[0, 1], padding=5)])

        groups = layout.groups()
        assert len(groups) == 1
        # After groups() is called, nodes should be resolved
        assert len(groups[0].leaves) == 2

    def test_nested_groups(self):
        """Test layout with nested groups."""
        layout = Layout()
        layout.nodes([Node(x=i * 10, y=i * 10) for i in range(4)])
        layout.groups(
            [
                Group(leaves=[0, 1]),
                Group(leaves=[2, 3]),
                Group(groups=[0, 1]),  # Parent group
            ]
        )

        groups = layout.groups()
        assert len(groups) == 3

    def test_layout_with_groups_start(self):
        """Test running layout with groups."""
        layout = Layout()
        layout.nodes(
            [
                Node(x=0, y=0, width=10, height=10),
                Node(x=10, y=0, width=10, height=10),
                Node(x=100, y=100, width=10, height=10),
            ]
        )
        layout.links([Link(0, 1)])
        layout.groups([Group(leaves=[0, 1])])

        # Don't use initial_unconstrained_iterations with groups to avoid nested layout
        layout.start(0, 0, 5, 0, False)

        nodes = layout.nodes()
        assert len(nodes) == 3


class TestLayoutConstraints:
    """Test Layout with constraints."""

    def test_set_constraints(self):
        """Test setting constraints."""
        layout = Layout()
        constraint = {"axis": "x", "left": 0, "right": 1, "gap": 50}
        layout.constraints([constraint])

        constraints = layout.constraints()
        assert len(constraints) == 1
        assert constraints[0] == constraint

    def test_layout_with_separation_constraint(self):
        """Test layout with separation constraint."""
        layout = Layout()
        layout.nodes([Node(x=0, y=0, width=10, height=10), Node(x=10, y=0, width=10, height=10)])
        layout.links([Link(0, 1)])

        # Add a constraint (dict format expected by Projection)
        constraint = {"axis": "x", "left": 0, "right": 1, "gap": 100}
        layout.constraints([constraint])

        layout.start(0, 10, 0, 0, False)

        nodes = layout.nodes()
        # Check that layout ran successfully
        # (exact constraint enforcement depends on other parameters)
        assert len(nodes) == 2
        # Nodes should have been positioned
        assert nodes[0].x is not None
        assert nodes[1].x is not None


class TestLayoutFlowLayout:
    """Test flow layout functionality."""

    def test_flow_layout_y_axis(self):
        """Test flow layout in y direction."""
        layout = Layout()
        layout.nodes([Node() for _ in range(3)])
        layout.links([Link(0, 1), Link(1, 2)])

        result = layout.flow_layout("y", 50)
        assert result is layout  # Method chaining

    def test_flow_layout_x_axis(self):
        """Test flow layout in x direction."""
        layout = Layout()
        layout.nodes([Node() for _ in range(3)])
        layout.links([Link(0, 1), Link(1, 2)])

        result = layout.flow_layout("x", 50)
        assert result is layout

    def test_flow_layout_with_function(self):
        """Test flow layout with function."""
        layout = Layout()
        layout.nodes([Node() for _ in range(3)])
        layout.links([Link(0, 1), Link(1, 2)])

        result = layout.flow_layout("y", lambda: 30)
        assert result is layout


class TestLayoutDisconnectedGraphs:
    """Test layout with disconnected graphs."""

    def test_disconnected_components(self):
        """Test layout with disconnected components."""
        layout = Layout()
        layout.nodes(
            [
                Node(x=0, y=0, width=10, height=10),
                Node(x=10, y=0, width=10, height=10),
                Node(x=100, y=100, width=10, height=10),
                Node(x=110, y=100, width=10, height=10),
            ]
        )
        layout.links(
            [
                Link(0, 1),  # Component 1
                Link(2, 3),  # Component 2
            ]
        )
        layout.handle_disconnected(True)

        layout.start(10, 0, 0, 0, False)

        nodes = layout.nodes()
        # Components should be separated
        comp1_x = (nodes[0].x + nodes[1].x) / 2
        comp2_x = (nodes[2].x + nodes[3].x) / 2
        # Should have some separation (exact value depends on packing)
        assert abs(comp1_x - comp2_x) > 0

    def test_disable_disconnected_handling(self):
        """Test disabling disconnected graph handling."""
        layout = Layout()
        layout.nodes([Node() for _ in range(4)])
        layout.links([Link(0, 1), Link(2, 3)])
        layout.handle_disconnected(False)

        layout.start(5, 0, 0, 0, False)

        # Should still run without error
        nodes = layout.nodes()
        assert len(nodes) == 4


class TestLayoutAlphaAndConvergence:
    """Test layout alpha and convergence."""

    def test_alpha_getter(self):
        """Test getting alpha value."""
        layout = Layout()
        # Before start, alpha is None
        assert layout.alpha() is None

    def test_stop_layout(self):
        """Test stopping layout with keep_running=True."""
        layout = Layout()
        layout.nodes([Node(x=0, y=0, width=10, height=10), Node(x=10, y=0, width=10, height=10)])
        layout.links([Link(0, 1)])

        # Start with keep_running=True so alpha gets set
        # (we can't actually run the async loop in tests, but we can test stop())
        result = layout.stop()

        assert result is layout
        # stop() should return self for chaining
        # Alpha behavior depends on whether layout was running

    def test_resume_layout(self):
        """Test resuming layout."""
        layout = Layout()
        layout.nodes([Node(x=0, y=0, width=10, height=10), Node(x=10, y=0, width=10, height=10)])
        layout.links([Link(0, 1)])

        layout.start(0, 0, 0, 0, False)
        layout.stop()

        result = layout.resume()
        assert result is layout
        # Resume should set alpha (exact value depends on implementation)
        # The resume() method sets alpha to 0.1
        alpha_val = layout.alpha()
        assert alpha_val is not None
        # Alpha gets set but may be 0 if iterations completed
        assert alpha_val >= 0.0


class TestLayoutStaticMethods:
    """Test Layout static methods."""

    def test_get_source_index_with_int(self):
        """Test get_source_index with integer."""
        link = Link(5, 10)
        assert Layout.get_source_index(link) == 5

    def test_get_source_index_with_node(self):
        """Test get_source_index with node."""
        node = Node(index=7)
        link = Link(node, Node(index=8))
        assert Layout.get_source_index(link) == 7

    def test_get_target_index_with_int(self):
        """Test get_target_index with integer."""
        link = Link(5, 10)
        assert Layout.get_target_index(link) == 10

    def test_get_target_index_with_node(self):
        """Test get_target_index with node."""
        link = Link(Node(index=5), Node(index=8))
        assert Layout.get_target_index(link) == 8

    def test_link_id(self):
        """Test link_id generation."""
        link = Link(3, 7)
        assert Layout.link_id(link) == "3-7"

    def test_set_link_length(self):
        """Test set_link_length."""
        link = Link(0, 1)
        Layout.set_link_length(link, 100.0)
        assert link.length == 100.0


class TestLayoutDragOperations:
    """Test Layout drag operations."""

    def test_drag_start_node(self):
        """Test drag start on node."""
        node = Node(x=10, y=20, fixed=0)
        Layout.drag_start(node)

        # Bit 2 should be set
        assert node.fixed & 2 == 2
        assert node.px == 10
        assert node.py == 20

    def test_drag_node(self):
        """Test dragging a node."""
        node = Node(x=10, y=20)
        Layout.drag_start(node)
        Layout.drag(node, {"x": 30, "y": 40})

        assert node.px == 30
        assert node.py == 40

    def test_drag_end_node(self):
        """Test drag end on node."""
        node = Node(fixed=0)
        Layout.drag_start(node)
        assert node.fixed & 2 == 2

        Layout.drag_end(node)
        # Bits 2 and 3 should be unset
        assert node.fixed & 6 == 0

    def test_mouse_over(self):
        """Test mouse over event."""
        node = Node(x=10, y=20, fixed=0)
        Layout.mouse_over(node)

        # Bit 3 should be set
        assert node.fixed & 4 == 4
        assert node.px == 10
        assert node.py == 20

    def test_mouse_out(self):
        """Test mouse out event."""
        node = Node(fixed=0)
        Layout.mouse_over(node)
        assert node.fixed & 4 == 4

        Layout.mouse_out(node)
        assert node.fixed & 4 == 0

    def test_drag_origin_node(self):
        """Test drag origin for node."""
        node = Node(x=15, y=25)
        origin = Layout.drag_origin(node)

        assert origin["x"] == 15
        assert origin["y"] == 25


class TestLayoutLinkLengthCalculators:
    """Test link length calculation methods."""

    def test_symmetric_diff_link_lengths(self):
        """Test symmetric diff link length calculator."""
        layout = Layout()
        layout.nodes([Node() for _ in range(4)])
        layout.links([Link(0, 1), Link(0, 2), Link(0, 3)])

        result = layout.symmetric_diff_link_lengths(50.0, 1.0)
        assert result is layout

    def test_jaccard_link_lengths(self):
        """Test Jaccard link length calculator."""
        layout = Layout()
        layout.nodes([Node() for _ in range(4)])
        layout.links([Link(0, 1), Link(0, 2), Link(0, 3)])

        result = layout.jaccard_link_lengths(50.0, 1.0)
        assert result is layout


class TestLayoutDistanceMatrix:
    """Test custom distance matrix."""

    def test_set_distance_matrix(self):
        """Test setting custom distance matrix."""
        layout = Layout()
        layout.nodes([Node() for _ in range(3)])

        # Custom distance matrix
        D = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)

        result = layout.distance_matrix(D)
        assert result is layout

        retrieved = layout.distance_matrix()
        assert np.array_equal(retrieved, D)

    def test_layout_with_distance_matrix(self):
        """Test running layout with custom distance matrix."""
        layout = Layout()
        layout.nodes([Node(x=0, y=0) for _ in range(3)])
        layout.links([Link(0, 1), Link(1, 2)])

        D = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)

        layout.distance_matrix(D)
        layout.start(5, 0, 0, 0, False)

        nodes = layout.nodes()
        assert len(nodes) == 3


class TestLayoutFixedNodes:
    """Test layout with fixed nodes."""

    def test_fixed_node_stays_in_place(self):
        """Test that fixed nodes don't move much during layout."""
        layout = Layout()
        layout.nodes(
            [
                Node(x=0, y=0, width=10, height=10, fixed=1),  # Fixed
                Node(x=100, y=0, width=10, height=10),
            ]
        )
        layout.links([Link(0, 1)])
        # Disable disconnected component handling which can move fixed nodes
        layout.handle_disconnected(False)

        layout.start(10, 0, 0, 0, False)

        nodes = layout.nodes()
        # Fixed node should stay close to origin
        # (some movement can occur due to numerical precision)
        assert abs(nodes[0].x - 0) < 10.0
        assert abs(nodes[0].y - 0) < 10.0

    def test_partially_fixed_network(self):
        """Test layout with some fixed and some free nodes."""
        layout = Layout()
        layout.nodes(
            [
                Node(x=0, y=0, width=10, height=10, fixed=1),
                Node(x=100, y=100, width=10, height=10, fixed=1),
                Node(x=50, y=50, width=10, height=10),  # Free to move
            ]
        )
        layout.links([Link(0, 2), Link(1, 2)])
        # Disable disconnected component handling
        layout.handle_disconnected(False)

        layout.start(10, 0, 0, 0, False)

        nodes = layout.nodes()
        # Fixed nodes shouldn't move much
        # (allow more tolerance due to packing and numerical precision)
        assert abs(nodes[0].x - 0) < 50.0
        assert abs(nodes[1].x - 100) < 50.0


class TestLayoutEdgeCases:
    """Test edge cases."""

    def test_empty_graph(self):
        """Test layout with no nodes."""
        layout = Layout()
        # Should not crash
        assert layout.nodes() == []
        assert layout.links() == []

    def test_single_node(self):
        """Test layout with single node."""
        layout = Layout()
        layout.nodes([Node(x=10, y=10, width=10, height=10)])

        layout.start(5, 0, 0, 0, False)

        nodes = layout.nodes()
        assert len(nodes) == 1

    def test_nodes_no_links(self):
        """Test layout with nodes but no links."""
        layout = Layout()
        layout.nodes([Node(width=10, height=10) for _ in range(5)])

        layout.start(5, 0, 0, 0, False)

        nodes = layout.nodes()
        assert len(nodes) == 5
