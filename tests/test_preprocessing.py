"""Tests for graph preprocessing utilities."""

from graph_layout import (
    assign_layers_longest_path,
    connected_components,
    count_crossings,
    detect_cycle,
    has_cycle,
    is_connected,
    minimize_crossings_barycenter,
    remove_cycles,
    topological_sort,
)


class TestCycleDetection:
    """Tests for cycle detection functions."""

    def test_detect_cycle_in_cyclic_graph(self):
        """Should detect cycle in a graph with a cycle."""
        links = [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
            {"source": 2, "target": 0},  # Back edge creating cycle
        ]
        cycle = detect_cycle(3, links)
        assert cycle is not None
        assert len(cycle) >= 2

    def test_detect_cycle_in_acyclic_graph(self):
        """Should return None for acyclic graph."""
        links = [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
            {"source": 0, "target": 2},
        ]
        cycle = detect_cycle(3, links)
        assert cycle is None

    def test_has_cycle_true(self):
        """has_cycle should return True for cyclic graph."""
        links = [
            {"source": 0, "target": 1},
            {"source": 1, "target": 0},
        ]
        assert has_cycle(2, links) is True

    def test_has_cycle_false(self):
        """has_cycle should return False for acyclic graph."""
        links = [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
        ]
        assert has_cycle(3, links) is False

    def test_detect_cycle_empty_graph(self):
        """Empty graph should have no cycles."""
        assert detect_cycle(0, []) is None
        assert detect_cycle(5, []) is None

    def test_detect_cycle_self_loop(self):
        """Self-loop should be detected as cycle."""
        links = [{"source": 0, "target": 0}]
        cycle = detect_cycle(1, links)
        assert cycle is not None


class TestCycleRemoval:
    """Tests for cycle removal function."""

    def test_remove_cycles_makes_acyclic(self):
        """Removing cycles should produce an acyclic graph."""
        links = [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
            {"source": 2, "target": 0},  # Creates cycle
        ]
        new_links, reversed_idx = remove_cycles(3, links)
        assert has_cycle(3, new_links) is False
        assert len(reversed_idx) >= 1

    def test_remove_cycles_preserves_acyclic(self):
        """Acyclic graph should remain unchanged."""
        links = [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
        ]
        new_links, reversed_idx = remove_cycles(3, links)
        assert len(reversed_idx) == 0
        assert has_cycle(3, new_links) is False

    def test_remove_cycles_bidirectional(self):
        """Should handle bidirectional edges."""
        links = [
            {"source": 0, "target": 1},
            {"source": 1, "target": 0},
        ]
        new_links, reversed_idx = remove_cycles(2, links)
        assert has_cycle(2, new_links) is False


class TestTopologicalSort:
    """Tests for topological sort."""

    def test_topological_sort_simple(self):
        """Simple DAG should have valid topological order."""
        links = [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
        ]
        order = topological_sort(3, links)
        assert order is not None
        assert order.index(0) < order.index(1) < order.index(2)

    def test_topological_sort_diamond(self):
        """Diamond DAG should have valid topological order."""
        links = [
            {"source": 0, "target": 1},
            {"source": 0, "target": 2},
            {"source": 1, "target": 3},
            {"source": 2, "target": 3},
        ]
        order = topological_sort(4, links)
        assert order is not None
        assert order.index(0) < order.index(1)
        assert order.index(0) < order.index(2)
        assert order.index(1) < order.index(3)
        assert order.index(2) < order.index(3)

    def test_topological_sort_cyclic_returns_none(self):
        """Cyclic graph should return None."""
        links = [
            {"source": 0, "target": 1},
            {"source": 1, "target": 0},
        ]
        order = topological_sort(2, links)
        assert order is None

    def test_topological_sort_empty(self):
        """Empty graph should return empty list."""
        order = topological_sort(0, [])
        assert order == []

    def test_topological_sort_disconnected(self):
        """Disconnected graph should include all nodes."""
        links = [{"source": 0, "target": 1}]
        order = topological_sort(3, links)
        assert order is not None
        assert set(order) == {0, 1, 2}


class TestConnectedComponents:
    """Tests for connected component detection."""

    def test_connected_components_single(self):
        """Connected graph should have one component."""
        links = [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
        ]
        components = connected_components(3, links)
        assert len(components) == 1
        assert set(components[0]) == {0, 1, 2}

    def test_connected_components_multiple(self):
        """Disconnected graph should have multiple components."""
        links = [
            {"source": 0, "target": 1},
            {"source": 2, "target": 3},
        ]
        components = connected_components(4, links)
        assert len(components) == 2

    def test_connected_components_isolated(self):
        """Isolated nodes should each be their own component."""
        components = connected_components(3, [])
        assert len(components) == 3

    def test_is_connected_true(self):
        """Connected graph should return True."""
        links = [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
        ]
        assert is_connected(3, links) is True

    def test_is_connected_false(self):
        """Disconnected graph should return False."""
        links = [{"source": 0, "target": 1}]
        assert is_connected(3, links) is False

    def test_is_connected_empty(self):
        """Empty or single-node graph should be connected."""
        assert is_connected(0, []) is True
        assert is_connected(1, []) is True


class TestLayerAssignment:
    """Tests for layer assignment."""

    def test_assign_layers_simple_chain(self):
        """Simple chain should have sequential layers."""
        links = [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
        ]
        layers = assign_layers_longest_path(3, links)
        assert len(layers) == 3
        assert layers[0] == [0]
        assert layers[1] == [1]
        assert layers[2] == [2]

    def test_assign_layers_fork(self):
        """Fork should put children in same layer."""
        links = [
            {"source": 0, "target": 1},
            {"source": 0, "target": 2},
        ]
        layers = assign_layers_longest_path(3, links)
        assert len(layers) == 2
        assert layers[0] == [0]
        assert set(layers[1]) == {1, 2}

    def test_assign_layers_diamond(self):
        """Diamond DAG should have correct layer assignment."""
        links = [
            {"source": 0, "target": 1},
            {"source": 0, "target": 2},
            {"source": 1, "target": 3},
            {"source": 2, "target": 3},
        ]
        layers = assign_layers_longest_path(4, links)
        assert len(layers) == 3
        assert 0 in layers[0]
        assert 3 in layers[2]

    def test_assign_layers_empty(self):
        """Empty graph should return empty layers."""
        layers = assign_layers_longest_path(0, [])
        assert layers == []


class TestCrossingMinimization:
    """Tests for crossing minimization."""

    def test_minimize_crossings_reduces_count(self):
        """Crossing minimization should not increase crossings."""
        # Create a graph with potential crossings
        layers = [[0, 1], [2, 3]]
        links = [
            {"source": 0, "target": 3},  # Crosses with below
            {"source": 1, "target": 2},
        ]

        initial_crossings = count_crossings(layers, links)
        new_layers = minimize_crossings_barycenter(layers, links, iterations=10)
        final_crossings = count_crossings(new_layers, links)

        assert final_crossings <= initial_crossings

    def test_minimize_crossings_preserves_nodes(self):
        """All nodes should remain after minimization."""
        layers = [[0, 1, 2], [3, 4, 5]]
        links = [
            {"source": 0, "target": 4},
            {"source": 1, "target": 3},
            {"source": 2, "target": 5},
        ]

        new_layers = minimize_crossings_barycenter(layers, links)

        all_nodes_before = set(sum(layers, []))
        all_nodes_after = set(sum(new_layers, []))
        assert all_nodes_before == all_nodes_after


class TestCountCrossings:
    """Tests for crossing counting."""

    def test_count_crossings_none(self):
        """Parallel edges should have no crossings."""
        layers = [[0, 1], [2, 3]]
        links = [
            {"source": 0, "target": 2},
            {"source": 1, "target": 3},
        ]
        assert count_crossings(layers, links) == 0

    def test_count_crossings_one(self):
        """Crossing edges should be counted."""
        layers = [[0, 1], [2, 3]]
        links = [
            {"source": 0, "target": 3},
            {"source": 1, "target": 2},
        ]
        assert count_crossings(layers, links) == 1

    def test_count_crossings_empty(self):
        """Empty graph should have no crossings."""
        assert count_crossings([], []) == 0
        assert count_crossings([[0, 1]], []) == 0
