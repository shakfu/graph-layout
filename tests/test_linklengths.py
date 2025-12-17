"""Tests for linklengths module."""

from graph_layout.cola.linklengths import (
    AlignmentConstraint,
    AlignmentSpecification,
    LinkAccessor,
    LinkLengthAccessor,
    LinkSepAccessor,
    SeparationConstraint,
    generate_directed_edge_constraints,
    jaccard_link_lengths,
    strongly_connected_components,
    symmetric_diff_link_lengths,
)


class SimpleLink:
    """Simple link for testing."""

    def __init__(self, source: int, target: int, length: float = 1.0, min_sep: float = 1.0):
        self.source = source
        self.target = target
        self.length = length
        self.min_sep = min_sep


class SimpleLinkAccessor(LinkAccessor[SimpleLink]):
    """Simple link accessor implementation."""

    def get_source_index(self, l: SimpleLink) -> int:
        return l.source

    def get_target_index(self, l: SimpleLink) -> int:
        return l.target


class SimpleLinkLengthAccessor(LinkLengthAccessor[SimpleLink]):
    """Simple link length accessor implementation."""

    def get_source_index(self, l: SimpleLink) -> int:
        return l.source

    def get_target_index(self, l: SimpleLink) -> int:
        return l.target

    def set_length(self, l: SimpleLink, value: float) -> None:
        l.length = value


class SimpleLinkSepAccessor(LinkSepAccessor[SimpleLink]):
    """Simple link separation accessor implementation."""

    def get_source_index(self, l: SimpleLink) -> int:
        return l.source

    def get_target_index(self, l: SimpleLink) -> int:
        return l.target

    def get_min_separation(self, l: SimpleLink) -> float:
        return l.min_sep


class TestSymmetricDiffLinkLengths:
    """Test symmetric difference link length computation."""

    def test_simple_graph(self):
        """Test with simple graph."""
        #  0 -- 1 -- 2
        links = [SimpleLink(0, 1), SimpleLink(1, 2)]

        la = SimpleLinkLengthAccessor()
        symmetric_diff_link_lengths(links, la)

        # Lengths should be updated
        assert links[0].length > 1.0
        assert links[1].length > 1.0

    def test_with_weight(self):
        """Test with custom weight."""
        links = [SimpleLink(0, 1), SimpleLink(1, 2)]

        la = SimpleLinkLengthAccessor()
        symmetric_diff_link_lengths(links, la, w=2.0)

        # With higher weight, differences should be more pronounced
        assert links[0].length >= 1.0


class TestJaccardLinkLengths:
    """Test Jaccard similarity link length computation."""

    def test_simple_graph(self):
        """Test with simple graph."""
        links = [SimpleLink(0, 1), SimpleLink(1, 2), SimpleLink(2, 3)]

        la = SimpleLinkLengthAccessor()
        jaccard_link_lengths(links, la)

        # Lengths should be updated
        for link in links:
            assert link.length >= 1.0

    def test_triangle_graph(self):
        """Test with triangle graph."""
        # Triangle: all nodes connected
        links = [SimpleLink(0, 1), SimpleLink(1, 2), SimpleLink(2, 0)]

        la = SimpleLinkLengthAccessor()
        jaccard_link_lengths(links, la, w=1.0)

        # All links should have updated lengths
        assert all(link.length >= 1.0 for link in links)


class TestSeparationConstraint:
    """Test SeparationConstraint class."""

    def test_create_constraint(self):
        """Test constraint creation."""
        c = SeparationConstraint("x", 0, 1, 5.0)

        assert c.axis == "x"
        assert c.left == 0
        assert c.right == 1
        assert c.gap == 5.0
        assert c.equality == False

    def test_equality_constraint(self):
        """Test equality constraint."""
        c = SeparationConstraint("y", 0, 1, 10.0, equality=True)

        assert c.equality == True


class TestAlignmentConstraint:
    """Test AlignmentConstraint class."""

    def test_create_alignment(self):
        """Test alignment constraint creation."""
        offsets = [AlignmentSpecification(0, 0.0), AlignmentSpecification(1, 5.0)]

        c = AlignmentConstraint("x", offsets)

        assert c.axis == "x"
        assert len(c.offsets) == 2
        assert c.type == "alignment"


class TestStronglyConnectedComponents:
    """Test Tarjan's SCC algorithm."""

    def test_single_node(self):
        """Test with single node."""
        la = SimpleLinkAccessor()
        components = strongly_connected_components(1, [], la)

        assert len(components) == 1
        assert components[0] == [0]

    def test_linear_graph(self):
        """Test with linear directed graph."""
        # 0 -> 1 -> 2
        links = [SimpleLink(0, 1), SimpleLink(1, 2)]

        la = SimpleLinkAccessor()
        components = strongly_connected_components(3, links, la)

        # Each node is its own SCC
        assert len(components) == 3

    def test_cycle(self):
        """Test with cycle."""
        # 0 -> 1 -> 2 -> 0 (cycle)
        links = [SimpleLink(0, 1), SimpleLink(1, 2), SimpleLink(2, 0)]

        la = SimpleLinkAccessor()
        components = strongly_connected_components(3, links, la)

        # All nodes should be in one SCC
        assert len(components) == 1
        assert len(components[0]) == 3

    def test_two_components(self):
        """Test with two separate components."""
        # 0 -> 1 -> 0 (cycle)
        # 2 -> 3 -> 2 (cycle)
        links = [SimpleLink(0, 1), SimpleLink(1, 0), SimpleLink(2, 3), SimpleLink(3, 2)]

        la = SimpleLinkAccessor()
        components = strongly_connected_components(4, links, la)

        # Should have 2 SCCs
        assert len(components) == 2
        assert all(len(c) == 2 for c in components)

    def test_complex_graph(self):
        """Test with more complex graph."""
        # 0 -> 1 -> 2 -> 3
        #      ^         |
        #      |_________|
        # Creates SCC with 1, 2, 3
        links = [
            SimpleLink(0, 1),
            SimpleLink(1, 2),
            SimpleLink(2, 3),
            SimpleLink(3, 1),  # Back edge creating cycle
        ]

        la = SimpleLinkAccessor()
        components = strongly_connected_components(4, links, la)

        # Should have 2 components: {0} and {1,2,3}
        assert len(components) == 2


class TestGenerateDirectedEdgeConstraints:
    """Test directed edge constraint generation."""

    def test_simple_dag(self):
        """Test with simple DAG."""
        # 0 -> 1 -> 2
        links = [SimpleLink(0, 1, min_sep=5.0), SimpleLink(1, 2, min_sep=5.0)]

        la = SimpleLinkSepAccessor()
        constraints = generate_directed_edge_constraints(3, links, "x", la)

        # Should generate constraints for both edges
        assert len(constraints) == 2
        assert all(c.axis == "x" for c in constraints)

    def test_with_cycle(self):
        """Test with cycle (SCC)."""
        # 0 -> 1 -> 2 -> 0 (cycle)
        links = [SimpleLink(0, 1), SimpleLink(1, 2), SimpleLink(2, 0)]

        la = SimpleLinkSepAccessor()
        constraints = generate_directed_edge_constraints(3, links, "x", la)

        # No constraints within SCC
        assert len(constraints) == 0

    def test_mixed_graph(self):
        """Test with mix of SCC and DAG."""
        # 0 -> 1 <-> 2 (1 and 2 form SCC)
        # 2 -> 3
        links = [
            SimpleLink(0, 1, min_sep=5.0),
            SimpleLink(1, 2, min_sep=5.0),
            SimpleLink(2, 1, min_sep=5.0),  # Back edge
            SimpleLink(2, 3, min_sep=5.0),
        ]

        la = SimpleLinkSepAccessor()
        constraints = generate_directed_edge_constraints(4, links, "y", la)

        # Should generate constraints for edges crossing SCC boundaries
        assert len(constraints) > 0
        assert all(c.axis == "y" for c in constraints)

    def test_constraint_properties(self):
        """Test that constraints have correct properties."""
        links = [SimpleLink(0, 1, min_sep=10.0)]

        la = SimpleLinkSepAccessor()
        constraints = generate_directed_edge_constraints(2, links, "x", la)

        assert len(constraints) == 1
        c = constraints[0]
        assert c.left == 0
        assert c.right == 1
        assert c.gap == 10.0
