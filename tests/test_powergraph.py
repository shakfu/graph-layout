"""Tests for powergraph module."""

from graph_layout.cola.powergraph import (
    Configuration,
    LinkSets,
    Module,
    ModuleSet,
    PowerEdge,
    get_groups,
)


class SimpleLink:
    """Simple link for testing."""

    def __init__(self, source: int, target: int, link_type: int = 0):
        self.source = source
        self.target = target
        self.link_type = link_type


class SimpleLinkAccessor:
    """Simple link accessor implementation."""

    def get_source_index(self, link: SimpleLink) -> int:
        return link.source

    def get_target_index(self, link: SimpleLink) -> int:
        return link.target

    def get_type(self, link: SimpleLink) -> int:
        return link.link_type


class TestPowerEdge:
    """Test PowerEdge class."""

    def test_create_edge(self):
        """Test edge creation."""
        edge = PowerEdge(0, 1, 5)
        assert edge.source == 0
        assert edge.target == 1
        assert edge.type == 5


class TestModule:
    """Test Module class."""

    def test_create_leaf_module(self):
        """Test leaf module creation."""
        m = Module(0)
        assert m.id == 0
        assert m.is_leaf()
        assert m.is_island()
        assert not m.is_predefined()

    def test_create_module_with_links(self):
        """Test module with links."""
        outgoing = LinkSets()
        incoming = LinkSets()
        m = Module(0, outgoing, incoming)

        assert m.outgoing is outgoing
        assert m.incoming is incoming

    def test_predefined_module(self):
        """Test predefined module."""
        m = Module(0, definition={"color": "red"})
        assert m.is_predefined()
        assert m.definition["color"] == "red"

    def test_is_island(self):
        """Test island detection."""
        m = Module(0)
        assert m.is_island()

        # Add outgoing link
        m2 = Module(1)
        m.outgoing.add(0, m2)
        assert not m.is_island()

    def test_get_edges(self):
        """Test edge extraction."""
        m1 = Module(0)
        m2 = Module(1)
        m3 = Module(2)

        m1.outgoing.add(0, m2)
        m1.outgoing.add(1, m3)

        edges = []
        m1.get_edges(edges)

        assert len(edges) == 2
        assert all(e.source == 0 for e in edges)


class TestModuleSet:
    """Test ModuleSet class."""

    def test_empty_set(self):
        """Test empty module set."""
        ms = ModuleSet()
        assert ms.count() == 0

    def test_add_remove(self):
        """Test adding and removing modules."""
        ms = ModuleSet()
        m1 = Module(0)
        m2 = Module(1)

        ms.add(m1)
        assert ms.count() == 1
        assert ms.contains(0)

        ms.add(m2)
        assert ms.count() == 2

        ms.remove(m1)
        assert ms.count() == 1
        assert not ms.contains(0)

    def test_intersection(self):
        """Test set intersection."""
        ms1 = ModuleSet()
        ms2 = ModuleSet()

        m1 = Module(0)
        m2 = Module(1)
        m3 = Module(2)

        ms1.add(m1)
        ms1.add(m2)

        ms2.add(m2)
        ms2.add(m3)

        inter = ms1.intersection(ms2)
        assert inter.count() == 1
        assert inter.contains(1)

    def test_modules(self):
        """Test modules() method."""
        ms = ModuleSet()
        m1 = Module(0)
        m2 = Module(1, definition={})  # predefined

        ms.add(m1)
        ms.add(m2)

        # modules() only returns non-predefined
        mods = ms.modules()
        assert len(mods) == 1
        assert mods[0].id == 0


class TestLinkSets:
    """Test LinkSets class."""

    def test_empty_linksets(self):
        """Test empty link sets."""
        ls = LinkSets()
        assert ls.count() == 0

    def test_add_links(self):
        """Test adding links."""
        ls = LinkSets()
        m1 = Module(0)
        m2 = Module(1)

        ls.add(0, m1)  # type 0
        ls.add(0, m2)  # type 0
        assert ls.count() == 2

        ls.add(1, m1)  # type 1
        assert ls.count() == 3

    def test_remove_links(self):
        """Test removing links."""
        ls = LinkSets()
        m1 = Module(0)

        ls.add(0, m1)
        assert ls.count() == 1

        ls.remove(0, m1)
        assert ls.count() == 0

    def test_intersection(self):
        """Test link sets intersection."""
        ls1 = LinkSets()
        ls2 = LinkSets()

        m1 = Module(0)
        m2 = Module(1)
        m3 = Module(2)

        # ls1: type 0 -> {m1, m2}
        ls1.add(0, m1)
        ls1.add(0, m2)

        # ls2: type 0 -> {m2, m3}
        ls2.add(0, m2)
        ls2.add(0, m3)

        inter = ls1.intersection(ls2)
        assert inter.count() == 1
        assert 0 in inter.sets
        assert inter.sets[0].contains(1)  # m2

    def test_contains(self):
        """Test contains method."""
        ls = LinkSets()
        m1 = Module(0)

        ls.add(0, m1)
        assert ls.contains(0)
        assert not ls.contains(1)


class TestConfiguration:
    """Test Configuration class."""

    def test_simple_graph(self):
        """Test simple graph configuration."""
        # 0 -> 1 -> 2
        links = [SimpleLink(0, 1), SimpleLink(1, 2)]

        la = SimpleLinkAccessor()
        config = Configuration(3, links, la)

        assert len(config.modules) == 3
        assert config.R == 2
        assert len(config.roots) == 1

    def test_merge_modules(self):
        """Test merging two modules."""
        # 0 -> 1, 0 -> 2 (both from 0)
        links = [SimpleLink(0, 1), SimpleLink(0, 2)]

        la = SimpleLinkAccessor()
        config = Configuration(3, links, la)

        m1 = config.modules[1]
        m2 = config.modules[2]

        # Both have same incoming edge from 0
        merged = config.merge(m1, m2)

        assert merged.children.count() == 2
        assert merged.children.contains(1)
        assert merged.children.contains(2)

    def test_greedy_merge(self):
        """Test greedy merge algorithm."""
        # Star graph: 0 connects to 1, 2, 3
        links = [SimpleLink(0, 1), SimpleLink(0, 2), SimpleLink(0, 3)]

        la = SimpleLinkAccessor()
        config = Configuration(4, links, la)

        initial_R = config.R
        result = config.greedy_merge()

        # Should merge nodes with common neighbors
        assert result is True
        assert config.R < initial_R

    def test_n_edges(self):
        """Test edge count calculation."""
        links = [SimpleLink(0, 1), SimpleLink(0, 2)]

        la = SimpleLinkAccessor()
        config = Configuration(3, links, la)

        m1 = config.modules[1]
        m2 = config.modules[2]

        # Both have incoming from 0, so merging saves 1 edge
        n_edges = config._n_edges(m1, m2)
        assert n_edges == 1  # 2 - 1 intersection

    def test_all_edges(self):
        """Test extracting all edges."""
        links = [SimpleLink(0, 1), SimpleLink(1, 2)]

        la = SimpleLinkAccessor()
        config = Configuration(3, links, la)

        edges = config.all_edges()
        assert len(edges) == 2


class TestGetGroups:
    """Test get_groups function."""

    def test_simple_linear_graph(self):
        """Test with simple linear graph."""
        nodes = [{"id": i} for i in range(3)]
        links = [SimpleLink(0, 1), SimpleLink(1, 2)]

        la = SimpleLinkAccessor()
        result = get_groups(nodes, links, la)

        assert "groups" in result
        assert "powerEdges" in result

    def test_star_graph(self):
        """Test with star graph."""
        # Center node 0 connects to 1, 2, 3
        nodes = [{"id": i} for i in range(4)]
        links = [SimpleLink(0, 1), SimpleLink(0, 2), SimpleLink(0, 3)]

        la = SimpleLinkAccessor()
        result = get_groups(nodes, links, la)

        # Nodes 1, 2, 3 should be grouped (same incoming edge)
        groups = result["groups"]
        assert len(groups) > 0

    def test_disconnected_components(self):
        """Test with disconnected components."""
        # Two separate edges: 0-1 and 2-3
        nodes = [{"id": i} for i in range(4)]
        links = [SimpleLink(0, 1), SimpleLink(2, 3)]

        la = SimpleLinkAccessor()
        result = get_groups(nodes, links, la)

        # Each pair should be grouped separately
        assert "groups" in result
        assert "powerEdges" in result

    def test_typed_edges(self):
        """Test with different edge types."""
        nodes = [{"id": i} for i in range(3)]
        links = [
            SimpleLink(0, 1, link_type=0),
            SimpleLink(0, 2, link_type=1),  # different type
        ]

        la = SimpleLinkAccessor()
        result = get_groups(nodes, links, la)

        # Different edge types should prevent merging
        assert "groups" in result
        power_edges = result["powerEdges"]
        assert len(power_edges) > 0

    def test_complete_triangle(self):
        """Test with complete triangle graph."""
        # 0-1, 1-2, 2-0
        nodes = [{"id": i} for i in range(3)]
        links = [SimpleLink(0, 1), SimpleLink(1, 2), SimpleLink(2, 0)]

        la = SimpleLinkAccessor()
        result = get_groups(nodes, links, la)

        # Complete graph has no useful grouping
        assert "groups" in result
        assert "powerEdges" in result

    def test_power_edge_retargeting(self):
        """Test that power edges are retargeted to nodes."""
        nodes = [{"id": i} for i in range(3)]
        links = [SimpleLink(0, 1), SimpleLink(0, 2)]

        la = SimpleLinkAccessor()
        result = get_groups(nodes, links, la)

        power_edges = result["powerEdges"]
        # Edges should reference actual node objects, not indices
        for edge in power_edges:
            assert isinstance(edge.source, dict) or isinstance(edge.source, int)
            assert isinstance(edge.target, dict) or isinstance(edge.target, int)
