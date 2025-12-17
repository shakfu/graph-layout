"""
Power graph clustering algorithm.

This module implements hierarchical graph clustering using greedy merging
based on edge intersection patterns.
"""

from __future__ import annotations

from typing import Any, Callable, Generic, Optional, TypeVar

from .linklengths import LinkAccessor

T = TypeVar("T")


class LinkTypeAccessor(LinkAccessor[T]):
    """Link accessor that also provides edge type information."""

    def get_type(self, link: T) -> int:
        """Return a unique identifier for the type of the link."""
        raise NotImplementedError


class PowerEdge:
    """Edge in the power graph."""

    def __init__(self, source: Any, target: Any, edge_type: int):
        self.source = source
        self.target = target
        self.type = edge_type


class ModuleSet:
    """Set of modules indexed by ID."""

    def __init__(self) -> None:
        self.table: dict[int, Module] = {}

    def count(self) -> int:
        """Return number of modules in set."""
        return len(self.table)

    def intersection(self, other: ModuleSet) -> ModuleSet:
        """Return intersection of this set with another."""
        result = ModuleSet()
        for key in self.table:
            if key in other.table:
                result.table[key] = self.table[key]
        return result

    def intersection_count(self, other: ModuleSet) -> int:
        """Return count of intersection."""
        return self.intersection(other).count()

    def contains(self, module_id: int) -> bool:
        """Check if module ID is in set."""
        return module_id in self.table

    def add(self, m: Module) -> None:
        """Add module to set."""
        self.table[m.id] = m

    def remove(self, m: Module) -> None:
        """Remove module from set."""
        if m.id in self.table:
            del self.table[m.id]

    def for_all(self, func: Callable[["Module"], None]) -> None:
        """Apply function to all modules."""
        for module in self.table.values():
            func(module)

    def modules(self) -> list[Module]:
        """Return list of non-predefined modules."""
        result = []
        for module in self.table.values():
            if not module.is_predefined():
                result.append(module)
        return result


class LinkSets:
    """Links organized by type."""

    def __init__(self) -> None:
        self.sets: dict[int, ModuleSet] = {}
        self.n: int = 0

    def count(self) -> int:
        """Return total number of links."""
        return self.n

    def contains(self, module_id: int) -> bool:
        """Check if any link set contains module ID."""
        for module_set in self.sets.values():
            if module_set.contains(module_id):
                return True
        return False

    def add(self, linktype: int, m: Module) -> None:
        """Add module to link set of given type."""
        if linktype not in self.sets:
            self.sets[linktype] = ModuleSet()
        self.sets[linktype].add(m)
        self.n += 1

    def remove(self, linktype: int, m: Module) -> None:
        """Remove module from link set of given type."""
        if linktype in self.sets:
            ms = self.sets[linktype]
            ms.remove(m)
            if ms.count() == 0:
                del self.sets[linktype]
            self.n -= 1

    def for_all(self, func: Callable[[ModuleSet, int], None]) -> None:
        """Apply function to all (ModuleSet, linktype) pairs."""
        for linktype, module_set in self.sets.items():
            func(module_set, linktype)

    def for_all_modules(self, func: Callable[["Module"], None]) -> None:
        """Apply function to all modules across all link types."""

        def wrapper(ms: ModuleSet, lt: int) -> None:
            ms.for_all(func)

        self.for_all(wrapper)

    def intersection(self, other: LinkSets) -> LinkSets:
        """Return intersection of link sets."""
        result = LinkSets()
        for linktype, ms in self.sets.items():
            if linktype in other.sets:
                i = ms.intersection(other.sets[linktype])
                n = i.count()
                if n > 0:
                    result.sets[linktype] = i
                    result.n += n
        return result


class Module:
    """A module (cluster) in the power graph."""

    def __init__(
        self,
        module_id: int,
        outgoing: Optional[LinkSets] = None,
        incoming: Optional[LinkSets] = None,
        children: Optional[ModuleSet] = None,
        definition: Optional[dict] = None,
    ):
        self.id = module_id
        self.outgoing = outgoing if outgoing is not None else LinkSets()
        self.incoming = incoming if incoming is not None else LinkSets()
        self.children = children if children is not None else ModuleSet()
        self.definition = definition
        self.gid: Optional[int] = None

    def get_edges(self, es: list[PowerEdge]) -> None:
        """Add this module's edges to list."""

        def add_edges(ms: ModuleSet, edgetype: int) -> None:
            def add_edge(target: Module) -> None:
                es.append(PowerEdge(self.id, target.id, edgetype))

            ms.for_all(add_edge)

        self.outgoing.for_all(add_edges)

    def is_leaf(self) -> bool:
        """Check if this is a leaf module."""
        return self.children.count() == 0

    def is_island(self) -> bool:
        """Check if this module has no connections."""
        return self.outgoing.count() == 0 and self.incoming.count() == 0

    def is_predefined(self) -> bool:
        """Check if this module was predefined."""
        return self.definition is not None


class Configuration(Generic[T]):
    """Power graph configuration manager."""

    def __init__(
        self,
        n: int,
        edges: list[T],
        link_accessor: LinkTypeAccessor[T],
        root_group: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize configuration.

        Args:
            n: Number of nodes
            edges: List of edges
            link_accessor: Accessor for edge properties
            root_group: Optional predefined group structure
        """
        self.modules: list[Module] = [None] * n  # type: ignore
        self.roots: list[ModuleSet] = []
        self.link_accessor = link_accessor

        if root_group:
            self._init_modules_from_group(root_group)
        else:
            self.roots.append(ModuleSet())
            for i in range(n):
                module = Module(i)
                self.modules[i] = module
                self.roots[0].add(module)

        self.R = len(edges)

        for e in edges:
            s = self.modules[link_accessor.get_source_index(e)]
            t = self.modules[link_accessor.get_target_index(e)]
            edge_type = link_accessor.get_type(e)
            s.outgoing.add(edge_type, t)
            t.incoming.add(edge_type, s)

    def _init_modules_from_group(self, group: dict) -> ModuleSet:
        """Initialize modules from predefined group structure."""
        module_set = ModuleSet()
        self.roots.append(module_set)

        for node in group.get("leaves", []):
            node_id = node.get("id", node) if isinstance(node, dict) else node
            module = Module(node_id)
            self.modules[node_id] = module
            module_set.add(module)

        for j, child in enumerate(group.get("groups", [])):
            # Propagate group properties
            definition = {}
            for prop, value in child.items():
                if prop not in ["leaves", "groups"]:
                    definition[prop] = value

            # Use negative module id to avoid clashes
            child_set = self._init_modules_from_group(child)
            module = Module(-1 - j, LinkSets(), LinkSets(), child_set, definition)
            module_set.add(module)
            self.modules.append(module)

        return module_set

    def merge(self, a: Module, b: Module, k: int = 0) -> Module:
        """
        Merge two modules.

        Args:
            a: First module
            b: Second module
            k: Root index

        Returns:
            New merged module
        """
        in_int = a.incoming.intersection(b.incoming)
        out_int = a.outgoing.intersection(b.outgoing)

        children = ModuleSet()
        children.add(a)
        children.add(b)

        m = Module(len(self.modules), out_int, in_int, children)
        self.modules.append(m)

        def update(s: LinkSets, incoming_attr: str, outgoing_attr: str) -> None:
            def process_linktype(ms: ModuleSet, linktype: int) -> None:
                def process_node(n: Module) -> None:
                    nls = getattr(n, incoming_attr)
                    nls.add(linktype, m)
                    nls.remove(linktype, a)
                    nls.remove(linktype, b)
                    getattr(a, outgoing_attr).remove(linktype, n)
                    getattr(b, outgoing_attr).remove(linktype, n)

                ms.for_all(process_node)

            s.for_all(process_linktype)

        update(out_int, "incoming", "outgoing")
        update(in_int, "outgoing", "incoming")

        self.R -= in_int.count() + out_int.count()
        self.roots[k].remove(a)
        self.roots[k].remove(b)
        self.roots[k].add(m)

        return m

    def _root_merges(self, k: int = 0) -> list[dict]:
        """Get all possible merges for root modules."""
        rs = self.roots[k].modules()
        n = len(rs)
        merges = []
        ctr = 0

        for i in range(n - 1):
            for j in range(i + 1, n):
                a = rs[i]
                b = rs[j]
                merges.append({"id": ctr, "nEdges": self._n_edges(a, b), "a": a, "b": b})
                ctr += 1

        return merges

    def greedy_merge(self) -> bool:
        """
        Perform one greedy merge.

        Returns:
            True if merge was performed, False otherwise
        """
        for i in range(len(self.roots)):
            # Handle single nested module case
            if len(self.roots[i].modules()) < 2:
                continue

            # Find merge that removes most edges
            ms = sorted(self._root_merges(i), key=lambda x: (x["nEdges"], x["id"]))

            m = ms[0]
            if m["nEdges"] >= self.R:
                continue

            self.merge(m["a"], m["b"], i)
            return True

        return False

    def _n_edges(self, a: Module, b: Module) -> int:
        """Calculate number of edges after merging a and b."""
        in_int = a.incoming.intersection(b.incoming)
        out_int = a.outgoing.intersection(b.outgoing)
        return self.R - in_int.count() - out_int.count()

    def get_group_hierarchy(self, retargeted_edges: list[PowerEdge]) -> list[dict]:
        """
        Get group hierarchy structure.

        Args:
            retargeted_edges: Output list for retargeted edges

        Returns:
            List of groups
        """
        groups: list[dict] = []
        root: dict = {}
        _to_groups(self.roots[0], root, groups)

        es = self.all_edges()
        for e in es:
            a = self.modules[e.source]
            b = self.modules[e.target]

            source = e.source if a.gid is None else groups[a.gid]
            target = e.target if b.gid is None else groups[b.gid]

            retargeted_edges.append(PowerEdge(source, target, e.type))

        return groups

    def all_edges(self) -> list[PowerEdge]:
        """Get all edges in the power graph."""
        es: list[PowerEdge] = []
        Configuration._get_edges(self.roots[0], es)
        return es

    @staticmethod
    def _get_edges(modules: ModuleSet, es: list[PowerEdge]) -> None:
        """Recursively collect edges from modules."""

        def process_module(m: Module) -> None:
            m.get_edges(es)
            Configuration._get_edges(m.children, es)

        modules.for_all(process_module)


def _to_groups(modules: ModuleSet, group: dict[str, Any], groups: list[dict[str, Any]]) -> None:
    """Convert module hierarchy to group structure."""

    def process_module(m: Module) -> None:
        if m.is_leaf():
            if "leaves" not in group:
                group["leaves"] = []
            group["leaves"].append(m.id)
        else:
            g: dict[str, Any] = group
            m.gid = len(groups)

            if not m.is_island() or m.is_predefined():
                g = {"id": m.gid}
                if m.is_predefined() and m.definition is not None:
                    # Apply original group properties
                    for prop, value in m.definition.items():
                        g[prop] = value

                if "groups" not in group:
                    group["groups"] = []
                group["groups"].append(m.gid)
                groups.append(g)

            _to_groups(m.children, g, groups)

    modules.for_all(process_module)


def get_groups(
    nodes: list[Any],
    links: list[T],
    link_accessor: LinkTypeAccessor[T],
    root_group: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Generate power graph groups from node and link structure.

    Args:
        nodes: List of nodes
        links: List of links
        link_accessor: Accessor for link properties
        root_group: Optional predefined group structure

    Returns:
        Dictionary with 'groups' and 'powerEdges' keys
    """
    n = len(nodes)
    config = Configuration(n, links, link_accessor, root_group)

    while config.greedy_merge():
        pass

    power_edges: list[PowerEdge] = []
    groups = config.get_group_hierarchy(power_edges)

    # Replace node indices with actual node objects
    for e in power_edges:
        if isinstance(e.source, int):
            e.source = nodes[e.source]
        if isinstance(e.target, int):
            e.target = nodes[e.target]

    return {"groups": groups, "powerEdges": power_edges}
