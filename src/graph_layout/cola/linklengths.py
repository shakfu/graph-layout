"""
Link length utilities and constraint generation.

This module provides utilities for computing link lengths based on graph
structure and generating constraints for directed graphs.
"""

from __future__ import annotations

import math
from typing import Callable, Generic, Literal, Optional, TypeVar

T = TypeVar("T")
Axis = Literal["x", "y"]


class LinkAccessor(Generic[T]):
    """Base class for link accessors."""

    def get_source_index(self, l: T) -> int:
        """Get source node index for a link."""
        raise NotImplementedError

    def get_target_index(self, l: T) -> int:
        """Get target node index for a link."""
        raise NotImplementedError


class LinkLengthAccessor(LinkAccessor[T]):
    """Link accessor with length setting capability."""

    def set_length(self, l: T, value: float) -> None:
        """Set the length for a link."""
        raise NotImplementedError


class LinkSepAccessor(LinkAccessor[T]):
    """Link accessor with minimum separation."""

    def get_min_separation(self, l: T) -> float:
        """Get minimum separation for a link."""
        raise NotImplementedError


def _union_count(a: dict, b: dict) -> int:
    """Compute the size of the union of two sets."""
    u = {}
    for i in a:
        u[i] = True
    for i in b:
        u[i] = True
    return len(u)


def _intersection_count(a: dict, b: dict) -> int:
    """Compute the size of the intersection of two sets."""
    n = 0
    for i in a:
        if i in b:
            n += 1
    return n


def _get_neighbours(links: list[T], la: LinkAccessor[T]) -> dict[int, dict]:
    """
    Get neighbor sets for all nodes.

    Args:
        links: List of links
        la: Link accessor

    Returns:
        Dict mapping node indices to their neighbor sets
    """
    neighbours: dict[int, dict] = {}

    def add_neighbours(u: int, v: int) -> None:
        if u not in neighbours:
            neighbours[u] = {}
        neighbours[u][v] = True

    for e in links:
        u = la.get_source_index(e)
        v = la.get_target_index(e)
        add_neighbours(u, v)
        add_neighbours(v, u)

    return neighbours


def _compute_link_lengths(
    links: list[T], w: float, f: Callable[[dict, dict], float], la: LinkLengthAccessor[T]
) -> None:
    """
    Compute and set link lengths based on neighbor similarity.

    Args:
        links: List of links
        w: Weight factor
        f: Function to compute similarity between neighbor sets
        la: Link length accessor
    """
    neighbours = _get_neighbours(links, la)

    for l in links:
        a = neighbours[la.get_source_index(l)]
        b = neighbours[la.get_target_index(l)]
        la.set_length(l, 1 + w * f(a, b))


def symmetric_diff_link_lengths(links: list[T], la: LinkLengthAccessor[T], w: float = 1.0) -> None:
    """
    Modify link lengths based on symmetric difference of neighbors.

    Args:
        links: List of links
        la: Link length accessor
        w: Weight factor
    """

    def f(a: dict, b: dict) -> float:
        return math.sqrt(_union_count(a, b) - _intersection_count(a, b))

    _compute_link_lengths(links, w, f, la)


def jaccard_link_lengths(links: list[T], la: LinkLengthAccessor[T], w: float = 1.0) -> None:
    """
    Modify link lengths based on Jaccard similarity of neighbors.

    Args:
        links: List of links
        la: Link length accessor
        w: Weight factor
    """

    def f(a: dict, b: dict) -> float:
        min_size = min(len(a), len(b))
        if min_size < 1.1:
            return 0.0
        return _intersection_count(a, b) / _union_count(a, b)

    _compute_link_lengths(links, w, f, la)


class SeparationConstraint:
    """Separation constraint between two nodes."""

    def __init__(self, axis: Axis, left: int, right: int, gap: float = 0.0, equality: bool = False):
        self.type = "separation"
        self.axis = axis
        self.left = left
        self.right = right
        self.gap = gap
        self.equality = equality


class AlignmentSpecification:
    """Specification for alignment of a node."""

    def __init__(self, node: int, offset: float = 0.0):
        self.node = node
        self.offset = offset


class AlignmentConstraint:
    """Alignment constraint for multiple nodes."""

    def __init__(self, axis: Axis, offsets: list[AlignmentSpecification]):
        self.type = "alignment"
        self.axis = axis
        self.offsets = offsets


def generate_directed_edge_constraints(
    n: int, links: list[T], axis: Axis, la: LinkSepAccessor[T]
) -> list[SeparationConstraint]:
    """
    Generate separation constraints for directed edges.

    Generates constraints for all edges unless both source and sink
    are in the same strongly connected component.

    Args:
        n: Number of nodes
        links: List of links
        axis: Axis for constraints
        la: Link accessor with minimum separation

    Returns:
        List of separation constraints
    """
    components = strongly_connected_components(n, links, la)

    # Map each node to its component index
    nodes = {}
    for i, c in enumerate(components):
        for v in c:
            nodes[v] = i

    constraints: list[SeparationConstraint] = []
    for l in links:
        ui = la.get_source_index(l)
        vi = la.get_target_index(l)
        u = nodes[ui]
        v = nodes[vi]

        if u != v:
            constraints.append(
                SeparationConstraint(axis=axis, left=ui, right=vi, gap=la.get_min_separation(l))
            )

    return constraints


def strongly_connected_components(
    num_vertices: int, edges: list[T], la: LinkAccessor[T]
) -> list[list[int]]:
    """
    Tarjan's algorithm for finding strongly connected components.

    Args:
        num_vertices: Number of vertices
        edges: List of edges
        la: Link accessor

    Returns:
        List of strongly connected components, each a list of node indices
    """

    class Node:
        def __init__(self, id: int):
            self.id = id
            self.out: list[Node] = []
            self.index: Optional[int] = None
            self.lowlink: Optional[int] = None
            self.on_stack: bool = False

    nodes = [Node(i) for i in range(num_vertices)]
    index = 0
    stack: list[Node] = []
    components: list[list[int]] = []

    def strong_connect(v: Node) -> None:
        nonlocal index

        # Set the depth index for v to the smallest unused index
        v.index = v.lowlink = index
        index += 1
        stack.append(v)
        v.on_stack = True

        # Consider successors of v
        for w in v.out:
            if w.index is None:
                # Successor w has not yet been visited; recurse on it
                strong_connect(w)
                if v.lowlink is not None and w.lowlink is not None:
                    v.lowlink = min(v.lowlink, w.lowlink)
            elif w.on_stack:
                # Successor w is in stack and hence in the current SCC
                if v.lowlink is not None and w.index is not None:
                    v.lowlink = min(v.lowlink, w.index)

        # If v is a root node, pop the stack and generate an SCC
        if v.lowlink == v.index:
            # Start a new strongly connected component
            component = []
            while stack:
                w = stack.pop()
                w.on_stack = False
                # Add w to current strongly connected component
                component.append(w)
                if w == v:
                    break
            # Output the current strongly connected component
            components.append([node.id for node in component])

    # Build graph
    for e in edges:
        v = nodes[la.get_source_index(e)]
        w = nodes[la.get_target_index(e)]
        v.out.append(w)

    # Find SCCs
    for v in nodes:
        if v.index is None:
            strong_connect(v)

    return components
