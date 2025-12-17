"""
Batch layout operations.

This module provides utility functions for grid-based layouts and
power graph layouts with edge routing.
"""

from __future__ import annotations

from typing import Any

from .geom import Point
from .gridrouter import GridRouter
from .layout import Layout, Node


def gridify(
    pg_layout: dict, nudge_gap: float, margin: float, group_margin: float
) -> list[list[list[Point]]]:
    """
    Apply grid-based edge routing to a power graph layout.

    Args:
        pg_layout: Power graph layout dict with 'cola' and 'powerGraph' keys
        nudge_gap: Spacing between parallel edge segments
        margin: Space around nodes
        group_margin: Space around groups

    Returns:
        List of routed edge paths
    """
    cola = pg_layout["cola"]
    power_graph = pg_layout["powerGraph"]

    # Run initial layout
    cola.start(0, 0, 0, 10, False)

    # Set up grid router
    gridrouter = _route(cola.nodes(), cola.groups(), margin, group_margin)

    # Route edges
    return gridrouter.route_edges(
        power_graph.powerEdges,
        nudge_gap,
        lambda e: e.source.routerNode.id,
        lambda e: e.target.routerNode.id,
    )


def _route(nodes: list[Node], groups: list[Any], margin: float, group_margin: float) -> GridRouter:
    """
    Create a grid router for the given nodes and groups.

    Args:
        nodes: List of nodes
        groups: List of groups
        margin: Space around nodes
        group_margin: Space around groups

    Returns:
        Configured GridRouter
    """
    # Create router nodes for each node
    for d in nodes:
        d.routerNode = type(
            "RouterNode",
            (),
            {"name": getattr(d, "name", None), "bounds": d.bounds.inflate(-margin)},
        )()

    # Create router nodes for each group
    for d in groups:
        children = []

        # Add group children indices
        if hasattr(d, "groups") and d.groups:
            children.extend([len(nodes) + g.id for g in d.groups])

        # Add leaf children indices
        if hasattr(d, "leaves") and d.leaves:
            children.extend([leaf.index for leaf in d.leaves])

        d.routerNode = type(
            "RouterNode", (), {"bounds": d.bounds.inflate(-group_margin), "children": children}
        )()

    # Combine nodes and groups
    grid_router_nodes = []
    for i, d in enumerate(nodes + groups):
        d.routerNode.id = i
        grid_router_nodes.append(d.routerNode)

    # Create accessor for router nodes
    class RouterNodeAccessor:
        def get_children(self, v: Any) -> list[int]:
            return getattr(v, "children", [])

        def get_bounds(self, v: Any):
            return v.bounds

    return GridRouter(grid_router_nodes, RouterNodeAccessor(), margin - group_margin)


def power_graph_grid_layout(
    graph: dict[str, Any], size: list[float], grouppadding: float
) -> dict[str, Any]:
    """
    Create a power graph layout with grid-based edge routing.

    Args:
        graph: Graph dict with 'nodes' and 'links' keys
        size: Layout size [width, height]
        grouppadding: Padding around groups

    Returns:
        Dict with 'cola' (Layout) and 'powerGraph' keys
    """
    # Initialize node indices
    for i, v in enumerate(graph["nodes"]):
        v.index = i

    # Compute power graph
    power_graph = None

    def power_graph_callback(d):
        nonlocal power_graph
        power_graph = d
        for v in power_graph.groups:
            v.padding = grouppadding

    Layout().avoid_overlaps(False).nodes(graph["nodes"]).links(graph["links"]).power_graph_groups(
        power_graph_callback
    )

    # Construct flat graph with dummy nodes for groups
    n = len(graph["nodes"])
    edges = []
    vs = graph["nodes"][:]

    # Set indices
    for i, v in enumerate(vs):
        v.index = i

    # Add group nodes and edges
    for g in power_graph.groups:
        source_ind = g.id + n
        g.index = source_ind
        vs.append(g)

        # Add edges from group to leaves
        if hasattr(g, "leaves") and g.leaves:
            for v in g.leaves:
                edges.append({"source": source_ind, "target": v.index})

        # Add edges from group to subgroups
        if hasattr(g, "groups") and g.groups:
            for gg in g.groups:
                edges.append({"source": source_ind, "target": gg.id + n})

    # Add power edges
    for e in power_graph.powerEdges:
        edges.append({"source": e.source.index, "target": e.target.index})

    # Layout flat graph with dummy nodes
    Layout().size(size).nodes(vs).links(edges).avoid_overlaps(False).link_distance(
        30
    ).symmetric_diff_link_lengths(5).convergence_threshold(1e-4).start(100, 0, 0, 0, False)

    # Final layout with group constraints
    cola = (
        Layout()
        .convergence_threshold(1e-3)
        .size(size)
        .avoid_overlaps(True)
        .nodes(graph["nodes"])
        .links(graph["links"])
        .group_compactness(1e-4)
        .link_distance(30)
        .symmetric_diff_link_lengths(5)
        .power_graph_groups(power_graph_callback)
        .start(50, 0, 100, 0, False)
    )

    return {"cola": cola, "powerGraph": power_graph}
