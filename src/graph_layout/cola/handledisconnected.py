"""
Handle disconnected graph components.

This module provides utilities for separating disconnected components
and packing them efficiently in the layout space.
"""

from __future__ import annotations

import math
from typing import Any

# Packing configuration
PADDING = 10
GOLDEN_SECTION = (1 + math.sqrt(5)) / 2
FLOAT_EPSILON = 0.0001
MAX_ITERATIONS = 100


def separate_graphs(nodes: list[Any], links: list[Any]) -> list[dict]:
    """
    Find connected components in a graph.

    Args:
        nodes: List of nodes with 'index' attribute
        links: List of links with 'source' and 'target' attributes

    Returns:
        List of components, each with 'array' containing nodes
    """
    marks: dict[int, int] = {}
    ways: dict[int, list[Any]] = {}
    graphs: list[dict[str, list[Any]]] = []
    clusters = 0

    # Build adjacency list
    for link in links:
        n1 = link.source
        n2 = link.target

        if n1.index in ways:
            ways[n1.index].append(n2)
        else:
            ways[n1.index] = [n2]

        if n2.index in ways:
            ways[n2.index].append(n1)
        else:
            ways[n2.index] = [n1]

    # Find connected components using DFS
    def explore_node(n: Any, is_new: bool) -> None:
        nonlocal clusters

        if n.index in marks:
            return

        if is_new:
            clusters += 1
            graphs.append({"array": []})

        marks[n.index] = clusters
        graphs[clusters - 1]["array"].append(n)

        adjacent = ways.get(n.index)
        if not adjacent:
            return

        for adj in adjacent:
            explore_node(adj, False)

    # Explore all nodes
    for node in nodes:
        if node.index not in marks:
            explore_node(node, True)

    return graphs


def apply_packing(
    graphs: list[dict],
    w: float,
    h: float,
    node_size: float = 0,
    desired_ratio: float = 1.0,
    center_graph: bool = True,
) -> None:
    """
    Apply box packing algorithm to position disconnected graph components.

    Args:
        graphs: List of graph components with 'array' of nodes
        w: SVG width
        h: SVG height
        node_size: Default node size
        desired_ratio: Desired aspect ratio
        center_graph: Whether to center the final layout
    """
    if len(graphs) == 0:
        return

    svg_width = w
    svg_height = h
    init_x = 0
    init_y = 0

    # Calculate bounding boxes
    _calculate_bb(graphs, node_size)

    # Apply packing algorithm
    real_width, real_height = _apply_packing_algorithm(graphs, desired_ratio)

    # Position nodes
    if center_graph:
        _put_nodes_to_positions(graphs, svg_width, svg_height, real_width, real_height)


def _calculate_bb(graphs: list[dict], node_size: float) -> None:
    """Calculate bounding boxes for all graph components."""
    for g in graphs:
        _calculate_single_bb(g, node_size)


def _calculate_single_bb(graph: dict, node_size: float) -> None:
    """Calculate bounding box for a single graph component."""
    min_x = float("inf")
    min_y = float("inf")
    max_x = -float("inf")
    max_y = -float("inf")

    for v in graph["array"]:
        # Use node's width/height if available and not None, else use node_size
        w = (getattr(v, "width", None) or node_size) / 2
        h = (getattr(v, "height", None) or node_size) / 2

        max_x = max(v.x + w, max_x)
        min_x = min(v.x - w, min_x)
        max_y = max(v.y + h, max_y)
        min_y = min(v.y - h, min_y)

    graph["width"] = max_x - min_x
    graph["height"] = max_y - min_y


def _put_nodes_to_positions(
    graphs: list[dict], svg_width: float, svg_height: float, real_width: float, real_height: float
) -> None:
    """Position nodes in their final locations."""
    for g in graphs:
        # Calculate current graph center
        center_x = sum(node.x for node in g["array"]) / len(g["array"])
        center_y = sum(node.y for node in g["array"]) / len(g["array"])

        # Calculate top-left corner
        corner_x = center_x - g["width"] / 2
        corner_y = center_y - g["height"] / 2

        # Calculate offset to center everything
        offset_x = g["x"] - corner_x + svg_width / 2 - real_width / 2
        offset_y = g["y"] - corner_y + svg_height / 2 - real_height / 2

        # Apply offset to all nodes
        for node in g["array"]:
            node.x += offset_x
            node.y += offset_y


def _apply_packing_algorithm(graphs: list[dict], desired_ratio: float) -> tuple[float, float]:
    """
    Apply golden section search for optimal packing.

    Returns:
        Tuple of (real_width, real_height)
    """
    # Sort by height (descending)
    graphs.sort(key=lambda g: g["height"], reverse=True)

    # Find minimum width
    min_width = min(g["width"] for g in graphs)

    # Binary search bounds
    left = min_width
    right = _get_entire_width(graphs)

    # Golden section search
    curr_best_f = float("inf")
    curr_best = 0
    iteration_counter = 0

    x1 = right - (right - left) / GOLDEN_SECTION
    x2 = left + (right - left) / GOLDEN_SECTION
    f_x1 = _step(graphs, x1, desired_ratio)[2]
    f_x2 = _step(graphs, x2, desired_ratio)[2]

    flag = -1  # which to recompute

    dx = float("inf")
    df = float("inf")

    while (dx > min_width or df > FLOAT_EPSILON) and iteration_counter < MAX_ITERATIONS:
        if flag != 1:
            x1 = right - (right - left) / GOLDEN_SECTION
            f_x1 = _step(graphs, x1, desired_ratio)[2]

        if flag != 0:
            x2 = left + (right - left) / GOLDEN_SECTION
            f_x2 = _step(graphs, x2, desired_ratio)[2]

        dx = abs(x1 - x2)
        df = abs(f_x1 - f_x2)

        if f_x1 < curr_best_f:
            curr_best_f = f_x1
            curr_best = x1

        if f_x2 < curr_best_f:
            curr_best_f = f_x2
            curr_best = x2

        if f_x1 > f_x2:
            left = x1
            x1 = x2
            f_x1 = f_x2
            flag = 1
        else:
            right = x2
            x2 = x1
            f_x2 = f_x1
            flag = 0

        iteration_counter += 1

    # Final step with best width
    real_width, real_height, _ = _step(graphs, curr_best, desired_ratio)
    return real_width, real_height


def _step(graphs: list[dict], max_width: float, desired_ratio: float) -> tuple[float, float, float]:
    """
    One iteration of packing optimization.

    Returns:
        Tuple of (real_width, real_height, cost)
    """
    line: list[dict[str, Any]] = []
    real_width: float = 0.0
    real_height: float = 0.0
    global_bottom: float = 0.0

    for g in graphs:
        real_width, real_height, global_bottom = _put_rect(
            g, max_width, line, real_width, real_height, global_bottom
        )

    ratio = real_width / real_height if real_height > 0 else float("inf")
    cost = abs(ratio - desired_ratio)

    return real_width, real_height, cost


def _put_rect(
    rect: dict,
    max_width: float,
    line: list[dict],
    real_width: float,
    real_height: float,
    global_bottom: float,
) -> tuple[float, float, float]:
    """
    Find position for one rectangle.

    Returns:
        Updated (real_width, real_height, global_bottom)
    """
    parent = None

    # Find parent in current line
    for item in line:
        can_fit_height = item.get("space_left", 0) >= rect["height"]
        can_fit_width = (
            item["x"] + item["width"] + rect["width"] + PADDING - max_width
        ) <= FLOAT_EPSILON

        if can_fit_height and can_fit_width:
            parent = item
            break

    line.append(rect)

    if parent is not None:
        # Place next to parent
        rect["x"] = parent["x"] + parent["width"] + PADDING
        rect["y"] = parent["bottom"]
        rect["space_left"] = rect["height"]
        rect["bottom"] = rect["y"]
        parent["space_left"] -= rect["height"] + PADDING
        parent["bottom"] += rect["height"] + PADDING
    else:
        # Start new row
        rect["y"] = global_bottom
        global_bottom += rect["height"] + PADDING
        rect["x"] = 0
        rect["bottom"] = rect["y"]
        rect["space_left"] = rect["height"]

    # Update real dimensions
    if rect["y"] + rect["height"] - real_height > -FLOAT_EPSILON:
        real_height = rect["y"] + rect["height"]

    if rect["x"] + rect["width"] - real_width > -FLOAT_EPSILON:
        real_width = rect["x"] + rect["width"]

    return real_width, real_height, global_bottom


def _get_entire_width(graphs: list[dict[str, Any]]) -> float:
    """Calculate total width if all graphs placed in a line."""
    return float(sum(g["width"] + PADDING for g in graphs))
