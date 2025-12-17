"""
Layout quality metrics.

Provides quantitative measures of layout quality:
- Edge crossings: Number of intersecting edges
- Stress: How well distances match ideal distances
- Edge length variance: Uniformity of edge lengths
- Angular resolution: Minimum angle between edges at each node

All metrics work with final node positions from any layout algorithm.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, List, Optional, Sequence, Tuple, Union

from .types import Link, Node


def edge_crossings(nodes: Sequence[Node], links: Sequence[Link]) -> int:
    """
    Count the number of edge crossings in the layout.

    Two edges cross if their line segments intersect (excluding
    shared endpoints).

    Args:
        nodes: List of positioned nodes
        links: List of links

    Returns:
        Number of edge crossings

    Time Complexity: O(m^2) where m = number of edges
    """
    crossings = 0
    n_links = len(links)

    for i in range(n_links):
        for j in range(i + 1, n_links):
            if _edges_cross(nodes, links[i], links[j]):
                crossings += 1

    return crossings


def _get_link_index(endpoint: Union[Node, int]) -> int:
    """Get index from a link endpoint (Node or int)."""
    if isinstance(endpoint, int):
        return endpoint
    return endpoint.index if endpoint.index is not None else 0


def _edges_cross(nodes: Sequence[Node], e1: Link, e2: Link) -> bool:
    """Check if two edges cross (not at shared endpoints)."""
    s1 = _get_link_index(e1.source)
    t1 = _get_link_index(e1.target)
    s2 = _get_link_index(e2.source)
    t2 = _get_link_index(e2.target)

    # Skip if edges share an endpoint
    if s1 == s2 or s1 == t2 or t1 == s2 or t1 == t2:
        return False

    # Bounds check
    n = len(nodes)
    if not (0 <= s1 < n and 0 <= t1 < n and 0 <= s2 < n and 0 <= t2 < n):
        return False

    # Get coordinates
    p1 = (nodes[s1].x, nodes[s1].y)
    p2 = (nodes[t1].x, nodes[t1].y)
    p3 = (nodes[s2].x, nodes[s2].y)
    p4 = (nodes[t2].x, nodes[t2].y)

    return _segments_intersect(p1, p2, p3, p4)


def _segments_intersect(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float],
) -> bool:
    """Check if line segments (p1,p2) and (p3,p4) intersect."""

    def ccw(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> bool:
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


def stress(
    nodes: Sequence[Node],
    ideal_distances: Optional[Sequence[Sequence[float]]] = None,
    links: Optional[Sequence[Link]] = None,
    edge_length: float = 100.0,
) -> float:
    """
    Compute the stress value of the layout.

    Stress measures how well actual pairwise distances match ideal distances:
    stress = sum_ij (w_ij * (d_ij - D_ij)^2) / sum_ij (w_ij * D_ij^2)

    where d_ij is actual distance, D_ij is ideal distance,
    w_ij = 1/D_ij^2 (weight).

    Args:
        nodes: List of positioned nodes
        ideal_distances: n x n matrix of ideal distances.
                        If None, computed from shortest paths using links.
        links: Links (required if ideal_distances not provided)
        edge_length: Base edge length for computing ideal distances

    Returns:
        Normalized stress value (0 = perfect, higher = worse)
    """
    n = len(nodes)
    if n < 2:
        return 0.0

    if ideal_distances is None:
        if links is None:
            raise ValueError("Must provide either ideal_distances or links")
        ideal_distances = _compute_ideal_distances(nodes, links, edge_length)

    numerator = 0.0
    denominator = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            ideal_d = ideal_distances[i][j]
            if ideal_d <= 0 or ideal_d == float("inf"):
                continue

            # Actual distance
            dx = nodes[i].x - nodes[j].x
            dy = nodes[i].y - nodes[j].y
            actual_d = math.sqrt(dx * dx + dy * dy)

            # Weight
            w_ij = 1.0 / (ideal_d * ideal_d)

            diff = actual_d - ideal_d
            numerator += w_ij * diff * diff
            denominator += w_ij * ideal_d * ideal_d

    if denominator == 0:
        return 0.0

    return numerator / denominator


def _compute_ideal_distances(
    nodes: Sequence[Node],
    links: Sequence[Link],
    edge_length: float,
) -> List[List[float]]:
    """Compute ideal distances based on shortest path lengths."""
    n = len(nodes)
    dist: List[List[float]] = [[float("inf")] * n for _ in range(n)]

    # Build adjacency
    adj: List[List[int]] = [[] for _ in range(n)]
    for link in links:
        s = _get_link_index(link.source)
        t = _get_link_index(link.target)
        if 0 <= s < n and 0 <= t < n:
            adj[s].append(t)
            adj[t].append(s)

    # BFS from each node
    for start in range(n):
        dist[start][start] = 0
        queue: deque[int] = deque([start])
        while queue:
            curr = queue.popleft()
            for neighbor in adj[curr]:
                if dist[start][neighbor] == float("inf"):
                    dist[start][neighbor] = dist[start][curr] + 1
                    queue.append(neighbor)

    # Convert hop count to ideal distance
    for i in range(n):
        for j in range(n):
            if dist[i][j] != float("inf"):
                dist[i][j] *= edge_length

    return dist


def edge_length_variance(nodes: Sequence[Node], links: Sequence[Link]) -> float:
    """
    Compute the variance of edge lengths.

    Lower variance indicates more uniform edge lengths.

    Args:
        nodes: List of positioned nodes
        links: List of links

    Returns:
        Variance of edge lengths
    """
    if len(links) == 0:
        return 0.0

    lengths = _get_edge_lengths(nodes, links)
    if not lengths:
        return 0.0

    mean = sum(lengths) / len(lengths)
    variance = sum((length - mean) ** 2 for length in lengths) / len(lengths)

    return variance


def _get_edge_lengths(nodes: Sequence[Node], links: Sequence[Link]) -> List[float]:
    """Get list of edge lengths."""
    n = len(nodes)
    lengths = []

    for link in links:
        s = _get_link_index(link.source)
        t = _get_link_index(link.target)

        if 0 <= s < n and 0 <= t < n:
            dx = nodes[s].x - nodes[t].x
            dy = nodes[s].y - nodes[t].y
            lengths.append(math.sqrt(dx * dx + dy * dy))

    return lengths


def edge_length_uniformity(nodes: Sequence[Node], links: Sequence[Link]) -> float:
    """
    Compute edge length uniformity (0-1, higher is better).

    Returns:
        1 - (std_dev / mean), clamped to [0, 1]
    """
    if len(links) == 0:
        return 1.0

    lengths = _get_edge_lengths(nodes, links)
    if not lengths:
        return 1.0

    mean = sum(lengths) / len(lengths)
    if mean == 0:
        return 0.0

    variance = sum((length - mean) ** 2 for length in lengths) / len(lengths)
    std_dev = math.sqrt(variance)

    return max(0.0, min(1.0, 1.0 - std_dev / mean))


def angular_resolution(nodes: Sequence[Node], links: Sequence[Link]) -> float:
    """
    Compute minimum angular resolution at any node.

    Angular resolution is the minimum angle between any two edges
    incident to the same node. Higher is better (max possible is
    360/degree for each node).

    Args:
        nodes: List of positioned nodes
        links: List of links

    Returns:
        Minimum angle (in degrees) between adjacent edges at any node.
        Returns 360.0 if no node has multiple edges.
    """
    n = len(nodes)
    if n == 0 or len(links) == 0:
        return 360.0

    # Build incident edges for each node
    incident: List[List[int]] = [[] for _ in range(n)]
    for link in links:
        s = _get_link_index(link.source)
        t = _get_link_index(link.target)
        if 0 <= s < n and 0 <= t < n:
            incident[s].append(t)
            incident[t].append(s)

    min_angle = 360.0

    for node_idx in range(n):
        neighbors = incident[node_idx]
        if len(neighbors) < 2:
            continue

        # Compute angles to each neighbor
        angles = []
        for neighbor_idx in neighbors:
            dx = nodes[neighbor_idx].x - nodes[node_idx].x
            dy = nodes[neighbor_idx].y - nodes[node_idx].y
            angle = math.atan2(dy, dx)
            angles.append(angle)

        # Sort angles
        angles.sort()

        # Find minimum angle between consecutive edges
        for i in range(len(angles)):
            next_i = (i + 1) % len(angles)
            diff = angles[next_i] - angles[i]
            if diff <= 0:
                diff += 2 * math.pi

            angle_deg = math.degrees(diff)
            min_angle = min(min_angle, angle_deg)

    return min_angle


def layout_quality_summary(
    nodes: Sequence[Node],
    links: Sequence[Link],
    edge_length: float = 100.0,
) -> dict[str, Any]:
    """
    Compute a summary of layout quality metrics.

    Args:
        nodes: List of positioned nodes
        links: List of links
        edge_length: Base edge length for stress calculation

    Returns:
        Dictionary with all metrics:
        - edge_crossings: Number of edge crossings
        - stress: Normalized stress value
        - edge_length_variance: Variance of edge lengths
        - edge_length_uniformity: Uniformity score (0-1)
        - angular_resolution: Minimum angle in degrees
    """
    return {
        "edge_crossings": edge_crossings(nodes, links),
        "stress": stress(nodes, links=links, edge_length=edge_length),
        "edge_length_variance": edge_length_variance(nodes, links),
        "edge_length_uniformity": edge_length_uniformity(nodes, links),
        "angular_resolution": angular_resolution(nodes, links),
    }


__all__ = [
    "edge_crossings",
    "stress",
    "edge_length_variance",
    "edge_length_uniformity",
    "angular_resolution",
    "layout_quality_summary",
]
