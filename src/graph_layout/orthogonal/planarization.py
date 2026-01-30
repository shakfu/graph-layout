"""
Planarization utilities for orthogonal layout.

Provides planarity testing and crossing vertex insertion for
handling non-planar graphs in orthogonal layouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, cast

# Try to import Cython-optimized functions
try:
    from .._speedups import _find_edge_crossings, _segments_intersect

    _USE_CYTHON = True
except ImportError:
    _USE_CYTHON = False


@dataclass
class CrossingVertex:
    """
    A dummy vertex inserted at an edge crossing.

    When two edges cross in a non-planar graph, we insert a dummy
    vertex at the crossing point and split both edges.
    """

    index: int  # Index in the augmented graph
    x: float  # Crossing x coordinate
    y: float  # Crossing y coordinate
    edge1: tuple[int, int]  # First original edge (source, target)
    edge2: tuple[int, int]  # Second original edge (source, target)


@dataclass
class PlanarizedGraph:
    """
    Result of planarizing a graph.

    Contains the augmented graph with crossing vertices and
    mappings to recover the original structure.
    """

    num_original_nodes: int
    num_total_nodes: int  # Including crossing vertices

    # Augmented edges (may reference crossing vertices)
    edges: list[tuple[int, int]]

    # Crossing vertices
    crossings: list[CrossingVertex]

    # Map from augmented edge index to original edge index
    edge_to_original: dict[int, int]

    # Map from original edge to list of augmented edges (path through crossings)
    original_to_edges: dict[int, list[int]]

    def is_crossing_vertex(self, node: int) -> bool:
        """Check if a node is a crossing vertex."""
        return node >= self.num_original_nodes


def _segments_intersect_py(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> Optional[tuple[float, float]]:
    """
    Pure Python implementation of segment intersection.

    Check if two line segments intersect and return intersection point.
    Segments are (p1, p2) and (p3, p4).

    Returns:
        Intersection point (x, y) if segments intersect, None otherwise
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    # Calculate denominators
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(denom) < 1e-10:
        # Lines are parallel
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    # Check if intersection is within both segments (exclusive of endpoints)
    eps = 1e-10
    if eps < t < 1 - eps and eps < u < 1 - eps:
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return (ix, iy)

    return None


def segments_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> Optional[tuple[float, float]]:
    """
    Check if two line segments intersect and return intersection point.

    Uses Cython-optimized implementation when available.
    Segments are (p1, p2) and (p3, p4).

    Returns:
        Intersection point (x, y) if segments intersect, None otherwise
    """
    if _USE_CYTHON:
        result = _segments_intersect(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p4[0], p4[1])
        return cast(Optional[tuple[float, float]], result)
    return _segments_intersect_py(p1, p2, p3, p4)


def _find_edge_crossings_py(
    positions: list[tuple[float, float]],
    edges: list[tuple[int, int]],
) -> list[tuple[int, int, float, float]]:
    """
    Pure Python implementation of edge crossing detection.

    Find all edge crossings in a graph with given positions.

    Args:
        positions: List of (x, y) positions for each node
        edges: List of (source, target) edges

    Returns:
        List of (edge1_idx, edge2_idx, cross_x, cross_y) for each crossing
    """
    crossings = []

    for i, (s1, t1) in enumerate(edges):
        for j, (s2, t2) in enumerate(edges):
            if j <= i:
                continue

            # Skip edges that share a vertex
            if s1 == s2 or s1 == t2 or t1 == s2 or t1 == t2:
                continue

            # Check for intersection
            p1 = positions[s1]
            p2 = positions[t1]
            p3 = positions[s2]
            p4 = positions[t2]

            intersection = _segments_intersect_py(p1, p2, p3, p4)
            if intersection:
                crossings.append((i, j, intersection[0], intersection[1]))

    return crossings


def find_edge_crossings(
    positions: list[tuple[float, float]],
    edges: list[tuple[int, int]],
) -> list[tuple[int, int, float, float]]:
    """
    Find all edge crossings in a graph with given positions.

    Uses Cython-optimized implementation when available.

    Args:
        positions: List of (x, y) positions for each node
        edges: List of (source, target) edges

    Returns:
        List of (edge1_idx, edge2_idx, cross_x, cross_y) for each crossing
    """
    if _USE_CYTHON:
        result = _find_edge_crossings(positions, edges)
        return cast(list[tuple[int, int, float, float]], result)
    return _find_edge_crossings_py(positions, edges)


def planarize_graph(
    num_nodes: int,
    edges: list[tuple[int, int]],
    positions: Optional[list[tuple[float, float]]] = None,
) -> PlanarizedGraph:
    """
    Planarize a graph by inserting crossing vertices.

    If positions are not provided, uses a simple initial layout to
    detect crossings. For best results, provide approximate positions.

    Args:
        num_nodes: Number of nodes in original graph
        edges: List of (source, target) edges
        positions: Optional node positions for crossing detection

    Returns:
        PlanarizedGraph with crossing vertices inserted
    """
    if not edges:
        return PlanarizedGraph(
            num_original_nodes=num_nodes,
            num_total_nodes=num_nodes,
            edges=[],
            crossings=[],
            edge_to_original={},
            original_to_edges={i: [] for i in range(len(edges))},
        )

    # If no positions provided, create simple grid layout for crossing detection
    if positions is None:
        # Simple circular layout for crossing detection
        import math

        positions = []
        for i in range(num_nodes):
            angle = 2 * math.pi * i / num_nodes
            positions.append((math.cos(angle), math.sin(angle)))

    # Find all crossings
    raw_crossings = find_edge_crossings(positions, edges)

    if not raw_crossings:
        # Graph is already planar (with given positions)
        return PlanarizedGraph(
            num_original_nodes=num_nodes,
            num_total_nodes=num_nodes,
            edges=list(edges),
            crossings=[],
            edge_to_original={i: i for i in range(len(edges))},
            original_to_edges={i: [i] for i in range(len(edges))},
        )

    # Group crossings by edge
    edge_crossings: dict[int, list[tuple[int, float, float, float]]] = {
        i: [] for i in range(len(edges))
    }

    for edge1_idx, edge2_idx, cx, cy in raw_crossings:
        s1, t1 = edges[edge1_idx]
        s2, t2 = edges[edge2_idx]

        # Calculate parameter t along each edge for sorting
        p1 = positions[s1]
        p2 = positions[t1]
        p3 = positions[s2]
        p4 = positions[t2]

        # t parameter for edge1
        dx1 = p2[0] - p1[0]
        dy1 = p2[1] - p1[1]
        if abs(dx1) > abs(dy1):
            t1_param = (cx - p1[0]) / dx1 if abs(dx1) > 1e-10 else 0.5
        else:
            t1_param = (cy - p1[1]) / dy1 if abs(dy1) > 1e-10 else 0.5

        # t parameter for edge2
        dx2 = p4[0] - p3[0]
        dy2 = p4[1] - p3[1]
        if abs(dx2) > abs(dy2):
            t2_param = (cx - p3[0]) / dx2 if abs(dx2) > 1e-10 else 0.5
        else:
            t2_param = (cy - p3[1]) / dy2 if abs(dy2) > 1e-10 else 0.5

        edge_crossings[edge1_idx].append((edge2_idx, t1_param, cx, cy))
        edge_crossings[edge2_idx].append((edge1_idx, t2_param, cx, cy))

    # Sort crossings along each edge by t parameter
    for edge_idx in edge_crossings:
        edge_crossings[edge_idx].sort(key=lambda x: x[1])

    # Create crossing vertices
    crossing_vertices: list[CrossingVertex] = []
    crossing_map: dict[tuple[int, int], int] = {}  # (edge1, edge2) -> crossing vertex index

    next_vertex_idx = num_nodes

    for edge1_idx, edge2_idx, cx, cy in raw_crossings:
        key = (min(edge1_idx, edge2_idx), max(edge1_idx, edge2_idx))
        if key not in crossing_map:
            cv = CrossingVertex(
                index=next_vertex_idx,
                x=cx,
                y=cy,
                edge1=edges[edge1_idx],
                edge2=edges[edge2_idx],
            )
            crossing_vertices.append(cv)
            crossing_map[key] = next_vertex_idx
            next_vertex_idx += 1

    # Build augmented edge list
    augmented_edges: list[tuple[int, int]] = []
    edge_to_original: dict[int, int] = {}
    original_to_edges: dict[int, list[int]] = {i: [] for i in range(len(edges))}

    for edge_idx, (src, tgt) in enumerate(edges):
        crossings_on_edge = edge_crossings[edge_idx]

        if not crossings_on_edge:
            # No crossings, keep original edge
            aug_idx = len(augmented_edges)
            augmented_edges.append((src, tgt))
            edge_to_original[aug_idx] = edge_idx
            original_to_edges[edge_idx].append(aug_idx)
        else:
            # Split edge at crossings
            current_node = src
            for other_edge, t_param, cx, cy in crossings_on_edge:
                key = (min(edge_idx, other_edge), max(edge_idx, other_edge))
                crossing_node = crossing_map[key]

                # Add edge segment
                aug_idx = len(augmented_edges)
                augmented_edges.append((current_node, crossing_node))
                edge_to_original[aug_idx] = edge_idx
                original_to_edges[edge_idx].append(aug_idx)

                current_node = crossing_node

            # Add final segment to target
            aug_idx = len(augmented_edges)
            augmented_edges.append((current_node, tgt))
            edge_to_original[aug_idx] = edge_idx
            original_to_edges[edge_idx].append(aug_idx)

    return PlanarizedGraph(
        num_original_nodes=num_nodes,
        num_total_nodes=next_vertex_idx,
        edges=augmented_edges,
        crossings=crossing_vertices,
        edge_to_original=edge_to_original,
        original_to_edges=original_to_edges,
    )


def is_planar_quick_check(num_nodes: int, num_edges: int) -> bool:
    """
    Quick check if graph could possibly be planar using Euler's formula.

    For a planar graph: E <= 3V - 6 (for V >= 3)

    Returns:
        True if graph might be planar, False if definitely not planar
    """
    if num_nodes < 3:
        return True
    return num_edges <= 3 * num_nodes - 6


__all__ = [
    "CrossingVertex",
    "PlanarizedGraph",
    "segments_intersect",
    "find_edge_crossings",
    "planarize_graph",
    "is_planar_quick_check",
]
