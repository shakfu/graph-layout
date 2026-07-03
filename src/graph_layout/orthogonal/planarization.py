"""
Planarization utilities for orthogonal layout.

Provides planarity testing and crossing vertex insertion for
handling non-planar graphs in orthogonal layouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, cast

from ..planarity import PlanarEmbedding, check_planarity

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


def _insertion_crossings(
    total_nodes: int,
    segments: list[list[int]],
    u: int,
    v: int,
) -> Optional[list[int]]:
    """Find the segments crossed when inserting edge (u, v) into a planar graph.

    ``segments`` is the current (planar) graph as ``[a, b, orig]`` triples. The
    edge is routed through the embedding's faces, crossing the fewest existing
    segments (a shortest path in the dual graph from a face incident to ``u`` to
    a face incident to ``v``).

    Returns the crossed segment indices in order from ``u`` to ``v`` (empty if
    the edge can be added without any crossing), or ``None`` if no face path
    exists (e.g. the endpoints are in different connected components -- such an
    edge can be drawn around everything, so it needs no crossing either).
    """
    from collections import deque

    edge_list = [(a, b) for a, b, _ in segments]
    result = check_planarity(total_nodes, edge_list)
    if not result.is_planar or result.embedding is None:
        return None

    faces = PlanarEmbedding(result.embedding).faces()
    if not faces:
        return None

    face_of: dict[tuple[int, int], int] = {}
    faces_at: dict[int, set[int]] = {}
    for fi, face in enumerate(faces):
        for a, b in face:
            face_of[(a, b)] = fi
            faces_at.setdefault(a, set()).add(fi)
            faces_at.setdefault(b, set()).add(fi)

    start = faces_at.get(u, set())
    goal = faces_at.get(v, set())
    if not start or not goal:
        return None
    if start & goal:
        return []  # u and v already share a face: no crossing needed

    seg_index: dict[frozenset[int], int] = {}
    for si, (a, b, _) in enumerate(segments):
        seg_index[frozenset((a, b))] = si

    # Multi-source BFS over faces; crossing a shared segment costs one step.
    prev_face: dict[int, Optional[int]] = {f: None for f in start}
    prev_seg: dict[int, frozenset[int]] = {}
    visited = set(start)
    queue = deque(start)
    found: Optional[int] = None
    while queue and found is None:
        f = queue.popleft()
        for a, b in faces[f]:
            opp = face_of.get((b, a))
            if opp is None or opp in visited:
                continue
            visited.add(opp)
            prev_face[opp] = f
            prev_seg[opp] = frozenset((a, b))
            if opp in goal:
                found = opp
                break
            queue.append(opp)

    if found is None:
        return None

    crossed: list[int] = []
    cur: Optional[int] = found
    while cur is not None and prev_face.get(cur) is not None:
        crossed.append(seg_index[prev_seg[cur]])
        cur = prev_face[cur]
    crossed.reverse()
    return crossed


def planarize_graph(
    num_nodes: int,
    edges: list[tuple[int, int]],
    positions: Optional[list[tuple[float, float]]] = None,
) -> PlanarizedGraph:
    """
    Planarize a graph by inserting crossing (dummy) vertices.

    Topological planarization (Topology-Shape-Metrics): a maximal planar subgraph
    is embedded, then the remaining edges are reinserted one at a time along a
    minimum-crossing path through the current embedding's faces, each crossing
    becoming a degree-four dummy vertex. The result is a planar graph, and the
    crossings depend only on the graph's topology -- not on any drawing. A
    genuinely planar graph therefore gains no crossings regardless of how its
    nodes happen to be positioned.

    Args:
        num_nodes: Number of nodes in original graph
        edges: List of (source, target) edges
        positions: Optional node positions, used only to give the dummy vertices
            an approximate coordinate (never to decide whether a crossing exists)

    Returns:
        PlanarizedGraph with crossing vertices inserted
    """
    n = num_nodes

    # Self-loops never cross anything; carry them through untouched.
    loops = [(i, u, v) for i, (u, v) in enumerate(edges) if u == v]
    proper = [(i, u, v) for i, (u, v) in enumerate(edges) if u != v]

    # 1. Greedy maximal planar subgraph; edges that would break planarity are
    #    deferred for reinsertion.
    accepted: list[tuple[int, int]] = []
    subgraph: list[tuple[int, int, int]] = []  # (orig, u, v)
    deferred: list[tuple[int, int, int]] = []
    for orig, u, v in proper:
        if check_planarity(n, accepted + [(u, v)]).is_planar:
            accepted.append((u, v))
            subgraph.append((orig, u, v))
        else:
            deferred.append((orig, u, v))

    # 2. Incremental edge reinsertion. ``segments`` holds the current planar
    #    graph as [a, b, orig] triples; dummy vertices are appended as needed.
    segments: list[list[int]] = [[u, v, orig] for orig, u, v in subgraph]
    next_idx = n
    crossings: list[CrossingVertex] = []

    for orig_d, u, v in deferred:
        crossed = _insertion_crossings(next_idx, segments, u, v)
        if not crossed:
            segments.append([u, v, orig_d])
            continue
        prev = u
        for seg_ref in crossed:
            a, b, orig_s = segments[seg_ref]
            dummy = next_idx
            next_idx += 1
            # Split the crossed segment at the dummy and route the new edge
            # through it.
            segments[seg_ref] = [a, dummy, orig_s]
            segments.append([dummy, b, orig_s])
            segments.append([prev, dummy, orig_d])
            crossings.append(
                CrossingVertex(
                    index=dummy,
                    x=0.0,
                    y=0.0,
                    edge1=edges[orig_d],
                    edge2=edges[orig_s],
                )
            )
            prev = dummy
        segments.append([prev, v, orig_d])

    # 3. Assemble the augmented edge list and the original-edge -> segments maps.
    augmented_edges: list[tuple[int, int]] = []
    edge_to_original: dict[int, int] = {}
    by_orig: dict[int, list[int]] = {}
    for a, b, orig in segments:
        aug_idx = len(augmented_edges)
        augmented_edges.append((a, b))
        edge_to_original[aug_idx] = orig
        by_orig.setdefault(orig, []).append(aug_idx)
    for orig, _u, _v in loops:
        aug_idx = len(augmented_edges)
        augmented_edges.append(edges[orig])
        edge_to_original[aug_idx] = orig
        by_orig.setdefault(orig, []).append(aug_idx)

    original_to_edges: dict[int, list[int]] = {i: [] for i in range(len(edges))}
    for orig, seg_idxs in by_orig.items():
        original_to_edges[orig] = _order_segment_path(edges[orig], seg_idxs, augmented_edges)

    # Give dummies an approximate coordinate from the crossing edges' endpoints.
    if positions is not None:
        for cv in crossings:
            pts = [positions[p] for p in (*cv.edge1, *cv.edge2) if 0 <= p < len(positions)]
            if pts:
                cv.x = sum(p[0] for p in pts) / len(pts)
                cv.y = sum(p[1] for p in pts) / len(pts)

    return PlanarizedGraph(
        num_original_nodes=num_nodes,
        num_total_nodes=next_idx,
        edges=augmented_edges,
        crossings=crossings,
        edge_to_original=edge_to_original,
        original_to_edges=original_to_edges,
    )


def _order_segment_path(
    original_edge: tuple[int, int],
    seg_idxs: list[int],
    augmented_edges: list[tuple[int, int]],
) -> list[int]:
    """Order an original edge's segments into a path from its source to target."""
    if len(seg_idxs) <= 1:
        return list(seg_idxs)
    src, tgt = original_edge
    adj: dict[int, list[tuple[int, int]]] = {}
    for ai in seg_idxs:
        a, b = augmented_edges[ai]
        adj.setdefault(a, []).append((b, ai))
        adj.setdefault(b, []).append((a, ai))
    path: list[int] = []
    used: set[int] = set()
    cur = src
    while cur != tgt:
        nxt = next(((w, ai) for w, ai in adj.get(cur, []) if ai not in used), None)
        if nxt is None:
            break
        path.append(nxt[1])
        used.add(nxt[1])
        cur = nxt[0]
    return path


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
