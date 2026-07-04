"""Shared bend-optimal realization for the Topology-Shape-Metrics layouts.

Both :class:`GIOTTOLayout` and :class:`KandinskyLayout` compute a bend-minimal
orthogonal *representation* (angles + bends, from min-cost flow) and then need to
turn it into an actual drawing: assign compass directions to segments, assign
integer coordinates (with rectangularization), and scale the grid onto the
canvas as node boxes and routed edges. This module holds that shared machinery
so the two layouts realize the representation identically.

Two entry points:

* :func:`bend_optimal_representation` -- from a planar embedding, produce the
  ``(representation, faces, expansion)`` triple, expanding vertices of degree
  > 4 into cages (see :mod:`.expansion`) when requested.
* :func:`realize_bend_optimal_drawing` -- from that triple, produce the
  ``(node_boxes, orthogonal_edges)`` drawing, or ``None`` if the representation
  is not a realizable shape (callers then fall back to a heuristic router).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

from .expansion import Expansion, cage_face_indices, expand_high_degree
from .metrics import (
    EAST,
    NORTH,
    SOUTH,
    WEST,
    compute_coordinates,
    compute_orthogonal_shape,
)
from .orthogonalization import (
    Face,
    OrthogonalRepresentation,
    build_flow_network,
    compute_faces,
    compute_orthogonal_representation,
    flow_to_orthogonal_rep,
    solve_min_cost_flow_simple,
)
from .types import NodeBox, OrthogonalEdge, Port, Side

if TYPE_CHECKING:
    from ..planarity._embedding import PlanarEmbedding


def _opp(direction: int) -> int:
    """Opposite compass direction (metrics quarter-turn encoding)."""
    return (direction + 2) % 4


def bend_optimal_representation(
    num_nodes: int,
    edges: list[tuple[int, int]],
    embedding: PlanarEmbedding,
    *,
    allow_expansion: bool = True,
) -> tuple[Optional[OrthogonalRepresentation], list[Face], Optional[Expansion]]:
    """Compute the bend-minimal representation from a planar embedding.

    Returns ``(representation, faces, expansion)``. When ``allow_expansion`` is
    True and some vertex has degree > 4, the graph is first expanded into cage
    cycles and the flow runs on the expanded graph (``expansion`` is non-None
    and ``faces`` describe the expanded graph). Returns ``(None, [], None)`` when
    the expanded structure is inconsistent or the flow is infeasible, so callers
    fall back to a heuristic router.

    ``edges`` may contain self-loops and multi-edges; the downstream stages
    sanitize them.
    """
    expansion = expand_high_degree(num_nodes, embedding) if allow_expansion else None
    if expansion is None:
        rep = compute_orthogonal_representation(num_nodes, edges, embedding=embedding)
        faces = compute_faces(num_nodes, edges, embedding=embedding)
        return rep, faces, None

    # Degree > 4: run the pipeline on the expanded graph, with the cage faces
    # forced rectangular and the cage cycle edges kept straight (box sides).
    faces = compute_faces(expansion.num_nodes, expansion.edges, embedding=expansion.embedding)
    cage_faces = cage_face_indices(expansion, faces)
    if cage_faces is None:
        return None, [], None
    network = build_flow_network(
        expansion.num_nodes,
        expansion.edges,
        faces,
        cage_faces=cage_faces,
        rigid_edges=expansion.cage_edges,
    )
    if not solve_min_cost_flow_simple(network):
        return None, [], None
    rep = flow_to_orthogonal_rep(network, expansion.edges)
    return rep, faces, expansion


def _port_fraction(box: NodeBox, side: Side, point: tuple[float, float]) -> float:
    """Fraction along ``side`` of ``box`` where the port at ``point`` sits."""
    if side in (Side.NORTH, Side.SOUTH):
        span = box.width
        frac = (point[0] - box.left) / span if span > 0 else 0.5
    else:
        span = box.height
        frac = (point[1] - box.top) / span if span > 0 else 0.5
    return min(1.0, max(0.0, frac))


def realize_bend_optimal_drawing(
    *,
    num_nodes: int,
    link_endpoints: Sequence[tuple[int, int]],
    node_sizes: Sequence[tuple[float, float]],
    orthogonal_rep: Optional[OrthogonalRepresentation],
    faces: list[Face],
    expansion: Optional[Expansion],
    canvas_size: tuple[float, float],
    cell: float,
) -> Optional[tuple[list[NodeBox], list[OrthogonalEdge]]]:
    """Draw the representation via Topology-Shape-Metrics.

    Computes the shape (compass direction of every segment) and integer
    coordinates, then scales them onto the canvas. Returns
    ``(node_boxes, orthogonal_edges)``: node boxes are centered on grid points
    (expanded degree > 4 vertices become boxes spanning their cage rectangle),
    and each edge carries ports on the box sides with bends from the route.
    Returns ``None`` when the representation is not a realizable shape, so the
    caller can fall back without the drawing regressing.

    Args:
        num_nodes: Number of original graph vertices.
        link_endpoints: ``(src, tgt)`` per link, indexed by edge id (the port's
            ``edge`` field). Self-loops and out-of-range endpoints are skipped.
        node_sizes: ``(width, height)`` per original vertex.
        orthogonal_rep: The bend-minimal representation.
        faces: Faces of the (possibly expanded) embedding the rep describes.
        expansion: Cage expansion, or None when no vertex exceeds degree 4.
        canvas_size: Target ``(width, height)`` to center the drawing in.
        cell: Grid spacing in canvas units (one unit of the integer drawing).
    """
    n = num_nodes
    if not orthogonal_rep or not faces or n == 0:
        return None

    edges = [(s, t) for (s, t) in link_endpoints if 0 <= s < n and 0 <= t < n and s != t]

    shape = compute_orthogonal_shape(faces, orthogonal_rep)
    if not shape.valid:
        return None
    coord_edges = expansion.edges if expansion is not None else edges
    drawing = compute_coordinates(shape, coord_edges, faces=faces)
    if not drawing.valid:
        return None

    # Every drawn vertex needs a position: all originals (minus the ones replaced
    # by cages) plus every cage vertex.
    needed: set[int] = set(range(n))
    if expansion is not None:
        needed -= set(expansion.cages.keys())
        for ids in expansion.cages.values():
            needed.update(ids)
    if not needed.issubset(drawing.vertex_positions.keys()):
        return None

    # --- Scale the integer grid onto the canvas ---------------------------
    gxs = [p[0] for p in drawing.vertex_positions.values()]
    gys = [p[1] for p in drawing.vertex_positions.values()]
    for route in drawing.edge_routes.values():
        gxs += [p[0] for p in route]
        gys += [p[1] for p in route]
    min_x, max_x = min(gxs), max(gxs)
    min_y, max_y = min(gys), max(gys)

    span_x = (max_x - min_x) * cell
    span_y = (max_y - min_y) * cell
    canvas_w, canvas_h = canvas_size
    off_x = (canvas_w - span_x) / 2.0
    off_y = (canvas_h - span_y) / 2.0

    def to_canvas(gx: float, gy: float) -> tuple[float, float]:
        # Flip y so grid-North (+y) is up on screen (smaller screen y).
        return (off_x + (gx - min_x) * cell, off_y + (max_y - gy) * cell)

    dir_to_side = {EAST: Side.EAST, WEST: Side.WEST, NORTH: Side.NORTH, SOUTH: Side.SOUTH}

    # Canonical dart per undirected edge, mirroring compute_coordinates: the
    # stored route runs along the orientation the edge has in coord_edges, which
    # need not match the link orientation.
    canonical_dart: dict[tuple[int, int], tuple[int, int]] = {}
    for u, v in coord_edges:
        ckey = (min(u, v), max(u, v))
        if (u, v) in shape.edge_shapes:
            canonical_dart[ckey] = (u, v)
        elif (v, u) in shape.edge_shapes:
            canonical_dart[ckey] = (v, u)

    # --- Rebuild node boxes ----------------------------------------------
    node_boxes: list[NodeBox] = []
    for i in range(n):
        width, height = node_sizes[i]
        width = float(width)
        height = float(height)
        if expansion is not None and i in expansion.cages:
            pts = [drawing.vertex_positions[c] for c in expansion.cages[i]]
            gx_min, gx_max = min(p[0] for p in pts), max(p[0] for p in pts)
            gy_min, gy_max = min(p[1] for p in pts), max(p[1] for p in pts)
            cx, cy = to_canvas((gx_min + gx_max) / 2.0, (gy_min + gy_max) / 2.0)
            width += (gx_max - gx_min) * cell
            height += (gy_max - gy_min) * cell
        else:
            gx, gy = drawing.vertex_positions[i]
            cx, cy = to_canvas(gx, gy)
        node_boxes.append(NodeBox(index=i, x=cx, y=cy, width=width, height=height))

    # --- Build orthogonal edges from the routes --------------------------
    # Cage cycle edges are not drawn (they are the box boundary).
    orthogonal_edges: list[OrthogonalEdge] = []
    for edge_idx, (src, tgt) in enumerate(link_endpoints):
        if not (0 <= src < n and 0 <= tgt < n) or src == tgt:
            continue
        # Endpoints in the drawn (possibly expanded) graph.
        if expansion is not None:
            mapped = expansion.dart_map.get((src, tgt))
            if mapped is None:
                continue
            dsrc, dtgt = mapped
        else:
            dsrc, dtgt = src, tgt
        key = (min(dsrc, dtgt), max(dsrc, dtgt))
        edge_route = drawing.edge_routes.get(key)
        dart = canonical_dart.get(key)
        es = shape.edge_shapes.get(dart) if dart is not None else None
        if edge_route is None or es is None or dart is None:
            continue

        # The canonical route runs dart.tail -> dart.head; orient it so it goes
        # src -> tgt for this link.
        pts = list(edge_route)
        if dart[0] != dsrc:
            pts = pts[::-1]
        src_side = dir_to_side[es.start_direction if dart[0] == dsrc else _opp(es.end_direction)]
        tgt_side = dir_to_side[_opp(es.end_direction) if dart[1] == dtgt else es.start_direction]

        src_frac = _port_fraction(node_boxes[src], src_side, to_canvas(*pts[0]))
        tgt_frac = _port_fraction(node_boxes[tgt], tgt_side, to_canvas(*pts[-1]))

        interior = [to_canvas(gx, gy) for gx, gy in pts[1:-1]]
        orthogonal_edges.append(
            OrthogonalEdge(
                source=src,
                target=tgt,
                source_port=Port(node=src, side=src_side, position=src_frac, edge=edge_idx),
                target_port=Port(node=tgt, side=tgt_side, position=tgt_frac, edge=edge_idx),
                bends=interior,
            )
        )
    return node_boxes, orthogonal_edges


__all__ = [
    "bend_optimal_representation",
    "realize_bend_optimal_drawing",
]
