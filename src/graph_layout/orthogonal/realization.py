"""Shared bend-optimal realization for the Topology-Shape-Metrics layouts.

Both :class:`GIOTTOLayout` and :class:`KandinskyLayout` compute a bend-minimal
orthogonal *representation* (angles + bends, from min-cost flow) and then need to
turn it into an actual drawing: assign compass directions to segments, assign
integer coordinates (with rectangularization), and scale the grid onto the
canvas as node boxes and routed edges. This module holds that shared machinery
so the two layouts realize the representation identically.

Entry points:

* :func:`bend_optimal_representation` -- from a planar embedding, produce the
  ``(representation, faces, expansion)`` triple, expanding vertices of degree
  > 4 into cages (see :mod:`.expansion`) when requested.
* :func:`realize_bend_optimal_drawing` -- from that triple, produce the
  ``(node_boxes, orthogonal_edges)`` drawing, or ``None`` if the representation
  is not a realizable shape (callers then fall back to a heuristic router).
* :func:`realize_planarized_drawing` -- for non-planar input: realize the
  *planarized* graph (crossings replaced by degree-4 dummy vertices) and
  reassemble each original edge's polyline through its crossing points, which
  become the edges' bends. Returns ``None`` outside its scope (original degree
  > 4, or the crossing gadget not straight-through) so callers fall back.
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
    from ..planarity.embedders import PlanarEmbedder
    from .planarization import PlanarizedGraph

_DIR_TO_SIDE = {EAST: Side.EAST, WEST: Side.WEST, NORTH: Side.NORTH, SOUTH: Side.SOUTH}


def _opp(direction: int) -> int:
    """Opposite compass direction (metrics quarter-turn encoding)."""
    return (direction + 2) % 4


def _grid_direction(p: tuple[int, int], q: tuple[int, int]) -> Optional[int]:
    """Compass direction (metrics encoding) of the axis-aligned step p -> q."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    if dx > 0 and dy == 0:
        return EAST
    if dx < 0 and dy == 0:
        return WEST
    if dy > 0 and dx == 0:
        return NORTH
    if dy < 0 and dx == 0:
        return SOUTH
    return None  # diagonal or zero-length (should not occur in a valid drawing)


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


def _dummy_rotations_alternate(planarized: PlanarizedGraph, embedding: PlanarEmbedding) -> bool:
    """Whether every crossing dummy alternates its two original edges.

    A degree-4 dummy draws its two original edges straight through (a clean
    crossing) only when the two edges alternate in its rotation -- ``e1 e2 e1
    e2`` -- so the halves of each edge leave in opposite directions. (With
    the flow model forcing all four corners of a degree-4 vertex to 90
    degrees, alternation is the only remaining condition.) The embedder
    produces alternating rotations in practice; this guards the rare case
    where it does not, so the drawing never shows a bent pseudo-crossing.
    """
    aug = planarized.edges
    e2o = planarized.edge_to_original
    for cv in planarized.crossings:
        d = cv.index
        rot = embedding.rotation.get(d, [])
        if len(rot) != 4:
            return False
        neighbor_edge: dict[int, int] = {}
        for ai, (a, b) in enumerate(aug):
            if a == d:
                neighbor_edge.setdefault(b, e2o[ai])
            elif b == d:
                neighbor_edge.setdefault(a, e2o[ai])
        groups = [neighbor_edge.get(w) for w in rot]
        if not (groups[0] == groups[2] and groups[1] == groups[3] and groups[0] != groups[1]):
            return False
    return True


def realize_planarized_drawing(
    *,
    planarized: PlanarizedGraph,
    link_endpoints: Sequence[tuple[int, int]],
    node_sizes: Sequence[tuple[float, float]],
    embedder: PlanarEmbedder,
    canvas_size: tuple[float, float],
    cell: float,
) -> Optional[tuple[list[NodeBox], list[OrthogonalEdge]]]:
    """Draw a non-planar graph bend-optimally via its planarization.

    The planarized graph (crossings replaced by degree-4 dummy vertices) is
    planar, so the Topology-Shape-Metrics pipeline realizes it directly. Each
    original edge's polyline is then reassembled by walking its augmented
    segments through the crossing dummies; the dummy grid points become the
    edge's bend points, so the two edges of every crossing pass through a shared
    point (a clean orthogonal crossing). Node boxes are built for the original
    vertices only -- dummies are drawn implicitly as the crossing points.

    Returns ``None`` (caller falls back to the heuristic router) when the input
    is out of scope: an original vertex of degree > 4 (would need cage
    expansion, not combined with crossings here), a non-alternating crossing
    gadget, or an unrealizable shape.

    Args:
        planarized: The planarized graph (from :func:`planarize_graph`).
        link_endpoints: ``(src, tgt)`` per link, indexed by edge id.
        node_sizes: ``(width, height)`` per original vertex.
        embedder: Planar embedding strategy for the augmented graph.
        canvas_size: Target ``(width, height)`` to center the drawing in.
        cell: Grid spacing in canvas units.
    """
    from ..planarity import check_planarity

    n = planarized.num_original_nodes
    total = planarized.num_total_nodes
    aug = planarized.edges
    if n == 0 or not planarized.crossings:
        return None

    # Combining crossings with degree > 4 cage expansion is out of scope; the
    # crossing path requires every original vertex to be degree <= 4.
    degree = [0] * n
    for src, tgt in link_endpoints:
        if 0 <= src < n and 0 <= tgt < n and src != tgt:
            degree[src] += 1
            degree[tgt] += 1
    if any(d > 4 for d in degree):
        return None

    result = check_planarity(total, list(aug))
    if not result.is_planar:
        return None
    embedding = embedder.embed(total, list(aug), planarity_result=result)
    if not _dummy_rotations_alternate(planarized, embedding):
        return None

    rep, faces, _expansion = bend_optimal_representation(
        total, list(aug), embedding, allow_expansion=False
    )
    if rep is None or not faces:
        return None
    shape = compute_orthogonal_shape(faces, rep)
    if not shape.valid:
        return None
    drawing = compute_coordinates(shape, list(aug), faces=faces)
    if not drawing.valid:
        return None
    if set(range(total)) - set(drawing.vertex_positions):
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
        return (off_x + (gx - min_x) * cell, off_y + (max_y - gy) * cell)

    # Canonical dart per augmented edge (route is stored tail -> head).
    canonical_dart: dict[tuple[int, int], tuple[int, int]] = {}
    for u, v in aug:
        ckey = (min(u, v), max(u, v))
        if (u, v) in shape.edge_shapes:
            canonical_dart[ckey] = (u, v)
        elif (v, u) in shape.edge_shapes:
            canonical_dart[ckey] = (v, u)

    # Node boxes for the original vertices only.
    node_boxes: list[NodeBox] = []
    for i in range(n):
        width, height = node_sizes[i]
        cx, cy = to_canvas(*drawing.vertex_positions[i])
        node_boxes.append(NodeBox(index=i, x=cx, y=cy, width=float(width), height=float(height)))

    dummy_grid = {tuple(drawing.vertex_positions[cv.index]) for cv in planarized.crossings}

    # Reassemble each original edge's polyline through its augmented segments.
    orthogonal_edges: list[OrthogonalEdge] = []
    for edge_idx, (src, tgt) in enumerate(link_endpoints):
        if not (0 <= src < n and 0 <= tgt < n) or src == tgt:
            continue
        path = planarized.original_to_edges.get(edge_idx, [])
        if not path:
            continue

        grid: list[tuple[int, int]] = [drawing.vertex_positions[src]]
        cur = src
        ok = True
        for ai in path:
            a, b = aug[ai]
            key = (min(a, b), max(a, b))
            seg_route: Optional[list[tuple[int, int]]] = drawing.edge_routes.get(key)
            dart: Optional[tuple[int, int]] = canonical_dart.get(key)
            if seg_route is None or dart is None:
                ok = False
                break
            seg = list(seg_route) if dart[0] == cur else list(reversed(seg_route))
            cur = dart[1] if dart[0] == cur else dart[0]
            grid.extend(seg[1:])
        if not ok or len(grid) < 2 or cur != tgt:
            return None

        src_side = _grid_direction(grid[0], grid[1])
        tgt_side = _grid_direction(grid[-1], grid[-2])
        if src_side is None or tgt_side is None:
            return None
        # Each crossing dummy must be a straight-through point (in == out).
        for i in range(1, len(grid) - 1):
            if tuple(grid[i]) in dummy_grid:
                if _grid_direction(grid[i - 1], grid[i]) != _grid_direction(grid[i], grid[i + 1]):
                    return None

        bends = [to_canvas(gx, gy) for gx, gy in grid[1:-1]]
        src_port = Port(node=src, side=_DIR_TO_SIDE[src_side], position=0.5, edge=edge_idx)
        tgt_port = Port(node=tgt, side=_DIR_TO_SIDE[tgt_side], position=0.5, edge=edge_idx)
        orthogonal_edges.append(
            OrthogonalEdge(
                source=src,
                target=tgt,
                source_port=src_port,
                target_port=tgt_port,
                bends=bends,
            )
        )
    return node_boxes, orthogonal_edges


__all__ = [
    "bend_optimal_representation",
    "realize_bend_optimal_drawing",
    "realize_planarized_drawing",
]
