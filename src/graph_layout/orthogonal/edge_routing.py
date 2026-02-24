"""Constraint-aware edge routing for orthogonal layouts.

Provides global edge routing with:
- Even port distribution along node sides
- Self-loop routing with 4 bends
- Parallel edge separation
- Basic obstacle-aware segment detouring

Shared by both KandinskyLayout and GIOTTOLayout.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Optional

from .types import NodeBox, OrthogonalEdge, Port, Side


def _opposite_sides() -> dict[Side, Side]:
    return {
        Side.NORTH: Side.SOUTH,
        Side.SOUTH: Side.NORTH,
        Side.EAST: Side.WEST,
        Side.WEST: Side.EAST,
    }


_ADJACENT_SIDE: dict[Side, Side] = {
    Side.NORTH: Side.EAST,
    Side.EAST: Side.SOUTH,
    Side.SOUTH: Side.WEST,
    Side.WEST: Side.NORTH,
}


def determine_port_sides(
    src_box: NodeBox,
    tgt_box: NodeBox,
) -> tuple[Side, Side]:
    """Determine best port sides based on relative node position.

    Uses the same heuristic as the original Kandinsky/GIOTTO layouts:
    prefer vertical connections for hierarchical layouts.
    """
    dx = tgt_box.x - src_box.x
    dy = tgt_box.y - src_box.y

    if abs(dy) > abs(dx) * 0.5:
        if dy > 0:
            return (Side.SOUTH, Side.NORTH)
        else:
            return (Side.NORTH, Side.SOUTH)
    else:
        if dx > 0:
            return (Side.EAST, Side.WEST)
        else:
            return (Side.WEST, Side.EAST)


def assign_ports(
    boxes: list[NodeBox],
    edges: list[tuple[int, int]],
    edge_sides: list[tuple[Side, Side]],
) -> list[tuple[Port, Port]]:
    """Assign ports for all edges with proper distribution.

    For each node, distributes ports evenly along each side.
    Multiple edges on the same side get offset positions.

    Args:
        boxes: Node boxes for all nodes.
        edges: Edge list as (source, target) pairs.
        edge_sides: Pre-determined sides for each edge as (src_side, tgt_side).

    Returns:
        List of (source_port, target_port) for each edge.
    """
    # Group edges by (node, side) to compute port offsets
    # node_side_edges[node][side] -> list of (edge_idx, is_source)
    node_side_edges: dict[int, dict[Side, list[tuple[int, bool]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for ei, ((src, tgt), (src_side, tgt_side)) in enumerate(zip(edges, edge_sides)):
        node_side_edges[src][src_side].append((ei, True))
        if src != tgt:  # self-loop handled separately
            node_side_edges[tgt][tgt_side].append((ei, False))

    # Compute offsets: for k edges on the same side, positions are
    # (i+1)/(k+1) for i in 0..k-1, evenly distributed along the side
    port_offsets: dict[tuple[int, Side, int], float] = {}  # (node, side, edge_idx) -> offset

    for node, sides in node_side_edges.items():
        for side, entries in sides.items():
            k = len(entries)
            for i, (ei, _is_source) in enumerate(entries):
                offset = (i + 1) / (k + 1)
                port_offsets[(node, side, ei)] = offset

    # Build Port objects
    result: list[tuple[Port, Port]] = []
    for ei, ((src, tgt), (src_side, tgt_side)) in enumerate(zip(edges, edge_sides)):
        src_offset = port_offsets.get((src, src_side, ei), 0.5)
        tgt_offset = port_offsets.get((tgt, tgt_side, ei), 0.5)

        src_port = Port(node=src, side=src_side, position=src_offset, edge=ei)
        tgt_port = Port(node=tgt, side=tgt_side, position=tgt_offset, edge=ei)
        result.append((src_port, tgt_port))

    return result


def route_self_loop(
    box: NodeBox,
    port_out: Port,
    port_in: Port,
    edge_separation: float,
) -> list[tuple[float, float]]:
    """Route a self-loop edge with 4 bends around a corner of the node.

    The loop exits from port_out's side and enters at port_in's side,
    forming a small rectangle outside the node corner.

    Args:
        box: The node box.
        port_out: Outgoing port.
        port_in: Incoming port.
        edge_separation: Gap for the loop offset.

    Returns:
        List of 4 bend coordinates.
    """
    out_pos = box.get_port_position(port_out.side, port_out.position)
    in_pos = box.get_port_position(port_in.side, port_in.position)

    # Determine the corner direction based on the two sides
    out_side = port_out.side
    in_side = port_in.side

    # Compute offset direction for each side
    def _outward(side: Side, amount: float) -> tuple[float, float]:
        if side == Side.NORTH:
            return (0, -amount)
        elif side == Side.SOUTH:
            return (0, amount)
        elif side == Side.EAST:
            return (amount, 0)
        else:  # WEST
            return (-amount, 0)

    d_out = _outward(out_side, edge_separation)
    d_in = _outward(in_side, edge_separation)

    # 4-bend path: out_pos -> bend1 -> corner -> bend2 -> in_pos
    bend1 = (out_pos[0] + d_out[0], out_pos[1] + d_out[1])
    corner = (out_pos[0] + d_out[0] + d_in[0], out_pos[1] + d_out[1] + d_in[1])
    bend2 = (in_pos[0] + d_in[0], in_pos[1] + d_in[1])

    return [bend1, corner, bend2]


def _ensure_orthogonal(
    src: tuple[float, float],
    tgt: tuple[float, float],
    bends: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Fix diagonal segments and simplify the bend path.

    1. Insert L-shaped bends for any diagonal segment.
    2. Remove duplicate consecutive points (zero-length segments).
    3. Remove collinear middle points (redundant bends on the same axis).
    """
    # --- Phase 1: fix diagonals ---
    points = [src] + bends + [tgt]
    fixed: list[tuple[float, float]] = []

    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        if i > 0:
            fixed.append((x1, y1))
        if abs(x1 - x2) > 1e-6 and abs(y1 - y2) > 1e-6:
            # Diagonal segment: insert an L-bend (go vertical first)
            fixed.append((x1, y2))

    # --- Phase 2: remove duplicate consecutive points ---
    deduped: list[tuple[float, float]] = []
    for pt in fixed:
        if deduped and abs(pt[0] - deduped[-1][0]) < 1e-6 and abs(pt[1] - deduped[-1][1]) < 1e-6:
            continue
        deduped.append(pt)

    # Also deduplicate against src (first point) and tgt (last point)
    while deduped and abs(deduped[0][0] - src[0]) < 1e-6 and abs(deduped[0][1] - src[1]) < 1e-6:
        deduped.pop(0)
    while deduped and abs(deduped[-1][0] - tgt[0]) < 1e-6 and abs(deduped[-1][1] - tgt[1]) < 1e-6:
        deduped.pop()

    # --- Phase 3: remove collinear middle points ---
    # Reconstruct the full path and remove interior points that lie on the
    # same axis line as their neighbours.
    full = [src] + deduped + [tgt]
    simplified: list[tuple[float, float]] = []
    for i in range(1, len(full) - 1):
        px, py = full[i - 1]
        cx, cy = full[i]
        nx, ny = full[i + 1]
        # Same horizontal line
        if abs(py - cy) < 1e-6 and abs(cy - ny) < 1e-6:
            continue
        # Same vertical line
        if abs(px - cx) < 1e-6 and abs(cx - nx) < 1e-6:
            continue
        simplified.append((cx, cy))

    return simplified


def route_edge(
    src_box: NodeBox,
    tgt_box: NodeBox,
    src_port: Port,
    tgt_port: Port,
    obstacles: list[NodeBox],
    edge_separation: float,
) -> list[tuple[float, float]]:
    """Route a single edge with basic obstacle awareness.

    Uses the 5-case bend logic (same as original _compute_edge_route),
    then checks for segment-node overlaps and adds detour bends if needed.

    Args:
        src_box: Source node box.
        tgt_box: Target node box.
        src_port: Source port.
        tgt_port: Target port.
        obstacles: Other node boxes to avoid.
        edge_separation: Gap for detour offsets.

    Returns:
        List of bend coordinates.
    """
    src_pos = src_box.get_port_position(src_port.side, src_port.position)
    tgt_pos = tgt_box.get_port_position(tgt_port.side, tgt_port.position)

    sx, sy = src_pos
    tx, ty = tgt_pos

    src_side = src_port.side
    tgt_side = tgt_port.side

    bends: list[tuple[float, float]] = []

    if src_side in (Side.NORTH, Side.SOUTH) and tgt_side in (Side.NORTH, Side.SOUTH):
        if abs(sx - tx) < 1e-6:
            pass  # aligned vertically
        else:
            mid_y = (sy + ty) / 2
            bends = [(sx, mid_y), (tx, mid_y)]

    elif src_side in (Side.EAST, Side.WEST) and tgt_side in (Side.EAST, Side.WEST):
        if abs(sy - ty) < 1e-6:
            pass  # aligned horizontally
        else:
            mid_x = (sx + tx) / 2
            bends = [(mid_x, sy), (mid_x, ty)]

    elif src_side in (Side.NORTH, Side.SOUTH) and tgt_side in (Side.EAST, Side.WEST):
        bends = [(sx, ty)]

    elif src_side in (Side.EAST, Side.WEST) and tgt_side in (Side.NORTH, Side.SOUTH):
        bends = [(tx, sy)]

    else:
        # Same direction exits
        offset = edge_separation * 2

        if src_side == Side.SOUTH and tgt_side == Side.SOUTH:
            detour_y = max(sy, ty) + offset
            bends = [(sx, detour_y), (tx, detour_y)]
        elif src_side == Side.NORTH and tgt_side == Side.NORTH:
            detour_y = min(sy, ty) - offset
            bends = [(sx, detour_y), (tx, detour_y)]
        elif src_side == Side.EAST and tgt_side == Side.EAST:
            detour_x = max(sx, tx) + offset
            bends = [(detour_x, sy), (detour_x, ty)]
        elif src_side == Side.WEST and tgt_side == Side.WEST:
            detour_x = min(sx, tx) - offset
            bends = [(detour_x, sy), (detour_x, ty)]

    # Check for obstacle overlaps and add detours
    bends = _add_obstacle_detours(
        src_pos, tgt_pos, bends, obstacles, src_box, tgt_box, edge_separation
    )

    # Ensure all segments are orthogonal by fixing any diagonal segments.
    bends = _ensure_orthogonal(src_pos, tgt_pos, bends)

    return bends


def _add_obstacle_detours(
    src_pos: tuple[float, float],
    tgt_pos: tuple[float, float],
    bends: list[tuple[float, float]],
    obstacles: list[NodeBox],
    src_box: NodeBox,
    tgt_box: NodeBox,
    edge_separation: float,
) -> list[tuple[float, float]]:
    """Route edge around obstacles using visibility graph routing.

    Builds an orthogonal visibility graph from the source/target positions
    and obstacle corners, then finds the shortest orthogonal path using
    BFS (minimizing number of bends).

    Falls back to simple detour heuristic if visibility routing fails.
    """
    if not obstacles:
        return bends

    # Filter obstacles to only those between src and tgt
    real_obstacles = [
        obs for obs in obstacles if obs.index != src_box.index and obs.index != tgt_box.index
    ]
    if not real_obstacles:
        return bends

    # Check if any segment of the current route intersects an obstacle
    points = [src_pos] + bends + [tgt_pos]
    has_intersection = False
    for i in range(len(points) - 1):
        for obs in real_obstacles:
            if _segment_intersects_box(points[i], points[i + 1], obs):
                has_intersection = True
                break
        if has_intersection:
            break

    if not has_intersection:
        return bends

    # Build visibility graph and find shortest orthogonal path
    vis_path = _visibility_graph_route(src_pos, tgt_pos, real_obstacles, edge_separation)
    if vis_path is not None:
        # vis_path includes src and tgt; extract bends (intermediate points)
        return vis_path[1:-1]

    # Fallback: simple detour heuristic
    return _simple_detour_route(src_pos, tgt_pos, bends, real_obstacles, edge_separation)


def _build_visibility_graph(
    src: tuple[float, float],
    tgt: tuple[float, float],
    obstacles: list[NodeBox],
    margin: float,
) -> tuple[list[tuple[float, float]], dict[int, list[int]]]:
    """Build an orthogonal visibility graph.

    Nodes are the source, target, and offset corners of each obstacle box.
    Two nodes are connected if they share an x or y coordinate and the
    axis-aligned segment between them doesn't pass through any obstacle.

    Returns:
        (node_positions, adjacency_dict)
    """
    nodes: list[tuple[float, float]] = [src, tgt]

    # Add offset corner points for each obstacle
    for obs in obstacles:
        corners = [
            (obs.left - margin, obs.top - margin),
            (obs.right + margin, obs.top - margin),
            (obs.left - margin, obs.bottom + margin),
            (obs.right + margin, obs.bottom + margin),
        ]
        nodes.extend(corners)

    n = len(nodes)
    adj: dict[int, list[int]] = {i: [] for i in range(n)}

    # Connect nodes that share an x or y coordinate with clear line-of-sight
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi = nodes[i]
            xj, yj = nodes[j]

            # Must share x or y (axis-aligned connection)
            if abs(xi - xj) < 1e-6 or abs(yi - yj) < 1e-6:
                # Check line-of-sight
                blocked = False
                for obs in obstacles:
                    if _segment_intersects_box(nodes[i], nodes[j], obs):
                        blocked = True
                        break
                if not blocked:
                    adj[i].append(j)
                    adj[j].append(i)

    # Also add L-shaped connections through shared coordinate lines
    # For each pair of non-aligned nodes, add a via point if it's clear
    via_nodes: list[tuple[float, float]] = []
    via_start = n
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi = nodes[i]
            xj, yj = nodes[j]
            if abs(xi - xj) < 1e-6 or abs(yi - yj) < 1e-6:
                continue  # Already aligned

            # Two possible via points: (xi, yj) and (xj, yi)
            for via in [(xi, yj), (xj, yi)]:
                # Check if via point is inside any obstacle
                inside = False
                for obs in obstacles:
                    if (
                        obs.left - 1e-6 < via[0] < obs.right + 1e-6
                        and obs.top - 1e-6 < via[1] < obs.bottom + 1e-6
                    ):
                        inside = True
                        break
                if inside:
                    continue

                # Check if both segments to via are clear
                seg1_clear = True
                seg2_clear = True
                for obs in obstacles:
                    if _segment_intersects_box(nodes[i], via, obs):
                        seg1_clear = False
                        break
                if not seg1_clear:
                    continue
                for obs in obstacles:
                    if _segment_intersects_box(via, nodes[j], obs):
                        seg2_clear = False
                        break
                if not seg2_clear:
                    continue

                # Add via node
                via_idx = via_start + len(via_nodes)
                via_nodes.append(via)
                adj[via_idx] = [i, j]
                adj[i].append(via_idx)
                adj[j].append(via_idx)

    nodes.extend(via_nodes)
    return nodes, adj


def _visibility_graph_route(
    src: tuple[float, float],
    tgt: tuple[float, float],
    obstacles: list[NodeBox],
    margin: float,
) -> Optional[list[tuple[float, float]]]:
    """Find shortest orthogonal path from src to tgt avoiding obstacles.

    Uses BFS on the visibility graph (minimizes number of bends/segments).

    Returns:
        List of waypoints from src to tgt (inclusive), or None if no path.
    """
    nodes, adj = _build_visibility_graph(src, tgt, obstacles, margin)

    if not adj[0] and not adj[1]:
        return None

    # BFS from node 0 (src) to node 1 (tgt)
    from collections import deque

    visited = {0}
    queue: deque[tuple[int, list[int]]] = deque([(0, [0])])

    while queue:
        current, path = queue.popleft()
        if current == 1:
            return [nodes[i] for i in path]
        for nb in adj[current]:
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, path + [nb]))

    return None


def _simple_detour_route(
    src_pos: tuple[float, float],
    tgt_pos: tuple[float, float],
    bends: list[tuple[float, float]],
    obstacles: list[NodeBox],
    edge_separation: float,
) -> list[tuple[float, float]]:
    """Simple detour heuristic fallback for obstacle avoidance."""
    points = [src_pos] + bends + [tgt_pos]
    new_bends: list[tuple[float, float]] = []

    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]

        detoured = False
        for obs in obstacles:
            if _segment_intersects_box(p1, p2, obs):
                detour = _compute_detour(p1, p2, obs, edge_separation)
                new_bends.extend(detour)
                detoured = True
                break

        if not detoured and i > 0:
            new_bends.append(p1)

    if not new_bends and bends:
        return bends
    return new_bends if new_bends else bends


def _segment_intersects_box(
    p1: tuple[float, float],
    p2: tuple[float, float],
    box: NodeBox,
) -> bool:
    """Check if an axis-aligned segment passes through a node box."""
    x1, y1 = p1
    x2, y2 = p2

    if abs(x1 - x2) < 1e-6:
        # Vertical segment
        min_y = min(y1, y2)
        max_y = max(y1, y2)
        return (box.left < x1 < box.right) and (min_y < box.bottom) and (max_y > box.top)
    elif abs(y1 - y2) < 1e-6:
        # Horizontal segment
        min_x = min(x1, x2)
        max_x = max(x1, x2)
        return (box.top < y1 < box.bottom) and (min_x < box.right) and (max_x > box.left)

    return False


def _compute_detour(
    p1: tuple[float, float],
    p2: tuple[float, float],
    obs: NodeBox,
    edge_separation: float,
) -> list[tuple[float, float]]:
    """Compute detour bends around an obstacle box."""
    x1, y1 = p1
    x2, y2 = p2

    if abs(x1 - x2) < 1e-6:
        if x1 > obs.x:
            detour_x = obs.right + edge_separation
        else:
            detour_x = obs.left - edge_separation
        return [(detour_x, y1), (detour_x, y2)]
    else:
        if y1 > obs.y:
            detour_y = obs.bottom + edge_separation
        else:
            detour_y = obs.top - edge_separation
        return [(x1, detour_y), (x2, detour_y)]


def route_all_edges(
    boxes: list[NodeBox],
    edges: list[tuple[int, int]],
    edge_indices: list[int],
    edge_separation: float = 15.0,
    port_constraints: Optional[dict[tuple[int, int], tuple[Optional[Side], Optional[Side]]]] = None,
    side_fn: Optional[Callable[[NodeBox, NodeBox], tuple[Side, Side]]] = None,
) -> list[OrthogonalEdge]:
    """Route all edges with global awareness.

    Pipeline:
    1. Classify edges: normal, self-loop, parallel
    2. Determine port sides (using constraints or heuristic)
    3. Assign ports (distributing multiple edges per side)
    4. Route each edge (self-loops, normal, with obstacle awareness)

    Args:
        boxes: Node boxes indexed by node id.
        edges: Edge list as (source, target) pairs.
        edge_indices: Original edge index for each edge (for port tracking).
        edge_separation: Minimum gap between parallel edge segments.
        port_constraints: Optional mapping (src, tgt) -> (src_side, tgt_side)
            where None means "use heuristic".
        side_fn: Optional custom function to determine port sides.

    Returns:
        List of OrthogonalEdge for each input edge.
    """
    if not edges or not boxes:
        return []

    box_map = {box.index: box for box in boxes}

    # Step 1: Determine port sides for each edge
    edge_sides: list[tuple[Side, Side]] = []
    for src, tgt in edges:
        if src == tgt:
            # Self-loop: exit EAST, enter SOUTH (default corner)
            edge_sides.append((Side.EAST, Side.SOUTH))
            continue

        src_box = box_map.get(src)
        tgt_box = box_map.get(tgt)
        if src_box is None or tgt_box is None:
            edge_sides.append((Side.SOUTH, Side.NORTH))
            continue

        # Check for explicit port constraints
        constrained = False
        if port_constraints and (src, tgt) in port_constraints:
            cs, ct = port_constraints[(src, tgt)]
            heuristic = (side_fn or determine_port_sides)(src_box, tgt_box)
            final_s = cs if cs is not None else heuristic[0]
            final_t = ct if ct is not None else heuristic[1]
            edge_sides.append((final_s, final_t))
            constrained = True

        if not constrained:
            sides = (side_fn or determine_port_sides)(src_box, tgt_box)
            edge_sides.append(sides)

    # Step 2: Assign ports with even distribution
    ports = assign_ports(boxes, edges, edge_sides)

    # Step 3: Route each edge
    result: list[OrthogonalEdge] = []
    for ei, ((src, tgt), (src_port, tgt_port)) in enumerate(zip(edges, ports)):
        src_box = box_map.get(src)
        tgt_box = box_map.get(tgt)

        if src_box is None or tgt_box is None:
            # Skip edges with missing boxes
            result.append(
                OrthogonalEdge(
                    source=src,
                    target=tgt,
                    source_port=src_port,
                    target_port=tgt_port,
                    bends=[],
                )
            )
            continue

        if src == tgt:
            # Self-loop routing
            bends = route_self_loop(src_box, src_port, tgt_port, edge_separation)
        else:
            # Normal edge routing with obstacle awareness
            # Obstacles are all boxes except src and tgt
            obstacles = [b for b in boxes if b.index != src and b.index != tgt]
            bends = route_edge(src_box, tgt_box, src_port, tgt_port, obstacles, edge_separation)

        edge_idx = edge_indices[ei] if ei < len(edge_indices) else ei
        # Update port edge indices
        src_port_final = Port(
            node=src_port.node,
            side=src_port.side,
            position=src_port.position,
            edge=edge_idx,
        )
        tgt_port_final = Port(
            node=tgt_port.node,
            side=tgt_port.side,
            position=tgt_port.position,
            edge=edge_idx,
        )

        result.append(
            OrthogonalEdge(
                source=src,
                target=tgt,
                source_port=src_port_final,
                target_port=tgt_port_final,
                bends=bends,
            )
        )

    return result


def nudge_overlapping_segments(
    edges: list[OrthogonalEdge],
    boxes: list[NodeBox],
    edge_separation: float = 15.0,
) -> list[OrthogonalEdge]:
    """Separate overlapping/coincident parallel edge segments.

    Post-processing step that finds groups of edge segments on the same
    axis coordinate (within tolerance) and spreads them apart by
    edge_separation increments, centered on the original position.

    Args:
        edges: Routed orthogonal edges.
        boxes: Node boxes (for computing port positions).
        edge_separation: Minimum gap between parallel segments.

    Returns:
        New list of OrthogonalEdge with adjusted bend points.
    """
    if not edges:
        return edges

    box_map = {box.index: box for box in boxes}

    # Collect all segments from all edges
    # segment_info: list of (edge_idx, seg_idx, x1, y1, x2, y2, is_horizontal)
    all_segments: list[tuple[int, int, float, float, float, float, bool]] = []

    for ei, edge in enumerate(edges):
        src_box = box_map.get(edge.source)
        tgt_box = box_map.get(edge.target)
        if src_box is None or tgt_box is None:
            continue
        src_pos = src_box.get_port_position(edge.source_port.side, edge.source_port.position)
        tgt_pos = tgt_box.get_port_position(edge.target_port.side, edge.target_port.position)

        points = [src_pos] + list(edge.bends) + [tgt_pos]
        for si in range(len(points) - 1):
            x1, y1 = points[si]
            x2, y2 = points[si + 1]
            if abs(y1 - y2) < 1e-6:
                # Horizontal segment
                all_segments.append((ei, si, min(x1, x2), y1, max(x1, x2), y2, True))
            elif abs(x1 - x2) < 1e-6:
                # Vertical segment
                all_segments.append((ei, si, x1, min(y1, y2), x2, max(y1, y2), False))

    # Group horizontal segments by y-coordinate
    h_groups: dict[float, list[tuple[int, int, float, float, float, float, bool]]] = {}
    v_groups: dict[float, list[tuple[int, int, float, float, float, float, bool]]] = {}

    threshold = edge_separation * 0.5

    for seg in all_segments:
        ei, si, x1, y1, x2, y2, is_h = seg
        if is_h:
            # Group by y coordinate
            found = False
            for key in list(h_groups.keys()):
                if abs(key - y1) < threshold:
                    h_groups[key].append(seg)
                    found = True
                    break
            if not found:
                h_groups[y1] = [seg]
        else:
            # Group by x coordinate
            found = False
            for key in list(v_groups.keys()):
                if abs(key - x1) < threshold:
                    v_groups[key].append(seg)
                    found = True
                    break
            if not found:
                v_groups[x1] = [seg]

    # For groups with overlapping segments, compute offsets
    # A segment overlaps another if they share the same axis coordinate
    # AND their spans on the other axis overlap
    nudge_map: dict[tuple[int, int], float] = {}  # (edge_idx, seg_idx) -> offset

    def _spans_overlap(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> bool:
        return a_lo < b_hi and b_lo < a_hi

    for group_coord, segs in h_groups.items():
        if len(segs) <= 1:
            continue
        # Find overlapping subsets
        overlapping = _find_overlapping_clusters(segs, is_horizontal=True)
        for cluster in overlapping:
            if len(cluster) <= 1:
                continue
            n_segs = len(cluster)
            for i, seg in enumerate(cluster):
                offset = (i - (n_segs - 1) / 2.0) * edge_separation
                nudge_map[(seg[0], seg[1])] = offset

    for group_coord, segs in v_groups.items():
        if len(segs) <= 1:
            continue
        overlapping = _find_overlapping_clusters(segs, is_horizontal=False)
        for cluster in overlapping:
            if len(cluster) <= 1:
                continue
            n_segs = len(cluster)
            for i, seg in enumerate(cluster):
                offset = (i - (n_segs - 1) / 2.0) * edge_separation
                nudge_map[(seg[0], seg[1])] = offset

    if not nudge_map:
        return edges

    # Apply nudges to bend points
    result: list[OrthogonalEdge] = []
    for ei, edge in enumerate(edges):
        src_box = box_map.get(edge.source)
        tgt_box = box_map.get(edge.target)
        if src_box is None or tgt_box is None:
            result.append(edge)
            continue

        src_pos = src_box.get_port_position(edge.source_port.side, edge.source_port.position)
        tgt_pos = tgt_box.get_port_position(edge.target_port.side, edge.target_port.position)

        points = [src_pos] + list(edge.bends) + [tgt_pos]
        n_pts = len(points)
        new_points = list(points)

        # Apply nudges to intermediate points (bends only, not src/tgt)
        for si in range(n_pts - 1):
            seg_key = (ei, si)
            if seg_key not in nudge_map:
                continue
            offset = nudge_map[seg_key]
            x1, y1 = points[si]
            x2, y2 = points[si + 1]

            if abs(y1 - y2) < 1e-6:
                # Horizontal segment: nudge y
                if si > 0:
                    new_points[si] = (new_points[si][0], new_points[si][1] + offset)
                if si + 1 < n_pts - 1:
                    new_points[si + 1] = (new_points[si + 1][0], new_points[si + 1][1] + offset)
            elif abs(x1 - x2) < 1e-6:
                # Vertical segment: nudge x
                if si > 0:
                    new_points[si] = (new_points[si][0] + offset, new_points[si][1])
                if si + 1 < n_pts - 1:
                    new_points[si + 1] = (new_points[si + 1][0] + offset, new_points[si + 1][1])

        # Extract bends (exclude first and last points)
        new_bends = new_points[1:-1]

        # Nudging can create diagonal segments when port endpoints are
        # not moved but adjacent bends are.  Fix by inserting L-shaped
        # connecting bends so every segment is strictly orthogonal.
        new_bends = _ensure_orthogonal(src_pos, tgt_pos, new_bends)

        result.append(
            OrthogonalEdge(
                source=edge.source,
                target=edge.target,
                source_port=edge.source_port,
                target_port=edge.target_port,
                bends=new_bends,
            )
        )

    return result


def _find_overlapping_clusters(
    segs: list[tuple[int, int, float, float, float, float, bool]],
    is_horizontal: bool,
) -> list[list[tuple[int, int, float, float, float, float, bool]]]:
    """Find clusters of segments that overlap on the cross-axis span."""
    if not segs:
        return []

    clusters: list[list[tuple[int, int, float, float, float, float, bool]]] = []

    for seg in segs:
        _, _, x1, y1, x2, y2, _ = seg
        if is_horizontal:
            span_lo, span_hi = x1, x2
        else:
            span_lo, span_hi = y1, y2

        placed = False
        for cluster in clusters:
            # Check if this segment overlaps with any in the cluster
            for cseg in cluster:
                _, _, cx1, cy1, cx2, cy2, _ = cseg
                if is_horizontal:
                    clo, chi = cx1, cx2
                else:
                    clo, chi = cy1, cy2
                if span_lo < chi and clo < span_hi:
                    cluster.append(seg)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            clusters.append([seg])

    return clusters


__all__ = [
    "assign_ports",
    "determine_port_sides",
    "nudge_overlapping_segments",
    "route_all_edges",
    "route_edge",
    "route_self_loop",
]
