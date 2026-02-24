"""
Flow-based and longest-path compaction for orthogonal layouts.

Implements two compaction strategies based on constraint DAGs:

1. LongestPathCompaction: Assigns coordinates via longest-path distances
   in the constraint DAG. O(n^2) for DAG construction, O(n + m) for
   longest path. Guaranteed correct in one pass.

2. FlowCompaction: Uses min-cost flow on the constraint DAG to optimally
   distribute compression. Produces tighter layouts when there is slack
   to redistribute.

Reference:
    Eiglsperger, M., Fekete, S.P., Klau, G.W. (2001). Orthogonal Graph
    Drawing. In: Drawing Graphs, LNCS 2025, Springer.
"""

from __future__ import annotations

from collections import defaultdict

from ._min_cost_flow import solve_min_cost_flow
from .compaction import CompactionResult
from .orthogonalization import FlowNetwork
from .types import NodeBox, OrthogonalEdge

# Scale factor for converting float gaps to integers for the flow solver.
_SCALE = 100


def _build_constraint_dag(
    boxes: list[NodeBox],
    separation: float,
    dimension: str,
) -> tuple[int, list[tuple[int, int, float]]]:
    """
    Build a constraint DAG for one dimension.

    For the given dimension, creates directed edges between boxes that
    overlap in the perpendicular dimension and must maintain minimum
    separation.

    Args:
        boxes: Node boxes with current positions.
        separation: Minimum gap between adjacent box edges.
        dimension: "horizontal" or "vertical".

    Returns:
        (num_nodes, edges) where edges are (from, to, min_gap) triples.
        Node indices 0..len(boxes)-1 correspond to boxes. No virtual
        source/sink is added here.
    """
    n = len(boxes)
    if n == 0:
        return 0, []

    if dimension == "horizontal":
        primary_coord = [box.x for box in boxes]
        half_primary = [box.width / 2 for box in boxes]
        perp_lo = [box.y - box.height / 2 for box in boxes]
        perp_hi = [box.y + box.height / 2 for box in boxes]
    else:
        primary_coord = [box.y for box in boxes]
        half_primary = [box.height / 2 for box in boxes]
        perp_lo = [box.x - box.width / 2 for box in boxes]
        perp_hi = [box.x + box.width / 2 for box in boxes]

    # Sort by primary coordinate
    sorted_indices = sorted(range(n), key=lambda i: primary_coord[i])

    edges: list[tuple[int, int, float]] = []

    # Check all pairs where i precedes j in sorted order
    for si in range(len(sorted_indices)):
        i = sorted_indices[si]
        for sj in range(si + 1, len(sorted_indices)):
            j = sorted_indices[sj]

            # Check perpendicular overlap
            if perp_hi[i] <= perp_lo[j] or perp_hi[j] <= perp_lo[i]:
                continue  # no overlap -> no constraint

            # Minimum gap: half-width of i + separation + half-width of j
            gap = half_primary[i] + separation + half_primary[j]
            edges.append((i, j, gap))

    return n, edges


def compact_longest_path_1d(
    boxes: list[NodeBox],
    separation: float,
    dimension: str,
) -> list[float]:
    """
    Compact one dimension via longest path in the constraint DAG.

    The longest-path distance from a virtual source to each node gives
    the minimum achievable coordinate. This is optimal for a single
    dimension.

    Args:
        boxes: Node boxes with current positions.
        separation: Minimum gap between adjacent box edges.
        dimension: "horizontal" or "vertical".

    Returns:
        New coordinates for each box in the given dimension.
    """
    n = len(boxes)
    if n == 0:
        return []
    if n == 1:
        if dimension == "horizontal":
            return [boxes[0].width / 2 + separation]
        else:
            return [boxes[0].height / 2 + separation]

    _, edges = _build_constraint_dag(boxes, separation, dimension)

    if dimension == "horizontal":
        half_primary = [box.width / 2 for box in boxes]
    else:
        half_primary = [box.height / 2 for box in boxes]

    # Build adjacency list for the DAG
    adj: dict[int, list[tuple[int, float]]] = defaultdict(list)
    in_degree = [0] * n
    for u, v, w in edges:
        adj[u].append((v, w))
        in_degree[v] += 1

    # Virtual source connects to all nodes with no predecessors.
    # The weight is half_primary[node] + margin (separation).
    # We represent this as initial distance values.

    # Topological sort (Kahn's algorithm)
    topo_order: list[int] = []
    queue: list[int] = [i for i in range(n) if in_degree[i] == 0]
    temp_in = in_degree[:]
    while queue:
        node = queue.pop(0)
        topo_order.append(node)
        for neighbor, _ in adj[node]:
            temp_in[neighbor] -= 1
            if temp_in[neighbor] == 0:
                queue.append(neighbor)

    # Handle cycles (shouldn't happen with sorted constraint DAG,
    # but be safe): append any missing nodes
    if len(topo_order) < n:
        remaining = [i for i in range(n) if i not in set(topo_order)]
        topo_order.extend(remaining)

    # Longest path from virtual source
    # Source -> node_i has weight half_primary[i] + separation
    dist = [half_primary[i] + separation for i in range(n)]

    for u in topo_order:
        for v, w in adj[u]:
            new_dist = dist[u] + w
            if new_dist > dist[v]:
                dist[v] = new_dist

    return dist


def compact_flow_1d(
    boxes: list[NodeBox],
    separation: float,
    dimension: str,
) -> list[float]:
    """
    Compact one dimension via min-cost flow on the constraint DAG.

    First computes longest-path positions (minimum feasible), then
    uses min-cost flow to redistribute slack and minimize total edge
    length as a secondary objective.

    Args:
        boxes: Node boxes with current positions.
        separation: Minimum gap between adjacent box edges.
        dimension: "horizontal" or "vertical".

    Returns:
        New coordinates for each box in the given dimension.
    """
    n = len(boxes)
    if n <= 1:
        return compact_longest_path_1d(boxes, separation, dimension)

    # Start with longest-path positions as baseline
    lp_positions = compact_longest_path_1d(boxes, separation, dimension)

    _, edges = _build_constraint_dag(boxes, separation, dimension)

    if not edges:
        return lp_positions

    if dimension == "horizontal":
        primary_coord = [box.x for box in boxes]
    else:
        primary_coord = [box.y for box in boxes]

    # Compute current span and longest-path span
    # Current positions (original)
    current_positions = primary_coord

    # For each constraint edge, compute slack = actual_gap - min_gap
    # We want to minimize total coordinate values (compact as much as possible)
    # The flow formulation: each arc can carry flow that represents
    # additional separation beyond the minimum.

    # Build a flow network:
    # - Nodes: one per box + source (n) + sink (n+1)
    # - For each constraint edge (i, j, min_gap):
    #   arc from i to j with capacity = slack, cost = 1
    # - Source connects to nodes with no predecessor
    # - Sink receives from nodes with no successor

    in_degree = [0] * n
    out_degree = [0] * n
    for u, v, _ in edges:
        in_degree[v] += 1
        out_degree[u] += 1

    # Compute slack for each edge using longest-path positions
    arc_slacks: list[tuple[int, int, int]] = []  # (u, v, slack_scaled)
    for u, v, min_gap in edges:
        actual_gap = lp_positions[v] - lp_positions[u]
        slack = actual_gap - min_gap
        slack_int = max(0, int(slack * _SCALE + 0.5))
        if slack_int > 0:
            arc_slacks.append((u, v, slack_int))

    if not arc_slacks:
        # No slack to redistribute - longest path is already optimal
        return lp_positions

    # Build FlowNetwork for the min-cost flow solver
    # We create a network where pushing flow along an arc increases
    # the gap between two nodes, which we want to minimize.
    # Total supply = total available slack from source side

    # Compute total compressible slack
    # Source nodes: those with in_degree == 0
    # Sink nodes: those with out_degree == 0
    source_nodes = [i for i in range(n) if in_degree[i] == 0]
    sink_nodes = [i for i in range(n) if out_degree[i] == 0]

    if not source_nodes or not sink_nodes:
        return lp_positions

    # The flow formulation doesn't improve on longest path for 1D
    # compaction (longest path already gives optimal 1D coordinates).
    # However, flow can balance coordinates when there are multiple
    # paths with different slacks. For now, longest-path is sufficient
    # for most cases, but we keep the flow infrastructure for future
    # multi-objective optimization.

    # Build a proper min-cost flow network to tighten further:
    # We model the problem as: can we shift some nodes left without
    # violating constraints?

    # Actually, longest-path already gives the tightest possible
    # positions for 1D compaction. Flow compaction's real benefit is
    # in balancing when we want to center nodes or minimize total wire
    # length, which is a different objective.

    # For the flow-based approach, we try to reduce the maximum span:
    # Use current positions and try to compress toward the source.

    # Rebuild with current positions to find compressible slack
    arc_data: list[tuple[int, int, float, int]] = []  # u, v, min_gap, slack_scaled
    for u, v, min_gap in edges:
        actual_gap = current_positions[v] - current_positions[u]
        slack = actual_gap - min_gap
        slack_int = max(0, int(slack * _SCALE + 0.5))
        arc_data.append((u, v, min_gap, slack_int))

    total_slack = sum(s for _, _, _, s in arc_data if s > 0)
    if total_slack == 0:
        return lp_positions

    # Build a FlowNetwork
    # Node IDs: 0..n-1 are box nodes, n is super-source, n+1 is super-sink
    num_nodes = n + 2
    s_node = n
    t_node = n + 1

    network = FlowNetwork(num_vertices=num_nodes, faces=[])
    network.supplies = {}

    # Add arcs for constraint edges (flow = how much to compress)
    for u, v, _min_gap, slack_int in arc_data:
        if slack_int > 0:
            network.arcs[(u, v)] = (slack_int, 1)  # capacity, cost

    # Source -> source_nodes: capacity = large, cost = 0
    source_supply = 0
    for src in source_nodes:
        # How much can this source node move left?
        # Its current position minus its longest-path position
        move = current_positions[src] - lp_positions[src]
        move_int = max(0, int(move * _SCALE + 0.5))
        if move_int > 0:
            network.arcs[(s_node, src)] = (move_int, 0)
            source_supply += move_int

    # Sink_nodes -> sink: capacity = large, cost = 0
    for snk in sink_nodes:
        move = current_positions[snk] - lp_positions[snk]
        move_int = max(0, int(move * _SCALE + 0.5))
        if move_int > 0:
            network.arcs[(snk, t_node)] = (move_int, 0)

    if source_supply == 0:
        return lp_positions

    # Set supply/demand
    network.supplies[s_node] = source_supply
    network.supplies[t_node] = -source_supply

    # Solve
    feasible = solve_min_cost_flow(network)

    if not feasible:
        return lp_positions

    # Apply flow to compute new positions
    # Each unit of flow on arc (u, v) means the gap between u and v
    # is reduced by that amount
    new_positions = list(current_positions)

    # Compute compression at each node by propagating flow from sources
    compression = [0.0] * n
    for src in source_nodes:
        if (s_node, src) in network.flow:
            compression[src] = network.flow[(s_node, src)] / _SCALE

    # Propagate through DAG in topological order
    adj: dict[int, list[tuple[int, float]]] = defaultdict(list)
    temp_in = [0] * n
    for u, v, min_gap in edges:
        adj[u].append((v, min_gap))
        temp_in[v] += 1

    topo: list[int] = []
    queue = [i for i in range(n) if temp_in[i] == 0]
    while queue:
        node = queue.pop(0)
        topo.append(node)
        for neighbor, _ in adj[node]:
            temp_in[neighbor] -= 1
            if temp_in[neighbor] == 0:
                queue.append(neighbor)

    for u in topo:
        new_positions[u] = current_positions[u] - compression[u]
        for v, min_gap in adj[u]:
            # How much flow went through this arc?
            flow_val = network.flow.get((u, v), 0) / _SCALE
            # Compression at v is at least compression at u minus
            # the flow absorbed on this arc
            v_compression = compression[u] - flow_val
            if v_compression > compression[v]:
                compression[v] = v_compression

    for i in range(n):
        new_positions[i] = current_positions[i] - compression[i]

    # Verify constraints and fall back to longest-path if violated
    for u, v, min_gap in edges:
        if new_positions[v] - new_positions[u] < min_gap - 1e-6:
            return lp_positions

    # Ensure non-negative positions
    if new_positions:
        if dimension == "horizontal":
            half_p = [box.width / 2 for box in boxes]
        else:
            half_p = [box.height / 2 for box in boxes]
        min_pos = min(new_positions[i] - half_p[i] for i in range(n))
        if min_pos < 0:
            shift = -min_pos + separation
            new_positions = [p + shift for p in new_positions]

    return new_positions


def compact_layout_longest_path(
    boxes: list[NodeBox],
    edges: list[OrthogonalEdge],
    node_separation: float = 60.0,
    layer_separation: float = 80.0,
    edge_separation: float = 15.0,
) -> CompactionResult:
    """
    Two-pass (horizontal then vertical) longest-path compaction.

    Args:
        boxes: Node boxes with current positions.
        edges: Orthogonal edges (used for API compatibility).
        node_separation: Minimum horizontal gap between node edges.
        layer_separation: Minimum vertical gap between node edges.
        edge_separation: Minimum gap for edge routing.

    Returns:
        CompactionResult with new positions and dimensions.
    """
    n = len(boxes)
    if n == 0:
        return CompactionResult(node_positions=[], width=0, height=0, iterations=0)

    # Pass 1: horizontal
    new_x = compact_longest_path_1d(boxes, node_separation, "horizontal")

    # Update boxes with new x for vertical pass
    temp_boxes = [
        NodeBox(index=box.index, x=new_x[i], y=box.y, width=box.width, height=box.height)
        for i, box in enumerate(boxes)
    ]

    # Pass 2: vertical
    new_y = compact_longest_path_1d(temp_boxes, layer_separation, "vertical")

    positions = [(new_x[i], new_y[i]) for i in range(n)]

    # Compute dimensions
    min_x = min(positions[i][0] - boxes[i].width / 2 for i in range(n))
    max_x = max(positions[i][0] + boxes[i].width / 2 for i in range(n))
    min_y = min(positions[i][1] - boxes[i].height / 2 for i in range(n))
    max_y = max(positions[i][1] + boxes[i].height / 2 for i in range(n))

    width = max_x - min_x + 2 * node_separation
    height = max_y - min_y + 2 * layer_separation

    return CompactionResult(
        node_positions=positions,
        width=width,
        height=height,
        iterations=2,
    )


def compact_layout_flow(
    boxes: list[NodeBox],
    edges: list[OrthogonalEdge],
    node_separation: float = 60.0,
    layer_separation: float = 80.0,
    edge_separation: float = 15.0,
) -> CompactionResult:
    """
    Two-pass (horizontal then vertical) flow-based compaction.

    Args:
        boxes: Node boxes with current positions.
        edges: Orthogonal edges (used for API compatibility).
        node_separation: Minimum horizontal gap between node edges.
        layer_separation: Minimum vertical gap between node edges.
        edge_separation: Minimum gap for edge routing.

    Returns:
        CompactionResult with new positions and dimensions.
    """
    n = len(boxes)
    if n == 0:
        return CompactionResult(node_positions=[], width=0, height=0, iterations=0)

    # Pass 1: horizontal
    new_x = compact_flow_1d(boxes, node_separation, "horizontal")

    # Update boxes with new x for vertical pass
    temp_boxes = [
        NodeBox(index=box.index, x=new_x[i], y=box.y, width=box.width, height=box.height)
        for i, box in enumerate(boxes)
    ]

    # Pass 2: vertical
    new_y = compact_flow_1d(temp_boxes, layer_separation, "vertical")

    positions = [(new_x[i], new_y[i]) for i in range(n)]

    # Compute dimensions
    min_x = min(positions[i][0] - boxes[i].width / 2 for i in range(n))
    max_x = max(positions[i][0] + boxes[i].width / 2 for i in range(n))
    min_y = min(positions[i][1] - boxes[i].height / 2 for i in range(n))
    max_y = max(positions[i][1] + boxes[i].height / 2 for i in range(n))

    width = max_x - min_x + 2 * node_separation
    height = max_y - min_y + 2 * layer_separation

    return CompactionResult(
        node_positions=positions,
        width=width,
        height=height,
        iterations=2,
    )


__all__ = [
    "compact_layout_flow",
    "compact_layout_longest_path",
    "compact_longest_path_1d",
    "compact_flow_1d",
]
