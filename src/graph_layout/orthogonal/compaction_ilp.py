"""
ILP-based optimal compaction for orthogonal layout.

Uses scipy.optimize.milp to solve an Integer Linear Program that minimizes
the total layout area while respecting node separation and edge routing
constraints.

If scipy is not available, falls back to the greedy compaction algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import NodeBox, OrthogonalEdge

# Try to import scipy for ILP solving
try:
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


@dataclass
class ILPCompactionResult:
    """Result of ILP-based compaction."""

    node_positions: list[tuple[float, float]]  # New (x, y) positions for nodes
    width: float  # Total width of compacted layout
    height: float  # Total height of compacted layout
    optimal: bool  # True if ILP solved optimally, False if fallback used
    solver_status: str  # Status message from solver


def is_scipy_available() -> bool:
    """Check if scipy is available for ILP solving."""
    return _HAS_SCIPY


def compact_layout_ilp(
    boxes: list[NodeBox],
    edges: list[OrthogonalEdge],
    node_separation: float = 60.0,
    layer_separation: float = 80.0,
    edge_separation: float = 15.0,
    max_time: float = 10.0,
) -> ILPCompactionResult:
    """
    Compact layout using Integer Linear Programming.

    Formulates compaction as an optimization problem:
    - Variables: x[i], y[i] for each node center, plus W (width), H (height)
    - Objective: Minimize W + H (total area proxy)
    - Constraints: Node separation, relative ordering preservation

    If scipy is not available, falls back to greedy compaction.

    Args:
        boxes: Node boxes with current positions
        edges: Edges with routing information
        node_separation: Minimum horizontal gap between nodes
        layer_separation: Minimum vertical gap between layers
        edge_separation: Minimum gap for edge routing
        max_time: Maximum solver time in seconds

    Returns:
        ILPCompactionResult with new positions and dimensions
    """
    n = len(boxes)
    if n == 0:
        return ILPCompactionResult(
            node_positions=[],
            width=0.0,
            height=0.0,
            optimal=True,
            solver_status="empty_graph",
        )

    if not _HAS_SCIPY:
        # Fall back to greedy compaction
        from .compaction import compact_layout

        result = compact_layout(
            boxes=boxes,
            edges=edges,
            node_separation=node_separation,
            layer_separation=layer_separation,
            edge_separation=edge_separation,
        )
        return ILPCompactionResult(
            node_positions=result.node_positions,
            width=result.width,
            height=result.height,
            optimal=False,
            solver_status="scipy_unavailable",
        )

    return _solve_ilp_compaction(
        boxes=boxes,
        edges=edges,
        node_separation=node_separation,
        layer_separation=layer_separation,
        edge_separation=edge_separation,
        max_time=max_time,
    )


def _solve_ilp_compaction(
    boxes: list[NodeBox],
    edges: list[OrthogonalEdge],
    node_separation: float,
    layer_separation: float,
    edge_separation: float,
    max_time: float,
) -> ILPCompactionResult:
    """
    Solve the ILP compaction problem using scipy.

    Variables layout:
    - x[0..n-1]: x-coordinates of node centers
    - y[n..2n-1]: y-coordinates of node centers
    - W (index 2n): total width
    - H (index 2n+1): total height

    Total variables: 2n + 2
    """
    n = len(boxes)

    # Cache box properties
    box_x = [box.x for box in boxes]
    box_y = [box.y for box in boxes]
    box_width = [box.width for box in boxes]
    box_height = [box.height for box in boxes]

    # Collect parent-child edges (cross-layer edges) for centering objective.
    centering_edges: list[tuple[int, int]] = []
    for edge in edges:
        src, tgt = edge.source, edge.target
        if src < n and tgt < n and abs(box_y[src] - box_y[tgt]) > 1e-6:
            centering_edges.append((src, tgt))
    num_centering = len(centering_edges)

    # Number of variables: x coords + y coords + W + H + d_e (centering aux)
    num_vars = 2 * n + 2 + num_centering
    idx_W = 2 * n
    idx_H = 2 * n + 1
    idx_d_start = 2 * n + 2  # auxiliary variables d_e for |x_src - x_tgt|

    # Objective: minimize W + H + alpha * sum(d_e)
    # alpha balances area minimization vs parent-child centering.
    alpha = 0.5
    c = np.zeros(num_vars)
    c[idx_W] = 1.0
    c[idx_H] = 1.0
    for k in range(num_centering):
        c[idx_d_start + k] = alpha

    # Build constraint matrices
    # We need A_ub @ x <= b_ub for inequality constraints
    A_ub_rows = []
    b_ub_rows = []

    # Non-overlap constraints for ALL pairs of nodes.
    # For each pair (i,j), we choose a separation axis based on initial
    # relative positions: separate horizontally if they're mostly side-by-side,
    # vertically if they're mostly stacked, or both if ambiguous.
    for i in range(n):
        for j in range(i + 1, n):
            dx = box_x[j] - box_x[i]
            dy = box_y[j] - box_y[i]

            # Determine separation axis based on initial layout geometry
            # Horizontal separation needed: nodes on the same row (similar y)
            # Vertical separation needed: nodes in the same column (similar x)
            same_x = abs(dx) < 1e-6
            same_y = abs(dy) < 1e-6

            if same_x and same_y:
                # Identical position: separate in both dimensions
                h_gap = box_width[i] / 2 + node_separation + box_width[j] / 2
                row = np.zeros(num_vars)
                row[i] = 1.0
                row[j] = -1.0
                A_ub_rows.append(row)
                b_ub_rows.append(-h_gap)
                continue

            if same_x:
                # Same column: vertical separation
                a, b = (i, j) if dy > 0 else (j, i)
                gap = box_height[a] / 2 + layer_separation + box_height[b] / 2
                row = np.zeros(num_vars)
                row[n + a] = 1.0
                row[n + b] = -1.0
                A_ub_rows.append(row)
                b_ub_rows.append(-gap)
                continue

            if same_y:
                # Same row: horizontal separation
                a, b = (i, j) if dx > 0 else (j, i)
                gap = box_width[a] / 2 + node_separation + box_width[b] / 2
                row = np.zeros(num_vars)
                row[a] = 1.0
                row[b] = -1.0
                A_ub_rows.append(row)
                b_ub_rows.append(-gap)
                continue

            # Different x and y: choose axis that gives the larger ratio
            # (i.e., the axis along which they're more separated).
            # This preserves the relative layout direction.
            x_gap_needed = (box_width[i] + box_width[j]) / 2 + node_separation
            y_gap_needed = (box_height[i] + box_height[j]) / 2 + layer_separation
            x_ratio = abs(dx) / x_gap_needed
            y_ratio = abs(dy) / y_gap_needed

            if x_ratio >= y_ratio:
                # Primarily separated horizontally: add x-constraint
                a, b = (i, j) if dx > 0 else (j, i)
                gap = box_width[a] / 2 + edge_separation + box_width[b] / 2
                row = np.zeros(num_vars)
                row[a] = 1.0
                row[b] = -1.0
                A_ub_rows.append(row)
                b_ub_rows.append(-gap)
            else:
                # Primarily separated vertically: add y-constraint
                a, b = (i, j) if dy > 0 else (j, i)
                gap = box_height[a] / 2 + edge_separation + box_height[b] / 2
                row = np.zeros(num_vars)
                row[n + a] = 1.0
                row[n + b] = -1.0
                A_ub_rows.append(row)
                b_ub_rows.append(-gap)

    # Centering constraints: linearize |x_src - x_tgt| via auxiliary d_e.
    # d_e >= x_src - x_tgt  =>  x_src - x_tgt - d_e <= 0
    # d_e >= x_tgt - x_src  =>  x_tgt - x_src - d_e <= 0
    for k, (src, tgt) in enumerate(centering_edges):
        d_idx = idx_d_start + k
        # d_e >= x_src - x_tgt
        row = np.zeros(num_vars)
        row[src] = 1.0
        row[tgt] = -1.0
        row[d_idx] = -1.0
        A_ub_rows.append(row)
        b_ub_rows.append(0.0)
        # d_e >= x_tgt - x_src
        row = np.zeros(num_vars)
        row[tgt] = 1.0
        row[src] = -1.0
        row[d_idx] = -1.0
        A_ub_rows.append(row)
        b_ub_rows.append(0.0)

    # Edge-based constraints: connected nodes in different layers must maintain
    # layer_separation vertically to preserve hierarchical structure.
    for edge in edges:
        src, tgt = edge.source, edge.target
        if src >= n or tgt >= n:
            continue
        dy = box_y[tgt] - box_y[src]
        if abs(dy) < 1e-6:
            continue  # Same layer
        if dy > 0:
            top_idx, bottom_idx = src, tgt
        else:
            top_idx, bottom_idx = tgt, src
        gap = box_height[top_idx] / 2 + layer_separation + box_height[bottom_idx] / 2
        row = np.zeros(num_vars)
        row[n + top_idx] = 1.0  # y_top
        row[n + bottom_idx] = -1.0  # -y_bottom
        A_ub_rows.append(row)
        b_ub_rows.append(-gap)

    # Constraint 3: W bounds all x coordinates
    # For each i: x[i] + w[i]/2 <= W - margin
    # => x[i] - W <= -w[i]/2 - margin
    margin = node_separation
    for i in range(n):
        row = np.zeros(num_vars)
        row[i] = 1.0  # x_i
        row[idx_W] = -1.0  # -W
        A_ub_rows.append(row)
        b_ub_rows.append(-box_width[i] / 2 - margin)

    # Constraint 4: H bounds all y coordinates
    # For each i: y[i] + h[i]/2 <= H - margin
    for i in range(n):
        row = np.zeros(num_vars)
        row[n + i] = 1.0  # y_i
        row[idx_H] = -1.0  # -H
        A_ub_rows.append(row)
        b_ub_rows.append(-box_height[i] / 2 - margin)

    # Constraint 5: All coordinates positive with margin
    # x[i] >= w[i]/2 + margin => -x[i] <= -w[i]/2 - margin
    for i in range(n):
        row = np.zeros(num_vars)
        row[i] = -1.0
        A_ub_rows.append(row)
        b_ub_rows.append(-box_width[i] / 2 - margin)

    # y[i] >= h[i]/2 + margin => -y[i] <= -h[i]/2 - margin
    for i in range(n):
        row = np.zeros(num_vars)
        row[n + i] = -1.0
        A_ub_rows.append(row)
        b_ub_rows.append(-box_height[i] / 2 - margin)

    # Build constraint matrix
    if A_ub_rows:
        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_rows)
    else:
        A_ub = np.zeros((1, num_vars))
        b_ub = np.zeros(1)

    # Variable bounds
    # x, y coordinates: [0, inf)
    # W, H: [min_size, inf)
    min_w = max(box_width) + 2 * margin if box_width else margin
    min_h = max(box_height) + 2 * margin if box_height else margin

    lb = np.zeros(num_vars)
    ub = np.full(num_vars, np.inf)
    lb[idx_W] = min_w
    lb[idx_H] = min_h

    bounds = Bounds(lb=lb, ub=ub)

    # Create linear constraint
    constraints = LinearConstraint(A_ub, -np.inf, b_ub)

    # Solve the LP (continuous relaxation - positions don't need to be integers)
    try:
        result = milp(
            c=c,
            constraints=constraints,
            bounds=bounds,
            options={"time_limit": max_time},
        )

        if result.success:
            x_vals = result.x
            positions = [(x_vals[i], x_vals[n + i]) for i in range(n)]
            width = x_vals[idx_W]
            height = x_vals[idx_H]

            return ILPCompactionResult(
                node_positions=positions,
                width=float(width),
                height=float(height),
                optimal=True,
                solver_status="optimal",
            )
        else:
            # Solver failed, fall back to greedy
            from .compaction import compact_layout

            fallback = compact_layout(
                boxes=boxes,
                edges=edges,
                node_separation=node_separation,
                layer_separation=layer_separation,
                edge_separation=edge_separation,
            )
            return ILPCompactionResult(
                node_positions=fallback.node_positions,
                width=fallback.width,
                height=fallback.height,
                optimal=False,
                solver_status=f"solver_failed: {result.message}",
            )

    except Exception as e:
        # Any solver error, fall back to greedy
        from .compaction import compact_layout

        fallback = compact_layout(
            boxes=boxes,
            edges=edges,
            node_separation=node_separation,
            layer_separation=layer_separation,
            edge_separation=edge_separation,
        )
        return ILPCompactionResult(
            node_positions=fallback.node_positions,
            width=fallback.width,
            height=fallback.height,
            optimal=False,
            solver_status=f"exception: {e}",
        )


__all__ = [
    "ILPCompactionResult",
    "compact_layout_ilp",
    "is_scipy_available",
]
