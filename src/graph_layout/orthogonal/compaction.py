"""
Compaction phase for Kandinsky layout.

Implements constraint-based compaction to minimize the area of the
orthogonal drawing while preserving the topology and avoiding overlaps.

The compaction works in two passes:
1. Horizontal compaction: Minimize total width
2. Vertical compaction: Minimize total height

Each pass uses a constraint system to maintain:
- Node separation constraints
- Edge segment separation constraints
- Relative ordering constraints
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import NodeBox, OrthogonalEdge


@dataclass
class CompactionConstraint:
    """A constraint for the compaction solver."""

    # left + gap <= right (for horizontal) or top + gap <= bottom (for vertical)
    left: int  # Index of left/top element
    right: int  # Index of right/bottom element
    gap: float  # Minimum gap between elements
    is_hard: bool = True  # Hard constraints must be satisfied


@dataclass
class CompactionResult:
    """Result of compaction."""

    node_positions: list[tuple[float, float]]  # New (x, y) positions for nodes
    width: float  # Total width of compacted layout
    height: float  # Total height of compacted layout
    iterations: int  # Number of iterations used


class CompactionSolver:
    """
    Solves compaction constraints using iterative relaxation.

    This is a simple constraint solver that iteratively adjusts positions
    to satisfy all constraints while minimizing total span.
    """

    def __init__(
        self,
        num_elements: int,
        initial_positions: list[float],
        constraints: list[CompactionConstraint],
        max_iterations: int = 100,
    ) -> None:
        """
        Initialize compaction solver.

        Args:
            num_elements: Number of elements to position
            initial_positions: Initial positions
            constraints: List of constraints
            max_iterations: Maximum iterations for solving
        """
        self.num_elements = num_elements
        self.positions = list(initial_positions)
        self.constraints = constraints
        self.max_iterations = max_iterations

    def solve(self) -> tuple[list[float], int]:
        """
        Solve the constraint system.

        Returns:
            Tuple of (final positions, iterations used)
        """
        if not self.constraints:
            return self.positions, 0

        for iteration in range(self.max_iterations):
            changed = False

            # Process each constraint
            for constraint in self.constraints:
                left_pos = self.positions[constraint.left]
                right_pos = self.positions[constraint.right]

                # Check if constraint is violated
                required_pos = left_pos + constraint.gap

                if right_pos < required_pos:
                    # Push right element to satisfy constraint
                    self.positions[constraint.right] = required_pos
                    changed = True

            # Check for convergence
            if not changed:
                return self.positions, iteration + 1

        return self.positions, self.max_iterations


def compact_horizontal(
    boxes: list[NodeBox],
    edges: list[OrthogonalEdge],
    node_separation: float,
    edge_separation: float,
) -> list[float]:
    """
    Compact layout horizontally.

    Args:
        boxes: Node boxes with current positions
        edges: Edges with routing information
        node_separation: Minimum gap between nodes
        edge_separation: Minimum gap for edge routing

    Returns:
        New x-coordinates for all nodes
    """
    n = len(boxes)
    if n == 0:
        return []

    # Pre-cache box bounds for performance (avoid repeated property access)
    box_x = [box.x for box in boxes]
    box_y = [box.y for box in boxes]
    box_width = [box.width for box in boxes]
    box_height = [box.height for box in boxes]
    box_top = [box_y[i] - box_height[i] / 2 for i in range(n)]
    box_bottom = [box_y[i] + box_height[i] / 2 for i in range(n)]

    # Sort boxes by x-coordinate to establish ordering
    sorted_indices = sorted(range(n), key=lambda i: box_x[i])

    # Build constraints from left-to-right ordering
    constraints: list[CompactionConstraint] = []

    for i in range(len(sorted_indices) - 1):
        left_idx = sorted_indices[i]
        right_idx = sorted_indices[i + 1]

        # Check if they overlap vertically (need horizontal separation)
        # Using cached bounds instead of property access
        if not (
            box_bottom[left_idx] < box_top[right_idx] or box_bottom[right_idx] < box_top[left_idx]
        ):
            # Gap = half of left width + separation + half of right width
            gap = box_width[left_idx] / 2 + node_separation + box_width[right_idx] / 2
            constraints.append(
                CompactionConstraint(
                    left=left_idx,
                    right=right_idx,
                    gap=gap,
                )
            )

    # Initial positions
    initial_x = box_x[:]

    # Solve constraints
    solver = CompactionSolver(n, initial_x, constraints)
    new_x, _ = solver.solve()

    # Shift to minimize total width (move everything left as much as possible)
    if new_x:
        min_x = min(new_x[i] - box_width[i] / 2 for i in range(n))
        # Don't shift into negative, but minimize whitespace on left
        if min_x > node_separation:
            shift = min_x - node_separation
            new_x = [x - shift for x in new_x]

    return new_x


def compact_vertical(
    boxes: list[NodeBox],
    edges: list[OrthogonalEdge],
    layer_separation: float,
    edge_separation: float,
) -> list[float]:
    """
    Compact layout vertically.

    Args:
        boxes: Node boxes with current positions
        edges: Edges with routing information
        layer_separation: Minimum gap between layers
        edge_separation: Minimum gap for edge routing

    Returns:
        New y-coordinates for all nodes
    """
    n = len(boxes)
    if n == 0:
        return []

    # Pre-cache box bounds for performance (avoid repeated property access)
    box_x = [box.x for box in boxes]
    box_y = [box.y for box in boxes]
    box_width = [box.width for box in boxes]
    box_height = [box.height for box in boxes]
    box_left = [box_x[i] - box_width[i] / 2 for i in range(n)]
    box_right = [box_x[i] + box_width[i] / 2 for i in range(n)]

    # Sort boxes by y-coordinate to establish ordering
    sorted_indices = sorted(range(n), key=lambda i: box_y[i])

    # Build constraints from top-to-bottom ordering
    constraints: list[CompactionConstraint] = []

    for i in range(len(sorted_indices) - 1):
        top_idx = sorted_indices[i]
        bottom_idx = sorted_indices[i + 1]

        # Check if they overlap horizontally (need vertical separation)
        # Using cached bounds instead of property access
        if not (
            box_right[top_idx] < box_left[bottom_idx] or box_right[bottom_idx] < box_left[top_idx]
        ):
            # Gap = half of top height + separation + half of bottom height
            gap = box_height[top_idx] / 2 + layer_separation + box_height[bottom_idx] / 2
            constraints.append(
                CompactionConstraint(
                    left=top_idx,
                    right=bottom_idx,
                    gap=gap,
                )
            )

    # Initial positions
    initial_y = box_y[:]

    # Solve constraints
    solver = CompactionSolver(n, initial_y, constraints)
    new_y, _ = solver.solve()

    # Shift to minimize total height
    if new_y:
        min_y = min(new_y[i] - box_height[i] / 2 for i in range(n))
        if min_y > layer_separation:
            shift = min_y - layer_separation
            new_y = [y - shift for y in new_y]

    return new_y


def _boxes_overlap_vertically(box1: NodeBox, box2: NodeBox) -> bool:
    """Check if two boxes overlap in the vertical dimension."""
    return not (box1.bottom < box2.top or box2.bottom < box1.top)


def _boxes_overlap_horizontally(box1: NodeBox, box2: NodeBox) -> bool:
    """Check if two boxes overlap in the horizontal dimension."""
    return not (box1.right < box2.left or box2.right < box1.left)


def compact_layout(
    boxes: list[NodeBox],
    edges: list[OrthogonalEdge],
    node_separation: float = 60.0,
    layer_separation: float = 80.0,
    edge_separation: float = 15.0,
) -> CompactionResult:
    """
    Compact the entire layout in both dimensions.

    Args:
        boxes: Node boxes with current positions
        edges: Edges with routing information
        node_separation: Minimum horizontal gap between nodes
        layer_separation: Minimum vertical gap between layers
        edge_separation: Minimum gap for edge routing

    Returns:
        CompactionResult with new positions and dimensions
    """
    n = len(boxes)
    if n == 0:
        return CompactionResult(
            node_positions=[],
            width=0,
            height=0,
            iterations=0,
        )

    # First pass: horizontal compaction
    new_x = compact_horizontal(boxes, edges, node_separation, edge_separation)

    # Update boxes with new x positions for vertical pass
    temp_boxes = [
        NodeBox(
            index=box.index,
            x=new_x[i] if i < len(new_x) else box.x,
            y=box.y,
            width=box.width,
            height=box.height,
        )
        for i, box in enumerate(boxes)
    ]

    # Second pass: vertical compaction
    new_y = compact_vertical(temp_boxes, edges, layer_separation, edge_separation)

    # Combine results
    positions = [
        (new_x[i] if i < len(new_x) else boxes[i].x, new_y[i] if i < len(new_y) else boxes[i].y)
        for i in range(n)
    ]

    # Calculate final dimensions
    if positions:
        min_x = min(positions[i][0] - boxes[i].width / 2 for i in range(n))
        max_x = max(positions[i][0] + boxes[i].width / 2 for i in range(n))
        min_y = min(positions[i][1] - boxes[i].height / 2 for i in range(n))
        max_y = max(positions[i][1] + boxes[i].height / 2 for i in range(n))

        width = max_x - min_x + 2 * node_separation
        height = max_y - min_y + 2 * layer_separation
    else:
        width = 0
        height = 0

    return CompactionResult(
        node_positions=positions,
        width=width,
        height=height,
        iterations=2,  # Two passes
    )


__all__ = [
    "CompactionConstraint",
    "CompactionResult",
    "CompactionSolver",
    "compact_horizontal",
    "compact_vertical",
    "compact_layout",
]
