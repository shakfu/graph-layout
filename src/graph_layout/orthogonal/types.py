"""
Type definitions for orthogonal layout algorithms.

Provides data structures for representing orthogonal graph drawings,
including ports, faces, and orthogonal representations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Side(Enum):
    """Side of a node where edges can connect."""

    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"

    def opposite(self) -> Side:
        """Get the opposite side."""
        opposites = {
            Side.NORTH: Side.SOUTH,
            Side.SOUTH: Side.NORTH,
            Side.EAST: Side.WEST,
            Side.WEST: Side.EAST,
        }
        return opposites[self]

    def is_horizontal(self) -> bool:
        """Check if this side is on a horizontal edge of the node."""
        return self in (Side.NORTH, Side.SOUTH)

    def is_vertical(self) -> bool:
        """Check if this side is on a vertical edge of the node."""
        return self in (Side.EAST, Side.WEST)


class BendDirection(Enum):
    """Direction of a bend in an edge."""

    LEFT = 1  # 90° counter-clockwise turn
    RIGHT = -1  # 90° clockwise turn


@dataclass
class Port:
    """
    A connection point on a node side.

    Ports represent where edges attach to nodes. Multiple edges
    can share a side but have different ports.
    """

    node: int  # Node index
    side: Side  # Which side of the node
    position: float = 0.5  # Position along side (0.0 to 1.0)
    edge: int | None = None  # Edge index using this port

    def __hash__(self) -> int:
        return hash((self.node, self.side, self.position, self.edge))


@dataclass
class EdgeSegment:
    """
    A segment of an orthogonal edge.

    Orthogonal edges consist of horizontal and vertical segments.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    is_horizontal: bool

    @property
    def length(self) -> float:
        """Get segment length."""
        if self.is_horizontal:
            return abs(self.x2 - self.x1)
        return abs(self.y2 - self.y1)


@dataclass
class OrthogonalEdge:
    """
    An edge in orthogonal representation.

    Contains the routing information including source/target ports
    and any bends in between.
    """

    source: int  # Source node index
    target: int  # Target node index
    source_port: Port  # Port on source node
    target_port: Port  # Port on target node
    bends: list[tuple[float, float]] = field(default_factory=list)  # Bend coordinates
    segments: list[EdgeSegment] = field(default_factory=list)  # Computed segments


@dataclass
class NodeBox:
    """
    A node represented as a box in orthogonal layout.

    Nodes are rectangles with edges connecting to their sides.
    """

    index: int
    x: float  # Center x
    y: float  # Center y
    width: float
    height: float

    @property
    def left(self) -> float:
        """Left edge x coordinate."""
        return self.x - self.width / 2

    @property
    def right(self) -> float:
        """Right edge x coordinate."""
        return self.x + self.width / 2

    @property
    def top(self) -> float:
        """Top edge y coordinate."""
        return self.y - self.height / 2

    @property
    def bottom(self) -> float:
        """Bottom edge y coordinate."""
        return self.y + self.height / 2

    def get_port_position(self, side: Side, offset: float = 0.5) -> tuple[float, float]:
        """
        Get the (x, y) position of a port on this node.

        Args:
            side: Which side of the node
            offset: Position along the side (0.0 to 1.0)

        Returns:
            (x, y) coordinates of the port
        """
        if side == Side.NORTH:
            return (self.left + self.width * offset, self.top)
        elif side == Side.SOUTH:
            return (self.left + self.width * offset, self.bottom)
        elif side == Side.WEST:
            return (self.left, self.top + self.height * offset)
        else:  # EAST
            return (self.right, self.top + self.height * offset)


@dataclass
class GridCell:
    """A cell in the routing grid."""

    row: int
    col: int
    occupied: bool = False
    node: int | None = None  # Node index if occupied by a node


@dataclass
class RoutingGrid:
    """
    Grid for edge routing in orthogonal layout.

    The grid is used to route edges while avoiding nodes and other edges.
    """

    rows: int
    cols: int
    cell_width: float
    cell_height: float
    origin_x: float = 0.0
    origin_y: float = 0.0
    cells: list[list[GridCell]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize grid cells."""
        if not self.cells:
            self.cells = [
                [GridCell(row=r, col=c) for c in range(self.cols)] for r in range(self.rows)
            ]

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        """Convert grid coordinates to world coordinates."""
        x = self.origin_x + col * self.cell_width
        y = self.origin_y + row * self.cell_height
        return (x, y)

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        col = int((x - self.origin_x) / self.cell_width)
        row = int((y - self.origin_y) / self.cell_height)
        # Clamp to valid range
        col = max(0, min(col, self.cols - 1))
        row = max(0, min(row, self.rows - 1))
        return (row, col)

    def is_valid(self, row: int, col: int) -> bool:
        """Check if grid position is valid."""
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_free(self, row: int, col: int) -> bool:
        """Check if grid cell is free for routing."""
        if not self.is_valid(row, col):
            return False
        return not self.cells[row][col].occupied

    def mark_occupied(self, row: int, col: int, node: int | None = None) -> None:
        """Mark a cell as occupied."""
        if self.is_valid(row, col):
            self.cells[row][col].occupied = True
            self.cells[row][col].node = node


__all__ = [
    "Side",
    "BendDirection",
    "Port",
    "EdgeSegment",
    "OrthogonalEdge",
    "NodeBox",
    "GridCell",
    "RoutingGrid",
]
