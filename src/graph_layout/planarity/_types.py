"""Internal data structures for LR-planarity testing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Edge type: directed edge tuple (source, target)
Edge = Optional[tuple[int, int]]


@dataclass
class PlanarityResult:
    """Result of a planarity test.

    Attributes:
        is_planar: Whether the graph is planar.
        embedding: If planar, a dict mapping each vertex to its clockwise
            neighbor ordering (rotation system). None if non-planar.
        kuratowski_edges: Reserved for Phase 2 -- edges forming a Kuratowski
            subgraph (K5 or K3,3) when non-planar. Currently always None.
        kuratowski_type: Reserved for Phase 2 -- "K5", "K3,3", or None.
    """

    is_planar: bool
    embedding: Optional[dict[int, list[int]]] = None
    kuratowski_edges: Optional[list[tuple[int, int]]] = None
    kuratowski_type: Optional[str] = None


@dataclass
class Interval:
    """An interval of back edges in the LR-planarity algorithm.

    low and high are back edge tuples (v, w) where v is a descendant
    and w is an ancestor. None means the interval is empty.
    """

    low: Edge = None
    high: Edge = None

    def empty(self) -> bool:
        return self.low is None and self.high is None

    def copy(self) -> Interval:
        return Interval(low=self.low, high=self.high)


@dataclass
class ConflictPair:
    """A left/right interval pair on the constraint stack.

    During LR testing, each back edge creates constraints that are tracked
    as pairs of intervals. When constraints conflict (cannot be assigned
    to left or right without overlap), the graph is non-planar.
    """

    left: Interval = field(default_factory=Interval)
    right: Interval = field(default_factory=Interval)

    def swap(self) -> None:
        self.left, self.right = self.right, self.left
