"""
Common types for graph layout algorithms.

This module provides the fundamental types used across all layout algorithms:
- Node: Graph vertex with position and properties
- Link: Edge connecting two nodes
- Group: Hierarchical grouping of nodes
- EventType: Layout lifecycle events
- Event: Event payload for callbacks
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Callable, Generic, Optional, Sequence, TypedDict, TypeVar, Union


class EventType(IntEnum):
    """
    Layout lifecycle events.

    - start: Layout iterations have begun
    - tick: Fired once per iteration (for animation)
    - end: Layout has converged or stopped
    """

    start = 0
    tick = 1
    end = 2


class Event(TypedDict, total=False):
    """Event payload passed to event listeners."""

    type: EventType
    alpha: float
    stress: Optional[float]
    listener: Optional[Callable[[], None]]


class Node:
    """
    Graph node with position and properties.

    Attributes:
        index: Index in nodes array (set by layout)
        x: X coordinate (centroid)
        y: Y coordinate (centroid)
        width: Node width (for overlap avoidance)
        height: Node height (for overlap avoidance)
        fixed: Bit mask for fixed position (0 = free, nonzero = fixed)
        bounds: Bounding rectangle (set by layout algorithms)
        innerBounds: Inner bounding rectangle (for groups)
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize node with optional properties."""
        self.index: Optional[int] = kwargs.get("index")
        self.x: float = kwargs.get("x", 0.0)
        self.y: float = kwargs.get("y", 0.0)
        self.width: Optional[float] = kwargs.get("width")
        self.height: Optional[float] = kwargs.get("height")
        self.fixed: int = kwargs.get("fixed", 0)

        # Bounds (used by layout algorithms for overlap avoidance)
        self.bounds: Optional[Any] = kwargs.get("bounds")
        self.innerBounds: Optional[Any] = kwargs.get("innerBounds")

        # Internal: previous position (for drag operations)
        self.px: Optional[float] = None
        self.py: Optional[float] = None

        # Internal: parent group reference
        self.parent: Optional[Group] = None

        # Copy any additional custom properties
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def __repr__(self) -> str:
        return f"Node(index={self.index}, x={self.x:.2f}, y={self.y:.2f})"


class Link:
    """
    Edge connecting two nodes.

    Attributes:
        source: Source node or node index
        target: Target node or node index
        length: Ideal edge length (optional)
        weight: Edge weight/strength (optional)
    """

    def __init__(
        self,
        source: Union[Node, int],
        target: Union[Node, int],
        length: Optional[float] = None,
        weight: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize link between two nodes.

        Args:
            source: Source node or node index (required)
            target: Target node or node index (required)
            length: Ideal edge length (optional)
            weight: Edge weight/strength (optional)

        Raises:
            ValueError: If source or target is None
        """
        if source is None:
            raise ValueError("Link source cannot be None")
        if target is None:
            raise ValueError("Link target cannot be None")

        self.source = source
        self.target = target
        self.length = length
        self.weight = weight

        # Copy any additional custom properties
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def __repr__(self) -> str:
        if isinstance(self.source, int):
            src: Any = self.source
        else:
            src = getattr(self.source, "index", None)
        if isinstance(self.target, int):
            tgt: Any = self.target
        else:
            tgt = getattr(self.target, "index", None)
        return f"Link({src} -> {tgt})"


class Group:
    """
    Hierarchical group of nodes.

    Attributes:
        leaves: List of nodes (or indices) in this group
        groups: List of child groups (or indices)
        padding: Padding around group contents
        bounds: Bounding rectangle (set by layout algorithms)
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize group with optional properties."""
        self.leaves: Optional[list[Union[Node, int]]] = kwargs.get("leaves")
        self.groups: Optional[list[Union[Group, int]]] = kwargs.get("groups")
        self.padding: float = kwargs.get("padding", 1.0)
        self.parent: Optional[Group] = None
        self.index: Optional[int] = None

        # Bounds (used by layout algorithms)
        self.bounds: Optional[Any] = kwargs.get("bounds")

        # Copy any additional custom properties
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def __repr__(self) -> str:
        n_leaves = len(self.leaves) if self.leaves else 0
        n_groups = len(self.groups) if self.groups else 0
        return f"Group(leaves={n_leaves}, groups={n_groups})"


def is_group(obj: Any) -> bool:
    """Check if an object is a Group (has leaves or groups attribute)."""
    return hasattr(obj, "leaves") or hasattr(obj, "groups")


# Type aliases for callbacks
LinkNumericPropertyAccessor = Callable[[Link], float]


# Generic type variables for custom node/link types
NodeT = TypeVar("NodeT", bound=Node)
LinkT = TypeVar("LinkT", bound=Link)


class LinkAccessor(Generic[LinkT]):
    """Protocol for accessing link properties with custom link types."""

    @staticmethod
    def get_source_index(link: LinkT) -> int:
        """Get source node index from a link."""
        if isinstance(link.source, int):
            return link.source
        return link.source.index  # type: ignore

    @staticmethod
    def get_target_index(link: LinkT) -> int:
        """Get target node index from a link."""
        if isinstance(link.target, int):
            return link.target
        return link.target.index  # type: ignore


# Type aliases for Pythonic API
# These allow flexible input types while maintaining type safety
NodeLike = Union[Node, dict[str, Any], Any]
"""Input type for nodes: Node objects, dicts, or objects with node attributes."""

LinkLike = Union[Link, dict[str, Any], Any]
"""Input type for links: Link objects, dicts, or objects with source/target."""

GroupLike = Union[Group, dict[str, Any], Any]
"""Input type for groups: Group objects, dicts, or objects with leaves/groups."""

SizeType = Union[tuple[float, float], list[float], Sequence[float]]
"""Canvas size: (width, height) tuple, list, or sequence."""


__all__ = [
    "EventType",
    "Event",
    "Node",
    "Link",
    "Group",
    "is_group",
    "LinkNumericPropertyAccessor",
    "NodeT",
    "LinkT",
    "LinkAccessor",
    # Pythonic API type aliases
    "NodeLike",
    "LinkLike",
    "GroupLike",
    "SizeType",
]
