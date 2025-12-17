"""
Base classes for graph layout algorithms.

This module provides abstract base classes that define the common interface
and shared functionality for all layout algorithms:

- BaseLayout: Abstract base with event system, node/link management
- IterativeLayout: For animated layouts with tick loop (force-directed, etc.)
- StaticLayout: For single-pass layouts (circular, tree, etc.)
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

if TYPE_CHECKING:
    from typing_extensions import Self

from .types import (
    Event,
    EventType,
    Group,
    GroupLike,
    Link,
    LinkLike,
    Node,
    NodeLike,
    SizeType,
)
from .validation import (
    validate_canvas_size,
    validate_group_indices,
    validate_link_indices,
)


class BaseLayout(ABC):
    """
    Abstract base class for all layout algorithms.

    Provides shared infrastructure:
    - Event system (start/tick/end events)
    - Node/link/group management via properties
    - Position initialization
    - Canvas size management

    Example:
        layout = SomeLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
        )
        layout.run()

        # Access results via properties
        for node in layout.nodes:
            print(f"Node {node.index}: ({node.x}, {node.y})")
    """

    def __init__(
        self,
        *,
        nodes: Optional[Sequence[NodeLike]] = None,
        links: Optional[Sequence[LinkLike]] = None,
        groups: Optional[Sequence[GroupLike]] = None,
        size: SizeType = (1.0, 1.0),
        random_seed: Optional[int] = None,
        on_start: Optional[Callable[[Optional[Event]], None]] = None,
        on_tick: Optional[Callable[[Optional[Event]], None]] = None,
        on_end: Optional[Callable[[Optional[Event]], None]] = None,
    ) -> None:
        """
        Initialize layout with configuration.

        Args:
            nodes: List of nodes (Node objects, dicts, or objects with attributes)
            links: List of links (Link objects or dicts with source/target)
            groups: List of groups (Group objects or dicts)
            size: Canvas size as (width, height)
            random_seed: Random seed for reproducible layouts
            on_start: Callback for start event
            on_tick: Callback for tick event (iterative layouts)
            on_end: Callback for end event
        """
        self._nodes: list[Node] = []
        self._links: list[Link] = []
        self._groups: list[Group] = []
        self._canvas_size: tuple[float, float] = (1.0, 1.0)
        self._events: dict[EventType, Callable[[Optional[Event]], None]] = {}
        self._random_seed: Optional[int] = None

        # Set initial values via properties (triggers normalization)
        if nodes is not None:
            self.nodes = nodes
        if links is not None:
            self.links = links
        if groups is not None:
            self.groups = groups
        self.size = size
        if random_seed is not None:
            self.random_seed = random_seed

        # Register event callbacks
        if on_start:
            self._events[EventType.start] = on_start
        if on_tick:
            self._events[EventType.tick] = on_tick
        if on_end:
            self._events[EventType.end] = on_end

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def nodes(self) -> list[Node]:
        """Get the list of nodes."""
        return self._nodes

    @nodes.setter
    def nodes(self, value: Sequence[NodeLike]) -> None:
        """Set nodes from a sequence of Node objects, dicts, or objects."""
        self._nodes = []
        for node_data in value:
            if isinstance(node_data, Node):
                self._nodes.append(node_data)
            elif isinstance(node_data, dict):
                self._nodes.append(Node(**node_data))
            else:
                # Generic object - copy attributes
                node = Node()
                for attr in ["index", "x", "y", "width", "height", "fixed"]:
                    if hasattr(node_data, attr):
                        setattr(node, attr, getattr(node_data, attr))
                # Copy any additional attributes
                for attr in dir(node_data):
                    if not attr.startswith("_") and not hasattr(node, attr):
                        setattr(node, attr, getattr(node_data, attr))
                self._nodes.append(node)

    @property
    def links(self) -> list[Link]:
        """Get the list of links."""
        return self._links

    @links.setter
    def links(self, value: Sequence[LinkLike]) -> None:
        """Set links from a sequence of Link objects, dicts, or objects."""
        self._links = []
        for link_data in value:
            if isinstance(link_data, Link):
                self._links.append(link_data)
            elif isinstance(link_data, dict):
                self._links.append(Link(**link_data))
            else:
                # Generic object - extract source/target
                source = getattr(link_data, "source", 0)
                target = getattr(link_data, "target", 0)
                length = getattr(link_data, "length", None)
                weight = getattr(link_data, "weight", None)
                self._links.append(Link(source, target, length, weight))

    @property
    def groups(self) -> list[Group]:
        """Get the list of groups."""
        return self._groups

    @groups.setter
    def groups(self, value: Sequence[GroupLike]) -> None:
        """Set groups from a sequence of Group objects, dicts, or objects."""
        self._groups = []
        for group_data in value:
            if isinstance(group_data, Group):
                self._groups.append(group_data)
            elif isinstance(group_data, dict):
                self._groups.append(Group(**group_data))
            else:
                group = Group()
                for attr in ["leaves", "groups", "padding"]:
                    if hasattr(group_data, attr):
                        setattr(group, attr, getattr(group_data, attr))
                self._groups.append(group)

    @property
    def size(self) -> tuple[float, float]:
        """Get canvas size as (width, height)."""
        return self._canvas_size

    @size.setter
    def size(self, value: SizeType) -> None:
        """
        Set canvas size.

        Args:
            value: (width, height) tuple, list, or sequence

        Raises:
            InvalidCanvasSizeError: If width or height is not positive.
        """
        width, height = validate_canvas_size(value)
        self._canvas_size = (width, height)

    @property
    def random_seed(self) -> Optional[int]:
        """Get random seed for reproducible layouts."""
        return self._random_seed

    @random_seed.setter
    def random_seed(self, value: Optional[int]) -> None:
        """Set random seed for reproducible layouts."""
        self._random_seed = value

    # -------------------------------------------------------------------------
    # Event System
    # -------------------------------------------------------------------------

    def on(self, event: EventType | str, callback: Callable[[Optional[Event]], None]) -> Self:
        """
        Subscribe to a layout event.

        Args:
            event: Event type (EventType enum or string name)
            callback: Function to call when event fires

        Returns:
            self (for chaining)
        """
        if isinstance(event, str):
            event = EventType[event]
        self._events[event] = callback
        return self

    def trigger(self, event: Event) -> None:
        """
        Trigger an event, calling the registered callback.

        Args:
            event: Event payload with type and optional data
        """
        event_type = event.get("type")
        if event_type is not None and event_type in self._events:
            self._events[event_type](event)

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate(self) -> Self:
        """
        Validate current configuration.

        Checks that all link/group references point to valid node indices.
        Called automatically by run() but can be called early for fail-fast
        behavior.

        Returns:
            self (for chaining)

        Raises:
            InvalidLinkError: If any link references an invalid node index.
            InvalidGroupError: If any group references an invalid node/group index.
        """
        if self._links:
            validate_link_indices(self._links, len(self._nodes), strict=True)
        if self._groups:
            validate_group_indices(self._groups, len(self._nodes), strict=True)
        return self

    # -------------------------------------------------------------------------
    # Lifecycle Methods
    # -------------------------------------------------------------------------

    @abstractmethod
    def run(self, **kwargs: Any) -> Self:
        """
        Run the layout algorithm.

        This is the main entry point. Implementations should:
        1. Initialize node positions if needed
        2. Run the layout algorithm
        3. Fire appropriate events

        Returns:
            self (for chaining)
        """
        pass

    def stop(self) -> Self:
        """
        Stop the layout (for iterative layouts).

        Returns:
            self (for chaining)
        """
        return self

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def _initialize_indices(self) -> None:
        """Assign indices to nodes that don't have them."""
        for i, node in enumerate(self._nodes):
            if node.index is None:
                node.index = i

    def _initialize_positions(
        self, random_init: bool = True, center: Optional[tuple[float, float]] = None
    ) -> None:
        """
        Initialize node positions.

        Fixed nodes (node.fixed != 0) retain their original positions.

        Args:
            random_init: If True, randomize positions for non-fixed nodes.
            center: Center point for initialization. Defaults to canvas center.
        """
        if center is None:
            center = (self._canvas_size[0] / 2, self._canvas_size[1] / 2)

        if self._random_seed is not None:
            random.seed(self._random_seed)

        w, h = self._canvas_size
        for node in self._nodes:
            # Skip fixed nodes - preserve their positions
            if node.fixed:
                continue

            if random_init or (node.x == 0.0 and node.y == 0.0):
                # Initialize to random position within canvas
                node.x = random.uniform(0, w)
                node.y = random.uniform(0, h)

    def _center_graph(self) -> None:
        """Center the graph within the canvas."""
        if not self._nodes:
            return

        # Find bounding box
        min_x = min(n.x for n in self._nodes)
        max_x = max(n.x for n in self._nodes)
        min_y = min(n.y for n in self._nodes)
        max_y = max(n.y for n in self._nodes)

        # Calculate offset to center
        graph_cx = (min_x + max_x) / 2
        graph_cy = (min_y + max_y) / 2
        canvas_cx = self._canvas_size[0] / 2
        canvas_cy = self._canvas_size[1] / 2

        dx = canvas_cx - graph_cx
        dy = canvas_cy - graph_cy

        # Translate all nodes
        for node in self._nodes:
            node.x += dx
            node.y += dy

    def _get_source_index(self, link: Link) -> int:
        """Get source node index from a link."""
        if isinstance(link.source, int):
            return link.source
        return link.source.index if link.source.index is not None else 0

    def _get_target_index(self, link: Link) -> int:
        """Get target node index from a link."""
        if isinstance(link.target, int):
            return link.target
        return link.target.index if link.target.index is not None else 0

    def _build_adjacency(self) -> list[list[int]]:
        """Build adjacency list from links.

        Invalid indices (out of bounds) are silently skipped to prevent crashes.
        Use validate() first to ensure all indices are valid.
        """
        n = len(self._nodes)
        adj: list[list[int]] = [[] for _ in self._nodes]
        for link in self._links:
            src = self._get_source_index(link)
            tgt = self._get_target_index(link)
            # Skip invalid indices to prevent IndexError
            if 0 <= src < n and 0 <= tgt < n:
                adj[src].append(tgt)
                adj[tgt].append(src)
        return adj


class IterativeLayout(BaseLayout):
    """
    Base class for iterative/animated layout algorithms.

    Provides:
    - Alpha (temperature/cooling) management
    - Tick-based iteration loop
    - Convergence checking

    Example:
        layout = SomeForceLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
            iterations=300,
            alpha_min=0.001,
        )
        layout.run()
    """

    def __init__(
        self,
        *,
        nodes: Optional[Sequence[NodeLike]] = None,
        links: Optional[Sequence[LinkLike]] = None,
        groups: Optional[Sequence[GroupLike]] = None,
        size: SizeType = (1.0, 1.0),
        random_seed: Optional[int] = None,
        on_start: Optional[Callable[[Optional[Event]], None]] = None,
        on_tick: Optional[Callable[[Optional[Event]], None]] = None,
        on_end: Optional[Callable[[Optional[Event]], None]] = None,
        # IterativeLayout-specific parameters
        alpha: float = 1.0,
        alpha_min: float = 0.001,
        alpha_decay: float = 0.99,
        iterations: int = 300,
    ) -> None:
        """
        Initialize iterative layout.

        Args:
            nodes: List of nodes
            links: List of links
            groups: List of groups
            size: Canvas size as (width, height)
            random_seed: Random seed for reproducible layouts
            on_start: Callback for start event
            on_tick: Callback for tick event
            on_end: Callback for end event
            alpha: Initial alpha/temperature (0 to 1)
            alpha_min: Minimum alpha for convergence threshold
            alpha_decay: Alpha decay rate per iteration (0 to 1)
            iterations: Maximum number of iterations
        """
        super().__init__(
            nodes=nodes,
            links=links,
            groups=groups,
            size=size,
            random_seed=random_seed,
            on_start=on_start,
            on_tick=on_tick,
            on_end=on_end,
        )
        self._alpha: float = max(0.0, min(1.0, float(alpha)))
        self._alpha_min: float = float(alpha_min)
        self._alpha_decay: float = max(0.0, min(1.0, float(alpha_decay)))
        self._running: bool = False
        self._iterations: int = max(1, int(iterations))

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def alpha(self) -> float:
        """Get current alpha (temperature/energy)."""
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        """Set alpha (temperature/energy), clamped to [0, 1]."""
        self._alpha = max(0.0, min(1.0, float(value)))

    @property
    def alpha_min(self) -> float:
        """Get minimum alpha (convergence threshold)."""
        return self._alpha_min

    @alpha_min.setter
    def alpha_min(self, value: float) -> None:
        """Set minimum alpha (convergence threshold)."""
        self._alpha_min = float(value)

    @property
    def alpha_decay(self) -> float:
        """Get alpha decay rate."""
        return self._alpha_decay

    @alpha_decay.setter
    def alpha_decay(self, value: float) -> None:
        """Set alpha decay rate, clamped to [0, 1]."""
        self._alpha_decay = max(0.0, min(1.0, float(value)))

    @property
    def iterations(self) -> int:
        """Get maximum iterations."""
        return self._iterations

    @iterations.setter
    def iterations(self, value: int) -> None:
        """Set maximum iterations (minimum 1)."""
        self._iterations = max(1, int(value))

    # -------------------------------------------------------------------------
    # Lifecycle Methods
    # -------------------------------------------------------------------------

    @abstractmethod
    def tick(self) -> bool:
        """
        Perform one iteration of the layout.

        Returns:
            True if converged/done, False if more iterations needed.
        """
        pass

    def kick(self) -> None:
        """Run tick() repeatedly until convergence or max iterations."""
        for _ in range(self._iterations):
            if self.tick():
                break

    def resume(self) -> Self:
        """Resume layout with alpha reset to 0.1."""
        self._alpha = 0.1
        self._running = True
        self.trigger({"type": EventType.start, "alpha": self._alpha})
        self.kick()
        return self

    def stop(self) -> Self:
        """Stop the layout."""
        self._alpha = 0.0
        self._running = False
        return self


class StaticLayout(BaseLayout):
    """
    Base class for single-pass layout algorithms.

    These layouts compute positions in one pass without iteration.
    Examples: circular, tree, spectral layouts.

    Example:
        layout = CircularLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
            radius=200,
        )
        layout.run()
    """

    def run(self, **kwargs: Any) -> Self:
        """
        Run the layout algorithm.

        Fires start event, computes layout, fires end event.

        Args:
            **kwargs: Additional arguments passed to _compute()

        Returns:
            self (for chaining)
        """
        self._initialize_indices()
        self.trigger({"type": EventType.start, "alpha": 1.0})

        # Subclasses implement _compute()
        self._compute(**kwargs)

        if kwargs.get("center_graph", True):
            self._center_graph()

        self.trigger({"type": EventType.end, "alpha": 0.0})
        return self

    @abstractmethod
    def _compute(self, **kwargs: Any) -> None:
        """
        Compute node positions.

        Subclasses must implement this to perform the actual layout computation.
        """
        pass


__all__ = [
    "BaseLayout",
    "IterativeLayout",
    "StaticLayout",
]
