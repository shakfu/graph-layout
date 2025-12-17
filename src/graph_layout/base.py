"""
Base classes for graph layout algorithms.

This module provides abstract base classes that define the common interface
and shared functionality for all layout algorithms:

- BaseLayout: Abstract base with event system, node/link management, fluent API
- IterativeLayout: For animated layouts with tick loop (force-directed, etc.)
- StaticLayout: For single-pass layouts (circular, tree, etc.)
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union

from typing_extensions import Self

from .types import Event, EventType, Group, Link, Node


class BaseLayout(ABC):
    """
    Abstract base class for all layout algorithms.

    Provides shared infrastructure:
    - Event system (start/tick/end events)
    - Fluent configuration API (method chaining)
    - Node/link/group management
    - Position initialization
    - Canvas size management
    """

    def __init__(self) -> None:
        """Initialize layout with default parameters."""
        self._nodes: list[Node] = []
        self._links: list[Link] = []
        self._groups: list[Group] = []
        self._canvas_size: list[float] = [1.0, 1.0]
        self._events: dict[EventType, Callable[[Optional[Event]], None]] = {}
        self._random_seed: Optional[int] = None

    # -------------------------------------------------------------------------
    # Fluent Configuration API
    # -------------------------------------------------------------------------

    def nodes(self, v: Optional[list[Any]] = None) -> Union[list[Node], Self]:
        """
        Get or set nodes.

        Args:
            v: List of nodes (Node objects, dicts, or objects with attributes).
               If None, returns current nodes.

        Returns:
            Current nodes (if v is None) or self (for chaining).
        """
        if v is None:
            return self._nodes

        self._nodes = []
        for node_data in v:
            if isinstance(node_data, Node):
                self._nodes.append(node_data)
            elif isinstance(node_data, dict):
                self._nodes.append(Node(**node_data))
            else:
                # Generic object - copy attributes
                node = Node()
                for attr in ['index', 'x', 'y', 'width', 'height', 'fixed']:
                    if hasattr(node_data, attr):
                        setattr(node, attr, getattr(node_data, attr))
                # Copy any additional attributes
                for attr in dir(node_data):
                    if not attr.startswith('_') and not hasattr(node, attr):
                        setattr(node, attr, getattr(node_data, attr))
                self._nodes.append(node)
        return self

    def links(self, v: Optional[list[Any]] = None) -> Union[list[Link], Self]:
        """
        Get or set links.

        Args:
            v: List of links (Link objects or dicts with source/target).
               If None, returns current links.

        Returns:
            Current links (if v is None) or self (for chaining).
        """
        if v is None:
            return self._links

        self._links = []
        for link_data in v:
            if isinstance(link_data, Link):
                self._links.append(link_data)
            elif isinstance(link_data, dict):
                self._links.append(Link(**link_data))
            else:
                # Generic object - extract source/target
                source = getattr(link_data, 'source', 0)
                target = getattr(link_data, 'target', 0)
                length = getattr(link_data, 'length', None)
                weight = getattr(link_data, 'weight', None)
                self._links.append(Link(source, target, length, weight))
        return self

    def groups(self, v: Optional[list[Any]] = None) -> Union[list[Group], Self]:
        """
        Get or set groups.

        Args:
            v: List of groups (Group objects or dicts).
               If None, returns current groups.

        Returns:
            Current groups (if v is None) or self (for chaining).
        """
        if v is None:
            return self._groups

        self._groups = []
        for group_data in v:
            if isinstance(group_data, Group):
                self._groups.append(group_data)
            elif isinstance(group_data, dict):
                self._groups.append(Group(**group_data))
            else:
                group = Group()
                for attr in ['leaves', 'groups', 'padding']:
                    if hasattr(group_data, attr):
                        setattr(group, attr, getattr(group_data, attr))
                self._groups.append(group)
        return self

    def size(self, v: Optional[list[float]] = None) -> Union[list[float], Self]:
        """
        Get or set canvas size.

        Args:
            v: [width, height] of the layout area.
               If None, returns current size.

        Returns:
            Current size (if v is None) or self (for chaining).
        """
        if v is None:
            return self._canvas_size
        self._canvas_size = [float(v[0]), float(v[1])]
        return self

    def random_seed(self, seed: Optional[int] = None) -> Union[Optional[int], Self]:
        """
        Get or set random seed for reproducible layouts.

        Args:
            seed: Random seed value. If None (getter), returns current seed.

        Returns:
            Current seed (if called as getter) or self (for chaining).
        """
        if seed is None:
            return self._random_seed
        self._random_seed = seed
        return self

    # -------------------------------------------------------------------------
    # Event System
    # -------------------------------------------------------------------------

    def on(
        self,
        event: Union[EventType, str],
        callback: Callable[[Optional[Event]], None]
    ) -> Self:
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
        event_type = event.get('type')
        if event_type is not None and event_type in self._events:
            self._events[event_type](event)

    # -------------------------------------------------------------------------
    # Lifecycle Methods
    # -------------------------------------------------------------------------

    @abstractmethod
    def start(self, **kwargs: Any) -> Self:
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
        self,
        random_init: bool = True,
        center: Optional[tuple[float, float]] = None
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
        """Build adjacency list from links."""
        adj: list[list[int]] = [[] for _ in self._nodes]
        for link in self._links:
            src = self._get_source_index(link)
            tgt = self._get_target_index(link)
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
    """

    def __init__(self) -> None:
        super().__init__()
        self._alpha: float = 1.0
        self._alpha_min: float = 0.001
        self._alpha_decay: float = 0.99
        self._running: bool = False
        self._iterations: int = 300

    def alpha(self, v: Optional[float] = None) -> Union[float, Self]:
        """
        Get or set alpha (temperature/energy).

        Args:
            v: Alpha value (0 to 1). If None, returns current alpha.

        Returns:
            Current alpha (if v is None) or self (for chaining).
        """
        if v is None:
            return self._alpha
        self._alpha = max(0.0, min(1.0, float(v)))
        return self

    def alpha_min(self, v: Optional[float] = None) -> Union[float, Self]:
        """
        Get or set minimum alpha (convergence threshold).

        Args:
            v: Minimum alpha value. If None, returns current value.

        Returns:
            Current alpha_min (if v is None) or self (for chaining).
        """
        if v is None:
            return self._alpha_min
        self._alpha_min = float(v)
        return self

    def alpha_decay(self, v: Optional[float] = None) -> Union[float, Self]:
        """
        Get or set alpha decay rate.

        Args:
            v: Decay multiplier (0 to 1). If None, returns current value.

        Returns:
            Current alpha_decay (if v is None) or self (for chaining).
        """
        if v is None:
            return self._alpha_decay
        self._alpha_decay = max(0.0, min(1.0, float(v)))
        return self

    def iterations(self, v: Optional[int] = None) -> Union[int, Self]:
        """
        Get or set maximum iterations.

        Args:
            v: Maximum iterations. If None, returns current value.

        Returns:
            Current iterations (if v is None) or self (for chaining).
        """
        if v is None:
            return self._iterations
        self._iterations = max(1, int(v))
        return self

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
        self.trigger({'type': EventType.start, 'alpha': self._alpha})
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
    """

    def start(self, **kwargs: Any) -> Self:
        """
        Run the layout algorithm.

        Fires start event, computes layout, fires end event.

        Returns:
            self (for chaining)
        """
        self._initialize_indices()
        self.trigger({'type': EventType.start, 'alpha': 1.0})

        # Subclasses implement _compute()
        self._compute(**kwargs)

        if kwargs.get('center_graph', True):
            self._center_graph()

        self.trigger({'type': EventType.end, 'alpha': 0.0})
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
