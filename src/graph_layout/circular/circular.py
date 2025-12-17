"""
Circular layout algorithm.

Places all nodes evenly distributed on a circle.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Sequence, Union

from ..base import StaticLayout
from ..types import (
    Event,
    GroupLike,
    LinkLike,
    Node,
    NodeLike,
    SizeType,
)


class CircularLayout(StaticLayout):
    """
    Circular layout - positions nodes on a circle.

    Nodes are placed evenly spaced around a circle. The order can be
    customized using a sort function.

    Example:
        layout = CircularLayout(
            nodes=[{}, {}, {}, {}, {}],
            links=[...],
            size=(800, 600),
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
        # Circular-specific parameters
        radius: Optional[float] = None,
        start_angle: float = 0.0,
        sort_by: Optional[Union[str, Callable[[Node], Any]]] = None,
    ) -> None:
        """
        Initialize Circular layout.

        Args:
            nodes: List of nodes
            links: List of links
            groups: List of groups
            size: Canvas size as (width, height)
            random_seed: Random seed for reproducible layouts
            on_start: Callback for start event
            on_tick: Callback for tick event
            on_end: Callback for end event
            radius: Circle radius. If None, auto-computed from canvas size.
            start_angle: Starting angle in radians (default 0).
            sort_by: Sort key for node ordering. Options:
                - None: Keep original order
                - 'degree': Sort by node degree (connections)
                - callable: Custom function taking a Node and returning a sort key
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

        # Circular-specific configuration
        self._radius: Optional[float] = radius
        self._start_angle: float = float(start_angle)
        self._sort_by: Optional[Union[str, Callable[[Node], Any]]] = sort_by

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def radius(self) -> Optional[float]:
        """Get circle radius (None = auto-computed)."""
        return self._radius

    @radius.setter
    def radius(self, value: Optional[float]) -> None:
        """Set circle radius."""
        self._radius = float(value) if value is not None else None

    @property
    def start_angle(self) -> float:
        """Get starting angle in radians."""
        return self._start_angle

    @start_angle.setter
    def start_angle(self, value: float) -> None:
        """Set starting angle in radians."""
        self._start_angle = float(value)

    @property
    def sort_by(self) -> Optional[Union[str, Callable[[Node], Any]]]:
        """Get sort key for node ordering."""
        return self._sort_by

    @sort_by.setter
    def sort_by(self, value: Optional[Union[str, Callable[[Node], Any]]]) -> None:
        """Set sort key for node ordering."""
        self._sort_by = value

    # -------------------------------------------------------------------------
    # Layout Computation
    # -------------------------------------------------------------------------

    def _compute_degree(self, node_idx: int) -> int:
        """Compute degree (number of connections) for a node."""
        degree = 0
        for link in self._links:
            src = self._get_source_index(link)
            tgt = self._get_target_index(link)
            if src == node_idx or tgt == node_idx:
                degree += 1
        return degree

    def _get_sorted_indices(self) -> list[int]:
        """Get node indices in sorted order."""
        n = len(self._nodes)
        indices = list(range(n))

        if self._sort_by is None:
            return indices

        if self._sort_by == "degree":
            # Sort by degree (descending)
            degrees = [self._compute_degree(i) for i in range(n)]
            indices.sort(key=lambda i: -degrees[i])
        elif callable(self._sort_by):
            # Custom sort function
            sort_fn = self._sort_by  # Store in local for proper type narrowing
            indices.sort(key=lambda i: sort_fn(self._nodes[i]))

        return indices

    def _compute(self, **kwargs: Any) -> None:
        """Compute circular layout positions."""
        n = len(self._nodes)
        if n == 0:
            return

        # Calculate center and radius
        cx = self._canvas_size[0] / 2
        cy = self._canvas_size[1] / 2

        if self._radius is not None:
            radius = self._radius
        else:
            # Auto-compute radius to fit in canvas with padding
            max_radius = min(self._canvas_size[0], self._canvas_size[1]) / 2 - 50
            radius = max(50, max_radius)

        # Get sorted node indices
        sorted_indices = self._get_sorted_indices()

        # Place nodes around the circle
        angle_step = 2 * math.pi / n if n > 0 else 0

        for pos, node_idx in enumerate(sorted_indices):
            angle = self._start_angle + pos * angle_step
            self._nodes[node_idx].x = cx + radius * math.cos(angle)
            self._nodes[node_idx].y = cy + radius * math.sin(angle)


__all__ = ["CircularLayout"]
