"""
Circular layout algorithm.

Places all nodes evenly distributed on a circle.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Union

from typing_extensions import Self

from ..base import StaticLayout
from ..types import Node


class CircularLayout(StaticLayout):
    """
    Circular layout - positions nodes on a circle.

    Nodes are placed evenly spaced around a circle. The order can be
    customized using a sort function.

    Example:
        layout = (CircularLayout()
            .nodes([{}, {}, {}, {}, {}])
            .links([...])
            .size([800, 600])
            .start())
    """

    def __init__(self) -> None:
        super().__init__()
        self._radius: Optional[float] = None  # Auto-computed if None
        self._start_angle: float = 0.0
        self._sort_by: Optional[Union[str, Callable[[Node], Any]]] = None

    # -------------------------------------------------------------------------
    # Configuration Methods
    # -------------------------------------------------------------------------

    def radius(self, r: Optional[float] = None) -> Union[Optional[float], Self]:
        """
        Get or set the circle radius.

        If None, radius is computed automatically based on canvas size.

        Args:
            r: Radius value. If None, returns current value.

        Returns:
            Current value or self for chaining.
        """
        if r is None:
            return self._radius
        self._radius = float(r) if r else None
        return self

    def start_angle(self, angle: Optional[float] = None) -> Union[float, Self]:
        """
        Get or set the starting angle in radians.

        Args:
            angle: Start angle. If None, returns current value.

        Returns:
            Current value or self for chaining.
        """
        if angle is None:
            return self._start_angle
        self._start_angle = float(angle)
        return self

    def sort_by(
        self, key: Optional[Union[str, Callable[[Node], Any]]] = None
    ) -> Union[Optional[Union[str, Callable]], Self]:
        """
        Get or set the sort key for node ordering.

        Options:
        - None: Keep original order
        - 'degree': Sort by node degree (connections)
        - callable: Custom function taking a Node and returning a sort key

        Args:
            key: Sort key. If None when called as getter, returns current value.

        Returns:
            Current value or self for chaining.
        """
        if key is None:
            return self._sort_by
        self._sort_by = key
        return self

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

        if self._sort_by == 'degree':
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
