"""
Random layout algorithm.

Places nodes at random positions within the canvas bounds.
Useful as a baseline for comparing layout quality metrics
and as a starting point for iterative algorithms.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

if TYPE_CHECKING:
    from typing_extensions import Self

from ..base import StaticLayout
from ..types import (
    Event,
    GroupLike,
    LinkLike,
    NodeLike,
    SizeType,
)


class RandomLayout(StaticLayout):
    """
    Random layout - positions nodes randomly within canvas bounds.

    This is a trivial but essential baseline layout. Nodes are placed
    at uniformly random positions within the specified canvas area.

    Use cases:
        - Baseline for comparing layout quality metrics
        - Starting point for iterative algorithms (force-directed, etc.)
        - Quick visualization when structure doesn't matter
        - Testing and debugging

    Example:
        layout = RandomLayout(
            nodes=[{}, {}, {}, {}, {}],
            links=[...],
            size=(800, 600),
        )
        layout.run()

        # With margin to keep nodes away from edges
        layout = RandomLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
            margin=50,  # 50px padding from canvas edges
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
        # Random-specific parameters
        margin: float = 0.0,
    ) -> None:
        """
        Initialize Random layout.

        Args:
            nodes: List of nodes
            links: List of links (not used for positioning, but stored for consistency)
            groups: List of groups
            size: Canvas size as (width, height)
            random_seed: Random seed for reproducible layouts
            on_start: Callback for start event
            on_tick: Callback for tick event
            on_end: Callback for end event
            margin: Padding from canvas edges (default 0). Nodes will be placed
                within the area [margin, width-margin] x [margin, height-margin].
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

        # Random-specific configuration
        self._margin: float = max(0.0, float(margin))

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def margin(self) -> float:
        """Get margin (padding from canvas edges)."""
        return self._margin

    @margin.setter
    def margin(self, value: float) -> None:
        """Set margin (padding from canvas edges)."""
        self._margin = max(0.0, float(value))

    # -------------------------------------------------------------------------
    # Layout Computation
    # -------------------------------------------------------------------------

    def run(self, **kwargs: Any) -> Self:
        """
        Run the random layout algorithm.

        By default, centering is disabled since random positions are already
        uniformly distributed within the canvas. This also preserves fixed
        node positions.

        Args:
            center_graph: If True, center the graph after layout. Defaults to False.
            **kwargs: Additional arguments passed to parent.

        Returns:
            self (for chaining)
        """
        # Default to no centering for random layout
        if "center_graph" not in kwargs:
            kwargs["center_graph"] = False
        return super().run(**kwargs)

    def _compute(self, **kwargs: Any) -> None:
        """Compute random layout positions."""
        n = len(self._nodes)
        if n == 0:
            return

        # Seed random number generator if specified
        if self._random_seed is not None:
            random.seed(self._random_seed)

        # Calculate placement bounds with margin
        width, height = self._canvas_size
        min_x = self._margin
        max_x = width - self._margin
        min_y = self._margin
        max_y = height - self._margin

        # Handle case where margin is too large for canvas
        if max_x <= min_x:
            min_x = 0.0
            max_x = width
        if max_y <= min_y:
            min_y = 0.0
            max_y = height

        # Place each node at a random position
        for node in self._nodes:
            # Skip fixed nodes - preserve their positions
            if node.fixed:
                continue

            node.x = random.uniform(min_x, max_x)
            node.y = random.uniform(min_y, max_y)


__all__ = ["RandomLayout"]
