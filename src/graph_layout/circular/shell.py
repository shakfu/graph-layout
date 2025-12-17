"""
Shell layout algorithm.

Places nodes in concentric circles (shells) based on grouping or degree.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Sequence

from ..base import StaticLayout
from ..types import (
    Event,
    GroupLike,
    LinkLike,
    NodeLike,
    SizeType,
)


class ShellLayout(StaticLayout):
    """
    Shell layout - positions nodes in concentric circles.

    Nodes are grouped into shells (rings) and placed on circles of
    increasing radii. Groups can be specified explicitly or computed
    automatically based on node degree.

    Example:
        # Explicit shells
        layout = ShellLayout(
            nodes=[{}, {}, {}, {}, {}, {}],
            shells=[[0], [1, 2], [3, 4, 5]],  # Center, inner ring, outer ring
            size=(800, 600),
        )
        layout.run()

        # Auto-group by degree
        layout = ShellLayout(
            nodes=[...],
            links=[...],
            auto_shells=3,  # 3 shells based on degree
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
        # Shell-specific parameters
        shells: Optional[list[list[int]]] = None,
        auto_shells: int = 0,
        radius_step: float = 100.0,
        start_angle: float = 0.0,
    ) -> None:
        """
        Initialize Shell layout.

        Args:
            nodes: List of nodes
            links: List of links
            groups: List of groups
            size: Canvas size as (width, height)
            random_seed: Random seed for reproducible layouts
            on_start: Callback for start event
            on_tick: Callback for tick event
            on_end: Callback for end event
            shells: Explicit shell assignments. Each shell is a list of node indices.
                Shells are ordered from center (index 0) to outermost.
            auto_shells: Number of shells to auto-generate based on node degree.
                Higher-degree nodes go in inner shells. Ignored if shells is set.
            radius_step: Radius increment between shells.
            start_angle: Starting angle in radians.
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

        # Shell-specific configuration
        self._shells: Optional[list[list[int]]] = shells
        self._n_auto_shells: int = max(0, int(auto_shells)) if shells is None else 0
        self._radius_step: float = float(radius_step)
        self._start_angle: float = float(start_angle)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def shells(self) -> Optional[list[list[int]]]:
        """Get explicit shell assignments."""
        return self._shells

    @shells.setter
    def shells(self, value: Optional[list[list[int]]]) -> None:
        """Set explicit shell assignments."""
        self._shells = value
        if value is not None:
            self._n_auto_shells = 0

    @property
    def auto_shells(self) -> int:
        """Get number of auto-generated shells."""
        return self._n_auto_shells

    @auto_shells.setter
    def auto_shells(self, value: int) -> None:
        """Set number of auto-generated shells."""
        self._n_auto_shells = max(0, int(value))
        if value > 0:
            self._shells = None

    @property
    def radius_step(self) -> float:
        """Get radius increment between shells."""
        return self._radius_step

    @radius_step.setter
    def radius_step(self, value: float) -> None:
        """Set radius increment between shells."""
        self._radius_step = float(value)

    @property
    def start_angle(self) -> float:
        """Get starting angle in radians."""
        return self._start_angle

    @start_angle.setter
    def start_angle(self, value: float) -> None:
        """Set starting angle in radians."""
        self._start_angle = float(value)

    # -------------------------------------------------------------------------
    # Shell Computation
    # -------------------------------------------------------------------------

    def _compute_degrees(self) -> list[int]:
        """Compute degree for each node."""
        n = len(self._nodes)
        degrees = [0] * n

        for link in self._links:
            src = self._get_source_index(link)
            tgt = self._get_target_index(link)
            degrees[src] += 1
            degrees[tgt] += 1

        return degrees

    def _compute_auto_shells(self) -> list[list[int]]:
        """Compute shells automatically based on node degree."""
        n = len(self._nodes)
        if n == 0:
            return []

        if self._n_auto_shells <= 0:
            return [list(range(n))]

        # Compute degrees
        degrees = self._compute_degrees()

        # Sort nodes by degree (descending)
        sorted_nodes = sorted(range(n), key=lambda i: -degrees[i])

        # Distribute into shells
        n_shells = min(self._n_auto_shells, n)
        computed_shells: list[list[int]] = [[] for _ in range(n_shells)]

        # Distribute evenly across shells
        for i, node in enumerate(sorted_nodes):
            shell_idx = i * n_shells // n
            computed_shells[shell_idx].append(node)

        return computed_shells

    def _get_shells(self) -> list[list[int]]:
        """Get shells (explicit or auto-computed)."""
        if self._shells is not None:
            return self._shells

        if self._n_auto_shells > 0:
            return self._compute_auto_shells()

        # Default: all nodes in one shell
        return [list(range(len(self._nodes)))]

    # -------------------------------------------------------------------------
    # Layout Computation
    # -------------------------------------------------------------------------

    def _compute(self, **kwargs: Any) -> None:
        """Compute shell layout positions."""
        n = len(self._nodes)
        if n == 0:
            return

        computed_shells = self._get_shells()
        if not computed_shells:
            return

        # Calculate center
        cx = self._canvas_size[0] / 2
        cy = self._canvas_size[1] / 2

        # Calculate radius step based on canvas size
        max_radius = min(self._canvas_size[0], self._canvas_size[1]) / 2 - 50
        n_shells = len(computed_shells)

        if n_shells > 1:
            radius_step = min(self._radius_step, max_radius / n_shells)
        else:
            radius_step = max_radius

        # Place each shell
        for shell_idx, shell_nodes in enumerate(computed_shells):
            if not shell_nodes:
                continue

            # Radius for this shell (inner shells have smaller radius)
            if shell_idx == 0 and len(shell_nodes) == 1:
                # Single node in center
                radius = 0.0
            else:
                radius = (shell_idx + 1) * radius_step if n_shells > 1 else max_radius

            # Place nodes in this shell
            n_in_shell = len(shell_nodes)
            angle_step = 2 * math.pi / n_in_shell if n_in_shell > 0 else 0

            for pos, node_idx in enumerate(shell_nodes):
                if radius == 0:
                    # Center node
                    self._nodes[node_idx].x = cx
                    self._nodes[node_idx].y = cy
                else:
                    angle = self._start_angle + pos * angle_step
                    self._nodes[node_idx].x = cx + radius * math.cos(angle)
                    self._nodes[node_idx].y = cy + radius * math.sin(angle)


__all__ = ["ShellLayout"]
