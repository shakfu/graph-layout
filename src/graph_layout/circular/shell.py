"""
Shell layout algorithm.

Places nodes in concentric circles (shells) based on grouping or degree.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Union

from typing_extensions import Self

from ..base import StaticLayout


class ShellLayout(StaticLayout):
    """
    Shell layout - positions nodes in concentric circles.

    Nodes are grouped into shells (rings) and placed on circles of
    increasing radii. Groups can be specified explicitly or computed
    automatically based on node degree.

    Example:
        # Explicit shells
        layout = (ShellLayout()
            .nodes([{}, {}, {}, {}, {}, {}])
            .shells([[0], [1, 2], [3, 4, 5]])  # Center, inner ring, outer ring
            .size([800, 600])
            .start())

        # Auto-group by degree
        layout = (ShellLayout()
            .nodes([...])
            .links([...])
            .auto_shells(3)  # 3 shells based on degree
            .size([800, 600])
            .start())
    """

    def __init__(self) -> None:
        super().__init__()
        self._shells: Optional[list[list[int]]] = None
        self._n_auto_shells: int = 0
        self._radius_step: float = 100.0
        self._start_angle: float = 0.0

    # -------------------------------------------------------------------------
    # Configuration Methods
    # -------------------------------------------------------------------------

    def shells(
        self, shells: Optional[list[list[int]]] = None
    ) -> Union[Optional[list[list[int]]], Self]:
        """
        Get or set explicit shell assignments.

        Each shell is a list of node indices. Shells are ordered from
        center (index 0) to outermost.

        Args:
            shells: List of shells. If None, returns current value.

        Returns:
            Current value or self for chaining.
        """
        if shells is None:
            return self._shells
        self._shells = shells
        self._n_auto_shells = 0  # Disable auto
        return self

    def auto_shells(self, n: Optional[int] = None) -> Union[int, Self]:
        """
        Get or set automatic shell grouping by degree.

        Nodes are automatically grouped into n shells based on their
        degree (number of connections). High-degree nodes are placed
        in inner shells.

        Args:
            n: Number of shells. If None, returns current value.

        Returns:
            Current value or self for chaining.
        """
        if n is None:
            return self._n_auto_shells
        self._n_auto_shells = max(1, int(n))
        self._shells = None  # Disable explicit shells
        return self

    def radius_step(self, step: Optional[float] = None) -> Union[float, Self]:
        """
        Get or set the radius increment between shells.

        Args:
            step: Radius step. If None, returns current value.

        Returns:
            Current value or self for chaining.
        """
        if step is None:
            return self._radius_step
        self._radius_step = float(step)
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
        shells: list[list[int]] = [[] for _ in range(n_shells)]

        # Distribute evenly across shells
        for i, node in enumerate(sorted_nodes):
            shell_idx = i * n_shells // n
            shells[shell_idx].append(node)

        return shells

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

        shells = self._get_shells()
        if not shells:
            return

        # Calculate center
        cx = self._canvas_size[0] / 2
        cy = self._canvas_size[1] / 2

        # Calculate radius step based on canvas size
        max_radius = min(self._canvas_size[0], self._canvas_size[1]) / 2 - 50
        n_shells = len(shells)

        if n_shells > 1:
            radius_step = min(self._radius_step, max_radius / n_shells)
        else:
            radius_step = max_radius

        # Place each shell
        for shell_idx, shell_nodes in enumerate(shells):
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
