"""
Radial tree layout algorithm.

Places tree nodes in concentric circles, with the root at the center
and children placed at increasing radii.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Union

from typing_extensions import Self

from ..base import StaticLayout


class RadialTreeLayout(StaticLayout):
    """
    Radial tree layout.

    Places the root at the center and children in concentric rings.
    Each subtree is assigned an angular wedge proportional to its size.

    Example:
        layout = (RadialTreeLayout()
            .nodes([{}, {}, {}, {}, {}])
            .links([
                {'source': 0, 'target': 1},
                {'source': 0, 'target': 2},
                {'source': 1, 'target': 3},
                {'source': 1, 'target': 4},
            ])
            .size([800, 600])
            .start())
    """

    def __init__(self) -> None:
        super().__init__()
        self._root_index: Optional[int] = None
        self._level_radius: float = 100.0  # Radius increment per level
        self._start_angle: float = 0.0  # Starting angle in radians
        self._sweep_angle: float = 2 * math.pi  # Total sweep angle

        # Internal state
        self._subtree_sizes: dict[int, int] = {}

    # -------------------------------------------------------------------------
    # Configuration Methods
    # -------------------------------------------------------------------------

    def root(self, index: Optional[int] = None) -> Union[Optional[int], Self]:
        """
        Get or set the root node index.

        Args:
            index: Root node index. If None, returns current value.

        Returns:
            Current value or self for chaining.
        """
        if index is None:
            return self._root_index
        self._root_index = index
        return self

    def level_radius(self, radius: Optional[float] = None) -> Union[float, Self]:
        """
        Get or set the radius increment per tree level.

        Args:
            radius: Radius increment. If None, returns current value.

        Returns:
            Current value or self for chaining.
        """
        if radius is None:
            return self._level_radius
        self._level_radius = float(radius)
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

    def sweep_angle(self, angle: Optional[float] = None) -> Union[float, Self]:
        """
        Get or set the total sweep angle in radians.

        Default is 2*pi (full circle). Use smaller values for partial trees.

        Args:
            angle: Sweep angle. If None, returns current value.

        Returns:
            Current value or self for chaining.
        """
        if angle is None:
            return self._sweep_angle
        self._sweep_angle = float(angle)
        return self

    # -------------------------------------------------------------------------
    # Tree Construction
    # -------------------------------------------------------------------------

    def _find_root(self) -> int:
        """Find a suitable root node."""
        if self._root_index is not None:
            return self._root_index

        n = len(self._nodes)
        if n == 0:
            return -1

        # Find node with no incoming edges
        has_parent = [False] * n
        for link in self._links:
            tgt = self._get_target_index(link)
            has_parent[tgt] = True

        for i in range(n):
            if not has_parent[i]:
                return i

        return 0

    def _build_children_map(self) -> dict[int, list[int]]:
        """Build map from parent index to list of child indices."""
        children: dict[int, list[int]] = {}
        for link in self._links:
            src = self._get_source_index(link)
            tgt = self._get_target_index(link)
            if src not in children:
                children[src] = []
            children[src].append(tgt)
        return children

    def _compute_subtree_sizes(
        self, node: int, children_map: dict[int, list[int]], visited: set[int]
    ) -> int:
        """Compute size of subtree rooted at node."""
        if node in visited:
            return 0

        visited.add(node)
        size = 1  # Count this node

        for child in children_map.get(node, []):
            size += self._compute_subtree_sizes(child, children_map, visited)

        self._subtree_sizes[node] = size
        return size

    # -------------------------------------------------------------------------
    # Layout Computation
    # -------------------------------------------------------------------------

    def _compute(self, **kwargs: Any) -> None:
        """Compute radial tree positions."""
        n = len(self._nodes)
        if n == 0:
            return

        root_idx = self._find_root()
        if root_idx < 0:
            return

        children_map = self._build_children_map()

        # Compute subtree sizes
        self._subtree_sizes = {}
        self._compute_subtree_sizes(root_idx, children_map, set())

        # Center of layout
        cx = self._canvas_size[0] / 2
        cy = self._canvas_size[1] / 2

        # Position root at center
        self._nodes[root_idx].x = cx
        self._nodes[root_idx].y = cy

        # Layout children recursively
        self._layout_subtree(
            root_idx,
            children_map,
            level=1,
            angle_start=self._start_angle,
            angle_sweep=self._sweep_angle,
            cx=cx,
            cy=cy,
            visited={root_idx}
        )

    def _layout_subtree(
        self,
        parent: int,
        children_map: dict[int, list[int]],
        level: int,
        angle_start: float,
        angle_sweep: float,
        cx: float,
        cy: float,
        visited: set[int]
    ) -> None:
        """Recursively layout a subtree."""
        children = children_map.get(parent, [])
        if not children:
            return

        # Filter out already visited nodes
        children = [c for c in children if c not in visited]
        if not children:
            return

        # Calculate radius for this level
        radius = level * self._level_radius

        # Total weight of children (based on subtree sizes)
        total_weight = sum(self._subtree_sizes.get(c, 1) for c in children)

        # Assign angular wedges proportional to subtree size
        current_angle = angle_start

        for child in children:
            visited.add(child)

            # Angular wedge for this child's subtree
            child_weight = self._subtree_sizes.get(child, 1)
            child_sweep = angle_sweep * (child_weight / total_weight)

            # Position at center of wedge
            child_angle = current_angle + child_sweep / 2

            # Convert polar to Cartesian
            x = cx + radius * math.cos(child_angle)
            y = cy + radius * math.sin(child_angle)

            self._nodes[child].x = x
            self._nodes[child].y = y

            # Layout child's subtree
            self._layout_subtree(
                child,
                children_map,
                level + 1,
                current_angle,
                child_sweep,
                cx,
                cy,
                visited
            )

            current_angle += child_sweep


__all__ = ["RadialTreeLayout"]
