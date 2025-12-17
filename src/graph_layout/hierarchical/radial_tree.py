"""
Radial tree layout algorithm.

Places tree nodes in concentric circles, with the root at the center
and children placed at increasing radii.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Optional, Sequence

from ..base import StaticLayout
from ..types import (
    Event,
    GroupLike,
    LinkLike,
    NodeLike,
    SizeType,
)


class TreeStructureWarning(UserWarning):
    """Warning issued when graph structure doesn't match tree assumptions."""

    pass


class RadialTreeLayout(StaticLayout):
    """
    Radial tree layout.

    Places the root at the center and children in concentric rings.
    Each subtree is assigned an angular wedge proportional to its size.

    Example:
        layout = RadialTreeLayout(
            nodes=[{}, {}, {}, {}, {}],
            links=[
                {'source': 0, 'target': 1},
                {'source': 0, 'target': 2},
                {'source': 1, 'target': 3},
                {'source': 1, 'target': 4},
            ],
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
        # RadialTree-specific parameters
        root: Optional[int] = None,
        level_radius: float = 100.0,
        start_angle: float = 0.0,
        sweep_angle: float = 2 * math.pi,
    ) -> None:
        """
        Initialize RadialTree layout.

        Args:
            nodes: List of nodes
            links: List of links
            groups: List of groups
            size: Canvas size as (width, height)
            random_seed: Random seed for reproducible layouts
            on_start: Callback for start event
            on_tick: Callback for tick event
            on_end: Callback for end event
            root: Root node index. If None, auto-detected.
            level_radius: Radius increment per tree level.
            start_angle: Starting angle in radians (default 0).
            sweep_angle: Total sweep angle in radians (default 2*pi for full circle).
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

        # RadialTree-specific configuration
        self._root_index: Optional[int] = root
        self._level_radius: float = float(level_radius)
        self._start_angle: float = float(start_angle)
        self._sweep_angle: float = float(sweep_angle)

        # Internal state
        self._subtree_sizes: dict[int, int] = {}

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def root(self) -> Optional[int]:
        """Get root node index."""
        return self._root_index

    @root.setter
    def root(self, value: Optional[int]) -> None:
        """Set root node index."""
        self._root_index = value

    @property
    def level_radius(self) -> float:
        """Get radius increment per tree level."""
        return self._level_radius

    @level_radius.setter
    def level_radius(self, value: float) -> None:
        """Set radius increment per tree level."""
        self._level_radius = float(value)

    @property
    def start_angle(self) -> float:
        """Get starting angle in radians."""
        return self._start_angle

    @start_angle.setter
    def start_angle(self, value: float) -> None:
        """Set starting angle in radians."""
        self._start_angle = float(value)

    @property
    def sweep_angle(self) -> float:
        """Get total sweep angle in radians."""
        return self._sweep_angle

    @sweep_angle.setter
    def sweep_angle(self, value: float) -> None:
        """Set total sweep angle in radians."""
        self._sweep_angle = float(value)

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

        warnings.warn(
            "No root node found (all nodes have incoming edges). "
            "This suggests the graph is not a tree. "
            "RadialTreeLayout is designed for trees; results may be suboptimal.",
            TreeStructureWarning,
            stacklevel=3,
        )
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
        visited: set[int] = set()
        self._compute_subtree_sizes(root_idx, children_map, visited)

        # Check for disconnected nodes
        unvisited = [i for i in range(n) if i not in visited]
        if unvisited:
            warnings.warn(
                f"Found {len(unvisited)} disconnected node(s) not reachable from root. "
                "These nodes will be placed at arbitrary positions. "
                "RadialTreeLayout is designed for connected trees.",
                TreeStructureWarning,
                stacklevel=2,
            )

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
            visited={root_idx},
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
        visited: set[int],
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
                child, children_map, level + 1, current_angle, child_sweep, cx, cy, visited
            )

            current_angle += child_sweep


__all__ = ["RadialTreeLayout", "TreeStructureWarning"]
