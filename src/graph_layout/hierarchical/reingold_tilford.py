"""
Reingold-Tilford tree layout algorithm.

Based on the paper:
"Tidier Drawings of Trees" by Reingold and Tilford (1981)

Extended with improvements from:
"A Node-Positioning Algorithm for General Trees" by Walker (1990)
"""

from __future__ import annotations

import warnings
from collections import deque
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


class TreeNode:
    """Internal tree node representation for layout computation."""

    def __init__(self, index: int) -> None:
        self.index = index
        self.children: list[TreeNode] = []
        self.parent: Optional[TreeNode] = None
        self.depth: int = 0

        # Layout coordinates
        self.x: float = 0.0
        self.y: float = 0.0

        # Reingold-Tilford fields
        self.mod: float = 0.0  # Modifier for subtree shift
        self.thread: Optional[TreeNode] = None
        self.ancestor: TreeNode = self
        self.prelim: float = 0.0
        self.change: float = 0.0
        self.shift: float = 0.0
        self.number: int = 0  # Position among siblings


class ReingoldTilfordLayout(StaticLayout):
    """
    Reingold-Tilford tidy tree layout.

    Positions nodes in a tree with:
    - Parent centered above children
    - Subtrees separated to avoid overlap
    - Minimum horizontal distance between nodes

    The tree is built from the graph by finding a root node and
    performing a traversal. Works best with actual tree graphs.

    Example:
        layout = ReingoldTilfordLayout(
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
        # ReingoldTilford-specific parameters
        root: Optional[int] = None,
        node_separation: float = 1.0,
        level_separation: float = 1.0,
        orientation: str = "top-to-bottom",
    ) -> None:
        """
        Initialize Reingold-Tilford layout.

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
            node_separation: Horizontal separation between sibling nodes.
            level_separation: Vertical separation between tree levels.
            orientation: Layout direction - 'top-to-bottom', 'bottom-to-top',
                'left-to-right', or 'right-to-left'.
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

        # ReingoldTilford-specific configuration
        self._root_index: Optional[int] = root
        self._node_separation: float = float(node_separation)
        self._level_separation: float = float(level_separation)
        valid_orientations = {"top-to-bottom", "bottom-to-top", "left-to-right", "right-to-left"}
        if orientation not in valid_orientations:
            raise ValueError(f"orientation must be one of {valid_orientations}")
        self._orientation: str = orientation

        # Internal state
        self._tree_nodes: list[TreeNode] = []
        self._tree_root: Optional[TreeNode] = None

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
    def node_separation(self) -> float:
        """Get horizontal separation between sibling nodes."""
        return self._node_separation

    @node_separation.setter
    def node_separation(self, value: float) -> None:
        """Set horizontal separation between sibling nodes."""
        self._node_separation = float(value)

    @property
    def level_separation(self) -> float:
        """Get vertical separation between tree levels."""
        return self._level_separation

    @level_separation.setter
    def level_separation(self, value: float) -> None:
        """Set vertical separation between tree levels."""
        self._level_separation = float(value)

    @property
    def orientation(self) -> str:
        """Get tree orientation."""
        return self._orientation

    @orientation.setter
    def orientation(self, value: str) -> None:
        """Set tree orientation."""
        valid = {"top-to-bottom", "bottom-to-top", "left-to-right", "right-to-left"}
        if value not in valid:
            raise ValueError(f"orientation must be one of {valid}")
        self._orientation = value

    # -------------------------------------------------------------------------
    # Tree Construction
    # -------------------------------------------------------------------------

    def _find_root(self) -> int:
        """Find a suitable root node (node with no incoming edges, or node 0)."""
        if self._root_index is not None:
            return self._root_index

        n = len(self._nodes)
        if n == 0:
            return -1

        # Find nodes with no parents
        has_parent = [False] * n

        for link in self._links:
            tgt = self._get_target_index(link)
            has_parent[tgt] = True

        # Find first node without a parent
        for i in range(n):
            if not has_parent[i]:
                return i

        # Fallback to node 0
        warnings.warn(
            "No root node found (all nodes have incoming edges). "
            "This suggests the graph is not a tree. "
            "Reingold-Tilford layout is designed for trees; results may be suboptimal.",
            TreeStructureWarning,
            stacklevel=4,
        )
        return 0

    def _build_tree(self, root_idx: int) -> TreeNode:
        """Build tree structure from graph via BFS from root."""
        n = len(self._nodes)
        self._tree_nodes = [TreeNode(i) for i in range(n)]
        visited = [False] * n

        # Build adjacency (directed: parent -> children)
        children: list[list[int]] = [[] for _ in range(n)]
        for link in self._links:
            src = self._get_source_index(link)
            tgt = self._get_target_index(link)
            children[src].append(tgt)

        # BFS to build tree
        tree_root = self._tree_nodes[root_idx]
        tree_root.depth = 0
        visited[root_idx] = True
        queue = deque([tree_root])

        while queue:
            node = queue.popleft()
            for child_idx in children[node.index]:
                if not visited[child_idx]:
                    visited[child_idx] = True
                    child = self._tree_nodes[child_idx]
                    child.parent = node
                    child.depth = node.depth + 1
                    node.children.append(child)
                    queue.append(child)

        # Number siblings
        self._number_siblings(tree_root)

        # Check for disconnected nodes
        unvisited = [i for i in range(n) if not visited[i]]
        if unvisited:
            warnings.warn(
                f"Found {len(unvisited)} disconnected node(s) not reachable from root. "
                "These nodes will be placed at arbitrary positions. "
                "Reingold-Tilford layout is designed for connected trees.",
                TreeStructureWarning,
                stacklevel=3,
            )

        return tree_root

    def _number_siblings(self, node: TreeNode) -> None:
        """Assign sibling numbers (position among siblings)."""
        for i, child in enumerate(node.children):
            child.number = i
            self._number_siblings(child)

    # -------------------------------------------------------------------------
    # Reingold-Tilford Algorithm
    # -------------------------------------------------------------------------

    def _first_walk(self, v: TreeNode) -> None:
        """
        First walk: compute preliminary x-coordinates (bottom-up).
        """
        if not v.children:
            # Leaf node
            if v.number > 0 and v.parent:
                # Has a left sibling
                left_sibling = v.parent.children[v.number - 1]
                v.prelim = left_sibling.prelim + self._node_separation
            else:
                v.prelim = 0
        else:
            # Internal node
            default_ancestor = v.children[0]

            for child in v.children:
                self._first_walk(child)
                default_ancestor = self._apportion(child, default_ancestor)

            self._execute_shifts(v)

            # Position at midpoint of children
            first_child = v.children[0]
            last_child = v.children[-1]
            midpoint = (first_child.prelim + last_child.prelim) / 2

            if v.number > 0 and v.parent:
                left_sibling = v.parent.children[v.number - 1]
                v.prelim = left_sibling.prelim + self._node_separation
                v.mod = v.prelim - midpoint
            else:
                v.prelim = midpoint

    def _apportion(self, v: TreeNode, default_ancestor: TreeNode) -> TreeNode:
        """
        Apportion: separate subtrees and thread for contour tracing.
        """
        if v.number > 0 and v.parent:
            left_sibling = v.parent.children[v.number - 1]

            v_inner_right = v
            v_outer_right = v
            v_inner_left = left_sibling
            v_outer_left = v.parent.children[0]

            s_inner_right = v.mod
            s_outer_right = v.mod
            s_inner_left = v_inner_left.mod
            s_outer_left = v_outer_left.mod

            while self._next_right(v_inner_left) and self._next_left(v_inner_right):
                next_inner_left = self._next_right(v_inner_left)
                next_inner_right = self._next_left(v_inner_right)
                next_outer_left = self._next_left(v_outer_left)
                next_outer_right = self._next_right(v_outer_right)

                # These are guaranteed non-None by the while condition
                assert next_inner_left is not None
                assert next_inner_right is not None
                assert next_outer_left is not None
                assert next_outer_right is not None

                v_inner_left = next_inner_left
                v_inner_right = next_inner_right
                v_outer_left = next_outer_left
                v_outer_right = next_outer_right

                v_outer_right.ancestor = v

                shift = (
                    (v_inner_left.prelim + s_inner_left)
                    - (v_inner_right.prelim + s_inner_right)
                    + self._node_separation
                )

                if shift > 0:
                    ancestor = self._ancestor(v_inner_left, v, default_ancestor)
                    self._move_subtree(ancestor, v, shift)
                    s_inner_right += shift
                    s_outer_right += shift

                s_inner_left += v_inner_left.mod
                s_inner_right += v_inner_right.mod
                s_outer_left += v_outer_left.mod
                s_outer_right += v_outer_right.mod

            if self._next_right(v_inner_left) and not self._next_right(v_outer_right):
                v_outer_right.thread = self._next_right(v_inner_left)
                v_outer_right.mod += s_inner_left - s_outer_right

            if self._next_left(v_inner_right) and not self._next_left(v_outer_left):
                v_outer_left.thread = self._next_left(v_inner_right)
                v_outer_left.mod += s_inner_right - s_outer_left
                default_ancestor = v

        return default_ancestor

    def _next_left(self, v: TreeNode) -> Optional[TreeNode]:
        """Get next node on left contour."""
        if v.children:
            return v.children[0]
        return v.thread

    def _next_right(self, v: TreeNode) -> Optional[TreeNode]:
        """Get next node on right contour."""
        if v.children:
            return v.children[-1]
        return v.thread

    def _ancestor(self, v_inner_left: TreeNode, v: TreeNode, default: TreeNode) -> TreeNode:
        """Find ancestor of v_inner_left that is a sibling of v."""
        if v_inner_left.ancestor.parent == v.parent:
            return v_inner_left.ancestor
        return default

    def _move_subtree(self, wl: TreeNode, wr: TreeNode, shift: float) -> None:
        """Move subtree rooted at wr by shift amount."""
        subtrees = wr.number - wl.number
        if subtrees > 0:
            wr.change -= shift / subtrees
            wr.shift += shift
            wl.change += shift / subtrees
            wr.prelim += shift
            wr.mod += shift

    def _execute_shifts(self, v: TreeNode) -> None:
        """Execute accumulated shifts for children of v."""
        shift = 0.0
        change = 0.0
        for child in reversed(v.children):
            child.prelim += shift
            child.mod += shift
            change += child.change
            shift += child.shift + change

    def _second_walk(self, v: TreeNode, mod: float = 0.0) -> None:
        """
        Second walk: compute final positions (top-down).
        """
        v.x = v.prelim + mod
        v.y = v.depth * self._level_separation

        for child in v.children:
            self._second_walk(child, mod + v.mod)

    # -------------------------------------------------------------------------
    # Layout Computation
    # -------------------------------------------------------------------------

    def _compute(self, **kwargs: Any) -> None:
        """Compute node positions using Reingold-Tilford algorithm."""
        n = len(self._nodes)
        if n == 0:
            return

        # Find root and build tree
        root_idx = self._find_root()
        if root_idx < 0:
            return

        self._tree_root = self._build_tree(root_idx)

        # Run Reingold-Tilford
        self._first_walk(self._tree_root)
        self._second_walk(self._tree_root)

        # Apply orientation and scale to fit canvas
        self._apply_layout()

    def _apply_layout(self) -> None:
        """Apply computed positions to actual nodes, with orientation and scaling."""
        if not self._tree_nodes:
            return

        # Find bounds
        min_x = min(tn.x for tn in self._tree_nodes)
        max_x = max(tn.x for tn in self._tree_nodes)
        min_y = min(tn.y for tn in self._tree_nodes)
        max_y = max(tn.y for tn in self._tree_nodes)

        width = max_x - min_x if max_x > min_x else 1
        height = max_y - min_y if max_y > min_y else 1

        # Calculate scale to fit canvas with padding
        padding = 50
        canvas_w = self._canvas_size[0] - 2 * padding
        canvas_h = self._canvas_size[1] - 2 * padding

        # Apply positions
        for tn in self._tree_nodes:
            node = self._nodes[tn.index]

            # Normalize to [0, 1]
            nx = (tn.x - min_x) / width if width > 0 else 0.5
            ny = (tn.y - min_y) / height if height > 0 else 0.5

            # Apply orientation
            if self._orientation == "top-to-bottom":
                node.x = padding + nx * canvas_w
                node.y = padding + ny * canvas_h
            elif self._orientation == "bottom-to-top":
                node.x = padding + nx * canvas_w
                node.y = self._canvas_size[1] - padding - ny * canvas_h
            elif self._orientation == "left-to-right":
                node.x = padding + ny * canvas_w
                node.y = padding + nx * canvas_h
            elif self._orientation == "right-to-left":
                node.x = self._canvas_size[0] - padding - ny * canvas_w
                node.y = padding + nx * canvas_h


__all__ = ["ReingoldTilfordLayout", "TreeStructureWarning"]
