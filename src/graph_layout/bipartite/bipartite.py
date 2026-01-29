"""
Bipartite layout algorithm.

Places nodes in two parallel rows/columns for bipartite graphs,
minimizing edge crossings between the rows.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Sequence

if TYPE_CHECKING:
    pass

from ..base import StaticLayout
from ..types import (
    Event,
    GroupLike,
    LinkLike,
    NodeLike,
    SizeType,
)


class BipartiteLayout(StaticLayout):
    """
    Bipartite layout algorithm.

    Places nodes in two parallel rows (or columns) for bipartite graphs.
    The two vertex sets are positioned on opposite sides, with edges
    connecting between them.

    The layout automatically detects the bipartite structure if not
    provided, or allows explicit specification of the two sets.

    Edge crossings are minimized using the barycenter heuristic.

    Example:
        # User-item bipartite graph
        users = [{}, {}, {}]  # indices 0, 1, 2
        items = [{}, {}, {}, {}]  # indices 3, 4, 5, 6
        nodes = users + items

        links = [
            {'source': 0, 'target': 3},
            {'source': 0, 'target': 4},
            {'source': 1, 'target': 4},
            {'source': 1, 'target': 5},
            {'source': 2, 'target': 5},
            {'source': 2, 'target': 6},
        ]

        layout = BipartiteLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
            top_set=[0, 1, 2],  # Users on top
            bottom_set=[3, 4, 5, 6],  # Items on bottom
        )
        layout.run()

    Attributes:
        top_nodes: Indices of nodes in the top/left set
        bottom_nodes: Indices of nodes in the bottom/right set
        is_bipartite: Whether the graph is bipartite
    """

    def __init__(
        self,
        *,
        nodes: Optional[Sequence[NodeLike]] = None,
        links: Optional[Sequence[LinkLike]] = None,
        groups: Optional[Sequence[GroupLike]] = None,
        size: SizeType = (800, 600),
        random_seed: Optional[int] = None,
        on_start: Optional[Callable[[Optional[Event]], None]] = None,
        on_tick: Optional[Callable[[Optional[Event]], None]] = None,
        on_end: Optional[Callable[[Optional[Event]], None]] = None,
        # Bipartite-specific parameters
        top_set: Optional[Sequence[int]] = None,
        bottom_set: Optional[Sequence[int]] = None,
        orientation: Literal["horizontal", "vertical"] = "horizontal",
        layer_separation: float = 200,
        node_separation: float = 50,
        minimize_crossings: bool = True,
        crossing_iterations: int = 4,
    ) -> None:
        """
        Initialize Bipartite layout.

        Args:
            nodes: List of nodes
            links: List of links
            groups: List of groups
            size: Canvas size as (width, height)
            random_seed: Random seed for reproducible layouts
            on_start: Callback for start event
            on_tick: Callback for tick event
            on_end: Callback for end event
            top_set: Indices of nodes in top/left set (auto-detected if None)
            bottom_set: Indices of nodes in bottom/right set (auto-detected if None)
            orientation: "horizontal" (top/bottom) or "vertical" (left/right)
            layer_separation: Distance between the two rows/columns
            node_separation: Minimum gap between adjacent nodes in same row
            minimize_crossings: If True, reorder nodes to minimize edge crossings
            crossing_iterations: Number of iterations for crossing minimization
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

        # Bipartite-specific configuration
        self._top_set = list(top_set) if top_set is not None else None
        self._bottom_set = list(bottom_set) if bottom_set is not None else None
        self._orientation = orientation
        self._layer_separation = float(layer_separation)
        self._node_separation = float(node_separation)
        self._minimize_crossings = bool(minimize_crossings)
        self._crossing_iterations = int(crossing_iterations)

        # Output data
        self._top_nodes: list[int] = []
        self._bottom_nodes: list[int] = []
        self._is_bipartite: bool = True

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def top_set(self) -> Optional[list[int]]:
        """Get the user-specified top set."""
        return self._top_set

    @top_set.setter
    def top_set(self, value: Optional[Sequence[int]]) -> None:
        """Set the top set."""
        self._top_set = list(value) if value is not None else None

    @property
    def bottom_set(self) -> Optional[list[int]]:
        """Get the user-specified bottom set."""
        return self._bottom_set

    @bottom_set.setter
    def bottom_set(self, value: Optional[Sequence[int]]) -> None:
        """Set the bottom set."""
        self._bottom_set = list(value) if value is not None else None

    @property
    def orientation(self) -> str:
        """Get layout orientation."""
        return self._orientation

    @orientation.setter
    def orientation(self, value: Literal["horizontal", "vertical"]) -> None:
        """Set layout orientation."""
        self._orientation = value

    @property
    def layer_separation(self) -> float:
        """Get distance between the two rows/columns."""
        return self._layer_separation

    @layer_separation.setter
    def layer_separation(self, value: float) -> None:
        """Set distance between the two rows/columns."""
        self._layer_separation = max(0.0, float(value))

    @property
    def node_separation(self) -> float:
        """Get minimum gap between adjacent nodes."""
        return self._node_separation

    @node_separation.setter
    def node_separation(self, value: float) -> None:
        """Set minimum gap between adjacent nodes."""
        self._node_separation = max(0.0, float(value))

    @property
    def minimize_crossings(self) -> bool:
        """Get whether crossing minimization is enabled."""
        return self._minimize_crossings

    @minimize_crossings.setter
    def minimize_crossings(self, value: bool) -> None:
        """Set whether to minimize crossings."""
        self._minimize_crossings = bool(value)

    @property
    def top_nodes(self) -> list[int]:
        """Get nodes in the top/left set (after layout)."""
        return self._top_nodes

    @property
    def bottom_nodes(self) -> list[int]:
        """Get nodes in the bottom/right set (after layout)."""
        return self._bottom_nodes

    @property
    def is_bipartite(self) -> bool:
        """Check if the graph is bipartite."""
        return self._is_bipartite

    # -------------------------------------------------------------------------
    # Layout Computation
    # -------------------------------------------------------------------------

    def _compute(self, **kwargs: Any) -> None:
        """Compute bipartite layout."""
        n = len(self._nodes)
        if n == 0:
            return

        # Build adjacency list
        adj: dict[int, list[int]] = defaultdict(list)
        for link in self._links:
            src = self._get_source_index(link)
            tgt = self._get_target_index(link)
            if 0 <= src < n and 0 <= tgt < n and src != tgt:
                adj[src].append(tgt)
                adj[tgt].append(src)

        # Determine bipartite sets
        if self._top_set is not None and self._bottom_set is not None:
            # Use user-specified sets
            self._top_nodes = [i for i in self._top_set if 0 <= i < n]
            self._bottom_nodes = [i for i in self._bottom_set if 0 <= i < n]
            self._is_bipartite = self._verify_bipartite(adj)
        else:
            # Auto-detect bipartite structure
            self._is_bipartite, self._top_nodes, self._bottom_nodes = self._detect_bipartite(n, adj)

        # Handle non-bipartite graphs by forcing a split
        if not self._is_bipartite or (not self._top_nodes and not self._bottom_nodes):
            # Fall back to splitting nodes by index
            mid = n // 2
            self._top_nodes = list(range(mid))
            self._bottom_nodes = list(range(mid, n))

        # Minimize edge crossings
        if self._minimize_crossings:
            self._top_nodes, self._bottom_nodes = self._minimize_edge_crossings(
                self._top_nodes, self._bottom_nodes, adj
            )

        # Position nodes
        self._position_nodes()

    def _detect_bipartite(
        self, n: int, adj: dict[int, list[int]]
    ) -> tuple[bool, list[int], list[int]]:
        """
        Detect bipartite structure using BFS coloring.

        Returns:
            Tuple of (is_bipartite, set_a, set_b)
        """
        color: dict[int, int] = {}
        set_a: list[int] = []
        set_b: list[int] = []

        for start in range(n):
            if start in color:
                continue

            # BFS from this node
            queue = deque([start])
            color[start] = 0
            set_a.append(start)

            while queue:
                node = queue.popleft()
                node_color = color[node]
                next_color = 1 - node_color

                for neighbor in adj[node]:
                    if neighbor in color:
                        if color[neighbor] == node_color:
                            # Same color as neighbor - not bipartite
                            return False, [], []
                    else:
                        color[neighbor] = next_color
                        if next_color == 0:
                            set_a.append(neighbor)
                        else:
                            set_b.append(neighbor)
                        queue.append(neighbor)

        return True, set_a, set_b

    def _verify_bipartite(self, adj: dict[int, list[int]]) -> bool:
        """
        Verify that user-specified sets form a valid bipartite partition.

        Returns:
            True if no edges within the same set
        """
        top_set = set(self._top_nodes)
        bottom_set = set(self._bottom_nodes)

        # Check no edges within top set
        for node in self._top_nodes:
            for neighbor in adj[node]:
                if neighbor in top_set:
                    return False

        # Check no edges within bottom set
        for node in self._bottom_nodes:
            for neighbor in adj[node]:
                if neighbor in bottom_set:
                    return False

        return True

    def _minimize_edge_crossings(
        self,
        top_nodes: list[int],
        bottom_nodes: list[int],
        adj: dict[int, list[int]],
    ) -> tuple[list[int], list[int]]:
        """
        Minimize edge crossings using barycenter heuristic.

        Alternates between fixing one layer and reordering the other
        based on the average position of neighbors.

        Returns:
            Reordered (top_nodes, bottom_nodes)
        """
        top_order = list(top_nodes)
        bottom_order = list(bottom_nodes)

        top_set = set(top_nodes)
        bottom_set = set(bottom_nodes)

        for iteration in range(self._crossing_iterations):
            # Reorder bottom based on top positions
            if top_order:
                top_pos = {node: i for i, node in enumerate(top_order)}
                bottom_order = self._barycenter_order(bottom_order, top_pos, adj, top_set)

            # Reorder top based on bottom positions
            if bottom_order:
                bottom_pos = {node: i for i, node in enumerate(bottom_order)}
                top_order = self._barycenter_order(top_order, bottom_pos, adj, bottom_set)

        return top_order, bottom_order

    def _barycenter_order(
        self,
        nodes: list[int],
        neighbor_positions: dict[int, int],
        adj: dict[int, list[int]],
        neighbor_set: set[int],
    ) -> list[int]:
        """
        Reorder nodes by barycenter (average neighbor position).

        Args:
            nodes: Nodes to reorder
            neighbor_positions: Position of each neighbor
            adj: Adjacency list
            neighbor_set: Set of valid neighbors to consider

        Returns:
            Reordered node list
        """
        barycenters: list[tuple[float, int]] = []

        for node in nodes:
            # Get positions of neighbors in the other set
            neighbor_pos = [
                neighbor_positions[n]
                for n in adj[node]
                if n in neighbor_set and n in neighbor_positions
            ]

            if neighbor_pos:
                barycenter = sum(neighbor_pos) / len(neighbor_pos)
            else:
                # No neighbors - keep original relative position
                barycenter = float(nodes.index(node))

            barycenters.append((barycenter, node))

        # Sort by barycenter
        barycenters.sort(key=lambda x: x[0])
        return [node for _, node in barycenters]

    def _position_nodes(self) -> None:
        """Position nodes in two rows/columns."""
        canvas_width, canvas_height = self._canvas_size

        if self._orientation == "horizontal":
            # Top row and bottom row
            self._position_horizontal(canvas_width, canvas_height)
        else:
            # Left column and right column
            self._position_vertical(canvas_width, canvas_height)

    def _position_horizontal(self, width: float, height: float) -> None:
        """Position nodes in top and bottom rows."""
        center_y = height / 2
        top_y = center_y - self._layer_separation / 2
        bottom_y = center_y + self._layer_separation / 2

        # Position top row
        self._position_row(self._top_nodes, top_y, width)

        # Position bottom row
        self._position_row(self._bottom_nodes, bottom_y, width)

    def _position_vertical(self, width: float, height: float) -> None:
        """Position nodes in left and right columns."""
        center_x = width / 2
        left_x = center_x - self._layer_separation / 2
        right_x = center_x + self._layer_separation / 2

        # Position left column
        self._position_column(self._top_nodes, left_x, height)

        # Position right column
        self._position_column(self._bottom_nodes, right_x, height)

    def _position_row(self, nodes: list[int], y: float, width: float) -> None:
        """Position a row of nodes horizontally."""
        n = len(nodes)
        if n == 0:
            return

        # Calculate total width needed
        total_width = (n - 1) * self._node_separation

        # Center the row
        start_x = (width - total_width) / 2

        for i, node_idx in enumerate(nodes):
            x = start_x + i * self._node_separation
            self._nodes[node_idx].x = x
            self._nodes[node_idx].y = y

    def _position_column(self, nodes: list[int], x: float, height: float) -> None:
        """Position a column of nodes vertically."""
        n = len(nodes)
        if n == 0:
            return

        # Calculate total height needed
        total_height = (n - 1) * self._node_separation

        # Center the column
        start_y = (height - total_height) / 2

        for i, node_idx in enumerate(nodes):
            y = start_y + i * self._node_separation
            self._nodes[node_idx].x = x
            self._nodes[node_idx].y = y


def is_bipartite(
    num_nodes: int,
    edges: list[tuple[int, int]],
) -> tuple[bool, list[int], list[int]]:
    """
    Check if a graph is bipartite and return the two sets.

    Args:
        num_nodes: Number of nodes
        edges: List of (source, target) edges

    Returns:
        Tuple of (is_bipartite, set_a, set_b)
    """
    adj: dict[int, list[int]] = defaultdict(list)
    for src, tgt in edges:
        if 0 <= src < num_nodes and 0 <= tgt < num_nodes:
            adj[src].append(tgt)
            adj[tgt].append(src)

    color: dict[int, int] = {}
    set_a: list[int] = []
    set_b: list[int] = []

    for start in range(num_nodes):
        if start in color:
            continue

        queue = deque([start])
        color[start] = 0
        set_a.append(start)

        while queue:
            node = queue.popleft()
            node_color = color[node]
            next_color = 1 - node_color

            for neighbor in adj[node]:
                if neighbor in color:
                    if color[neighbor] == node_color:
                        return False, [], []
                else:
                    color[neighbor] = next_color
                    if next_color == 0:
                        set_a.append(neighbor)
                    else:
                        set_b.append(neighbor)
                    queue.append(neighbor)

    return True, set_a, set_b


def _merge_count_inversions(arr: list[int]) -> tuple[list[int], int]:
    """
    Count inversions in an array using merge sort.

    An inversion is a pair (i, j) where i < j but arr[i] > arr[j].
    This runs in O(n log n) time.

    Args:
        arr: Array of integers

    Returns:
        Tuple of (sorted_array, inversion_count)
    """
    n = len(arr)
    if n <= 1:
        return arr, 0

    mid = n // 2
    left, left_inv = _merge_count_inversions(arr[:mid])
    right, right_inv = _merge_count_inversions(arr[mid:])

    merged = []
    inversions = left_inv + right_inv
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            # All remaining elements in left are greater than right[j]
            inversions += len(left) - i
            j += 1

    merged.extend(left[i:])
    merged.extend(right[j:])

    return merged, inversions


def count_crossings(
    top_order: list[int],
    bottom_order: list[int],
    edges: list[tuple[int, int]],
) -> int:
    """
    Count the number of edge crossings in a bipartite layout.

    Uses O(m log m) algorithm based on inversion counting with merge sort,
    where m is the number of edges.

    Args:
        top_order: Order of nodes in top row (left to right)
        bottom_order: Order of nodes in bottom row (left to right)
        edges: List of edges between top and bottom sets

    Returns:
        Number of edge crossings
    """
    top_pos = {node: i for i, node in enumerate(top_order)}
    bottom_pos = {node: i for i, node in enumerate(bottom_order)}
    top_set = set(top_order)
    bottom_set = set(bottom_order)

    # Get edges as (top_position, bottom_position)
    edge_positions: list[tuple[int, int]] = []
    for src, tgt in edges:
        if src in top_set and tgt in bottom_set:
            edge_positions.append((top_pos[src], bottom_pos[tgt]))
        elif src in bottom_set and tgt in top_set:
            edge_positions.append((top_pos[tgt], bottom_pos[src]))

    if not edge_positions:
        return 0

    # Sort edges by top position, then extract bottom positions
    # Crossings = inversions in bottom positions after sorting by top
    edge_positions.sort(key=lambda x: (x[0], x[1]))
    bottom_positions = [b for _, b in edge_positions]

    # Count inversions in O(m log m)
    _, crossings = _merge_count_inversions(bottom_positions)

    return crossings


__all__ = [
    "BipartiteLayout",
    "is_bipartite",
    "count_crossings",
]
