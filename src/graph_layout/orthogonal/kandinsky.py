"""
Kandinsky orthogonal layout algorithm.

Produces orthogonal drawings where edges use only horizontal and vertical
segments. Suitable for UML diagrams, ER diagrams, flowcharts, and circuit
schematics.

The Kandinsky model supports vertices of arbitrary degree (unlike simpler
orthogonal models limited to degree 4) by allowing multiple edges to exit
from the same side of a node.
"""

from __future__ import annotations

import math
from collections import defaultdict
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
from .types import (
    NodeBox,
    OrthogonalEdge,
    Port,
    RoutingGrid,
    Side,
)


class KandinskyLayout(StaticLayout):
    """
    Kandinsky orthogonal layout algorithm.

    Produces orthogonal drawings where all edges consist of horizontal
    and vertical segments only. This is ideal for diagrams where a
    structured, rectilinear appearance is desired.

    The algorithm works in three phases:
    1. **Placement**: Position nodes on a grid
    2. **Port Assignment**: Determine which side of each node edges exit from
    3. **Edge Routing**: Route edges using only horizontal/vertical segments

    This MVP implementation uses a simplified approach suitable for small
    to medium graphs. For optimal bend minimization, a full min-cost flow
    formulation would be needed.

    Example:
        layout = KandinskyLayout(
            nodes=[{}, {}, {}, {}],
            links=[
                {'source': 0, 'target': 1},
                {'source': 1, 'target': 2},
                {'source': 2, 'target': 3},
                {'source': 3, 'target': 0},
            ],
            size=(800, 600),
        )
        layout.run()

        # Access edge routing information
        for edge in layout.orthogonal_edges:
            print(f"Edge {edge.source}->{edge.target}: {len(edge.bends)} bends")

    Attributes:
        orthogonal_edges: List of OrthogonalEdge with routing information
        node_boxes: List of NodeBox with position and size information
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
        # Kandinsky-specific parameters
        node_width: float = 60,
        node_height: float = 40,
        node_separation: float = 60,
        edge_separation: float = 15,
        layer_separation: float = 80,
    ) -> None:
        """
        Initialize Kandinsky layout.

        Args:
            nodes: List of nodes
            links: List of links
            groups: List of groups
            size: Canvas size as (width, height)
            random_seed: Random seed for reproducible layouts
            on_start: Callback for start event
            on_tick: Callback for tick event
            on_end: Callback for end event
            node_width: Default width of node boxes
            node_height: Default height of node boxes
            node_separation: Minimum horizontal gap between nodes
            edge_separation: Minimum gap between parallel edge segments
            layer_separation: Vertical gap between layers (for hierarchical arrangement)
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

        # Kandinsky-specific configuration
        self._node_width = float(node_width)
        self._node_height = float(node_height)
        self._node_separation = float(node_separation)
        self._edge_separation = float(edge_separation)
        self._layer_separation = float(layer_separation)

        # Output data
        self._orthogonal_edges: list[OrthogonalEdge] = []
        self._node_boxes: list[NodeBox] = []
        self._routing_grid: Optional[RoutingGrid] = None

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def node_width(self) -> float:
        """Get default node width."""
        return self._node_width

    @node_width.setter
    def node_width(self, value: float) -> None:
        """Set default node width."""
        self._node_width = max(1.0, float(value))

    @property
    def node_height(self) -> float:
        """Get default node height."""
        return self._node_height

    @node_height.setter
    def node_height(self, value: float) -> None:
        """Set default node height."""
        self._node_height = max(1.0, float(value))

    @property
    def node_separation(self) -> float:
        """Get minimum gap between nodes."""
        return self._node_separation

    @node_separation.setter
    def node_separation(self, value: float) -> None:
        """Set minimum gap between nodes."""
        self._node_separation = max(0.0, float(value))

    @property
    def edge_separation(self) -> float:
        """Get minimum gap between parallel edges."""
        return self._edge_separation

    @edge_separation.setter
    def edge_separation(self, value: float) -> None:
        """Set minimum gap between parallel edges."""
        self._edge_separation = max(0.0, float(value))

    @property
    def layer_separation(self) -> float:
        """Get vertical gap between layers."""
        return self._layer_separation

    @layer_separation.setter
    def layer_separation(self, value: float) -> None:
        """Set vertical gap between layers."""
        self._layer_separation = max(0.0, float(value))

    @property
    def orthogonal_edges(self) -> list[OrthogonalEdge]:
        """Get orthogonal edge routing information."""
        return self._orthogonal_edges

    @property
    def node_boxes(self) -> list[NodeBox]:
        """Get node box information."""
        return self._node_boxes

    # -------------------------------------------------------------------------
    # Layout Computation
    # -------------------------------------------------------------------------

    def _compute(self, **kwargs: Any) -> None:
        """Compute Kandinsky orthogonal layout."""
        n = len(self._nodes)
        if n == 0:
            return

        # Phase 1: Assign layers (simple topological ordering)
        layers = self._assign_layers()

        # Phase 2: Position nodes on grid
        self._position_nodes(layers)

        # Phase 3: Assign ports and route edges
        self._route_edges()

        # Update node positions from boxes
        for i, box in enumerate(self._node_boxes):
            self._nodes[i].x = box.x
            self._nodes[i].y = box.y

    def _assign_layers(self) -> list[list[int]]:
        """
        Assign nodes to layers using longest path layering.

        Returns:
            List of layers, each containing node indices
        """
        n = len(self._nodes)
        if n == 0:
            return []

        # Build adjacency list
        adj: list[list[int]] = [[] for _ in range(n)]
        in_degree = [0] * n

        for link in self._links:
            src = self._get_source_index(link)
            tgt = self._get_target_index(link)
            if 0 <= src < n and 0 <= tgt < n and src != tgt:
                adj[src].append(tgt)
                in_degree[tgt] += 1

        # Find nodes with no incoming edges (roots)
        roots = [i for i in range(n) if in_degree[i] == 0]
        if not roots:
            # Graph has cycles, just use node 0 as root
            roots = [0]

        # Compute longest path from any root to each node
        layer_assignment = [-1] * n
        visited = [False] * n

        def dfs(node: int, depth: int) -> None:
            if visited[node] and layer_assignment[node] >= depth:
                return
            visited[node] = True
            layer_assignment[node] = max(layer_assignment[node], depth)
            for neighbor in adj[node]:
                dfs(neighbor, depth + 1)

        for root in roots:
            dfs(root, 0)

        # Assign unvisited nodes to layer 0
        for i in range(n):
            if layer_assignment[i] < 0:
                layer_assignment[i] = 0

        # Group nodes by layer
        max_layer = max(layer_assignment) if layer_assignment else 0
        layers: list[list[int]] = [[] for _ in range(max_layer + 1)]
        for node, layer in enumerate(layer_assignment):
            layers[layer].append(node)

        return layers

    def _position_nodes(self, layers: list[list[int]]) -> None:
        """
        Position nodes on a grid based on layer assignment.

        Args:
            layers: List of layers containing node indices
        """
        n = len(self._nodes)
        self._node_boxes = []

        if n == 0:
            return

        canvas_width, canvas_height = self._canvas_size

        # Calculate grid dimensions
        num_layers = len(layers)
        max_layer_size = max(len(layer) for layer in layers) if layers else 1

        # Calculate spacing
        cell_width = self._node_width + self._node_separation
        cell_height = self._node_height + self._layer_separation

        # Calculate total size needed
        total_width = max_layer_size * cell_width
        total_height = num_layers * cell_height

        # Calculate offsets to center the layout
        offset_x = (canvas_width - total_width) / 2 + cell_width / 2
        offset_y = (canvas_height - total_height) / 2 + cell_height / 2

        # Position nodes
        node_positions: dict[int, tuple[float, float]] = {}

        for layer_idx, layer in enumerate(layers):
            layer_width = len(layer) * cell_width
            layer_offset_x = (canvas_width - layer_width) / 2 + cell_width / 2

            for pos_in_layer, node_idx in enumerate(layer):
                x = layer_offset_x + pos_in_layer * cell_width
                y = offset_y + layer_idx * cell_height
                node_positions[node_idx] = (x, y)

        # Create node boxes
        for i in range(n):
            x, y = node_positions.get(i, (canvas_width / 2, canvas_height / 2))

            # Get node dimensions (use defaults or from node attributes)
            node = self._nodes[i]
            width = getattr(node, "width", None) or self._node_width
            height = getattr(node, "height", None) or self._node_height

            box = NodeBox(
                index=i,
                x=x,
                y=y,
                width=float(width),
                height=float(height),
            )
            self._node_boxes.append(box)

    def _route_edges(self) -> None:
        """Route edges between positioned nodes."""
        self._orthogonal_edges = []

        if not self._node_boxes:
            return

        # Group edges by source node to assign ports
        edges_by_source: dict[int, list[tuple[int, int]]] = defaultdict(list)
        edges_by_target: dict[int, list[tuple[int, int]]] = defaultdict(list)

        for edge_idx, link in enumerate(self._links):
            src = self._get_source_index(link)
            tgt = self._get_target_index(link)
            if 0 <= src < len(self._node_boxes) and 0 <= tgt < len(self._node_boxes):
                edges_by_source[src].append((edge_idx, tgt))
                edges_by_target[tgt].append((edge_idx, src))

        # Route each edge
        for edge_idx, link in enumerate(self._links):
            src = self._get_source_index(link)
            tgt = self._get_target_index(link)

            if not (0 <= src < len(self._node_boxes) and 0 <= tgt < len(self._node_boxes)):
                continue

            src_box = self._node_boxes[src]
            tgt_box = self._node_boxes[tgt]

            # Determine best sides based on relative positions
            src_side, tgt_side = self._determine_port_sides(src_box, tgt_box)

            # Calculate port positions
            src_port = Port(node=src, side=src_side, edge=edge_idx)
            tgt_port = Port(node=tgt, side=tgt_side, edge=edge_idx)

            # Get port coordinates
            src_pos = src_box.get_port_position(src_side)
            tgt_pos = tgt_box.get_port_position(tgt_side)

            # Route the edge with orthogonal segments
            bends = self._compute_edge_route(src_pos, tgt_pos, src_side, tgt_side)

            ortho_edge = OrthogonalEdge(
                source=src,
                target=tgt,
                source_port=src_port,
                target_port=tgt_port,
                bends=bends,
            )

            self._orthogonal_edges.append(ortho_edge)

    def _determine_port_sides(
        self, src_box: NodeBox, tgt_box: NodeBox
    ) -> tuple[Side, Side]:
        """
        Determine which sides of source and target nodes to use for an edge.

        Uses relative position heuristic: if target is below source, exit
        from south and enter from north, etc.

        Args:
            src_box: Source node box
            tgt_box: Target node box

        Returns:
            Tuple of (source_side, target_side)
        """
        dx = tgt_box.x - src_box.x
        dy = tgt_box.y - src_box.y

        # Prefer vertical connections for hierarchical layouts
        if abs(dy) > abs(dx) * 0.5:
            # Primarily vertical relationship
            if dy > 0:
                return (Side.SOUTH, Side.NORTH)
            else:
                return (Side.NORTH, Side.SOUTH)
        else:
            # Primarily horizontal relationship
            if dx > 0:
                return (Side.EAST, Side.WEST)
            else:
                return (Side.WEST, Side.EAST)

    def _compute_edge_route(
        self,
        src_pos: tuple[float, float],
        tgt_pos: tuple[float, float],
        src_side: Side,
        tgt_side: Side,
    ) -> list[tuple[float, float]]:
        """
        Compute bend points for an orthogonal edge route.

        Args:
            src_pos: Source port position (x, y)
            tgt_pos: Target port position (x, y)
            src_side: Side of source node
            tgt_side: Side of target node

        Returns:
            List of bend point coordinates
        """
        sx, sy = src_pos
        tx, ty = tgt_pos

        bends: list[tuple[float, float]] = []

        # Simple routing: determine if we need 0, 1, or 2 bends

        # Check if direct orthogonal connection is possible
        if src_side in (Side.NORTH, Side.SOUTH) and tgt_side in (Side.NORTH, Side.SOUTH):
            # Both vertical exits
            if abs(sx - tx) < 1e-6:
                # Aligned vertically, no bends needed
                pass
            else:
                # Need horizontal segment in middle
                mid_y = (sy + ty) / 2
                bends = [(sx, mid_y), (tx, mid_y)]

        elif src_side in (Side.EAST, Side.WEST) and tgt_side in (Side.EAST, Side.WEST):
            # Both horizontal exits
            if abs(sy - ty) < 1e-6:
                # Aligned horizontally, no bends needed
                pass
            else:
                # Need vertical segment in middle
                mid_x = (sx + tx) / 2
                bends = [(mid_x, sy), (mid_x, ty)]

        elif src_side in (Side.NORTH, Side.SOUTH) and tgt_side in (Side.EAST, Side.WEST):
            # Source vertical, target horizontal - one bend
            bends = [(sx, ty)]

        elif src_side in (Side.EAST, Side.WEST) and tgt_side in (Side.NORTH, Side.SOUTH):
            # Source horizontal, target vertical - one bend
            bends = [(tx, sy)]

        else:
            # Same direction exits (e.g., both SOUTH) - need detour
            # This is a simplified approach; full Kandinsky would optimize this
            offset = self._edge_separation * 2

            if src_side == Side.SOUTH and tgt_side == Side.SOUTH:
                detour_y = max(sy, ty) + offset
                bends = [(sx, detour_y), (tx, detour_y)]
            elif src_side == Side.NORTH and tgt_side == Side.NORTH:
                detour_y = min(sy, ty) - offset
                bends = [(sx, detour_y), (tx, detour_y)]
            elif src_side == Side.EAST and tgt_side == Side.EAST:
                detour_x = max(sx, tx) + offset
                bends = [(detour_x, sy), (detour_x, ty)]
            elif src_side == Side.WEST and tgt_side == Side.WEST:
                detour_x = min(sx, tx) - offset
                bends = [(detour_x, sy), (detour_x, ty)]

        return bends


__all__ = ["KandinskyLayout"]
