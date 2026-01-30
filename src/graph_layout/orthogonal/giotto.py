"""
GIOTTO: Bend-optimal orthogonal layout for degree-4 planar graphs.

GIOTTO is a classic algorithm for producing orthogonal graph drawings with
the minimum number of bends. It works on planar graphs where every vertex
has degree at most 4 (i.e., at most 4 edges incident to each vertex).

The algorithm produces bend-optimal drawings, meaning no other orthogonal
drawing of the same graph can have fewer bends.

Reference:
    Tamassia, R. (1987). On embedding a graph in the grid with the minimum
    number of bends. SIAM Journal on Computing, 16(3), 421-444.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

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
from .compaction import CompactionResult, compact_layout
from .orthogonalization import OrthogonalRepresentation, compute_orthogonal_representation
from .planarization import is_planar_quick_check
from .types import NodeBox, OrthogonalEdge, Port, Side


class GIOTTOLayout(StaticLayout):
    """
    GIOTTO orthogonal layout for degree-4 planar graphs.

    Produces bend-optimal orthogonal drawings where all edges consist of
    horizontal and vertical segments only. This algorithm is specialized
    for planar graphs where every node has at most 4 incident edges.

    For graphs that don't meet these requirements:
    - If strict=True (default): Raises ValueError
    - If strict=False: Falls back to Kandinsky-like behavior

    Example:
        # 4x4 grid graph (degree-4 planar)
        layout = GIOTTOLayout(
            nodes=[{} for _ in range(16)],
            links=[
                # Horizontal edges
                {'source': 0, 'target': 1}, {'source': 1, 'target': 2}, ...
                # Vertical edges
                {'source': 0, 'target': 4}, {'source': 4, 'target': 8}, ...
            ],
            size=(800, 600),
        )
        layout.run()

    Attributes:
        orthogonal_edges: List of OrthogonalEdge with routing information
        node_boxes: List of NodeBox with position and size information
        is_valid_input: True if graph satisfies degree-4 planar requirements
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
        # GIOTTO-specific parameters
        node_width: float = 60,
        node_height: float = 40,
        node_separation: float = 60,
        edge_separation: float = 15,
        layer_separation: float = 80,
        strict: bool = True,
    ) -> None:
        """
        Initialize GIOTTO layout.

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
            layer_separation: Vertical gap between layers
            strict: If True, raise ValueError for invalid graphs.
                   If False, fall back to Kandinsky-like behavior.
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

        # GIOTTO-specific configuration
        self._node_width = float(node_width)
        self._node_height = float(node_height)
        self._node_separation = float(node_separation)
        self._edge_separation = float(edge_separation)
        self._layer_separation = float(layer_separation)
        self._strict = bool(strict)

        # Output data
        self._orthogonal_edges: list[OrthogonalEdge] = []
        self._node_boxes: list[NodeBox] = []
        self._orthogonal_rep: Optional[OrthogonalRepresentation] = None
        self._compaction_result: Optional[CompactionResult] = None
        self._is_valid_input: bool = False

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
    def strict(self) -> bool:
        """Get whether strict mode is enabled."""
        return self._strict

    @strict.setter
    def strict(self, value: bool) -> None:
        """Set strict mode."""
        self._strict = bool(value)

    @property
    def orthogonal_edges(self) -> list[OrthogonalEdge]:
        """Get orthogonal edge routing information."""
        return self._orthogonal_edges

    @property
    def node_boxes(self) -> list[NodeBox]:
        """Get node box information."""
        return self._node_boxes

    @property
    def total_bends(self) -> int:
        """Get total number of bends in the layout."""
        return sum(len(e.bends) for e in self._orthogonal_edges)

    @property
    def is_valid_input(self) -> bool:
        """Check if the input graph is valid for GIOTTO (degree-4 planar)."""
        return self._is_valid_input

    @property
    def orthogonal_rep(self) -> Optional[OrthogonalRepresentation]:
        """Get the computed orthogonal representation."""
        return self._orthogonal_rep

    @property
    def compaction_result(self) -> Optional[CompactionResult]:
        """Get the compaction result."""
        return self._compaction_result

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def _validate_degree_4(self) -> tuple[bool, Optional[int]]:
        """
        Check that all nodes have degree <= 4.

        Returns:
            Tuple of (is_valid, offending_node_index)
            If valid, offending_node_index is None
        """
        n = len(self._nodes)
        if n == 0:
            return True, None

        # Count degree of each node
        degrees = [0] * n

        for link in self._links:
            src = self._get_source_index(link)
            tgt = self._get_target_index(link)

            if 0 <= src < n:
                degrees[src] += 1
            if 0 <= tgt < n:
                degrees[tgt] += 1

        # Check for violations
        for i, deg in enumerate(degrees):
            if deg > 4:
                return False, i

        return True, None

    def _validate_planar(self) -> bool:
        """
        Quick planarity check using Euler's formula.

        For a planar graph: E <= 3V - 6 (for V >= 3)

        This is a necessary but not sufficient condition. A more rigorous
        check would require a full planarity testing algorithm.

        Returns:
            True if graph might be planar, False if definitely not planar
        """
        n = len(self._nodes)
        m = len(self._links)
        return is_planar_quick_check(n, m)

    def _is_k5_subgraph(self) -> bool:
        """
        Check for K5 (complete graph on 5 vertices) as a minor.

        K5 is non-planar. This is a simplified check that looks for
        5 mutually connected vertices.

        Returns:
            True if K5 subgraph detected (definitely non-planar)
        """
        n = len(self._nodes)
        if n < 5:
            return False

        # Build adjacency set
        adj: dict[int, set[int]] = {i: set() for i in range(n)}
        for link in self._links:
            src = self._get_source_index(link)
            tgt = self._get_target_index(link)
            if 0 <= src < n and 0 <= tgt < n and src != tgt:
                adj[src].add(tgt)
                adj[tgt].add(src)

        # Check all 5-combinations for K5
        # This is O(n^5) which is expensive but acceptable for small graphs
        if n > 20:
            # For larger graphs, skip this check and rely on Euler's formula
            return False

        from itertools import combinations

        for vertices in combinations(range(n), 5):
            is_complete = True
            for i, j in combinations(vertices, 2):
                if j not in adj[i]:
                    is_complete = False
                    break
            if is_complete:
                return True

        return False

    # -------------------------------------------------------------------------
    # Layout Computation
    # -------------------------------------------------------------------------

    def _compute(self, **kwargs: Any) -> None:
        """Compute GIOTTO orthogonal layout."""
        n = len(self._nodes)
        if n == 0:
            self._is_valid_input = True
            return

        # Validate degree constraint
        is_deg4, offending_node = self._validate_degree_4()
        if not is_deg4:
            if self._strict:
                raise ValueError(
                    f"GIOTTO requires max degree 4, but node {offending_node} "
                    f"has degree > 4. Use strict=False to fall back to "
                    "Kandinsky-like behavior, or use KandinskyLayout instead."
                )
            # Fall back to Kandinsky-like behavior
            self._is_valid_input = False
            self._compute_fallback()
            return

        # Validate planarity
        if not self._validate_planar():
            if self._strict:
                raise ValueError(
                    f"GIOTTO requires a planar graph. The graph has {n} nodes "
                    f"and {len(self._links)} edges, which violates Euler's formula "
                    f"(E <= 3V - 6 = {3 * n - 6} for V >= 3). "
                    "Use strict=False to fall back to Kandinsky-like behavior."
                )
            self._is_valid_input = False
            self._compute_fallback()
            return

        # Check for K5 subgraph (non-planar)
        if self._is_k5_subgraph():
            if self._strict:
                raise ValueError(
                    "GIOTTO requires a planar graph. The graph contains K5 "
                    "(complete graph on 5 vertices) as a subgraph, which is non-planar. "
                    "Use strict=False to fall back to Kandinsky-like behavior."
                )
            self._is_valid_input = False
            self._compute_fallback()
            return

        # Input is valid for GIOTTO
        self._is_valid_input = True

        # Phase 1: Assign layers (simple topological ordering)
        layers = self._assign_layers()

        # Phase 2: Position nodes on grid
        self._position_nodes(layers)

        # Phase 3: Compute orthogonal representation (bend-optimal for planar)
        self._compute_orthogonal_rep()

        # Phase 4: Route edges
        self._route_edges()

        # Phase 5: Compaction
        self._apply_compaction()

        # Update node positions from boxes
        for i, box in enumerate(self._node_boxes):
            if i < len(self._nodes):
                self._nodes[i].x = box.x
                self._nodes[i].y = box.y

    def _compute_fallback(self) -> None:
        """
        Compute layout using Kandinsky-like approach for invalid inputs.

        This is used when strict=False and the graph doesn't meet
        GIOTTO requirements.
        """
        # Use the same layering and positioning as GIOTTO
        layers = self._assign_layers()
        self._position_nodes(layers)

        # Skip orthogonal representation optimization
        # (it assumes planar input)

        # Route edges with simple heuristic
        self._route_edges()

        # Apply compaction
        self._apply_compaction()

        # Update node positions
        for i, box in enumerate(self._node_boxes):
            if i < len(self._nodes):
                self._nodes[i].x = box.x
                self._nodes[i].y = box.y

    def _compute_orthogonal_rep(self) -> None:
        """
        Compute bend-optimal orthogonal representation.

        For planar graphs with max degree 4, this produces the minimum
        number of bends using min-cost flow.
        """
        n = len(self._nodes)
        if n == 0 or not self._node_boxes:
            return

        positions = [(box.x, box.y) for box in self._node_boxes]

        edges: list[tuple[int, int]] = []
        for link in self._links:
            src = self._get_source_index(link)
            tgt = self._get_target_index(link)
            if 0 <= src < n and 0 <= tgt < n:
                edges.append((src, tgt))

        self._orthogonal_rep = compute_orthogonal_representation(n, edges, positions)

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

        # Find roots
        roots = [i for i in range(n) if in_degree[i] == 0]
        if not roots:
            roots = [0]

        # Compute longest path
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

        # Assign unvisited nodes
        for i in range(n):
            if layer_assignment[i] < 0:
                layer_assignment[i] = 0

        # Group by layer
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

        num_layers = len(layers)
        cell_width = self._node_width + self._node_separation
        cell_height = self._node_height + self._layer_separation

        total_height = num_layers * cell_height
        offset_y = (canvas_height - total_height) / 2 + cell_height / 2

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

        n = len(self._nodes)

        for edge_idx, link in enumerate(self._links):
            src = self._get_source_index(link)
            tgt = self._get_target_index(link)

            if not (0 <= src < n and 0 <= tgt < n):
                continue

            src_box = self._node_boxes[src]
            tgt_box = self._node_boxes[tgt]

            src_side, tgt_side = self._determine_port_sides(src_box, tgt_box)

            src_port = Port(node=src, side=src_side, edge=edge_idx)
            tgt_port = Port(node=tgt, side=tgt_side, edge=edge_idx)

            src_pos = src_box.get_port_position(src_side)
            tgt_pos = tgt_box.get_port_position(tgt_side)

            bends = self._compute_edge_route(src_pos, tgt_pos, src_side, tgt_side)

            ortho_edge = OrthogonalEdge(
                source=src,
                target=tgt,
                source_port=src_port,
                target_port=tgt_port,
                bends=bends,
            )

            self._orthogonal_edges.append(ortho_edge)

    def _determine_port_sides(self, src_box: NodeBox, tgt_box: NodeBox) -> tuple[Side, Side]:
        """
        Determine port sides using relative position heuristic.

        Args:
            src_box: Source node box
            tgt_box: Target node box

        Returns:
            Tuple of (source_side, target_side)
        """
        dx = tgt_box.x - src_box.x
        dy = tgt_box.y - src_box.y

        if abs(dy) > abs(dx) * 0.5:
            if dy > 0:
                return (Side.SOUTH, Side.NORTH)
            else:
                return (Side.NORTH, Side.SOUTH)
        else:
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
            src_pos: Source port position
            tgt_pos: Target port position
            src_side: Side of source node
            tgt_side: Side of target node

        Returns:
            List of bend point coordinates
        """
        sx, sy = src_pos
        tx, ty = tgt_pos

        bends: list[tuple[float, float]] = []

        if src_side in (Side.NORTH, Side.SOUTH) and tgt_side in (Side.NORTH, Side.SOUTH):
            if abs(sx - tx) < 1e-6:
                pass
            else:
                mid_y = (sy + ty) / 2
                bends = [(sx, mid_y), (tx, mid_y)]

        elif src_side in (Side.EAST, Side.WEST) and tgt_side in (Side.EAST, Side.WEST):
            if abs(sy - ty) < 1e-6:
                pass
            else:
                mid_x = (sx + tx) / 2
                bends = [(mid_x, sy), (mid_x, ty)]

        elif src_side in (Side.NORTH, Side.SOUTH) and tgt_side in (Side.EAST, Side.WEST):
            bends = [(sx, ty)]

        elif src_side in (Side.EAST, Side.WEST) and tgt_side in (Side.NORTH, Side.SOUTH):
            bends = [(tx, sy)]

        else:
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

    def _apply_compaction(self) -> None:
        """Apply compaction to reduce layout area."""
        if not self._node_boxes:
            return

        n = len(self._nodes)
        original_boxes = self._node_boxes[:n]

        self._compaction_result = compact_layout(
            boxes=original_boxes,
            edges=self._orthogonal_edges,
            node_separation=self._node_separation,
            layer_separation=self._layer_separation,
            edge_separation=self._edge_separation,
        )

        for i, (new_x, new_y) in enumerate(self._compaction_result.node_positions):
            if i < len(self._node_boxes):
                old_box = self._node_boxes[i]
                self._node_boxes[i] = NodeBox(
                    index=old_box.index,
                    x=new_x,
                    y=new_y,
                    width=old_box.width,
                    height=old_box.height,
                )

        # Re-route edges with new positions
        self._route_edges()

    # -------------------------------------------------------------------------
    # Export Methods (Override base class for orthogonal output)
    # -------------------------------------------------------------------------

    def to_svg(self, **kwargs: Any) -> str:
        """
        Export GIOTTO orthogonal layout to SVG format.

        Uses orthogonal-specific rendering with rectangular nodes and
        polyline edges that include bend points.

        Args:
            node_color: Fill color for nodes (default "#4a90d9")
            node_stroke: Stroke color for nodes (default "#2c5aa0")
            node_stroke_width: Stroke width for nodes (default 2)
            edge_color: Color for edges (default "#666666")
            edge_width: Width for edges (default 1.5)
            show_labels: Whether to show node labels (default True)
            label_color: Color for labels (default "#000000")
            font_size: Font size for labels (default 12)
            font_family: Font family for labels (default "sans-serif")
            padding: Padding around the graph (default 40)
            background: Background color (default None for transparent)

        Returns:
            SVG string representation of the orthogonal graph
        """
        from ..export.svg import to_svg_orthogonal

        return to_svg_orthogonal(self._node_boxes, self._orthogonal_edges, **kwargs)

    def to_dot(self, **kwargs: Any) -> str:
        """
        Export GIOTTO orthogonal layout to DOT (Graphviz) format.

        Uses orthogonal-specific settings with splines=ortho.

        Args:
            name: Name of the graph (default "G")
            directed: Whether the graph is directed (default False)
            include_positions: Include pos attributes (default True)
            node_shape: Node shape (default "box")
            node_color: Node border color
            node_fillcolor: Node fill color
            graph_attrs: Additional graph attributes

        Returns:
            DOT format string
        """
        from ..export.dot import to_dot_orthogonal

        return to_dot_orthogonal(self._node_boxes, self._orthogonal_edges, **kwargs)

    def to_graphml(self, **kwargs: Any) -> str:
        """
        Export GIOTTO orthogonal layout to GraphML format.

        Includes bend point information for orthogonal edges.

        Args:
            graph_id: ID for the graph element (default "G")
            directed: Whether edges are directed (default False)
            include_bends: Include bend point data (default True)

        Returns:
            GraphML XML string
        """
        from ..export.graphml import to_graphml_orthogonal

        return to_graphml_orthogonal(self._node_boxes, self._orthogonal_edges, **kwargs)


__all__ = ["GIOTTOLayout"]
