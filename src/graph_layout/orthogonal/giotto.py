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
from ..planarity import PlanarityResult, check_planarity
from ..planarity.embedders import MaxFaceEmbedder, PlanarEmbedder
from ..preprocessing import connected_components
from ..types import (
    Event,
    GroupLike,
    LinkLike,
    NodeLike,
    SizeType,
)
from .compaction import CompactionResult, compact_layout
from .compaction_flow import compact_layout_flow, compact_layout_longest_path
from .edge_routing import route_all_edges
from .expansion import Expansion
from .orthogonalization import (
    Face,
    OrthogonalRepresentation,
    compute_faces,
    compute_orthogonal_representation,
)
from .planarization import is_planar_quick_check
from .realization import (
    bend_optimal_representation,
    pack_component_drawings,
    realize_bend_optimal_drawing,
)
from .types import NodeBox, OrthogonalEdge, Port, Side


class GIOTTOLayout(StaticLayout):
    """
    GIOTTO orthogonal layout for planar graphs.

    Produces bend-optimal orthogonal drawings where all edges consist of
    horizontal and vertical segments only. The bend-minimal drawing covers
    connected planar graphs: vertices of degree > 4 are expanded into cage
    rectangles (Kandinsky-style node boxes with several edges per side), and
    bridges / cut vertices are handled with per-corner angles.

    For graphs outside the strict degree-4 planar contract:
    - If strict=True (default): Raises ValueError
    - If strict=False: degree > 4 planar graphs still draw bend-optimally
      (via expansion); non-planar graphs fall back to Kandinsky-like
      heuristic routing

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
        embedder: Optional[PlanarEmbedder] = None,
        compaction_method: str = "greedy",
        bend_optimal: bool = True,
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
            embedder: Planar embedding strategy. Defaults to MaxFaceEmbedder.
            compaction_method: Method to use for compaction. Options:
                - "greedy": Greedy constraint-based compaction (default)
                - "flow": Flow-based compaction using min-cost flow
                - "longest_path": Longest-path compaction on constraint DAG
            bend_optimal: Drive the drawing from the bend-minimal orthogonal
                representation (Topology-Shape-Metrics with rectangularization;
                degree > 4 vertices via cage expansion). Default True;
                out-of-domain inputs (non-planar, disconnected) silently fall
                back to the heuristic router -- check ``used_bend_optimal``.
                Set False to force the heuristic router.
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
        self._embedder: PlanarEmbedder = embedder or MaxFaceEmbedder()
        self._compaction_method = compaction_method
        # When True (default), drive the drawing from the bend-minimal
        # orthogonal representation (Topology-Shape-Metrics) rather than the
        # geometric routing heuristic, whenever the representation is a
        # realizable shape. Rectangularization, per-corner angles (bridges /
        # cut vertices) and cage expansion (degree > 4) cover all connected
        # planar graphs; other inputs fall back to the heuristic.
        self._bend_optimal = bool(bend_optimal)
        # Whether the last run actually drew from the bend-minimal representation
        # (True) or fell back to the heuristic router (False). Out-of-domain
        # inputs (non-planar, disconnected) silently fall back.
        self._used_bend_optimal = False

        # Output data
        self._orthogonal_edges: list[OrthogonalEdge] = []
        self._node_boxes: list[NodeBox] = []
        self._orthogonal_rep: Optional[OrthogonalRepresentation] = None
        self._expansion: Optional[Expansion] = None
        self._faces: list[Face] = []
        self._compaction_result: Optional[CompactionResult] = None
        self._is_valid_input: bool = False
        self._planarity_result: Optional[PlanarityResult] = None

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
    def bend_optimal(self) -> bool:
        """Whether the bend-minimal drawing was *requested* (see also
        :attr:`used_bend_optimal` for whether it was actually applied)."""
        return self._bend_optimal

    @bend_optimal.setter
    def bend_optimal(self, value: bool) -> None:
        self._bend_optimal = bool(value)

    @property
    def used_bend_optimal(self) -> bool:
        """Whether the last :meth:`run` actually drew from the bend-minimal
        representation (True) rather than falling back to the heuristic router.

        Requesting ``bend_optimal=True`` does not guarantee it is used: the
        representation must be a realizable orthogonal shape, so out-of-domain
        inputs (non-planar, disconnected) fall back to the heuristic router.
        This flag lets callers detect that silent fallback -- e.g. to warn, or
        to decide whether the drawing is bend-minimal.
        """
        return self._used_bend_optimal

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
        Definitive planarity check using LR-planarity algorithm.

        Uses linear-time LR-planarity test to determine whether the graph
        is planar. Stores the PlanarityResult so the embedding can be reused.

        Returns:
            True if graph is planar, False otherwise.
        """
        n = len(self._nodes)
        m = len(self._links)

        # Quick Euler formula pre-check (O(1))
        if not is_planar_quick_check(n, m):
            return False

        # Build edge list
        edges: list[tuple[int, int]] = []
        for link in self._links:
            src = self._get_source_index(link)
            tgt = self._get_target_index(link)
            if 0 <= src < n and 0 <= tgt < n and src != tgt:
                edges.append((src, tgt))

        result = check_planarity(n, edges)
        self._planarity_result = result
        return result.is_planar

    # -------------------------------------------------------------------------
    # Layout Computation
    # -------------------------------------------------------------------------

    def _compute(self, **kwargs: Any) -> None:
        """Compute GIOTTO orthogonal layout."""
        # Reset per-run; only set True if the bend-optimal path drives the draw.
        self._used_bend_optimal = False
        self._expansion = None
        n = len(self._nodes)
        if n == 0:
            self._is_valid_input = True
            return

        # Validate degree constraint
        is_deg4, offending_node = self._validate_degree_4()
        if not is_deg4 and self._strict:
            raise ValueError(
                f"GIOTTO requires max degree 4, but node {offending_node} "
                f"has degree > 4. Use strict=False to fall back to "
                "Kandinsky-like behavior, or use KandinskyLayout instead."
            )

        # Validate planarity (linear-time LR-planarity test)
        is_planar = self._validate_planar()
        if not is_planar and self._strict:
            raise ValueError(
                f"GIOTTO requires a planar graph. The graph with {n} nodes "
                f"and {len(self._links)} edges is non-planar. "
                "Use strict=False to fall back to Kandinsky-like behavior."
            )

        self._is_valid_input = is_deg4 and is_planar

        # Phase 1: Assign layers (simple topological ordering)
        layers = self._assign_layers()

        # Phase 2: Position nodes on grid
        self._position_nodes(layers)

        # Phase 2.5: Disconnected planar graphs. A single planar embedding
        # spanning several components is ill-defined, so the shared realizer only
        # handles connected input. Draw each component bend-optimally in its own
        # frame and pack the drawings side by side; fall back (below) if any
        # component is out of the bend-optimal domain.
        if self._bend_optimal and is_planar:
            components = connected_components(
                n, self._links, self._get_source_index, self._get_target_index
            )
            if len(components) > 1 and self._draw_disconnected_bend_optimal(components):
                self._used_bend_optimal = True
                for i, box in enumerate(self._node_boxes):
                    if i < len(self._nodes):
                        self._nodes[i].x = box.x
                        self._nodes[i].y = box.y
                return

        # Phase 3+4: For planar inputs, compute the bend-minimal orthogonal
        # representation (expanding degree > 4 vertices into cages when
        # necessary) and, if requested and realizable, draw directly from it.
        # Degree > 4 without bend_optimal skips the representation entirely
        # (the plain flow model is infeasible there).
        if is_planar and (is_deg4 or self._bend_optimal):
            self._compute_orthogonal_rep()
            self._used_bend_optimal = (
                bool(self._bend_optimal) and self._apply_bend_optimal_drawing()
            )

        if not self._used_bend_optimal:
            # Heuristic router: the path for non-planar graphs, disabled
            # bend_optimal, and representations that turn out not realizable.
            self._route_edges()
            self._apply_compaction()

        # Update node positions from boxes
        for i, box in enumerate(self._node_boxes):
            if i < len(self._nodes):
                self._nodes[i].x = box.x
                self._nodes[i].y = box.y

    def _apply_bend_optimal_drawing(self) -> bool:
        """Draw from the orthogonal representation via the shared realizer.

        Delegates to :func:`realization.realize_bend_optimal_drawing`, which
        computes the shape and integer coordinates and scales them onto the
        canvas. Returns False -- leaving the heuristic pipeline to run -- when
        the representation is not a realizable shape, so the drawing never
        regresses.
        """
        n = len(self._nodes)
        if n == 0:
            return False

        link_endpoints = [
            (self._get_source_index(link), self._get_target_index(link)) for link in self._links
        ]
        node_sizes = [
            (
                float(getattr(self._nodes[i], "width", None) or self._node_width),
                float(getattr(self._nodes[i], "height", None) or self._node_height),
            )
            for i in range(n)
        ]
        cell = max(self._node_width, self._node_height) + self._node_separation

        result = realize_bend_optimal_drawing(
            num_nodes=n,
            link_endpoints=link_endpoints,
            node_sizes=node_sizes,
            orthogonal_rep=self._orthogonal_rep,
            faces=self._faces,
            expansion=self._expansion,
            canvas_size=self._canvas_size,
            cell=cell,
        )
        if result is None:
            return False
        self._node_boxes, self._orthogonal_edges = result
        return True

    def _draw_disconnected_bend_optimal(self, components: list[list[int]]) -> bool:
        """Draw a disconnected planar graph by packing per-component drawings.

        Each connected component is drawn bend-optimally in its own frame (via a
        recursive :class:`GIOTTOLayout` on the component's subgraph, or placed
        directly for an isolated vertex), then the drawings are shelf-packed into
        one non-overlapping layout. Returns False -- leaving the caller to fall
        back to the heuristic router for the whole graph -- if any non-trivial
        component is outside the bend-optimal domain, so ``used_bend_optimal``
        stays truthful.
        """
        n = len(self._nodes)
        drawings: list[tuple[list[NodeBox], list[OrthogonalEdge]]] = []
        for comp in components:
            comp_sorted = sorted(comp)
            result = self._draw_component(comp_sorted)
            if result is None:
                return False
            drawings.append(result)

        packed_boxes, packed_edges = pack_component_drawings(
            drawings, separation=self._node_separation, canvas_size=self._canvas_size
        )

        boxes_by_index: list[Optional[NodeBox]] = [None] * n
        for box in packed_boxes:
            if 0 <= box.index < n:
                boxes_by_index[box.index] = box
        if any(b is None for b in boxes_by_index):
            return False  # some vertex went undrawn -- fail safe

        self._node_boxes = [b for b in boxes_by_index if b is not None]
        self._orthogonal_edges = packed_edges
        return True

    def _draw_component(
        self, comp_sorted: list[int]
    ) -> Optional[tuple[list[NodeBox], list[OrthogonalEdge]]]:
        """Draw one connected component, returning (boxes, edges) in *global*
        vertex indices, or None if the component is not bend-optimal drawable."""
        if len(comp_sorted) == 1:
            # Isolated vertex: no edges, so zero bends -- trivially optimal.
            orig = comp_sorted[0]
            node = self._nodes[orig]
            width = float(getattr(node, "width", None) or self._node_width)
            height = float(getattr(node, "height", None) or self._node_height)
            return [NodeBox(index=orig, x=0.0, y=0.0, width=width, height=height)], []

        local_of = {orig: i for i, orig in enumerate(comp_sorted)}
        comp_set = set(comp_sorted)

        sub_nodes: list[dict[str, Any]] = []
        for orig in comp_sorted:
            node = self._nodes[orig]
            sub_nodes.append(
                {
                    "width": getattr(node, "width", None) or self._node_width,
                    "height": getattr(node, "height", None) or self._node_height,
                }
            )

        sub_links: list[dict[str, int]] = []
        for link in self._links:
            src = self._get_source_index(link)
            tgt = self._get_target_index(link)
            if src in comp_set and tgt in comp_set:
                sub_links.append({"source": local_of[src], "target": local_of[tgt]})

        child = GIOTTOLayout(
            nodes=sub_nodes,
            links=sub_links,
            size=self._canvas_size,
            node_width=self._node_width,
            node_height=self._node_height,
            node_separation=self._node_separation,
            edge_separation=self._edge_separation,
            layer_separation=self._layer_separation,
            strict=False,
            embedder=self._embedder,
            compaction_method=self._compaction_method,
            bend_optimal=True,
        )
        child.run()
        if not child.used_bend_optimal:
            return None

        k = len(comp_sorted)
        boxes = [
            NodeBox(
                index=comp_sorted[b.index],
                x=b.x,
                y=b.y,
                width=b.width,
                height=b.height,
            )
            for b in child.node_boxes
            if 0 <= b.index < k
        ]
        edges: list[OrthogonalEdge] = []
        for e in child.orthogonal_edges:
            if not (0 <= e.source < k and 0 <= e.target < k):
                continue
            gsrc = comp_sorted[e.source]
            gtgt = comp_sorted[e.target]
            edges.append(
                OrthogonalEdge(
                    source=gsrc,
                    target=gtgt,
                    source_port=Port(
                        node=gsrc,
                        side=e.source_port.side,
                        position=e.source_port.position,
                        edge=e.source_port.edge,
                    ),
                    target_port=Port(
                        node=gtgt,
                        side=e.target_port.side,
                        position=e.target_port.position,
                        edge=e.target_port.edge,
                    ),
                    bends=list(e.bends),
                )
            )
        return boxes, edges

    def _compute_orthogonal_rep(self) -> None:
        """
        Compute bend-optimal orthogonal representation.

        Uses the configured embedder to obtain a combinatorial planar
        embedding, then runs min-cost flow bend minimization on the
        resulting face structure. Vertices of degree > 4 are first expanded
        into cage cycles (see :mod:`.expansion`); the representation and faces
        then describe the expanded graph and ``self._expansion`` holds the
        mapping back to original vertices.
        """
        n = len(self._nodes)
        if n == 0 or not self._node_boxes:
            return

        edges: list[tuple[int, int]] = []
        for link in self._links:
            src = self._get_source_index(link)
            tgt = self._get_target_index(link)
            if 0 <= src < n and 0 <= tgt < n:
                edges.append((src, tgt))

        # Try to get a combinatorial embedding from the embedder
        embedding = None
        try:
            embedding = self._embedder.embed(n, edges, planarity_result=self._planarity_result)
        except (ValueError, AttributeError):
            pass

        if embedding is None:
            # Fallback: use position-based face computation
            positions = [(box.x, box.y) for box in self._node_boxes]
            self._orthogonal_rep = compute_orthogonal_representation(n, edges, positions)
            self._faces = compute_faces(n, edges, positions)
            return

        # Expanding degree > 4 vertices into cages is only useful when we intend
        # to draw from the representation (bend_optimal); otherwise keep the
        # plain rep (exposed via the ``orthogonal_rep`` property).
        rep, faces, expansion = bend_optimal_representation(
            n, edges, embedding, allow_expansion=self._bend_optimal
        )
        self._orthogonal_rep = rep
        self._faces = faces
        self._expansion = expansion

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

        # Compute longest path. The graph may be cyclic (orthogonal layouts are
        # usually drawn from undirected planar graphs), so guard against back
        # edges with an on-path stack -- otherwise a cycle drives depth upward
        # without bound and the DFS recurses forever.
        layer_assignment = [-1] * n
        visited = [False] * n
        on_path = [False] * n

        def dfs(node: int, depth: int) -> None:
            if on_path[node]:
                return  # back edge -> ignore to keep the layering acyclic
            if visited[node] and layer_assignment[node] >= depth:
                return
            visited[node] = True
            on_path[node] = True
            layer_assignment[node] = max(layer_assignment[node], depth)
            for neighbor in adj[node]:
                dfs(neighbor, depth + 1)
            on_path[node] = False

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
        """Route edges using global constraint-aware routing."""
        self._orthogonal_edges = []

        if not self._node_boxes:
            return

        n = len(self._nodes)

        edges: list[tuple[int, int]] = []
        edge_indices: list[int] = []
        for edge_idx, link in enumerate(self._links):
            src = self._get_source_index(link)
            tgt = self._get_target_index(link)
            if 0 <= src < n and 0 <= tgt < n:
                edges.append((src, tgt))
                edge_indices.append(edge_idx)

        self._orthogonal_edges = route_all_edges(
            boxes=self._node_boxes[:n],
            edges=edges,
            edge_indices=edge_indices,
            edge_separation=self._edge_separation,
        )

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

        if self._compaction_method == "flow":
            self._compaction_result = compact_layout_flow(
                boxes=original_boxes,
                edges=self._orthogonal_edges,
                node_separation=self._node_separation,
                layer_separation=self._layer_separation,
                edge_separation=self._edge_separation,
            )
        elif self._compaction_method == "longest_path":
            self._compaction_result = compact_layout_longest_path(
                boxes=original_boxes,
                edges=self._orthogonal_edges,
                node_separation=self._node_separation,
                layer_separation=self._layer_separation,
                edge_separation=self._edge_separation,
            )
        else:
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
