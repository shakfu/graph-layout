"""
Sugiyama layered graph layout algorithm.

Based on the framework from:
"Methods for Visual Understanding of Hierarchical System Structures"
by Sugiyama, Tagawa, and Toda (1981)

This algorithm produces layered layouts for directed graphs (DAGs)
with the following phases:
1. Cycle removal (if needed)
2. Layer assignment
3. Crossing minimization
4. Horizontal coordinate assignment
"""

from __future__ import annotations

import warnings
from collections import deque
from typing import Any, Callable, Optional, Sequence

from ..base import StaticLayout
from ..preprocessing import remove_cycles
from ..types import (
    Event,
    GroupLike,
    LinkLike,
    NodeLike,
    SizeType,
)


class GraphStructureWarning(UserWarning):
    """Warning issued when graph structure doesn't match algorithm assumptions."""

    pass


class SugiyamaLayout(StaticLayout):
    """
    Sugiyama layered DAG layout.

    Arranges nodes in horizontal layers with edges flowing downward.
    Minimizes edge crossings and produces aesthetically pleasing layouts.

    Example:
        layout = SugiyamaLayout(
            nodes=[{}, {}, {}, {}, {}],
            links=[
                {'source': 0, 'target': 1},
                {'source': 0, 'target': 2},
                {'source': 1, 'target': 3},
                {'source': 2, 'target': 3},
                {'source': 3, 'target': 4},
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
        # Sugiyama-specific parameters
        layer_separation: float = 100.0,
        node_separation: float = 50.0,
        orientation: str = "top-to-bottom",
        crossing_iterations: int = 24,
    ) -> None:
        """
        Initialize Sugiyama layout.

        Args:
            nodes: List of nodes
            links: List of links
            groups: List of groups
            size: Canvas size as (width, height)
            random_seed: Random seed for reproducible layouts
            on_start: Callback for start event
            on_tick: Callback for tick event
            on_end: Callback for end event
            layer_separation: Vertical separation between layers.
            node_separation: Horizontal separation between nodes in same layer.
            orientation: Layout direction - 'top-to-bottom', 'bottom-to-top',
                'left-to-right', or 'right-to-left'.
            crossing_iterations: Number of iterations for crossing minimization.
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

        # Sugiyama-specific configuration
        self._layer_separation: float = float(layer_separation)
        self._node_separation: float = float(node_separation)
        valid_orientations = {"top-to-bottom", "bottom-to-top", "left-to-right", "right-to-left"}
        if orientation not in valid_orientations:
            raise ValueError(f"orientation must be one of {valid_orientations}")
        self._orientation: str = orientation
        self._crossing_iterations: int = max(1, int(crossing_iterations))

        # Internal state
        self._layers: list[list[int]] = []
        self._node_layer: dict[int, int] = {}
        self._node_position: dict[int, int] = {}

        # Dummy-node bookkeeping (virtual nodes on edges that span >1 layer).
        # Dummy ids are integers >= len(nodes) so they never collide with real
        # node indices. Segments are the adjacent-layer edges of the expanded
        # graph (long edges replaced by dummy chains) that crossing
        # minimization and coordinate assignment operate on.
        self._dummy_nodes: set[int] = set()
        self._segments: list[tuple[int, int]] = []
        self._dummy_pos: dict[int, tuple[float, float]] = {}
        # Original (acyclic) edge -> ordered list of dummy ids it passes through.
        self._edge_chains: dict[tuple[int, int], list[int]] = {}
        # Original (acyclic) edge -> ordered list of (x, y) bend points, one per
        # dummy the edge passes through. Empty for edges between adjacent layers.
        self._edge_bends: dict[tuple[int, int], list[tuple[float, float]]] = {}

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def layer_separation(self) -> float:
        """Get vertical separation between layers."""
        return self._layer_separation

    @layer_separation.setter
    def layer_separation(self, value: float) -> None:
        """Set vertical separation between layers."""
        self._layer_separation = float(value)

    @property
    def node_separation(self) -> float:
        """Get horizontal separation between nodes in same layer."""
        return self._node_separation

    @node_separation.setter
    def node_separation(self, value: float) -> None:
        """Set horizontal separation between nodes in same layer."""
        self._node_separation = float(value)

    @property
    def orientation(self) -> str:
        """Get layout orientation."""
        return self._orientation

    @orientation.setter
    def orientation(self, value: str) -> None:
        """Set layout orientation."""
        valid = {"top-to-bottom", "bottom-to-top", "left-to-right", "right-to-left"}
        if value not in valid:
            raise ValueError(f"orientation must be one of {valid}")
        self._orientation = value

    @property
    def crossing_iterations(self) -> int:
        """Get number of crossing minimization iterations."""
        return self._crossing_iterations

    @crossing_iterations.setter
    def crossing_iterations(self, value: int) -> None:
        """Set number of crossing minimization iterations."""
        self._crossing_iterations = max(1, int(value))

    @property
    def edge_bends(self) -> dict[tuple[int, int], list[tuple[float, float]]]:
        """Bend points for edges that span more than one layer.

        Maps each such edge (as an acyclic ``(source, target)`` index pair) to
        the ordered list of ``(x, y)`` points where it passes through a dummy
        node between its endpoints. Edges between adjacent layers do not appear.
        Populated after :meth:`run`.
        """
        return self._edge_bends

    # -------------------------------------------------------------------------
    # Phase 1: Layer Assignment (Longest Path)
    # -------------------------------------------------------------------------

    def _build_acyclic_edges(self) -> list[tuple[int, int]]:
        """Normalize links into a self-loop-free, acyclic edge list.

        This is the cycle-removal phase (previously declared but never invoked):
        back edges are reversed so the remaining phases see a DAG in which every
        edge points strictly downward. Self-loops are dropped -- they have no
        meaning in a layered drawing and would otherwise push a node to the
        bottom layer.
        """
        n = len(self._nodes)
        raw: list[dict[str, int]] = []
        for link in self._links:
            src = self._get_source_index(link)
            tgt = self._get_target_index(link)
            if 0 <= src < n and 0 <= tgt < n and src != tgt:
                raw.append({"source": src, "target": tgt})

        new_links, _reversed = remove_cycles(
            n,
            raw,
            get_source=lambda link: link["source"],
            get_target=lambda link: link["target"],
        )

        edges: list[tuple[int, int]] = []
        for new_link in new_links:
            src = new_link["source"]
            tgt = new_link["target"]
            if src != tgt:
                edges.append((src, tgt))
        return edges

    def _assign_layers(self, edges: list[tuple[int, int]]) -> None:
        """Assign nodes to layers using the longest-path algorithm.

        Operates on the acyclic ``edges`` from :meth:`_build_acyclic_edges`, so
        every edge is guaranteed to point downward and the longest path is at
        most ``n - 1`` edges.
        """
        n = len(self._nodes)
        if n == 0:
            return

        # Build adjacency lists
        outgoing: list[list[int]] = [[] for _ in range(n)]
        incoming: list[list[int]] = [[] for _ in range(n)]

        for src, tgt in edges:
            outgoing[src].append(tgt)
            incoming[tgt].append(src)

        # Find sources (nodes with no incoming edges)
        sources = [i for i in range(n) if not incoming[i]]
        if not sources:
            warnings.warn(
                "No source nodes found (all nodes have incoming edges). "
                "This suggests the graph contains cycles. "
                "Sugiyama layout is designed for DAGs; results may be suboptimal.",
                GraphStructureWarning,
                stacklevel=3,
            )
            sources = [0]  # Fallback

        # Compute longest path from any source
        # Cap at n layers to guarantee termination on cyclic graphs.
        # In a DAG with n nodes the longest path has at most n-1 edges,
        # so any layer >= n is proof of a cycle.
        max_layer = n - 1
        layers = [-1] * n
        queue: deque[int] = deque()

        for src in sources:
            layers[src] = 0
            queue.append(src)

        while queue:
            node = queue.popleft()
            for child in outgoing[node]:
                new_layer = layers[node] + 1
                if new_layer > layers[child] and new_layer <= max_layer:
                    layers[child] = new_layer
                    queue.append(child)

        # Handle disconnected nodes
        disconnected = [i for i in range(n) if layers[i] < 0]
        if disconnected:
            warnings.warn(
                f"Found {len(disconnected)} disconnected node(s) unreachable from sources. "
                "These will be placed in layer 0.",
                GraphStructureWarning,
                stacklevel=3,
            )
            for i in disconnected:
                layers[i] = 0

        # Group nodes by layer
        max_layer = max(layers) if layers else 0
        self._layers = [[] for _ in range(max_layer + 1)]

        for i in range(n):
            layer = layers[i]
            self._layers[layer].append(i)
            self._node_layer[i] = layer

    # -------------------------------------------------------------------------
    # Phase 2a: Dummy-Node Insertion (proper layering)
    # -------------------------------------------------------------------------

    def _insert_dummy_nodes(self, edges: list[tuple[int, int]]) -> None:
        """Split every edge that spans more than one layer into a dummy chain.

        Without this, crossing minimization would average the positions of
        neighbors that live several layers apart (positions in different layers
        are not comparable), long edges would exert no ordering force on the
        layers they pass through, and crossings involving them would be
        uncounted. After this step every segment connects adjacent layers, so
        the barycenter heuristic is well defined.
        """
        n = len(self._nodes)
        self._dummy_nodes = set()
        self._segments = []
        self._edge_bends = {}

        next_id = n
        chains: dict[tuple[int, int], list[int]] = {}

        for u, v in edges:
            lu = self._node_layer[u]
            lv = self._node_layer[v]
            # Longest-path layering on the acyclic edge set guarantees lv > lu;
            # normalize defensively in case an edge is level or reversed.
            if lv < lu:
                u, v, lu, lv = v, u, lv, lu

            if lv - lu <= 1:
                self._segments.append((u, v))
                continue

            chain = [u]
            prev = u
            for layer in range(lu + 1, lv):
                dummy = next_id
                next_id += 1
                self._dummy_nodes.add(dummy)
                self._node_layer[dummy] = layer
                self._layers[layer].append(dummy)
                self._segments.append((prev, dummy))
                chain.append(dummy)
                prev = dummy
            self._segments.append((prev, v))
            chain.append(v)
            chains[(u, v)] = chain

        self._edge_chains = chains

    # -------------------------------------------------------------------------
    # Phase 2b: Crossing Minimization (Barycenter Method)
    # -------------------------------------------------------------------------

    def _minimize_crossings(self) -> None:
        """Minimize edge crossings using the barycenter heuristic.

        Operates on the expanded graph (real nodes plus dummies) via the
        adjacent-layer ``self._segments``, and keeps the ordering with the
        fewest crossings seen across sweeps rather than whatever the last sweep
        happens to produce (barycenter sweeps can oscillate).
        """
        if len(self._layers) < 2:
            return

        # Segment adjacency (dict-keyed: dummy ids exceed the real-node range)
        outgoing: dict[int, list[int]] = {}
        incoming: dict[int, list[int]] = {}
        for src, tgt in self._segments:
            outgoing.setdefault(src, []).append(tgt)
            incoming.setdefault(tgt, []).append(src)

        # Initialize positions from current layer order
        for layer in self._layers:
            for pos, node in enumerate(layer):
                self._node_position[node] = pos

        best_layers = [list(layer) for layer in self._layers]
        best_crossings = self._count_segment_crossings(outgoing)

        # Iterate: alternate sweeping down and up
        for iteration in range(self._crossing_iterations):
            if iteration % 2 == 0:
                for layer_idx in range(1, len(self._layers)):
                    self._order_layer_by_barycenter(layer_idx, incoming)
            else:
                for layer_idx in range(len(self._layers) - 2, -1, -1):
                    self._order_layer_by_barycenter(layer_idx, outgoing)

            crossings = self._count_segment_crossings(outgoing)
            if crossings < best_crossings:
                best_crossings = crossings
                best_layers = [list(layer) for layer in self._layers]

        # Restore the best ordering found.
        self._layers = best_layers
        for layer in self._layers:
            for pos, node in enumerate(layer):
                self._node_position[node] = pos

    def _order_layer_by_barycenter(self, layer_idx: int, adj: dict[int, list[int]]) -> None:
        """Reorder a layer based on the barycenter of its neighbors."""
        layer = self._layers[layer_idx]
        if not layer:
            return

        barycenters: list[tuple[float, int]] = []
        for node in layer:
            neighbors = adj.get(node)
            if neighbors:
                positions = [self._node_position.get(nb, 0) for nb in neighbors]
                barycenter = sum(positions) / len(positions)
            else:
                # No neighbors in the adjacent layer: keep current position.
                barycenter = float(self._node_position.get(node, 0))
            barycenters.append((barycenter, node))

        # Stable sort by barycenter preserves relative order on ties.
        barycenters.sort(key=lambda x: x[0])

        self._layers[layer_idx] = [node for _, node in barycenters]
        for pos, (_, node) in enumerate(barycenters):
            self._node_position[node] = pos

    def _count_segment_crossings(self, outgoing: dict[int, list[int]]) -> int:
        """Count crossings between all pairs of adjacent layers.

        With dummy nodes every segment connects consecutive layers, so this is
        the standard inversion count on endpoint positions and is exact for the
        current ordering.
        """
        total = 0
        for layer_idx in range(len(self._layers) - 1):
            # Collect (upper_pos, lower_pos) for every segment leaving this layer.
            seg_positions: list[tuple[int, int]] = []
            for node in self._layers[layer_idx]:
                up = self._node_position.get(node, 0)
                for tgt in outgoing.get(node, []):
                    if self._node_layer.get(tgt) == layer_idx + 1:
                        seg_positions.append((up, self._node_position.get(tgt, 0)))
            # Count pairs (a, b) that cross: a.up < b.up but a.low > b.low.
            for i in range(len(seg_positions)):
                ui, li = seg_positions[i]
                for j in range(i + 1, len(seg_positions)):
                    uj, lj = seg_positions[j]
                    if (ui < uj and li > lj) or (ui > uj and li < lj):
                        total += 1
        return total

    # -------------------------------------------------------------------------
    # Phase 3: Coordinate Assignment
    # -------------------------------------------------------------------------

    def _assign_coordinates(self) -> None:
        """Assign final x,y coordinates to nodes."""
        if not self._layers:
            return

        padding = 50
        canvas_w = self._canvas_size[0] - 2 * padding
        canvas_h = self._canvas_size[1] - 2 * padding

        n_layers = len(self._layers)
        max_layer_size = max(len(layer) for layer in self._layers)

        # Calculate separation based on canvas size
        if n_layers > 1:
            layer_sep = min(self._layer_separation, canvas_h / (n_layers - 1))
        else:
            layer_sep = 0

        if max_layer_size > 1:
            node_sep = min(self._node_separation, canvas_w / (max_layer_size - 1))
        else:
            node_sep = 0

        n = len(self._nodes)
        self._dummy_pos = {}

        # Assign coordinates based on layer position. Dummy nodes occupy real
        # horizontal slots (so long edges are routed around, not through, other
        # nodes) but are recorded separately rather than written to self._nodes.
        for layer_idx, layer in enumerate(self._layers):
            layer_size = len(layer)

            # Center layer horizontally
            layer_width = (layer_size - 1) * node_sep if layer_size > 1 else 0
            start_offset = (canvas_w - layer_width) / 2 if layer_size > 1 else canvas_w / 2

            for pos, node_idx in enumerate(layer):
                # Within-layer position (horizontal in top-to-bottom)
                within_pos = start_offset + pos * node_sep if layer_size > 1 else start_offset

                # Layer position (vertical in top-to-bottom)
                layer_pos = layer_idx * layer_sep if n_layers > 1 else canvas_h / 2

                # Apply orientation
                if self._orientation == "top-to-bottom":
                    px = padding + within_pos
                    py = padding + layer_pos
                elif self._orientation == "bottom-to-top":
                    px = padding + within_pos
                    py = self._canvas_size[1] - padding - layer_pos
                elif self._orientation == "left-to-right":
                    px = padding + layer_pos
                    py = padding + within_pos
                else:  # right-to-left
                    px = self._canvas_size[0] - padding - layer_pos
                    py = padding + within_pos

                if node_idx < n:
                    node = self._nodes[node_idx]
                    node.x = px
                    node.y = py
                else:
                    self._dummy_pos[node_idx] = (px, py)

        # Record bend points (dummy coordinates) per original long edge.
        for edge, chain in self._edge_chains.items():
            bends = [self._dummy_pos[d] for d in chain[1:-1] if d in self._dummy_pos]
            if bends:
                self._edge_bends[edge] = bends

    # -------------------------------------------------------------------------
    # Layout Computation
    # -------------------------------------------------------------------------

    def _compute(self, **kwargs: Any) -> None:
        """Compute Sugiyama layout."""
        n = len(self._nodes)
        if n == 0:
            return

        # Reset per-run state (layouts may be re-run).
        self._layers = []
        self._node_layer = {}
        self._node_position = {}
        self._dummy_nodes = set()
        self._segments = []
        self._dummy_pos = {}
        self._edge_bends = {}
        self._edge_chains = {}

        # Phase 0: Cycle removal (reverse back edges to obtain a DAG)
        edges = self._build_acyclic_edges()

        # Phase 1: Layer assignment (longest path)
        self._assign_layers(edges)

        # Phase 2a: Insert dummy nodes for edges spanning multiple layers
        self._insert_dummy_nodes(edges)

        # Phase 2b: Crossing minimization (barycenter over the expanded graph)
        self._minimize_crossings()

        # Phase 3: Coordinate assignment
        self._assign_coordinates()


__all__ = ["SugiyamaLayout", "GraphStructureWarning"]
