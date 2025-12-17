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

    # -------------------------------------------------------------------------
    # Phase 1: Layer Assignment (Longest Path)
    # -------------------------------------------------------------------------

    def _assign_layers(self) -> None:
        """Assign nodes to layers using longest path algorithm."""
        n = len(self._nodes)
        if n == 0:
            return

        # Build adjacency lists
        outgoing: list[list[int]] = [[] for _ in range(n)]
        incoming: list[list[int]] = [[] for _ in range(n)]

        for link in self._links:
            src = self._get_source_index(link)
            tgt = self._get_target_index(link)
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
        layers = [-1] * n
        queue: deque[int] = deque()

        for src in sources:
            layers[src] = 0
            queue.append(src)

        while queue:
            node = queue.popleft()
            for child in outgoing[node]:
                new_layer = layers[node] + 1
                if new_layer > layers[child]:
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
    # Phase 2: Crossing Minimization (Barycenter Method)
    # -------------------------------------------------------------------------

    def _minimize_crossings(self) -> None:
        """Minimize edge crossings using barycenter heuristic."""
        if len(self._layers) < 2:
            return

        # Build adjacency for barycenter computation
        n = len(self._nodes)
        outgoing: list[list[int]] = [[] for _ in range(n)]
        incoming: list[list[int]] = [[] for _ in range(n)]

        for link in self._links:
            src = self._get_source_index(link)
            tgt = self._get_target_index(link)
            outgoing[src].append(tgt)
            incoming[tgt].append(src)

        # Initialize positions
        for layer in self._layers:
            for pos, node in enumerate(layer):
                self._node_position[node] = pos

        # Iterate: alternate sweeping down and up
        for iteration in range(self._crossing_iterations):
            if iteration % 2 == 0:
                # Sweep down
                for layer_idx in range(1, len(self._layers)):
                    self._order_layer_by_barycenter(layer_idx, incoming, direction="down")
            else:
                # Sweep up
                for layer_idx in range(len(self._layers) - 2, -1, -1):
                    self._order_layer_by_barycenter(layer_idx, outgoing, direction="up")

    def _order_layer_by_barycenter(
        self, layer_idx: int, adj: list[list[int]], direction: str
    ) -> None:
        """Reorder a layer based on barycenter of adjacent layer."""
        layer = self._layers[layer_idx]
        if not layer:
            return

        # Compute barycenter for each node
        barycenters: list[tuple[float, int]] = []

        for node in layer:
            neighbors = adj[node]
            if neighbors:
                # Barycenter = average position of neighbors
                positions = [self._node_position.get(n, 0) for n in neighbors]
                barycenter = sum(positions) / len(positions)
            else:
                # Keep current position
                barycenter = self._node_position.get(node, 0)

            barycenters.append((barycenter, node))

        # Sort by barycenter
        barycenters.sort(key=lambda x: x[0])

        # Update layer and positions
        self._layers[layer_idx] = [node for _, node in barycenters]
        for pos, (_, node) in enumerate(barycenters):
            self._node_position[node] = pos

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

        # Assign coordinates based on layer position
        for layer_idx, layer in enumerate(self._layers):
            layer_size = len(layer)

            # Center layer horizontally
            layer_width = (layer_size - 1) * node_sep if layer_size > 1 else 0
            start_offset = (canvas_w - layer_width) / 2 if layer_size > 1 else canvas_w / 2

            for pos, node_idx in enumerate(layer):
                node = self._nodes[node_idx]

                # Within-layer position (horizontal in top-to-bottom)
                within_pos = start_offset + pos * node_sep if layer_size > 1 else start_offset

                # Layer position (vertical in top-to-bottom)
                layer_pos = layer_idx * layer_sep if n_layers > 1 else canvas_h / 2

                # Apply orientation
                if self._orientation == "top-to-bottom":
                    node.x = padding + within_pos
                    node.y = padding + layer_pos
                elif self._orientation == "bottom-to-top":
                    node.x = padding + within_pos
                    node.y = self._canvas_size[1] - padding - layer_pos
                elif self._orientation == "left-to-right":
                    node.x = padding + layer_pos
                    node.y = padding + within_pos
                elif self._orientation == "right-to-left":
                    node.x = self._canvas_size[0] - padding - layer_pos
                    node.y = padding + within_pos

    # -------------------------------------------------------------------------
    # Layout Computation
    # -------------------------------------------------------------------------

    def _compute(self, **kwargs: Any) -> None:
        """Compute Sugiyama layout."""
        n = len(self._nodes)
        if n == 0:
            return

        # Phase 1: Layer assignment
        self._assign_layers()

        # Phase 2: Crossing minimization
        self._minimize_crossings()

        # Phase 3: Coordinate assignment
        self._assign_coordinates()


__all__ = ["SugiyamaLayout", "GraphStructureWarning"]
