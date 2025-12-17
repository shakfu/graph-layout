"""
ColaLayoutAdapter - Unified interface for Cola Layout.

Wraps the Cola Layout class to provide an interface consistent with
the BaseLayout hierarchy, enabling polymorphic usage while preserving
all Cola-specific functionality.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Union, cast

from ..base import IterativeLayout
from ..types import (
    Event,
    EventType,
    Group,
    GroupLike,
    Link,
    LinkLike,
    Node,
    NodeLike,
    SizeType,
)
from .layout import Layout as ColaLayout


class ColaLayoutAdapter(IterativeLayout):
    """
    Adapter that wraps Cola Layout with a BaseLayout-compatible interface.

    This adapter enables Cola layouts to be used interchangeably with
    other layout algorithms while preserving access to Cola-specific
    features like constraints, overlap avoidance, and groups.

    Example:
        layout = ColaLayoutAdapter(
            nodes=nodes,
            links=links,
            size=(800, 600),
            link_distance=100.0,
            avoid_overlaps=True,
        )
        layout.run()

        # Access underlying Cola for advanced features
        layout.cola.power_graph_groups()
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
        # IterativeLayout parameters
        alpha: float = 1.0,
        alpha_min: float = 0.001,
        alpha_decay: float = 0.99,
        iterations: int = 300,
        # Cola-specific parameters
        link_distance: Union[float, Callable[[Link], float]] = 20.0,
        avoid_overlaps: bool = False,
        handle_disconnected: bool = True,
        convergence_threshold: float = 0.01,
        default_node_size: float = 10.0,
        group_compactness: float = 1e-6,
        constraints: Optional[list[Any]] = None,
    ) -> None:
        """
        Initialize Cola layout adapter.

        Args:
            nodes: List of nodes
            links: List of links
            groups: List of groups
            size: Canvas size as (width, height)
            random_seed: Random seed for reproducible layouts
            on_start: Callback for start event
            on_tick: Callback for tick event
            on_end: Callback for end event
            alpha: Initial alpha/temperature (0 to 1)
            alpha_min: Minimum alpha for convergence threshold
            alpha_decay: Alpha decay rate per iteration (0 to 1)
            iterations: Maximum number of iterations
            link_distance: Ideal link distance (fixed value or callable).
            avoid_overlaps: Whether to prevent node overlaps.
            handle_disconnected: Whether to pack disconnected components.
            convergence_threshold: Stop when stress change falls below this.
            default_node_size: Size for nodes without width/height.
            group_compactness: Higher values make groups more compact.
            constraints: List of constraint objects.
        """
        # Initialize base class (but don't call super init yet)
        # We need to set up the Cola instance first
        self._cola = ColaLayout()
        self._started = False

        # Configure Cola with provided parameters
        self._cola.default_node_size(default_node_size)
        self._cola.convergence_threshold(convergence_threshold)
        self._cola.handle_disconnected(handle_disconnected)
        self._cola.avoid_overlaps(avoid_overlaps)
        self._cola.link_distance(link_distance)
        self._cola.group_compactness(group_compactness)
        if constraints:
            self._cola.constraints(constraints)

        # Forward events from Cola to base event system
        def forward_start(e: Optional[Event]) -> None:
            if e:
                self.trigger(e)

        def forward_tick(e: Optional[Event]) -> None:
            if e:
                self._alpha = e.get("alpha", self._alpha)
                self.trigger(e)

        def forward_end(e: Optional[Event]) -> None:
            if e:
                self.trigger(e)

        self._cola.on(EventType.start, forward_start)
        self._cola.on(EventType.tick, forward_tick)
        self._cola.on(EventType.end, forward_end)

        # Now initialize parent class
        super().__init__(
            nodes=nodes,
            links=links,
            groups=groups,
            size=size,
            random_seed=random_seed,
            on_start=on_start,
            on_tick=on_tick,
            on_end=on_end,
            alpha=alpha,
            alpha_min=alpha_min,
            alpha_decay=alpha_decay,
            iterations=iterations,
        )

        # Store Cola-specific config
        self._link_distance: Union[float, Callable[[Link], float]] = link_distance
        self._avoid_overlaps: bool = avoid_overlaps
        self._handle_disconnected: bool = handle_disconnected
        self._convergence_threshold: float = convergence_threshold
        self._default_node_size: float = default_node_size
        self._group_compactness: float = group_compactness
        self._constraints: Optional[list[Any]] = constraints

    # -------------------------------------------------------------------------
    # Override base properties to sync with Cola
    # -------------------------------------------------------------------------

    @property
    def nodes(self) -> list[Node]:
        """Get the list of nodes."""
        return cast(list[Node], self._cola.nodes())

    @nodes.setter
    def nodes(self, value: Sequence[NodeLike]) -> None:
        """Set nodes from a sequence."""
        # Convert to internal format
        super(ColaLayoutAdapter, type(self)).nodes.fset(self, value)  # type: ignore[attr-defined]
        # Sync to Cola
        self._cola.nodes(self._nodes)

    @property
    def links(self) -> list[Link]:
        """Get the list of links."""
        return cast(list[Link], self._cola.links())

    @links.setter
    def links(self, value: Sequence[LinkLike]) -> None:
        """Set links from a sequence."""
        super(ColaLayoutAdapter, type(self)).links.fset(self, value)  # type: ignore[attr-defined]
        self._cola.links(self._links)

    @property
    def groups(self) -> list[Group]:
        """Get the list of groups."""
        return cast(list[Group], self._cola.groups())

    @groups.setter
    def groups(self, value: Sequence[GroupLike]) -> None:
        """Set groups from a sequence."""
        super(ColaLayoutAdapter, type(self)).groups.fset(self, value)  # type: ignore[attr-defined]
        self._cola.groups(self._groups)

    @property
    def size(self) -> tuple[float, float]:
        """Get canvas size as (width, height)."""
        return self._canvas_size

    @size.setter
    def size(self, value: SizeType) -> None:
        """Set canvas size."""
        super(ColaLayoutAdapter, type(self)).size.fset(self, value)  # type: ignore[attr-defined]
        self._cola.size(list(self._canvas_size))

    # -------------------------------------------------------------------------
    # Cola-Specific Properties
    # -------------------------------------------------------------------------

    @property
    def link_distance(self) -> Union[float, Callable[[Link], float]]:
        """Get link distance setting."""
        return self._link_distance

    @link_distance.setter
    def link_distance(self, value: Union[float, Callable[[Link], float]]) -> None:
        """Set link distance."""
        self._link_distance = value
        self._cola.link_distance(value)

    @property
    def avoid_overlaps(self) -> bool:
        """Get whether overlap avoidance is enabled."""
        return self._avoid_overlaps

    @avoid_overlaps.setter
    def avoid_overlaps(self, value: bool) -> None:
        """Enable/disable overlap avoidance."""
        self._avoid_overlaps = bool(value)
        self._cola.avoid_overlaps(value)

    @property
    def handle_disconnected(self) -> bool:
        """Get whether disconnected component handling is enabled."""
        return self._handle_disconnected

    @handle_disconnected.setter
    def handle_disconnected(self, value: bool) -> None:
        """Enable/disable disconnected component handling."""
        self._handle_disconnected = bool(value)
        self._cola.handle_disconnected(value)

    @property
    def convergence_threshold(self) -> float:
        """Get convergence threshold."""
        return self._convergence_threshold

    @convergence_threshold.setter
    def convergence_threshold(self, value: float) -> None:
        """Set convergence threshold."""
        self._convergence_threshold = float(value)
        self._cola.convergence_threshold(value)

    @property
    def default_node_size(self) -> float:
        """Get default node size."""
        return self._default_node_size

    @default_node_size.setter
    def default_node_size(self, value: float) -> None:
        """Set default node size."""
        self._default_node_size = float(value)
        self._cola.default_node_size(value)

    @property
    def group_compactness(self) -> float:
        """Get group compactness."""
        return self._group_compactness

    @group_compactness.setter
    def group_compactness(self, value: float) -> None:
        """Set group compactness."""
        self._group_compactness = float(value)
        self._cola.group_compactness(value)

    @property
    def constraints(self) -> Optional[list[Any]]:
        """Get constraints."""
        return self._constraints

    @constraints.setter
    def constraints(self, value: Optional[list[Any]]) -> None:
        """Set constraints."""
        self._constraints = value
        if value:
            self._cola.constraints(value)

    @property
    def cola(self) -> ColaLayout:
        """
        Access the underlying Cola Layout for advanced features.

        Use this for features not exposed through the adapter interface,
        such as:
        - power_graph_groups()
        - prepare_edge_routing() / route_edge()
        - drag_start() / drag() / drag_end()
        """
        return self._cola

    # -------------------------------------------------------------------------
    # Layout Implementation
    # -------------------------------------------------------------------------

    def tick(self) -> bool:
        """Perform one iteration. Returns True when converged."""
        return self._cola.tick()

    def run(self, **kwargs: Any) -> "ColaLayoutAdapter":
        """
        Run the layout algorithm.

        Keyword Args:
            unconstrained_iterations: Iterations without constraints (default: 0)
            user_constraint_iterations: Iterations with user constraints (default: 0)
            all_constraints_iterations: Iterations with all constraints (default: iterations/3)
            grid_snap_iterations: Iterations with grid snapping (default: 0)
            keep_running: Whether to keep running until converged (default: True)
            center_graph: Center the graph after layout (default: True)

        Returns:
            self for chaining
        """
        # Sync data to Cola if set via properties
        if self._nodes:
            self._cola.nodes(self._nodes)
        if self._links:
            self._cola.links(self._links)
        if self._groups:
            self._cola.groups(self._groups)
        self._cola.size(list(self._canvas_size))

        # Get iteration counts
        unconstrained = kwargs.get("unconstrained_iterations", 0)
        user_constraint = kwargs.get("user_constraint_iterations", 0)
        all_constraints = kwargs.get("all_constraints_iterations", self._iterations // 3)
        grid_snap = kwargs.get("grid_snap_iterations", 0)
        keep_running = kwargs.get("keep_running", True)
        center = kwargs.get("center_graph", True)

        # Fire start event (Cola doesn't fire it in batch mode)
        self._alpha = 1.0
        self.trigger({"type": EventType.start, "alpha": self._alpha})

        self._cola.start(
            unconstrained,
            user_constraint,
            all_constraints,
            grid_snap,
            keep_running,
            center,
        )

        self._started = True

        # Fire end event if not keep_running (Cola doesn't fire it in batch mode)
        if not keep_running:
            self._alpha = 0.0
            self.trigger({"type": EventType.end, "alpha": 0.0})

        return self

    def stop(self) -> "ColaLayoutAdapter":
        """Stop the layout."""
        self._cola.stop()
        self._running = False
        return self

    def resume(self) -> "ColaLayoutAdapter":
        """Resume layout after stopping."""
        self._cola.resume()
        self._running = True
        return self

    def kick(self) -> None:
        """Run tick() repeatedly until convergence."""
        self._cola.kick()

    # -------------------------------------------------------------------------
    # Cola-Specific Methods
    # -------------------------------------------------------------------------

    def flow_layout(
        self,
        axis: str = "y",
        min_separation: Optional[Union[float, Callable[[Link], float]]] = None,
    ) -> "ColaLayoutAdapter":
        """
        Configure flow layout (directed graph layout).

        Adds constraints to make edges flow in a consistent direction.

        Args:
            axis: Flow axis ('x' or 'y', default 'y' = top-to-bottom)
            min_separation: Minimum separation between nodes

        Returns:
            self for chaining
        """
        if min_separation is not None:
            self._cola.flow_layout(axis, min_separation)
        else:
            self._cola.flow_layout(axis)
        return self

    def symmetric_diff_link_lengths(
        self, ideal_length: float, w: float = 1.0
    ) -> "ColaLayoutAdapter":
        """
        Compute link lengths using symmetric difference.

        Args:
            ideal_length: Base ideal length for edges
            w: Weight factor

        Returns:
            self for chaining
        """
        self._cola.symmetric_diff_link_lengths(ideal_length, w)
        return self

    def jaccard_link_lengths(self, ideal_length: float, w: float = 1.0) -> "ColaLayoutAdapter":
        """
        Compute link lengths using Jaccard coefficient.

        Args:
            ideal_length: Base ideal length for edges
            w: Weight factor

        Returns:
            self for chaining
        """
        self._cola.jaccard_link_lengths(ideal_length, w)
        return self


__all__ = ["ColaLayoutAdapter"]
