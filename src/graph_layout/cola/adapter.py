"""
ColaLayoutAdapter - Unified interface for Cola Layout.

Wraps the Cola Layout class to provide an interface consistent with
the BaseLayout hierarchy, enabling polymorphic usage while preserving
all Cola-specific functionality.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union, cast

from typing_extensions import Self

from ..base import IterativeLayout
from ..types import Event, EventType, Group, Link, Node
from .layout import Layout as ColaLayout


class ColaLayoutAdapter(IterativeLayout):
    """
    Adapter that wraps Cola Layout with a BaseLayout-compatible interface.

    This adapter enables Cola layouts to be used interchangeably with
    other layout algorithms while preserving access to Cola-specific
    features like constraints, overlap avoidance, and groups.

    Example:
        # Standard BaseLayout interface
        layout = (ColaLayoutAdapter()
            .nodes(nodes)
            .links(links)
            .size([800, 600])
            .start())

        # Cola-specific features still available
        layout.avoid_overlaps(True)
        layout.constraints(constraints)

        # Access underlying Cola for advanced features
        layout.cola.power_graph_groups()
    """

    def __init__(self) -> None:
        super().__init__()
        self._cola = ColaLayout()
        self._started = False
        # Set a default node size to avoid None errors in packing
        self._cola.default_node_size(10)

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

    # -------------------------------------------------------------------------
    # BaseLayout Interface - Delegation
    # -------------------------------------------------------------------------

    def nodes(self, v: Optional[list[Any]] = None) -> Union[list[Node], Self]:
        """Get or set nodes."""
        if v is None:
            return cast(list[Node], self._cola.nodes())
        self._cola.nodes(v)
        self._nodes = cast(list[Node], self._cola.nodes())
        return self

    def links(self, v: Optional[list[Any]] = None) -> Union[list[Link], Self]:
        """Get or set links."""
        if v is None:
            return cast(list[Link], self._cola.links())
        self._cola.links(v)
        self._links = cast(list[Link], self._cola.links())
        return self

    def groups(self, v: Optional[list[Any]] = None) -> Union[list[Group], Self]:
        """Get or set groups."""
        if v is None:
            return cast(list[Group], self._cola.groups())
        self._cola.groups(v)
        self._groups = cast(list[Group], self._cola.groups())
        return self

    def size(self, v: Optional[list[float]] = None) -> Union[list[float], Self]:
        """Get or set canvas size."""
        if v is None:
            return cast(list[float], self._cola.size())
        # Use parent validation
        super().size(v)
        self._cola.size(v)
        return self

    # -------------------------------------------------------------------------
    # IterativeLayout Interface
    # -------------------------------------------------------------------------

    def alpha(self, v: Optional[float] = None) -> Union[float, Self]:
        """Get or set alpha (temperature)."""
        if v is None:
            a = cast(Optional[float], self._cola.alpha())
            return a if a is not None else 0.0
        self._cola.alpha(v)
        self._alpha = v
        return self

    def alpha_min(self, v: Optional[float] = None) -> Union[float, Self]:
        """
        Get or set minimum alpha (convergence threshold).

        Note: Maps to Cola's convergence_threshold().
        """
        if v is None:
            return cast(float, self._cola.convergence_threshold())
        self._cola.convergence_threshold(v)
        self._alpha_min = v
        return self

    def iterations(self, v: Optional[int] = None) -> Union[int, Self]:
        """
        Get or set maximum iterations.

        Note: Cola uses different iteration parameters in start().
        This sets a default that is divided among constraint phases.
        """
        if v is None:
            return self._iterations
        self._iterations = max(1, int(v))
        return self

    def tick(self) -> bool:
        """Perform one iteration. Returns True when converged."""
        return self._cola.tick()

    def start(self, **kwargs: Any) -> Self:
        """
        Start the layout algorithm.

        Keyword Args:
            iterations: Total iterations (default: 300), divided among phases
            initial_unconstrained_iterations: Iterations without constraints
            initial_user_constraint_iterations: Iterations with user constraints
            initial_all_constraints_iterations: Iterations with all constraints
            grid_snap_iterations: Iterations with grid snapping
            keep_running: Whether to keep running until converged (default: True)
            center_graph: Center the graph after layout (default: True)

        Returns:
            self for chaining
        """
        # Get iteration counts
        total_iters = kwargs.get("iterations", self._iterations)

        # Map to Cola's multi-phase iteration scheme
        unconstrained = kwargs.get("initial_unconstrained_iterations", 0)
        user_constraint = kwargs.get("initial_user_constraint_iterations", 0)
        all_constraints = kwargs.get(
            "initial_all_constraints_iterations", total_iters // 3
        )
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

    def stop(self) -> Self:
        """Stop the layout."""
        self._cola.stop()
        self._running = False
        return self

    def resume(self) -> Self:
        """Resume layout after stopping."""
        self._cola.resume()
        self._running = True
        return self

    def kick(self) -> None:
        """Run tick() repeatedly until convergence."""
        self._cola.kick()

    # -------------------------------------------------------------------------
    # Cola-Specific Methods (Passthrough)
    # -------------------------------------------------------------------------

    def link_distance(
        self, v: Optional[Union[float, Callable[[Link], float]]] = None
    ) -> Union[Union[float, Callable[[Link], float]], Self]:
        """
        Get or set link distance.

        Args:
            v: Fixed distance (float) or function (link) -> distance.
               If None, returns current value.

        Returns:
            Current value or self for chaining.
        """
        if v is None:
            return cast(Union[float, Callable[[Link], float]], self._cola.link_distance())
        self._cola.link_distance(v)
        return self

    def avoid_overlaps(self, v: Optional[bool] = None) -> Union[bool, Self]:
        """
        Get or set overlap avoidance.

        When enabled, nodes are constrained to not overlap.

        Args:
            v: Enable/disable overlap avoidance. If None, returns current value.

        Returns:
            Current value or self for chaining.
        """
        if v is None:
            return cast(bool, self._cola.avoid_overlaps())
        self._cola.avoid_overlaps(v)
        return self

    def handle_disconnected(self, v: Optional[bool] = None) -> Union[bool, Self]:
        """
        Get or set disconnected component handling.

        When enabled, disconnected components are laid out separately
        and then packed together.

        Args:
            v: Enable/disable. If None, returns current value.

        Returns:
            Current value or self for chaining.
        """
        if v is None:
            return cast(bool, self._cola.handle_disconnected())
        self._cola.handle_disconnected(v)
        return self

    def convergence_threshold(self, v: Optional[float] = None) -> Union[float, Self]:
        """
        Get or set convergence threshold.

        Layout stops when stress change falls below this value.

        Args:
            v: Threshold value. If None, returns current value.

        Returns:
            Current value or self for chaining.
        """
        if v is None:
            return cast(float, self._cola.convergence_threshold())
        self._cola.convergence_threshold(v)
        return self

    def constraints(self, v: Optional[list] = None) -> Union[list, Self]:
        """
        Get or set constraints.

        Args:
            v: List of constraint objects. If None, returns current value.

        Returns:
            Current value or self for chaining.
        """
        if v is None:
            return cast(list, self._cola.constraints())
        self._cola.constraints(v)
        return self

    def flow_layout(
        self,
        axis: str = "y",
        min_separation: Optional[Union[float, Callable[[Link], float]]] = None,
    ) -> Self:
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
    ) -> Self:
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

    def jaccard_link_lengths(self, ideal_length: float, w: float = 1.0) -> Self:
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

    def default_node_size(self, v: Optional[float] = None) -> Union[float, Self]:
        """
        Get or set default node size.

        Used when nodes don't specify width/height.

        Args:
            v: Default size. If None, returns current value.

        Returns:
            Current value or self for chaining.
        """
        if v is None:
            return cast(float, self._cola.default_node_size())
        self._cola.default_node_size(v)
        return self

    def group_compactness(self, v: Optional[float] = None) -> Union[float, Self]:
        """
        Get or set group compactness.

        Higher values make groups more compact.

        Args:
            v: Compactness value. If None, returns current value.

        Returns:
            Current value or self for chaining.
        """
        if v is None:
            return cast(float, self._cola.group_compactness())
        self._cola.group_compactness(v)
        return self

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


__all__ = ["ColaLayoutAdapter"]
