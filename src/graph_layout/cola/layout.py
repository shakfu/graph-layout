"""
Main 2D force-directed graph layout engine.

This module implements the Layout class which provides:
- Node positioning with force-directed algorithm
- Link length constraints
- Group/cluster support with containment constraints
- Overlap avoidance
- Power graph integration
- Flow layouts
- Event system (start/tick/end events)
- Disconnected graph handling
"""

from __future__ import annotations

from typing import Callable, Optional, TypedDict, Union

import numpy as np

from ..types import (
    Event,
    EventType,
    Group,
    Link,
    LinkNumericPropertyAccessor,
    Node,
    is_group,
)
from .descent import Descent
from .geom import Point, TangentVisibilityGraph
from .handledisconnected import apply_packing, separate_graphs
from .linklengths import (
    AlignmentConstraint,
    SeparationConstraint,
    generate_directed_edge_constraints,
    jaccard_link_lengths,
    symmetric_diff_link_lengths,
)
from .powergraph import LinkTypeAccessor, get_groups
from .rectangle import Projection, make_edge_between, make_edge_to
from .shortestpaths import Calculator


class InputNode(TypedDict, total=False):
    """
    Input node specification (TypedDict for documentation).

    Attributes:
        index: Index in nodes array, initialized by Layout.start()
        x: x coordinate, computed by layout as the node's centroid
        y: y coordinate, computed by layout as the node's centroid
        width: Width of the node's bounding box (for overlap avoidance)
        height: Height of the node's bounding box (for overlap avoidance)
        fixed: Selective bit mask. !=0 means layout will not move
    """

    index: int
    x: float
    y: float
    width: float
    height: float
    fixed: int


class Layout:
    """
    Main interface to cola layout.

    This class provides a fluent API for configuring and running
    force-directed graph layout with constraints.
    """

    def __init__(self):
        """Initialize layout with default parameters."""
        self._canvasSize: list[float] = [1.0, 1.0]
        self._linkDistance: Union[float, LinkNumericPropertyAccessor] = 20.0
        self._defaultNodeSize: float = 10.0
        self._linkLengthCalculator: Optional[Callable[[], None]] = None
        self._linkType: Optional[Union[Callable, int]] = None
        self._avoidOverlaps: bool = False
        self._handleDisconnected: bool = True
        self._alpha: Optional[float] = None
        self._lastStress: Optional[float] = None
        self._running: bool = False
        self._nodes: list[Node] = []
        self._groups: list[Group] = []
        self._rootGroup: Optional[Group] = None
        self._links: list[Link] = []
        self._constraints: list[Union[SeparationConstraint, AlignmentConstraint]] = []
        self._distanceMatrix: Optional[np.ndarray] = None
        self._descent: Optional[Descent] = None
        self._directedLinkConstraints: Optional[dict] = None
        self._threshold: float = 0.01
        self._visibilityGraph: Optional[TangentVisibilityGraph] = None
        self._groupCompactness: float = 1e-6

        # Event system - can be overridden by subclasses
        self.event: Optional[dict] = None

    def on(self, e: Union[EventType, str], listener: Callable[[Optional[Event]], None]) -> Layout:
        """
        Subscribe a listener to an event.

        Args:
            e: Event type (EventType enum or string name)
            listener: Function to call when event fires

        Returns:
            self for method chaining
        """
        if self.event is None:
            self.event = {}

        if isinstance(e, str):
            # Convert string to EventType
            event_type = EventType[e]
            self.event[event_type] = listener
        else:
            self.event[e] = listener

        return self

    def trigger(self, e: Event) -> None:
        """
        Trigger an event by calling registered listeners.

        Subclasses can override this method to replace with a more
        sophisticated eventing mechanism.

        Args:
            e: Event to trigger
        """
        if self.event and e["type"] in self.event:
            self.event[e["type"]](e)

    def kick(self) -> None:
        """
        Kick off the iteration tick loop.

        Calls tick() repeatedly until tick returns True (converged).
        Subclass and override with something fancier (e.g. dispatch tick on a timer).
        """
        while not self.tick():
            pass

    def tick(self) -> bool:
        """
        Iterate the layout once.

        Returns:
            True when layout converged, False otherwise
        """
        if self._alpha < self._threshold:
            self._running = False
            self.trigger({"type": EventType.end, "alpha": 0.0, "stress": self._lastStress})
            self._alpha = 0.0
            return True

        n = len(self._nodes)
        m = len(self._links)

        # Clear locks and re-add fixed nodes
        self._descent.locks.clear()
        for i in range(n):
            o = self._nodes[i]
            if o.fixed:
                if o.px is None or o.py is None:
                    o.px = o.x
                    o.py = o.y
                p = np.array([o.px, o.py])
                self._descent.locks.add(i, p)

        # Run one step of gradient descent
        s1 = self._descent.runge_kutta()

        if s1 == 0:
            self._alpha = 0.0
        elif self._lastStress is not None:
            self._alpha = s1

        self._lastStress = s1

        self._update_node_positions()

        self.trigger({"type": EventType.tick, "alpha": self._alpha, "stress": self._lastStress})

        return False

    def _update_node_positions(self) -> None:
        """Copy positions from descent instance into node center coords."""
        x = self._descent.x[0]
        y = self._descent.x[1]

        for i in range(len(self._nodes) - 1, -1, -1):
            o = self._nodes[i]
            o.x = x[i]
            o.y = y[i]

    def nodes(self, v: Optional[list] = None) -> Union[list[Node], Layout]:
        """
        Get or set the list of nodes.

        If nodes has not been set, but links has, then we instantiate a nodes list here,
        of the correct size, before returning it.

        Args:
            v: Optional list of nodes to set

        Returns:
            Current nodes list if v is None, otherwise self for chaining
        """
        if v is None:
            if len(self._nodes) == 0 and len(self._links) > 0:
                # If we have links but no nodes, create the nodes array with empty objects
                # In this case links are expected to be numeric indices for nodes 0..n-1
                n = 0
                for l in self._links:
                    source_idx = l.source if isinstance(l.source, int) else 0
                    target_idx = l.target if isinstance(l.target, int) else 0
                    n = max(n, source_idx, target_idx)

                n += 1
                self._nodes = [Node() for _ in range(n)]

            return self._nodes

        # Convert dict-like objects to Node instances
        self._nodes = []
        for node_data in v:
            if isinstance(node_data, Node):
                self._nodes.append(node_data)
            elif isinstance(node_data, dict):
                self._nodes.append(Node(**node_data))
            else:
                # Treat as an object with attributes
                node = Node()
                for attr in dir(node_data):
                    if not attr.startswith("_"):
                        setattr(node, attr, getattr(node_data, attr))
                self._nodes.append(node)

        return self

    def groups(self, x: Optional[list] = None) -> Union[list[Group], Layout]:
        """
        Get or set the list of hierarchical groups.

        Args:
            x: Optional list of groups to set

        Returns:
            Current groups list if x is None, otherwise self for chaining
        """
        if x is None:
            return self._groups

        # Convert to Group instances and set up hierarchy
        self._groups = []
        for group_data in x:
            if isinstance(group_data, Group):
                self._groups.append(group_data)
            elif isinstance(group_data, dict):
                self._groups.append(Group(**group_data))
            else:
                group = Group()
                for attr in dir(group_data):
                    if not attr.startswith("_"):
                        setattr(group, attr, getattr(group_data, attr))
                self._groups.append(group)

        self._rootGroup = Group()

        # Process groups to resolve indices and set parent links
        for g in self._groups:
            if g.padding is None:
                g.padding = 1.0

            if g.leaves is not None:
                new_leaves = []
                for i, v in enumerate(g.leaves):
                    if isinstance(v, int):
                        node = self._nodes[v]
                        node.parent = g
                        new_leaves.append(node)
                    else:
                        v.parent = g
                        new_leaves.append(v)
                g.leaves = new_leaves

            if g.groups is not None:
                new_groups = []
                for i, gi in enumerate(g.groups):
                    if isinstance(gi, int):
                        child_group = self._groups[gi]
                        child_group.parent = g
                        new_groups.append(child_group)
                    else:
                        gi.parent = g
                        new_groups.append(gi)
                g.groups = new_groups

        # Set root group
        self._rootGroup.leaves = [
            v for v in self._nodes if not hasattr(v, "parent") or v.parent is None
        ]
        self._rootGroup.groups = [
            g for g in self._groups if not hasattr(g, "parent") or g.parent is None
        ]

        return self

    def power_graph_groups(self, f: Callable) -> Layout:
        """
        Generate power graph groups from the current graph structure.

        Args:
            f: Callback function to receive the power graph result

        Returns:
            self for method chaining
        """
        g = get_groups(self._nodes, self._links, self.link_accessor, self._rootGroup)

        # Convert dict result to object-like structure for compatibility
        power_graph = type(
            "PowerGraph", (), {"groups": g["groups"], "powerEdges": g["powerEdges"]}
        )()

        self.groups(power_graph.groups)
        f(power_graph)
        return self

    def avoid_overlaps(self, v: Optional[bool] = None) -> Union[bool, Layout]:
        """
        Get or set whether to avoid overlaps.

        If true, the layout will not permit overlaps of node bounding boxes
        (defined by width and height properties on nodes).

        Args:
            v: Optional value to set

        Returns:
            Current value if v is None, otherwise self for chaining
        """
        if v is None:
            return self._avoidOverlaps

        self._avoidOverlaps = v
        return self

    def handle_disconnected(self, v: Optional[bool] = None) -> Union[bool, Layout]:
        """
        Get or set whether to handle disconnected components.

        If true, the final step of start() will nicely pack connected components.
        Works best if start() is called with reasonable iterations and nodes have
        bounding boxes.

        Args:
            v: Optional value to set

        Returns:
            Current value if v is None, otherwise self for chaining
        """
        if v is None:
            return self._handleDisconnected

        self._handleDisconnected = v
        return self

    def flow_layout(self, axis: str = "y", min_separation: Union[float, Callable] = None) -> Layout:
        """
        Generate constraints for directed flow layout.

        Causes constraints to be generated such that directed graphs are laid out
        either left-to-right or top-to-bottom. A separation constraint is generated
        in the selected axis for each edge not involved in a cycle.

        Args:
            axis: 'x' for left-to-right, 'y' for top-to-bottom
            min_separation: Either a number or function returning minimum spacing

        Returns:
            self for method chaining
        """
        if isinstance(min_separation, (int, float)):
            sep_value = min_separation
            get_min_sep = lambda: sep_value
        else:
            get_min_sep = min_separation if min_separation else lambda: 0

        self._directedLinkConstraints = {"axis": axis, "getMinSeparation": get_min_sep}
        return self

    def links(self, x: Optional[list] = None) -> Union[list[Link], Layout]:
        """
        Get or set links defined as source, target pairs.

        Args:
            x: Optional list of links to set

        Returns:
            Current links list if x is None, otherwise self for chaining
        """
        if x is None:
            return self._links

        # Convert to Link instances
        self._links = []
        for link_data in x:
            if isinstance(link_data, Link):
                self._links.append(link_data)
            elif isinstance(link_data, dict):
                self._links.append(Link(**link_data))
            else:
                link = Link(
                    source=getattr(link_data, "source", 0), target=getattr(link_data, "target", 0)
                )
                for attr in dir(link_data):
                    if not attr.startswith("_") and attr not in ("source", "target"):
                        setattr(link, attr, getattr(link_data, attr))
                self._links.append(link)

        return self

    def constraints(
        self, c: Optional[list[Union[SeparationConstraint, AlignmentConstraint]]] = None
    ) -> Union[list[Union[SeparationConstraint, AlignmentConstraint]], Layout]:
        """
        Get or set list of constraints.

        Args:
            c: Optional list of constraints to set

        Returns:
            Current constraints if c is None, otherwise self for chaining
        """
        if c is None:
            return self._constraints

        self._constraints = c
        return self

    def distance_matrix(
        self, d: Optional[np.ndarray] = None
    ) -> Union[Optional[np.ndarray], Layout]:
        """
        Get or set matrix of ideal distances between all pairs of nodes.

        If unspecified, ideal distances will be based on shortest path distance.

        Args:
            d: Optional distance matrix to set

        Returns:
            Current distance matrix if d is None, otherwise self for chaining
        """
        if d is None:
            return self._distanceMatrix

        self._distanceMatrix = d
        return self

    def size(self, x: Optional[list[float]] = None) -> Union[list[float], Layout]:
        """
        Get or set canvas size [width, height].

        Currently only used to determine midpoint for initial node positions.

        Args:
            x: Optional size to set

        Returns:
            Current size if x is None, otherwise self for chaining
        """
        if x is None:
            return self._canvasSize

        self._canvasSize = x
        return self

    def default_node_size(self, x: Optional[float] = None) -> Union[float, Layout]:
        """
        Get or set default node size.

        Default size (both width and height) to use in packing if node
        width/height are not specified.

        Args:
            x: Optional size to set

        Returns:
            Current default size if x is None, otherwise self for chaining
        """
        if x is None:
            return self._defaultNodeSize

        self._defaultNodeSize = x
        return self

    def group_compactness(self, x: Optional[float] = None) -> Union[float, Layout]:
        """
        Get or set group compactness.

        The strength of attraction between group boundaries.

        Args:
            x: Optional compactness to set

        Returns:
            Current compactness if x is None, otherwise self for chaining
        """
        if x is None:
            return self._groupCompactness

        self._groupCompactness = x
        return self

    def link_distance(
        self, x: Optional[Union[float, LinkNumericPropertyAccessor]] = None
    ) -> Union[Union[float, LinkNumericPropertyAccessor], Layout]:
        """
        Get or set link distance.

        Links have an ideal distance. The layout will try to keep links as close
        as possible to this length.

        Args:
            x: Optional distance (number or function) to set

        Returns:
            Current link distance if x is None, otherwise self for chaining
        """
        if x is None:
            return self._linkDistance

        if callable(x):
            self._linkDistance = x
        else:
            self._linkDistance = float(x)

        self._linkLengthCalculator = None
        return self

    def link_type(self, f: Union[Callable, int]) -> Layout:
        """
        Set link type accessor.

        Args:
            f: Function or constant to determine link type

        Returns:
            self for method chaining
        """
        self._linkType = f
        return self

    def convergence_threshold(self, x: Optional[float] = None) -> Union[float, Layout]:
        """
        Get or set convergence threshold.

        Args:
            x: Optional threshold to set

        Returns:
            Current threshold if x is None, otherwise self for chaining
        """
        if x is None:
            return self._threshold

        self._threshold = float(x)
        return self

    def alpha(self, x: Optional[float] = None) -> Union[Optional[float], Layout]:
        """
        Get or set alpha (cooling parameter).

        Setting alpha > 0 will start/resume the layout if not already running.
        Setting alpha = 0 will stop the layout.

        Args:
            x: Optional alpha value to set

        Returns:
            Current alpha if x is None, otherwise self for chaining
        """
        if x is None:
            return self._alpha

        x = float(x)

        if self._alpha:  # Already running
            if x > 0:
                self._alpha = x  # Keep it hot
            else:
                self._alpha = 0  # Next tick will dispatch "end"
        elif x > 0:  # Fire it up
            if not self._running:
                self._running = True
                self._alpha = x
                self.trigger({"type": EventType.start, "alpha": self._alpha})
                self.kick()

        return self

    def get_link_length(self, link: Link) -> float:
        """
        Get the ideal length for a link.

        Args:
            link: The link

        Returns:
            Ideal length
        """
        if callable(self._linkDistance):
            return float(self._linkDistance(link))
        else:
            return float(self._linkDistance)

    @staticmethod
    def set_link_length(link: Link, length: float) -> None:
        """
        Set the length property on a link.

        Args:
            link: The link
            length: Length to set
        """
        link.length = length

    def get_link_type(self, link: Link) -> int:
        """
        Get the type of a link.

        Args:
            link: The link

        Returns:
            Link type (0 if no type function set)
        """
        if callable(self._linkType):
            return self._linkType(link)
        else:
            return 0

    @property
    def link_accessor(self) -> LinkTypeAccessor:
        """
        Get link accessor for this layout.

        Returns:
            LinkTypeAccessor with methods for accessing link properties
        """
        layout = self

        class LayoutLinkAccessor(LinkTypeAccessor):
            def get_source_index(self, l: Link) -> int:
                return Layout.get_source_index(l)

            def get_target_index(self, l: Link) -> int:
                return Layout.get_target_index(l)

            def set_length(self, l: Link, value: float) -> None:
                Layout.set_link_length(l, value)

            def get_type(self, l: Link) -> int:
                return layout.get_link_type(l)

        return LayoutLinkAccessor()

    def symmetric_diff_link_lengths(self, ideal_length: float, w: float = 1.0) -> Layout:
        """
        Compute ideal link lengths based on symmetric difference of neighbor sets.

        Creates extra space around hub-nodes in dense graphs. Calculation is based on
        symmetric difference in neighbor sets of source and target:
        sqrt(|a union b| - |a intersection b|)

        Args:
            ideal_length: Base length when source and target have no common neighbors
            w: Multiplier for the effect of length adjustment

        Returns:
            self for method chaining
        """
        self.link_distance(lambda l: ideal_length * (l.length if l.length else 1.0))
        self._linkLengthCalculator = lambda: symmetric_diff_link_lengths(
            self._links, self.link_accessor, w
        )
        return self

    def jaccard_link_lengths(self, ideal_length: float, w: float = 1.0) -> Layout:
        """
        Compute ideal link lengths based on Jaccard coefficient of neighbor sets.

        Creates extra space around hub-nodes in dense graphs. Calculation is based on
        Jaccard coefficient: |a intersection b| / |a union b|

        Args:
            ideal_length: Base length when source and target have no common neighbors
            w: Multiplier for the effect of length adjustment

        Returns:
            self for method chaining
        """
        self.link_distance(lambda l: ideal_length * (l.length if l.length else 1.0))
        self._linkLengthCalculator = lambda: jaccard_link_lengths(
            self._links, self.link_accessor, w
        )
        return self

    def start(
        self,
        initial_unconstrained_iterations: int = 0,
        initial_user_constraint_iterations: int = 0,
        initial_all_constraints_iterations: int = 0,
        grid_snap_iterations: int = 0,
        keep_running: bool = True,
        center_graph: bool = True,
    ) -> Layout:
        """
        Start the layout process.

        Args:
            initial_unconstrained_iterations: Unconstrained initial layout iterations
            initial_user_constraint_iterations: Iterations with user-specified constraints
            initial_all_constraints_iterations: Iterations with all constraints including non-overlap
            grid_snap_iterations: Iterations of grid snap (pulls nodes to grid cells)
            keep_running: Keep iterating asynchronously via tick method
            center_graph: Center graph on restart

        Returns:
            self for method chaining
        """
        nodes_list = self.nodes()
        n = len(nodes_list)
        N = n + 2 * len(self._groups)
        m = len(self._links)
        w = self._canvasSize[0]
        h = self._canvasSize[1]

        x = np.zeros(N)
        y = np.zeros(N)

        G = None

        ao = self._avoidOverlaps

        # Initialize node positions and indices
        for i, v in enumerate(self._nodes):
            v.index = i
            if not hasattr(v, "x") or v.x is None:
                v.x = w / 2.0
                v.y = h / 2.0
            x[i] = v.x
            y[i] = v.y

        # Calculate link lengths if calculator is set
        if self._linkLengthCalculator:
            self._linkLengthCalculator()

        # Compute distance matrix
        if self._distanceMatrix is not None:
            distances = self._distanceMatrix
        else:
            # Construct distance matrix based on shortest paths
            calc = Calculator(
                N,
                self._links,
                Layout.get_source_index,
                Layout.get_target_index,
                self.get_link_length,
            )
            distances = calc.distance_matrix()

            # Create adjacency matrix G
            G = Descent.create_square_matrix(N, lambda i, j: 2.0)

            # Convert link source/target from indices to node references
            for l in self._links:
                if isinstance(l.source, int):
                    l.source = self._nodes[l.source]
                if isinstance(l.target, int):
                    l.target = self._nodes[l.target]

            # Fill in G with edge weights
            for e in self._links:
                u = Layout.get_source_index(e)
                v = Layout.get_target_index(e)
                weight = e.weight if e.weight else 1.0
                G[u][v] = weight
                G[v][u] = weight

        # Create D matrix from distances
        D = Descent.create_square_matrix(N, lambda i, j: distances[i][j])

        # Handle groups
        if self._rootGroup and self._rootGroup.groups is not None:
            i = n

            def add_attraction(i: int, j: int, strength: float, ideal_distance: float):
                G[i][j] = strength
                G[j][i] = strength
                D[i][j] = ideal_distance
                D[j][i] = ideal_distance

            for g in self._groups:
                add_attraction(i, i + 1, self._groupCompactness, 0.1)

                if g.bounds is None:
                    x[i] = w / 2.0
                    y[i] = h / 2.0
                    i += 1
                    x[i] = w / 2.0
                    y[i] = h / 2.0
                    i += 1
                else:
                    x[i] = g.bounds.x
                    y[i] = g.bounds.y
                    i += 1
                    x[i] = g.bounds.X
                    y[i] = g.bounds.Y
                    i += 1
        else:
            self._rootGroup = Group(leaves=self._nodes, groups=[])

        # Set up constraints
        cur_constraints = self._constraints if self._constraints else []

        if self._directedLinkConstraints:
            # Add get_min_separation to link accessor
            link_acc = self.link_accessor
            link_acc.get_min_separation = self._directedLinkConstraints["getMinSeparation"]

            cur_constraints = cur_constraints + generate_directed_edge_constraints(
                n, self._links, self._directedLinkConstraints["axis"], link_acc
            )

        # Initialize descent
        self.avoid_overlaps(False)
        self._descent = Descent(np.array([x, y]), D)

        # Add locks for fixed nodes
        self._descent.locks.clear()
        for i in range(n):
            o = self._nodes[i]
            if o.fixed:
                o.px = o.x
                o.py = o.y
                p = np.array([o.x, o.y])
                self._descent.locks.add(i, p)

        self._descent.threshold = self._threshold

        # Initial layout
        self._initial_layout(initial_unconstrained_iterations, x, y)

        # Apply user constraints
        if len(cur_constraints) > 0:
            self._descent.project = Projection(
                self._nodes, self._groups, self._rootGroup, cur_constraints
            ).project_functions()

        self._descent.run(initial_user_constraint_iterations)
        self._separate_overlapping_components(w, h, center_graph)

        # Apply all constraints including overlap avoidance
        self.avoid_overlaps(ao)
        if ao:
            for i, v in enumerate(self._nodes):
                v.x = x[i]
                v.y = y[i]

            self._descent.project = Projection(
                self._nodes, self._groups, self._rootGroup, cur_constraints, True
            ).project_functions()

            for i, v in enumerate(self._nodes):
                x[i] = v.x
                y[i] = v.y

        # Allow non-connected nodes to relax (p-stress)
        self._descent.G = G
        self._descent.run(initial_all_constraints_iterations)

        # Grid snap
        if grid_snap_iterations:
            self._descent.snap_strength = 1000.0
            self._descent.snap_grid_size = self._nodes[0].width
            self._descent.num_grid_snap_nodes = n
            self._descent.scale_snap_by_max_h = n != N

            G0 = Descent.create_square_matrix(N, lambda i, j: G[i][j] if i >= n or j >= n else 0.0)
            self._descent.G = G0
            self._descent.run(grid_snap_iterations)

        self._update_node_positions()
        self._separate_overlapping_components(w, h, center_graph)

        return self.resume() if keep_running else self

    def _initial_layout(self, iterations: int, x: np.ndarray, y: np.ndarray) -> None:
        """
        Perform initial layout with groups.

        Args:
            iterations: Number of iterations
            x: X coordinates array
            y: Y coordinates array
        """
        if len(self._groups) > 0 and iterations > 0:
            # Construct flat graph with dummy nodes for groups
            n = len(self._nodes)

            edges = [{"source": e.source.index, "target": e.target.index} for e in self._links]

            vs = [{"index": v.index} for v in self._nodes]

            for i, g in enumerate(self._groups):
                g.index = n + i
                vs.append({"index": g.index})

            # Add edges from groups to their children
            for g in self._groups:
                if g.leaves is not None:
                    for v in g.leaves:
                        edges.append({"source": g.index, "target": v.index})

                if g.groups is not None:
                    for gg in g.groups:
                        edges.append({"source": g.index, "target": gg.index})

            # Layout flat graph
            flat_layout = Layout()
            flat_layout.size(self.size())
            flat_layout.nodes(vs)
            flat_layout.links(edges)
            flat_layout.avoid_overlaps(False)
            flat_layout.link_distance(self.link_distance())
            flat_layout.symmetric_diff_link_lengths(5)
            flat_layout.convergence_threshold(1e-4)
            flat_layout.start(iterations, 0, 0, 0, False)

            # Copy positions back
            for v in self._nodes:
                x[v.index] = vs[v.index]["x"]
                y[v.index] = vs[v.index]["y"]
        else:
            self._descent.run(iterations)

    def _separate_overlapping_components(
        self, width: float, height: float, center_graph: bool = True
    ) -> None:
        """
        Recalculate node positions for disconnected graphs.

        Args:
            width: Canvas width
            height: Canvas height
            center_graph: Whether to center the graph
        """
        if self._distanceMatrix is None and self._handleDisconnected:
            x = self._descent.x[0]
            y = self._descent.x[1]

            for i, v in enumerate(self._nodes):
                v.x = x[i]
                v.y = y[i]

            graphs = separate_graphs(self._nodes, self._links)
            apply_packing(graphs, width, height, self._defaultNodeSize, 1, center_graph)

            for i, v in enumerate(self._nodes):
                self._descent.x[0][i] = v.x
                self._descent.x[1][i] = v.y

                if v.bounds:
                    v.bounds.set_x_centre(v.x)
                    v.bounds.set_y_centre(v.y)

    def resume(self) -> Layout:
        """
        Resume layout iterations.

        Returns:
            self for method chaining
        """
        return self.alpha(0.1)

    def stop(self) -> Layout:
        """
        Stop layout iterations.

        Returns:
            self for method chaining
        """
        return self.alpha(0.0)

    def prepare_edge_routing(self, node_margin: float = 0.0) -> None:
        """
        Find a visibility graph over the set of nodes.

        Assumes all nodes have a bounds property (Rectangle) and that
        no pair of bounds overlaps.

        Args:
            node_margin: Margin around nodes for routing
        """
        self._visibilityGraph = TangentVisibilityGraph(
            [v.bounds.inflate(-node_margin).vertices() for v in self._nodes]
        )

    def route_edge(self, edge, ah: float = 5.0, draw=None):
        """
        Find a route avoiding node bounds for the given edge.

        Assumes the visibility graph has been created (by prepare_edge_routing)
        and that nodes have an index property.

        Args:
            edge: The edge to generate a route for
            ah: Arrow head size (distance to shorten end of edge)
            draw: Optional callback for debugging

        Returns:
            List of points for the route
        """
        line_data = []

        # Create a copy of visibility graph
        vg2 = TangentVisibilityGraph(
            self._visibilityGraph.P, V=self._visibilityGraph.V, E=self._visibilityGraph.E
        )

        port1 = Point(edge.source.x, edge.source.y)
        port2 = Point(edge.target.x, edge.target.y)

        start = vg2.add_point(port1, edge.source.index)
        end = vg2.add_point(port2, edge.target.index)

        vg2.add_edge_if_visible(port1, port2, edge.source.index, edge.target.index)

        if draw is not None:
            draw(vg2)

        # Find shortest path
        sp_calc = Calculator(
            len(vg2.V), vg2.E, lambda e: e.source.id, lambda e: e.target.id, lambda e: e.length()
        )

        shortest_path = sp_calc.path_from_node_to_node(start.id, end.id)

        if len(shortest_path) == 1 or len(shortest_path) == len(vg2.V):
            route = make_edge_between(edge.source.innerBounds, edge.target.innerBounds, ah)
            line_data = [route["sourceIntersection"], route["arrowStart"]]
        else:
            n = len(shortest_path) - 2
            p = vg2.V[shortest_path[n]].p
            q = vg2.V[shortest_path[0]].p

            line_data = [edge.source.innerBounds.ray_intersection(p.x, p.y)]

            for i in range(n, -1, -1):
                line_data.append(vg2.V[shortest_path[i]].p)

            line_data.append(make_edge_to(q, edge.target.innerBounds, ah))

        return line_data

    @staticmethod
    def get_source_index(e: Link) -> int:
        """
        Get source index from a link.

        The link source may be a node index or a reference to a node.

        Args:
            e: The link

        Returns:
            Source node index
        """
        if isinstance(e.source, int):
            return e.source
        else:
            return e.source.index

    @staticmethod
    def get_target_index(e: Link) -> int:
        """
        Get target index from a link.

        The link target may be a node index or a reference to a node.

        Args:
            e: The link

        Returns:
            Target node index
        """
        if isinstance(e.target, int):
            return e.target
        else:
            return e.target.index

    @staticmethod
    def link_id(e: Link) -> str:
        """
        Get a string ID for a link.

        Args:
            e: The link

        Returns:
            String ID in format "source-target"
        """
        return f"{Layout.get_source_index(e)}-{Layout.get_target_index(e)}"

    @staticmethod
    def drag_start(d: Union[Node, Group]) -> None:
        """
        Handle drag start event.

        The fixed property has three bits:
        - Bit 1: Set externally (e.g., d.fixed = True) and persists
        - Bit 2: Stores dragging state (mousedown to mouseup)
        - Bit 3: Stores hover state (mouseover to mouseout)

        Args:
            d: Node or group being dragged
        """
        if is_group(d):
            Layout._store_offset(d, Layout.drag_origin(d))
        else:
            Layout._stop_node(d)
            d.fixed |= 2  # Set bit 2

    @staticmethod
    def _stop_node(v: Node) -> None:
        """
        Stop a node by clobbering desired positions.

        Args:
            v: Node to stop
        """
        v.px = v.x
        v.py = v.y

    @staticmethod
    def _store_offset(d: Group, origin: dict) -> None:
        """
        Store offsets for nodes relative to group center.

        Args:
            d: Group being dragged
            origin: Origin position with 'x' and 'y' keys
        """
        if d.leaves is not None:
            for v in d.leaves:
                v.fixed |= 2
                Layout._stop_node(v)
                v._dragGroupOffsetX = v.x - origin["x"]
                v._dragGroupOffsetY = v.y - origin["y"]

        if d.groups is not None:
            for g in d.groups:
                Layout._store_offset(g, origin)

    @staticmethod
    def drag_origin(d: Union[Node, Group]) -> dict:
        """
        Get drag origin for a node or group.

        The drag origin is taken as the center of the node or group.

        Args:
            d: Node or group

        Returns:
            Dict with 'x' and 'y' keys
        """
        if is_group(d):
            return {"x": d.bounds.cx(), "y": d.bounds.cy()}
        else:
            return {"x": d.x, "y": d.y}

    @staticmethod
    def drag(d: Union[Node, Group], position: dict) -> None:
        """
        Handle drag event.

        For groups, the drag translation is propagated to all children.

        Args:
            d: Node or group being dragged
            position: New position with 'x' and 'y' keys
        """
        if is_group(d):
            if d.leaves is not None:
                for v in d.leaves:
                    d.bounds.set_x_centre(position["x"])
                    d.bounds.set_y_centre(position["y"])
                    v.px = v._dragGroupOffsetX + position["x"]
                    v.py = v._dragGroupOffsetY + position["y"]

            if d.groups is not None:
                for g in d.groups:
                    Layout.drag(g, position)
        else:
            d.px = position["x"]
            d.py = position["y"]

    @staticmethod
    def drag_end(d: Union[Node, Group]) -> None:
        """
        Handle drag end event.

        Unsets bits 2 and 3 so user-set locks persist between drags.

        Args:
            d: Node or group being dragged
        """
        if is_group(d):
            if d.leaves is not None:
                for v in d.leaves:
                    Layout.drag_end(v)
                    if hasattr(v, "_dragGroupOffsetX"):
                        delattr(v, "_dragGroupOffsetX")
                    if hasattr(v, "_dragGroupOffsetY"):
                        delattr(v, "_dragGroupOffsetY")

            if d.groups is not None:
                for g in d.groups:
                    Layout.drag_end(g)
        else:
            d.fixed &= ~6  # Unset bits 2 and 3

    @staticmethod
    def mouse_over(d: Node) -> None:
        """
        Handle mouse over event (temporarily locks node).

        Args:
            d: Node
        """
        d.fixed |= 4  # Set bit 3
        d.px = d.x
        d.py = d.y  # Set velocity to zero

    @staticmethod
    def mouse_out(d: Node) -> None:
        """
        Handle mouse out event (unlocks node).

        Args:
            d: Node
        """
        d.fixed &= ~4  # Unset bit 3
