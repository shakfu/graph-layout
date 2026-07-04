"""
Orthogonalization phase for Kandinsky layout.

Implements bend minimization using a flow network formulation.
Based on Tamassia's algorithm for orthogonal graph drawing.

The key insight is that the problem of minimizing bends can be
formulated as a min-cost flow problem:
- Vertices supply flow based on their degree (4 - degree)
- Faces consume flow based on their size
- Flow on arcs represents angles at vertices and bends on edges
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..planarity._embedding import PlanarEmbedding


class AngleType(Enum):
    """Angle types in orthogonal representation."""

    DEGREE_90 = 1  # 90° angle (right angle)
    DEGREE_180 = 2  # 180° angle (straight)
    DEGREE_270 = 3  # 270° angle (reflex)
    DEGREE_0 = 0  # 0° angle (Kandinsky only - same direction)


@dataclass
class OrthogonalRepresentation:
    """
    Orthogonal representation of a graph.

    Stores the shape of an orthogonal drawing without coordinates.
    The shape consists of:
    - Angle assignments at each vertex-face incidence
    - Bend sequences along each edge
    """

    # For each (vertex, face) pair: angle in units of 90°
    # angle = 1 means 90°, angle = 2 means 180°, etc.
    # Only filled for corners whose vertex borders the face exactly once;
    # see ``corner_angles`` for the general (per-corner) storage.
    vertex_face_angles: dict[tuple[int, int], int] = field(default_factory=dict)

    # Per-corner angles keyed by the *incoming dart*: the corner following
    # directed edge (a, b) in its face walk sits at vertex b, and every dart
    # appears in exactly one face walk, so the key is unique. This is what
    # supports bridges / cut vertices, where a vertex borders the same face
    # at several corners and a (vertex, face) key would collide.
    corner_angles: dict[tuple[int, int], int] = field(default_factory=dict)

    # For each directed edge (u, v): list of bend directions
    # +1 = left turn (counter-clockwise), -1 = right turn (clockwise)
    edge_bends: dict[tuple[int, int], list[int]] = field(default_factory=dict)

    def corner_angle(self, dart: tuple[int, int], face_index: int) -> Optional[int]:
        """Angle (in 90° units) at the corner following ``dart`` in its face.

        Prefers the per-corner storage; falls back to the legacy
        (vertex, face) keyed dict for representations built by hand.
        """
        angle = self.corner_angles.get(dart)
        if angle is not None:
            return angle
        return self.vertex_face_angles.get((dart[1], face_index))

    # Total number of bends
    @property
    def total_bends(self) -> int:
        """Count total bends across all edges."""
        return sum(len(bends) for bends in self.edge_bends.values())


@dataclass
class Face:
    """A face in the planar embedding."""

    index: int
    vertices: list[int]  # Vertices in clockwise order
    edges: list[tuple[int, int]]  # Edges (as directed pairs) around face
    is_outer: bool = False


@dataclass
class FlowNetwork:
    """
    Flow network for orthogonalization.

    Nodes are vertices and faces of the graph.
    Arcs represent angle assignments and bends.
    """

    # Number of original graph vertices
    num_vertices: int

    # Faces
    faces: list[Face]

    # Node supplies (positive) and demands (negative)
    # Vertices: supply = 4 - degree
    # Faces: demand = |vertices on face| - 4 (inner) or +4 (outer)
    supplies: dict[int, int] = field(default_factory=dict)

    # Arcs: (from_node, to_node) -> (capacity, cost)
    arcs: dict[tuple[int, int], tuple[int, int]] = field(default_factory=dict)

    # Flow solution: (from_node, to_node) -> flow amount
    flow: dict[tuple[int, int], int] = field(default_factory=dict)

    # Per-edge bend arcs. Bends are modeled one edge at a time (not one
    # face-pair at a time) so that two faces sharing several edges get an
    # independent bend variable per edge. Keyed by the undirected edge
    # (min(u, v), max(u, v)) -> (dart_f1, arc_toward_f2, arc_toward_f1) where
    # dart_f1 is the directed edge bordering the first face and each arc is the
    # (from_node, intermediate_node) pair whose flow is the number of bends of
    # that edge turning toward the corresponding face.
    bend_arcs: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = (
        field(default_factory=dict)
    )


def _sanitize_edges(
    edges: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Remove self-loops and deduplicate multi-edges for face computation.

    Returns:
        Tuple of (clean_edges, removed_self_loops).
    """
    self_loops: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    clean: list[tuple[int, int]] = []
    for u, v in edges:
        if u == v:
            self_loops.append((u, v))
            continue
        canon = (min(u, v), max(u, v))
        if canon not in seen:
            seen.add(canon)
            clean.append((u, v))
    return clean, self_loops


def compute_faces(
    num_nodes: int,
    edges: list[tuple[int, int]],
    positions: Optional[list[tuple[float, float]]] = None,
    embedding: Optional[PlanarEmbedding] = None,
) -> list[Face]:
    """
    Compute faces of a planar graph given its embedding.

    When *embedding* is provided, faces are derived directly from the
    combinatorial rotation system and the embedding's ``outer_face_index``
    is used to mark the outer face.  This avoids the fragile position-based
    angular sorting.

    Otherwise falls back to using *positions* (or largest-face heuristic).

    Self-loops and duplicate multi-edges are filtered before face computation
    (they do not participate in the planar face structure).

    Args:
        num_nodes: Number of vertices
        edges: List of undirected edges as (u, v) pairs
        positions: Optional vertex positions for determining embedding
        embedding: Optional PlanarEmbedding from a planarity test / embedder

    Returns:
        List of Face objects
    """
    if num_nodes == 0:
        return []

    # Sanitize: strip self-loops and duplicate multi-edges
    clean_edges, _self_loops = _sanitize_edges(edges)

    # ---- Fast path: use combinatorial embedding directly ----
    if embedding is not None:
        # Validate embedding consistency; fall back to legacy if broken
        if not embedding.verify():
            import warnings

            warnings.warn(
                "PlanarEmbedding failed verification; falling back to position-based "
                "face computation.",
                stacklevel=2,
            )
            return _compute_faces_legacy(num_nodes, clean_edges, positions)

        raw_faces = embedding.faces()
        faces: list[Face] = []
        for fi, directed_edges in enumerate(raw_faces):
            verts = [u for u, _ in directed_edges]
            face = Face(
                index=fi,
                vertices=verts,
                edges=directed_edges,
                is_outer=(fi == embedding.outer_face_index),
            )
            faces.append(face)

        # If outer_face_index was not set, fall back to largest face
        if embedding.outer_face_index is None and faces:
            max_face = max(faces, key=lambda f: len(f.vertices))
            max_face.is_outer = True

        return faces

    return _compute_faces_legacy(num_nodes, clean_edges, positions)


def _compute_faces_legacy(
    num_nodes: int,
    edges: list[tuple[int, int]],
    positions: Optional[list[tuple[float, float]]] = None,
) -> list[Face]:
    """Legacy position-based face computation.

    Edges must already be sanitized (no self-loops, no duplicate multi-edges).
    Handles disconnected components by tracing faces per component and
    selecting the global outer face from the largest face overall.
    """
    # Build adjacency with angular ordering
    adj: dict[int, list[int]] = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Deduplicate neighbor lists (guards against any residual multi-edge entries)
    for v in adj:
        adj[v] = sorted(set(adj[v]))

    if positions:
        # Sort neighbors by angle from each vertex
        import math

        for v in adj:
            vx, vy = positions[v]

            def angle_key(u: int, _vx: float = vx, _vy: float = vy) -> float:
                ux, uy = positions[u]
                return math.atan2(uy - _vy, ux - _vx)

            adj[v].sort(key=angle_key)

    # Find faces by walking around the embedding
    # Each directed edge (u, v) belongs to exactly one face
    used_edges: set[tuple[int, int]] = set()
    faces_list: list[Face] = []

    def next_edge(u: int, v: int) -> tuple[int, int]:
        """Get next edge after (u, v) in clockwise order around v."""
        neighbors = adj[v]
        if not neighbors:
            return (v, u)
        try:
            idx = neighbors.index(u)
        except ValueError:
            idx = 0
        # Next in clockwise order (previous in the sorted list)
        next_idx = (idx - 1) % len(neighbors)
        return (v, neighbors[next_idx])

    for u, v in edges:
        for start_u, start_v in [(u, v), (v, u)]:
            if (start_u, start_v) in used_edges:
                continue

            # Walk around face
            face_vertices: list[int] = []
            face_edges: list[tuple[int, int]] = []
            curr_u, curr_v = start_u, start_v

            max_steps = 2 * len(edges) + 2  # safety bound
            steps = 0
            while steps < max_steps:
                if (curr_u, curr_v) in used_edges:
                    break
                used_edges.add((curr_u, curr_v))
                face_vertices.append(curr_u)
                face_edges.append((curr_u, curr_v))
                curr_u, curr_v = next_edge(curr_u, curr_v)
                steps += 1
                if curr_u == start_u and curr_v == start_v:
                    break

            if face_vertices:
                face = Face(
                    index=len(faces_list),
                    vertices=face_vertices,
                    edges=face_edges,
                )
                faces_list.append(face)

    # Identify outer face (largest face by vertex count, or use positions)
    if faces_list:
        if positions:
            # Find vertices that participate in edges
            verts_in_adj = [v for v in range(num_nodes) if v in adj and adj[v]]
            if verts_in_adj:
                min_x = min(positions[v][0] for v in verts_in_adj)
                leftmost = [v for v in verts_in_adj if positions[v][0] == min_x]
                for face in faces_list:
                    if any(v in leftmost for v in face.vertices):
                        area = 0.0
                        for i, v in enumerate(face.vertices):
                            u = face.vertices[(i + 1) % len(face.vertices)]
                            area += positions[v][0] * positions[u][1]
                            area -= positions[u][0] * positions[v][1]
                        if area < 0:
                            face.is_outer = True
                            break
            if not any(f.is_outer for f in faces_list):
                max_face = max(faces_list, key=lambda f: len(f.vertices))
                max_face.is_outer = True
        else:
            max_face = max(faces_list, key=lambda f: len(f.vertices))
            max_face.is_outer = True

    return faces_list


def build_flow_network(
    num_nodes: int,
    edges: list[tuple[int, int]],
    faces: list[Face],
    *,
    cage_faces: Optional[set[int]] = None,
    rigid_edges: Optional[set[tuple[int, int]]] = None,
) -> FlowNetwork:
    """
    Build the min-cost flow network for orthogonalization.

    Network structure:
    - Node IDs 0..num_nodes-1 are graph vertices
    - Node IDs num_nodes..num_nodes+len(faces)-1 are faces
    - Arcs from vertices to adjacent faces (angle arcs)
    - Arcs between adjacent faces (bend arcs)

    Self-loops are filtered from ``edges`` before degree computation so they
    do not corrupt supply values.

    Args:
        num_nodes: Number of graph vertices
        edges: Graph edges
        faces: Computed faces
        cage_faces: Face indices that must be drawn as rectangles (the cage
            faces of expanded degree > 4 vertices): their corner angles are
            capped at 180 degrees.
        rigid_edges: Canonical (min, max) edges that must stay straight (cage
            cycle edges -- a node box side cannot bend): no bend arcs.

    Returns:
        FlowNetwork ready for solving
    """
    network = FlowNetwork(num_vertices=num_nodes, faces=faces)

    # Filter self-loops for degree computation
    clean_edges = [(u, v) for u, v in edges if u != v]

    # Compute vertex degrees (only count vertices that appear in faces)
    degrees: dict[int, int] = defaultdict(int)
    for u, v in clean_edges:
        degrees[u] += 1
        degrees[v] += 1

    # Determine which vertices participate in faces
    face_vertices: set[int] = set()
    for face in faces:
        face_vertices.update(face.vertices)

    # Vertex supplies: 4 - degree (since sum of angles at vertex = 360° = 4*90 degrees)
    # Only include vertices that actually appear in the face structure
    for v in face_vertices:
        supply = 4 - degrees.get(v, 0)
        if supply != 0:
            network.supplies[v] = supply

    # Face demands
    for face in faces:
        face_node = num_nodes + face.index
        num_vertices_on_face = len(face.vertices)
        if face.is_outer:
            # Outer face: angles sum to (n+2)*90 degrees where n = #vertices
            demand = -(num_vertices_on_face + 4)
        else:
            # Inner face: angles sum to (n-2)*90 degrees where n = #vertices
            demand = -(num_vertices_on_face - 4)

        if demand != 0:
            network.supplies[face_node] = demand

    # Build edge-to-faces mapping
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face in faces:
        for edge in face.edges:
            edge_faces[edge].append(face.index)
            # Also add reverse for undirected lookup
            edge_faces[(edge[1], edge[0])].append(face.index)

    # Arcs from vertices to faces (angle assignments)
    # Capacity = 3 per corner (angle can be up to 360 degrees = flow 3 above the
    # 90-degree minimum), times the number of corners the vertex has on the
    # face. A vertex borders a face more than once at cut vertices / bridges;
    # the arc then carries the total over those corners and the extraction
    # distributes it per corner.
    # Cost = 0 (angles don't cost anything)
    # Corners of cage faces (expanded degree > 4 vertices) are capped at 180
    # degrees (flow 1) so the cage is drawn as a rectangle.
    for face in faces:
        face_node = num_nodes + face.index
        per_corner = 1 if cage_faces and face.index in cage_faces else 3
        multiplicity: dict[int, int] = defaultdict(int)
        for v in face.vertices:
            multiplicity[v] += 1
        for v, k in multiplicity.items():
            network.arcs[(v, face_node)] = (per_corner * k, 0)

    # Bend arcs (one bend variable *per edge*, not per face-pair).
    #
    # In Tamassia's flow model a bend of edge e is a unit of flow between the
    # two faces e separates, at cost 1. Modeling it as a single arc per
    # face-pair (the previous approach) is wrong whenever two faces share more
    # than one edge -- e.g. the two length-2 sides of a theta graph -- because
    # every shared edge would then read the same lumped flow.
    #
    # To give each edge an independent variable while keeping the arc dict
    # keyed by node pairs (no parallel arcs), route each edge's bends through a
    # unique intermediate node: for edge e between faces f1 and f2,
    #   f1 -> em1 (cost 1) -> f2      flow = bends of e turning toward f2
    #   f2 -> em2 (cost 0? no: cost 1) -> f1   flow = bends turning toward f1
    # The cost sits on the first hop so each bend costs exactly 1; the second
    # hop is free. Capacity is generous rather than the old hard cap of 4.
    #
    # Determine the (ordered) pair of faces bordering each undirected edge, in
    # face-index order of first appearance, and remember the *directed* edge
    # (dart) that borders the first face. The dart is needed so bends can be
    # attributed to the correct side with the correct turn sign: a bend flowing
    # f1 -> f2 is convex (a +1 quarter turn) on the dart bordering f1 and reflex
    # (-1) on the dart bordering f2. Getting this incidence right is what makes
    # every face turn by +/-4 (a valid orthogonal representation).
    edge_face_pair: dict[tuple[int, int], list[int]] = {}
    edge_first_dart: dict[tuple[int, int], tuple[int, int]] = {}
    for face in faces:
        for u, v in face.edges:
            key = (min(u, v), max(u, v))
            lst = edge_face_pair.setdefault(key, [])
            if face.index not in lst:
                lst.append(face.index)
                if len(lst) == 1:
                    edge_first_dart[key] = (u, v)  # dart bordering f1

    bend_capacity = 2 * len(edges) + 4  # effectively uncapped for a planar graph
    next_node = num_nodes + len(faces)
    for key in sorted(edge_face_pair):
        face_list = edge_face_pair[key]
        if len(face_list) < 2 or face_list[0] == face_list[1]:
            # Bridge (same face on both sides) or boundary edge: not bendable
            # in the flow model.
            continue
        if rigid_edges and key in rigid_edges:
            # Cage cycle edge: a node box side must stay straight.
            continue
        f1, f2 = face_list[0], face_list[1]
        dart_f1 = edge_first_dart[key]
        f1_node = num_nodes + f1
        f2_node = num_nodes + f2

        em1 = next_node
        next_node += 1
        em2 = next_node
        next_node += 1

        # Bends of this edge turning toward f2.
        network.arcs[(f1_node, em1)] = (bend_capacity, 1)
        network.arcs[(em1, f2_node)] = (bend_capacity, 0)
        # Bends turning toward f1.
        network.arcs[(f2_node, em2)] = (bend_capacity, 1)
        network.arcs[(em2, f1_node)] = (bend_capacity, 0)

        network.bend_arcs[key] = (dart_f1, (f1_node, em1), (f2_node, em2))

    return network


def _solve_bellman_ford(network: FlowNetwork) -> bool:
    """
    Solve min-cost flow using successive shortest paths with Bellman-Ford.

    Kept as a private reference implementation for equivalence testing.
    """
    # Initialize flow to zero
    network.flow = {arc: 0 for arc in network.arcs}

    # Build residual graph
    # Residual capacity = capacity - flow (forward) or flow (backward)
    def get_residual_capacity(u: int, v: int) -> int:
        if (u, v) in network.arcs:
            cap, _ = network.arcs[(u, v)]
            return cap - network.flow.get((u, v), 0)
        elif (v, u) in network.arcs:
            return network.flow.get((v, u), 0)
        return 0

    def get_cost(u: int, v: int) -> int:
        if (u, v) in network.arcs:
            _, cost = network.arcs[(u, v)]
            return cost
        elif (v, u) in network.arcs:
            _, cost = network.arcs[(v, u)]
            return -cost
        return 0

    # Find all nodes
    all_nodes: set[int] = set()
    for v in range(network.num_vertices):
        all_nodes.add(v)
    for face in network.faces:
        all_nodes.add(network.num_vertices + face.index)
    # Include auxiliary nodes referenced only by arcs (per-edge bend nodes).
    for u_arc, v_arc in network.arcs:
        all_nodes.add(u_arc)
        all_nodes.add(v_arc)

    # Get supply nodes and demand nodes
    supply_nodes = [(n, s) for n, s in network.supplies.items() if s > 0]
    demand_nodes = [(n, -s) for n, s in network.supplies.items() if s < 0]

    # Simple approach: send flow from each supply to demands
    for supply_node, supply_amount in supply_nodes:
        remaining_supply = supply_amount

        while remaining_supply > 0:
            # Find shortest path to any demand node using Bellman-Ford
            dist = {n: float("inf") for n in all_nodes}
            parent: dict[int, Optional[int]] = {n: None for n in all_nodes}
            dist[supply_node] = 0

            # Bellman-Ford relaxation
            for _ in range(len(all_nodes)):
                updated = False
                for (u, v), (cap, cost) in network.arcs.items():
                    if get_residual_capacity(u, v) > 0:
                        if dist[u] + cost < dist[v]:
                            dist[v] = dist[u] + cost
                            parent[v] = u
                            updated = True
                    # Backward arc
                    if network.flow.get((u, v), 0) > 0:
                        if dist[v] - cost < dist[u]:
                            dist[u] = dist[v] - cost
                            parent[u] = v
                            updated = True
                if not updated:
                    break

            # Find reachable demand node
            best_demand = None
            best_dist = float("inf")
            for demand_node, demand_amount in demand_nodes:
                if dist[demand_node] < best_dist and demand_amount > 0:
                    best_dist = dist[demand_node]
                    best_demand = demand_node

            if best_demand is None or best_dist == float("inf"):
                # No path found, flow might not be feasible
                break

            # Find path
            path = []
            curr: Optional[int] = best_demand
            while curr is not None and curr != supply_node:
                prev = parent[curr]
                if prev is not None:
                    path.append((prev, curr))
                curr = prev

            if not path:
                break

            path.reverse()

            # Find bottleneck capacity
            bottleneck = remaining_supply
            for u, v in path:
                cap = get_residual_capacity(u, v)
                bottleneck = min(bottleneck, cap)

            # Also limit by remaining demand
            for i, (demand_node, demand_amount) in enumerate(demand_nodes):
                if demand_node == best_demand:
                    bottleneck = min(bottleneck, demand_amount)
                    demand_nodes[i] = (demand_node, demand_amount - bottleneck)
                    break

            if bottleneck <= 0:
                break

            # Augment flow
            for u, v in path:
                if (u, v) in network.arcs:
                    network.flow[(u, v)] = network.flow.get((u, v), 0) + bottleneck
                else:
                    network.flow[(v, u)] = network.flow.get((v, u), 0) - bottleneck

            remaining_supply -= bottleneck

    return True


def solve_min_cost_flow_simple(network: FlowNetwork) -> bool:
    """
    Solve min-cost flow for orthogonalization bend minimization.

    Delegates to the Dijkstra-based successive shortest paths solver
    which is O(V^2 log V) for planar graphs, replacing the previous
    Bellman-Ford implementation.

    Args:
        network: Flow network to solve

    Returns:
        True if a feasible flow was found, False otherwise
    """
    from ._min_cost_flow import solve_min_cost_flow

    return solve_min_cost_flow(network)


def flow_to_orthogonal_rep(
    network: FlowNetwork,
    edges: list[tuple[int, int]],
) -> OrthogonalRepresentation:
    """
    Convert flow solution to orthogonal representation.

    Args:
        network: Solved flow network
        edges: Original graph edges

    Returns:
        OrthogonalRepresentation with angle and bend assignments
    """
    ortho_rep = OrthogonalRepresentation()

    num_nodes = network.num_vertices

    # Extract per-corner angles from flow. The corner following dart (a, b) in
    # the face walk sits at vertex b. A vertex may border the same face at
    # several corners (cut vertices / bridges); the arc's flow is then the
    # total over those corners and is distributed greedily in walk order --
    # any distribution with each angle in [1, 4] yields a valid representation
    # with the same bend count, because face turn sums and per-vertex angle
    # sums depend only on the totals.
    for face in network.faces:
        face_node = num_nodes + face.index
        corners_by_vertex: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for dart in face.edges:
            corners_by_vertex[dart[1]].append(dart)
        for v, corner_darts in corners_by_vertex.items():
            remaining = network.flow.get((v, face_node), 0)
            for dart in corner_darts:
                extra = min(3, remaining)
                remaining -= extra
                # Angle = 1 + flow (minimum angle is 90° = 1 unit)
                ortho_rep.corner_angles[dart] = 1 + extra
            if len(corner_darts) == 1:
                # Legacy (vertex, face) key -- unambiguous only for single corners
                ortho_rep.vertex_face_angles[(v, face.index)] = ortho_rep.corner_angles[
                    corner_darts[0]
                ]

    # Extract bends per edge from its own intermediate-node flow. Each edge has
    # its own pair of bend arcs, so two edges shared by the same face pair no
    # longer read (and duplicate) a single lumped flow value.
    #
    # Sign convention (this is what makes every face turn by +/-4): a unit of
    # flow f1 -> f2 is a bend that is convex -- a +1 quarter turn -- when the
    # edge is traversed along the dart bordering f1, and reflex (-1) along the
    # reverse dart bordering f2. Attributing the turn to the dart that actually
    # borders each face is essential; keying by the raw edge tuple (the old
    # behaviour) left the sign unaligned with the embedding and produced
    # inconsistent, unrealizable representations.
    for _key, (dart_f1, arc_toward_f2, arc_toward_f1) in network.bend_arcs.items():
        flow_toward_f2 = network.flow.get(arc_toward_f2, 0)
        flow_toward_f1 = network.flow.get(arc_toward_f1, 0)

        # Turns along the dart bordering f1.
        bends_f1 = [1] * flow_toward_f2 + [-1] * flow_toward_f1
        if bends_f1:
            dart_f2 = (dart_f1[1], dart_f1[0])
            ortho_rep.edge_bends[dart_f1] = bends_f1
            ortho_rep.edge_bends[dart_f2] = [-b for b in bends_f1]

    return ortho_rep


def compute_orthogonal_representation(
    num_nodes: int,
    edges: list[tuple[int, int]],
    positions: Optional[list[tuple[float, float]]] = None,
    embedding: Optional[PlanarEmbedding] = None,
) -> OrthogonalRepresentation:
    """
    Compute optimal orthogonal representation using min-cost flow.

    This is the main entry point for the orthogonalization phase.

    Args:
        num_nodes: Number of vertices
        edges: Graph edges
        positions: Optional vertex positions for embedding
        embedding: Optional PlanarEmbedding (preferred over positions)

    Returns:
        OrthogonalRepresentation with minimized bends
    """
    if num_nodes == 0 or not edges:
        return OrthogonalRepresentation()

    # Sanitize edges for flow network (compute_faces does its own sanitization)
    clean_edges, _self_loops = _sanitize_edges(edges)
    if not clean_edges:
        return OrthogonalRepresentation()

    # Compute faces
    faces = compute_faces(num_nodes, clean_edges, positions, embedding=embedding)

    if not faces:
        return OrthogonalRepresentation()

    # Build flow network
    network = build_flow_network(num_nodes, clean_edges, faces)

    # Solve min-cost flow
    solve_min_cost_flow_simple(network)

    # Convert to orthogonal representation
    ortho_rep = flow_to_orthogonal_rep(network, clean_edges)

    return ortho_rep


__all__ = [
    "AngleType",
    "OrthogonalRepresentation",
    "Face",
    "FlowNetwork",
    "compute_faces",
    "build_flow_network",
    "solve_min_cost_flow_simple",
    "flow_to_orthogonal_rep",
    "compute_orthogonal_representation",
]
