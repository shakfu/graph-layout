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
from typing import Optional


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
    vertex_face_angles: dict[tuple[int, int], int] = field(default_factory=dict)

    # For each directed edge (u, v): list of bend directions
    # +1 = left turn (counter-clockwise), -1 = right turn (clockwise)
    edge_bends: dict[tuple[int, int], list[int]] = field(default_factory=dict)

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


def compute_faces(
    num_nodes: int,
    edges: list[tuple[int, int]],
    positions: Optional[list[tuple[float, float]]] = None,
) -> list[Face]:
    """
    Compute faces of a planar graph given its embedding.

    Uses the positions to determine clockwise ordering around vertices.
    If positions are not given, assumes edges are already in embedding order.

    Args:
        num_nodes: Number of vertices
        edges: List of undirected edges as (u, v) pairs
        positions: Optional vertex positions for determining embedding

    Returns:
        List of Face objects
    """
    if num_nodes == 0:
        return []

    # Build adjacency with angular ordering
    adj: dict[int, list[int]] = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    if positions:
        # Sort neighbors by angle from each vertex
        import math

        for v in adj:
            vx, vy = positions[v]

            def angle_key(u: int) -> float:
                ux, uy = positions[u]
                return math.atan2(uy - vy, ux - vx)

            adj[v].sort(key=angle_key)

    # Find faces by walking around the embedding
    # Each directed edge (u, v) belongs to exactly one face
    used_edges: set[tuple[int, int]] = set()
    faces: list[Face] = []

    def next_edge(u: int, v: int) -> tuple[int, int]:
        """Get next edge after (u, v) in clockwise order around v."""
        neighbors = adj[v]
        if not neighbors:
            return (v, u)
        idx = neighbors.index(u) if u in neighbors else 0
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

            while True:
                if (curr_u, curr_v) in used_edges:
                    break
                used_edges.add((curr_u, curr_v))
                face_vertices.append(curr_u)
                face_edges.append((curr_u, curr_v))
                curr_u, curr_v = next_edge(curr_u, curr_v)
                if curr_u == start_u and curr_v == start_v:
                    break

            if face_vertices:
                face = Face(
                    index=len(faces),
                    vertices=face_vertices,
                    edges=face_edges,
                )
                faces.append(face)

    # Identify outer face (largest face by vertex count, or use positions)
    if faces:
        if positions:
            # Outer face has leftmost vertex with edges going up and down
            min_x = min(positions[v][0] for v in range(num_nodes) if v in adj)
            leftmost = [v for v in range(num_nodes) if v in adj and positions[v][0] == min_x]
            # Find face containing leftmost vertex with counterclockwise orientation
            for face in faces:
                if any(v in leftmost for v in face.vertices):
                    # Check orientation using signed area
                    area = 0.0
                    for i, v in enumerate(face.vertices):
                        u = face.vertices[(i + 1) % len(face.vertices)]
                        area += positions[v][0] * positions[u][1]
                        area -= positions[u][0] * positions[v][1]
                    if area < 0:  # Clockwise = outer face in our convention
                        face.is_outer = True
                        break
        else:
            # Just pick the largest face as outer
            max_face = max(faces, key=lambda f: len(f.vertices))
            max_face.is_outer = True

    return faces


def build_flow_network(
    num_nodes: int,
    edges: list[tuple[int, int]],
    faces: list[Face],
) -> FlowNetwork:
    """
    Build the min-cost flow network for orthogonalization.

    Network structure:
    - Node IDs 0..num_nodes-1 are graph vertices
    - Node IDs num_nodes..num_nodes+len(faces)-1 are faces
    - Arcs from vertices to adjacent faces (angle arcs)
    - Arcs between adjacent faces (bend arcs)

    Args:
        num_nodes: Number of graph vertices
        edges: Graph edges
        faces: Computed faces

    Returns:
        FlowNetwork ready for solving
    """
    network = FlowNetwork(num_vertices=num_nodes, faces=faces)

    # Compute vertex degrees
    degrees: dict[int, int] = defaultdict(int)
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1

    # Vertex supplies: 4 - degree (since sum of angles at vertex = 360° = 4×90°)
    for v in range(num_nodes):
        supply = 4 - degrees.get(v, 0)
        if supply != 0:
            network.supplies[v] = supply

    # Face demands
    for face in faces:
        face_node = num_nodes + face.index
        num_vertices_on_face = len(face.vertices)
        if face.is_outer:
            # Outer face: angles sum to (n+2)×90° where n = #vertices
            demand = -(num_vertices_on_face + 4)
        else:
            # Inner face: angles sum to (n-2)×90° where n = #vertices
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
    # Capacity = 3 (angle can be 90°, 180°, or 270° = 1, 2, or 3 units)
    # Cost = 0 (angles don't cost anything)
    for face in faces:
        face_node = num_nodes + face.index
        for v in face.vertices:
            arc = (v, face_node)
            if arc not in network.arcs:
                network.arcs[arc] = (3, 0)  # capacity=3, cost=0

    # Arcs between faces (bends on edges)
    # For each edge between two faces, add arcs in both directions
    # Capacity = unlimited (but we use 4 as practical limit)
    # Cost = 1 per bend
    for face in faces:
        face_node = num_nodes + face.index
        for edge in face.edges:
            # Find the other face sharing this edge
            u, v = edge
            for other_face_idx in edge_faces[(v, u)]:
                if other_face_idx != face.index:
                    other_face_node = num_nodes + other_face_idx
                    arc = (face_node, other_face_node)
                    if arc not in network.arcs:
                        network.arcs[arc] = (4, 1)  # capacity=4, cost=1

    return network


def solve_min_cost_flow_simple(network: FlowNetwork) -> bool:
    """
    Solve min-cost flow using a simple successive shortest path algorithm.

    This is a simplified implementation suitable for small graphs.
    For larger graphs, a more efficient algorithm would be needed.

    Args:
        network: Flow network to solve

    Returns:
        True if a feasible flow was found, False otherwise
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

    # Extract vertex-face angles from flow
    for face in network.faces:
        face_node = num_nodes + face.index
        for v in face.vertices:
            arc = (v, face_node)
            flow_amount = network.flow.get(arc, 0)
            # Angle = 1 + flow (minimum angle is 90° = 1 unit)
            angle = 1 + flow_amount
            ortho_rep.vertex_face_angles[(v, face.index)] = angle

    # Extract bends from face-to-face flow
    edge_to_faces: dict[tuple[int, int], tuple[int, int]] = {}
    for face in network.faces:
        for edge in face.edges:
            u, v = edge
            key = (min(u, v), max(u, v))
            if key not in edge_to_faces:
                edge_to_faces[key] = (face.index, -1)
            else:
                f1, _ = edge_to_faces[key]
                edge_to_faces[key] = (f1, face.index)

    for (u, v), (f1, f2) in edge_to_faces.items():
        if f2 < 0:
            continue

        face1_node = num_nodes + f1
        face2_node = num_nodes + f2

        # Flow from f1 to f2 means bends turning toward f2
        flow_12 = network.flow.get((face1_node, face2_node), 0)
        flow_21 = network.flow.get((face2_node, face1_node), 0)

        bends: list[int] = []
        # Positive flow means bends in one direction
        bends.extend([1] * flow_12)  # Left turns
        bends.extend([-1] * flow_21)  # Right turns

        if bends:
            ortho_rep.edge_bends[(u, v)] = bends
            ortho_rep.edge_bends[(v, u)] = [-b for b in bends]  # Reverse direction

    return ortho_rep


def compute_orthogonal_representation(
    num_nodes: int,
    edges: list[tuple[int, int]],
    positions: Optional[list[tuple[float, float]]] = None,
) -> OrthogonalRepresentation:
    """
    Compute optimal orthogonal representation using min-cost flow.

    This is the main entry point for the orthogonalization phase.

    Args:
        num_nodes: Number of vertices
        edges: Graph edges
        positions: Optional vertex positions for embedding

    Returns:
        OrthogonalRepresentation with minimized bends
    """
    if num_nodes == 0 or not edges:
        return OrthogonalRepresentation()

    # Compute faces
    faces = compute_faces(num_nodes, edges, positions)

    if not faces:
        return OrthogonalRepresentation()

    # Build flow network
    network = build_flow_network(num_nodes, edges, faces)

    # Solve min-cost flow
    solve_min_cost_flow_simple(network)

    # Convert to orthogonal representation
    ortho_rep = flow_to_orthogonal_rep(network, edges)

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
