"""
Shortest paths calculation using Dijkstra's algorithm.

This module provides efficient all-pairs shortest path calculation
using Dijkstra's algorithm with a pairing heap priority queue.
"""

from __future__ import annotations

from typing import Callable, Generic, Optional, TypeVar, cast

from .pqueue import PairingHeap, PriorityQueue

T = TypeVar("T")


class Neighbour:
    """Represents a neighbor with distance."""

    def __init__(self, id: int, distance: float):
        self.id = id
        self.distance = distance


class Node:
    """Graph node for shortest path calculation."""

    def __init__(self, id: int):
        self.id = id
        self.neighbours: list[Neighbour] = []
        self.d: float = 0.0  # Current shortest distance
        self.prev: Optional[Node] = None  # Previous node in shortest path
        self.q: Optional[PairingHeap[Node]] = None  # Priority queue node


class QueueEntry:
    """Entry in priority queue for path finding with previous cost."""

    def __init__(self, node: Node, prev: Optional[QueueEntry], d: float):
        self.node = node
        self.prev = prev
        self.d = d


class Calculator(Generic[T]):
    """
    Calculator for all-pairs shortest paths or shortest paths from a single node.

    Uses Dijkstra's algorithm with a priority queue for efficiency.
    """

    def __init__(
        self,
        n: int,
        edges: list[T],
        get_source_index: Callable[[T], int],
        get_target_index: Callable[[T], int],
        get_length: Callable[[T], float],
    ):
        """
        Initialize shortest path calculator.

        Args:
            n: Number of nodes
            edges: List of edges
            get_source_index: Function to get source node index from edge
            get_target_index: Function to get target node index from edge
            get_length: Function to get edge length
        """
        self.n = n
        self.edges = edges

        # Build adjacency list
        self.neighbours = [Node(i) for i in range(n)]

        for edge in edges:
            u = get_source_index(edge)
            v = get_target_index(edge)
            d = get_length(edge)
            self.neighbours[u].neighbours.append(Neighbour(v, d))
            self.neighbours[v].neighbours.append(Neighbour(u, d))

    def distance_matrix(self) -> list[list[float]]:
        """
        Compute all-pairs shortest paths using Johnson's algorithm.

        Returns:
            Matrix of shortest distances between all pairs of nodes
        """
        D: list[list[float]] = []
        for i in range(self.n):
            D.append(cast(list[float], self._dijkstra_neighbours(i)))
        return D

    def distances_from_node(self, start: int) -> list[float]:
        """
        Get shortest paths from a specified start node.

        Args:
            start: Starting node index

        Returns:
            Array of shortest distances from start to all other nodes
        """
        return cast(list[float], self._dijkstra_neighbours(start))

    def path_from_node_to_node(self, start: int, end: int) -> list[int]:
        """
        Find shortest path from start to end node.

        Args:
            start: Start node index
            end: End node index

        Returns:
            List of node indices in the path (excluding start, including end)
        """
        return cast(list[int], self._dijkstra_neighbours(start, end))

    def path_from_node_to_node_with_prev_cost(
        self, start: int, end: int, prev_cost: Callable[[int, int, int], float]
    ) -> list[int]:
        """
        Find shortest path with custom cost function based on previous edge.

        This allows penalizing bends or other path characteristics.

        Args:
            start: Start node index
            end: End node index
            prev_cost: Function(prev_node, current_node, next_node) -> cost

        Returns:
            List of node indices in the path
        """
        q: PriorityQueue[QueueEntry] = PriorityQueue(lambda a, b: a.d <= b.d)
        u = self.neighbours[start]
        qu = QueueEntry(u, None, 0)
        visited_from: dict[str, float] = {}
        q.push(qu)

        while not q.empty():
            popped = q.pop()
            assert popped is not None  # Queue not empty
            qu = popped
            u = qu.node

            if u.id == end:
                break

            for neighbour in u.neighbours:
                v = self.neighbours[neighbour.id]

                # Don't double back
                if qu.prev and v.id == qu.prev.node.id:
                    continue

                # Don't retraverse an edge if already explored from lower cost
                vid_uid = f"{v.id},{u.id}"
                if vid_uid in visited_from and visited_from[vid_uid] <= qu.d:
                    continue

                # Calculate cost including previous edge penalty
                cc = prev_cost(qu.prev.node.id, u.id, v.id) if qu.prev else 0
                t = qu.d + neighbour.distance + cc

                # Store cost of this traversal
                visited_from[vid_uid] = t
                q.push(QueueEntry(v, qu, t))

        # Reconstruct path
        path: list[int] = []
        while qu.prev:
            qu = qu.prev
            path.append(qu.node.id)

        return path

    def _dijkstra_neighbours(self, start: int, dest: int = -1) -> list[float] | list[int]:
        """
        Run Dijkstra's algorithm from start node.

        Args:
            start: Starting node index
            dest: Optional destination node (if specified, returns path instead of distances)

        Returns:
            Either array of distances to all nodes, or path to dest if dest specified
        """
        q: PriorityQueue[Node] = PriorityQueue(lambda a, b: a.d <= b.d)
        d: list[float] = [0.0] * self.n

        # Initialize all nodes
        for i, node in enumerate(self.neighbours):
            node.d = 0.0 if i == start else float("inf")
            node.q = q.push(node)

        while not q.empty():
            u = q.pop()
            assert u is not None  # Queue not empty
            d[u.id] = u.d

            # If we reached destination, reconstruct path
            if u.id == dest:
                path: list[int] = []
                v = u
                while v.prev is not None:
                    path.append(v.prev.id)
                    v = v.prev
                return path

            # Relax edges
            for neighbour in u.neighbours:
                v = self.neighbours[neighbour.id]
                t = u.d + neighbour.distance

                if u.d != float("inf") and v.d > t:
                    v.d = t
                    v.prev = u
                    assert v.q is not None  # Node was added to queue
                    q.reduce_key(v.q, v, lambda e, heap_q: setattr(e, "q", heap_q))

        return d
