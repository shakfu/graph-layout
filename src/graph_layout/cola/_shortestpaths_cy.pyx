# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Cython-optimized shortest paths calculation using Dijkstra's algorithm.

This module provides high-performance all-pairs shortest path calculation
using Dijkstra's algorithm with an optimized pairing heap priority queue.
"""

from __future__ import annotations
from typing import Callable, TypeVar
from ._pqueue_cy cimport FastPriorityQueue, PairingHeapNode

cimport cython

T = TypeVar("T")


cdef class Neighbour:
    """Represents a neighbor with distance."""
    cdef public int id
    cdef public double distance

    def __init__(self, int id, double distance):
        self.id = id
        self.distance = distance


cdef class Node:
    """Graph node for shortest path calculation."""
    cdef public int id
    cdef public list neighbours
    cdef public double d
    cdef public Node prev
    cdef public PairingHeapNode q

    def __init__(self, int id):
        self.id = id
        self.neighbours = []
        self.d = 0.0
        self.prev = None
        self.q = None


cdef class Calculator:
    """
    Calculator for all-pairs shortest paths or shortest paths from a single node.

    Uses Dijkstra's algorithm with a priority queue for efficiency.
    """
    cdef public int n
    cdef public list edges
    cdef public list neighbours
    cdef public object get_source_index
    cdef public object get_target_index
    cdef public object get_length

    def __init__(
        self,
        int n,
        list edges,
        object get_source_index,
        object get_target_index,
        object get_length,
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
        cdef int u, v, i
        cdef double d
        cdef object edge

        self.n = n
        self.edges = edges
        self.get_source_index = get_source_index
        self.get_target_index = get_target_index
        self.get_length = get_length

        # Build adjacency list
        self.neighbours = [Node(i) for i in range(n)]

        for edge in edges:
            u = get_source_index(edge)
            v = get_target_index(edge)
            d = get_length(edge)
            (<Node>self.neighbours[u]).neighbours.append(Neighbour(v, d))
            (<Node>self.neighbours[v]).neighbours.append(Neighbour(u, d))

    cpdef list distance_matrix(self):
        """
        Compute all-pairs shortest paths.

        Returns:
            Matrix of shortest distances between all pairs of nodes
        """
        cdef list D = []
        cdef int i

        for i in range(self.n):
            D.append(self._dijkstra_neighbours(i, -1))

        return D

    cpdef list distances_from_node(self, int start):
        """
        Get shortest paths from a specified start node.

        Args:
            start: Starting node index

        Returns:
            Array of shortest distances from start to all other nodes
        """
        return self._dijkstra_neighbours(start, -1)

    cpdef list path_from_node_to_node(self, int start, int end):
        """
        Find shortest path from start to end node.

        Args:
            start: Start node index
            end: End node index

        Returns:
            List of node indices in the path (excluding start, including end)
        """
        return self._dijkstra_neighbours(start, end)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef list _dijkstra_neighbours(self, int start, int dest):
        """
        Run Dijkstra's algorithm from start node.

        Args:
            start: Starting node index
            dest: Optional destination node (if specified, returns path instead of distances)

        Returns:
            Either array of distances to all nodes, or path to dest if dest specified
        """
        cdef FastPriorityQueue q = FastPriorityQueue()
        cdef list d = [0.0] * self.n
        cdef Node u, v, node
        cdef Neighbour neighbour
        cdef double t
        cdef list path
        cdef int i

        # Initialize all nodes
        for i in range(self.n):
            node = <Node>self.neighbours[i]
            if i == start:
                node.d = 0.0
            else:
                node.d = float('inf')
            node.q = q.push(node, node.d)

        while not q.empty():
            u = <Node>q.pop()
            d[u.id] = u.d

            # If we reached destination, reconstruct path
            if u.id == dest:
                path = []
                v = u
                while v.prev is not None:
                    path.append(v.prev.id)
                    v = v.prev
                return path

            # Relax edges
            for neighbour in u.neighbours:
                v = <Node>self.neighbours[neighbour.id]
                t = u.d + neighbour.distance

                if u.d != float('inf') and v.d > t:
                    v.d = t
                    v.prev = u
                    q.reduce_key(v.q, v, v.d)

        return d


def create_calculator(
    int n,
    list edges,
    object get_source_index,
    object get_target_index,
    object get_length,
) -> Calculator:
    """
    Factory function to create a Calculator instance.

    Args:
        n: Number of nodes
        edges: List of edges
        get_source_index: Function to get source node index from edge
        get_target_index: Function to get target node index from edge
        get_length: Function to get edge length

    Returns:
        Calculator instance
    """
    return Calculator(n, edges, get_source_index, get_target_index, get_length)
