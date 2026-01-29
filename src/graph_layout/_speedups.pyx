# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Cython-optimized implementations for graph layout algorithms.

This module provides high-performance implementations of:
- Priority Queue (Pairing Heap) for Dijkstra's algorithm
- Shortest paths calculation using Dijkstra's algorithm
- Force calculations for Fruchterman-Reingold layout
- Force calculations for ForceAtlas2 layout
- QuadTree for Barnes-Hut force approximation

These are used by various layout algorithms for performance-critical operations.
"""

from __future__ import annotations

cimport cython

# =============================================================================
# Priority Queue (Pairing Heap)
# =============================================================================

cdef class PairingHeapNode:
    """
    Pairing Heap node (Cython-optimized).

    Attributes:
        elem: The element stored in this node
        priority: Priority value for comparison
        subheaps: List of child heaps
    """

    def __init__(self, object elem=None, double priority=float('inf')):
        self.elem = elem
        self.priority = priority
        self.subheaps = []

    cpdef bint empty(self):
        """Check if the heap is empty."""
        return self.elem is None

    cpdef PairingHeapNode merge(self, PairingHeapNode heap2):
        """
        Merge two heaps.

        Args:
            heap2: Heap to merge with

        Returns:
            Merged heap root
        """
        if self.empty():
            return heap2
        elif heap2.empty():
            return self
        elif self.priority <= heap2.priority:
            self.subheaps.append(heap2)
            return self
        else:
            heap2.subheaps.append(self)
            return heap2

    cpdef PairingHeapNode remove_min(self):
        """
        Remove and return heap with minimum element removed.

        Returns:
            New heap root
        """
        if self.empty():
            return PairingHeapNode(None, float('inf'))
        else:
            return self.merge_pairs()

    cpdef PairingHeapNode merge_pairs(self):
        """
        Merge all subheaps in pairs (part of remove_min operation).

        Returns:
            Merged heap
        """
        cdef int n = len(self.subheaps)
        cdef PairingHeapNode first_pair, remaining

        if n == 0:
            return PairingHeapNode(None, float('inf'))
        elif n == 1:
            return self.subheaps[0]
        else:
            # Merge pairs from end
            first_pair = self.subheaps.pop().merge(self.subheaps.pop())
            remaining = self.merge_pairs()
            return first_pair.merge(remaining)

    cpdef PairingHeapNode decrease_key(self, PairingHeapNode subheap, object new_elem, double new_priority):
        """
        Decrease the key of an element in a subheap.

        Args:
            subheap: Subheap containing the element
            new_elem: New element value
            new_priority: New (smaller) priority value

        Returns:
            New heap root
        """
        cdef PairingHeapNode new_heap = subheap.remove_min()
        cdef PairingHeapNode pairing_node

        # Reassign subheap values to preserve tree structure
        subheap.elem = new_heap.elem
        subheap.priority = new_heap.priority
        subheap.subheaps = new_heap.subheaps

        # Create new node with decreased value
        pairing_node = PairingHeapNode(new_elem, new_priority)

        return self.merge(pairing_node)


cdef class FastPriorityQueue:
    """
    Min priority queue backed by a pairing heap (Cython-optimized).

    Provides O(1) insertion and find-min, O(log n) amortized delete-min,
    and O(log n) amortized decrease-key operations.
    """

    def __init__(self):
        """Initialize priority queue."""
        self.root = None

    cpdef bint empty(self):
        """
        Check if queue is empty.

        Returns:
            True if no elements in queue
        """
        return self.root is None or self.root.elem is None

    cpdef object top(self):
        """
        Get the top element (min element) without removing it.

        Returns:
            Minimum element, or None if queue is empty
        """
        if self.empty():
            return None
        return self.root.elem

    cpdef PairingHeapNode push(self, object elem, double priority):
        """
        Push element onto the heap.

        Args:
            elem: Element to push
            priority: Priority value (lower is better)

        Returns:
            Heap node for the inserted element
        """
        cdef PairingHeapNode pairing_node = PairingHeapNode(elem, priority)

        if self.empty():
            self.root = pairing_node
        else:
            self.root = self.root.merge(pairing_node)

        return pairing_node

    cpdef object pop(self):
        """
        Remove and return the minimum element.

        Returns:
            Minimum element, or None if queue is empty
        """
        cdef object obj

        if self.empty():
            return None

        obj = self.root.elem
        self.root = self.root.remove_min()
        return obj

    cpdef void reduce_key(self, PairingHeapNode heap_node, object new_elem, double new_priority):
        """
        Reduce the key value of the specified heap node.

        Args:
            heap_node: Heap node containing the element to update
            new_elem: New element value
            new_priority: New (smaller) priority value
        """
        self.root = self.root.decrease_key(heap_node, new_elem, new_priority)


# =============================================================================
# Shortest Paths (Dijkstra's Algorithm)
# =============================================================================

cdef class Neighbour:
    """Represents a neighbor with distance."""

    def __init__(self, int id, double distance):
        self.id = id
        self.distance = distance


cdef class Node:
    """Graph node for shortest path calculation."""

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


# =============================================================================
# Force-Directed Layout Calculations
# =============================================================================

from libc.math cimport sqrt


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void _compute_repulsive_forces(
    double[:] pos_x,
    double[:] pos_y,
    double[:] disp_x,
    double[:] disp_y,
    double k_sq,
    int n
) noexcept:
    """
    Compute repulsive forces between all pairs of nodes (O(n^2)).

    Uses Fruchterman-Reingold repulsive force: f_r = k^2 / d

    Args:
        pos_x: Array of node x positions
        pos_y: Array of node y positions
        disp_x: Array to accumulate x displacements (modified in place)
        disp_y: Array to accumulate y displacements (modified in place)
        k_sq: Square of optimal distance (k^2)
        n: Number of nodes
    """
    cdef int i, j
    cdef double dx, dy, dist_sq, dist, force, fx, fy

    for i in range(n):
        for j in range(i + 1, n):
            dx = pos_x[i] - pos_x[j]
            dy = pos_y[i] - pos_y[j]
            dist_sq = dx * dx + dy * dy

            if dist_sq < 1e-10:
                dist_sq = 1e-10

            dist = sqrt(dist_sq)
            force = k_sq / dist

            fx = (dx / dist) * force
            fy = (dy / dist) * force

            disp_x[i] += fx
            disp_y[i] += fy
            disp_x[j] -= fx
            disp_y[j] -= fy


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void _compute_attractive_forces(
    double[:] pos_x,
    double[:] pos_y,
    double[:] disp_x,
    double[:] disp_y,
    int[:] sources,
    int[:] targets,
    double k,
    int m
) noexcept:
    """
    Compute attractive forces along edges.

    Uses Fruchterman-Reingold attractive force: f_a = d^2 / k

    Args:
        pos_x: Array of node x positions
        pos_y: Array of node y positions
        disp_x: Array to accumulate x displacements (modified in place)
        disp_y: Array to accumulate y displacements (modified in place)
        sources: Array of source node indices for each edge
        targets: Array of target node indices for each edge
        k: Optimal distance
        m: Number of edges
    """
    cdef int e, src, tgt
    cdef double dx, dy, dist_sq, dist, force, fx, fy

    for e in range(m):
        src = sources[e]
        tgt = targets[e]

        dx = pos_x[src] - pos_x[tgt]
        dy = pos_y[src] - pos_y[tgt]
        dist_sq = dx * dx + dy * dy

        if dist_sq < 1e-10:
            continue

        dist = sqrt(dist_sq)
        force = dist_sq / k

        fx = (dx / dist) * force
        fy = (dy / dist) * force

        disp_x[src] -= fx
        disp_y[src] -= fy
        disp_x[tgt] += fx
        disp_y[tgt] += fy


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void _apply_displacements(
    double[:] pos_x,
    double[:] pos_y,
    double[:] disp_x,
    double[:] disp_y,
    unsigned char[:] fixed,
    double temperature,
    double min_x,
    double min_y,
    double max_x,
    double max_y,
    int n
) noexcept:
    """
    Apply displacement vectors to node positions, limited by temperature.

    Args:
        pos_x: Array of node x positions (modified in place)
        pos_y: Array of node y positions (modified in place)
        disp_x: Array of x displacements
        disp_y: Array of y displacements
        fixed: Array of fixed flags (1 = fixed, 0 = free)
        temperature: Maximum displacement magnitude
        min_x, min_y, max_x, max_y: Bounding box
        n: Number of nodes
    """
    cdef int i
    cdef double disp_len, scale

    for i in range(n):
        if fixed[i]:
            continue

        disp_len = sqrt(disp_x[i] * disp_x[i] + disp_y[i] * disp_y[i])

        if disp_len > 0:
            scale = min(disp_len, temperature) / disp_len
            pos_x[i] += disp_x[i] * scale
            pos_y[i] += disp_y[i] * scale

        # Clamp to bounds
        if pos_x[i] < min_x:
            pos_x[i] = min_x
        elif pos_x[i] > max_x:
            pos_x[i] = max_x

        if pos_y[i] < min_y:
            pos_y[i] = min_y
        elif pos_y[i] > max_y:
            pos_y[i] = max_y


# =============================================================================
# QuadTree for Barnes-Hut Approximation
# =============================================================================

cdef class QuadTreeBody:
    """A body with position and mass for quadtree force calculations."""

    def __init__(self, double x, double y, double mass=1.0, int index=-1):
        self.x = x
        self.y = y
        self.mass = mass
        self.index = index


cdef class QuadTreeNode:
    """
    A node in the quadtree for Barnes-Hut approximation.
    """

    def __init__(self, double x, double y, double half_size):
        self.x = x
        self.y = y
        self.half_size = half_size
        self.com_x = 0.0
        self.com_y = 0.0
        self.total_mass = 0.0
        self.body = None
        self.children = None

    cdef inline bint is_leaf(self):
        return self.children is None

    cdef inline bint is_empty(self):
        return self.body is None and self.children is None

    cdef inline int get_quadrant(self, double px, double py):
        """Get quadrant: 0=NW, 1=NE, 2=SW, 3=SE"""
        cdef int east = px >= self.x
        cdef int south = py >= self.y
        return (2 if south else 0) + (1 if east else 0)


cdef class FastQuadTree:
    """
    Barnes-Hut quadtree for O(n log n) force approximation.
    """

    def __init__(self, double min_x, double min_y, double max_x, double max_y, double theta=0.5):
        cdef double cx = (min_x + max_x) / 2
        cdef double cy = (min_y + max_y) / 2
        cdef double hs = max(max_x - min_x, max_y - min_y) / 2

        self.root = QuadTreeNode(cx, cy, hs)
        self.theta = theta
        self.body_count = 0

    cpdef void insert(self, double x, double y, double mass, int index):
        """Insert a body into the quadtree."""
        cdef QuadTreeBody body = QuadTreeBody(x, y, mass, index)
        self._insert_into(self.root, body, 0)
        self.body_count += 1

    cdef void _insert_into(self, QuadTreeNode node, QuadTreeBody body, int depth):
        """Recursively insert body into subtree with depth limit."""
        cdef QuadTreeBody existing
        cdef int quadrant
        cdef double hs, cx, cy

        # Max depth of 50 prevents stack overflow from coincident points
        # At depth 50, cell size is original_size / 2^50, essentially zero
        if depth > 50:
            # At max depth, just update mass distribution in place
            if node.body is not None:
                # Merge with existing body (weighted average position)
                node.body.mass += body.mass
                node.body.x = (node.body.x + body.x) / 2
                node.body.y = (node.body.y + body.y) / 2
            else:
                node.body = body
            return

        if node.is_empty():
            node.body = body
            return

        if node.is_leaf():
            existing = node.body
            node.body = None
            node.children = [None, None, None, None]
            if existing is not None:
                self._insert_into_child(node, existing, depth)

        self._insert_into_child(node, body, depth)

    cdef void _insert_into_child(self, QuadTreeNode node, QuadTreeBody body, int depth):
        """Insert body into appropriate child."""
        cdef int quadrant = node.get_quadrant(body.x, body.y)
        cdef double hs, cx, cy
        cdef QuadTreeNode child

        if node.children[quadrant] is None:
            hs = node.half_size / 2
            cx = node.x + hs * (1 if quadrant & 1 else -1)
            cy = node.y + hs * (1 if quadrant & 2 else -1)
            node.children[quadrant] = QuadTreeNode(cx, cy, hs)

        child = <QuadTreeNode>node.children[quadrant]
        self._insert_into(child, body, depth + 1)

    cpdef void compute_mass_distribution(self):
        """Compute center of mass for all nodes."""
        self._compute_mass(self.root)

    cdef void _compute_mass(self, QuadTreeNode node):
        """Recursively compute mass distribution."""
        cdef double total_mass, weighted_x, weighted_y
        cdef QuadTreeNode child
        cdef int i

        if node is None:
            return

        if node.is_leaf():
            if node.body is not None:
                node.total_mass = node.body.mass
                node.com_x = node.body.x
                node.com_y = node.body.y
            return

        total_mass = 0.0
        weighted_x = 0.0
        weighted_y = 0.0

        for i in range(4):
            child = <QuadTreeNode>node.children[i] if node.children[i] is not None else None
            if child is not None:
                self._compute_mass(child)
                total_mass += child.total_mass
                weighted_x += child.com_x * child.total_mass
                weighted_y += child.com_y * child.total_mass

        node.total_mass = total_mass
        if total_mass > 0:
            node.com_x = weighted_x / total_mass
            node.com_y = weighted_y / total_mass

    cpdef tuple calculate_force(self, double bx, double by, int body_index, double k_sq):
        """
        Calculate repulsive force on a body using Barnes-Hut approximation.

        Args:
            bx, by: Body position
            body_index: Body index (to skip self-interaction)
            k_sq: Repulsion constant (k^2)

        Returns:
            (fx, fy) force tuple
        """
        cdef double fx = 0.0, fy = 0.0
        self._calc_force(self.root, bx, by, body_index, k_sq, &fx, &fy)
        return (fx, fy)

    cdef void _calc_force(
        self,
        QuadTreeNode node,
        double bx,
        double by,
        int body_index,
        double k_sq,
        double* fx,
        double* fy
    ):
        """Recursively calculate force contribution."""
        cdef double dx, dy, dist_sq, dist, force, size
        cdef QuadTreeNode child
        cdef int i

        if node is None or node.is_empty():
            return

        # Skip self-interaction
        if node.is_leaf() and node.body is not None:
            if node.body.index == body_index:
                return

        dx = bx - node.com_x
        dy = by - node.com_y
        dist_sq = dx * dx + dy * dy

        if dist_sq < 1e-10:
            return

        dist = sqrt(dist_sq)
        size = node.half_size * 2

        # Barnes-Hut criterion: size/dist < theta
        if node.is_leaf() or (size / dist) < self.theta:
            # Treat as single mass
            force = k_sq * node.total_mass / dist
            fx[0] += (dx / dist) * force
            fy[0] += (dy / dist) * force
            return

        # Recurse into children
        for i in range(4):
            child = <QuadTreeNode>node.children[i] if node.children[i] is not None else None
            if child is not None:
                self._calc_force(child, bx, by, body_index, k_sq, fx, fy)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void _compute_repulsive_forces_barnes_hut(
    double[:] pos_x,
    double[:] pos_y,
    double[:] disp_x,
    double[:] disp_y,
    double k_sq,
    int n,
    double theta=0.5,
    double padding=10.0
) noexcept:
    """
    Compute repulsive forces using Barnes-Hut O(n log n) approximation.

    Args:
        pos_x: Array of node x positions
        pos_y: Array of node y positions
        disp_x: Array to accumulate x displacements (modified in place)
        disp_y: Array to accumulate y displacements (modified in place)
        k_sq: Square of optimal distance (k^2)
        n: Number of nodes
        theta: Barnes-Hut accuracy parameter (0=exact, 0.5=balanced)
        padding: Padding around bounding box
    """
    cdef double min_x, min_y, max_x, max_y
    cdef double fx, fy
    cdef int i
    cdef FastQuadTree tree

    if n == 0:
        return

    # Find bounds
    min_x = pos_x[0]
    max_x = pos_x[0]
    min_y = pos_y[0]
    max_y = pos_y[0]

    for i in range(1, n):
        if pos_x[i] < min_x:
            min_x = pos_x[i]
        elif pos_x[i] > max_x:
            max_x = pos_x[i]
        if pos_y[i] < min_y:
            min_y = pos_y[i]
        elif pos_y[i] > max_y:
            max_y = pos_y[i]

    min_x -= padding
    min_y -= padding
    max_x += padding
    max_y += padding

    # Build tree
    tree = FastQuadTree(min_x, min_y, max_x, max_y, theta)
    for i in range(n):
        tree.insert(pos_x[i], pos_y[i], 1.0, i)
    tree.compute_mass_distribution()

    # Calculate forces
    for i in range(n):
        fx, fy = tree.calculate_force(pos_x[i], pos_y[i], i, k_sq)
        disp_x[i] += fx
        disp_y[i] += fy


# =============================================================================
# ForceAtlas2 Layout Calculations
# =============================================================================

from libc.math cimport log


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void _compute_fa2_repulsive_forces(
    double[:] pos_x,
    double[:] pos_y,
    double[:] disp_x,
    double[:] disp_y,
    double[:] degrees,
    double scaling,
    int n
) noexcept:
    """
    Compute degree-weighted repulsive forces for ForceAtlas2 (O(n^2)).

    ForceAtlas2 repulsion: scaling * (deg_i + 1) * (deg_j + 1) / distance

    Args:
        pos_x: Array of node x positions
        pos_y: Array of node y positions
        disp_x: Array to accumulate x displacements (modified in place)
        disp_y: Array to accumulate y displacements (modified in place)
        degrees: Array of node degrees
        scaling: Repulsion scaling factor
        n: Number of nodes
    """
    cdef int i, j
    cdef double dx, dy, dist_sq, dist, force, fx, fy
    cdef double deg_i, deg_j, deg_factor

    for i in range(n):
        deg_i = degrees[i] + 1.0
        for j in range(i + 1, n):
            dx = pos_x[i] - pos_x[j]
            dy = pos_y[i] - pos_y[j]
            dist_sq = dx * dx + dy * dy

            if dist_sq < 1e-10:
                dist_sq = 1e-10

            dist = sqrt(dist_sq)
            deg_j = degrees[j] + 1.0
            deg_factor = deg_i * deg_j

            # ForceAtlas2 repulsion: scaling * (deg_i + 1) * (deg_j + 1) / distance
            force = scaling * deg_factor / dist

            fx = (dx / dist) * force
            fy = (dy / dist) * force

            disp_x[i] += fx
            disp_y[i] += fy
            disp_x[j] -= fx
            disp_y[j] -= fy


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void _compute_fa2_repulsive_forces_overlap(
    double[:] pos_x,
    double[:] pos_y,
    double[:] disp_x,
    double[:] disp_y,
    double[:] degrees,
    double[:] sizes,
    double scaling,
    int n
) noexcept:
    """
    Compute degree-weighted repulsive forces with overlap prevention for ForceAtlas2.

    Args:
        pos_x: Array of node x positions
        pos_y: Array of node y positions
        disp_x: Array to accumulate x displacements (modified in place)
        disp_y: Array to accumulate y displacements (modified in place)
        degrees: Array of node degrees
        sizes: Array of node sizes (radii)
        scaling: Repulsion scaling factor
        n: Number of nodes
    """
    cdef int i, j
    cdef double dx, dy, dist_sq, dist, adjusted_dist, force, fx, fy
    cdef double deg_i, deg_j, deg_factor, overlap_dist

    for i in range(n):
        deg_i = degrees[i] + 1.0
        for j in range(i + 1, n):
            dx = pos_x[i] - pos_x[j]
            dy = pos_y[i] - pos_y[j]
            dist_sq = dx * dx + dy * dy

            if dist_sq < 1e-10:
                dist_sq = 1e-10

            dist = sqrt(dist_sq)

            # Subtract node radii from distance
            overlap_dist = sizes[i] + sizes[j]
            adjusted_dist = dist - overlap_dist
            if adjusted_dist < 0.01:
                adjusted_dist = 0.01

            deg_j = degrees[j] + 1.0
            deg_factor = deg_i * deg_j
            force = scaling * deg_factor / adjusted_dist

            fx = (dx / dist) * force
            fy = (dy / dist) * force

            disp_x[i] += fx
            disp_y[i] += fy
            disp_x[j] -= fx
            disp_y[j] -= fy


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void _compute_fa2_attractive_forces(
    double[:] pos_x,
    double[:] pos_y,
    double[:] disp_x,
    double[:] disp_y,
    int[:] sources,
    int[:] targets,
    double[:] weights,
    double edge_weight_influence,
    bint linlog_mode,
    int m
) noexcept:
    """
    Compute attractive forces along edges for ForceAtlas2.

    Linear mode: force = weight * distance
    LinLog mode: force = weight * log(1 + distance)

    Args:
        pos_x: Array of node x positions
        pos_y: Array of node y positions
        disp_x: Array to accumulate x displacements (modified in place)
        disp_y: Array to accumulate y displacements (modified in place)
        sources: Array of source node indices for each edge
        targets: Array of target node indices for each edge
        weights: Array of edge weights
        edge_weight_influence: How much weight affects attraction (0 to 1)
        linlog_mode: Whether to use log attraction
        m: Number of edges
    """
    cdef int e, src, tgt
    cdef double dx, dy, dist, force, fx, fy, weight

    for e in range(m):
        src = sources[e]
        tgt = targets[e]

        dx = pos_x[src] - pos_x[tgt]
        dy = pos_y[src] - pos_y[tgt]
        dist = sqrt(dx * dx + dy * dy)

        if dist < 0.01:
            continue

        # Apply edge weight influence
        weight = weights[e]
        if edge_weight_influence < 1.0:
            weight = weight ** edge_weight_influence

        # ForceAtlas2 attraction
        if linlog_mode:
            force = weight * log(1.0 + dist)
        else:
            force = weight * dist

        fx = (dx / dist) * force
        fy = (dy / dist) * force

        disp_x[src] -= fx
        disp_y[src] -= fy
        disp_x[tgt] += fx
        disp_y[tgt] += fy


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void _compute_fa2_gravity(
    double[:] pos_x,
    double[:] pos_y,
    double[:] disp_x,
    double[:] disp_y,
    double[:] degrees,
    double gravity,
    double cx,
    double cy,
    bint strong_gravity_mode,
    int n
) noexcept:
    """
    Compute gravity forces toward center for ForceAtlas2.

    Gravity is degree-weighted: gravity * (deg + 1)
    Strong gravity mode: constant force regardless of distance
    Normal mode: force scales with distance

    Args:
        pos_x: Array of node x positions
        pos_y: Array of node y positions
        disp_x: Array to accumulate x displacements (modified in place)
        disp_y: Array to accumulate y displacements (modified in place)
        degrees: Array of node degrees
        gravity: Gravity strength
        cx, cy: Center coordinates
        strong_gravity_mode: Whether to use constant gravity
        n: Number of nodes
    """
    cdef int i
    cdef double dx, dy, dist, force, deg_factor

    if gravity <= 0:
        return

    for i in range(n):
        dx = cx - pos_x[i]
        dy = cy - pos_y[i]
        dist = sqrt(dx * dx + dy * dy)

        if dist < 0.01:
            continue

        deg_factor = degrees[i] + 1.0

        if strong_gravity_mode:
            force = gravity * deg_factor
        else:
            force = gravity * deg_factor * dist

        disp_x[i] += (dx / dist) * force
        disp_y[i] += (dy / dist) * force


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple _compute_fa2_swing_traction(
    double[:] disp_x,
    double[:] disp_y,
    double[:] prev_disp_x,
    double[:] prev_disp_y,
    double[:] degrees,
    int n
):
    """
    Compute global swing and traction for adaptive speed.

    Swing measures oscillation (force direction changing).
    Traction measures consistent movement (force direction stable).

    Args:
        disp_x: Array of current x displacements
        disp_y: Array of current y displacements
        prev_disp_x: Array of previous x displacements
        prev_disp_y: Array of previous y displacements
        degrees: Array of node degrees
        n: Number of nodes

    Returns:
        (total_swing, total_traction) tuple
    """
    cdef int i
    cdef double fx, fy, pfx, pfy
    cdef double swing_x, swing_y, swing
    cdef double trac_x, trac_y, traction
    cdef double deg_factor
    cdef double total_swing = 0.0
    cdef double total_traction = 0.0

    for i in range(n):
        fx = disp_x[i]
        fy = disp_y[i]
        pfx = prev_disp_x[i]
        pfy = prev_disp_y[i]

        # Swing: |F(t) - F(t-1)|
        swing_x = fx - pfx
        swing_y = fy - pfy
        swing = sqrt(swing_x * swing_x + swing_y * swing_y)

        # Traction: |F(t) + F(t-1)| / 2
        trac_x = (fx + pfx) / 2.0
        trac_y = (fy + pfy) / 2.0
        traction = sqrt(trac_x * trac_x + trac_y * trac_y)

        # Weight by degree
        deg_factor = degrees[i] + 1.0
        total_swing += deg_factor * swing
        total_traction += deg_factor * traction

    return (total_swing, total_traction)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double _apply_fa2_displacements(
    double[:] pos_x,
    double[:] pos_y,
    double[:] disp_x,
    double[:] disp_y,
    double[:] prev_disp_x,
    double[:] prev_disp_y,
    unsigned char[:] fixed,
    double global_speed,
    double max_displacement,
    int n
) noexcept:
    """
    Apply displacements with adaptive per-node speeds for ForceAtlas2.

    Per-node speed: global_speed / (1 + global_speed * sqrt(swing))

    Args:
        pos_x: Array of node x positions (modified in place)
        pos_y: Array of node y positions (modified in place)
        disp_x: Array of x displacements
        disp_y: Array of y displacements
        prev_disp_x: Array of previous x displacements
        prev_disp_y: Array of previous y displacements
        fixed: Array of fixed flags (1 = fixed, 0 = free)
        global_speed: Global adaptive speed
        max_displacement: Maximum displacement per iteration
        n: Number of nodes

    Returns:
        Total displacement (for convergence checking)
    """
    cdef int i
    cdef double fx, fy, pfx, pfy
    cdef double swing_x, swing_y, node_swing
    cdef double force_mag, node_speed, displacement, dx, dy
    cdef double total_displacement = 0.0

    for i in range(n):
        if fixed[i]:
            continue

        fx = disp_x[i]
        fy = disp_y[i]
        force_mag = sqrt(fx * fx + fy * fy)

        if force_mag < 0.01:
            continue

        # Compute per-node swing
        pfx = prev_disp_x[i]
        pfy = prev_disp_y[i]
        swing_x = fx - pfx
        swing_y = fy - pfy
        node_swing = sqrt(swing_x * swing_x + swing_y * swing_y)

        # Per-node speed: global_speed / (1 + global_speed * sqrt(swing))
        node_speed = global_speed / (1.0 + global_speed * sqrt(node_swing))

        # Limit displacement
        displacement = force_mag * node_speed
        if displacement > max_displacement:
            displacement = max_displacement

        # Apply displacement
        dx = (fx / force_mag) * displacement
        dy = (fy / force_mag) * displacement

        pos_x[i] += dx
        pos_y[i] += dy
        total_displacement += displacement

    return total_displacement


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void _compute_fa2_repulsive_forces_barnes_hut(
    double[:] pos_x,
    double[:] pos_y,
    double[:] disp_x,
    double[:] disp_y,
    double[:] degrees,
    double scaling,
    int n,
    double theta=1.2,
    double padding=10.0
) noexcept:
    """
    Compute degree-weighted repulsive forces using Barnes-Hut O(n log n) approximation.

    Args:
        pos_x: Array of node x positions
        pos_y: Array of node y positions
        disp_x: Array to accumulate x displacements (modified in place)
        disp_y: Array to accumulate y displacements (modified in place)
        degrees: Array of node degrees
        scaling: Repulsion scaling factor
        n: Number of nodes
        theta: Barnes-Hut accuracy parameter (FA2 uses 1.2 by default)
        padding: Padding around bounding box
    """
    cdef double min_x, min_y, max_x, max_y
    cdef double fx, fy, deg_i
    cdef int i
    cdef FastQuadTree tree

    if n == 0:
        return

    # Find bounds
    min_x = pos_x[0]
    max_x = pos_x[0]
    min_y = pos_y[0]
    max_y = pos_y[0]

    for i in range(1, n):
        if pos_x[i] < min_x:
            min_x = pos_x[i]
        elif pos_x[i] > max_x:
            max_x = pos_x[i]
        if pos_y[i] < min_y:
            min_y = pos_y[i]
        elif pos_y[i] > max_y:
            max_y = pos_y[i]

    min_x -= padding
    min_y -= padding
    max_x += padding
    max_y += padding

    # Build tree with degree+1 as mass for degree-weighted repulsion
    tree = FastQuadTree(min_x, min_y, max_x, max_y, theta)
    for i in range(n):
        deg_i = degrees[i] + 1.0
        tree.insert(pos_x[i], pos_y[i], deg_i, i)
    tree.compute_mass_distribution()

    # Calculate forces (scaling factor applied here)
    for i in range(n):
        fx, fy = tree.calculate_force(pos_x[i], pos_y[i], i, scaling)
        disp_x[i] += fx
        disp_y[i] += fy


# =============================================================================
# Orthogonal Layout Functions (Kandinsky)
# =============================================================================

def _segments_intersect(
    double p1x, double p1y,
    double p2x, double p2y,
    double p3x, double p3y,
    double p4x, double p4y,
):
    """
    Check if two line segments intersect and return intersection point.

    Uses parametric line intersection formula.
    Segments are (p1, p2) and (p3, p4).

    Returns:
        Tuple (x, y) of intersection point, or None if no intersection
    """
    cdef double d1x, d1y, d2x, d2y
    cdef double denom, t, u
    cdef double ix, iy
    cdef double eps = 1e-10

    d1x = p2x - p1x
    d1y = p2y - p1y
    d2x = p4x - p3x
    d2y = p4y - p3y

    denom = d1x * d2y - d1y * d2x

    if denom > -eps and denom < eps:
        # Lines are parallel
        return None

    t = ((p3x - p1x) * d2y - (p3y - p1y) * d2x) / denom
    u = ((p3x - p1x) * d1y - (p3y - p1y) * d1x) / denom

    # Check if intersection is within both segments (excluding endpoints)
    if t > eps and t < 1.0 - eps and u > eps and u < 1.0 - eps:
        ix = p1x + t * d1x
        iy = p1y + t * d1y
        return (ix, iy)

    return None


def _find_edge_crossings(
    list positions,
    list edges,
):
    """
    Find all edge crossings in a positioned graph.

    Args:
        positions: List of (x, y) tuples for each vertex
        edges: List of (u, v) edge tuples

    Returns:
        List of (edge_i, edge_j, x, y) tuples for each crossing
    """
    cdef int num_edges = len(edges)
    cdef int i, j
    cdef int u1, v1, u2, v2
    cdef double p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y

    crossings = []

    for i in range(num_edges):
        u1, v1 = edges[i]
        p1x, p1y = positions[u1]
        p2x, p2y = positions[v1]

        for j in range(i + 1, num_edges):
            u2, v2 = edges[j]

            # Skip edges that share a vertex (they can't cross in interior)
            if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                continue

            p3x, p3y = positions[u2]
            p4x, p4y = positions[v2]

            result = _segments_intersect(p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y)
            if result is not None:
                crossings.append((i, j, result[0], result[1]))

    return crossings
