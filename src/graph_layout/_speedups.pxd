# cython: language_level=3
"""
Cython header file for _speedups module.

Declares cdef classes for potential cross-module imports.
"""

# =============================================================================
# Priority Queue
# =============================================================================

cdef class PairingHeapNode:
    cdef public object elem
    cdef public double priority
    cdef public list subheaps

    cpdef bint empty(self)
    cpdef PairingHeapNode merge(self, PairingHeapNode heap2)
    cpdef PairingHeapNode remove_min(self)
    cpdef PairingHeapNode merge_pairs(self)
    cpdef PairingHeapNode decrease_key(self, PairingHeapNode subheap, object new_elem, double new_priority)


cdef class FastPriorityQueue:
    cdef public PairingHeapNode root

    cpdef bint empty(self)
    cpdef object top(self)
    cpdef PairingHeapNode push(self, object elem, double priority)
    cpdef object pop(self)
    cpdef void reduce_key(self, PairingHeapNode heap_node, object new_elem, double new_priority)


# =============================================================================
# Shortest Paths
# =============================================================================

cdef class Neighbour:
    cdef public int id
    cdef public double distance


cdef class Node:
    cdef public int id
    cdef public list neighbours
    cdef public double d
    cdef public Node prev
    cdef public PairingHeapNode q


cdef class Calculator:
    cdef public int n
    cdef public list edges
    cdef public list neighbours
    cdef public object get_source_index
    cdef public object get_target_index
    cdef public object get_length

    cpdef list distance_matrix(self)
    cpdef list distances_from_node(self, int start)
    cpdef list path_from_node_to_node(self, int start, int end)
    cdef list _dijkstra_neighbours(self, int start, int dest)


# =============================================================================
# Force-Directed Layout
# =============================================================================

cpdef void compute_repulsive_forces(
    double[:] pos_x,
    double[:] pos_y,
    double[:] disp_x,
    double[:] disp_y,
    double k_sq,
    int n
) noexcept

cpdef void compute_attractive_forces(
    double[:] pos_x,
    double[:] pos_y,
    double[:] disp_x,
    double[:] disp_y,
    int[:] sources,
    int[:] targets,
    double k,
    int m
) noexcept

cpdef void apply_displacements(
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
) noexcept


# =============================================================================
# QuadTree for Barnes-Hut
# =============================================================================

cdef class QuadTreeBody:
    cdef public double x
    cdef public double y
    cdef public double mass
    cdef public int index


cdef class QuadTreeNode:
    cdef public double x, y, half_size
    cdef public double com_x, com_y, total_mass
    cdef public QuadTreeBody body
    cdef public list children

    cdef inline bint is_leaf(self)
    cdef inline bint is_empty(self)
    cdef inline int get_quadrant(self, double px, double py)


cdef class FastQuadTree:
    cdef public QuadTreeNode root
    cdef public double theta
    cdef public int body_count

    cpdef void insert(self, double x, double y, double mass, int index)
    cdef void _insert_into(self, QuadTreeNode node, QuadTreeBody body, int depth)
    cdef void _insert_into_child(self, QuadTreeNode node, QuadTreeBody body, int depth)
    cpdef void compute_mass_distribution(self)
    cdef void _compute_mass(self, QuadTreeNode node)
    cpdef tuple calculate_force(self, double bx, double by, int body_index, double k_sq)
    cdef void _calc_force(self, QuadTreeNode node, double bx, double by, int body_index, double k_sq, double* fx, double* fy)


cpdef void compute_repulsive_forces_barnes_hut(
    double[:] pos_x,
    double[:] pos_y,
    double[:] disp_x,
    double[:] disp_y,
    double k_sq,
    int n,
    double theta=*,
    double padding=*
) noexcept
