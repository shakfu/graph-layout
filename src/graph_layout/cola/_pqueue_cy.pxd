# cython: language_level=3
"""
Cython header file for _pqueue_cy module.

Declares cdef classes for cross-module imports.
"""

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
