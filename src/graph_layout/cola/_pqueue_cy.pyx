# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Cython-optimized Priority Queue implementation using Pairing Heap.

This module provides a specialized, high-performance pairing heap for use
with shortest path calculations. It's optimized for Node objects with float priorities.
"""

from __future__ import annotations
from typing import Optional


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
