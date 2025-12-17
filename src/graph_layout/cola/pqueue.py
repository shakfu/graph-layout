"""
Priority Queue implementation using Pairing Heap.

This module provides a pairing heap-based priority queue with decrease-key support,
which is essential for efficient Dijkstra's algorithm implementation.
"""

from __future__ import annotations

from typing import Callable, Generic, Optional, TypeVar

T = TypeVar("T")


class PairingHeap(Generic[T]):
    """
    Pairing Heap data structure.

    A pairing heap is a type of heap data structure with relatively simple implementation
    and excellent practical amortized performance.
    """

    def __init__(self, elem: Optional[T] = None):
        self.elem = elem
        self.subheaps: list[PairingHeap[T]] = []

    def __str__(self) -> str:
        """String representation for debugging."""
        return self.to_string(lambda x: str(x))

    def to_string(self, selector: Callable[[T], str]) -> str:
        """
        Convert heap to string representation using custom selector.

        Args:
            selector: Function to convert element to string

        Returns:
            String representation of the heap
        """
        parts = []
        for subheap in self.subheaps:
            if subheap.elem is None:
                continue
            parts.append(subheap.to_string(selector))

        subheap_str = f"({','.join(parts)})" if parts else ""
        elem_str = selector(self.elem) if self.elem is not None else ""
        return elem_str + subheap_str

    def for_each(self, f: Callable[[T, "PairingHeap[T]"], None]) -> None:
        """
        Apply function to each element in the heap.

        Args:
            f: Function to apply to each (element, heap_node) pair
        """
        if not self.empty():
            assert self.elem is not None
            f(self.elem, self)
            for subheap in self.subheaps:
                subheap.for_each(f)

    def count(self) -> int:
        """Return the number of elements in the heap."""
        if self.empty():
            return 0
        return 1 + sum(h.count() for h in self.subheaps)

    def min(self) -> Optional[T]:
        """Return the minimum element without removing it."""
        return self.elem

    def empty(self) -> bool:
        """Check if the heap is empty."""
        return self.elem is None

    def contains(self, h: "PairingHeap[T]") -> bool:
        """
        Check if this heap contains the given heap node.

        Args:
            h: Heap node to search for

        Returns:
            True if h is contained in this heap
        """
        if self is h:
            return True
        return any(subheap.contains(h) for subheap in self.subheaps)

    def is_heap(self, less_than: Callable[[T, T], bool]) -> bool:
        """
        Verify heap property holds.

        Args:
            less_than: Comparison function

        Returns:
            True if heap property is satisfied
        """
        if self.elem is None:
            return True
        for h in self.subheaps:
            if h.elem is None:
                continue
            if not less_than(self.elem, h.elem) or not h.is_heap(less_than):
                return False
        return True

    def insert(self, obj: T, less_than: Callable[[T, T], bool]) -> "PairingHeap[T]":
        """
        Insert element into heap.

        Args:
            obj: Element to insert
            less_than: Comparison function

        Returns:
            New heap root
        """
        return self.merge(PairingHeap(obj), less_than)

    def merge(self, heap2: "PairingHeap[T]", less_than: Callable[[T, T], bool]) -> "PairingHeap[T]":
        """
        Merge two heaps.

        Args:
            heap2: Heap to merge with
            less_than: Comparison function

        Returns:
            Merged heap root
        """
        if self.empty():
            return heap2
        elif heap2.empty():
            return self
        else:
            # Both heaps non-empty, so both elem values are not None
            assert self.elem is not None and heap2.elem is not None
            if less_than(self.elem, heap2.elem):
                self.subheaps.append(heap2)
                return self
            else:
                heap2.subheaps.append(self)
                return heap2

    def remove_min(self, less_than: Callable[[T, T], bool]) -> Optional["PairingHeap[T]"]:
        """
        Remove and return heap with minimum element removed.

        Args:
            less_than: Comparison function

        Returns:
            New heap root, or None if heap was empty
        """
        if self.empty():
            return None
        else:
            return self.merge_pairs(less_than)

    def merge_pairs(self, less_than: Callable[[T, T], bool]) -> "PairingHeap[T]":
        """
        Merge all subheaps in pairs (part of remove_min operation).

        Args:
            less_than: Comparison function

        Returns:
            Merged heap
        """
        if len(self.subheaps) == 0:
            return PairingHeap(None)
        elif len(self.subheaps) == 1:
            return self.subheaps[0]
        else:
            # Merge pairs from end
            first_pair = self.subheaps.pop().merge(self.subheaps.pop(), less_than)
            remaining = self.merge_pairs(less_than)
            return first_pair.merge(remaining, less_than)

    def decrease_key(
        self,
        subheap: "PairingHeap[T]",
        new_value: T,
        set_heap_node: Optional[Callable[[T, "PairingHeap[T]"], None]],
        less_than: Callable[[T, T], bool],
    ) -> "PairingHeap[T]":
        """
        Decrease the key of an element in a subheap.

        Args:
            subheap: Subheap containing the element
            new_value: New (smaller) value for the element
            set_heap_node: Optional callback to update element's heap reference
            less_than: Comparison function

        Returns:
            New heap root
        """
        new_heap = subheap.remove_min(less_than)

        # Reassign subheap values to preserve tree structure
        if new_heap is not None:
            subheap.elem = new_heap.elem
            subheap.subheaps = new_heap.subheaps
            if set_heap_node is not None and new_heap.elem is not None:
                assert subheap.elem is not None
                set_heap_node(subheap.elem, subheap)
        else:
            subheap.elem = None
            subheap.subheaps = []

        # Create new node with decreased value
        pairing_node = PairingHeap(new_value)
        if set_heap_node is not None:
            set_heap_node(new_value, pairing_node)

        return self.merge(pairing_node, less_than)


class PriorityQueue(Generic[T]):
    """
    Min priority queue backed by a pairing heap.

    Provides O(1) insertion and find-min, O(log n) amortized delete-min,
    and O(log n) amortized decrease-key operations.
    """

    def __init__(self, less_than: Callable[[T, T], bool]):
        """
        Initialize priority queue.

        Args:
            less_than: Comparison function returning True if first arg < second arg
        """
        self.root: Optional[PairingHeap[T]] = None
        self.less_than = less_than

    def top(self) -> Optional[T]:
        """
        Get the top element (min element) without removing it.

        Returns:
            Minimum element, or None if queue is empty
        """
        if self.empty():
            return None
        assert self.root is not None
        return self.root.elem

    def push(self, *args: T) -> Optional[PairingHeap[T]]:
        """
        Push one or more elements onto the heap.

        Args:
            *args: Elements to push

        Returns:
            Heap node for the last inserted element
        """
        pairing_node = None
        for arg in args:
            pairing_node = PairingHeap(arg)
            if self.empty():
                self.root = pairing_node
            else:
                assert self.root is not None
                self.root = self.root.merge(pairing_node, self.less_than)
        return pairing_node

    def empty(self) -> bool:
        """
        Check if queue is empty.

        Returns:
            True if no elements in queue
        """
        return self.root is None or self.root.elem is None

    def is_heap(self) -> bool:
        """
        Verify heap property (for testing).

        Returns:
            True if queue is in valid state
        """
        if self.root is None:
            return True
        return self.root.is_heap(self.less_than)

    def for_each(self, f: Callable[[T, PairingHeap[T]], None]) -> None:
        """
        Apply function to each element.

        Args:
            f: Function to apply to each (element, heap_node) pair
        """
        if self.root:
            self.root.for_each(f)

    def pop(self) -> Optional[T]:
        """
        Remove and return the minimum element.

        Returns:
            Minimum element, or None if queue is empty
        """
        if self.empty():
            return None
        assert self.root is not None
        obj = self.root.min()
        self.root = self.root.remove_min(self.less_than)
        return obj

    def reduce_key(
        self,
        heap_node: PairingHeap[T],
        new_key: T,
        set_heap_node: Optional[Callable[[T, PairingHeap[T]], None]] = None,
    ) -> None:
        """
        Reduce the key value of the specified heap node.

        Args:
            heap_node: Heap node containing the element to update
            new_key: New (smaller) key value
            set_heap_node: Optional callback to update element's heap reference
        """
        assert self.root is not None
        self.root = self.root.decrease_key(heap_node, new_key, set_heap_node, self.less_than)

    def to_string(self, selector: Callable[[T], str]) -> str:
        """
        Convert queue to string representation.

        Args:
            selector: Function to convert elements to strings

        Returns:
            String representation
        """
        return self.root.to_string(selector) if self.root else ""

    def count(self) -> int:
        """
        Get number of elements in queue.

        Returns:
            Number of elements
        """
        return self.root.count() if self.root else 0
