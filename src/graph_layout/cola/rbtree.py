"""
Minimal Red-Black Tree adapter using sortedcontainers.SortedList.

This provides just the interface needed by rectangle.py without implementing
a full Red-Black Tree from scratch.
"""

from __future__ import annotations

from typing import Callable, Generic, Optional, TypeVar, cast

from sortedcontainers import SortedList

T = TypeVar("T")


class RBTreeIterator(Generic[T]):
    """Iterator for traversing sorted elements."""

    def __init__(self, tree: "RBTree[T]", index: int):
        self.tree = tree
        self.index = index

    def next(self) -> Optional[T]:
        """Get next element in sorted order."""
        self.index += 1
        if 0 <= self.index < len(self.tree._data):
            return cast(T, self.tree._data[self.index])
        return None

    def prev(self) -> Optional[T]:
        """Get previous element in sorted order."""
        self.index -= 1
        if 0 <= self.index < len(self.tree._data):
            return cast(T, self.tree._data[self.index])
        return None


class RBTree(Generic[T]):
    """
    Minimal Red-Black Tree adapter backed by SortedList.

    Provides just the interface needed for sweep line algorithms in rectangle.py.
    """

    def __init__(self, compare: Optional[Callable[[T, T], float]] = None):
        """
        Initialize tree.

        Args:
            compare: Comparison function returning negative, zero, or positive
        """
        if compare is not None:
            # SortedList expects a key function, so we wrap the comparator
            from functools import cmp_to_key

            # Wrap float comparator to return int for cmp_to_key
            def int_compare(a: T, b: T) -> int:
                result = compare(a, b)
                if result < 0:
                    return -1
                elif result > 0:
                    return 1
                return 0

            self._data: SortedList[T] = SortedList(key=cmp_to_key(int_compare))
        else:
            self._data = SortedList()

    def insert(self, item: T) -> None:
        """Insert an item into the tree."""
        self._data.add(item)

    def remove(self, item: T) -> None:
        """Remove an item from the tree."""
        self._data.remove(item)

    def find_iter(self, item: T) -> RBTreeIterator[T]:
        """
        Find an item and return an iterator positioned at it.

        Args:
            item: Item to find

        Returns:
            Iterator positioned at the item (or where it would be)
        """
        try:
            index = self._data.index(item)
        except ValueError:
            # If not found, use bisect to find insertion point
            index = self._data.bisect_left(item)
            if index > 0:
                index -= 1
        return RBTreeIterator(self, index)

    def iterator(self) -> RBTreeIterator[T]:
        """Get an iterator starting at the beginning."""
        return RBTreeIterator(self, -1)

    @property
    def size(self) -> int:
        """Get number of elements in tree."""
        return len(self._data)

    def is_empty(self) -> bool:
        """Check if tree is empty."""
        return len(self._data) == 0
