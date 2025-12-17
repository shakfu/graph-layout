"""
Spatial data structures for efficient force calculations.

Provides quadtree implementation for Barnes-Hut O(n log n) force approximation.
"""

from .quadtree import Body, QuadTree, QuadTreeNode

__all__ = ["Body", "QuadTree", "QuadTreeNode"]
