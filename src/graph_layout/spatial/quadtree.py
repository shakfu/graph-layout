"""
Quadtree implementation for Barnes-Hut force approximation.

The quadtree recursively subdivides 2D space into quadrants,
enabling O(n log n) approximate n-body force calculations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ..types import Node


@dataclass
class Body:
    """A body (node) with position and mass for force calculations."""

    x: float
    y: float
    mass: float = 1.0
    index: int = -1  # Original node index


@dataclass
class QuadTreeNode:
    """
    A node in the quadtree.

    Attributes:
        x, y: Center of this region
        half_size: Half the width/height of this region
        center_of_mass_x/y: Center of mass of bodies in this subtree
        total_mass: Total mass of bodies in this subtree
        body: Single body if this is a leaf (external) node
        children: Four child quadrants [NW, NE, SW, SE] if internal
    """

    x: float
    y: float
    half_size: float

    # Aggregated properties
    center_of_mass_x: float = 0.0
    center_of_mass_y: float = 0.0
    total_mass: float = 0.0

    # Content
    body: Optional[Body] = None
    children: Optional[List[Optional[QuadTreeNode]]] = None

    def is_leaf(self) -> bool:
        """True if this node has no children."""
        return self.children is None

    def is_empty(self) -> bool:
        """True if this node contains no bodies."""
        return self.body is None and self.children is None

    def contains(self, x: float, y: float) -> bool:
        """Check if point (x, y) is within this node's region."""
        return abs(x - self.x) <= self.half_size and abs(y - self.y) <= self.half_size

    def get_quadrant(self, x: float, y: float) -> int:
        """
        Get quadrant index for a point.

        Returns:
            0=NW, 1=NE, 2=SW, 3=SE
        """
        east = x >= self.x
        south = y >= self.y
        return (2 if south else 0) + (1 if east else 0)


class QuadTree:
    """
    Barnes-Hut quadtree for approximate force calculations.

    The Barnes-Hut algorithm uses a quadtree to approximate long-range
    forces. For distant clusters, the algorithm treats the cluster as
    a single body at its center of mass, reducing complexity from
    O(n^2) to O(n log n).

    Usage:
        tree = QuadTree(bounds=(0, 0, 1000, 1000))
        for node in nodes:
            tree.insert(Body(node.x, node.y, mass=1.0, index=node.index))
        tree.compute_mass_distribution()

        # Compute force on a body
        fx, fy = tree.calculate_force(body, repulsion_constant=10000)

    The theta parameter controls the accuracy/speed tradeoff:
    - theta = 0: Exact calculation (no approximation)
    - theta = 0.5: Good balance (recommended)
    - theta = 1.0+: Fast but less accurate
    """

    def __init__(
        self,
        bounds: Tuple[float, float, float, float],
        theta: float = 0.5,
    ):
        """
        Initialize quadtree.

        Args:
            bounds: (min_x, min_y, max_x, max_y) bounding box
            theta: Barnes-Hut threshold (0 = exact, higher = more approximation)
        """
        min_x, min_y, max_x, max_y = bounds
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        # Use max dimension to ensure square region
        half_size = max(max_x - min_x, max_y - min_y) / 2

        self.root = QuadTreeNode(center_x, center_y, half_size)
        self.theta = theta
        self.body_count = 0

    def insert(self, body: Body) -> None:
        """Insert a body into the quadtree."""
        self._insert_into(self.root, body)
        self.body_count += 1

    def _insert_into(self, node: QuadTreeNode, body: Body) -> None:
        """Recursively insert body into subtree rooted at node."""
        if node.is_empty():
            # Empty node becomes a leaf with this body
            node.body = body
            return

        if node.is_leaf():
            # Leaf with existing body - must subdivide
            existing = node.body
            node.body = None
            node.children = [None, None, None, None]

            # Re-insert existing body
            if existing is not None:
                self._insert_into_child(node, existing)

        # Insert new body into appropriate child
        self._insert_into_child(node, body)

    def _insert_into_child(self, node: QuadTreeNode, body: Body) -> None:
        """Insert body into the appropriate child of node."""
        quadrant = node.get_quadrant(body.x, body.y)

        if node.children is not None and node.children[quadrant] is None:
            # Create child node
            hs = node.half_size / 2
            cx = node.x + hs * (1 if quadrant & 1 else -1)
            cy = node.y + hs * (1 if quadrant & 2 else -1)
            node.children[quadrant] = QuadTreeNode(cx, cy, hs)

        if node.children is not None:
            child = node.children[quadrant]
            if child is not None:
                self._insert_into(child, body)

    def compute_mass_distribution(self) -> None:
        """Compute center of mass for all nodes (post-order traversal)."""
        self._compute_mass(self.root)

    def _compute_mass(self, node: QuadTreeNode) -> None:
        """Recursively compute mass distribution."""
        if node is None:
            return

        if node.is_leaf():
            if node.body:
                node.total_mass = node.body.mass
                node.center_of_mass_x = node.body.x
                node.center_of_mass_y = node.body.y
            return

        total_mass = 0.0
        weighted_x = 0.0
        weighted_y = 0.0

        if node.children:
            for child in node.children:
                if child is not None:
                    self._compute_mass(child)
                    total_mass += child.total_mass
                    weighted_x += child.center_of_mass_x * child.total_mass
                    weighted_y += child.center_of_mass_y * child.total_mass

        node.total_mass = total_mass
        if total_mass > 0:
            node.center_of_mass_x = weighted_x / total_mass
            node.center_of_mass_y = weighted_y / total_mass

    def calculate_force(
        self,
        body: Body,
        repulsion_constant: float = 1.0,
    ) -> Tuple[float, float]:
        """
        Calculate approximate repulsive force on a body.

        Uses Barnes-Hut approximation: if a cluster is sufficiently
        far away (size/distance < theta), treat it as a single mass.

        Args:
            body: The body to calculate force on
            repulsion_constant: k^2 in FR formula (F = k^2/d)

        Returns:
            (fx, fy) force vector (repulsive, pointing away from other bodies)
        """
        return self._calculate_force(self.root, body, repulsion_constant)

    def _calculate_force(
        self,
        node: QuadTreeNode,
        body: Body,
        k_sq: float,
    ) -> Tuple[float, float]:
        """Recursively calculate force contribution from node."""
        if node is None or node.is_empty():
            return 0.0, 0.0

        # Skip self-interaction (same body)
        if node.is_leaf() and node.body is not None:
            if node.body.index == body.index:
                return 0.0, 0.0

        dx = body.x - node.center_of_mass_x
        dy = body.y - node.center_of_mass_y
        dist_sq = dx * dx + dy * dy

        if dist_sq < 1e-10:
            # Skip very close or coincident bodies
            return 0.0, 0.0

        dist = math.sqrt(dist_sq)

        # Barnes-Hut criterion: s/d < theta
        # s = node size (2 * half_size), d = distance
        if node.is_leaf() or (node.half_size * 2 / dist) < self.theta:
            # Treat as single mass - calculate repulsive force
            # Repulsive force: F = k^2 * m / d (FR formula with mass)
            force = k_sq * node.total_mass / dist

            # Force direction: away from center of mass (positive dx/dy direction)
            fx = (dx / dist) * force
            fy = (dy / dist) * force
            return fx, fy

        # Node is too close - recurse into children
        fx, fy = 0.0, 0.0
        if node.children:
            for child in node.children:
                if child is not None:
                    cfx, cfy = self._calculate_force(child, body, k_sq)
                    fx += cfx
                    fy += cfy

        return fx, fy

    @classmethod
    def from_nodes(
        cls,
        nodes: Sequence[Node],
        padding: float = 10.0,
        theta: float = 0.5,
    ) -> QuadTree:
        """
        Build quadtree from a list of Node objects.

        Args:
            nodes: List of Node objects with x, y attributes
            padding: Padding around bounding box
            theta: Barnes-Hut threshold

        Returns:
            QuadTree with all nodes inserted and mass computed
        """
        if not nodes:
            return cls((0, 0, 100, 100), theta=theta)

        min_x = min(n.x for n in nodes) - padding
        min_y = min(n.y for n in nodes) - padding
        max_x = max(n.x for n in nodes) + padding
        max_y = max(n.y for n in nodes) + padding

        tree = cls((min_x, min_y, max_x, max_y), theta=theta)

        for i, node in enumerate(nodes):
            idx = node.index if node.index is not None else i
            tree.insert(Body(node.x, node.y, mass=1.0, index=idx))

        tree.compute_mass_distribution()
        return tree


__all__ = ["Body", "QuadTree", "QuadTreeNode"]
