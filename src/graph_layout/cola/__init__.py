"""
cola: Constraint-based graph layout algorithm.

Python port of the WebCola JavaScript library. Provides force-directed graph
layout with support for constraints, groups, overlap avoidance, and
hierarchical clustering.

Main classes:
- Layout: 2D constraint-based graph layout
- Layout3D: 3D extension of the layout algorithm
- Node, Link, Group: Graph structure components
- EventType: Layout event types for callbacks
"""

# Import shared types from the parent package
from ..types import Event, EventType, Group, Link, Node, is_group

# Import Cola-specific types and the Layout class
from .adapter import ColaLayoutAdapter
from .batch import gridify, power_graph_grid_layout
from .descent import Descent, Locks
from .geom import LineSegment, Point, TangentVisibilityGraph, convex_hull
from .gridrouter import GridRouter
from .handledisconnected import apply_packing, separate_graphs
from .layout import InputNode, Layout
from .layout3d import Layout3D, Link3D, Node3D
from .linklengths import (
    AlignmentConstraint,
    LinkLengthAccessor,
    SeparationConstraint,
    generate_directed_edge_constraints,
    jaccard_link_lengths,
    symmetric_diff_link_lengths,
)
from .powergraph import Configuration, PowerEdge, get_groups
from .rectangle import GraphNode, Projection, ProjectionGroup, Rectangle
from .shortestpaths import Calculator as ShortestPathCalculator
from .vpsc import Constraint, Solver, Variable

__all__ = [
    # Core layout
    "Layout",
    "Layout3D",
    "ColaLayoutAdapter",
    "Node",
    "Node3D",
    "Link",
    "Link3D",
    "Group",
    "EventType",
    "InputNode",
    "Event",
    "is_group",
    # Optimization
    "Descent",
    "Locks",
    # Constraints
    "Variable",
    "Constraint",
    "Solver",
    # Geometry
    "Rectangle",
    "Projection",
    "ProjectionGroup",
    "GraphNode",
    "Point",
    "LineSegment",
    "TangentVisibilityGraph",
    "convex_hull",
    # Algorithms
    "ShortestPathCalculator",
    "GridRouter",
    # Link lengths
    "LinkLengthAccessor",
    "SeparationConstraint",
    "AlignmentConstraint",
    "symmetric_diff_link_lengths",
    "jaccard_link_lengths",
    "generate_directed_edge_constraints",
    # Power graphs
    "get_groups",
    "PowerEdge",
    "Configuration",
    # Utilities
    "gridify",
    "power_graph_grid_layout",
    "separate_graphs",
    "apply_packing",
]
