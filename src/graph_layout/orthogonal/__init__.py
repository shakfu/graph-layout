"""
Orthogonal layout algorithms.

Orthogonal layouts position nodes and route edges using only horizontal
and vertical segments. This style is ideal for diagrams that require
a structured, rectilinear appearance such as:

- UML class diagrams
- Entity-relationship diagrams
- Flowcharts
- Circuit schematics
- Network topology diagrams

Available algorithms:
- KandinskyLayout: Supports arbitrary node degree, good for general use
"""

from .kandinsky import KandinskyLayout
from .types import (
    BendDirection,
    EdgeSegment,
    NodeBox,
    OrthogonalEdge,
    Port,
    RoutingGrid,
    Side,
)

__all__ = [
    # Layouts
    "KandinskyLayout",
    # Types
    "Side",
    "BendDirection",
    "Port",
    "EdgeSegment",
    "OrthogonalEdge",
    "NodeBox",
    "RoutingGrid",
]
