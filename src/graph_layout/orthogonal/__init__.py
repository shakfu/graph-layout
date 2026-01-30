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

from .compaction import (
    CompactionConstraint,
    CompactionResult,
    CompactionSolver,
    compact_horizontal,
    compact_layout,
    compact_vertical,
)
from .compaction_ilp import (
    ILPCompactionResult,
    compact_layout_ilp,
    is_scipy_available,
)
from .giotto import GIOTTOLayout
from .kandinsky import KandinskyLayout
from .orthogonalization import (
    AngleType,
    Face,
    FlowNetwork,
    OrthogonalRepresentation,
    build_flow_network,
    compute_faces,
    compute_orthogonal_representation,
    flow_to_orthogonal_rep,
    solve_min_cost_flow_simple,
)
from .planarization import (
    CrossingVertex,
    PlanarizedGraph,
    find_edge_crossings,
    is_planar_quick_check,
    planarize_graph,
    segments_intersect,
)
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
    "GIOTTOLayout",
    "KandinskyLayout",
    # Compaction
    "CompactionConstraint",
    "CompactionResult",
    "CompactionSolver",
    "compact_horizontal",
    "compact_vertical",
    "compact_layout",
    # ILP Compaction
    "ILPCompactionResult",
    "compact_layout_ilp",
    "is_scipy_available",
    # Orthogonalization
    "AngleType",
    "Face",
    "FlowNetwork",
    "OrthogonalRepresentation",
    "compute_faces",
    "build_flow_network",
    "solve_min_cost_flow_simple",
    "flow_to_orthogonal_rep",
    "compute_orthogonal_representation",
    # Planarization
    "CrossingVertex",
    "PlanarizedGraph",
    "planarize_graph",
    "find_edge_crossings",
    "segments_intersect",
    "is_planar_quick_check",
    # Types
    "Side",
    "BendDirection",
    "Port",
    "EdgeSegment",
    "OrthogonalEdge",
    "NodeBox",
    "RoutingGrid",
]
