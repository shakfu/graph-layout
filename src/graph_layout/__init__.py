"""
graph-layout: A collection of graph layout algorithms in Python.

This package provides various graph layout algorithms for positioning nodes
in network visualizations.

Available algorithms:
- cola: Constraint-based graph layout (port of WebCola)
- force: Force-directed layouts (Fruchterman-Reingold, Spring, Kamada-Kawai)
- hierarchical: Tree and DAG layouts (Reingold-Tilford, Radial, Sugiyama)
- circular: Circular and shell layouts
- spectral: Spectral/eigenvector-based layouts
"""

__version__ = "0.1.6"

# Shared types for all algorithms
# Base classes for building layouts
from .base import (
    BaseLayout,
    IterativeLayout,
    StaticLayout,
)

# Circular layouts
from .circular import (
    CircularLayout,
    ShellLayout,
)

# Cola layout (constraint-based)
from .cola import ColaLayoutAdapter, Layout, Layout3D

# Force-directed layouts
from .force import (
    FruchtermanReingoldLayout,
    KamadaKawaiLayout,
    SpringLayout,
)

# Hierarchical layouts
from .hierarchical import (
    RadialTreeLayout,
    ReingoldTilfordLayout,
    SugiyamaLayout,
)

# Metrics for layout quality evaluation
from .metrics import (
    angular_resolution,
    edge_crossings,
    edge_length_uniformity,
    edge_length_variance,
    layout_quality_summary,
    stress,
)

# Preprocessing utilities
from .preprocessing import (
    assign_layers_longest_path,
    connected_components,
    count_crossings,
    detect_cycle,
    has_cycle,
    is_connected,
    minimize_crossings_barycenter,
    remove_cycles,
    topological_sort,
)

# Spatial data structures
from .spatial import Body, QuadTree, QuadTreeNode

# Spectral layouts
from .spectral import SpectralLayout
from .types import (
    Event,
    EventType,
    Group,
    GroupLike,
    Link,
    LinkAccessor,
    LinkLike,
    Node,
    NodeLike,
    SizeType,
    is_group,
)

# Validation utilities
from .validation import (
    InvalidCanvasSizeError,
    InvalidGroupError,
    InvalidLinkError,
    ValidationError,
    validate_canvas_size,
    validate_group_indices,
    validate_link_indices,
)

__all__ = [
    # Version
    "__version__",
    # Shared types
    "Node",
    "Link",
    "Group",
    "EventType",
    "Event",
    "is_group",
    "LinkAccessor",
    # Type aliases for API
    "NodeLike",
    "LinkLike",
    "GroupLike",
    "SizeType",
    # Base classes
    "BaseLayout",
    "IterativeLayout",
    "StaticLayout",
    # Cola (constraint-based layout)
    "Layout",
    "Layout3D",
    "ColaLayoutAdapter",
    # Force-directed layouts
    "FruchtermanReingoldLayout",
    "SpringLayout",
    "KamadaKawaiLayout",
    # Hierarchical layouts
    "ReingoldTilfordLayout",
    "RadialTreeLayout",
    "SugiyamaLayout",
    # Circular layouts
    "CircularLayout",
    "ShellLayout",
    # Spectral layouts
    "SpectralLayout",
    # Metrics
    "edge_crossings",
    "stress",
    "edge_length_variance",
    "edge_length_uniformity",
    "angular_resolution",
    "layout_quality_summary",
    # Spatial data structures
    "Body",
    "QuadTree",
    "QuadTreeNode",
    # Validation
    "ValidationError",
    "InvalidCanvasSizeError",
    "InvalidLinkError",
    "InvalidGroupError",
    "validate_canvas_size",
    "validate_link_indices",
    "validate_group_indices",
    # Preprocessing
    "detect_cycle",
    "has_cycle",
    "remove_cycles",
    "topological_sort",
    "connected_components",
    "is_connected",
    "assign_layers_longest_path",
    "minimize_crossings_barycenter",
    "count_crossings",
]
