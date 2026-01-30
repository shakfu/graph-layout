"""
graph-layout: A collection of graph layout algorithms in Python.

This package provides various graph layout algorithms for positioning nodes
in network visualizations.

Available algorithms:
- basic: Simple utility layouts (Random)
- bipartite: Two-row layouts for bipartite graphs
- cola: Constraint-based graph layout (port of WebCola)
- force: Force-directed layouts (Fruchterman-Reingold, Spring, Kamada-Kawai)
- hierarchical: Tree and DAG layouts (Reingold-Tilford, Radial, Sugiyama)
- circular: Circular and shell layouts
- spectral: Spectral/eigenvector-based layouts
- orthogonal: Orthogonal layouts with horizontal/vertical edges (Kandinsky)
- export: Export to SVG, DOT (Graphviz), and GraphML formats
"""

__version__ = "0.1.7"

# Shared types for all algorithms
# Base classes for building layouts
from .base import (
    BaseLayout,
    IterativeLayout,
    StaticLayout,
)

# Basic layouts
from .basic import RandomLayout

# Bipartite layouts
from .bipartite import BipartiteLayout

# Circular layouts
from .circular import (
    CircularLayout,
    ShellLayout,
)

# Cola layout (constraint-based)
from .cola import ColaLayoutAdapter, Layout, Layout3D

# Export functions
from .export import (
    to_dot,
    to_dot_orthogonal,
    to_graphml,
    to_graphml_orthogonal,
    to_svg,
    to_svg_orthogonal,
)

# Force-directed layouts
from .force import (
    ForceAtlas2Layout,
    FruchtermanReingoldLayout,
    KamadaKawaiLayout,
    SpringLayout,
    YifanHuLayout,
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

# Orthogonal layouts
from .orthogonal import GIOTTOLayout, KandinskyLayout

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
    # Basic layouts
    "RandomLayout",
    # Bipartite layouts
    "BipartiteLayout",
    # Cola (constraint-based layout)
    "Layout",
    "Layout3D",
    "ColaLayoutAdapter",
    # Force-directed layouts
    "ForceAtlas2Layout",
    "FruchtermanReingoldLayout",
    "SpringLayout",
    "KamadaKawaiLayout",
    "YifanHuLayout",
    # Hierarchical layouts
    "ReingoldTilfordLayout",
    "RadialTreeLayout",
    "SugiyamaLayout",
    # Circular layouts
    "CircularLayout",
    "ShellLayout",
    # Spectral layouts
    "SpectralLayout",
    # Orthogonal layouts
    "GIOTTOLayout",
    "KandinskyLayout",
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
    # Export functions
    "to_svg",
    "to_svg_orthogonal",
    "to_dot",
    "to_dot_orthogonal",
    "to_graphml",
    "to_graphml_orthogonal",
]
