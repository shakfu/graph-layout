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

__version__ = "0.1.0"

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
from .cola import Layout, Layout3D

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

# Spectral layouts
from .spectral import SpectralLayout
from .types import (
    Event,
    EventType,
    Group,
    Link,
    LinkAccessor,
    Node,
    is_group,
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
    # Base classes
    "BaseLayout",
    "IterativeLayout",
    "StaticLayout",
    # Cola (constraint-based layout)
    "Layout",
    "Layout3D",
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
]
