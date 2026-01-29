# Kandinsky Orthogonal Layout Implementation

## Overview

The Kandinsky model is an orthogonal graph drawing algorithm that handles graphs with vertices of arbitrary degree (unlike GIOTTO which is limited to degree ≤ 4). It uses the **Topology-Shape-Metrics (TSM)** approach with multiple phases.

## Algorithm Background

### Why Kandinsky?

- Works on **any graph** (not just planar, not limited to degree 4)
- Practical for real-world use cases: UML diagrams, ER diagrams, flowcharts
- Used in commercial tools (yFiles, yEd)
- Well-documented with known approximation algorithms

### Key Papers

- Fößmeier & Kaufmann (1995): Original Kandinsky model
- Tamassia (1987): Min-cost flow for orthogonal drawings
- Eiglsperger (2003): 2-approximation algorithm for Kandinsky bend minimization

## Implementation

### File Structure

```
src/graph_layout/orthogonal/
├── __init__.py              # Module exports
├── kandinsky.py             # Main KandinskyLayout class
├── types.py                 # Core data structures (Side, Port, NodeBox, OrthogonalEdge)
├── planarization.py         # Edge crossing detection and vertex insertion
├── orthogonalization.py     # Bend minimization via min-cost flow (Tamassia)
└── compaction.py            # Constraint-based layout compaction
```

### Data Structures (`types.py`)

```python
class Side(Enum):
    """Side of a node box."""
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"

@dataclass
class Port:
    """A connection point on a node side."""
    node: int           # Node index
    side: Side          # Which side of the node
    edge: int           # Edge index using this port
    offset: float = 0.0 # Position along the side

@dataclass
class NodeBox:
    """A node represented as a rectangle."""
    index: int
    x: float            # Center x
    y: float            # Center y
    width: float
    height: float

    @property
    def left(self) -> float: ...
    @property
    def right(self) -> float: ...
    @property
    def top(self) -> float: ...
    @property
    def bottom(self) -> float: ...

    def get_port_position(self, side: Side) -> tuple[float, float]: ...

@dataclass
class OrthogonalEdge:
    """An edge with orthogonal routing."""
    source: int
    target: int
    source_port: Port
    target_port: Port
    bends: list[tuple[float, float]]  # Bend point coordinates
```

### Planarization (`planarization.py`)

Handles non-planar graphs by detecting edge crossings and inserting dummy vertices.

```python
@dataclass
class CrossingVertex:
    """A dummy vertex at an edge crossing."""
    index: int
    x: float
    y: float
    edge1: tuple[int, int]
    edge2: tuple[int, int]

@dataclass
class PlanarizedGraph:
    """Result of planarizing a graph."""
    num_original_nodes: int
    num_total_nodes: int
    edges: list[tuple[int, int]]
    crossings: list[CrossingVertex]
    edge_to_original: dict[int, int]
    original_to_edges: dict[int, list[int]]

def segments_intersect(p1, p2, p3, p4) -> Optional[tuple[float, float]]:
    """Check if two line segments intersect. Uses Cython when available."""

def find_edge_crossings(positions, edges) -> list[tuple[int, int, float, float]]:
    """Find all edge crossings. Uses Cython when available."""

def planarize_graph(num_nodes, edges, positions) -> PlanarizedGraph:
    """Insert crossing vertices at edge intersections."""
```

### Orthogonalization (`orthogonalization.py`)

Implements bend minimization using Tamassia's min-cost flow formulation.

```python
@dataclass
class OrthogonalRepresentation:
    """Orthogonal representation storing angles and bends."""
    vertex_face_angles: dict[tuple[int, int], int]  # (vertex, face) -> angle units
    edge_bends: dict[tuple[int, int], list[int]]    # (u, v) -> bend directions

    @property
    def total_bends(self) -> int: ...

@dataclass
class Face:
    """A face in the planar embedding."""
    index: int
    vertices: list[int]
    edges: list[tuple[int, int]]
    is_outer: bool = False

@dataclass
class FlowNetwork:
    """Min-cost flow network for orthogonalization."""
    num_vertices: int
    faces: list[Face]
    supplies: dict[int, int]           # Node supplies/demands
    arcs: dict[tuple[int, int], tuple[int, int]]  # (cap, cost)
    flow: dict[tuple[int, int], int]   # Solution

def compute_faces(num_nodes, edges, positions) -> list[Face]:
    """Compute faces of planar embedding with angular ordering."""

def build_flow_network(num_nodes, edges, faces) -> FlowNetwork:
    """Build min-cost flow network for orthogonalization."""

def solve_min_cost_flow_simple(network) -> bool:
    """Solve using successive shortest path algorithm."""

def compute_orthogonal_representation(num_nodes, edges, positions) -> OrthogonalRepresentation:
    """Main entry point - compute optimal orthogonal representation."""
```

### Compaction (`compaction.py`)

Reduces layout area while maintaining constraints.

```python
@dataclass
class CompactionConstraint:
    """A separation constraint: left + gap <= right."""
    left: int
    right: int
    gap: float
    is_hard: bool = True

@dataclass
class CompactionResult:
    """Result of compaction."""
    node_positions: list[tuple[float, float]]
    width: float
    height: float
    iterations: int

class CompactionSolver:
    """Iterative relaxation solver for constraints."""
    def solve(self) -> tuple[list[float], int]: ...

def compact_horizontal(boxes, edges, node_separation, edge_separation) -> list[float]:
    """Compact horizontally, returns new x-coordinates."""

def compact_vertical(boxes, edges, layer_separation, edge_separation) -> list[float]:
    """Compact vertically, returns new y-coordinates."""

def compact_layout(boxes, edges, ...) -> CompactionResult:
    """Full two-pass compaction (horizontal then vertical)."""
```

### Main Layout Class (`kandinsky.py`)

```python
class KandinskyLayout(StaticLayout):
    """
    Kandinsky orthogonal layout algorithm.

    Parameters:
        node_width: Width of node boxes (default: 60)
        node_height: Height of node boxes (default: 40)
        node_separation: Minimum gap between nodes (default: 60)
        edge_separation: Minimum gap between parallel edges (default: 15)
        layer_separation: Vertical gap between layers (default: 80)
        handle_crossings: Detect and handle edge crossings (default: True)
        optimize_bends: Use min-cost flow for bend minimization (default: True)
        compact: Apply compaction to reduce area (default: True)

    Properties:
        orthogonal_edges: List of OrthogonalEdge with routing info
        node_boxes: List of NodeBox with position/size info
        crossing_vertices: List of CrossingVertex (dummy nodes)
        num_crossings: Number of edge crossings detected
        orthogonal_rep: The computed OrthogonalRepresentation
        compaction_result: The CompactionResult
        total_bends: Total number of bends across all edges
    """
```

## Algorithm Flow

```
KandinskyLayout._compute():
    1. Layer Assignment
       - Topological ordering using longest path
       - Nodes assigned to layers based on DAG structure

    2. Node Positioning
       - Position nodes on grid based on layers
       - Center layout within canvas

    3. Planarization (if handle_crossings=True)
       - Find all edge crossings using segment intersection
       - Insert CrossingVertex at each crossing point
       - Split edges through crossing vertices

    4. Orthogonalization (if optimize_bends=True)
       - Compute faces of planar embedding
       - Build min-cost flow network
       - Solve for optimal angle/bend assignment
       - Extract OrthogonalRepresentation

    5. Compaction (if compact=True)
       - Horizontal pass: minimize width
       - Vertical pass: minimize height
       - Update node positions

    6. Edge Routing
       - Determine port sides based on relative positions
       - Compute orthogonal routes with bends
       - Create OrthogonalEdge objects
```

## Performance

The implementation includes several optimizations:

### Pure Python Optimizations
- **Cached box bounds**: Pre-compute `top`, `bottom`, `left`, `right` to avoid repeated property access
- **Removed redundant loops**: Eliminated O(n×edges×bends) loop in compaction

### Cython Optimizations (`_speedups.pyx`)
- `_segments_intersect()`: Fast line segment intersection
- `_find_edge_crossings()`: O(m²) crossing detection

### Benchmark Results

| Graph Size | Time |
|------------|------|
| 100 nodes, 224 edges | 0.045s |
| 500 nodes, 1230 edges | 0.78s |
| 1000 nodes, 2495 edges | 3.63s |

**Optimization impact**: 42x speedup from initial implementation (151s → 3.6s for 1000 nodes)

## Test Coverage

65 tests in `tests/test_kandinsky.py` covering:
- Basic functionality (layout runs, positions assigned)
- Configuration properties
- Layer assignment
- Edge routing and bends
- Event system
- NodeBox and Side utilities
- Segment intersection
- Edge crossing detection
- Graph planarization
- Face computation
- Orthogonal representation
- Compaction

## Usage Example

```python
from graph_layout import KandinskyLayout

nodes = [{} for _ in range(5)]
links = [
    {"source": 0, "target": 1},
    {"source": 1, "target": 2},
    {"source": 2, "target": 3},
    {"source": 3, "target": 4},
]

layout = KandinskyLayout(
    nodes=nodes,
    links=links,
    size=(800, 600),
    node_width=60,
    node_height=40,
    handle_crossings=True,
    optimize_bends=True,
    compact=True,
)
layout.run()

# Access results
for edge in layout.orthogonal_edges:
    print(f"Edge {edge.source}->{edge.target}: {len(edge.bends)} bends")

print(f"Total bends: {layout.total_bends}")
print(f"Edge crossings: {layout.num_crossings}")
```

## Future Improvements

Potential enhancements:
1. **ILP-based compaction**: Optimal area minimization (currently uses heuristic)
2. **Port constraints**: Allow user to specify edge exit sides
3. **Edge labels**: Consider label placement in routing
4. **Incremental layout**: Support for dynamic graph updates
5. **More Cython optimization**: Optimize `solve_min_cost_flow_simple()` and `_route_planarized_edges()`

## References

- [Orthogonal Graph Drawing with Constraints](https://publikationen.uni-tuebingen.de/xmlui/bitstream/handle/10900/49366/pdf/diss.pdf) - Comprehensive thesis
- [Implementing an Algorithm for Orthogonal Graph Layout](https://rtsys.informatik.uni-kiel.de/~biblio/downloads/theses/ocl-bt.pdf) - Implementation guide
- [OGDF Library](https://ogdf.uos.de/) - Open source C++ implementation
- Tamassia, R. (1987). "On embedding a graph in the grid with the minimum number of bends"
