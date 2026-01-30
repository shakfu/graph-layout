# graph-layout - Python Graph Layout Library

A collection of graph layout algorithms in Python.

## Layout Algorithms

| Family | Algorithm | Description |
|--------|-----------|-------------|
| **Basic** | `RandomLayout` | Random positions within canvas (baseline/starting point) |
| **Bipartite** | `BipartiteLayout` | Two parallel rows for bipartite graphs |
| **Cola** | `Layout` | Constraint-based layout with overlap avoidance (port of [WebCola](https://github.com/tgdwyer/WebCola)) |
| **Force-Directed** | `ForceAtlas2Layout` | Continuous layout with adaptive speeds (Gephi algorithm) |
| | `FruchtermanReingoldLayout` | Classic force-directed with cooling temperature |
| | `KamadaKawaiLayout` | Stress minimization based on graph-theoretic distances |
| | `SpringLayout` | Simple Hooke's law spring forces |
| | `YifanHuLayout` | Multilevel force-directed for medium-large graphs |
| **Hierarchical** | `SugiyamaLayout` | Layered DAG drawing (Sugiyama method) |
| | `ReingoldTilfordLayout` | Classic tree layout |
| | `RadialTreeLayout` | Radial tree with root at center |
| **Circular** | `CircularLayout` | Nodes arranged on a circle |
| | `ShellLayout` | Concentric circles by degree or grouping |
| **Spectral** | `SpectralLayout` | Laplacian eigenvector embedding |
| **Orthogonal** | `KandinskyLayout` | Edges use only horizontal/vertical segments |
| | `GIOTTOLayout` | Bend-optimal for degree-4 planar graphs |

## Installation

```bash
# Standard installation (includes Cython extensions for best performance)
pip install graph-layout

# With ILP compaction support (for optimal Kandinsky area minimization)
pip install graph-layout[ilp]

# Development installation
git clone https://github.com/shakfu/graph-layout.git
cd graph-layout
uv sync
```

## Quick Start

### Random Layout (Baseline)

Random layout places nodes at random positions. Useful as a baseline for comparing layout quality or as a starting point for iterative algorithms:

```python
from graph_layout import RandomLayout

nodes = [{} for _ in range(10)]
links = [{'source': i, 'target': (i + 1) % 10} for i in range(10)]

layout = RandomLayout(
    nodes=nodes,
    links=links,
    size=(800, 600),
    margin=50,        # Optional padding from edges
    random_seed=42,   # For reproducible layouts
)
layout.run()

for i, node in enumerate(layout.nodes):
    print(f"Node {i}: ({node.x:.1f}, {node.y:.1f})")
```

### Force-Directed Layout

```python
from graph_layout import FruchtermanReingoldLayout

nodes = [{} for _ in range(6)]
links = [
    {'source': 0, 'target': 1},
    {'source': 1, 'target': 2},
    {'source': 2, 'target': 0},
    {'source': 3, 'target': 4},
    {'source': 4, 'target': 5},
    {'source': 2, 'target': 3},
]

layout = FruchtermanReingoldLayout(
    nodes=nodes,
    links=links,
    size=(800, 600),
    iterations=300,
)
layout.run()

for i, node in enumerate(layout.nodes):
    print(f"Node {i}: ({node.x:.1f}, {node.y:.1f})")
```

### ForceAtlas2 Layout

ForceAtlas2 is designed for large network visualization with degree-weighted repulsion and adaptive speeds:

```python
from graph_layout import ForceAtlas2Layout

layout = ForceAtlas2Layout(
    nodes=nodes,
    links=links,
    size=(800, 600),
    scaling=2.0,           # Repulsion strength
    gravity=1.0,           # Pull toward center
    linlog_mode=True,      # Tighter clusters
    strong_gravity_mode=False,  # Distance-based gravity
)
layout.run()
```

### Yifan Hu Multilevel Layout

Yifan Hu is ideal for medium-large graphs (1K-100K nodes) using multilevel coarsening:

```python
from graph_layout import YifanHuLayout

layout = YifanHuLayout(
    nodes=nodes,
    links=links,
    size=(800, 600),
    use_barnes_hut=True,       # O(n log n) approximation
    coarsening_threshold=0.75, # Stop coarsening when ratio > 0.75
    min_coarsest_size=10,      # Minimum nodes in coarsest graph
)
layout.run()
```

### Cola (Constraint-Based) Layout

```python
from graph_layout import ColaLayoutAdapter

nodes = [
    {'x': 0, 'y': 0, 'width': 50, 'height': 30},
    {'x': 100, 'y': 0, 'width': 50, 'height': 30},
    {'x': 200, 'y': 0, 'width': 50, 'height': 30},
]
links = [
    {'source': 0, 'target': 1},
    {'source': 1, 'target': 2},
]

layout = ColaLayoutAdapter(
    nodes=nodes,
    links=links,
    avoid_overlaps=True,
    link_distance=100,
)
layout.run()
```

### Hierarchical Layout (Trees/DAGs)

```python
from graph_layout import SugiyamaLayout

# Tree structure
nodes = [{} for _ in range(7)]
links = [
    {'source': 0, 'target': 1},
    {'source': 0, 'target': 2},
    {'source': 1, 'target': 3},
    {'source': 1, 'target': 4},
    {'source': 2, 'target': 5},
    {'source': 2, 'target': 6},
]

layout = SugiyamaLayout(
    nodes=nodes,
    links=links,
    size=(800, 600),
    layer_separation=80,
    node_separation=50,
)
layout.run()
```

### Circular Layout

```python
from graph_layout import CircularLayout, ShellLayout

nodes = [{} for _ in range(10)]
links = [{'source': i, 'target': (i + 1) % 10} for i in range(10)]

# Simple circular
layout = CircularLayout(nodes=nodes, links=links, size=(800, 600))
layout.run()

# Concentric shells by degree
layout = ShellLayout(nodes=nodes, links=links, size=(800, 600), auto_shells=2)
layout.run()
```

### Spectral Layout

```python
from graph_layout import SpectralLayout

layout = SpectralLayout(
    nodes=nodes,
    links=links,
    size=(800, 600),
    normalized=True,
)
layout.run()
```

### Bipartite Layout

Bipartite layout places nodes in two parallel rows, ideal for user-item networks, author-paper relationships, or any bipartite graph:

```python
from graph_layout import BipartiteLayout

# User-item bipartite graph
nodes = [{} for _ in range(7)]  # 3 users + 4 items
links = [
    {'source': 0, 'target': 3},  # user 0 -> item 3
    {'source': 0, 'target': 4},
    {'source': 1, 'target': 4},
    {'source': 1, 'target': 5},
    {'source': 2, 'target': 5},
    {'source': 2, 'target': 6},
]

layout = BipartiteLayout(
    nodes=nodes,
    links=links,
    size=(800, 600),
    top_set=[0, 1, 2],       # Users on top row
    bottom_set=[3, 4, 5, 6], # Items on bottom row
    minimize_crossings=True, # Reorder to reduce edge crossings
)
layout.run()

# Check if graph is bipartite
print(f"Is bipartite: {layout.is_bipartite}")

# Count edge crossings (O(m log m) using inversion counting)
from graph_layout.bipartite import count_crossings
edges = [(0, 3), (0, 4), (1, 4), (1, 5), (2, 5), (2, 6)]  # Same as links above
crossings = count_crossings(layout.top_nodes, layout.bottom_nodes, edges)
print(f"Edge crossings: {crossings}")
```

**Algorithm insight:** Edge crossings in a bipartite drawing equal the number of *inversions* when edges are sorted by their top-layer position. This allows O(m log m) counting via merge sort instead of O(m²) pairwise comparison—a 188x speedup for 10,000 edges.

### Orthogonal Layout (Kandinsky)

Kandinsky layout produces diagrams where all edges use only horizontal and vertical segments. Ideal for UML diagrams, flowcharts, and ER diagrams. Uses a TSM (Topology-Shape-Metrics) approach:

```python
from graph_layout import KandinskyLayout

layout = KandinskyLayout(
    nodes=nodes,
    links=links,
    size=(800, 600),
    node_width=60,
    node_height=40,
    node_separation=60,
    handle_crossings=True,   # Insert crossing vertices for non-planar graphs
    optimize_bends=True,     # Minimize bends using min-cost flow
    compact=True,            # Compact layout to reduce area
    compaction_method="auto", # "auto", "greedy", or "ilp" (ILP requires scipy)
)
layout.run()

# Access edge routing information
for edge in layout.orthogonal_edges:
    print(f"Edge {edge.source}->{edge.target}: {len(edge.bends)} bends")

# Check crossing information
print(f"Edge crossings detected: {layout.num_crossings}")
print(f"Total bends: {layout.total_bends}")
```

#### Port Constraints

Specify which side of a node edges should exit/enter from:

```python
from graph_layout import KandinskyLayout
from graph_layout.orthogonal import Side

# Links with explicit port constraints
links = [
    {"source": 0, "target": 1, "source_side": Side.EAST, "target_side": Side.WEST},
    {"source": 1, "target": 2, "source_side": "south", "target_side": "north"},  # Strings work too
    {"source": 2, "target": 3},  # No constraint - uses heuristic
]

layout = KandinskyLayout(nodes=nodes, links=links, size=(800, 600))
layout.run()

# Verify constraints were applied
edge = layout.orthogonal_edges[0]
print(f"Edge exits from: {edge.source_port.side}")  # Side.EAST
```

### GIOTTO Layout (Degree-4 Planar)

GIOTTO produces bend-optimal orthogonal drawings for planar graphs where every node has at most 4 edges (degree <= 4). Based on Tamassia's algorithm:

```python
from graph_layout import GIOTTOLayout

# 3x3 grid graph (degree-4 planar)
nodes = [{} for _ in range(9)]
links = [
    # Horizontal edges
    {"source": 0, "target": 1}, {"source": 1, "target": 2},
    {"source": 3, "target": 4}, {"source": 4, "target": 5},
    {"source": 6, "target": 7}, {"source": 7, "target": 8},
    # Vertical edges
    {"source": 0, "target": 3}, {"source": 1, "target": 4}, {"source": 2, "target": 5},
    {"source": 3, "target": 6}, {"source": 4, "target": 7}, {"source": 5, "target": 8},
]

layout = GIOTTOLayout(
    nodes=nodes,
    links=links,
    size=(800, 600),
    strict=True,  # Raise error if graph doesn't meet requirements
)
layout.run()

print(f"Valid input: {layout.is_valid_input}")
print(f"Total bends: {layout.total_bends}")
```

Use `strict=False` to fall back to Kandinsky-like behavior for graphs that don't meet GIOTTO's requirements:

```python
# Graph with degree > 4 - would raise error with strict=True
layout = GIOTTOLayout(nodes=nodes, links=links, strict=False)
layout.run()  # Falls back to Kandinsky-like algorithm
```

## Visualization

Generate visualization images for all algorithms:

```bash
uv run python scripts/visualize.py
```

This creates images in `./build/` showing each algorithm's output.

## Algorithm Comparison

| Algorithm | Best For | Complexity | Features |
|-----------|----------|------------|----------|
| **Random** | Baselines, starting points | O(n) | Uniform distribution, reproducible |
| **Bipartite** | User-item, author-paper networks | O(n + m) | Auto-detection, crossing minimization |
| **Cola** | Constrained layouts, overlap avoidance | O(n^2) per iteration | Constraints, groups, 3D |
| **ForceAtlas2** | Large networks, community detection | O(n log n) with Barnes-Hut | Adaptive speed, degree-weighted |
| **Fruchterman-Reingold** | General graphs, aesthetics | O(n^2) per iteration | Temperature cooling |
| **Kamada-Kawai** | Small-medium graphs, stress minimization | O(n^2) per iteration | Graph-theoretic distances |
| **Spring** | Simple layouts, baselines | O(n^2) per iteration | Hooke's law |
| **Yifan Hu** | Medium-large graphs (1K-100K nodes) | O(n log n) with Barnes-Hut | Multilevel coarsening, adaptive step |
| **Sugiyama** | DAGs, hierarchies | O(n^2) | Layer-based, crossing minimization |
| **Reingold-Tilford** | Trees | O(n) | Compact, balanced |
| **Circular** | Ring structures, cycles | O(n) | Simple, predictable |
| **Shell** | Grouped/stratified data | O(n) | Degree-based grouping |
| **Spectral** | Clustering visualization | O(n^3) eigendecomp | Reveals structure |
| **Kandinsky** | UML, flowcharts, ER diagrams | O(m²) | Orthogonal edges, bend minimization, compaction, port constraints |
| **GIOTTO** | Degree-4 planar graphs | O(m²) | Bend-optimal orthogonal, validates planarity |

## Advanced Features

### Cola: Overlap Avoidance & Constraints

```python
from graph_layout import ColaLayoutAdapter
from graph_layout.cola.linklengths import SeparationConstraint

# Overlap avoidance
layout = ColaLayoutAdapter(
    nodes=nodes,
    links=links,
    avoid_overlaps=True,
)
layout.run()

# Hierarchical groups
groups = [{'leaves': [0, 1], 'padding': 10}, {'leaves': [2, 3], 'padding': 10}]
layout = ColaLayoutAdapter(
    nodes=nodes,
    links=links,
    groups=groups,
)
layout.run()

# Separation constraints
constraint = SeparationConstraint(axis='x', left=0, right=1, gap=50)
layout = ColaLayoutAdapter(
    nodes=nodes,
    links=links,
    constraints=[constraint],
)
layout.run()
```

### Event System (Animation)

```python
from graph_layout import FruchtermanReingoldLayout
from graph_layout.types import EventType

def on_tick(event):
    print(f"Alpha: {event['alpha']:.3f}")

layout = FruchtermanReingoldLayout(
    nodes=nodes,
    links=links,
    size=(800, 600),
    on_tick=on_tick,
)
layout.run()

# Or register events after construction
layout = FruchtermanReingoldLayout(nodes=nodes, links=links)
layout.on(EventType.tick, on_tick)
layout.run()
```

### 3D Layout

```python
from graph_layout.cola import Layout3D, Node3D, Link3D

nodes = [Node3D(0, 0, 0), Node3D(1, 0, 0), Node3D(0, 1, 0)]
links = [Link3D(0, 1), Link3D(1, 2), Link3D(2, 0)]

layout = Layout3D(nodes, links, ideal_link_length=1.0)
layout.start(iterations=100)
```

### Export Formats

All layout classes support exporting to SVG, DOT (Graphviz), and GraphML formats via methods:

```python
from graph_layout import CircularLayout

# Create and run a layout
layout = CircularLayout(
    nodes=[{"index": i} for i in range(5)],
    links=[{"source": i, "target": (i + 1) % 5} for i in range(5)],
    size=(400, 400),
).run()

# Export to SVG (web/print)
svg = layout.to_svg(node_color="#4a90d9", show_labels=True)
with open("graph.svg", "w") as f:
    f.write(svg)

# Export to DOT (Graphviz)
dot = layout.to_dot(directed=False, include_positions=True)
with open("graph.dot", "w") as f:
    f.write(dot)

# Export to GraphML (interchange format)
graphml = layout.to_graphml(include_positions=True)
with open("graph.graphml", "w") as f:
    f.write(graphml)
```

Orthogonal layouts (KandinskyLayout, GIOTTOLayout) automatically use orthogonal-specific export with rectangular nodes and bend points:

```python
from graph_layout import KandinskyLayout

layout = KandinskyLayout(nodes=nodes, links=links, size=(800, 600)).run()

# SVG with orthogonal edges (polylines with bends)
svg = layout.to_svg()  # Automatically uses orthogonal rendering

# GraphML with bend point data
graphml = layout.to_graphml()  # Includes bend coordinates and port sides

# DOT with splines=ortho
dot = layout.to_dot()  # Uses box nodes and ortho splines
```

Standalone functions are also available:

```python
from graph_layout import to_svg, to_dot, to_graphml

svg = to_svg(layout, node_color="#ff0000")
dot = to_dot(layout, directed=True)
graphml = to_graphml(layout)
```

### Configuration via Properties

All layout algorithms support configuration via constructor parameters and properties:

```python
from graph_layout import FruchtermanReingoldLayout

# Configure via constructor
layout = FruchtermanReingoldLayout(
    nodes=nodes,
    links=links,
    size=(800, 600),
    iterations=300,
    temperature=100.0,
    cooling_factor=0.95,
)

# Or modify properties after construction
layout = FruchtermanReingoldLayout()
layout.nodes = nodes
layout.links = links
layout.size = (800, 600)
layout.temperature = 50.0
layout.run()

# Access results via properties
for node in layout.nodes:
    print(f"({node.x}, {node.y})")
```

## Module Structure

```sh
graph_layout/
    __init__.py              # Top-level exports
    base.py                  # Base classes (BaseLayout, IterativeLayout, StaticLayout)
    types.py                 # Common types (Node, Link, Group, EventType)
    basic/                   # Basic utility layouts
        random.py            # RandomLayout
    bipartite/               # Bipartite layouts
        bipartite.py         # BipartiteLayout
    cola/                    # Constraint-based layout (WebCola port)
        layout.py            # Main 2D layout
        layout3d.py          # 3D layout
        adapter.py           # ColaLayoutAdapter (Pythonic API)
        descent.py           # Gradient descent optimizer
        vpsc.py              # VPSC constraint solver
        ...
    force/                   # Force-directed layouts
        force_atlas2.py
        fruchterman_reingold.py
        kamada_kawai.py
        spring.py
        yifan_hu.py
    hierarchical/            # Tree and DAG layouts
        sugiyama.py
        reingold_tilford.py
        radial_tree.py
    circular/                # Circular layouts
        circular.py
        shell.py
    spectral/                # Spectral methods
        spectral.py
    orthogonal/              # Orthogonal layouts
        kandinsky.py         # Kandinsky layout (arbitrary degree)
        giotto.py            # GIOTTO layout (degree-4 planar, bend-optimal)
        types.py             # NodeBox, Port, OrthogonalEdge, Side
        planarization.py     # Edge crossing detection and vertex insertion
        orthogonalization.py # Bend minimization via min-cost flow
        compaction.py        # Greedy layout area minimization
        compaction_ilp.py    # ILP-based optimal area minimization
    export/                  # Export to various formats
        svg.py               # to_svg, to_svg_orthogonal
        dot.py               # to_dot, to_dot_orthogonal (Graphviz)
        graphml.py           # to_graphml, to_graphml_orthogonal
```

## Performance

### Cython Speedups

This project includes a Cython `_speedups.pyx` module which provides significant speedups over pure Python:

| Algorithm | Cython Speedup | Notes |
|-----------|----------------|-------|
| **Fruchterman-Reingold** | **50-60x faster** | O(n²) force calculations |
| **ForceAtlas2** | **15-20x faster** | Degree-weighted forces |
| **Yifan Hu** | **5-7x faster** | Multilevel overhead in Python |
| Shortest paths (Dijkstra) | **5x faster** | Priority queue operations |

### Benchmark Results

Benchmarks on random scale-free graphs (Barabási-Albert model), 50 iterations:

| Algorithm | 500 nodes | 1,000 nodes | 5,000 nodes |
|-----------|-----------|-------------|-------------|
| **Random** | 0.001s | 0.002s | 0.015s |
| **Circular** | 0.001s | 0.002s | 0.015s |
| **Yifan Hu** | 0.007s | 0.014s | **0.077s** |
| **ForceAtlas2** | 0.031s | 0.066s | 0.402s |
| **FR + Barnes-Hut** | 0.082s | 0.188s | 1.277s |
| **Spectral** | 0.036s | 0.102s | 6.428s |
| **Fruchterman-Reingold** | 0.059s | -- | -- |
| **Kamada-Kawai** | 5.5s | -- | -- |
| **Kandinsky** | 0.78s | 3.6s | -- |

*Note: FR and KK use O(n²) and are too slow for graphs >500 nodes without Barnes-Hut. Kandinsky uses O(m²) for edge crossing detection.*

### Algorithmic Optimizations

Beyond Cython speedups, several algorithms use asymptotically better approaches:

| Function | Naive | Optimized | Technique |
|----------|-------|-----------|-----------|
| `count_crossings()` | O(m²) | O(m log m) | Merge sort inversion counting |
| Force repulsion | O(n²) | O(n log n) | Barnes-Hut quadtree |
| Yifan Hu layout | O(n²) | O(n log n) | Multilevel coarsening + Barnes-Hut |

**Recommendations by graph size:**
- **< 500 nodes**: Any algorithm works well
- **500-2,000 nodes**: Use Yifan Hu, ForceAtlas2, or FR+Barnes-Hut
- **> 2,000 nodes**: Use Yifan Hu (fastest) or ForceAtlas2 (best for communities)

### Barnes-Hut Approximation

ForceAtlas2 and Yifan Hu use Barnes-Hut O(n log n) approximation by default for graphs >50 nodes. For Fruchterman-Reingold, enable it manually:

```python
layout = FruchtermanReingoldLayout(
    nodes=nodes,
    links=links,
    use_barnes_hut=True,
    barnes_hut_theta=0.5,  # 0=exact, higher=faster but less accurate
)
```

### Running Benchmarks

```bash
# Generate benchmark graphs
uv run python scripts/generate_benchmark_graphs.py

# Run benchmarks
uv run python scripts/benchmark_layouts.py --graphs "large_*"
```

## Development

```bash
make test          # Run tests
make typecheck     # Type checking
make lint          # Lint code
make qa            # Run all qualtiy checks
```

## Credits

- **Cola**: Port of [WebCola](https://github.com/tgdwyer/WebCola) by Tim Dwyer (see also [libcola-releated papers](https://www.adaptagrams.org/documentation/libcola.html) in the [adaptagrams project](https://www.adaptagrams.org).
- **ForceAtlas2**: Based on "ForceAtlas2, a Continuous Graph Layout Algorithm for Handy Network Visualization" by Jacomy et al. (2014)
- **Fruchterman-Reingold**: Based on "Graph Drawing by Force-directed Placement" (1991)
- **Kamada-Kawai**: Based on "An Algorithm for Drawing General Undirected Graphs" (1989)
- **Yifan Hu**: Based on "Efficient and High Quality Force-Directed Graph Drawing" (2005)
- **Sugiyama**: Based on "Methods for Visual Understanding of Hierarchical System Structures" (1981)
- **Reingold-Tilford**: Based on "Tidier Drawings of Trees" (1981)
- **Kandinsky**: Based on the Kandinsky model and Tamassia's bend minimization algorithm (1987)
- **GIOTTO**: Based on Tamassia's "On Embedding a Graph in the Grid with the Minimum Number of Bends" (1987)

## License

MIT
