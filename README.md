# graph-layout - Python Graph Layout Library

A collection of graph layout algorithms in Python.

## Layout Algorithms

| Family | Algorithm | Description |
|--------|-----------|-------------|
| **Cola** | `Layout` | Constraint-based layout with overlap avoidance (port of [WebCola](https://github.com/tgdwyer/WebCola)) |
| **Force-Directed** | `FruchtermanReingoldLayout` | Classic force-directed with cooling temperature |
| | `KamadaKawaiLayout` | Stress minimization based on graph-theoretic distances |
| | `SpringLayout` | Simple Hooke's law spring forces |
| **Hierarchical** | `SugiyamaLayout` | Layered DAG drawing (Sugiyama method) |
| | `ReingoldTilfordLayout` | Classic tree layout |
| | `RadialTreeLayout` | Radial tree with root at center |
| **Circular** | `CircularLayout` | Nodes arranged on a circle |
| | `ShellLayout` | Concentric circles by degree or grouping |
| **Spectral** | `SpectralLayout` | Laplacian eigenvector embedding |

## Installation

```bash
# Standard installation (includes Cython extensions for best performance)
pip install graph-layout

# Development installation
git clone https://github.com/shakfu/graph-layout.git
cd graph-layout
uv sync
```

## Quick Start

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

## Visualization

Generate visualization images for all algorithms:

```bash
uv run python scripts/visualize.py
```

This creates images in `./build/` showing each algorithm's output.

## Algorithm Comparison

| Algorithm | Best For | Complexity | Features |
|-----------|----------|------------|----------|
| **Cola** | Constrained layouts, overlap avoidance | O(n^2) per iteration | Constraints, groups, 3D |
| **Fruchterman-Reingold** | General graphs, aesthetics | O(n^2) per iteration | Temperature cooling |
| **Kamada-Kawai** | Small-medium graphs, stress minimization | O(n^2) per iteration | Graph-theoretic distances |
| **Spring** | Simple layouts, baselines | O(n^2) per iteration | Hooke's law |
| **Sugiyama** | DAGs, hierarchies | O(n^2) | Layer-based, crossing minimization |
| **Reingold-Tilford** | Trees | O(n) | Compact, balanced |
| **Circular** | Ring structures, cycles | O(n) | Simple, predictable |
| **Shell** | Grouped/stratified data | O(n) | Degree-based grouping |
| **Spectral** | Clustering visualization | O(n^3) eigendecomp | Reveals structure |

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
    cola/                    # Constraint-based layout (WebCola port)
        layout.py            # Main 2D layout
        layout3d.py          # 3D layout
        adapter.py           # ColaLayoutAdapter (Pythonic API)
        descent.py           # Gradient descent optimizer
        vpsc.py              # VPSC constraint solver
        ...
    force/                   # Force-directed layouts
        fruchterman_reingold.py
        kamada_kawai.py
        spring.py
    hierarchical/            # Tree and DAG layouts
        sugiyama.py
        reingold_tilford.py
        radial_tree.py
    circular/                # Circular layouts
        circular.py
        shell.py
    spectral/                # Spectral methods
        spectral.py
```

## Performance

With Cython extensions enabled:

| Algorithm | Graph Size | Time |
|-----------|-----------|------|
| Fruchterman-Reingold | 20 nodes | 0.002s |
| Fruchterman-Reingold | 100 nodes | 0.010s |
| Fruchterman-Reingold | 500 nodes | 0.046s |
| Cola (constraint-based) | 500 nodes | 1.2s |
| Circular | 100 nodes | 0.001s |

For very large graphs (2000+ nodes), enable Barnes-Hut approximation for O(n log n) performance:

```python
layout = FruchtermanReingoldLayout(
    nodes=nodes,
    links=links,
    use_barnes_hut=True,
    barnes_hut_theta=0.5,  # 0=exact, higher=faster but less accurate
)
```

## Development

```bash
make test          # Run tests
make typecheck     # Type checking
make lint          # Lint code
make verify        # Run all checks
```

## Credits

- **Cola**: Port of [WebCola](https://github.com/tgdwyer/WebCola) by Tim Dwyer
- **Fruchterman-Reingold**: Based on "Graph Drawing by Force-directed Placement" (1991)
- **Kamada-Kawai**: Based on "An Algorithm for Drawing General Undirected Graphs" (1989)
- **Sugiyama**: Based on "Methods for Visual Understanding of Hierarchical System Structures" (1981)
- **Reingold-Tilford**: Based on "Tidier Drawings of Trees" (1981)

## License

MIT
