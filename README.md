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
| **Circular** | `CircularLayout` | Nodes arranged on a circle |
| | `ShellLayout` | Concentric circles by degree or grouping |
| **Spectral** | `SpectralLayout` | Laplacian eigenvector embedding |

## Installation

```bash
# Standard installation (with Cython extensions for best performance)
pip install graph-layout

# With optional scipy integration (faster shortest paths)
pip install graph-layout[fast]

# Development installation
git clone https://github.com/shakfu/graph-layout.git
cd graph-layout
uv sync
```

## Quick Start

### Force-Directed Layout

```python
from graph_layout.force import FruchtermanReingoldLayout

nodes = [{'x': 0, 'y': 0} for _ in range(6)]
links = [
    {'source': 0, 'target': 1},
    {'source': 1, 'target': 2},
    {'source': 2, 'target': 0},
    {'source': 3, 'target': 4},
    {'source': 4, 'target': 5},
    {'source': 2, 'target': 3},
]

layout = (FruchtermanReingoldLayout()
    .nodes(nodes)
    .links(links)
    .size([800, 600])
    .start())

for i, node in enumerate(layout.nodes()):
    print(f"Node {i}: ({node.x:.1f}, {node.y:.1f})")
```

### Cola (Constraint-Based) Layout

```python
from graph_layout.cola import Layout

nodes = [
    {'x': 0, 'y': 0, 'width': 50, 'height': 30},
    {'x': 100, 'y': 0, 'width': 50, 'height': 30},
    {'x': 200, 'y': 0, 'width': 50, 'height': 30},
]
links = [
    {'source': 0, 'target': 1},
    {'source': 1, 'target': 2},
]

layout = (Layout()
    .nodes(nodes)
    .links(links)
    .avoid_overlaps(True)
    .link_distance(100)
    .start())
```

### Hierarchical Layout (Trees/DAGs)

```python
from graph_layout.hierarchical import SugiyamaLayout

# Tree structure
nodes = [{'x': 0, 'y': 0} for _ in range(7)]
links = [
    {'source': 0, 'target': 1},
    {'source': 0, 'target': 2},
    {'source': 1, 'target': 3},
    {'source': 1, 'target': 4},
    {'source': 2, 'target': 5},
    {'source': 2, 'target': 6},
]

layout = (SugiyamaLayout()
    .nodes(nodes)
    .links(links)
    .layer_separation(80)
    .node_separation(50)
    .start())
```

### Circular Layout

```python
from graph_layout.circular import CircularLayout, ShellLayout

nodes = [{'x': 0, 'y': 0} for _ in range(10)]
links = [{'source': i, 'target': (i + 1) % 10} for i in range(10)]

# Simple circular
layout = CircularLayout().nodes(nodes).links(links).start()

# Concentric shells by degree
layout = ShellLayout().nodes(nodes).links(links).auto_shells(2).start()
```

### Spectral Layout

```python
from graph_layout.spectral import SpectralLayout

layout = (SpectralLayout()
    .nodes(nodes)
    .links(links)
    .normalized(True)
    .start())
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
from graph_layout.cola import Layout, Group
from graph_layout.cola.linklengths import SeparationConstraint

# Overlap avoidance
layout = Layout().nodes(nodes).links(links).avoid_overlaps(True).start()

# Hierarchical groups
groups = [Group(leaves=[0, 1], padding=10), Group(leaves=[2, 3], padding=10)]
layout = Layout().nodes(nodes).links(links).groups(groups).start()

# Separation constraints
constraint = SeparationConstraint(axis='x', left=0, right=1, gap=50)
layout = Layout().nodes(nodes).links(links).constraints([constraint]).start()
```

### Event System (Animation)

```python
from graph_layout.force import FruchtermanReingoldLayout
from graph_layout.types import EventType

layout = FruchtermanReingoldLayout().nodes(nodes).links(links)

def on_tick(event):
    print(f"Alpha: {event['alpha']:.3f}")

layout.on(EventType.tick, on_tick)
layout.start()
```

### 3D Layout

```python
from graph_layout.cola import Layout3D, Node3D, Link3D

nodes = [Node3D(0, 0, 0), Node3D(1, 0, 0), Node3D(0, 1, 0)]
links = [Link3D(0, 1), Link3D(1, 2), Link3D(2, 0)]

layout = Layout3D(nodes, links, ideal_link_length=1.0)
layout.start(iterations=100)
```

## Module Structure

```
graph_layout/
    __init__.py              # Top-level exports
    base.py                  # Base classes (BaseLayout, IterativeLayout, StaticLayout)
    types.py                 # Common types (Node, Link, Group, EventType)
    cola/                    # Constraint-based layout (WebCola port)
        layout.py            # Main 2D layout
        layout3d.py          # 3D layout
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
    circular/                # Circular layouts
        circular.py
        shell.py
    spectral/                # Spectral methods
        spectral.py
```

## Performance

With Cython extensions:
- Small graphs (20 nodes): ~0.02s
- Medium graphs (100 nodes): ~0.05s
- Large graphs (500 nodes): ~1.1s

See `docs/OPTIMIZATION_ANALYSIS.md` for detailed benchmarks.

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
