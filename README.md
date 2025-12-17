# graph-layout - Python Graph Layout Library

A collection of graph layout algorithms in Python. Currently includes:

- **Cola** - Constraint-based graph layout (port of [WebCola](https://github.com/tgdwyer/WebCola))

## Features

### Cola Algorithm
- **Force-Directed Layout** - 2D and 3D graph layouts using gradient descent
- **Constraint-Based** - Separation, alignment, and custom constraints
- **Overlap Avoidance** - Non-overlapping node placement
- **Hierarchical Groups** - Nested group layouts with containment
- **Power Graph** - Automatic hierarchical clustering
- **Grid Router** - Orthogonal edge routing
- **Event System** - Animation support with tick events
- **Fluent API** - Method chaining for easy configuration

## Installation

**Standard installation** (with Cython extensions for best performance):
```bash
pip install graph-layout
```

**With optional scipy integration** (for even faster shortest paths):
```bash
pip install graph-layout[fast]
```

**Development installation**:
```bash
# Clone repository
git clone https://github.com/shakfu/graph-layout.git
cd graph-layout

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

**Note**: Pre-built wheels with Cython extensions are available for:
- Linux (x86_64, aarch64)
- macOS (x86_64, arm64)
- Windows (amd64)

If you're on a different platform, the package will automatically fall back to pure Python (slower but functional).

## Quick Start

### Basic Usage

```python
from graph_layout.cola import Layout

# Create your graph
nodes = [
    {'x': 0, 'y': 0},           # node 0
    {'x': 100, 'y': 0},         # node 1
    {'x': 200, 'y': 0},         # node 2
]

edges = [
    {'source': 0, 'target': 1},
    {'source': 1, 'target': 2},
    {'source': 2, 'target': 0},
]

# Create and configure layout
layout = Layout()
layout.nodes(nodes)
layout.links(edges)
layout.start()

# After layout, nodes have computed x, y positions
print(f"Node 0: ({nodes[0]['x']:.2f}, {nodes[0]['y']:.2f})")
print(f"Node 1: ({nodes[1]['x']:.2f}, {nodes[1]['y']:.2f})")
print(f"Node 2: ({nodes[2]['x']:.2f}, {nodes[2]['y']:.2f})")
```

### Alternative Import Paths

```python
# Direct import from top-level (convenience)
from graph_layout import Layout, Layout3D

# Or from cola subpackage (explicit)
from graph_layout.cola import Layout, Layout3D, Node, Link, Group, EventType

# For specific modules
from graph_layout.cola.linklengths import SeparationConstraint
from graph_layout.cola.vpsc import Variable, Constraint, Solver
```

### Fluent API (Method Chaining)

```python
from graph_layout.cola import Layout

nodes = [{'x': 0, 'y': 0} for _ in range(5)]
edges = [
    {'source': 0, 'target': 1},
    {'source': 1, 'target': 2},
    {'source': 2, 'target': 3},
    {'source': 3, 'target': 4},
]

# Configure everything with method chaining
layout = (Layout()
    .size([800, 600])              # canvas size
    .nodes(nodes)
    .links(edges)
    .link_distance(100)            # desired edge length
    .convergence_threshold(0.01)   # when to stop
    .start(50)                     # run 50 iterations
)

# Nodes now have their positions
for i, node in enumerate(nodes):
    print(f"Node {i}: ({node['x']:.1f}, {node['y']:.1f})")
```

## Advanced Features

### Overlap Avoidance

Prevent nodes from overlapping by specifying widths and heights:

```python
from graph_layout.cola import Layout

nodes = [
    {'x': 0, 'y': 0, 'width': 50, 'height': 30},
    {'x': 50, 'y': 0, 'width': 50, 'height': 30},
    {'x': 100, 'y': 0, 'width': 50, 'height': 30},
]

edges = [
    {'source': 0, 'target': 1},
    {'source': 1, 'target': 2},
]

layout = (Layout()
    .nodes(nodes)
    .links(edges)
    .avoid_overlaps(True)    # prevent node overlaps
    .start()
)
```

### Event-Driven Layout (Animation)

Listen to layout events for animation:

```python
from graph_layout.cola import Layout, EventType

nodes = [{'x': 0, 'y': 0} for _ in range(10)]
edges = [{'source': i, 'target': i+1} for i in range(9)]

layout = Layout().nodes(nodes).links(edges)

# Listen to layout events
def on_tick(event):
    print(f"Tick {event['alpha']:.3f}, stress: {event['stress']:.2f}")

def on_end(event):
    print("Layout complete!")

layout.on(EventType.tick, on_tick)
layout.on(EventType.end, on_end)

layout.start(100)  # 100 iterations
```

### Hierarchical Groups

Create nested group layouts:

```python
from graph_layout.cola import Layout, Group

# Nodes
nodes = [
    {'x': 0, 'y': 0, 'width': 30, 'height': 30},    # 0
    {'x': 50, 'y': 0, 'width': 30, 'height': 30},   # 1
    {'x': 100, 'y': 0, 'width': 30, 'height': 30},  # 2
    {'x': 150, 'y': 0, 'width': 30, 'height': 30},  # 3
]

# Edges
edges = [
    {'source': 0, 'target': 1},
    {'source': 2, 'target': 3},
    {'source': 1, 'target': 2},
]

# Groups (clusters)
groups = [
    Group(leaves=[nodes[0], nodes[1]], padding=10),  # group 1
    Group(leaves=[nodes[2], nodes[3]], padding=10),  # group 2
]

# Layout with groups
layout = (Layout()
    .size([400, 300])
    .nodes(nodes)
    .links(edges)
    .groups(groups)
    .avoid_overlaps(True)
    .link_distance(80)
    .start()
)

# Access results
for i, node in enumerate(nodes):
    print(f"Node {i}: ({node['x']:.1f}, {node['y']:.1f})")
```

### 3D Layout

```python
from graph_layout.cola import Layout3D, Node3D, Link3D

nodes = [
    Node3D(0, 0, 0),
    Node3D(1, 0, 0),
    Node3D(0, 1, 0),
]

links = [
    Link3D(0, 1),
    Link3D(1, 2),
    Link3D(2, 0),
]

layout = Layout3D(nodes, links, ideal_link_length=1.0)
layout.start(iterations=100)

# Access 3D positions
for i, node in enumerate(nodes):
    print(f"Node {i}: ({node.x:.2f}, {node.y:.2f}, {node.z:.2f})")
```

### Constraints

```python
from graph_layout.cola import Layout
from graph_layout.cola.linklengths import SeparationConstraint, AlignmentConstraint

# Separation constraint
sep = SeparationConstraint(
    axis='x',        # 'x' or 'y'
    left=0,          # left node index
    right=1,         # right node index
    gap=50,          # minimum gap
    equality=False   # if True, gap is exact
)

# Alignment constraint
align = AlignmentConstraint(
    axis='y',
    offsets=[
        {'node': 0, 'offset': 0},
        {'node': 1, 'offset': 0},
        {'node': 2, 'offset': 0},
    ]
)

layout = Layout().nodes(nodes).links(edges).constraints([sep, align]).start()
```

## Development

### Setup

```bash
# Quick setup (recommended)
make sync

# Or using uv directly
uv sync
```

### Commands

```bash
make help          # Show all commands
make test          # Run tests
make test-coverage # Run with coverage
make typecheck     # Type checking with mypy
make format        # Format code
make lint          # Lint code
make verify        # Run all checks
make fix           # Auto-fix issues
make clean         # Clean artifacts
```

### Direct Commands

```bash
uv run pytest                                           # Run tests
uv run pytest --cov=src/graph_layout --cov-report=html  # Coverage
uv run mypy src/graph_layout                            # Type check
uv run ruff format src/graph_layout tests               # Format
uv run ruff check src/graph_layout tests                # Lint
```

## Module Overview

```
graph_layout/
    __init__.py           # Top-level exports (Layout, Layout3D)
    cola/                 # Cola algorithm package
        __init__.py       # Cola exports
        layout.py         # Main 2D force-directed layout
        layout3d.py       # 3D force-directed layout
        descent.py        # Gradient descent optimizer
        vpsc.py           # VPSC constraint solver
        rectangle.py      # Rectangle operations and overlap removal
        geom.py           # Computational geometry utilities
        powergraph.py     # Hierarchical graph clustering
        gridrouter.py     # Orthogonal edge routing
        shortestpaths.py  # Shortest path algorithms
        linklengths.py    # Link length utilities and constraints
        handledisconnected.py  # Disconnected component handling
        batch.py          # Batch layout operations
        pqueue.py         # Priority queue implementation
        rbtree.py         # Red-black tree implementation
```

## Performance

**Current performance** (with Cython extensions):
- Small graphs (20 nodes): ~0.02s
- Medium graphs (100 nodes): ~0.05s
- Large graphs (500 nodes): ~1.1s

**Optimizations**:
- **Cython-compiled shortest paths** - Dijkstra's algorithm compiled to C
- **Vectorized gradient descent** - NumPy broadcasting for derivative computation
- **Priority cascade**: Cython -> scipy (optional) -> pure Python fallback
- Efficient matrix operations for O(n^2) computations

See `docs/PERFORMANCE_COMPARISON.md` and `docs/OPTIMIZATION_ANALYSIS.md` for detailed benchmarks.

## Dependencies

**Runtime:**
- `numpy` - Matrix operations for gradient descent
- `sortedcontainers` - Sorted data structures for sweep algorithms

**Optional:**
- `scipy` - Faster shortest path algorithms (install with `graph-layout[fast]`)

**Development:**
- `pytest` - Testing framework
- `mypy` - Static type checking
- `ruff` - Fast Python linter

## Credits

The Cola algorithm is a Python port of [WebCola](https://github.com/tgdwyer/WebCola) by Tim Dwyer, which is itself a JavaScript port of `libcola` from the [adaptagrams library](https://www.adaptagrams.org).

Original WebCola paper:
> Tim Dwyer, Kim Marriott, and Michael Wybrow. 2009.
> "Dunnart: A constraint-based network diagram authoring tool."
> In Graph Drawing, pages 420-431. Springer.

## License

MIT (same as original WebCola)
