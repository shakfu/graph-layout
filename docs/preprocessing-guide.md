# Graph Preprocessing Guide

This guide explains how to use the preprocessing utilities in `graph-layout` to prepare graphs before layout.

## Overview

Graph preprocessing utilities help you:
- **Detect and remove cycles** - Required for hierarchical layouts (Sugiyama)
- **Find connected components** - Handle disconnected graphs
- **Compute topological order** - Understand graph structure
- **Assign layers** - Prepare for layered/hierarchical layouts
- **Minimize crossings** - Improve layered layout aesthetics

All functions work with the standard `graph-layout` link format (`{'source': int, 'target': int}`) and support custom accessors for other formats.

---

## Cycle Detection and Removal

### Why Remove Cycles?

Hierarchical layouts like `SugiyamaLayout` are designed for **directed acyclic graphs (DAGs)**. If your graph has cycles, you have two options:

1. **Let the algorithm warn you** - Sugiyama will still run but results may be suboptimal
2. **Preprocess to remove cycles** - Reverse some edges to make the graph acyclic

### Detecting Cycles

```python
from graph_layout import detect_cycle, has_cycle

nodes = [{} for _ in range(4)]
links = [
    {'source': 0, 'target': 1},
    {'source': 1, 'target': 2},
    {'source': 2, 'target': 3},
    {'source': 3, 'target': 1},  # Back edge creating cycle
]

# Quick boolean check
if has_cycle(len(nodes), links):
    print("Graph contains a cycle")

# Get the actual cycle
cycle = detect_cycle(len(nodes), links)
if cycle:
    print(f"Cycle found: {cycle}")  # e.g., [1, 2, 3, 1]
```

### Removing Cycles

The `remove_cycles()` function finds back edges using DFS and reverses them:

```python
from graph_layout import remove_cycles, has_cycle, SugiyamaLayout

# Original graph with cycle
links = [
    {'source': 0, 'target': 1},
    {'source': 1, 'target': 2},
    {'source': 2, 'target': 0},  # Creates cycle 0 -> 1 -> 2 -> 0
]

# Remove cycles
acyclic_links, reversed_indices = remove_cycles(3, links)

print(f"Reversed {len(reversed_indices)} edge(s): {reversed_indices}")
print(f"Is now acyclic: {not has_cycle(3, acyclic_links)}")

# Now safe for hierarchical layout
layout = SugiyamaLayout(
    nodes=[{} for _ in range(3)],
    links=acyclic_links,
    size=(800, 600),
)
layout.run()
```

**Note:** The reversed edges are tracked so you can render them differently (e.g., dashed lines) in your visualization.

---

## Topological Sorting

Topological sort orders nodes so that for every edge `u -> v`, node `u` comes before `v`. This is useful for:
- Understanding dependency order
- Processing nodes in correct sequence
- Verifying graph is acyclic

```python
from graph_layout import topological_sort

# Dependency graph: A -> B -> D, A -> C -> D
links = [
    {'source': 0, 'target': 1},  # A -> B
    {'source': 0, 'target': 2},  # A -> C
    {'source': 1, 'target': 3},  # B -> D
    {'source': 2, 'target': 3},  # C -> D
]

order = topological_sort(4, links)
if order is not None:
    print(f"Topological order: {order}")  # [0, 1, 2, 3] or [0, 2, 1, 3]
else:
    print("Graph has cycles - no topological order exists")
```

---

## Connected Components

### Finding Components

Disconnected graphs (multiple separate subgraphs) can cause issues with some layouts. Use `connected_components()` to identify them:

```python
from graph_layout import connected_components, is_connected

links = [
    {'source': 0, 'target': 1},  # Component 1
    {'source': 1, 'target': 2},
    {'source': 3, 'target': 4},  # Component 2
]

# Quick check
if not is_connected(5, links):
    print("Graph is disconnected")

# Get all components
components = connected_components(5, links)
print(f"Found {len(components)} components:")
for i, comp in enumerate(components):
    print(f"  Component {i}: nodes {comp}")
```

### Handling Disconnected Graphs

**Option 1: Layout each component separately**

```python
from graph_layout import connected_components, FruchtermanReingoldLayout

def layout_components(nodes, links, size):
    """Layout each component and arrange them side by side."""
    n = len(nodes)
    components = connected_components(n, links)

    # Create node index mapping for each component
    results = []
    x_offset = 0

    for comp in components:
        # Extract subgraph
        node_map = {old: new for new, old in enumerate(comp)}
        comp_nodes = [nodes[i] for i in comp]
        comp_links = [
            {'source': node_map[l['source']], 'target': node_map[l['target']]}
            for l in links
            if l['source'] in node_map and l['target'] in node_map
        ]

        # Layout component
        layout = FruchtermanReingoldLayout(
            nodes=comp_nodes,
            links=comp_links,
            size=(size[0] / len(components), size[1]),
        )
        layout.run()

        # Offset positions
        for node in layout.nodes:
            node.x += x_offset

        results.extend(zip(comp, layout.nodes))
        x_offset += size[0] / len(components)

    return results
```

**Option 2: Use Cola with `handle_disconnected=True`**

```python
from graph_layout import ColaLayoutAdapter

layout = ColaLayoutAdapter(
    nodes=nodes,
    links=links,
    handle_disconnected=True,  # Automatically handles disconnected components
)
layout.run()
```

---

## Layer Assignment

Layer assignment places nodes into horizontal layers for hierarchical visualization. This is the first step in the Sugiyama algorithm.

```python
from graph_layout import assign_layers_longest_path

# Tree structure
links = [
    {'source': 0, 'target': 1},
    {'source': 0, 'target': 2},
    {'source': 1, 'target': 3},
    {'source': 1, 'target': 4},
    {'source': 2, 'target': 5},
]

layers = assign_layers_longest_path(6, links)

for i, layer in enumerate(layers):
    print(f"Layer {i}: {layer}")

# Output:
# Layer 0: [0]        (root)
# Layer 1: [1, 2]     (children of root)
# Layer 2: [3, 4, 5]  (grandchildren)
```

### Using Layers for Custom Layout

```python
from graph_layout import assign_layers_longest_path, remove_cycles

def simple_hierarchical_layout(nodes, links, size, layer_gap=100, node_gap=50):
    """Simple hierarchical layout using preprocessing utilities."""
    n = len(nodes)

    # Make acyclic if needed
    acyclic_links, _ = remove_cycles(n, links)

    # Assign layers
    layers = assign_layers_longest_path(n, acyclic_links)

    # Position nodes
    width, height = size
    for layer_idx, layer in enumerate(layers):
        y = 50 + layer_idx * layer_gap
        layer_width = (len(layer) - 1) * node_gap
        start_x = (width - layer_width) / 2

        for pos, node_idx in enumerate(layer):
            nodes[node_idx]['x'] = start_x + pos * node_gap
            nodes[node_idx]['y'] = y

    return nodes, layers
```

---

## Crossing Minimization

Edge crossings make graphs harder to read. The barycenter heuristic reorders nodes within layers to reduce crossings.

```python
from graph_layout import (
    assign_layers_longest_path,
    minimize_crossings_barycenter,
    count_crossings,
)

links = [
    {'source': 0, 'target': 2},
    {'source': 0, 'target': 3},
    {'source': 1, 'target': 2},
    {'source': 1, 'target': 4},
]

# Get initial layers
layers = assign_layers_longest_path(5, links)
print(f"Initial crossings: {count_crossings(layers, links)}")

# Minimize crossings
optimized_layers = minimize_crossings_barycenter(layers, links, iterations=24)
print(f"After optimization: {count_crossings(optimized_layers, links)}")
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `layers` | List of layers from `assign_layers_longest_path()` | Required |
| `links` | Edge list | Required |
| `iterations` | Number of sweep iterations | 24 |

More iterations generally produce better results but take longer. 24 iterations is usually sufficient.

---

## Complete Preprocessing Pipeline

Here's a complete example combining all preprocessing steps:

```python
from graph_layout import (
    has_cycle,
    remove_cycles,
    is_connected,
    connected_components,
    assign_layers_longest_path,
    minimize_crossings_barycenter,
    count_crossings,
    SugiyamaLayout,
)

def preprocess_for_hierarchical(nodes, links, verbose=True):
    """
    Full preprocessing pipeline for hierarchical layout.

    Returns:
        Tuple of (processed_links, layers, reversed_edges, components)
    """
    n = len(nodes)

    # Step 1: Check connectivity
    if not is_connected(n, links):
        components = connected_components(n, links)
        if verbose:
            print(f"Warning: Graph has {len(components)} disconnected components")
    else:
        components = [list(range(n))]

    # Step 2: Remove cycles
    if has_cycle(n, links):
        processed_links, reversed_edges = remove_cycles(n, links)
        if verbose:
            print(f"Removed {len(reversed_edges)} cycle-causing edge(s)")
    else:
        processed_links = links
        reversed_edges = set()

    # Step 3: Assign layers
    layers = assign_layers_longest_path(n, processed_links)
    if verbose:
        print(f"Assigned nodes to {len(layers)} layers")

    # Step 4: Minimize crossings
    initial_crossings = count_crossings(layers, processed_links)
    layers = minimize_crossings_barycenter(layers, processed_links)
    final_crossings = count_crossings(layers, processed_links)
    if verbose:
        print(f"Reduced crossings from {initial_crossings} to {final_crossings}")

    return processed_links, layers, reversed_edges, components


# Usage
nodes = [{} for _ in range(6)]
links = [
    {'source': 0, 'target': 1},
    {'source': 1, 'target': 2},
    {'source': 2, 'target': 0},  # Cycle!
    {'source': 0, 'target': 3},
    {'source': 3, 'target': 4},
    {'source': 4, 'target': 5},
]

processed_links, layers, reversed, components = preprocess_for_hierarchical(nodes, links)

# Now use with Sugiyama (or custom layout using the layers)
layout = SugiyamaLayout(
    nodes=nodes,
    links=processed_links,
    size=(800, 600),
)
layout.run()
```

---

## Custom Link Formats

All preprocessing functions accept optional `get_source` and `get_target` callbacks for custom link formats:

```python
from graph_layout import has_cycle, topological_sort

# Using objects instead of dicts
class Edge:
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt

edges = [Edge(0, 1), Edge(1, 2), Edge(2, 0)]

# Provide custom accessors
has_cycles = has_cycle(
    3,
    edges,
    get_source=lambda e: e.src,
    get_target=lambda e: e.tgt,
)

# Works with tuples too
tuple_edges = [(0, 1), (1, 2)]
order = topological_sort(
    3,
    tuple_edges,
    get_source=lambda e: e[0],
    get_target=lambda e: e[1],
)
```

---

## API Reference

| Function | Description | Returns |
|----------|-------------|---------|
| `detect_cycle(n, links)` | Find a cycle in the graph | `list[int]` or `None` |
| `has_cycle(n, links)` | Check if graph has any cycle | `bool` |
| `remove_cycles(n, links)` | Make graph acyclic by reversing edges | `(list, set)` |
| `topological_sort(n, links)` | Get topological ordering | `list[int]` or `None` |
| `connected_components(n, links)` | Find all connected components | `list[list[int]]` |
| `is_connected(n, links)` | Check if graph is connected | `bool` |
| `assign_layers_longest_path(n, links)` | Assign nodes to layers | `list[list[int]]` |
| `minimize_crossings_barycenter(layers, links)` | Reduce edge crossings | `list[list[int]]` |
| `count_crossings(layers, links)` | Count edge crossings | `int` |
