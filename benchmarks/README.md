# Benchmark Graphs

This directory contains randomly generated graphs for benchmarking layout algorithms.

## Graph Formats

Each graph is saved in 4 formats:

| Format | Extension | Description |
|--------|-----------|-------------|
| **JSON** | `.json` | Native format for graph_layout library |
| **Edge List** | `.csv` | Simple `source,target` CSV format |
| **GraphML** | `.graphml` | XML-based standard format |
| **GML** | `.gml` | Graph Modelling Language |

## Available Graphs

| Name | Nodes | Edges | Model | Description |
|------|-------|-------|-------|-------------|
| `small_random` | 100 | ~250 | Erdős-Rényi | Quick tests |
| `small_scalefree` | 100 | ~300 | Barabási-Albert | Quick tests |
| `medium_random` | 500 | ~1,250 | Erdős-Rényi | Typical benchmarks |
| `medium_scalefree` | 500 | ~1,500 | Barabási-Albert | Typical benchmarks |
| `medium_smallworld` | 500 | ~1,500 | Watts-Strogatz | Typical benchmarks |
| `large_random` | 1,000 | ~2,500 | Erdős-Rényi | Performance testing |
| `large_scalefree` | 1,000 | ~3,000 | Barabási-Albert | Performance testing |
| `large_smallworld` | 1,000 | ~3,000 | Watts-Strogatz | Performance testing |
| `xlarge_scalefree` | 5,000 | ~15,000 | Barabási-Albert | Stress testing |
| `xlarge_random` | 5,000 | ~12,500 | Erdős-Rényi | Stress testing |
| `grid_small` | 100 | 180 | Grid | Regular structure |
| `grid_medium` | 900 | 1,740 | Grid | Regular structure |

## Graph Models

- **Erdős-Rényi (G(n,p))**: Each edge exists independently with probability p. Produces random graphs with Poisson degree distribution.

- **Barabási-Albert**: Scale-free network via preferential attachment. New nodes connect preferentially to high-degree nodes, producing power-law degree distribution (hubs).

- **Watts-Strogatz**: Small-world network. Starts with ring lattice, then randomly rewires edges. Has high clustering and short path lengths.

- **Grid**: Regular 2D lattice. Useful for testing algorithms on structured graphs.

## Usage

### Loading in Python

```python
import json
from pathlib import Path

# Load JSON format (native)
with open("benchmarks/graphs/medium_scalefree.json") as f:
    data = json.load(f)
nodes, links = data["nodes"], data["links"]

# Use with layout
from graph_layout import FruchtermanReingoldLayout

layout = FruchtermanReingoldLayout(
    nodes=nodes,
    links=links,
    size=(1000, 1000),
)
layout.run()
```

### Running Benchmarks

```bash
# Benchmark all algorithms on all graphs
uv run python scripts/benchmark_layouts.py

# Benchmark specific graphs
uv run python scripts/benchmark_layouts.py --graphs "medium_*"

# Benchmark specific algorithms
uv run python scripts/benchmark_layouts.py --algorithms "FR,FA2,YH"

# Custom iterations
uv run python scripts/benchmark_layouts.py --iterations 100
```

### Regenerating Graphs

```bash
uv run python scripts/generate_benchmark_graphs.py
```

## Benchmark Results (50 iterations)

### Medium Graphs (500 nodes)

| Algorithm | Time (s) | Notes |
|-----------|----------|-------|
| Random | 0.001 | O(n) baseline |
| Circular | 0.001 | O(n) |
| Yifan Hu | 0.007 | Multilevel + Barnes-Hut |
| ForceAtlas2 | 0.031 | Barnes-Hut |
| Spectral | 0.040 | Eigendecomposition |
| FR | 0.060 | Cython-accelerated |
| FR+Barnes-Hut | 0.082 | O(n log n) |
| Spring | 4.4 | Pure Python O(n²) |
| Kamada-Kawai | 5.5 | Stress minimization |

### Large Graphs (1000 nodes)

| Algorithm | Time (s) | Notes |
|-----------|----------|-------|
| Random | 0.002 | O(n) baseline |
| Circular | 0.002 | O(n) |
| Yifan Hu | 0.013 | Best for large graphs |
| ForceAtlas2 | 0.065 | Good for communities |
| Spectral | 0.11 | Reveals structure |
| FR+Barnes-Hut | 0.19 | General purpose |

**Recommendation**: For large graphs (>500 nodes), use:
- **Yifan Hu** for fastest layout
- **ForceAtlas2** for community visualization
- **FR+Barnes-Hut** for general aesthetics
