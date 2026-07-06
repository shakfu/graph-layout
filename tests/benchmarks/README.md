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
with open("tests/benchmarks/graphs/medium_scalefree.json") as f:
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

### Comparing against OGDF

`compare_ogdf.py` benchmarks graph-layout's layouts against the equivalent C++ layouts in [OGDF](https://ogdf.github.io/) (via the `ogdf-py` oracle) on the same graphs. It reports, per layout, wall-clock time, **scale-invariant normalized stress** (each drawing is optimally rescaled before scoring, so quality is compared independent of coordinate scale), and edge crossings on small graphs. Comparable layouts are grouped by family (stress/MDS, force-directed) so the numbers line up directly; OGDF sets the C++ speed/quality ceiling.

```bash
make bench-ogdf                          # default: small + medium graphs
make bench-ogdf ARGS="--all"             # include large / xlarge
make bench-ogdf ARGS="--graphs grid_small small_random"
```

Requires the `ogdf-py` oracle (installed by `uv sync` on supported platforms, or `make oracle-install`); the script exits cleanly with a message if it is absent.

#### Results: graph-layout vs OGDF

Barabasi-Albert scale-free graphs from 100 to 5000 nodes (`small_scalefree` ... `xlarge_scalefree`), a single run per cell on one machine (Apple Silicon), so times are indicative, not averaged. Stress is scale-invariant normalized stress (lower is better), sampled over 200k node pairs above that many pairs. Reproduce with `make bench-ogdf ARGS="--graphs small_scalefree medium_scalefree large_scalefree xlarge_scalefree"`.

**Layout time (seconds)** -- `gl.*` is graph-layout (pure Python/Cython), `ogdf.*` is OGDF (C++):

| Layout | 100 | 500 | 1000 | 5000 |
|--------|----:|----:|-----:|-----:|
| **Stress / MDS** | | | | |
| `gl.SMACOF` | 0.086 | 2.78 | 10.4 | 268.6 |
| `gl.KamadaKawai` | 2.06 | -- (capped) | -- | -- |
| `ogdf.StressMinimization` | 0.006 | 0.139 | 0.573 | 16.2 |
| `ogdf.PivotMDS` | 0.001 | 0.025 | 0.056 | 0.305 |
| `ogdf.SpringEmbedderKK` | 0.002 | 0.058 | 0.428 | 15.4 |
| **Force-directed** | | | | |
| `gl.FruchtermanReingold` | 0.028 | 0.231 | 0.697 | 13.0 |
| `gl.YifanHu` | 0.003 | 0.009 | 0.018 | 0.228 |
| `ogdf.FMMM` | 0.006 | 0.090 | 0.115 | 0.672 |
| `ogdf.GEM` | 0.009 | 0.068 | 0.134 | 0.675 |

**Speed ratio (graph-layout time / OGDF time)** for the same or closest algorithm; `<1` means graph-layout is faster:

| Comparison | 100 | 500 | 1000 | 5000 |
|------------|----:|----:|-----:|-----:|
| `SMACOF` / `StressMinimization` (identical algorithm) | 14x | 20x | 18x | 17x |
| `FruchtermanReingold` / `FMMM` | 5x | 3x | 6x | 19x |
| `YifanHu` / `FMMM` (both multilevel) | **0.5x** | **0.1x** | **0.16x** | **0.34x** |

Reading the numbers:

- **On the identical algorithm** (SMACOF is OGDF's StressMinimization -- both Guttman-transform stress majorization), OGDF's C++ is a stable **~15-20x faster** at essentially matched quality (stress within ~1% at every size). This is the pure-Python tax. At 5000 nodes `gl.SMACOF` takes **4.5 minutes** vs 16 seconds -- pure Python stress majorization is impractical at that scale.

- **`gl.YifanHu` is the standout**: a multilevel Barnes-Hut method that is *faster* than OGDF's flagship `FMMM` at every size (0.23s at 5000 nodes), paying ~20% higher stress for the speed. graph-layout's best layout beats OGDF's on raw throughput.

- **`gl.FruchtermanReingold`** stays within a small factor through 1000 nodes but falls to ~19x behind at 5000 (its repulsion cost grows faster than FMMM's multipole approximation).

- **`gl.KamadaKawai`** is pathologically slow (2s at *100* nodes) and capped at n<=200; use `SMACOF` for KK-style output at scale.

- **`ogdf.PivotMDS`** is the fastest layout overall (0.3s at 5000) but trades quality (highest stress among the stress methods) and emits NaN on disconnected graphs.

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
