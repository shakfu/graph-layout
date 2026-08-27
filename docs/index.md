# graph-layout

A collection of graph layout algorithms in Python: force-directed, constraint-based,
hierarchical, circular, spectral, orthogonal, and planar straight-line drawings, with
SVG, DOT, and GraphML export.

## Install

```bash
pip install graph-layout
```

The ILP-based compaction path needs SciPy:

```bash
pip install "graph-layout[ilp]"
```

## Quick start

Every layout takes `nodes`, `links`, and a canvas `size`, runs with `.run()`, and
exports with `.to_svg()`.

```graph-layout title="Circular layout of a 10-cycle" source node_radius=14
layout = CircularLayout(
    nodes=[{"index": i} for i in range(10)],
    links=[{"source": i, "target": (i + 1) % 10} for i in range(10)],
    size=(320, 320),
).run()
```

The figure above is not a checked-in image. It was produced by running the code
beside it during the documentation build, so it cannot drift from the API. See
[Embedding Visualizations](embedding-visualizations.md) for how to write one.

## Where to go next

| Page | Contents |
|------|----------|
| [Gallery](gallery.md) | One live figure per algorithm family |
| [Algorithms](algorithms-guide.md) | What each algorithm does and when to use it |
| [Preprocessing](preprocessing-guide.md) | Graph cleanup before layout |
| [Embedding Visualizations](embedding-visualizations.md) | Authoring live figures in these docs |

The [README](https://github.com/shakfu/graph-layout#readme) covers the full API,
metrics, and the constraint system.
