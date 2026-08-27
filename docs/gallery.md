# Gallery

Every figure on this page is rendered during the documentation build by executing
the code shown beneath it. Nothing here is a stored image.

## Force-directed

Nodes repel, edges pull. Good general-purpose layouts for graphs without an
inherent hierarchy. Pass `random_seed` to make a run reproducible.

```graph-layout title="Fruchterman-Reingold on a 12-cycle with chords" source=below node_radius=13
links = [{"source": i, "target": (i + 1) % 12} for i in range(12)]
links += [{"source": i, "target": (i + 5) % 12} for i in range(0, 12, 3)]

layout = FruchtermanReingoldLayout(
    nodes=[{"index": i} for i in range(12)],
    links=links,
    size=(360, 300),
    random_seed=7,
).run()
```

```graph-layout title="Kamada-Kawai, stress-minimising" source=below node_radius=13
layout = KamadaKawaiLayout(
    nodes=[{"index": i} for i in range(9)],
    links=[
        {"source": i, "target": j}
        for i in range(9)
        for j in range(i + 1, 9)
        if (j - i) in (1, 3)
    ],
    size=(340, 300),
    random_seed=7,
).run()
```

## Constraint-based

Cola adds separation constraints and overlap avoidance on top of stress
majorization. It refines an initial placement, so seed the nodes with `x` and `y`.

```graph-layout title="Cola with overlap avoidance" source=below node_radius=14
import math

n = 12
layout = ColaLayoutAdapter(
    nodes=[
        {
            "index": i,
            "x": 150 + 100 * math.cos(2 * math.pi * i / n),
            "y": 150 + 100 * math.sin(2 * math.pi * i / n),
            "width": 30,
            "height": 30,
        }
        for i in range(n)
    ],
    links=[{"source": i, "target": (i + 1) % n} for i in range(n)]
    + [{"source": i, "target": (i + 4) % n} for i in range(0, n, 4)],
    size=(340, 320),
    link_distance=55,
    avoid_overlaps=True,
).run()
```

## Hierarchical

Layer-based and tree-structured placements for directed or rooted graphs.

```graph-layout title="Sugiyama layered drawing of a DAG" source=below node_radius=13
layout = SugiyamaLayout(
    nodes=[{"index": i} for i in range(9)],
    links=[
        {"source": 0, "target": 1}, {"source": 0, "target": 2},
        {"source": 1, "target": 3}, {"source": 1, "target": 4},
        {"source": 2, "target": 5}, {"source": 3, "target": 6},
        {"source": 4, "target": 6}, {"source": 5, "target": 7},
        {"source": 6, "target": 8}, {"source": 7, "target": 8},
    ],
    size=(380, 320),
).run()
```

```graph-layout title="Reingold-Tilford tidy tree, complete binary tree" source=below node_radius=13
layout = ReingoldTilfordLayout(
    nodes=[{"index": i} for i in range(10)],
    links=[{"source": (i - 1) // 2, "target": i} for i in range(1, 10)],
    size=(380, 260),
).run()
```

```graph-layout title="Radial tree, ternary tree about the root" source=below node_radius=12
layout = RadialTreeLayout(
    nodes=[{"index": i} for i in range(13)],
    links=[{"source": (i - 1) // 3, "target": i} for i in range(1, 13)],
    size=(340, 340),
).run()
```

## Circular

```graph-layout title="Shell layout, hub plus two rings" source=below node_radius=12
links = [{"source": 0, "target": i} for i in range(1, 5)]
links += [
    {"source": i, "target": j}
    for i in range(1, 5)
    for j in range(5, 11)
    if (i + j) % 3 == 0
]

layout = ShellLayout(
    nodes=[{"index": i} for i in range(11)],
    links=links,
    size=(340, 340),
).run()
```

## Spectral

Positions come from the eigenvectors of the graph Laplacian, so the drawing is
deterministic for a given graph.

```graph-layout title="Spectral layout of a 12-cycle" source=below node_radius=13
layout = SpectralLayout(
    nodes=[{"index": i} for i in range(12)],
    links=[{"source": i, "target": i + 1} for i in range(11)]
    + [{"source": 11, "target": 0}],
    size=(340, 300),
).run()
```

## Bipartite

```graph-layout title="Bipartite layout, two node sets in columns" source=below node_radius=13
layout = BipartiteLayout(
    nodes=[{"index": i} for i in range(9)],
    links=[
        {"source": i, "target": j}
        for i in range(4)
        for j in range(4, 9)
        if (i + j) % 2 == 0
    ],
    size=(320, 300),
).run()
```

## Orthogonal

Edges run horizontally and vertically with explicit bends. These layouts override
`to_svg()` to draw rectangles and polylines, so the fence needs no extra options
beyond the ones the orthogonal renderer accepts -- `node_radius` is not one of them.

```graph-layout title="Kandinsky orthogonal drawing" source=below
layout = KandinskyLayout(
    nodes=[{"index": i} for i in range(6)],
    links=[
        {"source": 0, "target": 1}, {"source": 1, "target": 2},
        {"source": 2, "target": 3}, {"source": 3, "target": 0},
        {"source": 0, "target": 4}, {"source": 4, "target": 5},
        {"source": 5, "target": 2},
    ],
    size=(360, 300),
).run()
```

```graph-layout title="GIOTTO drawing of a cube graph" source=below
edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (0, 4), (1, 5), (2, 6), (3, 7),
    (4, 5), (5, 6), (6, 7), (7, 4),
]

layout = GIOTTOLayout(
    nodes=[{"index": i} for i in range(8)],
    links=[{"source": s, "target": t} for s, t in edges],
    size=(360, 320),
).run()
```

## Planar straight-line

Crossing-free drawings of planar graphs on an integer grid.

```graph-layout title="Schnyder realizer embedding" source=below node_radius=13
edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3), (0, 4), (1, 4)]

layout = SchnyderLayout(
    nodes=[{"index": i} for i in range(5)],
    links=[{"source": s, "target": t} for s, t in edges],
    size=(320, 300),
).run()
```
