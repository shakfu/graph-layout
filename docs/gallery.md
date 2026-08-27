# Gallery

Every figure on this page is produced during the documentation build by executing
the code that accompanies it. Nothing here is a stored image, and captions
reporting crossings, stress, or bend counts are measured at build time rather
than transcribed. See [Embedding Visualizations](embedding-visualizations.md).

## One graph, many algorithms

The Petersen graph -- 10 vertices, 15 edges, non-planar, vertex-transitive -- is a
useful common subject because no drawing of it is crossing-free and the
algorithms disagree visibly about what to do instead. Each block builds it the
same way:

```python
E = [(i, (i + 1) % 5) for i in range(5)]          # outer 5-cycle
E += [(i, i + 5) for i in range(5)]               # spokes
E += [(5 + i, 5 + (i + 2) % 5) for i in range(5)] # inner pentagram
N = [{"index": i} for i in range(10)]
L = [{"source": s, "target": t} for s, t in E]
```

```graph-layout class=inline node_radius=10
E = [(i, (i + 1) % 5) for i in range(5)] + [(i, i + 5) for i in range(5)]
E += [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
N = [{"index": i} for i in range(10)]
L = [{"source": s, "target": t} for s, t in E]

layout = CircularLayout(nodes=N, links=L, size=(300, 280)).run()
caption = f"Circular -- {edge_crossings(layout.nodes, layout.links)} crossings"
```

```graph-layout class=inline node_radius=10
E = [(i, (i + 1) % 5) for i in range(5)] + [(i, i + 5) for i in range(5)]
E += [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
N = [{"index": i} for i in range(10)]
L = [{"source": s, "target": t} for s, t in E]

layout = FruchtermanReingoldLayout(nodes=N, links=L, size=(300, 280), random_seed=5).run()
caption = f"Fruchterman-Reingold -- {edge_crossings(layout.nodes, layout.links)} crossings"
```

```graph-layout class=inline node_radius=10
E = [(i, (i + 1) % 5) for i in range(5)] + [(i, i + 5) for i in range(5)]
E += [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
N = [{"index": i} for i in range(10)]
L = [{"source": s, "target": t} for s, t in E]

layout = SpringLayout(nodes=N, links=L, size=(300, 280), random_seed=5).run()
caption = f"Spring -- {edge_crossings(layout.nodes, layout.links)} crossings"
```

```graph-layout class=inline node_radius=10
E = [(i, (i + 1) % 5) for i in range(5)] + [(i, i + 5) for i in range(5)]
E += [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
N = [{"index": i} for i in range(10)]
L = [{"source": s, "target": t} for s, t in E]

layout = KamadaKawaiLayout(nodes=N, links=L, size=(300, 280), random_seed=5).run()
caption = (
    f"Kamada-Kawai -- {edge_crossings(layout.nodes, layout.links)} crossings, "
    f"stress {stress(layout.nodes, layout.links):.3f}"
)
```

```graph-layout class=inline node_radius=10
E = [(i, (i + 1) % 5) for i in range(5)] + [(i, i + 5) for i in range(5)]
E += [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
N = [{"index": i} for i in range(10)]
L = [{"source": s, "target": t} for s, t in E]

layout = SMACOFLayout(nodes=N, links=L, size=(300, 280), random_seed=5).run()
caption = (
    f"SMACOF -- {edge_crossings(layout.nodes, layout.links)} crossings, "
    f"stress {stress(layout.nodes, layout.links):.3f}"
)
```

```graph-layout class=inline node_radius=10
E = [(i, (i + 1) % 5) for i in range(5)] + [(i, i + 5) for i in range(5)]
E += [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
N = [{"index": i} for i in range(10)]
L = [{"source": s, "target": t} for s, t in E]

layout = YifanHuLayout(nodes=N, links=L, size=(300, 280), random_seed=5).run()
caption = f"Yifan Hu -- {edge_crossings(layout.nodes, layout.links)} crossings"
```

```graph-layout class=inline node_radius=10
E = [(i, (i + 1) % 5) for i in range(5)] + [(i, i + 5) for i in range(5)]
E += [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
N = [{"index": i} for i in range(10)]
L = [{"source": s, "target": t} for s, t in E]

layout = ForceAtlas2Layout(nodes=N, links=L, size=(300, 280), random_seed=5, scaling=30.0).run()
caption = f"ForceAtlas2 -- {edge_crossings(layout.nodes, layout.links)} crossings"
```

```graph-layout class=inline node_radius=10
E = [(i, (i + 1) % 5) for i in range(5)] + [(i, i + 5) for i in range(5)]
E += [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
N = [{"index": i} for i in range(10)]
L = [{"source": s, "target": t} for s, t in E]

layout = SpectralLayout(nodes=N, links=L, size=(300, 280)).run()
caption = f"Spectral -- {edge_crossings(layout.nodes, layout.links)} crossings"
```

```graph-layout class=inline node_radius=10
E = [(i, (i + 1) % 5) for i in range(5)] + [(i, i + 5) for i in range(5)]
E += [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
N = [{"index": i} for i in range(10)]
L = [{"source": s, "target": t} for s, t in E]

layout = RandomLayout(nodes=N, links=L, size=(300, 280), random_seed=5).run()
caption = f"Random (baseline) -- {edge_crossings(layout.nodes, layout.links)} crossings"
```

## The same algorithm on different graph families

Force-directed placement is sensitive to the structure it is given. A grid
recovers its own geometry; a random graph does not have one to recover.

```graph-layout class=inline node_radius=9
edges = []
for r in range(4):
    for c in range(4):
        i = r * 4 + c
        if c < 3:
            edges.append((i, i + 1))
        if r < 3:
            edges.append((i, i + 4))

layout = FruchtermanReingoldLayout(
    nodes=[{"index": i} for i in range(16)],
    links=[{"source": s, "target": t} for s, t in edges],
    size=(320, 300),
    random_seed=11,
).run()
caption = f"4x4 grid -- {edge_crossings(layout.nodes, layout.links)} crossings"
```

```graph-layout class=inline node_radius=9
import random as rng

rng.seed(42)
n = 15
edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.2]
used = sorted({v for e in edges for v in e})
remap = {v: i for i, v in enumerate(used)}

layout = FruchtermanReingoldLayout(
    nodes=[{"index": i} for i in range(len(used))],
    links=[{"source": remap[s], "target": remap[t]} for s, t in edges],
    size=(320, 300),
    random_seed=3,
).run()
caption = (
    f"Erdos-Renyi n={len(used)}, {len(edges)} edges -- "
    f"{edge_crossings(layout.nodes, layout.links)} crossings"
)
```

```graph-layout class=inline node_radius=9
layout = FruchtermanReingoldLayout(
    nodes=[{"index": i} for i in range(15)],
    links=[{"source": (i - 1) // 2, "target": i} for i in range(1, 15)],
    size=(320, 300),
    random_seed=11,
).run()
caption = f"Binary tree, 15 nodes -- {edge_crossings(layout.nodes, layout.links)} crossings"
```

## Force-directed

Nodes repel, edges pull. Pass `random_seed` to make a run reproducible.

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
The pair below differs only in `avoid_overlaps`.

```graph-layout class=inline node_radius=14
import math

n = 12
nodes = [
    {
        "index": i,
        "x": 150 + 100 * math.cos(2 * math.pi * i / n),
        "y": 150 + 100 * math.sin(2 * math.pi * i / n),
        "width": 30,
        "height": 30,
    }
    for i in range(n)
]
layout = ColaLayoutAdapter(
    nodes=nodes,
    links=[{"source": i, "target": (i + 1) % n} for i in range(n)],
    size=(300, 280),
    link_distance=50,
    avoid_overlaps=False,
).run()
caption = "avoid_overlaps=False"
```

```graph-layout class=inline node_radius=14
import math

n = 12
nodes = [
    {
        "index": i,
        "x": 150 + 100 * math.cos(2 * math.pi * i / n),
        "y": 150 + 100 * math.sin(2 * math.pi * i / n),
        "width": 30,
        "height": 30,
    }
    for i in range(n)
]
layout = ColaLayoutAdapter(
    nodes=nodes,
    links=[{"source": i, "target": (i + 1) % n} for i in range(n)],
    size=(300, 280),
    link_distance=50,
    avoid_overlaps=True,
).run()
caption = "avoid_overlaps=True"
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

The same 15-node binary tree, drawn as a tidy tree and radially.

```graph-layout class=inline node_radius=10
layout = ReingoldTilfordLayout(
    nodes=[{"index": i} for i in range(15)],
    links=[{"source": (i - 1) // 2, "target": i} for i in range(1, 15)],
    size=(340, 240),
).run()
caption = "Reingold-Tilford"
```

```graph-layout class=inline node_radius=10
layout = RadialTreeLayout(
    nodes=[{"index": i} for i in range(15)],
    links=[{"source": (i - 1) // 2, "target": i} for i in range(1, 15)],
    size=(320, 320),
).run()
caption = "Radial tree"
```

## Circular

```graph-layout class=inline node_radius=11
layout = CircularLayout(
    nodes=[{"index": i} for i in range(12)],
    links=[{"source": i, "target": (i + 1) % 12} for i in range(12)]
    + [{"source": i, "target": (i + 4) % 12} for i in range(12)],
    size=(300, 300),
).run()
caption = "Circular, one ring"
```

```graph-layout class=inline node_radius=11
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
    size=(300, 300),
).run()
caption = "Shell, hub plus two rings"
```

## Spectral

Positions come from the eigenvectors of the graph Laplacian, so the drawing is
deterministic for a given graph and needs no seed.

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
`to_svg()` to draw rectangles and polylines, so `node_radius` does not apply --
the orthogonal renderer does not take it.

The three drawings below are the same graph under three routing strategies, with
the bend count each achieved. `bend_optimal=True` solves the bend-minimisation
flow problem rather than routing heuristically, and reports through
`used_bend_optimal` whether it applied or fell back.

```graph-layout class=inline
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (4, 5), (5, 2)]

layout = KandinskyLayout(
    nodes=[{"index": i} for i in range(6)],
    links=[{"source": s, "target": t} for s, t in edges],
    size=(320, 260),
).run()
caption = f"Default router -- {layout.total_bends} bends"
```

```graph-layout class=inline
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (4, 5), (5, 2)]

layout = KandinskyLayout(
    nodes=[{"index": i} for i in range(6)],
    links=[{"source": s, "target": t} for s, t in edges],
    size=(320, 260),
    embedder=OptimalFlexEmbedder(),
).run()
caption = f"OptimalFlexEmbedder -- {layout.total_bends} bends"
```

```graph-layout class=inline
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (4, 5), (5, 2)]

layout = KandinskyLayout(
    nodes=[{"index": i} for i in range(6)],
    links=[{"source": s, "target": t} for s, t in edges],
    size=(320, 260),
    bend_optimal=True,
).run()
n = layout.total_bends
caption = (
    f"bend_optimal=True -- {n} bend{'' if n == 1 else 's'} "
    f"(applied: {layout.used_bend_optimal})"
)
```

GIOTTO targets degree-4 planar graphs, where a bendless drawing is often
available.

```graph-layout class=inline
edges = []
for r in range(3):
    for c in range(3):
        i = r * 3 + c
        if c < 2:
            edges.append((i, i + 1))
        if r < 2:
            edges.append((i, i + 3))

layout = GIOTTOLayout(
    nodes=[{"index": i} for i in range(9)],
    links=[{"source": s, "target": t} for s, t in edges],
    size=(320, 300),
).run()
caption = f"3x3 grid -- {layout.total_bends} bends"
```

```graph-layout class=inline
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5), (2, 6), (3, 7)]
edges += [(4, 5), (5, 6), (6, 7), (7, 4)]

layout = GIOTTOLayout(
    nodes=[{"index": i} for i in range(8)],
    links=[{"source": s, "target": t} for s, t in edges],
    size=(320, 300),
).run()
caption = f"Cube Q3 -- {layout.total_bends} bends"
```

## Planar straight-line

Crossing-free drawings of planar graphs. Schnyder and FPP place vertices on an
integer grid, and Tutte solves for a barycentric embedding with the outer face
pinned. All three are straight-line, so `to_svg()` draws them faithfully.

```graph-layout class=inline node_radius=11
edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3), (0, 4), (1, 4)]
N = [{"index": i} for i in range(5)]
L = [{"source": s, "target": t} for s, t in edges]

layout = SchnyderLayout(nodes=N, links=L, size=(300, 280)).run()
caption = f"Schnyder -- {edge_crossings(layout.nodes, layout.links)} crossings"
```

```graph-layout class=inline node_radius=11
edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3), (0, 4), (1, 4)]
N = [{"index": i} for i in range(5)]
L = [{"source": s, "target": t} for s, t in edges]

layout = FPPLayout(nodes=N, links=L, size=(300, 280)).run()
caption = f"FPP -- {edge_crossings(layout.nodes, layout.links)} crossings"
```

```graph-layout class=inline node_radius=11
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5), (2, 6), (3, 7)]
edges += [(4, 5), (5, 6), (6, 7), (7, 4)]

layout = TutteLayout(
    nodes=[{"index": i} for i in range(8)],
    links=[{"source": s, "target": t} for s, t in edges],
    size=(300, 300),
).run()
caption = f"Tutte, cube Q3 -- {edge_crossings(layout.nodes, layout.links)} crossings"
```

## Mixed-Model

`MixedModelLayout` is a visibility representation, not a straight-line drawing:
each vertex is a horizontal bar and each edge a vertical segment attaching at a
distinct port, which is what buys its angular resolution. It exposes that
geometry as `vertex_bars` and `edge_routes`. Drawing it with `to_svg()` would
join bar centres with straight lines and invent crossings the layout does not
have, so the block below renders the real geometry and binds `svg` directly.

```graph-layout source=below
edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3), (0, 4), (1, 4)]
layout = MixedModelLayout(
    nodes=[{"index": i} for i in range(5)],
    links=[{"source": s, "target": t} for s, t in edges],
    size=(300, 280),
).run()

bars, routes = layout.vertex_bars, layout.edge_routes
xs = [v for x1, x2, _ in bars.values() for v in (x1, x2)]
ys = [y for _, _, y in bars.values()]
pad = 26
w, h = max(xs) - min(xs) + 2 * pad, max(ys) - min(ys) + 2 * pad
ox, oy = pad - min(xs), pad - min(ys)

parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" height="{h:.0f}" '
    f'viewBox="0 0 {w:.0f} {h:.0f}">'
]
for pts in routes.values():
    d = " ".join(
        f"{'M' if i == 0 else 'L'} {x + ox:.1f} {y + oy:.1f}" for i, (x, y) in enumerate(pts)
    )
    parts.append(f'<path d="{d}" stroke="#666" stroke-width="1.5" fill="none"/>')
for i, (x1, x2, y) in sorted(bars.items()):
    parts.append(
        f'<line x1="{x1 + ox:.1f}" y1="{y + oy:.1f}" x2="{x2 + ox:.1f}" y2="{y + oy:.1f}" '
        f'stroke="#2c5aa0" stroke-width="6" stroke-linecap="round"/>'
    )
    parts.append(
        f'<text x="{(x1 + x2) / 2 + ox:.1f}" y="{y + oy - 9:.1f}" font-size="10" '
        f'fill="currentColor" text-anchor="middle">{i}</text>'
    )
parts.append("</svg>")

svg = "\n".join(parts)
caption = f"Mixed-Model: {len(bars)} vertex bars, {len(routes)} bendless edges"
```

## Planarization

A non-planar graph has no crossing-free straight-line drawing. `PlanarizationLayout`
replaces crossings with dummy vertices, draws the resulting planar graph, and
reports how many it needed. The routed edges live in `edge_routes`; the interior
points of each route are the dummy crossings, marked in red below.

```graph-layout class=inline
links = [{"source": i, "target": j} for i in range(5) for j in range(i + 1, 5)]
layout = PlanarizationLayout(
    nodes=[{"index": i} for i in range(5)], links=links, size=(300, 280)
).run()

routes = layout.edge_routes
pts = [p for r in routes.values() for p in r]
pad = 26
xs, ys = [p[0] for p in pts], [p[1] for p in pts]
w, h = max(xs) - min(xs) + 2 * pad, max(ys) - min(ys) + 2 * pad
ox, oy = pad - min(xs), pad - min(ys)

parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" height="{h:.0f}" '
    f'viewBox="0 0 {w:.0f} {h:.0f}">'
]
for r in routes.values():
    d = " ".join(
        f"{'M' if i == 0 else 'L'} {x + ox:.1f} {y + oy:.1f}" for i, (x, y) in enumerate(r)
    )
    parts.append(f'<path d="{d}" stroke="#666" stroke-width="1.5" fill="none"/>')
for r in routes.values():
    for x, y in r[1:-1]:
        parts.append(f'<circle cx="{x + ox:.1f}" cy="{y + oy:.1f}" r="4" fill="#d64545"/>')
for i, node in enumerate(layout.nodes):
    parts.append(
        f'<circle cx="{node.x + ox:.1f}" cy="{node.y + oy:.1f}" r="11" fill="#4a90d9" '
        f'stroke="#2c5aa0" stroke-width="2"/>'
    )
    parts.append(
        f'<text x="{node.x + ox:.1f}" y="{node.y + oy:.1f}" font-size="10" fill="#fff" '
        f'text-anchor="middle" dominant-baseline="central">{i}</text>'
    )
parts.append("</svg>")

svg = "\n".join(parts)
caption = f"K5 -- {layout.crossing_count} crossing(s), shown in red"
```

```graph-layout class=inline
links = [{"source": i, "target": 3 + j} for i in range(3) for j in range(3)]
layout = PlanarizationLayout(
    nodes=[{"index": i} for i in range(6)], links=links, size=(300, 280)
).run()

routes = layout.edge_routes
pts = [p for r in routes.values() for p in r]
pad = 26
xs, ys = [p[0] for p in pts], [p[1] for p in pts]
w, h = max(xs) - min(xs) + 2 * pad, max(ys) - min(ys) + 2 * pad
ox, oy = pad - min(xs), pad - min(ys)

parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" height="{h:.0f}" '
    f'viewBox="0 0 {w:.0f} {h:.0f}">'
]
for r in routes.values():
    d = " ".join(
        f"{'M' if i == 0 else 'L'} {x + ox:.1f} {y + oy:.1f}" for i, (x, y) in enumerate(r)
    )
    parts.append(f'<path d="{d}" stroke="#666" stroke-width="1.5" fill="none"/>')
for r in routes.values():
    for x, y in r[1:-1]:
        parts.append(f'<circle cx="{x + ox:.1f}" cy="{y + oy:.1f}" r="4" fill="#d64545"/>')
for i, node in enumerate(layout.nodes):
    parts.append(
        f'<circle cx="{node.x + ox:.1f}" cy="{node.y + oy:.1f}" r="11" fill="#4a90d9" '
        f'stroke="#2c5aa0" stroke-width="2"/>'
    )
    parts.append(
        f'<text x="{node.x + ox:.1f}" y="{node.y + oy:.1f}" font-size="10" fill="#fff" '
        f'text-anchor="middle" dominant-baseline="central">{i}</text>'
    )
parts.append("</svg>")

svg = "\n".join(parts)
caption = f"K3,3 -- {layout.crossing_count} crossing(s), shown in red"
```

## Planarity testing and Kuratowski witnesses

`check_planarity` returns a certificate either way. For a non-planar graph the
certificate is a Kuratowski subgraph -- a subdivision of K5 or K3,3 -- which is a
checkable proof of non-planarity rather than a bare `False`. It is drawn in red
below over a circular embedding.

```graph-layout class=inline
import math

n, edges = 5, [(i, j) for i in range(5) for j in range(i + 1, 5)]
result = check_planarity(n, edges)
witness = {frozenset(e) for e in (result.kuratowski_edges or [])}

r, c = 88, 108
pos = [
    (c + r * math.cos(2 * math.pi * i / n - math.pi / 2),
     c + r * math.sin(2 * math.pi * i / n - math.pi / 2))
    for i in range(n)
]

parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{2 * c}" height="{2 * c}" '
    f'viewBox="0 0 {2 * c} {2 * c}">'
]
for s, t in edges:
    hit = frozenset((s, t)) in witness
    parts.append(
        f'<line x1="{pos[s][0]:.1f}" y1="{pos[s][1]:.1f}" '
        f'x2="{pos[t][0]:.1f}" y2="{pos[t][1]:.1f}" '
        f'stroke="{"#d64545" if hit else "#bbb"}" stroke-width="{2.5 if hit else 1.2}"/>'
    )
for i, (x, y) in enumerate(pos):
    parts.append(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="11" fill="#4a90d9" '
        f'stroke="#2c5aa0" stroke-width="2"/>'
    )
    parts.append(
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="10" fill="#fff" '
        f'text-anchor="middle" dominant-baseline="central">{i}</text>'
    )
parts.append("</svg>")

svg = "\n".join(parts)
caption = f"K5: planar={result.is_planar}, {result.kuratowski_type} witness, {len(witness)} edges"
```

```graph-layout class=inline
import math

n = 6
edges = [(i, 3 + j) for i in range(3) for j in range(3)]
result = check_planarity(n, edges)
witness = {frozenset(e) for e in (result.kuratowski_edges or [])}

r, c = 88, 108
pos = [
    (c + r * math.cos(2 * math.pi * i / n - math.pi / 2),
     c + r * math.sin(2 * math.pi * i / n - math.pi / 2))
    for i in range(n)
]

parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{2 * c}" height="{2 * c}" '
    f'viewBox="0 0 {2 * c} {2 * c}">'
]
for s, t in edges:
    hit = frozenset((s, t)) in witness
    parts.append(
        f'<line x1="{pos[s][0]:.1f}" y1="{pos[s][1]:.1f}" '
        f'x2="{pos[t][0]:.1f}" y2="{pos[t][1]:.1f}" '
        f'stroke="{"#d64545" if hit else "#bbb"}" stroke-width="{2.5 if hit else 1.2}"/>'
    )
for i, (x, y) in enumerate(pos):
    parts.append(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="11" fill="#4a90d9" '
        f'stroke="#2c5aa0" stroke-width="2"/>'
    )
    parts.append(
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="10" fill="#fff" '
        f'text-anchor="middle" dominant-baseline="central">{i}</text>'
    )
parts.append("</svg>")

svg = "\n".join(parts)
caption = f"K3,3: planar={result.is_planar}, {result.kuratowski_type} witness, {len(witness)} edges"
```

## Beyond this page

The repository also ships larger standalone demos that write self-contained HTML
into `build/`: `make showcase` covers every algorithm against a dozen graph
families, and `make demos` adds org-chart, social-network, and state-machine
scenarios.
