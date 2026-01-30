#!/usr/bin/env python3
"""
HTML/SVG showcase of graph layout algorithms.

Generates an interactive HTML page with SVG visualizations of all layout
algorithms applied to both random and structured (non-random) graphs.

Usage:
    uv run python scripts/showcase.py

Output:
    build/showcase.html
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Callable

# Import all layout algorithms
from graph_layout import (
    BipartiteLayout,
    CircularLayout,
    ColaLayoutAdapter,
    ForceAtlas2Layout,
    FruchtermanReingoldLayout,
    KamadaKawaiLayout,
    KandinskyLayout,
    RadialTreeLayout,
    ReingoldTilfordLayout,
    ShellLayout,
    SpectralLayout,
    SpringLayout,
    SugiyamaLayout,
    YifanHuLayout,
)
from graph_layout.cola import Layout as ColaLayout

# Output directory
BUILD_DIR = Path(__file__).parent.parent / "build"

# SVG dimensions
SVG_WIDTH = 400
SVG_HEIGHT = 350


@dataclass
class LayoutSpec:
    """Specification for a layout algorithm."""

    name: str
    cls: type | None
    params: dict[str, Any]
    description: str
    uses_cola_api: bool = False
    suitable_for: tuple[str, ...] = ("general",)


# Define all layout algorithms with their parameters
LAYOUTS: list[LayoutSpec] = [
    # Force-directed layouts
    LayoutSpec(
        name="Fruchterman-Reingold",
        cls=FruchtermanReingoldLayout,
        params={"iterations": 100},
        description="Classic force-directed with temperature cooling",
        suitable_for=("general", "random"),
    ),
    LayoutSpec(
        name="ForceAtlas2",
        cls=ForceAtlas2Layout,
        params={"iterations": 100, "scaling": 100.0, "gravity": 0.05},
        description="Degree-weighted repulsion, good for large networks",
        suitable_for=("general", "random"),
    ),
    LayoutSpec(
        name="Spring",
        cls=SpringLayout,
        params={"iterations": 100},
        description="Simple Hooke's law spring forces",
        suitable_for=("general", "random"),
    ),
    LayoutSpec(
        name="Kamada-Kawai",
        cls=KamadaKawaiLayout,
        params={},
        description="Stress minimization using graph-theoretic distances",
        suitable_for=("general", "random"),
    ),
    LayoutSpec(
        name="Yifan Hu",
        cls=YifanHuLayout,
        params={"iterations": 100},
        description="Multilevel with Barnes-Hut approximation",
        suitable_for=("general", "random"),
    ),
    # Circular layouts
    LayoutSpec(
        name="Circular",
        cls=CircularLayout,
        params={},
        description="Nodes arranged on a circle",
        suitable_for=("general", "random"),
    ),
    LayoutSpec(
        name="Shell (2 shells)",
        cls=ShellLayout,
        params={"auto_shells": 2},
        description="Concentric circles by degree",
        suitable_for=("general", "random"),
    ),
    # Spectral
    LayoutSpec(
        name="Spectral",
        cls=SpectralLayout,
        params={},
        description="Laplacian eigenvector embedding",
        suitable_for=("general", "random"),
    ),
    # Hierarchical layouts (best for trees/DAGs)
    LayoutSpec(
        name="Sugiyama",
        cls=SugiyamaLayout,
        params={},
        description="Layered DAG drawing with crossing minimization",
        suitable_for=("tree", "dag"),
    ),
    LayoutSpec(
        name="Reingold-Tilford",
        cls=ReingoldTilfordLayout,
        params={},
        description="Tree layout with compact positioning",
        suitable_for=("tree",),
    ),
    LayoutSpec(
        name="Radial Tree",
        cls=RadialTreeLayout,
        params={},
        description="Radial tree with root at center",
        suitable_for=("tree",),
    ),
    # Bipartite
    LayoutSpec(
        name="Bipartite",
        cls=BipartiteLayout,
        params={},
        description="Two parallel rows with crossing minimization",
        suitable_for=("bipartite",),
    ),
    # Orthogonal
    LayoutSpec(
        name="Kandinsky",
        cls=KandinskyLayout,
        params={"node_width": 30, "node_height": 20, "node_separation": 40},
        description="Orthogonal edges (horizontal/vertical only)",
        suitable_for=("general", "tree"),
    ),
    # Cola (constraint-based)
    LayoutSpec(
        name="Cola",
        cls=ColaLayout,
        params={"link_distance": 60},
        description="Constraint-based with overlap avoidance",
        uses_cola_api=True,
        suitable_for=("general", "random"),
    ),
]


def generate_erdos_renyi(n: int, p: float, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Generate Erdos-Renyi random graph, removing isolated nodes."""
    random.seed(seed)
    links = []
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                links.append({"source": i, "target": j})

    # Find all connected nodes
    connected = set()
    for link in links:
        connected.add(link["source"])
        connected.add(link["target"])

    # Create mapping from old indices to new indices
    old_to_new = {old: new for new, old in enumerate(sorted(connected))}

    # Remap links to use new indices
    remapped_links = [
        {"source": old_to_new[l["source"]], "target": old_to_new[l["target"]]}
        for l in links
    ]

    # Create nodes only for connected nodes
    nodes = [{} for _ in range(len(connected))]
    return nodes, remapped_links


def generate_tree_graph() -> tuple[list[dict], list[dict]]:
    """Generate a tree graph for hierarchical layouts."""
    nodes = [{} for _ in range(15)]
    links = [
        # Level 0 -> 1
        {"source": 0, "target": 1},
        {"source": 0, "target": 2},
        {"source": 0, "target": 3},
        # Level 1 -> 2
        {"source": 1, "target": 4},
        {"source": 1, "target": 5},
        {"source": 2, "target": 6},
        {"source": 2, "target": 7},
        {"source": 3, "target": 8},
        {"source": 3, "target": 9},
        # Level 2 -> 3
        {"source": 4, "target": 10},
        {"source": 5, "target": 11},
        {"source": 6, "target": 12},
        {"source": 8, "target": 13},
        {"source": 9, "target": 14},
    ]
    return nodes, links


def generate_grid_graph(rows: int = 4, cols: int = 4) -> tuple[list[dict], list[dict]]:
    """Generate a 2D grid graph."""
    nodes = [{} for _ in range(rows * cols)]
    links = []

    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            # Connect to right neighbor
            if c < cols - 1:
                links.append({"source": i, "target": i + 1})
            # Connect to bottom neighbor
            if r < rows - 1:
                links.append({"source": i, "target": i + cols})

    return nodes, links


def generate_bipartite_graph() -> tuple[list[dict], list[dict]]:
    """Generate a bipartite graph."""
    # 5 nodes in set A, 5 in set B
    nodes = [{} for _ in range(10)]
    links = [
        {"source": 0, "target": 5},
        {"source": 0, "target": 6},
        {"source": 1, "target": 6},
        {"source": 1, "target": 7},
        {"source": 2, "target": 5},
        {"source": 2, "target": 8},
        {"source": 3, "target": 7},
        {"source": 3, "target": 9},
        {"source": 4, "target": 8},
        {"source": 4, "target": 9},
    ]
    return nodes, links


def generate_petersen_graph() -> tuple[list[dict], list[dict]]:
    """Generate the Petersen graph (famous non-random graph)."""
    nodes = [{} for _ in range(10)]
    # Outer pentagon (0-4), inner pentagram (5-9)
    links = [
        # Outer pentagon
        {"source": 0, "target": 1},
        {"source": 1, "target": 2},
        {"source": 2, "target": 3},
        {"source": 3, "target": 4},
        {"source": 4, "target": 0},
        # Inner pentagram (skip-2 connections)
        {"source": 5, "target": 7},
        {"source": 7, "target": 9},
        {"source": 9, "target": 6},
        {"source": 6, "target": 8},
        {"source": 8, "target": 5},
        # Spokes connecting outer to inner
        {"source": 0, "target": 5},
        {"source": 1, "target": 6},
        {"source": 2, "target": 7},
        {"source": 3, "target": 8},
        {"source": 4, "target": 9},
    ]
    return nodes, links


def run_layout(spec: LayoutSpec, nodes: list[dict], links: list[dict]) -> Any:
    """Run a layout algorithm and return the layout object."""
    random.seed(42)  # Reproducible

    # Create fresh copies
    node_data = [{} for _ in nodes]
    link_data = [{"source": l["source"], "target": l["target"]} for l in links]

    if spec.uses_cola_api:
        # Cola uses different API
        node_data = [
            {"x": random.uniform(100, SVG_WIDTH - 100), "y": random.uniform(100, SVG_HEIGHT - 100), "width": 15, "height": 15}
            for _ in nodes
        ]
        layout = ColaLayout()
        layout.nodes(node_data).links(link_data).size([SVG_WIDTH, SVG_HEIGHT])
        layout.link_distance(spec.params.get("link_distance", 80))
        layout.start(20, 0, 20, 0, False)
        return layout

    # Standard Pythonic API
    layout = spec.cls(
        nodes=node_data,
        links=link_data,
        size=(SVG_WIDTH, SVG_HEIGHT),
        **spec.params,
    )
    layout.run()
    return layout


def layout_to_svg(
    layout: Any,
    spec: LayoutSpec,
    graph_name: str,
    show_orthogonal: bool = False,
) -> str:
    """Convert a layout to SVG."""
    # Get nodes and links
    if spec.uses_cola_api:
        nodes = layout.nodes()
        links = layout.links()
    else:
        nodes = layout.nodes
        links = layout.links

    # Build SVG
    svg_parts = [
        f'<svg width="{SVG_WIDTH}" height="{SVG_HEIGHT}" xmlns="http://www.w3.org/2000/svg">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
    ]

    # Draw edges
    if show_orthogonal and hasattr(layout, "orthogonal_edges"):
        # Draw orthogonal edges with bends
        for edge in layout.orthogonal_edges:
            src_idx = edge.source
            tgt_idx = edge.target

            if src_idx >= len(nodes) or tgt_idx >= len(nodes):
                continue

            src_node = nodes[src_idx]
            tgt_node = nodes[tgt_idx]

            # Build path through bends
            path_parts = [f"M {src_node.x:.1f} {src_node.y:.1f}"]
            for bx, by in edge.bends:
                path_parts.append(f"L {bx:.1f} {by:.1f}")
            path_parts.append(f"L {tgt_node.x:.1f} {tgt_node.y:.1f}")

            svg_parts.append(
                f'<path d="{" ".join(path_parts)}" stroke="#888" stroke-width="1.5" fill="none"/>'
            )
    else:
        # Draw straight edges
        for link in links:
            src_idx = link.source if isinstance(link.source, int) else link.source.index
            tgt_idx = link.target if isinstance(link.target, int) else link.target.index

            if src_idx >= len(nodes) or tgt_idx >= len(nodes):
                continue

            src = nodes[src_idx]
            tgt = nodes[tgt_idx]
            svg_parts.append(
                f'<line x1="{src.x:.1f}" y1="{src.y:.1f}" '
                f'x2="{tgt.x:.1f}" y2="{tgt.y:.1f}" '
                f'stroke="#888" stroke-width="1.5" opacity="0.6"/>'
            )

    # Draw nodes
    for i, node in enumerate(nodes):
        svg_parts.append(
            f'<circle cx="{node.x:.1f}" cy="{node.y:.1f}" r="8" '
            f'fill="#4a90d9" stroke="#fff" stroke-width="1.5"/>'
        )
        # Node label
        svg_parts.append(
            f'<text x="{node.x:.1f}" y="{node.y + 3:.1f}" '
            f'text-anchor="middle" fill="#fff" font-size="8" font-family="sans-serif">{i}</text>'
        )

    # Title with algorithm name and parameters
    params_str = ", ".join(f"{k}={v}" for k, v in spec.params.items()) if spec.params else ""
    title = f"{spec.name}"
    if params_str:
        title += f" ({params_str})"

    svg_parts.append(
        f'<text x="{SVG_WIDTH // 2}" y="20" text-anchor="middle" '
        f'fill="#333" font-size="12" font-weight="bold" font-family="sans-serif">{escape(title)}</text>'
    )
    svg_parts.append(
        f'<text x="{SVG_WIDTH // 2}" y="35" text-anchor="middle" '
        f'fill="#666" font-size="9" font-family="sans-serif">{escape(spec.description)}</text>'
    )
    svg_parts.append(
        f'<text x="{SVG_WIDTH // 2}" y="{SVG_HEIGHT - 10}" text-anchor="middle" '
        f'fill="#999" font-size="9" font-family="sans-serif">Graph: {escape(graph_name)}</text>'
    )

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def generate_html(sections: list[tuple[str, list[str]]]) -> str:
    """Generate the HTML page."""
    html_parts = [
        """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Graph Layout Algorithm Showcase</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f0f0f0;
            color: #333;
            line-height: 1.6;
        }
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            text-align: center;
        }
        header h1 {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        header p {
            opacity: 0.9;
        }
        main {
            max-width: 1800px;
            margin: 0 auto;
            padding: 2rem;
        }
        section {
            margin-bottom: 3rem;
        }
        section h2 {
            font-size: 1.5rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #667eea;
            color: #333;
        }
        .graph-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
            gap: 1.5rem;
        }
        .graph-card {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .graph-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 4px 16px rgba(0,0,0,0.15);
        }
        .graph-card svg {
            display: block;
            width: 100%;
            height: auto;
        }
        footer {
            text-align: center;
            padding: 2rem;
            color: #666;
            font-size: 0.9rem;
        }
        .legend {
            background: white;
            border-radius: 8px;
            padding: 1rem 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .legend h3 {
            margin-bottom: 0.5rem;
            color: #667eea;
        }
        .legend ul {
            list-style: none;
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
        }
        .legend li {
            background: #f8f8f8;
            padding: 0.3rem 0.8rem;
            border-radius: 4px;
            font-size: 0.85rem;
        }
    </style>
</head>
<body>
    <header>
        <h1>Graph Layout Algorithm Showcase</h1>
        <p>Comparing different layout algorithms on random and structured graphs</p>
    </header>
    <main>
        <div class="legend">
            <h3>Algorithm Categories</h3>
            <ul>
                <li><strong>Force-directed:</strong> FR, FA2, Spring, KK, Yifan Hu</li>
                <li><strong>Circular:</strong> Circular, Shell</li>
                <li><strong>Spectral:</strong> Laplacian eigenvector</li>
                <li><strong>Hierarchical:</strong> Sugiyama, Reingold-Tilford, Radial</li>
                <li><strong>Orthogonal:</strong> Kandinsky</li>
                <li><strong>Constraint-based:</strong> Cola</li>
            </ul>
        </div>
"""
    ]

    for section_title, svgs in sections:
        html_parts.append(f'        <section>\n            <h2>{escape(section_title)}</h2>')
        html_parts.append('            <div class="graph-grid">')
        for svg in svgs:
            html_parts.append(f'                <div class="graph-card">\n{svg}\n                </div>')
        html_parts.append("            </div>\n        </section>")

    html_parts.append(
        """    </main>
    <footer>
        <p>Generated by graph-layout library</p>
    </footer>
</body>
</html>"""
    )

    return "\n".join(html_parts)


def main() -> None:
    """Generate the showcase HTML."""
    BUILD_DIR.mkdir(exist_ok=True)

    # Define graphs to visualize
    graphs = {
        "Random (Erdos-Renyi n=15, p=0.2)": generate_erdos_renyi(15, 0.2, seed=42),
        "Petersen Graph": generate_petersen_graph(),
        "Tree (15 nodes)": generate_tree_graph(),
        "Grid (4x4)": generate_grid_graph(4, 4),
        "Bipartite (5+5)": generate_bipartite_graph(),
    }

    # Which layouts to use for which graph types
    graph_type_map = {
        "Random (Erdos-Renyi n=15, p=0.2)": ("general", "random"),
        "Petersen Graph": ("general",),
        "Tree (15 nodes)": ("tree", "general"),
        "Grid (4x4)": ("general",),
        "Bipartite (5+5)": ("bipartite", "general"),
    }

    sections = []

    for graph_name, (nodes, links) in graphs.items():
        print(f"\nProcessing: {graph_name}")
        print(f"  Nodes: {len(nodes)}, Edges: {len(links)}")

        svgs = []
        suitable_types = graph_type_map.get(graph_name, ("general",))

        for spec in LAYOUTS:
            # Check if this layout is suitable for this graph type
            if not any(t in spec.suitable_for for t in suitable_types):
                continue

            try:
                print(f"  Running {spec.name}...")
                layout = run_layout(spec, nodes, links)
                show_ortho = isinstance(layout, KandinskyLayout)
                svg = layout_to_svg(layout, spec, graph_name, show_orthogonal=show_ortho)
                svgs.append(svg)
            except Exception as e:
                print(f"    Error: {e}")

        sections.append((graph_name, svgs))

    # Generate HTML
    html = generate_html(sections)

    output_path = BUILD_DIR / "showcase.html"
    output_path.write_text(html)
    print(f"\nShowcase saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
