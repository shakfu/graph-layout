#!/usr/bin/env python3
"""
Org Chart Layout Showcase - comparing layout algorithms for organizational hierarchies.

This demo generates an interactive HTML page visualizing an organizational chart
structure using different layout algorithms to demonstrate which work best for
hierarchical business structures.

Usage:
    uv run python scripts/orgchart_showcase.py

Output:
    build/orgchart_showcase.html
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

# Import all layout algorithms
from graph_layout import (
    CircularLayout,
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
SVG_WIDTH = 500
SVG_HEIGHT = 450


@dataclass
class LayoutSpec:
    """Specification for a layout algorithm."""

    name: str
    cls: type | None
    params: dict[str, Any]
    description: str
    uses_cola_api: bool = False
    rating: str = ""  # How well suited for org charts


@dataclass
class OrgPerson:
    """Represents a person in the org chart."""

    id: int
    name: str
    title: str
    level: int  # Hierarchy level (0 = CEO)


# Define layouts with their suitability rating for org charts
LAYOUTS: list[LayoutSpec] = [
    # Hierarchical - Best for org charts
    LayoutSpec(
        name="Sugiyama",
        cls=SugiyamaLayout,
        params={},
        description="Layered hierarchical - IDEAL for org charts",
        rating="BEST",
    ),
    LayoutSpec(
        name="Reingold-Tilford",
        cls=ReingoldTilfordLayout,
        params={},
        description="Classic tree layout - excellent for strict hierarchies",
        rating="BEST",
    ),
    LayoutSpec(
        name="Radial Tree",
        cls=RadialTreeLayout,
        params={},
        description="CEO at center, reports radiate outward",
        rating="GOOD",
    ),
    # Orthogonal - Good for formal diagrams
    LayoutSpec(
        name="Kandinsky",
        cls=KandinskyLayout,
        params={"node_width": 60, "node_height": 30, "node_separation": 50},
        description="Orthogonal edges - clean corporate style",
        rating="GOOD",
    ),
    # Force-directed - Okay but not ideal
    LayoutSpec(
        name="Fruchterman-Reingold",
        cls=FruchtermanReingoldLayout,
        params={"iterations": 150},
        description="Force-directed - organic but less structured",
        rating="FAIR",
    ),
    LayoutSpec(
        name="ForceAtlas2",
        cls=ForceAtlas2Layout,
        params={"iterations": 150, "scaling": 150.0, "gravity": 0.03},
        description="Community detection - shows team clusters",
        rating="FAIR",
    ),
    LayoutSpec(
        name="Kamada-Kawai",
        cls=KamadaKawaiLayout,
        params={},
        description="Stress minimization - preserves distances",
        rating="FAIR",
    ),
    LayoutSpec(
        name="Spring",
        cls=SpringLayout,
        params={"iterations": 150},
        description="Simple spring model - may overlap",
        rating="FAIR",
    ),
    LayoutSpec(
        name="Yifan Hu",
        cls=YifanHuLayout,
        params={"iterations": 150},
        description="Multilevel force - fast for large orgs",
        rating="FAIR",
    ),
    # Circular - Shows everyone but loses hierarchy
    LayoutSpec(
        name="Circular",
        cls=CircularLayout,
        params={},
        description="Equal positioning - loses hierarchy info",
        rating="POOR",
    ),
    LayoutSpec(
        name="Shell (by level)",
        cls=ShellLayout,
        params={"auto_shells": 4},
        description="Concentric by hierarchy level",
        rating="FAIR",
    ),
    # Spectral
    LayoutSpec(
        name="Spectral",
        cls=SpectralLayout,
        params={},
        description="Eigenvector based - mathematical placement",
        rating="POOR",
    ),
    # Cola
    LayoutSpec(
        name="Cola",
        cls=ColaLayout,
        params={"link_distance": 80},
        description="Constraint-based with overlap avoidance",
        uses_cola_api=True,
        rating="FAIR",
    ),
]


def generate_orgchart() -> tuple[list[OrgPerson], list[dict]]:
    """Generate a realistic company org chart structure.

    Creates a typical corporate hierarchy:
    - CEO (1)
    - C-Suite executives (3)
    - VPs/Directors (6)
    - Managers (9)
    - Individual contributors (12)

    Total: 31 people
    """
    people = []
    links = []

    # Level 0: CEO
    people.append(OrgPerson(0, "Sarah Chen", "CEO", 0))

    # Level 1: C-Suite (report to CEO)
    c_suite = [
        OrgPerson(1, "Mike Torres", "CTO", 1),
        OrgPerson(2, "Lisa Park", "CFO", 1),
        OrgPerson(3, "James Wilson", "COO", 1),
    ]
    people.extend(c_suite)
    for person in c_suite:
        links.append({"source": 0, "target": person.id})

    # Level 2: VPs/Directors (report to C-Suite)
    vps = [
        # Report to CTO
        OrgPerson(4, "Emma Davis", "VP Engineering", 2),
        OrgPerson(5, "Alex Kim", "VP Product", 2),
        # Report to CFO
        OrgPerson(6, "Rachel Green", "Controller", 2),
        OrgPerson(7, "Tom Brown", "VP Finance", 2),
        # Report to COO
        OrgPerson(8, "Nina Shah", "VP Operations", 2),
        OrgPerson(9, "Chris Lee", "VP Sales", 2),
    ]
    people.extend(vps)
    # CTO reports
    links.append({"source": 1, "target": 4})
    links.append({"source": 1, "target": 5})
    # CFO reports
    links.append({"source": 2, "target": 6})
    links.append({"source": 2, "target": 7})
    # COO reports
    links.append({"source": 3, "target": 8})
    links.append({"source": 3, "target": 9})

    # Level 3: Managers (report to VPs)
    managers = [
        # Report to VP Engineering
        OrgPerson(10, "David Clark", "Eng Manager", 3),
        OrgPerson(11, "Sophie Martin", "Eng Manager", 3),
        # Report to VP Product
        OrgPerson(12, "Ryan Adams", "Product Manager", 3),
        # Report to Controller
        OrgPerson(13, "Kate Miller", "Accounting Mgr", 3),
        # Report to VP Finance
        OrgPerson(14, "Josh Wang", "FP&A Manager", 3),
        # Report to VP Operations
        OrgPerson(15, "Amy Taylor", "Ops Manager", 3),
        OrgPerson(16, "Ben Harris", "Ops Manager", 3),
        # Report to VP Sales
        OrgPerson(17, "Grace Liu", "Sales Manager", 3),
        OrgPerson(18, "Mark Johnson", "Sales Manager", 3),
    ]
    people.extend(managers)
    links.extend(
        [
            {"source": 4, "target": 10},
            {"source": 4, "target": 11},
            {"source": 5, "target": 12},
            {"source": 6, "target": 13},
            {"source": 7, "target": 14},
            {"source": 8, "target": 15},
            {"source": 8, "target": 16},
            {"source": 9, "target": 17},
            {"source": 9, "target": 18},
        ]
    )

    # Level 4: Individual Contributors (report to Managers)
    ics = [
        # Report to Eng Manager (David)
        OrgPerson(19, "Zoe White", "Sr Engineer", 4),
        OrgPerson(20, "Ian Scott", "Engineer", 4),
        # Report to Eng Manager (Sophie)
        OrgPerson(21, "Mia Young", "Sr Engineer", 4),
        OrgPerson(22, "Leo King", "Engineer", 4),
        # Report to Product Manager
        OrgPerson(23, "Ava Hill", "Assoc PM", 4),
        # Report to Accounting Mgr
        OrgPerson(24, "Ethan Moore", "Accountant", 4),
        # Report to FP&A Manager
        OrgPerson(25, "Olivia Reed", "Analyst", 4),
        # Report to Ops Manager (Amy)
        OrgPerson(26, "Liam Cox", "Ops Specialist", 4),
        # Report to Ops Manager (Ben)
        OrgPerson(27, "Chloe Ward", "Ops Specialist", 4),
        # Report to Sales Manager (Grace)
        OrgPerson(28, "Noah Price", "Sales Rep", 4),
        OrgPerson(29, "Emma Ross", "Sales Rep", 4),
        # Report to Sales Manager (Mark)
        OrgPerson(30, "Jack Bell", "Sales Rep", 4),
    ]
    people.extend(ics)
    links.extend(
        [
            {"source": 10, "target": 19},
            {"source": 10, "target": 20},
            {"source": 11, "target": 21},
            {"source": 11, "target": 22},
            {"source": 12, "target": 23},
            {"source": 13, "target": 24},
            {"source": 14, "target": 25},
            {"source": 15, "target": 26},
            {"source": 16, "target": 27},
            {"source": 17, "target": 28},
            {"source": 17, "target": 29},
            {"source": 18, "target": 30},
        ]
    )

    return people, links


def generate_small_orgchart() -> tuple[list[OrgPerson], list[dict]]:
    """Generate a smaller org chart for clearer visualization.

    CEO -> 3 Directors -> 6 Team members
    Total: 10 people
    """
    people = [
        OrgPerson(0, "CEO", "Chief Executive", 0),
        OrgPerson(1, "CTO", "Tech Lead", 1),
        OrgPerson(2, "CFO", "Finance Lead", 1),
        OrgPerson(3, "COO", "Operations Lead", 1),
        OrgPerson(4, "Eng 1", "Engineer", 2),
        OrgPerson(5, "Eng 2", "Engineer", 2),
        OrgPerson(6, "Fin 1", "Analyst", 2),
        OrgPerson(7, "Fin 2", "Analyst", 2),
        OrgPerson(8, "Ops 1", "Specialist", 2),
        OrgPerson(9, "Ops 2", "Specialist", 2),
    ]
    links = [
        {"source": 0, "target": 1},
        {"source": 0, "target": 2},
        {"source": 0, "target": 3},
        {"source": 1, "target": 4},
        {"source": 1, "target": 5},
        {"source": 2, "target": 6},
        {"source": 2, "target": 7},
        {"source": 3, "target": 8},
        {"source": 3, "target": 9},
    ]
    return people, links


def generate_startup_orgchart() -> tuple[list[OrgPerson], list[dict]]:
    """Generate a flat startup org chart.

    CEO with many direct reports (typical early-stage startup).
    """
    people = [
        OrgPerson(0, "Founder", "CEO", 0),
        OrgPerson(1, "Dev 1", "Full Stack", 1),
        OrgPerson(2, "Dev 2", "Backend", 1),
        OrgPerson(3, "Dev 3", "Frontend", 1),
        OrgPerson(4, "Designer", "UX/UI", 1),
        OrgPerson(5, "Marketing", "Growth", 1),
        OrgPerson(6, "Sales", "BD", 1),
        OrgPerson(7, "Support", "Customer Success", 1),
    ]
    links = [{"source": 0, "target": i} for i in range(1, 8)]
    return people, links


def run_layout(spec: LayoutSpec, people: list[OrgPerson], links: list[dict]) -> Any:
    """Run a layout algorithm and return the layout object."""
    random.seed(42)  # Reproducible

    # Create node data
    node_data = [{} for _ in people]
    link_data = [{"source": l["source"], "target": l["target"]} for l in links]

    if spec.uses_cola_api:
        node_data = [
            {
                "x": random.uniform(100, SVG_WIDTH - 100),
                "y": random.uniform(100, SVG_HEIGHT - 100),
                "width": 20,
                "height": 20,
            }
            for _ in people
        ]
        layout = ColaLayout()
        layout.nodes(node_data).links(link_data).size([SVG_WIDTH, SVG_HEIGHT])
        layout.link_distance(spec.params.get("link_distance", 80))
        layout.start(30, 0, 30, 0, False)
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


def get_level_color(level: int) -> str:
    """Get color based on hierarchy level."""
    colors = [
        "#e74c3c",  # Level 0: CEO - Red
        "#3498db",  # Level 1: C-Suite - Blue
        "#2ecc71",  # Level 2: VPs - Green
        "#f39c12",  # Level 3: Managers - Orange
        "#9b59b6",  # Level 4: ICs - Purple
    ]
    return colors[min(level, len(colors) - 1)]


def get_rating_color(rating: str) -> str:
    """Get badge color based on rating."""
    return {
        "BEST": "#27ae60",
        "GOOD": "#3498db",
        "FAIR": "#f39c12",
        "POOR": "#e74c3c",
    }.get(rating, "#95a5a6")


def center_layout(nodes: list, layout: Any, padding: int = 40) -> None:
    """Scale and center the layout within the SVG canvas.

    Modifies node positions in-place to fit and center the graph.
    Also adjusts orthogonal edge bend points if present.
    """
    if not nodes:
        return

    # Compute bounding box
    min_x = min(n.x for n in nodes)
    max_x = max(n.x for n in nodes)
    min_y = min(n.y for n in nodes)
    max_y = max(n.y for n in nodes)

    graph_width = max_x - min_x
    graph_height = max_y - min_y
    graph_cx = (min_x + max_x) / 2
    graph_cy = (min_y + max_y) / 2

    # Available space (accounting for header, footer, and padding)
    header_space = 50
    footer_space = 25
    available_width = SVG_WIDTH - 2 * padding
    available_height = SVG_HEIGHT - header_space - footer_space - 2 * padding

    # Calculate scale factor to fit graph in available space
    scale = 1.0
    if graph_width > 0 and graph_height > 0:
        scale_x = available_width / graph_width
        scale_y = available_height / graph_height
        scale = min(scale_x, scale_y, 1.0)  # Don't scale up, only down if needed

    # Target center position
    target_cx = SVG_WIDTH / 2
    target_cy = header_space + padding + available_height / 2

    # Apply scale around graph center, then translate to target center
    for node in nodes:
        node.x = (node.x - graph_cx) * scale + target_cx
        node.y = (node.y - graph_cy) * scale + target_cy

    # Apply same transform to orthogonal edge bends if present
    if hasattr(layout, "orthogonal_edges"):
        for edge in layout.orthogonal_edges:
            edge.bends = [
                ((bx - graph_cx) * scale + target_cx, (by - graph_cy) * scale + target_cy)
                for bx, by in edge.bends
            ]


def layout_to_svg(
    layout: Any,
    spec: LayoutSpec,
    people: list[OrgPerson],
    org_name: str,
    show_orthogonal: bool = False,
) -> str:
    """Convert a layout to SVG with org chart styling."""
    # Get nodes and links
    if spec.uses_cola_api:
        nodes = layout.nodes()
        links = layout.links()
    else:
        nodes = layout.nodes
        links = layout.links

    # Center the graph in the SVG canvas
    center_layout(nodes, layout)

    # Build SVG
    svg_parts = [
        f'<svg width="{SVG_WIDTH}" height="{SVG_HEIGHT}" xmlns="http://www.w3.org/2000/svg">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
    ]

    # Draw edges
    if show_orthogonal and hasattr(layout, "orthogonal_edges"):
        for edge in layout.orthogonal_edges:
            src_idx = edge.source
            tgt_idx = edge.target
            if src_idx >= len(nodes) or tgt_idx >= len(nodes):
                continue

            src_node = nodes[src_idx]
            tgt_node = nodes[tgt_idx]

            path_parts = [f"M {src_node.x:.1f} {src_node.y:.1f}"]
            for bx, by in edge.bends:
                path_parts.append(f"L {bx:.1f} {by:.1f}")
            path_parts.append(f"L {tgt_node.x:.1f} {tgt_node.y:.1f}")

            svg_parts.append(
                f'<path d="{" ".join(path_parts)}" stroke="#7f8c8d" stroke-width="2" fill="none"/>'
            )
    else:
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
                f'stroke="#bdc3c7" stroke-width="2"/>'
            )

    # Draw nodes as circles with person info
    for i, node in enumerate(nodes):
        if i >= len(people):
            continue

        person = people[i]
        color = get_level_color(person.level)
        radius = 12 if person.level == 0 else 10 if person.level <= 2 else 8

        # Node circle
        svg_parts.append(
            f'<circle cx="{node.x:.1f}" cy="{node.y:.1f}" r="{radius}" '
            f'fill="{color}" stroke="#fff" stroke-width="2"/>'
        )

        # Name label (abbreviated)
        name_abbrev = person.name.split()[0][:6] if " " in person.name else person.name[:6]
        svg_parts.append(
            f'<text x="{node.x:.1f}" y="{node.y + radius + 12:.1f}" '
            f'text-anchor="middle" fill="#2c3e50" font-size="7" '
            f'font-family="sans-serif">{escape(name_abbrev)}</text>'
        )

    # Title with rating badge
    rating_color = get_rating_color(spec.rating)
    svg_parts.append(
        f'<text x="{SVG_WIDTH // 2}" y="22" text-anchor="middle" '
        f'fill="#2c3e50" font-size="14" font-weight="bold" '
        f'font-family="sans-serif">{escape(spec.name)}</text>'
    )

    # Rating badge
    badge_x = SVG_WIDTH // 2 + len(spec.name) * 4 + 20
    svg_parts.append(
        f'<rect x="{badge_x - 20}" y="10" width="40" height="16" rx="8" fill="{rating_color}"/>'
    )
    svg_parts.append(
        f'<text x="{badge_x}" y="22" text-anchor="middle" fill="#fff" '
        f'font-size="8" font-weight="bold" font-family="sans-serif">'
        f"{escape(spec.rating)}</text>"
    )

    # Description
    svg_parts.append(
        f'<text x="{SVG_WIDTH // 2}" y="38" text-anchor="middle" '
        f'fill="#7f8c8d" font-size="9" font-family="sans-serif">'
        f"{escape(spec.description)}</text>"
    )

    # Legend at bottom
    svg_parts.append(
        f'<text x="{SVG_WIDTH // 2}" y="{SVG_HEIGHT - 8}" text-anchor="middle" '
        f'fill="#95a5a6" font-size="8" font-family="sans-serif">'
        f"{escape(org_name)} ({len(people)} people)</text>"
    )

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def generate_html(sections: list[tuple[str, str, list[str]]]) -> str:
    """Generate the HTML page.

    sections: list of (title, description, svgs)
    """
    html_parts = [
        """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Org Chart Layout Comparison</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #ecf0f1;
            color: #2c3e50;
            line-height: 1.6;
        }
        header {
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            color: white;
            padding: 2.5rem;
            text-align: center;
        }
        header h1 {
            font-size: 2.2rem;
            margin-bottom: 0.5rem;
        }
        header p {
            opacity: 0.9;
            font-size: 1.1rem;
        }
        main {
            max-width: 1800px;
            margin: 0 auto;
            padding: 2rem;
        }
        .intro {
            background: white;
            border-radius: 12px;
            padding: 1.5rem 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .intro h3 {
            color: #3498db;
            margin-bottom: 1rem;
        }
        .rating-guide {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin-top: 1rem;
        }
        .rating-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.4rem 1rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        .rating-best { background: #27ae60; color: white; }
        .rating-good { background: #3498db; color: white; }
        .rating-fair { background: #f39c12; color: white; }
        .rating-poor { background: #e74c3c; color: white; }
        .level-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin-top: 1rem;
        }
        .level-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .level-dot {
            width: 16px;
            height: 16px;
            border-radius: 50%;
        }
        section {
            margin-bottom: 3rem;
        }
        section h2 {
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 3px solid #3498db;
            color: #2c3e50;
        }
        section > p {
            color: #7f8c8d;
            margin-bottom: 1.5rem;
        }
        .graph-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(520px, 1fr));
            gap: 1.5rem;
        }
        .graph-card {
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .graph-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        }
        .graph-card svg {
            display: block;
            width: 100%;
            height: auto;
        }
        footer {
            text-align: center;
            padding: 2rem;
            color: #7f8c8d;
            font-size: 0.9rem;
        }
        .recommendation {
            background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
            color: white;
            padding: 1.5rem 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
        }
        .recommendation h3 {
            margin-bottom: 0.5rem;
        }
        .recommendation ul {
            margin-left: 1.5rem;
        }
    </style>
</head>
<body>
    <header>
        <h1>Org Chart Layout Comparison</h1>
        <p>Finding the best graph layout algorithm for organizational hierarchies</p>
    </header>
    <main>
        <div class="intro">
            <h3>Understanding the Ratings</h3>
            <p>Each layout algorithm is rated on how well it visualizes org chart hierarchies:</p>
            <div class="rating-guide">
                <span class="rating-badge rating-best">BEST - Designed for hierarchies</span>
                <span class="rating-badge rating-good">GOOD - Works well</span>
                <span class="rating-badge rating-fair">FAIR - Usable but not ideal</span>
                <span class="rating-badge rating-poor">POOR - Loses hierarchy information</span>
            </div>
            <h3 style="margin-top: 1.5rem;">Hierarchy Levels (by color)</h3>
            <div class="level-legend">
                <div class="level-item">
                    <span class="level-dot" style="background: #e74c3c;"></span>
                    <span>CEO</span>
                </div>
                <div class="level-item">
                    <span class="level-dot" style="background: #3498db;"></span>
                    <span>C-Suite</span>
                </div>
                <div class="level-item">
                    <span class="level-dot" style="background: #2ecc71;"></span>
                    <span>VPs/Directors</span>
                </div>
                <div class="level-item">
                    <span class="level-dot" style="background: #f39c12;"></span>
                    <span>Managers</span>
                </div>
                <div class="level-item">
                    <span class="level-dot" style="background: #9b59b6;"></span>
                    <span>Individual Contributors</span>
                </div>
            </div>
        </div>

        <div class="recommendation">
            <h3>Recommendation</h3>
            <p>For organizational charts, use:</p>
            <ul>
                <li><strong>Sugiyama</strong> - Best for top-down org charts</li>
                <li><strong>Reingold-Tilford</strong> - For strict tree hierarchies</li>
                <li><strong>Radial Tree</strong> - CEO at center layout</li>
                <li><strong>Kandinsky</strong> - Clean orthogonal lines</li>
            </ul>
        </div>
"""
    ]

    for section_title, section_desc, svgs in sections:
        html_parts.append(f"        <section>\n            <h2>{escape(section_title)}</h2>")
        html_parts.append(f"            <p>{escape(section_desc)}</p>")
        html_parts.append('            <div class="graph-grid">')
        for svg in svgs:
            card = f'                <div class="graph-card">\n{svg}\n                </div>'
            html_parts.append(card)
        html_parts.append("            </div>\n        </section>")

    html_parts.append(
        """    </main>
    <footer>
        <p>Generated by graph-layout library | Org Chart Visualization Demo</p>
    </footer>
</body>
</html>"""
    )

    return "\n".join(html_parts)


def main() -> None:
    """Generate the org chart showcase HTML."""
    BUILD_DIR.mkdir(exist_ok=True)

    # Define org charts to visualize
    org_charts = [
        (
            "Full Corporate Org Chart (31 people)",
            "A typical corporate hierarchy with CEO, C-Suite, VPs, Managers, and ICs",
            generate_orgchart(),
        ),
        (
            "Small Team Org Chart (10 people)",
            "A compact org structure - easier to see layout differences",
            generate_small_orgchart(),
        ),
        (
            "Flat Startup Org Chart (8 people)",
            "A flat organization with one leader and many direct reports",
            generate_startup_orgchart(),
        ),
    ]

    sections = []

    for org_name, org_desc, (people, links) in org_charts:
        print(f"\nProcessing: {org_name}")
        print(f"  People: {len(people)}, Reporting lines: {len(links)}")

        svgs = []
        for spec in LAYOUTS:
            try:
                print(f"  Running {spec.name}...")
                layout = run_layout(spec, people, links)
                show_ortho = isinstance(layout, KandinskyLayout)
                svg = layout_to_svg(layout, spec, people, org_name, show_orthogonal=show_ortho)
                svgs.append(svg)
            except Exception as e:
                print(f"    Error: {e}")

        sections.append((org_name, org_desc, svgs))

    # Generate HTML
    html = generate_html(sections)

    output_path = BUILD_DIR / "orgchart_showcase.html"
    output_path.write_text(html)
    print(f"\nOrg chart showcase saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
