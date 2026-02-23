#!/usr/bin/env python3
"""
State Machine Layout Showcase - comparing layout algorithms for workflow/state diagrams.

This demo generates an interactive HTML page visualizing workflow and pipeline
structures using different layout algorithms.

Note: Uses DAG (acyclic) workflows since Sugiyama requires acyclic graphs.
For cyclic state machines, use force-directed layouts.

Usage:
    uv run python scripts/state_machine_showcase.py

Output:
    build/state_machine_showcase.html
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from graph_layout import (
    CircularLayout,
    ForceAtlas2Layout,
    FruchtermanReingoldLayout,
    RadialTreeLayout,
    SpectralLayout,
    SpringLayout,
    SugiyamaLayout,
    YifanHuLayout,
)
from graph_layout.cola import Layout as ColaLayout

BUILD_DIR = Path(__file__).parent.parent / "build"
SVG_WIDTH = 500
SVG_HEIGHT = 450


@dataclass
class State:
    """Represents a state in the workflow."""

    id: int
    name: str
    state_type: str  # "start", "normal", "end", "decision"


@dataclass
class LayoutSpec:
    """Specification for a layout algorithm."""

    name: str
    cls: type | None
    params: dict[str, Any]
    description: str
    uses_cola_api: bool = False
    rating: str = ""


LAYOUTS: list[LayoutSpec] = [
    LayoutSpec(
        name="Sugiyama",
        cls=SugiyamaLayout,
        params={},
        description="Layered DAG - IDEAL for workflows",
        rating="BEST",
    ),
    LayoutSpec(
        name="Radial Tree",
        cls=RadialTreeLayout,
        params={},
        description="Radial from start state",
        rating="GOOD",
    ),
    LayoutSpec(
        name="Fruchterman-Reingold",
        cls=FruchtermanReingoldLayout,
        params={"iterations": 50},
        description="Force-directed layout",
        rating="FAIR",
    ),
    LayoutSpec(
        name="ForceAtlas2",
        cls=ForceAtlas2Layout,
        params={"iterations": 50, "scaling": 100.0, "gravity": 0.05},
        description="Community-aware forces",
        rating="FAIR",
    ),
    LayoutSpec(
        name="Spring",
        cls=SpringLayout,
        params={"iterations": 50},
        description="Simple spring model",
        rating="FAIR",
    ),
    LayoutSpec(
        name="Yifan Hu",
        cls=YifanHuLayout,
        params={"iterations": 50},
        description="Multilevel force-directed",
        rating="FAIR",
    ),
    LayoutSpec(
        name="Cola",
        cls=ColaLayout,
        params={"link_distance": 80},
        description="Constraint-based layout",
        uses_cola_api=True,
        rating="GOOD",
    ),
    LayoutSpec(
        name="Circular",
        cls=CircularLayout,
        params={},
        description="States on a circle",
        rating="POOR",
    ),
    LayoutSpec(
        name="Spectral",
        cls=SpectralLayout,
        params={},
        description="Eigenvector embedding",
        rating="POOR",
    ),
]


def generate_order_workflow() -> tuple[list[State], list[dict]]:
    """Generate an e-commerce order processing workflow (DAG)."""
    states = [
        State(0, "Cart", "start"),
        State(1, "Checkout", "normal"),
        State(2, "Payment", "decision"),
        State(3, "Processing", "normal"),
        State(4, "Shipping", "normal"),
        State(5, "Delivered", "end"),
        State(6, "Failed", "end"),
        State(7, "Cancelled", "end"),
    ]
    links = [
        {"source": 0, "target": 1},
        {"source": 1, "target": 2},
        {"source": 1, "target": 7},  # Abandon cart
        {"source": 2, "target": 3},  # Payment success
        {"source": 2, "target": 6},  # Payment failed
        {"source": 3, "target": 4},
        {"source": 4, "target": 5},
    ]
    return states, links


def generate_approval_workflow() -> tuple[list[State], list[dict]]:
    """Generate a document approval workflow (DAG)."""
    states = [
        State(0, "Draft", "start"),
        State(1, "Submit", "normal"),
        State(2, "Review", "decision"),
        State(3, "Revise", "normal"),
        State(4, "Approve", "normal"),
        State(5, "Published", "end"),
        State(6, "Rejected", "end"),
    ]
    links = [
        {"source": 0, "target": 1},
        {"source": 1, "target": 2},
        {"source": 2, "target": 3},  # Needs revision
        {"source": 2, "target": 4},  # Approved
        {"source": 2, "target": 6},  # Rejected
        {"source": 3, "target": 1},  # Resubmit (creates cycle, remove for Sugiyama)
        {"source": 4, "target": 5},
    ]
    # Remove cycle for DAG compatibility
    links = [l for l in links if not (l["source"] == 3 and l["target"] == 1)]
    links.append({"source": 3, "target": 6})  # Revise -> Rejected instead
    return states, links


def generate_ci_pipeline() -> tuple[list[State], list[dict]]:
    """Generate a CI/CD pipeline workflow (DAG)."""
    states = [
        State(0, "Push", "start"),
        State(1, "Build", "normal"),
        State(2, "Test", "decision"),
        State(3, "Deploy Stage", "normal"),
        State(4, "Deploy Prod", "normal"),
        State(5, "Success", "end"),
        State(6, "Failed", "end"),
    ]
    links = [
        {"source": 0, "target": 1},
        {"source": 1, "target": 2},
        {"source": 1, "target": 6},  # Build failed
        {"source": 2, "target": 3},  # Tests pass
        {"source": 2, "target": 6},  # Tests fail
        {"source": 3, "target": 4},
        {"source": 4, "target": 5},
        {"source": 4, "target": 6},  # Deploy failed
    ]
    return states, links


def generate_support_ticket() -> tuple[list[State], list[dict]]:
    """Generate a support ticket workflow (DAG)."""
    states = [
        State(0, "New", "start"),
        State(1, "Triage", "normal"),
        State(2, "Assigned", "normal"),
        State(3, "In Progress", "normal"),
        State(4, "Pending", "decision"),
        State(5, "Resolved", "end"),
        State(6, "Closed", "end"),
    ]
    links = [
        {"source": 0, "target": 1},
        {"source": 1, "target": 2},
        {"source": 1, "target": 6},  # Spam/duplicate
        {"source": 2, "target": 3},
        {"source": 3, "target": 4},
        {"source": 4, "target": 5},  # Customer confirms
        {"source": 4, "target": 3},  # More work needed (remove for DAG)
        {"source": 5, "target": 6},
    ]
    # Remove cycle
    links = [l for l in links if not (l["source"] == 4 and l["target"] == 3)]
    return states, links


def run_layout(spec: LayoutSpec, states: list[State], links: list[dict]) -> Any:
    """Run a layout algorithm."""
    random.seed(42)
    node_data = [{} for _ in states]
    link_data = [{"source": l["source"], "target": l["target"]} for l in links]

    if spec.uses_cola_api:
        node_data = [
            {
                "x": random.uniform(100, SVG_WIDTH - 100),
                "y": random.uniform(100, SVG_HEIGHT - 100),
                "width": 30,
                "height": 30,
            }
            for _ in states
        ]
        layout = ColaLayout()
        layout.nodes(node_data).links(link_data).size([SVG_WIDTH, SVG_HEIGHT])
        layout.link_distance(spec.params.get("link_distance", 80))
        layout.start(30, 0, 30, 0, False)
        return layout

    layout = spec.cls(
        nodes=node_data,
        links=link_data,
        size=(SVG_WIDTH, SVG_HEIGHT),
        **spec.params,
    )
    layout.run()
    return layout


def get_state_color(state_type: str) -> str:
    """Get color based on state type."""
    return {
        "start": "#27ae60",
        "normal": "#3498db",
        "decision": "#f39c12",
        "end": "#e74c3c",
    }.get(state_type, "#95a5a6")


def get_rating_color(rating: str) -> str:
    """Get badge color based on rating."""
    return {
        "BEST": "#27ae60",
        "GOOD": "#3498db",
        "FAIR": "#f39c12",
        "POOR": "#e74c3c",
    }.get(rating, "#95a5a6")


def center_layout(nodes: list, padding: int = 40) -> None:
    """Scale and center the layout within the SVG canvas."""
    if not nodes:
        return

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


def draw_arrow(x1: float, y1: float, x2: float, y2: float, color: str = "#7f8c8d") -> str:
    """Draw a line with an arrowhead."""
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx * dx + dy * dy)
    if length == 0:
        return ""

    dx /= length
    dy /= length

    node_radius = 25
    x1_adj = x1 + dx * node_radius
    y1_adj = y1 + dy * node_radius
    x2_adj = x2 - dx * node_radius
    y2_adj = y2 - dy * node_radius

    arrow_size = 8
    angle = math.atan2(dy, dx)
    arrow_angle = math.pi / 6

    ax1 = x2_adj - arrow_size * math.cos(angle - arrow_angle)
    ay1 = y2_adj - arrow_size * math.sin(angle - arrow_angle)
    ax2 = x2_adj - arrow_size * math.cos(angle + arrow_angle)
    ay2 = y2_adj - arrow_size * math.sin(angle + arrow_angle)

    return (
        f'<line x1="{x1_adj:.1f}" y1="{y1_adj:.1f}" '
        f'x2="{x2_adj:.1f}" y2="{y2_adj:.1f}" '
        f'stroke="{color}" stroke-width="2"/>'
        f'<polygon points="{x2_adj:.1f},{y2_adj:.1f} {ax1:.1f},{ay1:.1f} {ax2:.1f},{ay2:.1f}" '
        f'fill="{color}"/>'
    )


def layout_to_svg(layout: Any, spec: LayoutSpec, states: list[State], workflow_name: str) -> str:
    """Convert a layout to SVG."""
    if spec.uses_cola_api:
        nodes = layout.nodes()
        links = layout.links()
    else:
        nodes = layout.nodes
        links = layout.links

    center_layout(nodes)

    svg_parts = [
        f'<svg width="{SVG_WIDTH}" height="{SVG_HEIGHT}" xmlns="http://www.w3.org/2000/svg">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
    ]

    # Draw edges
    for link in links:
        src_idx = link.source if isinstance(link.source, int) else link.source.index
        tgt_idx = link.target if isinstance(link.target, int) else link.target.index
        if src_idx >= len(nodes) or tgt_idx >= len(nodes):
            continue
        src = nodes[src_idx]
        tgt = nodes[tgt_idx]
        svg_parts.append(draw_arrow(src.x, src.y, tgt.x, tgt.y))

    # Draw nodes
    for i, node in enumerate(nodes):
        if i >= len(states):
            continue
        state = states[i]
        color = get_state_color(state.state_type)
        box_w, box_h = 70, 28
        rx = 14 if state.state_type in ("start", "end") else 4

        svg_parts.append(
            f'<rect x="{node.x - box_w / 2:.1f}" y="{node.y - box_h / 2:.1f}" '
            f'width="{box_w}" height="{box_h}" rx="{rx}" '
            f'fill="{color}" stroke="#fff" stroke-width="2"/>'
        )
        svg_parts.append(
            f'<text x="{node.x:.1f}" y="{node.y + 4:.1f}" '
            f'text-anchor="middle" fill="#fff" font-size="9" '
            f'font-weight="bold" font-family="sans-serif">{escape(state.name)}</text>'
        )

    # Title
    rating_color = get_rating_color(spec.rating)
    svg_parts.append(
        f'<text x="{SVG_WIDTH // 2}" y="22" text-anchor="middle" '
        f'fill="#2c3e50" font-size="14" font-weight="bold" '
        f'font-family="sans-serif">{escape(spec.name)}</text>'
    )

    badge_x = SVG_WIDTH // 2 + len(spec.name) * 4 + 20
    svg_parts.append(
        f'<rect x="{badge_x - 20}" y="10" width="40" height="16" rx="8" fill="{rating_color}"/>'
    )
    svg_parts.append(
        f'<text x="{badge_x}" y="22" text-anchor="middle" fill="#fff" '
        f'font-size="8" font-weight="bold" font-family="sans-serif">{escape(spec.rating)}</text>'
    )

    svg_parts.append(
        f'<text x="{SVG_WIDTH // 2}" y="38" text-anchor="middle" '
        f'fill="#7f8c8d" font-size="9" font-family="sans-serif">{escape(spec.description)}</text>'
    )

    svg_parts.append(
        f'<text x="{SVG_WIDTH // 2}" y="{SVG_HEIGHT - 8}" text-anchor="middle" '
        f'fill="#95a5a6" font-size="8" font-family="sans-serif">'
        f"{escape(workflow_name)} ({len(states)} states)</text>"
    )

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def generate_html(sections: list[tuple[str, str, list[str]]]) -> str:
    """Generate the HTML page."""
    html_parts = [
        """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Workflow Layout Comparison</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #ecf0f1;
            color: #2c3e50;
            line-height: 1.6;
        }
        header {
            background: linear-gradient(135deg, #2c3e50 0%, #27ae60 100%);
            color: white;
            padding: 2.5rem;
            text-align: center;
        }
        header h1 { font-size: 2.2rem; margin-bottom: 0.5rem; }
        header p { opacity: 0.9; font-size: 1.1rem; }
        main { max-width: 1800px; margin: 0 auto; padding: 2rem; }
        .intro {
            background: white;
            border-radius: 12px;
            padding: 1.5rem 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .intro h3 { color: #27ae60; margin-bottom: 1rem; }
        .rating-guide, .state-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin-top: 1rem;
        }
        .rating-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.4rem 1rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        .rating-best { background: #27ae60; color: white; }
        .rating-good { background: #3498db; color: white; }
        .rating-fair { background: #f39c12; color: white; }
        .rating-poor { background: #e74c3c; color: white; }
        .state-item { display: flex; align-items: center; gap: 0.5rem; }
        .state-box { width: 20px; height: 14px; border-radius: 3px; }
        section { margin-bottom: 3rem; }
        section h2 {
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 3px solid #27ae60;
        }
        section > p { color: #7f8c8d; margin-bottom: 1.5rem; }
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
        .graph-card svg { display: block; width: 100%; height: auto; }
        footer { text-align: center; padding: 2rem; color: #7f8c8d; font-size: 0.9rem; }
        .recommendation {
            background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
            color: white;
            padding: 1.5rem 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
        }
        .recommendation h3 { margin-bottom: 0.5rem; }
        .recommendation ul { margin-left: 1.5rem; }
    </style>
</head>
<body>
    <header>
        <h1>Workflow Layout Comparison</h1>
        <p>Finding the best graph layout algorithm for pipelines and state diagrams</p>
    </header>
    <main>
        <div class="intro">
            <h3>Understanding the Ratings</h3>
            <p>Each layout is rated on how well it shows workflow progression:</p>
            <div class="rating-guide">
                <span class="rating-badge rating-best">BEST - Clear flow direction</span>
                <span class="rating-badge rating-good">GOOD - Readable structure</span>
                <span class="rating-badge rating-fair">FAIR - Usable but cluttered</span>
                <span class="rating-badge rating-poor">POOR - Loses flow info</span>
            </div>
            <h3 style="margin-top: 1.5rem;">State Types (by color)</h3>
            <div class="state-legend">
                <div class="state-item">
                    <span class="state-box" style="background: #27ae60; border-radius: 7px;"></span>
                    <span>Start</span>
                </div>
                <div class="state-item">
                    <span class="state-box" style="background: #3498db;"></span>
                    <span>Normal</span>
                </div>
                <div class="state-item">
                    <span class="state-box" style="background: #f39c12;"></span>
                    <span>Decision</span>
                </div>
                <div class="state-item">
                    <span class="state-box" style="background: #e74c3c; border-radius: 7px;"></span>
                    <span>End</span>
                </div>
            </div>
        </div>
        <div class="recommendation">
            <h3>Recommendation</h3>
            <p>For workflows and pipelines:</p>
            <ul>
                <li><strong>Sugiyama</strong> - Best for showing flow direction (top-to-bottom)</li>
                <li><strong>Cola</strong> - Good for constraint satisfaction</li>
                <li><strong>Radial Tree</strong> - Good for tree-like workflows</li>
            </ul>
        </div>
"""
    ]

    for title, desc, svgs in sections:
        html_parts.append(f"        <section>\n            <h2>{escape(title)}</h2>")
        html_parts.append(f"            <p>{escape(desc)}</p>")
        html_parts.append('            <div class="graph-grid">')
        for svg in svgs:
            card = f'                <div class="graph-card">\n{svg}\n                </div>'
            html_parts.append(card)
        html_parts.append("            </div>\n        </section>")

    html_parts.append(
        """    </main>
    <footer>
        <p>Generated by graph-layout library | Workflow Visualization Demo</p>
    </footer>
</body>
</html>"""
    )
    return "\n".join(html_parts)


def main() -> None:
    """Generate the workflow showcase HTML."""
    BUILD_DIR.mkdir(exist_ok=True)

    workflows = [
        (
            "E-Commerce Order Flow",
            "Order processing from cart to delivery",
            generate_order_workflow(),
        ),
        ("Document Approval", "Review and approval workflow", generate_approval_workflow()),
        ("CI/CD Pipeline", "Build, test, and deploy pipeline", generate_ci_pipeline()),
        ("Support Ticket", "Ticket lifecycle from new to closed", generate_support_ticket()),
    ]

    sections = []
    for name, desc, (states, links) in workflows:
        print(f"\nProcessing: {name}")
        print(f"  States: {len(states)}, Transitions: {len(links)}")

        svgs = []
        for spec in LAYOUTS:
            try:
                print(f"  Running {spec.name}...", end=" ", flush=True)
                layout = run_layout(spec, states, links)
                svg = layout_to_svg(layout, spec, states, name)
                svgs.append(svg)
                print("OK")
            except Exception as e:
                print(f"Error: {e}")

        sections.append((name, desc, svgs))

    html = generate_html(sections)
    output_path = BUILD_DIR / "state_machine_showcase.html"
    output_path.write_text(html)
    print(f"\nWorkflow showcase saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
