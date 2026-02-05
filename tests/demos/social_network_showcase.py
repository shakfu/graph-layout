#!/usr/bin/env python3
"""
Social Network Layout Showcase - comparing layout algorithms for social graphs.

This demo generates an interactive HTML page visualizing social network structures
using different layout algorithms to demonstrate which work best for showing
communities, influence, and connections.

Usage:
    uv run python scripts/social_network_showcase.py

Output:
    build/social_network_showcase.html
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
    RadialTreeLayout,
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
class Person:
    """Represents a person in the social network."""

    id: int
    name: str
    community: int  # Which community/group they belong to
    influence: int  # Number of connections (computed later)


@dataclass
class LayoutSpec:
    """Specification for a layout algorithm."""

    name: str
    cls: type | None
    params: dict[str, Any]
    description: str
    uses_cola_api: bool = False
    rating: str = ""  # How well suited for social networks


# Define layouts with their suitability rating for social networks
LAYOUTS: list[LayoutSpec] = [
    # Force-directed - Best for social networks
    LayoutSpec(
        name="ForceAtlas2",
        cls=ForceAtlas2Layout,
        params={"iterations": 200, "scaling": 80.0, "gravity": 0.05},
        description="Community detection - IDEAL for social networks",
        rating="BEST",
    ),
    LayoutSpec(
        name="Fruchterman-Reingold",
        cls=FruchtermanReingoldLayout,
        params={"iterations": 200},
        description="Classic force-directed - good clustering",
        rating="BEST",
    ),
    LayoutSpec(
        name="Yifan Hu",
        cls=YifanHuLayout,
        params={"iterations": 200},
        description="Fast multilevel - scales to large networks",
        rating="BEST",
    ),
    LayoutSpec(
        name="Kamada-Kawai",
        cls=KamadaKawaiLayout,
        params={},
        description="Stress minimization - preserves graph distances",
        rating="GOOD",
    ),
    LayoutSpec(
        name="Spring",
        cls=SpringLayout,
        params={"iterations": 200},
        description="Simple springs - may overlap on dense graphs",
        rating="GOOD",
    ),
    # Cola - Good with overlap avoidance
    LayoutSpec(
        name="Cola",
        cls=ColaLayout,
        params={"link_distance": 60},
        description="Constraint-based with overlap avoidance",
        uses_cola_api=True,
        rating="GOOD",
    ),
    # Circular - Shows structure but loses clustering
    LayoutSpec(
        name="Circular",
        cls=CircularLayout,
        params={},
        description="Equal positioning - loses community structure",
        rating="FAIR",
    ),
    LayoutSpec(
        name="Shell (by community)",
        cls=ShellLayout,
        params={"auto_shells": 3},
        description="Concentric by influence/degree",
        rating="FAIR",
    ),
    # Spectral - Mathematical but less intuitive
    LayoutSpec(
        name="Spectral",
        cls=SpectralLayout,
        params={},
        description="Eigenvector embedding - mathematical structure",
        rating="FAIR",
    ),
    # Hierarchical - Not ideal for social networks
    LayoutSpec(
        name="Sugiyama",
        cls=SugiyamaLayout,
        params={},
        description="Hierarchical - loses peer relationships",
        rating="POOR",
    ),
    LayoutSpec(
        name="Radial Tree",
        cls=RadialTreeLayout,
        params={},
        description="Tree from most connected - distorts structure",
        rating="POOR",
    ),
]


def generate_community_network() -> tuple[list[Person], list[dict]]:
    """Generate a social network with distinct communities.

    Creates 3 communities that are densely connected internally
    but sparsely connected between each other (classic community structure).
    """
    random.seed(42)
    people = []
    links = []

    # Community names and sizes
    communities = [
        ("Tech", 10),      # Tech enthusiasts
        ("Sports", 8),     # Sports fans
        ("Music", 9),      # Music lovers
    ]

    # Names for each community
    tech_names = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack"]
    sports_names = ["Kyle", "Luna", "Mike", "Nina", "Oscar", "Pam", "Quinn", "Rosa"]
    music_names = ["Sam", "Tina", "Uma", "Vince", "Wendy", "Xander", "Yuki", "Zara", "Alex"]
    all_names = [tech_names, sports_names, music_names]

    # Create people
    person_id = 0
    community_ranges = []  # (start_id, end_id) for each community

    for comm_idx, (comm_name, size) in enumerate(communities):
        start_id = person_id
        names = all_names[comm_idx]
        for i in range(size):
            people.append(Person(person_id, names[i], comm_idx, 0))
            person_id += 1
        community_ranges.append((start_id, person_id))

    # Create intra-community links (dense)
    for comm_idx, (start, end) in enumerate(community_ranges):
        members = list(range(start, end))
        # Each person connects to ~60% of their community
        for person in members:
            others = [m for m in members if m != person]
            num_connections = int(len(others) * 0.6)
            connections = random.sample(others, num_connections)
            for other in connections:
                if person < other:  # Avoid duplicates
                    links.append({"source": person, "target": other})

    # Create inter-community links (sparse bridges)
    # These are the "weak ties" that connect communities
    bridges = [
        (0, 10),   # Alice (Tech) - Kyle (Sports)
        (2, 18),   # Carol (Tech) - Sam (Music)
        (11, 20),  # Luna (Sports) - Tina (Music)
        (5, 14),   # Frank (Tech) - Oscar (Sports)
        (8, 22),   # Ivy (Tech) - Wendy (Music)
        (15, 25),  # Pam (Sports) - Zara (Music)
    ]
    for src, tgt in bridges:
        links.append({"source": src, "target": tgt})

    # Compute influence (degree)
    degree = {p.id: 0 for p in people}
    for link in links:
        degree[link["source"]] += 1
        degree[link["target"]] += 1
    for person in people:
        person.influence = degree[person.id]

    return people, links


def generate_influencer_network() -> tuple[list[Person], list[dict]]:
    """Generate a network with influencers (hub nodes).

    Some nodes have many connections (influencers), most have few (followers).
    Power-law degree distribution.
    """
    random.seed(123)
    people = []
    links = []

    # Create influencers (high degree)
    influencers = ["Influencer1", "Influencer2", "Influencer3"]
    for i, name in enumerate(influencers):
        people.append(Person(i, name, 0, 0))  # Community 0 = influencers

    # Create regular users
    num_regular = 25
    for i in range(num_regular):
        people.append(Person(len(influencers) + i, f"User{i+1}", 1, 0))

    # Connect users to influencers (preferential attachment style)
    regular_start = len(influencers)
    for user_id in range(regular_start, len(people)):
        # Each user follows 1-3 influencers
        num_follows = random.randint(1, 3)
        followed = random.sample(range(len(influencers)), num_follows)
        for inf_id in followed:
            links.append({"source": user_id, "target": inf_id})

        # Users also connect to each other (sparse)
        if random.random() < 0.3:
            other_users = [u for u in range(regular_start, len(people)) if u != user_id]
            if other_users:
                friend = random.choice(other_users)
                if user_id < friend:
                    links.append({"source": user_id, "target": friend})

    # Compute influence
    degree = {p.id: 0 for p in people}
    for link in links:
        degree[link["source"]] += 1
        degree[link["target"]] += 1
    for person in people:
        person.influence = degree[person.id]

    return people, links


def generate_small_world_network() -> tuple[list[Person], list[dict]]:
    """Generate a small-world network (Watts-Strogatz style).

    Starts with a ring lattice and rewires some edges randomly.
    Shows the "six degrees of separation" phenomenon.
    """
    random.seed(456)
    n = 20  # Number of nodes
    k = 4   # Each node connects to k nearest neighbors
    p = 0.3  # Rewiring probability

    people = [Person(i, f"P{i+1}", i % 3, 0) for i in range(n)]
    links = []
    edge_set = set()

    # Create ring lattice
    for i in range(n):
        for j in range(1, k // 2 + 1):
            target = (i + j) % n
            if i < target:
                edge_set.add((i, target))
            else:
                edge_set.add((target, i))

    # Rewire edges
    edges_to_add = []
    edges_to_remove = []
    for src, tgt in list(edge_set):
        if random.random() < p:
            # Rewire to random node
            new_target = random.randint(0, n - 1)
            while new_target == src or (min(src, new_target), max(src, new_target)) in edge_set:
                new_target = random.randint(0, n - 1)
            edges_to_remove.append((src, tgt))
            edges_to_add.append((min(src, new_target), max(src, new_target)))

    for e in edges_to_remove:
        edge_set.discard(e)
    for e in edges_to_add:
        edge_set.add(e)

    links = [{"source": s, "target": t} for s, t in edge_set]

    # Compute influence
    degree = {p.id: 0 for p in people}
    for link in links:
        degree[link["source"]] += 1
        degree[link["target"]] += 1
    for person in people:
        person.influence = degree[person.id]

    return people, links


def run_layout(spec: LayoutSpec, people: list[Person], links: list[dict]) -> Any:
    """Run a layout algorithm and return the layout object."""
    random.seed(42)

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

    layout = spec.cls(
        nodes=node_data,
        links=link_data,
        size=(SVG_WIDTH, SVG_HEIGHT),
        **spec.params,
    )
    layout.run()
    return layout


def get_community_color(community: int) -> str:
    """Get color based on community."""
    colors = [
        "#3498db",  # Blue - Tech
        "#e74c3c",  # Red - Sports
        "#2ecc71",  # Green - Music
        "#f39c12",  # Orange
        "#9b59b6",  # Purple
    ]
    return colors[community % len(colors)]


def get_rating_color(rating: str) -> str:
    """Get badge color based on rating."""
    return {
        "BEST": "#27ae60",
        "GOOD": "#3498db",
        "FAIR": "#f39c12",
        "POOR": "#e74c3c",
    }.get(rating, "#95a5a6")


def center_layout(nodes: list, layout: Any, padding: int = 40) -> None:
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


def layout_to_svg(
    layout: Any,
    spec: LayoutSpec,
    people: list[Person],
    network_name: str,
) -> str:
    """Convert a layout to SVG with social network styling."""
    if spec.uses_cola_api:
        nodes = layout.nodes()
        links = layout.links()
    else:
        nodes = layout.nodes
        links = layout.links

    center_layout(nodes, layout)

    svg_parts = [
        f'<svg width="{SVG_WIDTH}" height="{SVG_HEIGHT}" xmlns="http://www.w3.org/2000/svg">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
    ]

    # Draw edges first (underneath nodes)
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
            f'stroke="#bdc3c7" stroke-width="1" opacity="0.5"/>'
        )

    # Draw nodes
    for i, node in enumerate(nodes):
        if i >= len(people):
            continue

        person = people[i]
        color = get_community_color(person.community)
        # Size based on influence (connections)
        base_radius = 6
        radius = base_radius + min(person.influence, 10) * 0.5

        svg_parts.append(
            f'<circle cx="{node.x:.1f}" cy="{node.y:.1f}" r="{radius:.1f}" '
            f'fill="{color}" stroke="#fff" stroke-width="1.5" opacity="0.9"/>'
        )

    # Title with rating badge
    rating_color = get_rating_color(spec.rating)
    svg_parts.append(
        f'<text x="{SVG_WIDTH // 2}" y="22" text-anchor="middle" '
        f'fill="#2c3e50" font-size="14" font-weight="bold" '
        f'font-family="sans-serif">{escape(spec.name)}</text>'
    )

    badge_x = SVG_WIDTH // 2 + len(spec.name) * 4 + 20
    svg_parts.append(
        f'<rect x="{badge_x - 20}" y="10" width="40" height="16" rx="8" '
        f'fill="{rating_color}"/>'
    )
    svg_parts.append(
        f'<text x="{badge_x}" y="22" text-anchor="middle" fill="#fff" '
        f'font-size="8" font-weight="bold" font-family="sans-serif">'
        f'{escape(spec.rating)}</text>'
    )

    svg_parts.append(
        f'<text x="{SVG_WIDTH // 2}" y="38" text-anchor="middle" '
        f'fill="#7f8c8d" font-size="9" font-family="sans-serif">'
        f'{escape(spec.description)}</text>'
    )

    svg_parts.append(
        f'<text x="{SVG_WIDTH // 2}" y="{SVG_HEIGHT - 8}" text-anchor="middle" '
        f'fill="#95a5a6" font-size="8" font-family="sans-serif">'
        f'{escape(network_name)} ({len(people)} people, {len(links)} connections)</text>'
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
    <title>Social Network Layout Comparison</title>
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
            background: linear-gradient(135deg, #9b59b6 0%, #3498db 100%);
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
            color: #9b59b6;
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
        .community-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin-top: 1rem;
        }
        .community-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .community-dot {
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
            border-bottom: 3px solid #9b59b6;
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
        <h1>Social Network Layout Comparison</h1>
        <p>Finding the best graph layout algorithm for social network visualization</p>
    </header>
    <main>
        <div class="intro">
            <h3>Understanding the Ratings</h3>
            <p>Each layout algorithm is rated on how well it reveals social network structure:</p>
            <div class="rating-guide">
                <span class="rating-badge rating-best">BEST - Shows communities clearly</span>
                <span class="rating-badge rating-good">GOOD - Readable structure</span>
                <span class="rating-badge rating-fair">FAIR - Usable but loses info</span>
                <span class="rating-badge rating-poor">POOR - Distorts relationships</span>
            </div>
            <h3 style="margin-top: 1.5rem;">Visual Encoding</h3>
            <p><strong>Node size</strong> = influence (number of connections)</p>
            <p><strong>Node color</strong> = community membership</p>
            <div class="community-legend">
                <div class="community-item">
                    <span class="community-dot" style="background: #3498db;"></span>
                    <span>Community 1</span>
                </div>
                <div class="community-item">
                    <span class="community-dot" style="background: #e74c3c;"></span>
                    <span>Community 2</span>
                </div>
                <div class="community-item">
                    <span class="community-dot" style="background: #2ecc71;"></span>
                    <span>Community 3</span>
                </div>
            </div>
        </div>

        <div class="recommendation">
            <h3>Recommendation</h3>
            <p>For social networks, use:</p>
            <ul>
                <li><strong>ForceAtlas2</strong> - Best for community detection, widely used in network science</li>
                <li><strong>Fruchterman-Reingold</strong> - Classic choice, good clustering</li>
                <li><strong>Yifan Hu</strong> - Best for large networks (1000+ nodes)</li>
                <li><strong>Cola</strong> - When you need overlap avoidance</li>
            </ul>
        </div>
"""
    ]

    for section_title, section_desc, svgs in sections:
        html_parts.append(f'        <section>\n            <h2>{escape(section_title)}</h2>')
        html_parts.append(f'            <p>{escape(section_desc)}</p>')
        html_parts.append('            <div class="graph-grid">')
        for svg in svgs:
            html_parts.append(f'                <div class="graph-card">\n{svg}\n                </div>')
        html_parts.append("            </div>\n        </section>")

    html_parts.append(
        """    </main>
    <footer>
        <p>Generated by graph-layout library | Social Network Visualization Demo</p>
    </footer>
</body>
</html>"""
    )

    return "\n".join(html_parts)


def main() -> None:
    """Generate the social network showcase HTML."""
    BUILD_DIR.mkdir(exist_ok=True)

    networks = [
        (
            "Community Network (27 people)",
            "Three distinct communities with sparse inter-community bridges - tests community detection",
            generate_community_network(),
        ),
        (
            "Influencer Network (28 people)",
            "Hub-and-spoke structure with influencers (high degree) and followers - tests scale-free handling",
            generate_influencer_network(),
        ),
        (
            "Small World Network (20 people)",
            "Watts-Strogatz model with local clustering and random shortcuts - tests 'six degrees' structure",
            generate_small_world_network(),
        ),
    ]

    sections = []

    for network_name, network_desc, (people, links) in networks:
        print(f"\nProcessing: {network_name}")
        print(f"  People: {len(people)}, Connections: {len(links)}")

        svgs = []
        for spec in LAYOUTS:
            try:
                print(f"  Running {spec.name}...")
                layout = run_layout(spec, people, links)
                svg = layout_to_svg(layout, spec, people, network_name)
                svgs.append(svg)
            except Exception as e:
                print(f"    Error: {e}")

        sections.append((network_name, network_desc, svgs))

    html = generate_html(sections)

    output_path = BUILD_DIR / "social_network_showcase.html"
    output_path.write_text(html)
    print(f"\nSocial network showcase saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
