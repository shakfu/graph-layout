#!/usr/bin/env python3
"""
Visualization script for graph layout algorithms.

Generates images for all layout algorithms into ./build/

Usage:
    uv run python scripts/visualize.py
"""

from pathlib import Path

import matplotlib.pyplot as plt

from graph_layout.circular import CircularLayout, ShellLayout
from graph_layout.cola import Layout as ColaLayout
from graph_layout.force import FruchtermanReingoldLayout, KamadaKawaiLayout, SpringLayout
from graph_layout.hierarchical import ReingoldTilfordLayout, SugiyamaLayout
from graph_layout.spectral import SpectralLayout

# Output directory
BUILD_DIR = Path(__file__).parent.parent / "build"


def ensure_build_dir():
    """Create build directory if it doesn't exist."""
    BUILD_DIR.mkdir(exist_ok=True)


def visualize(layout, title="Graph Layout", ax=None):
    """Visualize a completed layout on an axis."""
    nodes = layout.nodes()
    links = layout.links()

    # Draw edges
    for link in links:
        src_idx = link.source if isinstance(link.source, int) else link.source.index
        tgt_idx = link.target if isinstance(link.target, int) else link.target.index
        ax.plot(
            [nodes[src_idx].x, nodes[tgt_idx].x],
            [nodes[src_idx].y, nodes[tgt_idx].y],
            "gray",
            alpha=0.5,
            linewidth=1,
        )

    # Draw nodes
    xs = [n.x for n in nodes]
    ys = [n.y for n in nodes]
    ax.scatter(xs, ys, s=100, c="steelblue", zorder=5, edgecolors="white", linewidth=1)

    # Label nodes
    for i, n in enumerate(nodes):
        ax.annotate(
            str(i), (n.x, n.y), ha="center", va="center", fontsize=8, color="white"
        )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")


def save_layout(LayoutClass, name, nodes, links, filename):
    """Generate and save a single layout image."""
    import random
    random.seed(42)  # Reproducible

    # Create fresh copies - Cola needs width/height and random initial positions
    if LayoutClass is ColaLayout:
        node_data = [
            {"x": random.uniform(100, 500), "y": random.uniform(100, 500), "width": 20, "height": 20}
            for _ in range(len(nodes))
        ]
    else:
        node_data = [{"x": 0, "y": 0} for _ in range(len(nodes))]
    link_data = [{"source": l["source"], "target": l["target"]} for l in links]

    layout = LayoutClass()
    layout.nodes(node_data).links(link_data).size([600, 600])

    # Algorithm-specific configuration
    if LayoutClass is ColaLayout:
        layout.link_distance(80)
        layout.start(20, 0, 20, 0, False)
    elif LayoutClass is ShellLayout:
        layout.auto_shells(2)  # Show concentric rings
        layout.start()
    else:
        layout.start()

    fig, ax = plt.subplots(figsize=(8, 8))
    visualize(layout, name, ax=ax)
    plt.tight_layout()

    filepath = BUILD_DIR / filename
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def save_comparison(algorithms, nodes, links, filename, title):
    """Generate and save a comparison image."""
    n = len(algorithms)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, (LayoutClass, name) in enumerate(algorithms):
        import random
        random.seed(42)  # Reproducible

        # Create fresh copies - Cola needs width/height and random initial positions
        if LayoutClass is ColaLayout:
            node_data = [
                {"x": random.uniform(100, 500), "y": random.uniform(100, 500), "width": 20, "height": 20}
                for _ in range(len(nodes))
            ]
        else:
            node_data = [{"x": 0, "y": 0} for _ in range(len(nodes))]
        link_data = [{"source": l["source"], "target": l["target"]} for l in links]

        layout = LayoutClass()
        layout.nodes(node_data).links(link_data).size([600, 600])

        # Algorithm-specific configuration
        if LayoutClass is ColaLayout:
            layout.link_distance(80)
            layout.start(20, 0, 20, 0, False)
        elif LayoutClass is ShellLayout:
            layout.auto_shells(2)  # Show concentric rings
            layout.start()
        else:
            layout.start()

        visualize(layout, name, ax=axes[i])

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    filepath = BUILD_DIR / filename
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def create_sample_graph():
    """Create a sample graph for demonstration."""
    nodes = [{"x": 0, "y": 0} for _ in range(12)]
    links = [
        # Outer ring
        {"source": 0, "target": 1},
        {"source": 1, "target": 2},
        {"source": 2, "target": 3},
        {"source": 3, "target": 4},
        {"source": 4, "target": 5},
        {"source": 5, "target": 0},
        # Inner connections
        {"source": 0, "target": 6},
        {"source": 1, "target": 7},
        {"source": 2, "target": 8},
        {"source": 3, "target": 9},
        {"source": 4, "target": 10},
        {"source": 5, "target": 11},
        # Inner ring
        {"source": 6, "target": 7},
        {"source": 7, "target": 8},
        {"source": 8, "target": 9},
        {"source": 9, "target": 10},
        {"source": 10, "target": 11},
        {"source": 11, "target": 6},
    ]
    return nodes, links


def create_tree_graph():
    """Create a tree graph for hierarchical layouts."""
    nodes = [{"x": 0, "y": 0} for _ in range(15)]
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


def generate_all():
    """Generate all visualization images."""
    ensure_build_dir()

    sample_nodes, sample_links = create_sample_graph()
    tree_nodes, tree_links = create_tree_graph()

    # Individual layouts with sample graph
    print("Generating individual layout images (sample graph)...")

    # Cola (constraint-based)
    save_layout(ColaLayout, "Cola (Constraint-Based)", sample_nodes, sample_links, "cola.png")

    force_layouts = [
        (FruchtermanReingoldLayout, "Fruchterman-Reingold", "fruchterman_reingold.png"),
        (KamadaKawaiLayout, "Kamada-Kawai", "kamada_kawai.png"),
        (SpringLayout, "Spring", "spring.png"),
        (SpectralLayout, "Spectral", "spectral.png"),
    ]
    for LayoutClass, name, filename in force_layouts:
        save_layout(LayoutClass, name, sample_nodes, sample_links, filename)

    circular_layouts = [
        (CircularLayout, "Circular", "circular.png"),
        (ShellLayout, "Shell", "shell.png"),
    ]
    for LayoutClass, name, filename in circular_layouts:
        save_layout(LayoutClass, name, sample_nodes, sample_links, filename)

    # Individual layouts with tree graph
    print("Generating individual layout images (tree graph)...")
    hierarchical_layouts = [
        (ReingoldTilfordLayout, "Reingold-Tilford", "reingold_tilford.png"),
        (SugiyamaLayout, "Sugiyama", "sugiyama.png"),
    ]
    for LayoutClass, name, filename in hierarchical_layouts:
        save_layout(LayoutClass, name, tree_nodes, tree_links, filename)

    # Comparison images
    print("Generating comparison images...")

    # Force-directed comparison
    save_comparison(
        [(c, n) for c, n, _ in force_layouts],
        sample_nodes,
        sample_links,
        "comparison_force_directed.png",
        "Force-Directed Layouts",
    )

    # Circular comparison
    save_comparison(
        [(c, n) for c, n, _ in circular_layouts],
        sample_nodes,
        sample_links,
        "comparison_circular.png",
        "Circular Layouts",
    )

    # Hierarchical comparison
    save_comparison(
        [(c, n) for c, n, _ in hierarchical_layouts],
        tree_nodes,
        tree_links,
        "comparison_hierarchical.png",
        "Hierarchical Layouts",
    )

    # All non-hierarchical layouts
    all_general = [
        (ColaLayout, "Cola"),
        (FruchtermanReingoldLayout, "Fruchterman-Reingold"),
        (KamadaKawaiLayout, "Kamada-Kawai"),
        (SpringLayout, "Spring"),
        (SpectralLayout, "Spectral"),
        (CircularLayout, "Circular"),
        (ShellLayout, "Shell"),
    ]
    save_comparison(
        all_general,
        sample_nodes,
        sample_links,
        "comparison_all_general.png",
        "All Layout Algorithms",
    )

    print()
    print(f"All images saved to: {BUILD_DIR.absolute()}")


if __name__ == "__main__":
    generate_all()
