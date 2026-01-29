#!/usr/bin/env python3
"""
Generate random benchmark graphs in various standard formats.

This script creates random graphs for benchmarking layout algorithms.
Supports multiple graph models and output formats.

Usage:
    uv run python scripts/generate_benchmark_graphs.py

Output formats:
    - JSON: Native format for graph_layout library
    - Edge list (CSV): Simple source,target pairs
    - GraphML: XML-based standard format
    - GML: Graph Modelling Language
"""

from __future__ import annotations

import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def generate_erdos_renyi(n: int, p: float, seed: int | None = None) -> tuple[list[dict], list[dict]]:
    """
    Generate Erdős-Rényi random graph G(n, p).

    Each possible edge exists independently with probability p.

    Args:
        n: Number of nodes
        p: Edge probability (0 to 1)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (nodes, links)
    """
    if seed is not None:
        random.seed(seed)

    nodes = [{"id": i} for i in range(n)]
    links = []

    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                links.append({"source": i, "target": j})

    return nodes, links


def generate_barabasi_albert(n: int, m: int, seed: int | None = None) -> tuple[list[dict], list[dict]]:
    """
    Generate Barabási-Albert preferential attachment graph.

    Creates a scale-free network where new nodes preferentially
    attach to high-degree nodes (rich get richer).

    Args:
        n: Number of nodes
        m: Number of edges to attach from each new node
        seed: Random seed for reproducibility

    Returns:
        Tuple of (nodes, links)
    """
    if seed is not None:
        random.seed(seed)

    if m < 1 or m >= n:
        raise ValueError(f"m must be >= 1 and < n, got m={m}, n={n}")

    nodes = [{"id": i} for i in range(n)]
    links = []

    # Start with a complete graph of m+1 nodes
    degrees = [0] * n
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            links.append({"source": i, "target": j})
            degrees[i] += 1
            degrees[j] += 1

    # Add remaining nodes with preferential attachment
    for new_node in range(m + 1, n):
        # Select m targets based on degree (preferential attachment)
        targets = set()
        total_degree = sum(degrees[:new_node])

        while len(targets) < m:
            r = random.random() * total_degree
            cumulative = 0
            for candidate in range(new_node):
                cumulative += degrees[candidate]
                if cumulative > r:
                    targets.add(candidate)
                    break

        # Add edges to selected targets
        for target in targets:
            links.append({"source": new_node, "target": target})
            degrees[new_node] += 1
            degrees[target] += 1

    return nodes, links


def generate_watts_strogatz(n: int, k: int, p: float, seed: int | None = None) -> tuple[list[dict], list[dict]]:
    """
    Generate Watts-Strogatz small-world graph.

    Creates a graph with high clustering and short path lengths.

    Args:
        n: Number of nodes
        k: Each node connected to k nearest neighbors in ring (must be even)
        p: Rewiring probability
        seed: Random seed for reproducibility

    Returns:
        Tuple of (nodes, links)
    """
    if seed is not None:
        random.seed(seed)

    if k % 2 != 0:
        k = k - 1  # Make even

    nodes = [{"id": i} for i in range(n)]

    # Start with ring lattice
    edges = set()
    for i in range(n):
        for j in range(1, k // 2 + 1):
            target = (i + j) % n
            edge = (min(i, target), max(i, target))
            edges.add(edge)

    # Rewire edges with probability p
    edges_list = list(edges)
    for i, (u, v) in enumerate(edges_list):
        if random.random() < p:
            # Rewire: keep u, find new v
            new_v = random.randint(0, n - 1)
            while new_v == u or (min(u, new_v), max(u, new_v)) in edges:
                new_v = random.randint(0, n - 1)

            edges.discard((u, v))
            edges.add((min(u, new_v), max(u, new_v)))

    links = [{"source": u, "target": v} for u, v in edges]
    return nodes, links


def generate_grid(rows: int, cols: int) -> tuple[list[dict], list[dict]]:
    """
    Generate a 2D grid graph.

    Args:
        rows: Number of rows
        cols: Number of columns

    Returns:
        Tuple of (nodes, links)
    """
    n = rows * cols
    nodes = [{"id": i, "row": i // cols, "col": i % cols} for i in range(n)]
    links = []

    for i in range(rows):
        for j in range(cols):
            node = i * cols + j
            # Right neighbor
            if j < cols - 1:
                links.append({"source": node, "target": node + 1})
            # Bottom neighbor
            if i < rows - 1:
                links.append({"source": node, "target": node + cols})

    return nodes, links


# =============================================================================
# Output format writers
# =============================================================================

def save_json(nodes: list[dict], links: list[dict], filepath: Path) -> None:
    """Save graph in JSON format (native to graph_layout)."""
    data = {
        "nodes": nodes,
        "links": links,
        "metadata": {
            "num_nodes": len(nodes),
            "num_edges": len(links),
        }
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def save_edge_list(nodes: list[dict], links: list[dict], filepath: Path) -> None:
    """Save graph as edge list CSV."""
    with open(filepath, "w") as f:
        f.write("# Edge list format: source,target\n")
        f.write(f"# Nodes: {len(nodes)}, Edges: {len(links)}\n")
        for link in links:
            f.write(f"{link['source']},{link['target']}\n")


def save_graphml(nodes: list[dict], links: list[dict], filepath: Path) -> None:
    """Save graph in GraphML format."""
    # Create XML structure
    graphml = ET.Element("graphml")
    graphml.set("xmlns", "http://graphml.graphdrawing.org/xmlns")

    graph = ET.SubElement(graphml, "graph")
    graph.set("id", "G")
    graph.set("edgedefault", "undirected")

    # Add nodes
    for node in nodes:
        n = ET.SubElement(graph, "node")
        n.set("id", f"n{node['id']}")

    # Add edges
    for i, link in enumerate(links):
        e = ET.SubElement(graph, "edge")
        e.set("id", f"e{i}")
        e.set("source", f"n{link['source']}")
        e.set("target", f"n{link['target']}")

    tree = ET.ElementTree(graphml)
    ET.indent(tree, space="  ")
    tree.write(filepath, encoding="unicode", xml_declaration=True)


def save_gml(nodes: list[dict], links: list[dict], filepath: Path) -> None:
    """Save graph in GML (Graph Modelling Language) format."""
    with open(filepath, "w") as f:
        f.write("graph [\n")
        f.write("  directed 0\n")

        for node in nodes:
            f.write("  node [\n")
            f.write(f"    id {node['id']}\n")
            f.write(f"    label \"{node['id']}\"\n")
            f.write("  ]\n")

        for link in links:
            f.write("  edge [\n")
            f.write(f"    source {link['source']}\n")
            f.write(f"    target {link['target']}\n")
            f.write("  ]\n")

        f.write("]\n")


# =============================================================================
# Main generation
# =============================================================================

def generate_benchmark_suite(output_dir: Path) -> None:
    """Generate a suite of benchmark graphs in various sizes and formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define benchmark graphs
    benchmarks: list[dict[str, Any]] = [
        # Small graphs for quick tests
        {
            "name": "small_random",
            "generator": lambda: generate_erdos_renyi(100, 0.05, seed=42),
            "description": "Small Erdős-Rényi random graph (100 nodes)",
        },
        {
            "name": "small_scalefree",
            "generator": lambda: generate_barabasi_albert(100, 3, seed=42),
            "description": "Small Barabási-Albert scale-free graph (100 nodes)",
        },

        # Medium graphs for typical benchmarks
        {
            "name": "medium_random",
            "generator": lambda: generate_erdos_renyi(500, 0.01, seed=42),
            "description": "Medium Erdős-Rényi random graph (500 nodes)",
        },
        {
            "name": "medium_scalefree",
            "generator": lambda: generate_barabasi_albert(500, 3, seed=42),
            "description": "Medium Barabási-Albert scale-free graph (500 nodes)",
        },
        {
            "name": "medium_smallworld",
            "generator": lambda: generate_watts_strogatz(500, 6, 0.3, seed=42),
            "description": "Medium Watts-Strogatz small-world graph (500 nodes)",
        },

        # Large graphs for stress testing
        {
            "name": "large_random",
            "generator": lambda: generate_erdos_renyi(1000, 0.005, seed=42),
            "description": "Large Erdős-Rényi random graph (1000 nodes)",
        },
        {
            "name": "large_scalefree",
            "generator": lambda: generate_barabasi_albert(1000, 3, seed=42),
            "description": "Large Barabási-Albert scale-free graph (1000 nodes)",
        },
        {
            "name": "large_smallworld",
            "generator": lambda: generate_watts_strogatz(1000, 6, 0.3, seed=42),
            "description": "Large Watts-Strogatz small-world graph (1000 nodes)",
        },

        # Very large graphs for algorithm scaling tests
        {
            "name": "xlarge_scalefree",
            "generator": lambda: generate_barabasi_albert(5000, 3, seed=42),
            "description": "Extra-large Barabási-Albert scale-free graph (5000 nodes)",
        },
        {
            "name": "xlarge_random",
            "generator": lambda: generate_erdos_renyi(5000, 0.001, seed=42),
            "description": "Extra-large Erdős-Rényi random graph (5000 nodes)",
        },

        # Grid graphs (regular structure)
        {
            "name": "grid_small",
            "generator": lambda: generate_grid(10, 10),
            "description": "Small 10x10 grid graph (100 nodes)",
        },
        {
            "name": "grid_medium",
            "generator": lambda: generate_grid(30, 30),
            "description": "Medium 30x30 grid graph (900 nodes)",
        },
    ]

    print(f"Generating {len(benchmarks)} benchmark graphs...")
    print(f"Output directory: {output_dir}\n")

    # Create index file
    index = []

    for bench in benchmarks:
        name = bench["name"]
        print(f"Generating {name}...")

        nodes, links = bench["generator"]()

        # Save in all formats
        save_json(nodes, links, output_dir / f"{name}.json")
        save_edge_list(nodes, links, output_dir / f"{name}.csv")
        save_graphml(nodes, links, output_dir / f"{name}.graphml")
        save_gml(nodes, links, output_dir / f"{name}.gml")

        info = {
            "name": name,
            "description": bench["description"],
            "num_nodes": len(nodes),
            "num_edges": len(links),
            "avg_degree": 2 * len(links) / len(nodes) if nodes else 0,
            "formats": ["json", "csv", "graphml", "gml"],
        }
        index.append(info)

        print(f"  -> {len(nodes)} nodes, {len(links)} edges, "
              f"avg degree: {info['avg_degree']:.2f}")

    # Save index
    with open(output_dir / "index.json", "w") as f:
        json.dump({"graphs": index}, f, indent=2)

    print(f"\nGenerated {len(benchmarks)} graphs in 4 formats each.")
    print(f"Index saved to {output_dir / 'index.json'}")


def load_json_graph(filepath: Path) -> tuple[list[dict], list[dict]]:
    """Load graph from JSON format."""
    with open(filepath) as f:
        data = json.load(f)
    return data["nodes"], data["links"]


if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / "benchmarks" / "graphs"
    generate_benchmark_suite(output_dir)
