"""
Profiling script for graph-layout performance analysis.

This script profiles various graph layout scenarios to identify bottlenecks.
"""

import cProfile
import pstats
import io
from pstats import SortKey
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.graph_layout.cola.layout import Layout


def create_graph(n_nodes, n_edges, with_size=True):
    """Create a random graph with n nodes and approximately n_edges edges."""
    if with_size:
        nodes = [{'x': 0, 'y': 0, 'width': 30, 'height': 30} for _ in range(n_nodes)]
    else:
        nodes = [{'x': 0, 'y': 0} for _ in range(n_nodes)]

    # Create random edges
    edges = []
    np.random.seed(42)
    for _ in range(n_edges):
        source = np.random.randint(0, n_nodes)
        target = np.random.randint(0, n_nodes)
        if source != target:
            edges.append({'source': source, 'target': target})

    return nodes, edges


def profile_small_graph():
    """Profile a small graph (20 nodes, 30 edges)."""
    nodes, edges = create_graph(20, 30, with_size=False)

    layout = Layout()
    layout.nodes(nodes)
    layout.links(edges)
    layout.link_distance(100)
    layout.handle_disconnected(False)  # Disable to avoid node_size issues
    layout.start(50, 0, 0, 0, False)


def profile_medium_graph():
    """Profile a medium graph (100 nodes, 200 edges)."""
    nodes, edges = create_graph(100, 200, with_size=False)

    layout = Layout()
    layout.nodes(nodes)
    layout.links(edges)
    layout.link_distance(100)
    layout.handle_disconnected(False)
    layout.start(50, 0, 0, 0, False)


def profile_large_graph():
    """Profile a large graph (500 nodes, 1000 edges)."""
    nodes, edges = create_graph(500, 1000, with_size=False)

    layout = Layout()
    layout.nodes(nodes)
    layout.links(edges)
    layout.link_distance(100)
    layout.handle_disconnected(False)
    layout.start(30, 0, 0, 0, False)


def profile_with_constraints():
    """Profile layout with overlap avoidance and constraints."""
    nodes, edges = create_graph(50, 100)

    layout = Layout()
    layout.nodes(nodes)
    layout.links(edges)
    layout.link_distance(100)
    layout.avoid_overlaps(True)
    layout.start(50, 20, 20, 0, False)


def profile_with_groups():
    """Profile layout with hierarchical groups."""
    from src.graph_layout.cola.layout import Group

    nodes, edges = create_graph(60, 100)

    # Create groups
    groups = [
        Group(leaves=list(range(0, 20)), padding=10),
        Group(leaves=list(range(20, 40)), padding=10),
        Group(leaves=list(range(40, 60)), padding=10),
    ]

    layout = Layout()
    layout.nodes(nodes)
    layout.links(edges)
    layout.groups(groups)
    layout.link_distance(100)
    layout.avoid_overlaps(True)
    layout.start(50, 20, 20, 0, False)


def benchmark_scenario(name, func):
    """Benchmark a scenario and print timing."""
    print(f"\n{'='*60}")
    print(f"Profiling: {name}")
    print('='*60)

    # Create profiler
    profiler = cProfile.Profile()

    # Run with profiling
    start_time = time.time()
    profiler.enable()
    func()
    profiler.disable()
    elapsed = time.time() - start_time

    print(f"\nTotal time: {elapsed:.3f}s")

    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(20)  # Top 20 functions

    print("\nTop 20 functions by cumulative time:")
    print(s.getvalue())

    return profiler


def main():
    """Run all profiling scenarios."""
    print("graph-layout Performance Profiling")
    print("=" * 60)

    scenarios = [
        ("Small Graph (20 nodes, 30 edges)", profile_small_graph),
        ("Medium Graph (100 nodes, 200 edges)", profile_medium_graph),
        ("Large Graph (500 nodes, 1000 edges)", profile_large_graph),
        ("With Constraints (50 nodes, overlap avoidance)", profile_with_constraints),
        ("With Groups (60 nodes, 3 groups)", profile_with_groups),
    ]

    profilers = {}
    for name, func in scenarios:
        profilers[name] = benchmark_scenario(name, func)

    # Save detailed profiles
    print("\n" + "="*60)
    print("Saving detailed profiles...")
    print("="*60)

    for name, profiler in profilers.items():
        filename = f"profile_{name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}.prof"
        profiler.dump_stats(filename)
        print(f"Saved: {filename}")

    print("\nTo view detailed profile, use:")
    print("  python -m pstats <profile_file>")
    print("  then type 'stats' or 'sort cumulative' and 'stats 50'")


if __name__ == "__main__":
    main()
