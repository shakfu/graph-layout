#!/usr/bin/env python3
"""
Profile Bipartite layout to identify optimization opportunities.
"""

import cProfile
import pstats
import time
from io import StringIO

from graph_layout import BipartiteLayout
from graph_layout.bipartite import count_crossings, is_bipartite


def generate_bipartite_graph(n_top: int, n_bottom: int, edge_density: float = 0.3):
    """Generate a random bipartite graph."""
    import random

    nodes = [{} for _ in range(n_top + n_bottom)]
    links = []

    for i in range(n_top):
        for j in range(n_top, n_top + n_bottom):
            if random.random() < edge_density:
                links.append({"source": i, "target": j})

    return nodes, links, list(range(n_top)), list(range(n_top, n_top + n_bottom))


def benchmark_bipartite(n_top: int, n_bottom: int, edge_density: float = 0.3):
    """Run BipartiteLayout and return elapsed time."""
    nodes, links, top_set, bottom_set = generate_bipartite_graph(n_top, n_bottom, edge_density)

    start = time.perf_counter()
    layout = BipartiteLayout(
        nodes=nodes,
        links=links,
        size=(1600, 1200),
        top_set=top_set,
        bottom_set=bottom_set,
        minimize_crossings=True,
        crossing_iterations=4,
    )
    layout.run()
    elapsed = time.perf_counter() - start

    return elapsed, len(links)


def profile_bipartite(n_top: int, n_bottom: int, edge_density: float = 0.3):
    """Profile BipartiteLayout and return stats."""
    nodes, links, top_set, bottom_set = generate_bipartite_graph(n_top, n_bottom, edge_density)

    profiler = cProfile.Profile()

    profiler.enable()
    layout = BipartiteLayout(
        nodes=nodes,
        links=links,
        size=(1600, 1200),
        top_set=top_set,
        bottom_set=bottom_set,
        minimize_crossings=True,
        crossing_iterations=4,
    )
    layout.run()
    profiler.disable()

    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(30)

    return stream.getvalue()


def benchmark_count_crossings(n_top: int, n_bottom: int, n_edges: int):
    """Benchmark the count_crossings utility function."""
    import random

    top_order = list(range(n_top))
    bottom_order = list(range(n_top, n_top + n_bottom))

    # Generate random edges
    edges = []
    for _ in range(n_edges):
        src = random.randint(0, n_top - 1)
        tgt = random.randint(n_top, n_top + n_bottom - 1)
        edges.append((src, tgt))

    start = time.perf_counter()
    crossings = count_crossings(top_order, bottom_order, edges)
    elapsed = time.perf_counter() - start

    return elapsed, crossings


def main():
    import random
    random.seed(42)

    print("=" * 70)
    print("BIPARTITE LAYOUT BENCHMARK")
    print("=" * 70)
    print()

    # Timing benchmarks
    print("TIMING BENCHMARKS (BipartiteLayout)")
    print("-" * 70)
    print(f"{'Graph':<30} {'Top':>8} {'Bottom':>8} {'Edges':>8} {'Time':>10}")
    print("-" * 70)

    test_cases = [
        (50, 50, 0.3),
        (100, 100, 0.3),
        (200, 200, 0.3),
        (500, 500, 0.2),
        (1000, 1000, 0.1),
    ]

    for n_top, n_bottom, density in test_cases:
        elapsed, n_edges = benchmark_bipartite(n_top, n_bottom, density)
        label = f"Bipartite ({n_top}x{n_bottom})"
        print(f"{label:<30} {n_top:>8} {n_bottom:>8} {n_edges:>8} {elapsed:>9.4f}s")

    print("-" * 70)
    print()

    # count_crossings benchmark
    print("COUNT_CROSSINGS BENCHMARK")
    print("-" * 70)
    print(f"{'Edges':<20} {'Time':>15} {'Crossings':>15}")
    print("-" * 70)

    for n_edges in [100, 500, 1000, 5000, 10000]:
        elapsed, crossings = benchmark_count_crossings(100, 100, n_edges)
        print(f"{n_edges:<20} {elapsed:>14.6f}s {crossings:>15}")

    print("-" * 70)
    print()

    # Detailed profiling
    print("=" * 70)
    print("DETAILED PROFILING (500x500 bipartite)")
    print("=" * 70)

    profile_output = profile_bipartite(500, 500, 0.2)
    print(profile_output)


if __name__ == "__main__":
    main()
