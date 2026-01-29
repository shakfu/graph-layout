#!/usr/bin/env python3
"""
Profile Kandinsky orthogonal layout to identify optimization hotspots.
"""

import cProfile
import json
import pstats
import time
from io import StringIO
from pathlib import Path

from graph_layout import KandinskyLayout


def load_graph(filepath: Path) -> tuple[list[dict], list[dict]]:
    """Load graph from JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    return data["nodes"], data["links"]


def benchmark_kandinsky(nodes: list, links: list, name: str) -> float:
    """Run Kandinsky layout and return elapsed time."""
    start = time.perf_counter()
    layout = KandinskyLayout(
        nodes=nodes,
        links=links,
        size=(1600, 1200),
        handle_crossings=True,
        optimize_bends=True,
        compact=True,
    )
    layout.run()
    elapsed = time.perf_counter() - start
    return elapsed


def profile_kandinsky(nodes: list, links: list, name: str) -> str:
    """Profile Kandinsky layout and return stats."""
    profiler = cProfile.Profile()

    profiler.enable()
    layout = KandinskyLayout(
        nodes=nodes,
        links=links,
        size=(1600, 1200),
        handle_crossings=True,
        optimize_bends=True,
        compact=True,
    )
    layout.run()
    profiler.disable()

    # Get stats
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(30)

    return stream.getvalue()


def main():
    benchmarks_dir = Path(__file__).parent.parent / "benchmarks" / "graphs"

    # Test graphs of increasing size
    test_graphs = [
        ("small_random.json", "Small Random (50 nodes)"),
        ("small_scalefree.json", "Small Scale-Free (50 nodes)"),
        ("medium_random.json", "Medium Random (200 nodes)"),
        ("medium_scalefree.json", "Medium Scale-Free (200 nodes)"),
        ("large_random.json", "Large Random (1000 nodes)"),
    ]

    print("=" * 70)
    print("KANDINSKY LAYOUT BENCHMARK")
    print("=" * 70)
    print()

    # First, run timing benchmarks
    print("TIMING BENCHMARKS")
    print("-" * 70)
    print(f"{'Graph':<35} {'Nodes':>8} {'Edges':>8} {'Time':>10}")
    print("-" * 70)

    results = []
    for filename, label in test_graphs:
        filepath = benchmarks_dir / filename
        if not filepath.exists():
            print(f"Skipping {filename} (not found)")
            continue

        nodes, links = load_graph(filepath)

        # Run benchmark
        elapsed = benchmark_kandinsky(nodes, links, label)
        results.append((label, len(nodes), len(links), elapsed))

        print(f"{label:<35} {len(nodes):>8} {len(links):>8} {elapsed:>9.3f}s")

    print("-" * 70)
    print()

    # Profile the medium graph for detailed analysis
    print("=" * 70)
    print("DETAILED PROFILING (Medium Random - 200 nodes)")
    print("=" * 70)

    medium_path = benchmarks_dir / "medium_random.json"
    if medium_path.exists():
        nodes, links = load_graph(medium_path)
        profile_output = profile_kandinsky(nodes, links, "Medium Random")
        print(profile_output)

    # Profile larger graph if timing allows
    print("=" * 70)
    print("DETAILED PROFILING (Large Random - 1000 nodes)")
    print("=" * 70)

    large_path = benchmarks_dir / "large_random.json"
    if large_path.exists():
        nodes, links = load_graph(large_path)

        # First check if it's reasonable to profile
        elapsed = benchmark_kandinsky(nodes, links, "Large Random")
        print(f"Total time: {elapsed:.3f}s")
        print()

        if elapsed < 60:  # Only profile if < 60 seconds
            profile_output = profile_kandinsky(nodes, links, "Large Random")
            print(profile_output)
        else:
            print("Skipping detailed profiling (too slow)")


if __name__ == "__main__":
    main()
