#!/usr/bin/env python3
"""
Benchmark layout algorithms using generated test graphs.

Usage:
    uv run python scripts/benchmark_layouts.py [--graphs PATTERN] [--algorithms ALGO,...]

Examples:
    uv run python scripts/benchmark_layouts.py
    uv run python scripts/benchmark_layouts.py --graphs "medium_*"
    uv run python scripts/benchmark_layouts.py --algorithms FR,FA2,YH
    uv run python scripts/benchmark_layouts.py --graphs "large_*" --iterations 50
"""

from __future__ import annotations

import argparse
import json
import time
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Callable

from graph_layout import (
    CircularLayout,
    ForceAtlas2Layout,
    FruchtermanReingoldLayout,
    KamadaKawaiLayout,
    RandomLayout,
    SpectralLayout,
    SpringLayout,
    YifanHuLayout,
)


def load_graph(filepath: Path) -> tuple[list[dict], list[dict]]:
    """Load graph from JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    return data["nodes"], data["links"]


def benchmark_layout(
    layout_class: type,
    nodes: list[dict],
    links: list[dict],
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Benchmark a single layout algorithm.

    Returns:
        Dict with timing and result info
    """
    # Create fresh node copies to avoid position carryover
    fresh_nodes = [{"id": n.get("id", i)} for i, n in enumerate(nodes)]

    start = time.perf_counter()
    layout = layout_class(nodes=fresh_nodes, links=links, **kwargs)
    layout.run()
    elapsed = time.perf_counter() - start

    return {
        "time_seconds": elapsed,
        "num_nodes": len(nodes),
        "num_edges": len(links),
    }


def run_benchmarks(
    graph_pattern: str = "*",
    algorithms: list[str] | None = None,
    iterations: int = 100,
    size: tuple[int, int] = (1000, 1000),
) -> list[dict]:
    """Run benchmarks on matching graphs."""

    graphs_dir = Path(__file__).parent.parent / "benchmarks" / "graphs"
    index_file = graphs_dir / "index.json"

    if not index_file.exists():
        print("No benchmark graphs found. Run generate_benchmark_graphs.py first.")
        return []

    with open(index_file) as f:
        index = json.load(f)

    # Define available algorithms
    all_algorithms: dict[str, tuple[type, dict[str, Any]]] = {
        "Random": (RandomLayout, {"random_seed": 42}),
        "Circular": (CircularLayout, {}),
        "FR": (FruchtermanReingoldLayout, {"iterations": iterations, "random_seed": 42}),
        "FR+BH": (FruchtermanReingoldLayout, {"iterations": iterations, "random_seed": 42, "use_barnes_hut": True}),
        "FA2": (ForceAtlas2Layout, {"iterations": iterations, "random_seed": 42}),
        "YH": (YifanHuLayout, {"iterations": iterations, "random_seed": 42}),
        "Spring": (SpringLayout, {"iterations": iterations, "random_seed": 42}),
        "KK": (KamadaKawaiLayout, {"iterations": min(iterations, 50), "random_seed": 42}),  # KK is slow
        "Spectral": (SpectralLayout, {}),
    }

    # Filter algorithms
    if algorithms:
        selected = {}
        for name in algorithms:
            if name in all_algorithms:
                selected[name] = all_algorithms[name]
            else:
                print(f"Warning: Unknown algorithm '{name}', skipping")
        all_algorithms = selected

    # Filter graphs
    matching_graphs = [
        g for g in index["graphs"]
        if fnmatch(g["name"], graph_pattern)
    ]

    if not matching_graphs:
        print(f"No graphs matching pattern '{graph_pattern}'")
        return []

    results = []

    print(f"\nBenchmarking {len(all_algorithms)} algorithms on {len(matching_graphs)} graphs")
    print(f"Iterations: {iterations}, Canvas: {size[0]}x{size[1]}")
    print("=" * 80)

    for graph_info in matching_graphs:
        name = graph_info["name"]
        filepath = graphs_dir / f"{name}.json"

        print(f"\n{name}: {graph_info['num_nodes']} nodes, {graph_info['num_edges']} edges")
        print("-" * 60)

        nodes, links = load_graph(filepath)

        for algo_name, (layout_class, kwargs) in all_algorithms.items():
            # Skip slow algorithms on large graphs
            if graph_info["num_nodes"] > 500 and algo_name == "KK":
                print(f"  {algo_name:12s}: SKIPPED (too slow for large graphs)")
                continue
            if graph_info["num_nodes"] > 2000 and algo_name in ["FR", "Spring"]:
                print(f"  {algo_name:12s}: SKIPPED (O(n^2) too slow, use Barnes-Hut)")
                continue

            try:
                result = benchmark_layout(
                    layout_class, nodes, links,
                    size=size, **kwargs
                )
                print(f"  {algo_name:12s}: {result['time_seconds']:.4f}s")

                results.append({
                    "graph": name,
                    "algorithm": algo_name,
                    **result,
                })
            except Exception as e:
                print(f"  {algo_name:12s}: ERROR - {e}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY (times in seconds)")
    print("=" * 80)

    # Create summary table
    graph_names = [g["name"] for g in matching_graphs]
    algo_names = list(all_algorithms.keys())

    # Header
    print(f"{'Graph':<25s}", end="")
    for algo in algo_names:
        print(f"{algo:>10s}", end="")
    print()
    print("-" * (25 + 10 * len(algo_names)))

    # Data rows
    for graph in graph_names:
        print(f"{graph:<25s}", end="")
        for algo in algo_names:
            matching = [r for r in results if r["graph"] == graph and r["algorithm"] == algo]
            if matching:
                print(f"{matching[0]['time_seconds']:>10.4f}", end="")
            else:
                print(f"{'--':>10s}", end="")
        print()

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark layout algorithms")
    parser.add_argument("--graphs", default="*", help="Graph name pattern (e.g., 'medium_*')")
    parser.add_argument("--algorithms", help="Comma-separated algorithm names (e.g., 'FR,FA2,YH')")
    parser.add_argument("--iterations", type=int, default=100, help="Iterations for iterative layouts")
    parser.add_argument("--output", help="Output JSON file for results")

    args = parser.parse_args()

    algorithms = args.algorithms.split(",") if args.algorithms else None

    results = run_benchmarks(
        graph_pattern=args.graphs,
        algorithms=algorithms,
        iterations=args.iterations,
    )

    if args.output and results:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
