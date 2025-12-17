"""
Profiling script for graph-layout performance analysis.

This script profiles all layout algorithms to identify bottlenecks and
compare performance across different graph sizes.
"""

import cProfile
import pstats
import io
from pstats import SortKey
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import random


def create_graph(n_nodes, n_edges, with_size=False, seed=42):
    """Create a random graph with n nodes and approximately n_edges edges."""
    random.seed(seed)
    np.random.seed(seed)

    if with_size:
        nodes = [
            {'x': random.uniform(0, 500), 'y': random.uniform(0, 500), 'width': 30, 'height': 30}
            for _ in range(n_nodes)
        ]
    else:
        nodes = [
            {'x': random.uniform(0, 500), 'y': random.uniform(0, 500)}
            for _ in range(n_nodes)
        ]

    # Create random edges
    edges = []
    for _ in range(n_edges):
        source = np.random.randint(0, n_nodes)
        target = np.random.randint(0, n_nodes)
        if source != target:
            edges.append({'source': source, 'target': target})

    return nodes, edges


# =============================================================================
# Cola Layout Profiles
# =============================================================================

def profile_cola_small():
    """Profile Cola layout: small graph (20 nodes, 30 edges)."""
    from graph_layout.cola.layout import Layout

    nodes, edges = create_graph(20, 30, with_size=False)
    layout = Layout()
    layout.nodes(nodes)
    layout.links(edges)
    layout.link_distance(100)
    layout.handle_disconnected(False)
    layout.start(50, 0, 0, 0, False)


def profile_cola_medium():
    """Profile Cola layout: medium graph (100 nodes, 200 edges)."""
    from graph_layout.cola.layout import Layout

    nodes, edges = create_graph(100, 200, with_size=False)
    layout = Layout()
    layout.nodes(nodes)
    layout.links(edges)
    layout.link_distance(100)
    layout.handle_disconnected(False)
    layout.start(50, 0, 0, 0, False)


def profile_cola_large():
    """Profile Cola layout: large graph (500 nodes, 1000 edges)."""
    from graph_layout.cola.layout import Layout

    nodes, edges = create_graph(500, 1000, with_size=False)
    layout = Layout()
    layout.nodes(nodes)
    layout.links(edges)
    layout.link_distance(100)
    layout.handle_disconnected(False)
    layout.start(30, 0, 0, 0, False)


# =============================================================================
# Fruchterman-Reingold Layout Profiles
# =============================================================================

def profile_fr_small():
    """Profile Fruchterman-Reingold: small graph (20 nodes, 30 edges)."""
    from graph_layout import FruchtermanReingoldLayout

    nodes, edges = create_graph(20, 30)
    layout = FruchtermanReingoldLayout(
        nodes=nodes, links=edges, size=(500, 500), iterations=100
    )
    layout.run()


def profile_fr_medium():
    """Profile Fruchterman-Reingold: medium graph (100 nodes, 200 edges)."""
    from graph_layout import FruchtermanReingoldLayout

    nodes, edges = create_graph(100, 200)
    layout = FruchtermanReingoldLayout(
        nodes=nodes, links=edges, size=(800, 800), iterations=100
    )
    layout.run()


def profile_fr_large():
    """Profile Fruchterman-Reingold: large graph (500 nodes, 1000 edges)."""
    from graph_layout import FruchtermanReingoldLayout

    nodes, edges = create_graph(500, 1000)
    layout = FruchtermanReingoldLayout(
        nodes=nodes, links=edges, size=(1000, 1000), iterations=50
    )
    layout.run()


def profile_fr_barnes_hut():
    """Profile Fruchterman-Reingold with Barnes-Hut: large graph (500 nodes)."""
    from graph_layout import FruchtermanReingoldLayout

    nodes, edges = create_graph(500, 1000)
    layout = FruchtermanReingoldLayout(
        nodes=nodes, links=edges, size=(1000, 1000), iterations=50,
        use_barnes_hut=True, barnes_hut_theta=0.5
    )
    layout.run()


# =============================================================================
# Kamada-Kawai Layout Profiles
# =============================================================================

def profile_kk_small():
    """Profile Kamada-Kawai: small graph (20 nodes, 30 edges)."""
    from graph_layout import KamadaKawaiLayout

    nodes, edges = create_graph(20, 30)
    layout = KamadaKawaiLayout(
        nodes=nodes, links=edges, size=(500, 500), iterations=100
    )
    layout.run()


def profile_kk_medium():
    """Profile Kamada-Kawai: medium graph (100 nodes, 200 edges)."""
    from graph_layout import KamadaKawaiLayout

    nodes, edges = create_graph(100, 200)
    layout = KamadaKawaiLayout(
        nodes=nodes, links=edges, size=(800, 800), iterations=100
    )
    layout.run()


# =============================================================================
# Spring Layout Profiles
# =============================================================================

def profile_spring_small():
    """Profile Spring layout: small graph (20 nodes, 30 edges)."""
    from graph_layout import SpringLayout

    nodes, edges = create_graph(20, 30)
    layout = SpringLayout(
        nodes=nodes, links=edges, size=(500, 500), iterations=100
    )
    layout.run()


def profile_spring_medium():
    """Profile Spring layout: medium graph (100 nodes, 200 edges)."""
    from graph_layout import SpringLayout

    nodes, edges = create_graph(100, 200)
    layout = SpringLayout(
        nodes=nodes, links=edges, size=(800, 800), iterations=100
    )
    layout.run()


# =============================================================================
# Static Layout Profiles
# =============================================================================

def profile_circular():
    """Profile Circular layout: medium graph (100 nodes)."""
    from graph_layout import CircularLayout

    nodes, edges = create_graph(100, 200)
    layout = CircularLayout(nodes=nodes, links=edges, size=(500, 500))
    layout.run()


def profile_spectral():
    """Profile Spectral layout: medium graph (100 nodes, 200 edges)."""
    from graph_layout import SpectralLayout

    nodes, edges = create_graph(100, 200)
    layout = SpectralLayout(nodes=nodes, links=edges, size=(500, 500))
    layout.run()


# =============================================================================
# Benchmarking Infrastructure
# =============================================================================

def benchmark_scenario(name, func, profile=True):
    """Benchmark a scenario and print timing."""
    print(f"\n{'-'*60}")
    print(f"  {name}")
    print('-'*60)

    if profile:
        profiler = cProfile.Profile()
        start_time = time.time()
        profiler.enable()
        func()
        profiler.disable()
        elapsed = time.time() - start_time

        # Print brief stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats(SortKey.CUMULATIVE)
        ps.print_stats(10)

        print(f"Time: {elapsed:.3f}s")
        print("\nTop 10 functions:")
        for line in s.getvalue().split('\n')[5:16]:
            if line.strip():
                print(line)

        return elapsed, profiler
    else:
        start_time = time.time()
        func()
        elapsed = time.time() - start_time
        print(f"Time: {elapsed:.3f}s")
        return elapsed, None


def main():
    """Run all profiling scenarios."""
    print("=" * 60)
    print("  graph-layout Performance Profiling")
    print("=" * 60)

    # Check Cython status
    try:
        from graph_layout import _speedups
        print("\nCython _speedups: ENABLED")
    except ImportError:
        print("\nCython _speedups: DISABLED (using pure Python)")

    from graph_layout.cola.shortestpaths import get_implementation
    print(f"Shortest paths: {get_implementation()}")

    scenarios = [
        # Cola layouts
        ("Cola: Small (20 nodes)", profile_cola_small),
        ("Cola: Medium (100 nodes)", profile_cola_medium),
        ("Cola: Large (500 nodes)", profile_cola_large),

        # Fruchterman-Reingold layouts
        ("FR: Small (20 nodes)", profile_fr_small),
        ("FR: Medium (100 nodes)", profile_fr_medium),
        ("FR: Large (500 nodes)", profile_fr_large),
        ("FR: Large + Barnes-Hut", profile_fr_barnes_hut),

        # Kamada-Kawai layouts
        ("KK: Small (20 nodes)", profile_kk_small),
        ("KK: Medium (100 nodes)", profile_kk_medium),

        # Spring layouts
        ("Spring: Small (20 nodes)", profile_spring_small),
        ("Spring: Medium (100 nodes)", profile_spring_medium),

        # Static layouts
        ("Circular (100 nodes)", profile_circular),
        ("Spectral (100 nodes)", profile_spectral),
    ]

    results = {}
    for name, func in scenarios:
        try:
            elapsed, _ = benchmark_scenario(name, func, profile=False)
            results[name] = elapsed
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = None

    # Print summary table
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"\n{'Algorithm':<35} {'Time':>10}")
    print("-" * 47)
    for name, elapsed in results.items():
        if elapsed is not None:
            print(f"{name:<35} {elapsed:>10.3f}s")
        else:
            print(f"{name:<35} {'ERROR':>10}")


if __name__ == "__main__":
    main()
