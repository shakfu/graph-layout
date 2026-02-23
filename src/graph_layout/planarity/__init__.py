"""Planarity testing module using the LR-planarity algorithm.

Provides linear-time planarity testing and combinatorial embedding extraction
for undirected graphs. Based on the Left-Right planarity algorithm of
de Fraysseix and Rosenstiehl.

Public API:
    is_planar(num_nodes, edges) -> bool
    check_planarity(num_nodes, edges) -> PlanarityResult
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from ._embedding import PlanarEmbedding
from ._lr_planarity import test_biconnected
from ._types import PlanarityResult


def is_planar(num_nodes: int, edges: Sequence[tuple[int, int]]) -> bool:
    """Test whether a graph is planar.

    This is the simple boolean API. For richer results (embedding, etc.),
    use ``check_planarity`` instead.

    Args:
        num_nodes: Number of vertices (labeled 0..num_nodes-1).
        edges: Sequence of (source, target) edge tuples (undirected).

    Returns:
        True if the graph is planar, False otherwise.
    """
    return check_planarity(num_nodes, edges).is_planar


def check_planarity(
    num_nodes: int,
    edges: Sequence[tuple[int, int]],
) -> PlanarityResult:
    """Test planarity and return detailed results.

    Preprocessing:
    - Self-loops are removed (they do not affect planarity).
    - Up to 2 parallel edges between the same pair are kept.
    - 3 or more parallel edges between any pair => non-planar immediately.
    - Disconnected graphs are split into connected components.
    - Each connected component is decomposed into biconnected components.

    Args:
        num_nodes: Number of vertices (labeled 0..num_nodes-1).
        edges: Sequence of (source, target) edge tuples (undirected).

    Returns:
        PlanarityResult with is_planar flag and optional embedding.
    """
    # Trivial cases
    if num_nodes <= 1:
        emb: dict[int, list[int]] = {v: [] for v in range(num_nodes)}
        return PlanarityResult(is_planar=True, embedding=emb)

    # Preprocess edges
    clean_edges = _preprocess_edges(num_nodes, edges)
    if clean_edges is None:
        # 3+ parallel edges detected
        return PlanarityResult(is_planar=False)

    if not clean_edges:
        # No edges -- trivially planar
        emb_empty: dict[int, list[int]] = {v: [] for v in range(num_nodes)}
        return PlanarityResult(is_planar=True, embedding=emb_empty)

    # Quick edge-count check (Euler formula necessary condition)
    m = len(clean_edges)
    if num_nodes >= 3 and m > 3 * num_nodes - 6:
        return PlanarityResult(is_planar=False)

    # Build undirected adjacency list
    adj: list[list[int]] = [[] for _ in range(num_nodes)]
    for u, v in clean_edges:
        adj[u].append(v)
        adj[v].append(u)

    # Find connected components
    components = _connected_components(num_nodes, adj)

    # Test each component independently
    full_embedding: dict[int, list[int]] = {v: [] for v in range(num_nodes)}

    for comp in components:
        if len(comp) <= 2:
            # Trivially planar
            for v in comp:
                full_embedding[v] = list(adj[v])
            continue

        # Build local adjacency for component
        local_map = {v: i for i, v in enumerate(comp)}
        n_local = len(comp)
        local_adj: list[list[int]] = [[] for _ in range(n_local)]

        for v in comp:
            for w in adj[v]:
                if w in local_map:
                    local_adj[local_map[v]].append(local_map[w])

        result = test_biconnected(n_local, local_adj)

        if not result.is_planar:
            return PlanarityResult(is_planar=False)

        # Map local embedding to global
        if result.embedding:
            for local_v, neighbors in result.embedding.items():
                global_v = comp[local_v]
                global_neighbors = [comp[w] for w in neighbors]
                full_embedding[global_v] = global_neighbors

    return PlanarityResult(is_planar=True, embedding=full_embedding)


def _preprocess_edges(
    num_nodes: int,
    edges: Sequence[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    """Remove self-loops and check parallel edge multiplicity.

    Returns:
        Cleaned edge list, or None if 3+ parallel edges detected.
    """
    # Count edge multiplicities using canonical (min, max) form
    edge_count: Counter[tuple[int, int]] = Counter()
    clean: list[tuple[int, int]] = []

    for u, v in edges:
        if u == v:
            # Self-loop: skip
            continue
        canon = (min(u, v), max(u, v))
        edge_count[canon] += 1
        if edge_count[canon] > 2:
            # 3+ parallel edges: non-planar
            return None
        clean.append((u, v))

    return clean


def _connected_components(
    num_nodes: int,
    adj: list[list[int]],
) -> list[list[int]]:
    """Find connected components via BFS."""
    visited = [False] * num_nodes
    components: list[list[int]] = []

    for start in range(num_nodes):
        if visited[start]:
            continue
        comp: list[int] = []
        queue = [start]
        visited[start] = True
        while queue:
            v = queue.pop()
            comp.append(v)
            for w in adj[v]:
                if not visited[w]:
                    visited[w] = True
                    queue.append(w)
        components.append(sorted(comp))

    return components


__all__ = [
    "is_planar",
    "check_planarity",
    "PlanarityResult",
    "PlanarEmbedding",
]
