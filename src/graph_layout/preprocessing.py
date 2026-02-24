"""
Graph preprocessing utilities.

This module provides reusable functions for preparing graphs before layout:
- Cycle detection and removal
- Topological sorting
- Connected component detection
- Layer assignment

These utilities are used internally by layout algorithms but can also be
used directly for graph analysis and manipulation.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Optional, Sequence, TypeVar, cast

from .types import LinkLike

T = TypeVar("T")


def _default_get_source(link: Any) -> int:
    """Default function to extract source index from a link."""
    return cast(int, link["source"] if isinstance(link, dict) else link.source)


def _default_get_target(link: Any) -> int:
    """Default function to extract target index from a link."""
    return cast(int, link["target"] if isinstance(link, dict) else link.target)


# =============================================================================
# Cycle Detection and Removal
# =============================================================================


def detect_cycle(
    n: int,
    links: Sequence[LinkLike],
    get_source: Optional[Callable[[Any], int]] = None,
    get_target: Optional[Callable[[Any], int]] = None,
) -> Optional[list[int]]:
    """
    Detect if a directed graph contains a cycle.

    Uses DFS-based cycle detection. Returns the first cycle found,
    or None if the graph is acyclic.

    Args:
        n: Number of nodes
        links: List of directed edges
        get_source: Function to extract source index from link (default: link['source'])
        get_target: Function to extract target index from link (default: link['target'])

    Returns:
        List of node indices forming a cycle, or None if acyclic.

    Example:
        >>> nodes = [{} for _ in range(3)]
        >>> links = [{'source': 0, 'target': 1}, {'source': 1, 'target': 2},
        ...          {'source': 2, 'target': 0}]
        >>> cycle = detect_cycle(3, links)
        >>> cycle is not None
        True
    """
    if get_source is None:
        get_source = _default_get_source
    if get_target is None:
        get_target = _default_get_target

    # Build adjacency list
    adj: list[list[int]] = [[] for _ in range(n)]
    for link in links:
        src = get_source(link)
        tgt = get_target(link)
        if 0 <= src < n and 0 <= tgt < n:
            adj[src].append(tgt)

    # DFS states: 0=unvisited, 1=visiting, 2=visited
    state = [0] * n

    def dfs(node: int, path: list[int]) -> Optional[list[int]]:
        state[node] = 1
        path.append(node)

        for neighbor in adj[node]:
            if state[neighbor] == 1:
                # Found cycle - extract it
                cycle_start = path.index(neighbor)
                return path[cycle_start:] + [neighbor]
            elif state[neighbor] == 0:
                result = dfs(neighbor, path)
                if result is not None:
                    return result

        path.pop()
        state[node] = 2
        return None

    for start in range(n):
        if state[start] == 0:
            result = dfs(start, [])
            if result is not None:
                return result

    return None


def has_cycle(
    n: int,
    links: Sequence[LinkLike],
    get_source: Optional[Callable[[Any], int]] = None,
    get_target: Optional[Callable[[Any], int]] = None,
) -> bool:
    """
    Check if a directed graph contains any cycle.

    Args:
        n: Number of nodes
        links: List of directed edges
        get_source: Function to extract source index from link
        get_target: Function to extract target index from link

    Returns:
        True if graph contains a cycle, False otherwise.
    """
    return detect_cycle(n, links, get_source, get_target) is not None


def remove_cycles(
    n: int,
    links: Sequence[LinkLike],
    get_source: Optional[Callable[[Any], int]] = None,
    get_target: Optional[Callable[[Any], int]] = None,
) -> tuple[list[dict[str, int]], set[int]]:
    """
    Remove cycles by reversing edges to make the graph acyclic.

    Uses a DFS-based approach to find back edges and reverse them.
    This implements a greedy feedback arc set approximation.

    Args:
        n: Number of nodes
        links: List of directed edges
        get_source: Function to extract source index from link
        get_target: Function to extract target index from link

    Returns:
        Tuple of (new_links, reversed_indices) where:
        - new_links: List of edges with cycles broken (reversed edges)
        - reversed_indices: Set of indices in original links that were reversed

    Example:
        >>> links = [{'source': 0, 'target': 1}, {'source': 1, 'target': 0}]
        >>> new_links, reversed = remove_cycles(2, links)
        >>> has_cycle(2, new_links)
        False
    """
    if get_source is None:
        get_source = _default_get_source
    if get_target is None:
        get_target = _default_get_target

    # Build edge list with indices
    edges: list[tuple[int, int, int]] = []  # (src, tgt, original_index)
    for i, link in enumerate(links):
        src = get_source(link)
        tgt = get_target(link)
        if 0 <= src < n and 0 <= tgt < n:
            edges.append((src, tgt, i))

    # Build adjacency list with edge indices
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]  # (neighbor, edge_index)
    for src, tgt, idx in edges:
        adj[src].append((tgt, idx))

    # DFS to find back edges
    state = [0] * n  # 0=unvisited, 1=visiting, 2=visited
    reversed_indices: set[int] = set()

    def dfs(node: int) -> None:
        state[node] = 1

        for neighbor, edge_idx in adj[node]:
            if state[neighbor] == 1:
                # Back edge found - mark for reversal
                reversed_indices.add(edge_idx)
            elif state[neighbor] == 0:
                dfs(neighbor)

        state[node] = 2

    for start in range(n):
        if state[start] == 0:
            dfs(start)

    # Create new link list with reversed edges
    new_links: list[dict[str, int]] = []
    for i, link in enumerate(links):
        src = get_source(link)
        tgt = get_target(link)
        if i in reversed_indices:
            new_links.append({"source": tgt, "target": src})
        else:
            new_links.append({"source": src, "target": tgt})

    return new_links, reversed_indices


# =============================================================================
# Topological Sort
# =============================================================================


def topological_sort(
    n: int,
    links: Sequence[LinkLike],
    get_source: Optional[Callable[[Any], int]] = None,
    get_target: Optional[Callable[[Any], int]] = None,
) -> Optional[list[int]]:
    """
    Compute a topological ordering of nodes in a directed acyclic graph.

    Uses Kahn's algorithm (BFS-based).

    Args:
        n: Number of nodes
        links: List of directed edges
        get_source: Function to extract source index from link
        get_target: Function to extract target index from link

    Returns:
        List of node indices in topological order, or None if graph has cycles.

    Example:
        >>> links = [{'source': 0, 'target': 1}, {'source': 1, 'target': 2}]
        >>> topological_sort(3, links)
        [0, 1, 2]
    """
    if get_source is None:
        get_source = _default_get_source
    if get_target is None:
        get_target = _default_get_target

    # Build adjacency list and compute in-degrees
    adj: list[list[int]] = [[] for _ in range(n)]
    in_degree = [0] * n

    for link in links:
        src = get_source(link)
        tgt = get_target(link)
        if 0 <= src < n and 0 <= tgt < n:
            adj[src].append(tgt)
            in_degree[tgt] += 1

    # Start with nodes that have no incoming edges
    queue: deque[int] = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)

    result: list[int] = []

    while queue:
        node = queue.popleft()
        result.append(node)

        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # If not all nodes processed, graph has a cycle
    if len(result) != n:
        return None

    return result


# =============================================================================
# Connected Components
# =============================================================================


def connected_components(
    n: int,
    links: Sequence[LinkLike],
    get_source: Optional[Callable[[Any], int]] = None,
    get_target: Optional[Callable[[Any], int]] = None,
    directed: bool = False,
) -> list[list[int]]:
    """
    Find connected components in a graph.

    Args:
        n: Number of nodes
        links: List of edges
        get_source: Function to extract source index from link
        get_target: Function to extract target index from link
        directed: If False (default), treat graph as undirected.
                  If True, find weakly connected components.

    Returns:
        List of components, where each component is a list of node indices.

    Example:
        >>> links = [{'source': 0, 'target': 1}, {'source': 2, 'target': 3}]
        >>> components = connected_components(4, links)
        >>> len(components)
        2
    """
    if get_source is None:
        get_source = _default_get_source
    if get_target is None:
        get_target = _default_get_target

    # Build undirected adjacency list
    adj: list[list[int]] = [[] for _ in range(n)]
    for link in links:
        src = get_source(link)
        tgt = get_target(link)
        if 0 <= src < n and 0 <= tgt < n:
            adj[src].append(tgt)
            adj[tgt].append(src)  # Always add reverse for connectivity

    visited = [False] * n
    components: list[list[int]] = []

    for start in range(n):
        if visited[start]:
            continue

        # BFS to find all nodes in this component
        component: list[int] = []
        queue: deque[int] = deque([start])
        visited[start] = True

        while queue:
            node = queue.popleft()
            component.append(node)

            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        components.append(component)

    return components


def is_connected(
    n: int,
    links: Sequence[LinkLike],
    get_source: Optional[Callable[[Any], int]] = None,
    get_target: Optional[Callable[[Any], int]] = None,
) -> bool:
    """
    Check if a graph is connected.

    Args:
        n: Number of nodes
        links: List of edges
        get_source: Function to extract source index from link
        get_target: Function to extract target index from link

    Returns:
        True if graph is connected, False otherwise.
    """
    if n <= 1:
        return True
    components = connected_components(n, links, get_source, get_target)
    return len(components) == 1


# =============================================================================
# Layer Assignment
# =============================================================================


def assign_layers_longest_path(
    n: int,
    links: Sequence[LinkLike],
    get_source: Optional[Callable[[Any], int]] = None,
    get_target: Optional[Callable[[Any], int]] = None,
) -> list[list[int]]:
    """
    Assign nodes to layers using the longest path algorithm.

    This is suitable for DAGs. Nodes are assigned to the earliest layer
    possible while respecting edge directions.

    Args:
        n: Number of nodes
        links: List of directed edges
        get_source: Function to extract source index from link
        get_target: Function to extract target index from link

    Returns:
        List of layers, where each layer is a list of node indices.
        Layer 0 contains source nodes (no incoming edges).

    Example:
        >>> links = [{'source': 0, 'target': 1}, {'source': 0, 'target': 2},
        ...          {'source': 1, 'target': 3}]
        >>> layers = assign_layers_longest_path(4, links)
        >>> layers[0]  # Source node
        [0]
    """
    if get_source is None:
        get_source = _default_get_source
    if get_target is None:
        get_target = _default_get_target

    if n == 0:
        return []

    # Build adjacency lists
    outgoing: list[list[int]] = [[] for _ in range(n)]
    incoming: list[list[int]] = [[] for _ in range(n)]

    for link in links:
        src = get_source(link)
        tgt = get_target(link)
        if 0 <= src < n and 0 <= tgt < n:
            outgoing[src].append(tgt)
            incoming[tgt].append(src)

    # Find sources
    sources = [i for i in range(n) if not incoming[i]]
    if not sources:
        sources = [0]  # Fallback for cyclic graphs

    # Compute longest path from sources.
    # Cap at n-1 to guarantee termination on cyclic graphs (a DAG with
    # n nodes has a longest path of at most n-1 edges).
    max_layer = n - 1
    node_layer = [-1] * n
    queue: deque[int] = deque()

    for src in sources:
        node_layer[src] = 0
        queue.append(src)

    while queue:
        node = queue.popleft()
        for child in outgoing[node]:
            new_layer = node_layer[node] + 1
            if new_layer > node_layer[child] and new_layer <= max_layer:
                node_layer[child] = new_layer
                queue.append(child)

    # Handle disconnected nodes
    for i in range(n):
        if node_layer[i] < 0:
            node_layer[i] = 0

    # Group by layer
    max_layer = max(node_layer) if node_layer else 0
    layers: list[list[int]] = [[] for _ in range(max_layer + 1)]
    for i in range(n):
        layers[node_layer[i]].append(i)

    return layers


# =============================================================================
# Crossing Minimization
# =============================================================================


def minimize_crossings_barycenter(
    layers: list[list[int]],
    links: Sequence[LinkLike],
    iterations: int = 24,
    get_source: Optional[Callable[[Any], int]] = None,
    get_target: Optional[Callable[[Any], int]] = None,
) -> list[list[int]]:
    """
    Minimize edge crossings between layers using the barycenter heuristic.

    Repeatedly sweeps through layers, reordering nodes based on the
    average position of their neighbors in adjacent layers.

    Args:
        layers: List of layers from assign_layers_longest_path()
        links: List of directed edges
        iterations: Number of sweep iterations
        get_source: Function to extract source index from link
        get_target: Function to extract target index from link

    Returns:
        Reordered layers with minimized crossings.

    Example:
        >>> layers = [[0], [1, 2], [3]]
        >>> links = [{'source': 0, 'target': 2}, {'source': 0, 'target': 1}]
        >>> new_layers = minimize_crossings_barycenter(layers, links)
    """
    if get_source is None:
        get_source = _default_get_source
    if get_target is None:
        get_target = _default_get_target

    if len(layers) < 2:
        return [list(layer) for layer in layers]

    # Build node-to-layer mapping
    node_layer: dict[int, int] = {}
    for layer_idx, layer in enumerate(layers):
        for node in layer:
            node_layer[node] = layer_idx

    # Build adjacency lists
    max_node = max(max(layer) for layer in layers if layer) + 1
    outgoing: list[list[int]] = [[] for _ in range(max_node)]
    incoming: list[list[int]] = [[] for _ in range(max_node)]

    for link in links:
        src = get_source(link)
        tgt = get_target(link)
        if src in node_layer and tgt in node_layer:
            outgoing[src].append(tgt)
            incoming[tgt].append(src)

    # Make mutable copy
    result = [list(layer) for layer in layers]

    # Track positions
    position: dict[int, int] = {}
    for layer in result:
        for pos, node in enumerate(layer):
            position[node] = pos

    def order_layer(layer_idx: int, adj: list[list[int]]) -> None:
        layer = result[layer_idx]
        if not layer:
            return

        barycenters: list[tuple[float, int]] = []
        for node in layer:
            neighbors = adj[node]
            if neighbors:
                avg = sum(position.get(n, 0) for n in neighbors) / len(neighbors)
            else:
                avg = position.get(node, 0)
            barycenters.append((avg, node))

        barycenters.sort(key=lambda x: x[0])
        result[layer_idx] = [node for _, node in barycenters]

        for pos, (_, node) in enumerate(barycenters):
            position[node] = pos

    # Iterate with alternating sweeps
    for i in range(iterations):
        if i % 2 == 0:
            # Sweep down
            for layer_idx in range(1, len(result)):
                order_layer(layer_idx, incoming)
        else:
            # Sweep up
            for layer_idx in range(len(result) - 2, -1, -1):
                order_layer(layer_idx, outgoing)

    return result


# =============================================================================
# Graph Metrics
# =============================================================================


def count_crossings(
    layers: list[list[int]],
    links: Sequence[LinkLike],
    get_source: Optional[Callable[[Any], int]] = None,
    get_target: Optional[Callable[[Any], int]] = None,
) -> int:
    """
    Count the number of edge crossings in a layered layout.

    Args:
        layers: List of layers, each containing node indices
        links: List of directed edges
        get_source: Function to extract source index from link
        get_target: Function to extract target index from link

    Returns:
        Number of edge crossings.
    """
    if get_source is None:
        get_source = _default_get_source
    if get_target is None:
        get_target = _default_get_target

    # Build node position map
    node_layer: dict[int, int] = {}
    node_pos: dict[int, int] = {}
    for layer_idx, layer in enumerate(layers):
        for pos, node in enumerate(layer):
            node_layer[node] = layer_idx
            node_pos[node] = pos

    # Group edges by layer pairs
    layer_edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for link in links:
        src = get_source(link)
        tgt = get_target(link)
        if src in node_layer and tgt in node_layer:
            l1, l2 = node_layer[src], node_layer[tgt]
            if l1 > l2:
                l1, l2 = l2, l1
                src, tgt = tgt, src
            key = (l1, l2)
            if key not in layer_edges:
                layer_edges[key] = []
            layer_edges[key].append((node_pos[src], node_pos[tgt]))

    # Count crossings for each layer pair
    total = 0
    for edges in layer_edges.values():
        for i, (s1, t1) in enumerate(edges):
            for s2, t2 in edges[i + 1 :]:
                # Two edges cross if one is "above" on left and "below" on right
                if (s1 < s2 and t1 > t2) or (s1 > s2 and t1 < t2):
                    total += 1

    return total


def assign_layers_width_bounded(
    n: int,
    links: Sequence[LinkLike],
    max_width: int = 0,
    get_source: Optional[Callable[[Any], int]] = None,
    get_target: Optional[Callable[[Any], int]] = None,
) -> list[list[int]]:
    """
    Assign nodes to layers using BFS with width-bounded merging.

    This algorithm is designed for undirected or loosely-directed graphs
    (e.g., grids) where longest-path layering produces undesirable diagonal
    layouts. It uses BFS to discover natural distance layers, then greedily
    merges adjacent thin layers to reduce the total number of layers while
    keeping each layer's width within a bound.

    Args:
        n: Number of nodes
        links: List of edges
        max_width: Maximum layer width. If 0 (default), uses
            ceil(n / ceil(sqrt(n))).
        get_source: Function to extract source index from link
        get_target: Function to extract target index from link

    Returns:
        List of layers, where each layer is a list of node indices.
    """
    if get_source is None:
        get_source = _default_get_source
    if get_target is None:
        get_target = _default_get_target

    if n == 0:
        return []
    if n == 1:
        return [[0]]

    # Build undirected adjacency
    adj: list[list[int]] = [[] for _ in range(n)]
    for link in links:
        src = get_source(link)
        tgt = get_target(link)
        if 0 <= src < n and 0 <= tgt < n and src != tgt:
            adj[src].append(tgt)
            adj[tgt].append(src)

    # Auto max_width: ceil(n / ceil(sqrt(n)))
    if max_width <= 0:
        import math

        target_layers = math.ceil(math.sqrt(n))
        max_width = math.ceil(n / target_layers)

    # Try a few candidate roots, pick the one giving the best (fewest) layers
    # after merging.
    candidates = _pick_bfs_candidates(n, adj)

    best_layers: list[list[int]] | None = None
    best_count = n + 1  # worst case: each node in its own layer

    for root in candidates:
        bfs_layers = _bfs_layers(n, adj, root)
        merged = _merge_layers(bfs_layers, max_width)
        if len(merged) < best_count:
            best_count = len(merged)
            best_layers = merged

    assert best_layers is not None
    return best_layers


def _pick_bfs_candidates(n: int, adj: list[list[int]]) -> list[int]:
    """Pick a small set of candidate BFS roots to try."""
    candidates: list[int] = [0]

    # Highest-degree node
    best_deg = len(adj[0])
    best_node = 0
    for i in range(1, n):
        if len(adj[i]) > best_deg:
            best_deg = len(adj[i])
            best_node = i
    if best_node != 0:
        candidates.append(best_node)

    # Degree-1 nodes (periphery heuristic): pick up to 2
    for i in range(n):
        if len(adj[i]) == 1 and i not in candidates:
            candidates.append(i)
            if len(candidates) >= 4:
                break

    # BFS-diameter endpoint heuristic: BFS from node 0, take the farthest
    # node, then BFS from that node, take the farthest again.
    farthest = _bfs_farthest(n, adj, 0)
    if farthest not in candidates:
        candidates.append(farthest)
    farthest2 = _bfs_farthest(n, adj, farthest)
    if farthest2 not in candidates:
        candidates.append(farthest2)

    return candidates


def _bfs_farthest(n: int, adj: list[list[int]], start: int) -> int:
    """Return the node farthest from start via BFS."""
    dist = [-1] * n
    dist[start] = 0
    queue: deque[int] = deque([start])
    farthest = start
    while queue:
        node = queue.popleft()
        for nb in adj[node]:
            if dist[nb] < 0:
                dist[nb] = dist[node] + 1
                queue.append(nb)
                farthest = nb
    return farthest


def _bfs_layers(n: int, adj: list[list[int]], root: int) -> list[list[int]]:
    """Compute BFS distance layers from root."""
    dist = [-1] * n
    dist[root] = 0
    queue: deque[int] = deque([root])
    max_dist = 0

    while queue:
        node = queue.popleft()
        for nb in adj[node]:
            if dist[nb] < 0:
                dist[nb] = dist[node] + 1
                if dist[nb] > max_dist:
                    max_dist = dist[nb]
                queue.append(nb)

    # Handle disconnected nodes: assign to layer 0
    for i in range(n):
        if dist[i] < 0:
            dist[i] = 0

    layers: list[list[int]] = [[] for _ in range(max_dist + 1)]
    for i in range(n):
        layers[dist[i]].append(i)

    return layers


def _merge_layers(layers: list[list[int]], max_width: int) -> list[list[int]]:
    """Greedily merge adjacent layers while respecting max_width."""
    if not layers:
        return []

    merged: list[list[int]] = [list(layers[0])]
    for layer in layers[1:]:
        if len(merged[-1]) + len(layer) <= max_width:
            merged[-1].extend(layer)
        else:
            merged.append(list(layer))

    return merged


__all__ = [
    "detect_cycle",
    "has_cycle",
    "remove_cycles",
    "topological_sort",
    "connected_components",
    "is_connected",
    "assign_layers_longest_path",
    "assign_layers_width_bounded",
    "minimize_crossings_barycenter",
    "count_crossings",
]
