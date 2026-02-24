"""
Efficient min-cost flow solver using successive shortest paths with
Dijkstra + Johnson potentials.

Replaces the Bellman-Ford-based solver for orthogonalization bend
minimization.  All original arc costs are non-negative ({0, 1}), so
initial potentials are zero and no Bellman-Ford bootstrap is needed.

Complexity: O(S * m * log n) where S = total supply (O(V)), m ~ 11V,
n ~ 3V for planar graphs => O(V^2 log V).
"""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .orthogonalization import FlowNetwork

_INF = float("inf")


def _dijkstra_reduced(
    source: int,
    graph: list[list[int]],
    head: list[int],
    cap: list[int],
    cost: list[int],
    pi: list[float],
    n: int,
) -> tuple[list[float], list[int]]:
    """Dijkstra on reduced costs.  Returns (dist, parent_arc) arrays."""
    dist: list[float] = [_INF] * n
    parent_arc: list[int] = [-1] * n
    dist[source] = 0.0

    # (reduced_cost, node)
    heap: list[tuple[float, int]] = [(0.0, source)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for arc_idx in graph[u]:
            if cap[arc_idx] <= 0:
                continue
            v = head[arc_idx]
            # reduced cost = cost[arc] + pi[u] - pi[v]
            rc = cost[arc_idx] + pi[u] - pi[v]
            nd = d + rc
            if nd < dist[v]:
                dist[v] = nd
                parent_arc[v] = arc_idx
                heapq.heappush(heap, (nd, v))

    return dist, parent_arc


def solve_min_cost_flow(network: FlowNetwork) -> bool:
    """
    Solve min-cost flow on *network* using successive shortest paths
    with Dijkstra and Johnson potentials.

    Reads ``network.arcs``, ``network.supplies``, ``network.num_vertices``,
    ``network.faces``.  Writes the solution into ``network.flow``.

    Returns True if a feasible flow was found (all supply / demand
    satisfied), False otherwise.
    """
    # -- Collect all node IDs and map to dense 0..n-1 ----------------------
    node_set: set[int] = set()
    for v in range(network.num_vertices):
        node_set.add(v)
    for face in network.faces:
        node_set.add(network.num_vertices + face.index)

    node_list = sorted(node_set)
    node_to_idx: dict[int, int] = {nd: i for i, nd in enumerate(node_list)}
    n = len(node_list)

    # -- Build arc-indexed residual graph ----------------------------------
    # Forward arc at index 2k, backward arc at 2k+1 (XOR trick).
    head: list[int] = []
    cap: list[int] = []
    cost_arr: list[int] = []
    graph: list[list[int]] = [[] for _ in range(n)]

    # Map from original arc -> forward arc index (for writing back flow)
    arc_fwd_idx: dict[tuple[int, int], int] = {}

    for (u_orig, v_orig), (c, w) in network.arcs.items():
        u = node_to_idx[u_orig]
        v = node_to_idx[v_orig]
        fwd = len(head)
        # forward arc
        head.append(v)
        cap.append(c)
        cost_arr.append(w)
        graph[u].append(fwd)
        # backward arc (initially zero capacity)
        head.append(u)
        cap.append(0)
        cost_arr.append(-w)
        graph[v].append(fwd + 1)

        arc_fwd_idx[(u_orig, v_orig)] = fwd

    # -- Supplies / demands ------------------------------------------------
    supply: list[int] = [0] * n
    for nd, s in network.supplies.items():
        if nd in node_to_idx:
            supply[node_to_idx[nd]] = s

    # Separate into excess (supply > 0) and deficit (supply < 0)
    excess_nodes: list[tuple[int, int]] = []  # (idx, amount)
    deficit_nodes: list[tuple[int, int]] = []
    for i in range(n):
        if supply[i] > 0:
            excess_nodes.append((i, supply[i]))
        elif supply[i] < 0:
            deficit_nodes.append((i, -supply[i]))

    # -- Add a super-source and super-sink ---------------------------------
    # Super-source (S) connects to all excess nodes; all deficit nodes
    # connect to super-sink (T).  Arcs have cost 0 and capacity = |supply|.
    S = n
    T = n + 1
    n2 = n + 2
    graph.append([])  # S
    graph.append([])  # T

    total_supply = 0
    for idx, amt in excess_nodes:
        fwd = len(head)
        head.append(idx)
        cap.append(amt)
        cost_arr.append(0)
        graph[S].append(fwd)
        head.append(S)
        cap.append(0)
        cost_arr.append(0)
        graph[idx].append(fwd + 1)
        total_supply += amt

    total_demand = 0
    for idx, amt in deficit_nodes:
        fwd = len(head)
        head.append(T)
        cap.append(amt)
        cost_arr.append(0)
        graph[idx].append(fwd)
        head.append(idx)
        cap.append(0)
        cost_arr.append(0)
        graph[T].append(fwd + 1)
        total_demand += amt

    # -- Johnson potentials (all original costs >= 0 => init to 0) ---------
    pi: list[float] = [0.0] * n2

    # -- Successive shortest paths from S to T -----------------------------
    flow_sent = 0

    while flow_sent < min(total_supply, total_demand):
        dist, parent_arc = _dijkstra_reduced(S, graph, head, cap, cost_arr, pi, n2)

        if dist[T] >= _INF:
            break  # no augmenting path

        # Trace path T -> S and find bottleneck
        bottleneck = min(total_supply, total_demand) - flow_sent
        v = T
        while v != S:
            a = parent_arc[v]
            bottleneck = min(bottleneck, cap[a])
            v = head[a ^ 1]  # tail of arc a

        if bottleneck <= 0:
            break

        # Augment flow along path
        v = T
        while v != S:
            a = parent_arc[v]
            cap[a] -= bottleneck
            cap[a ^ 1] += bottleneck
            v = head[a ^ 1]

        flow_sent += bottleneck

        # Update potentials
        for i in range(n2):
            if dist[i] < _INF:
                pi[i] += dist[i]

    # -- Write flow back to network ----------------------------------------
    network.flow = {}
    for (u_orig, v_orig), fwd in arc_fwd_idx.items():
        _, orig_cap = network.arcs[(u_orig, v_orig)]
        # flow on forward arc = original_capacity - residual_capacity
        f = network.arcs[(u_orig, v_orig)][0] - cap[fwd]
        network.flow[(u_orig, v_orig)] = f

    feasible = flow_sent >= min(total_supply, total_demand)
    return feasible


__all__ = ["solve_min_cost_flow"]
