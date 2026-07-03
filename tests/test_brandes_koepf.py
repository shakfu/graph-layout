"""Tests for Brandes-Köpf horizontal coordinate assignment."""

import random

from graph_layout.hierarchical._brandes_koepf import assign_x


def _build_random_proper_graph(seed: int):
    """A random proper layered graph with some inner (dummy) segments."""
    rng = random.Random(seed)
    n_layers = rng.randint(2, 6)
    layers = []
    nid = 0
    for _ in range(n_layers):
        w = rng.randint(1, 6)
        layers.append(list(range(nid, nid + w)))
        nid += w
    pos = {v: p for layer in layers for p, v in enumerate(layer)}
    upper = {v: [] for layer in layers for v in layer}
    lower = {v: [] for layer in layers for v in layer}
    for i in range(n_layers - 1):
        for u in layers[i]:
            for v in layers[i + 1]:
                if rng.random() < 0.3:
                    lower[u].append(v)
                    upper[v].append(u)
    for v in upper:
        upper[v].sort(key=lambda x: pos[x])
        lower[v].sort(key=lambda x: pos[x])
    is_dummy = set()
    for layer in layers[1:-1]:
        for v in layer:
            if len(upper[v]) == 1 and len(lower[v]) == 1 and rng.random() < 0.5:
                is_dummy.add(v)
    return layers, pos, upper, lower, is_dummy


class TestBrandesKopf:
    def test_no_overlap_invariant_random(self):
        """Within every layer, order is preserved and vertices keep >= delta
        separation, across many random proper layered graphs."""
        delta = 10.0
        for seed in range(500):
            layers, pos, upper, lower, is_dummy = _build_random_proper_graph(seed)
            x = assign_x(layers, pos, upper, lower, is_dummy, delta)
            for layer in layers:
                for a, b in zip(layer, layer[1:]):
                    assert x[b] - x[a] >= delta - 1e-6, f"seed {seed}: {a},{b} too close"

    def test_long_chain_is_straightened(self):
        """A chain of dummy vertices spanning several layers is aligned into a
        straight vertical line (all share one x-coordinate)."""
        # Layers: 0 -> d1 -> d2 -> 3 is the long-edge chain; 10/1/2/20 a parallel
        # chain so the alignment has something to separate against.
        layers = [[0, 10], [11, 1], [12, 2], [3, 20]]
        pos = {v: p for layer in layers for p, v in enumerate(layer)}
        upper = {v: [] for layer in layers for v in layer}
        lower = {v: [] for layer in layers for v in layer}

        def edge(u, v):
            lower[u].append(v)
            upper[v].append(u)

        edge(0, 11)
        edge(11, 12)
        edge(12, 3)  # the long-edge dummy chain
        edge(10, 1)
        edge(1, 2)
        edge(2, 20)  # parallel chain
        for v in upper:
            upper[v].sort(key=lambda x: pos[x])
            lower[v].sort(key=lambda x: pos[x])

        x = assign_x(layers, pos, upper, lower, {11, 12}, 10.0)
        chain = [0, 11, 12, 3]
        assert len({round(x[c], 6) for c in chain}) == 1

    def test_empty(self):
        assert assign_x([], {}, {}, {}, set(), 10.0) == {}
