"""Tests for topological graph planarization."""

import itertools
import random

from graph_layout.orthogonal.planarization import planarize_graph
from graph_layout.planarity import check_planarity


def _complete(n):
    return n, [(i, j) for i, j in itertools.combinations(range(n), 2)]


def _complete_bipartite(m, n):
    return m + n, [(i, m + j) for i in range(m) for j in range(n)]


class TestTopologicalPlanarization:
    def test_planar_graphs_get_no_crossings(self):
        """Every planar input yields zero crossings, whatever the shape."""
        cases = [
            (4, [(0, 1), (1, 2), (2, 3), (3, 0)]),  # square
            _complete(4),  # K4
            (4, [(0, 1), (0, 2), (0, 3)]),  # star / tree
            (4, [(0, 1), (2, 3)]),  # disjoint edges (a matching)
            # cube
            (
                8,
                [
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 0),
                    (4, 5),
                    (5, 6),
                    (6, 7),
                    (7, 4),
                    (0, 4),
                    (1, 5),
                    (2, 6),
                    (3, 7),
                ],
            ),
        ]
        for n, edges in cases:
            result = planarize_graph(n, edges)
            assert len(result.crossings) == 0
            assert result.num_total_nodes == n

    def test_crossing_numbers_match_known_values(self):
        """The reinsertion recovers the known crossing numbers of small graphs."""
        assert len(planarize_graph(*_complete(5)).crossings) == 1  # cr(K5) = 1
        assert len(planarize_graph(*_complete_bipartite(3, 3)).crossings) == 1  # cr(K3,3) = 1

    def test_augmented_graph_is_always_planar(self):
        """For any input, the planarized (augmented) graph is planar and every
        crossing dummy has degree four."""
        rng = random.Random(1)
        for _ in range(300):
            n = rng.randint(3, 8)
            possible = [(i, j) for i, j in itertools.combinations(range(n), 2)]
            edges = [e for e in possible if rng.random() < 0.5]
            if not edges:
                continue
            result = planarize_graph(n, edges)

            assert check_planarity(result.num_total_nodes, result.edges).is_planar

            # If the input was already planar, no crossings were introduced.
            if check_planarity(n, edges).is_planar:
                assert len(result.crossings) == 0

            # Dummy crossing vertices have degree four.
            degree: dict[int, int] = {}
            for a, b in result.edges:
                degree[a] = degree.get(a, 0) + 1
                degree[b] = degree.get(b, 0) + 1
            for cv in result.crossings:
                assert degree.get(cv.index, 0) == 4

    def test_original_edge_paths_are_connected(self):
        """Each original edge maps to a segment path from its source to target."""
        n, edges = _complete(5)  # K5 -> some edges get split by crossings
        result = planarize_graph(n, edges)
        for orig_idx, (src, tgt) in enumerate(edges):
            seg_ids = result.original_to_edges[orig_idx]
            assert seg_ids, f"edge {orig_idx} has no segments"
            # Walk the segment chain and confirm it runs src -> ... -> tgt.
            cur = src
            for si in seg_ids:
                a, b = result.edges[si]
                assert cur in (a, b)
                cur = b if cur == a else a
            assert cur == tgt
