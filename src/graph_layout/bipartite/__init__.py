"""
Bipartite layout algorithms.

Bipartite layouts position nodes in two parallel rows or columns,
suitable for graphs that can be partitioned into two disjoint sets
where edges only connect nodes from different sets.

Common use cases:
- User-item networks (recommendations)
- Author-paper networks (bibliometrics)
- Gene-disease networks (bioinformatics)
- Matching problems
"""

from .bipartite import BipartiteLayout, count_crossings, is_bipartite

__all__ = [
    "BipartiteLayout",
    "is_bipartite",
    "count_crossings",
]
