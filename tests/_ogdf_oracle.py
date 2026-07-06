"""Shared helpers for the OGDF differential-testing oracle.

``ogdf-py`` (https://github.com/shakfu/ogdf-py) provides Python bindings for the
C++ Open Graph Drawing Framework. Here it is used as an *optional, dev-only*
oracle: an independent, industrial-grade implementation (OGDF uses the
Boyer-Myrvold planarity test) of the same questions ``graph_layout`` answers with
its own hand-rolled code in ``graph_layout/planarity/``. Cross-checking against a
second, unrelated implementation catches bugs that a single codebase -- or two
codebases that happen to share a blind spot -- would miss.

It is a dev-only (never runtime) dependency, published on PyPI as ``ogdf-py``
(import name ``ogdf``). Because it is a compiled C++ extension with prebuilt
wheels only for CPython 3.10-3.13 on Linux/macOS, it is pinned into the dev group
in ``pyproject.toml`` behind an environment marker matching that wheel coverage,
so ``uv sync`` installs it there and skips it (rather than attempting a
from-source OGDF build) on Windows, Python 3.9, or 3.14+. Tests using these
helpers skip cleanly whenever ``ogdf`` is not importable, so the default
``make test`` run is unaffected by its absence.

To test graph-layout against a local, unreleased build of the sibling ``ogdf-py``
checkout instead of the pinned PyPI release, use ``make oracle-install``.
"""

from __future__ import annotations

import importlib.util
from typing import Any, Sequence

import pytest

HAS_OGDF = importlib.util.find_spec("ogdf") is not None

requires_ogdf = pytest.mark.skipif(
    not HAS_OGDF,
    reason="ogdf-py not installed (optional oracle; see tests/_ogdf_oracle.py)",
)


def require_ogdf_attr(attr: str) -> None:
    """Skip the calling test unless the installed ``ogdf`` build exposes ``attr``.

    ``ogdf-py`` is a curated subset that grows over releases: e.g. the PyPI 0.1.1
    wheel exposes ``is_connected``/``is_biconnected``/``connected_components`` but
    not ``cut_vertices``/``separation_pair``/``spqr_tree_summary`` (present in
    newer/local builds). This lets the finer connectivity oracles activate when
    run against a build that has them (``make oracle-install``) and skip cleanly
    on the pinned PyPI release, rather than erroring at call time.
    """
    import ogdf

    if not hasattr(ogdf, attr):
        pytest.skip(
            f"ogdf.{attr} not available in this ogdf-py build "
            f"(needs a newer release or `make oracle-install`)"
        )


def build_ogdf_graph(num_nodes: int, edges: Sequence[tuple[int, int]]) -> tuple[Any, list[Any]]:
    """Build an ``ogdf.Graph`` from graph-layout's ``(num_nodes, edges)`` model.

    Nodes are created in index order (``ogdf`` assigns ``0..num_nodes-1``
    matching insertion order), so the returned node list is indexable by the
    same integer labels graph-layout uses. Returns ``(graph, nodes)``.

    Importing ``ogdf`` here (not at module load) keeps this file importable when
    the oracle is absent, so ``requires_ogdf``-marked tests skip rather than
    erroring at collection time.
    """
    import ogdf

    g = ogdf.Graph()
    nodes = [g.new_node() for _ in range(num_nodes)]
    for u, v in edges:
        g.new_edge(nodes[u], nodes[v])
    return g, nodes
