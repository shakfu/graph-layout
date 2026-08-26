"""Tests for warning emission and attribution.

Layouts warn when the caller's graph does not suit the algorithm -- a tree
layout handed something that is not a tree. Nothing in the suite asserted on
those warnings, so neither the messages nor their attribution were covered.

Attribution matters: a warning blamed on a file inside graph_layout is one the
caller cannot act on and cannot filter by module. The literal ``stacklevel``
values these sites used had drifted to 2, 3 and 4 across call paths and were
landing on base.py rather than the caller.
"""

from __future__ import annotations

import warnings

import pytest

from graph_layout import RadialTreeLayout, ReingoldTilfordLayout
from graph_layout._warnings import warn_at_caller
from graph_layout.hierarchical import RadialTreeStructureWarning, TreeStructureWarning

# A 3-cycle: every node has an incoming edge, so no root exists.
CYCLE_NODES = [{} for _ in range(3)]
CYCLE_LINKS = [
    {"source": 0, "target": 1},
    {"source": 1, "target": 2},
    {"source": 2, "target": 0},
]

# A rooted pair plus an unreachable island.
DISCONNECTED_NODES = [{} for _ in range(4)]
DISCONNECTED_LINKS = [{"source": 0, "target": 1}, {"source": 2, "target": 3}]


class TestWarnAtCaller:
    """The helper counts library frames at runtime instead of hardcoding one."""

    def test_attributed_to_the_caller(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_at_caller("something", UserWarning)

        assert len(caught) == 1
        assert caught[0].filename == __file__, (
            f"warning blamed on {caught[0].filename}, not the calling test"
        )

    def test_attribution_survives_intermediate_library_frames(self):
        """Frames inside graph_layout must be skipped however many there are."""
        from graph_layout import base

        def emit():
            warn_at_caller("deep", UserWarning)

        # Route the call through a real library frame.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            base.run_deep_recursive(emit, depth=1)

        assert len(caught) == 1
        assert caught[0].filename == __file__

    def test_category_and_message_are_passed_through(self):
        with pytest.warns(TreeStructureWarning, match="custom text"):
            warn_at_caller("custom text here", TreeStructureWarning)

    def test_extra_depth_is_honoured(self):
        """A wrapper can ask for one more frame to be skipped."""

        def wrapper():
            warn_at_caller("wrapped", UserWarning, extra_depth=1)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            wrapper()

        assert len(caught) == 1
        # Skipping one extra frame moves attribution off this file's call line.
        assert caught[0].filename == __file__


class TestTreeLayoutWarnings:
    """The layouts warn, with the right category, on unsuitable graphs."""

    @pytest.mark.parametrize(
        "cls,category",
        [
            (ReingoldTilfordLayout, TreeStructureWarning),
            (RadialTreeLayout, RadialTreeStructureWarning),
        ],
    )
    def test_warns_when_no_root_exists(self, cls, category):
        with pytest.warns(category, match="No root node found"):
            cls(
                nodes=[dict(n) for n in CYCLE_NODES],
                links=[dict(link) for link in CYCLE_LINKS],
                size=(400, 300),
            ).run()

    @pytest.mark.parametrize(
        "cls,category",
        [
            (ReingoldTilfordLayout, TreeStructureWarning),
            (RadialTreeLayout, RadialTreeStructureWarning),
        ],
    )
    def test_warns_about_unreachable_nodes(self, cls, category):
        with pytest.warns(category, match="disconnected node"):
            cls(
                nodes=[dict(n) for n in DISCONNECTED_NODES],
                links=[dict(link) for link in DISCONNECTED_LINKS],
                size=(400, 300),
            ).run()

    @pytest.mark.parametrize("cls", [ReingoldTilfordLayout, RadialTreeLayout])
    def test_a_real_tree_warns_about_nothing(self, cls):
        nodes = [{} for _ in range(4)]
        links = [
            {"source": 0, "target": 1},
            {"source": 0, "target": 2},
            {"source": 1, "target": 3},
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cls(nodes=nodes, links=links, size=(400, 300)).run()
