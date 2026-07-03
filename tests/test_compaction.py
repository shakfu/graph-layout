"""Tests for the greedy (longest-path) orthogonal compaction (H8)."""

from graph_layout.orthogonal.compaction import (
    CompactionConstraint,
    CompactionSolver,
    compact_horizontal,
)
from graph_layout.orthogonal.types import NodeBox


def _box(i: int, x: float, y: float, w: float = 30.0, h: float = 20.0) -> NodeBox:
    return NodeBox(index=i, x=x, y=y, width=w, height=h)


def _h_overlap(xa: float, wa: float, xb: float, wb: float) -> bool:
    return not (xa + wa / 2 <= xb - wb / 2 or xb + wb / 2 <= xa - wa / 2)


class TestGreedyCompaction:
    """CompactionSolver must actually compact (close interior slack)."""

    def test_solver_pulls_elements_left(self):
        """The solver must pull elements to their leftmost feasible position.

        Regression (H8): the old solver only pushed right on gap violations and
        never closed interior slack, so a chain with large gaps stayed spread.
        """
        # Three elements at 0, 500, 1000 with min gap 100 between consecutive.
        constraints = [
            CompactionConstraint(left=0, right=1, gap=100.0),
            CompactionConstraint(left=1, right=2, gap=100.0),
        ]
        solver = CompactionSolver(3, [0.0, 500.0, 1000.0], constraints)
        positions, _ = solver.solve()
        # Longest-path packs them tight: 0, 100, 200.
        assert positions == [0.0, 100.0, 200.0]

    def test_compact_horizontal_reduces_width(self):
        """A spread row of boxes is compacted to the minimum feasible width."""
        boxes = [_box(0, 0, 0), _box(1, 200, 0), _box(2, 400, 0)]
        new_x = compact_horizontal(boxes, [], node_separation=60, edge_separation=15)
        span = max(new_x) - min(new_x)
        assert span < 400  # was 400
        # Minimal: consecutive centers separated by width/2 + sep + width/2 = 90.
        assert span == 180.0

    def test_compaction_keeps_nonconsecutive_overlaps_separated(self):
        """Non-consecutive boxes that overlap on the other axis must not collapse
        into an overlap when pulled left.

        Regression: with only consecutive-pair constraints, longest-path could
        place two vertically-overlapping boxes at the same x. All overlapping
        pairs are now constrained.
        """
        # box0 and box2 share the same y-range; box1 (between them in x) does not.
        boxes = [_box(0, 0, 0), _box(1, 100, 100), _box(2, 200, 0)]
        new_x = compact_horizontal(boxes, [], node_separation=60, edge_separation=15)
        assert not _h_overlap(new_x[0], 30, new_x[2], 30)
