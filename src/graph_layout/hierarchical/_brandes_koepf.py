"""Brandes-Köpf horizontal coordinate assignment for layered (Sugiyama) graphs.

Implements "Fast and Simple Horizontal Coordinate Assignment"
(Brandes & Köpf, 2002). Given a *proper* layered graph -- every edge connects
adjacent layers, long edges already split by dummy nodes -- with a fixed vertex
ordering per layer, it assigns an x-coordinate to every vertex so that:

  * the left-to-right order within each layer is preserved and vertices keep at
    least ``delta`` separation (no overlaps), and
  * edges are drawn as vertically as possible (aligning each vertex with the
    median of its neighbours), which straightens long-edge chains.

The public entry point is :func:`assign_x`. It runs the four combinations of
{align upward, align downward} x {leftmost, rightmost} biasing and balances them
into a single symmetric assignment, as in the paper.
"""

from __future__ import annotations

from typing import Optional

Neighbours = dict[int, list[int]]


def _mark_type1_conflicts(
    layers: list[list[int]],
    pos: dict[int, int],
    upper: Neighbours,
    is_dummy: set[int],
) -> set[tuple[int, int]]:
    """Mark type-1 conflicts: non-inner segments that cross an inner segment.

    An *inner* segment joins two dummy vertices. When a non-inner segment crosses
    an inner one the alignment must yield to the inner segment (so long edges stay
    straight); the crossing non-inner segments are returned as ``(upper, lower)``
    pairs to be skipped during alignment.
    """
    marked: set[tuple[int, int]] = set()

    def incident_to_inner(v: int) -> Optional[int]:
        """Return the position of v's upper neighbour if (upper, v) is inner."""
        if v not in is_dummy:
            return None
        for u in upper.get(v, []):
            if u in is_dummy:
                return pos[u]
        return None

    for i in range(1, len(layers) - 1):
        lower_layer = layers[i + 1]
        k0 = 0
        el = 0  # scan cursor over the lower layer
        for el1, v in enumerate(lower_layer):
            inner_upper_pos = incident_to_inner(v)
            if el1 == len(lower_layer) - 1 or inner_upper_pos is not None:
                k1 = len(layers[i]) - 1
                if inner_upper_pos is not None:
                    k1 = inner_upper_pos
                while el <= el1:
                    vl = lower_layer[el]
                    for u in upper.get(vl, []):
                        ku = pos[u]
                        if ku < k0 or ku > k1:
                            marked.add((u, vl))
                    el += 1
                k0 = k1
    return marked


def _vertical_alignment(
    layers: list[list[int]],
    pos: dict[int, int],
    predecessors: Neighbours,
    marked: set[tuple[int, int]],
    right_to_left: bool,
) -> tuple[dict[int, int], dict[int, int]]:
    """Align each vertex with the median of its ``predecessors``.

    ``predecessors`` is the neighbour map in the direction we align towards
    (upper neighbours for a downward pass, lower for an upward pass). ``layers``
    is already ordered so that the predecessor layer precedes each vertex's
    layer. ``right_to_left`` biases ties to the right median and scans each layer
    from right to left.

    Returns ``(root, align)`` describing the blocks: ``root[v]`` is the topmost
    vertex of v's block and following ``align`` cycles through the block.
    """
    root = {v: v for layer in layers for v in layer}
    align = {v: v for layer in layers for v in layer}

    for layer in layers:
        order = reversed(layer) if right_to_left else layer
        # r bounds the position already consumed so alignments stay planar.
        r = len(pos) if right_to_left else -1
        for v in order:
            preds = predecessors.get(v, [])
            d = len(preds)
            if d == 0:
                continue
            # Median predecessor(s): for even degree consider both medians.
            mids = [(d - 1) // 2, d // 2]
            if right_to_left:
                mids = mids[::-1]
            for m in mids:
                if align[v] != v:
                    break
                u = preds[m]
                if (u, v) in marked or (v, u) in marked:
                    continue
                pu = pos[u]
                if (right_to_left and pu < r) or (not right_to_left and pu > r):
                    align[u] = v
                    root[v] = root[u]
                    align[v] = root[v]
                    r = pu
    return root, align


def _horizontal_compaction(
    layers: list[list[int]],
    root: dict[int, int],
    delta: float,
    pack_right: bool = False,
) -> dict[int, float]:
    """Assign an x-coordinate to every block, packed as tight as possible.

    Every vertex takes its block root's coordinate, so a block (a chain of
    vertices aligned across layers) is drawn as a straight vertical line. Within
    each layer consecutive vertices must be separated by ``delta``, which -- since
    same-block vertices never share a layer -- is a set of ``x[root(b)] >=
    x[root(a)] + delta`` constraints between distinct block roots. The BK
    alignment keeps blocks non-crossing, so these constraints form a DAG and a
    longest-path (Bellman-Ford to a fixpoint) yields the tightest feasible
    coordinates -- equivalent to the paper's shift-class compaction but without
    its fragile bookkeeping. This guarantees the no-overlap / ordering invariant
    by construction.

    ``pack_right`` packs blocks against the right instead of the left (used by
    the rightmost-biased runs so the four runs are symmetric).
    """
    roots = {root[v] for layer in layers for v in layer}
    xr: dict[int, float] = {r: 0.0 for r in roots}

    # Longest path over the within-layer separation constraints; bounded by the
    # number of blocks (a DAG's longest path). Left packing raises each right
    # block off its left neighbour; right packing lowers each left block under
    # its right neighbour.
    for _ in range(len(roots) + 1):
        changed = False
        for layer in layers:
            for a, b in zip(layer, layer[1:]):
                ra, rb = root[a], root[b]
                if pack_right:
                    allowed = xr[rb] - delta
                    if xr[ra] > allowed + 1e-9:
                        xr[ra] = allowed
                        changed = True
                else:
                    required = xr[ra] + delta
                    if xr[rb] < required - 1e-9:
                        xr[rb] = required
                        changed = True
        if not changed:
            break

    return {v: xr[root[v]] for layer in layers for v in layer}


def assign_x(
    layers: list[list[int]],
    pos: dict[int, int],
    upper: Neighbours,
    lower: Neighbours,
    is_dummy: set[int],
    delta: float,
) -> dict[int, float]:
    """Assign a balanced Brandes-Köpf x-coordinate to every vertex.

    Args:
        layers: Ordered layers (lists of vertex ids), top layer first.
        pos: Vertex -> position within its layer.
        upper: Vertex -> its neighbours in the layer above, ordered by position.
        lower: Vertex -> its neighbours in the layer below, ordered by position.
        is_dummy: Set of dummy vertex ids (edge-bend vertices).
        delta: Minimum separation between adjacent vertices in a layer.

    Returns:
        Vertex -> x-coordinate. Within every layer the coordinates respect the
        given order and are separated by at least ``delta``.
    """
    if not layers:
        return {}

    marked = _mark_type1_conflicts(layers, pos, upper, is_dummy)

    # Four runs: {align to upper, align to lower} x {leftmost, rightmost}. The
    # rightmost runs bias median ties right and pack right, so the set is
    # symmetric and the balancing centres the drawing rather than biasing left.
    candidates: list[dict[int, float]] = []
    right_biased: list[bool] = []
    for upward in (False, True):
        run_layers = list(reversed(layers)) if upward else layers
        predecessors = lower if upward else upper
        for right_to_left in (False, True):
            root, _align = _vertical_alignment(run_layers, pos, predecessors, marked, right_to_left)
            candidates.append(_horizontal_compaction(run_layers, root, delta, right_to_left))
            right_biased.append(right_to_left)

    # Balance (Brandes-Köpf): shift each run onto the run of smallest width --
    # left-biased runs by aligning their minimum, right-biased runs by aligning
    # their maximum -- then take, per vertex, the average of the two median
    # coordinates.
    widths = [max(c.values()) - min(c.values()) for c in candidates]
    ref = widths.index(min(widths))
    ref_min = min(candidates[ref].values())
    ref_max = max(candidates[ref].values())

    aligned: list[dict[int, float]] = []
    for c, is_right in zip(candidates, right_biased):
        shift = ref_max - max(c.values()) if is_right else ref_min - min(c.values())
        aligned.append({v: xv + shift for v, xv in c.items()})

    balanced: dict[int, float] = {}
    for v in pos:
        vals = sorted(a[v] for a in aligned)
        balanced[v] = (vals[1] + vals[2]) / 2.0

    # The per-vertex median of the four runs can, in principle, place a couple of
    # vertices closer than delta. Enforce the ordering / separation invariant
    # with a final left-to-right pass per layer; it only nudges an offending
    # vertex and never reorders.
    for layer in layers:
        for a, b in zip(layer, layer[1:]):
            if balanced[b] < balanced[a] + delta:
                balanced[b] = balanced[a] + delta
    return balanced
