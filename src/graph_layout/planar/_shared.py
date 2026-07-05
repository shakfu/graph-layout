"""Shared infrastructure for planar straight-line drawing algorithms.

This module provides the combinatorial machinery that the grid-drawing
algorithms (Schnyder, de Fraysseix-Pach-Pollack) build on:

- :func:`build_embedding` -- extract a planar rotation system from an edge list
  via the LR-planarity test and choose an outer face.
- :func:`triangulate` -- augment a planar embedding to a *maximal* planar graph
  (every face a triangle) by inserting chords, tracking which edges were added.
- :func:`canonical_order` -- the de Fraysseix-Pach-Pollack canonical (shelling)
  ordering of a triangulation, the common substrate of both the shift algorithm
  and Schnyder's realizer.

A combinatorial planar embedding is stored as a *rotation system*: a mapping
from each vertex to the clockwise cyclic order of its neighbours, matching the
convention of :class:`graph_layout.planarity.PlanarEmbedding`. Faces are traced
with the same right-turn rule so that the two agree.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..planarity import check_planarity

Edge = tuple[int, int]
Rotation = dict[int, list[int]]


# ---------------------------------------------------------------------------
# Edge / embedding extraction
# ---------------------------------------------------------------------------


def normalize_edges(num_nodes: int, edges: Sequence[Edge]) -> list[Edge]:
    """Drop self-loops and duplicate (undirected) edges, keep in-range ones."""
    seen: set[Edge] = set()
    clean: list[Edge] = []
    for u, v in edges:
        if u == v:
            continue
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            continue
        key = (u, v) if u < v else (v, u)
        if key in seen:
            continue
        seen.add(key)
        clean.append((u, v))
    return clean


def trace_faces(rotation: Rotation) -> list[list[int]]:
    """Enumerate faces as vertex cycles using the clockwise right-turn rule.

    Returns each face as an ordered list of vertices ``[o0, o1, ...]`` whose
    boundary walk is ``o0 -> o1 -> ... -> o0``. Consistent with
    :meth:`PlanarEmbedding.faces` (which returns the same walks as directed
    edges).
    """
    rot_index: dict[int, dict[int, int]] = {
        v: {w: i for i, w in enumerate(nbrs)} for v, nbrs in rotation.items()
    }
    visited: set[Edge] = set()
    faces: list[list[int]] = []

    for v, nbrs in rotation.items():
        for w in nbrs:
            if (v, w) in visited:
                continue
            face: list[int] = []
            u, cur = v, w
            while (u, cur) not in visited:
                visited.add((u, cur))
                face.append(u)
                idx = rot_index[cur][u]
                ring = rotation[cur]
                nxt = ring[(idx - 1) % len(ring)]
                u, cur = cur, nxt
            faces.append(face)
    return faces


def build_embedding(
    num_nodes: int, edges: Sequence[Edge]
) -> Optional[tuple[Rotation, list[list[int]]]]:
    """Return ``(rotation, faces)`` for a connected planar simple graph.

    Returns ``None`` when the graph is non-planar, disconnected, or too small
    (fewer than 3 vertices) for a straight-line grid drawing.
    """
    clean = normalize_edges(num_nodes, edges)
    # Need at least a triangle's worth of vertices; missing edges are supplied by
    # triangulation, so a sparse-but-connected graph (e.g. a path or tree) is
    # fine as long as the Euler check below confirms a single component.
    if num_nodes < 3:
        return None

    result = check_planarity(num_nodes, clean)
    if not result.is_planar or result.embedding is None:
        return None

    rotation: Rotation = {v: list(result.embedding.get(v, [])) for v in range(num_nodes)}

    # Straight-line drawing needs a single connected component: an isolated
    # vertex or a second component has no place in one outer triangle.
    if any(len(rotation[v]) == 0 for v in range(num_nodes)):
        return None

    faces = trace_faces(rotation)
    # Euler check for connectivity: a connected planar graph has V - E + F = 2.
    if num_nodes - len(clean) + len(faces) != 2:
        return None

    return rotation, faces


# ---------------------------------------------------------------------------
# Triangulation
# ---------------------------------------------------------------------------


def _insert_after(ring: list[int], anchor: int, value: int) -> None:
    """Insert ``value`` immediately after ``anchor`` in ``ring``."""
    ring.insert(ring.index(anchor) + 1, value)


def _insert_before(ring: list[int], anchor: int, value: int) -> None:
    """Insert ``value`` immediately before ``anchor`` in ``ring``."""
    ring.insert(ring.index(anchor), value)


def triangulate(rotation: Rotation) -> tuple[Rotation, set[Edge]]:
    """Augment ``rotation`` to a maximal planar graph (every face a triangle).

    Repeatedly clips an *ear* off any face longer than three edges: given three
    consecutive boundary vertices ``a, m, b`` where ``a`` and ``b`` are not yet
    adjacent, the chord ``(a, b)`` is inserted, cutting the triangle ``(a, m, b)``
    off the face. The insertion positions in the rotation system are derived from
    the face-tracing rule so the embedding stays valid:

    - ``b`` goes immediately after ``m`` in ``a``'s clockwise order;
    - ``a`` goes immediately before ``m`` in ``b``'s clockwise order.

    Returns the mutated rotation and the set of added (undirected, ``min<max``)
    chords. Raises :class:`ValueError` if a face cannot be triangulated with a
    simple chord (every candidate ear diagonal already exists elsewhere), which
    would require a Steiner vertex.
    """
    rotation = {v: list(nbrs) for v, nbrs in rotation.items()}
    adj: list[set[int]] = [set() for _ in range(len(rotation))]
    for v, nbrs in rotation.items():
        for w in nbrs:
            adj[v].add(w)

    added: set[Edge] = set()

    # Bound iterations: a triangulation adds at most 3n - 6 - m edges.
    max_iters = 3 * len(rotation)
    iters = 0
    while True:
        faces = trace_faces(rotation)
        big = next((f for f in faces if len(f) > 3), None)
        if big is None:
            break
        iters += 1
        if iters > max_iters + 6:
            raise ValueError("triangulation failed to converge")

        k = len(big)
        chord_added = False
        for i in range(k):
            a = big[(i - 1) % k]
            m = big[i]
            b = big[(i + 1) % k]
            if a == b or b in adj[a]:
                continue
            # Insert the chord (a, b), clipping ear at m.
            _insert_after(rotation[a], m, b)
            _insert_before(rotation[b], m, a)
            adj[a].add(b)
            adj[b].add(a)
            added.add((a, b) if a < b else (b, a))
            chord_added = True
            break

        if not chord_added:
            raise ValueError(
                "face cannot be triangulated with a simple chord (would need a Steiner vertex)"
            )

    return rotation, added


# ---------------------------------------------------------------------------
# Canonical ordering
# ---------------------------------------------------------------------------


def choose_outer_triangle(rotation: Rotation, faces: list[list[int]]) -> tuple[int, int, int]:
    """Pick an outer triangle ``(v1, v2, vn)`` from a triangulated embedding.

    Prefers a triangle incident to an edge of a large original face so the
    drawing's outer boundary matches the graph's natural periphery. Any face is
    a valid choice for correctness; this only affects aesthetics.
    """
    # Largest original face gives a boundary edge to sit the outer triangle on.
    largest = max(faces, key=len)
    v1, v2 = largest[0], largest[1]

    # The (triangulated) face on the (v1 -> v2) side is our outer triangle.
    tri_faces = trace_faces(rotation)
    for f in tri_faces:
        assert len(f) == 3
        for i in range(3):
            if f[i] == v1 and f[(i + 1) % 3] == v2:
                return v1, v2, f[(i + 2) % 3]
    # Fallback: any triangle.
    f = tri_faces[0]
    return f[0], f[1], f[2]


def canonical_order(rotation: Rotation, outer: tuple[int, int, int]) -> list[int]:
    """Compute a de Fraysseix-Pach-Pollack canonical ordering.

    ``rotation`` must be a maximal planar (triangulated) embedding and
    ``outer = (v1, v2, vn)`` the outer-triangle vertices, with directed edge
    ``v1 -> v2`` on the outer face and ``vn`` the third (apex) corner.

    Returns the ordering ``[v1, v2, v3, ..., vn]`` such that for every
    ``k >= 3`` the subgraph induced by the first ``k`` vertices is 2-connected
    and internally triangulated, ``vk`` lies on its outer boundary, and ``vk``'s
    earlier neighbours are consecutive on that boundary.

    The order is built by reverse deletion: peel vertices off the current outer
    cycle, always removing one incident to no chord of the cycle.
    """
    v1, v2, vn = outer
    n = len(rotation)

    adj: list[set[int]] = [set() for _ in range(n)]
    for v, nbrs in rotation.items():
        for w in nbrs:
            adj[v].add(w)

    present = [True] * n

    # Initial outer cycle = the outer triangle itself, traced as the face on
    # directed edge v1 -> v2 (which yields [v1, v2, vn]). All other vertices lie
    # on the far side of this cycle and are peeled off one at a time.
    outer_cycle = _face_cycle(rotation, v1, v2)

    order: list[int] = []

    for k in range(n, 2, -1):
        if k == n:
            vk = vn
        else:
            picked = _pick_removable(outer_cycle, adj, present, v1, v2)
            if picked is None:
                raise ValueError("canonical ordering failed: no removable vertex")
            vk = picked
        order.append(vk)
        present[vk] = False
        outer_cycle = _update_cycle_after_removal(outer_cycle, vk, rotation, present)

    order.append(v2)
    order.append(v1)
    order.reverse()
    return order


def build_realizer(
    rotation: Rotation, order: list[int], outer: tuple[int, int, int]
) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    """Build a Schnyder wood (realizer) from a canonical ordering.

    Returns three parent maps ``(parent_1, parent_2, parent_3)`` giving, for each
    interior vertex, its outgoing edge in each of the three trees:

    - tree 1 (rooted at ``v1``): parent is the *leftmost* earlier neighbour;
    - tree 2 (rooted at ``v2``): parent is the *rightmost* earlier neighbour;
    - tree 3 (rooted at ``vn``): parent is the vertex that, when inserted, covered
      this one as a *middle* neighbour.

    This is the classical incremental realizer: adding vertex ``vk`` over the
    contiguous run ``w_1, ..., w_p`` of its earlier neighbours orients
    ``vk -> w_1`` in tree 1, ``vk -> w_p`` in tree 2, and ``w_j -> vk`` in tree 3
    for each middle ``w_j``. Every interior vertex ends with exactly one parent
    per tree.
    """
    v1, v2, _vn = outer
    n = len(rotation)
    adj: list[set[int]] = [set() for _ in range(n)]
    for v, nbrs in rotation.items():
        for w in nbrs:
            adj[v].add(w)
    pos = {v: i for i, v in enumerate(order)}

    parent_1: dict[int, int] = {}
    parent_2: dict[int, int] = {}
    parent_3: dict[int, int] = {}

    boundary = [v1, v2]
    for k in range(2, n):
        vk = order[k]
        lower = {w for w in adj[vk] if pos[w] < k}
        idxs = [i for i, b in enumerate(boundary) if b in lower]
        i0, i1 = idxs[0], idxs[-1]
        if idxs != list(range(i0, i1 + 1)):
            raise ValueError("realizer: earlier neighbours not contiguous")
        run = boundary[i0 : i1 + 1]
        parent_1[vk] = run[0]
        parent_2[vk] = run[-1]
        for wj in run[1:-1]:
            parent_3[wj] = vk
        boundary = boundary[: i0 + 1] + [vk] + boundary[i1:]

    return parent_1, parent_2, parent_3


def _face_cycle(rotation: Rotation, u: int, v: int) -> list[int]:
    """Trace the face to the right of directed edge ``u -> v`` as a vertex ring."""
    rot_index: dict[int, dict[int, int]] = {
        vtx: {w: i for i, w in enumerate(nbrs)} for vtx, nbrs in rotation.items()
    }
    cycle: list[int] = []
    a, b = u, v
    start = (u, v)
    while True:
        cycle.append(a)
        idx = rot_index[b][a]
        ring = rotation[b]
        nxt = ring[(idx - 1) % len(ring)]
        a, b = b, nxt
        if (a, b) == start:
            break
    return cycle


def _has_chord(v: int, outer_cycle: list[int], adj: list[set[int]], present: list[bool]) -> bool:
    """Does ``v`` have a chord: an edge to a non-adjacent vertex on the cycle?"""
    pos = {w: i for i, w in enumerate(outer_cycle)}
    i = pos[v]
    m = len(outer_cycle)
    left = outer_cycle[(i - 1) % m]
    right = outer_cycle[(i + 1) % m]
    for w in adj[v]:
        if not present[w] or w == v:
            continue
        if w in pos and w != left and w != right:
            return True
    return False


def _pick_removable(
    outer_cycle: list[int],
    adj: list[set[int]],
    present: list[bool],
    v1: int,
    v2: int,
) -> Optional[int]:
    """Choose a chord-free outer-cycle vertex other than the base ``v1, v2``."""
    for v in outer_cycle:
        if v == v1 or v == v2:
            continue
        if not _has_chord(v, outer_cycle, adj, present):
            return v
    return None


def _update_cycle_after_removal(
    outer_cycle: list[int],
    vk: int,
    rotation: Rotation,
    present: list[bool],
) -> list[int]:
    """Replace ``vk`` on the outer cycle by its now-exposed interior neighbours.

    ``vk`` sits between cycle neighbours ``p`` (previous) and ``q`` (next). Its
    remaining neighbours form a fan from ``p`` to ``q`` through the interior;
    the intermediate fan vertices become the new boundary between ``p`` and
    ``q`` (in reverse rotation order, matching the ring's orientation).
    """
    i = outer_cycle.index(vk)
    m = len(outer_cycle)
    p = outer_cycle[(i - 1) % m]
    q = outer_cycle[(i + 1) % m]

    ring = rotation[vk]
    # Present neighbours of vk in clockwise rotation order.
    present_nbrs = [w for w in ring if present[w]]

    # Rotate the neighbour ring so it starts at q and ends at p; the vertices
    # strictly between are the exposed fan. The outer cycle is oriented so that
    # the interior lies consistently; the fan from q to p (exclusive) in
    # clockwise order gives the replacement path.
    if q in present_nbrs:
        start = present_nbrs.index(q)
        rotated = present_nbrs[start:] + present_nbrs[:start]
    else:
        rotated = present_nbrs
    # rotated[0] == q; strip q at front and p at the (expected) end.
    fan = rotated[1:]
    if fan and fan[-1] == p:
        fan = fan[:-1]
    else:
        # p not last (degenerate wheel); drop any occurrence of p.
        fan = [w for w in fan if w != p]

    new_cycle = outer_cycle[:i] + fan + outer_cycle[i + 1 :]
    return new_cycle
