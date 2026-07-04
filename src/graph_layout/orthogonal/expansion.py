"""Vertex expansion (cages) for orthogonal drawing of degree > 4 vertices.

The Tamassia flow model realizes vertices as grid points with at most one edge
per compass direction, so it is limited to maximum degree 4. The classical way
to lift that limit (GIOTTO; also OGDF's planar orthogonal layouts) is *vertex
expansion*: every vertex of degree d > 4 is replaced by a cycle of d cage
vertices -- one per incident edge, in rotation order -- and drawn as a
rectangle. Each cage vertex has degree 3 (two cycle edges plus its original
edge), so the expanded graph is planar with maximum degree <= 4 and the
standard Topology-Shape-Metrics pipeline applies. Constraining the cage face
to be drawn as a rectangle (corner angles at most 180 degrees, no bends on
cycle edges) makes the cage a proper node box with multiple edges attaching
per side -- the familiar Kandinsky look.

This module builds the expanded graph and embedding; the flow-network
constraints live in :func:`orthogonalization.build_flow_network` (``cage_faces``
and ``rigid_edges``) and the mapping back to node boxes in ``GIOTTOLayout``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..planarity._embedding import PlanarEmbedding
from .orthogonalization import Face


@dataclass
class Expansion:
    """The expanded graph, embedding, and bookkeeping to map back.

    Original vertex ids keep their meaning for non-expanded vertices; each
    expanded vertex v is replaced by ``cages[v]`` (its id becomes isolated and
    is never drawn). ``dart_map`` rewrites original directed edges to expanded
    directed edges, e.g. for routing lookups.
    """

    num_nodes: int  # expanded id space size (original n + all cage vertices)
    edges: list[tuple[int, int]]  # rewritten original edges + cage cycle edges
    embedding: PlanarEmbedding  # rotation system of the expanded graph
    cages: dict[int, list[int]] = field(default_factory=dict)  # v -> cycle in rotation order
    cage_edges: set[tuple[int, int]] = field(default_factory=set)  # canonical cycle edges
    dart_map: dict[tuple[int, int], tuple[int, int]] = field(default_factory=dict)
    origin: dict[int, int] = field(default_factory=dict)  # cage vertex -> original vertex


def expand_high_degree(
    num_nodes: int,
    embedding: PlanarEmbedding,
    max_degree: int = 4,
) -> Optional[Expansion]:
    """Expand every vertex of degree > ``max_degree`` into a cage cycle.

    The cage vertices follow the vertex's rotation order, so the cyclic order
    of the original edges around the cage matches the embedding. With the
    rotation ``[c_next, c_prev, external]`` at each cage vertex, the increasing
    cycle darts (c_i, c_{i+1}) enclose the cage face and the surrounding faces
    traverse the cage in decreasing order, preserving planarity (one new face
    per expanded vertex).

    Returns None when no vertex exceeds ``max_degree`` (nothing to do).
    """
    rotation = embedding.rotation
    high = sorted(v for v, nbrs in rotation.items() if len(nbrs) > max_degree)
    if not high:
        return None

    expansion = Expansion(num_nodes=num_nodes, edges=[], embedding=embedding)

    # Allocate cage ids: port[(v, u)] is the cage vertex handling edge (v, u)
    # at v's end.
    next_id = num_nodes
    port: dict[tuple[int, int], int] = {}
    for v in high:
        nbrs = rotation[v]
        ids = list(range(next_id, next_id + len(nbrs)))
        next_id += len(nbrs)
        expansion.cages[v] = ids
        for i, u in enumerate(nbrs):
            port[(v, u)] = ids[i]
            expansion.origin[ids[i]] = v
    expansion.num_nodes = next_id

    def far(v_at: int, w: int) -> int:
        """Id replacing neighbor w of edge (v_at, w), as seen from v_at."""
        return port.get((w, v_at), w)

    # Rewritten rotation system.
    new_rotation: dict[int, list[int]] = {}
    for v, nbrs in rotation.items():
        if v in expansion.cages:
            continue
        new_rotation[v] = [far(v, w) for w in nbrs]
    for v in high:
        nbrs = rotation[v]
        ids = expansion.cages[v]
        d = len(ids)
        for i, u in enumerate(nbrs):
            new_rotation[ids[i]] = [ids[(i + 1) % d], ids[(i - 1) % d], far(v, u)]

    # Rewritten edges (each original undirected edge once, from the rotation)
    # plus the cage cycles.
    seen: set[tuple[int, int]] = set()
    for v, nbrs in rotation.items():
        for u in nbrs:
            key = (min(u, v), max(u, v))
            if key in seen:
                continue
            seen.add(key)
            pu = port.get((u, v), u)
            pv = port.get((v, u), v)
            expansion.edges.append((pu, pv))
            expansion.dart_map[(u, v)] = (pu, pv)
            expansion.dart_map[(v, u)] = (pv, pu)
    for v in high:
        ids = expansion.cages[v]
        d = len(ids)
        for i in range(d):
            a, b = ids[i], ids[(i + 1) % d]
            expansion.edges.append((a, b))
            expansion.cage_edges.add((min(a, b), max(a, b)))

    new_embedding = PlanarEmbedding(new_rotation)

    # Carry the outer face over: the expanded face containing the rewrite of
    # any dart of the original outer face.
    if embedding.outer_face_index is not None:
        original_faces = embedding.faces()
        outer = original_faces[embedding.outer_face_index]
        if outer:
            target = expansion.dart_map.get(outer[0])
            if target is not None:
                for fi, face in enumerate(new_embedding.faces()):
                    if target in face:
                        new_embedding.set_outer_face(fi)
                        break

    expansion.embedding = new_embedding
    return expansion


def cage_face_indices(expansion: Expansion, faces: list[Face]) -> Optional[set[int]]:
    """Face indices of the cage faces (one per expanded vertex).

    Each cage face must consist of exactly the increasing cycle darts of its
    cage; returns None if any does not (which would mean the expanded
    embedding is inconsistent).
    """
    dart_face: dict[tuple[int, int], int] = {}
    for face in faces:
        for dart in face.edges:
            dart_face[dart] = face.index
    face_by_index = {f.index: f for f in faces}

    result: set[int] = set()
    for _v, ids in expansion.cages.items():
        d = len(ids)
        fi = dart_face.get((ids[0], ids[1]))
        if fi is None:
            return None
        expected = {(ids[i], ids[(i + 1) % d]) for i in range(d)}
        if set(face_by_index[fi].edges) != expected:
            return None
        result.add(fi)
    return result


__all__ = ["Expansion", "expand_high_degree", "cage_face_indices"]
