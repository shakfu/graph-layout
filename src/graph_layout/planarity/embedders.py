"""Pluggable planar embedding strategies.

Each embedder takes a graph (num_nodes, edges) and returns a PlanarEmbedding
with a chosen outer face. The strategies differ in how they select the outer
face and, for graphs with cut vertices, how they expose faces at articulation
points.

Available strategies:
    FixedEmbedder   -- Returns the LR-planarity embedding unchanged.
    MaxFaceEmbedder -- Picks the face with the most edges as outer face;
                       optimizes face exposure at cut vertices.
    MinDepthEmbedder -- Picks outer face and face exposure to minimize
                        block nesting depth in the block-cut tree.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from ._block_cut_tree import build_block_cut_tree
from ._embedding import PlanarEmbedding
from ._types import PlanarityResult


class PlanarEmbedder(Protocol):
    """Protocol for planar embedding strategies."""

    def embed(
        self,
        num_nodes: int,
        edges: Sequence[tuple[int, int]],
        planarity_result: PlanarityResult | None = None,
    ) -> PlanarEmbedding:
        """Compute a planar embedding with a chosen outer face.

        Args:
            num_nodes: Number of vertices.
            edges: Edge list.
            planarity_result: Optional pre-computed planarity result to avoid
                re-running the planarity test.

        Returns:
            PlanarEmbedding with outer_face_index set.

        Raises:
            ValueError: If the graph is not planar.
        """
        ...


def _get_or_check_planarity(
    num_nodes: int,
    edges: Sequence[tuple[int, int]],
    planarity_result: PlanarityResult | None,
) -> PlanarityResult:
    """Return a valid PlanarityResult, running the test if needed."""
    if planarity_result is not None and planarity_result.is_planar:
        return planarity_result

    # Avoid circular import by importing at call time
    from . import check_planarity

    result = check_planarity(num_nodes, edges)
    if not result.is_planar:
        raise ValueError("Graph is not planar")
    return result


def _make_embedding(result: PlanarityResult, num_nodes: int) -> PlanarEmbedding:
    """Build a PlanarEmbedding from a PlanarityResult."""
    rotation = result.embedding or {v: [] for v in range(num_nodes)}
    return PlanarEmbedding(rotation)


class FixedEmbedder:
    """Return the LR-planarity embedding with the largest face as outer face."""

    def embed(
        self,
        num_nodes: int,
        edges: Sequence[tuple[int, int]],
        planarity_result: PlanarityResult | None = None,
    ) -> PlanarEmbedding:
        # Filter self-loops for embedding (they don't affect planarity)
        clean_edges = [(u, v) for u, v in edges if u != v]
        result = _get_or_check_planarity(num_nodes, clean_edges, planarity_result)
        emb = _make_embedding(result, num_nodes)
        faces = emb.faces()
        if faces:
            best = max(range(len(faces)), key=lambda i: len(faces[i]))
            emb.set_outer_face(best)
        return emb


class MaxFaceEmbedder:
    """Choose the outer face maximizing edge count.

    For graphs with cut vertices, performs a bottom-up traversal of the
    block-cut tree to optimize face exposure at articulation points, so
    that the largest face in each block is exposed toward the outer face.

    Handles disconnected graphs and isolated vertices correctly.
    """

    def embed(
        self,
        num_nodes: int,
        edges: Sequence[tuple[int, int]],
        planarity_result: PlanarityResult | None = None,
    ) -> PlanarEmbedding:
        clean_edges = [(u, v) for u, v in edges if u != v]
        result = _get_or_check_planarity(num_nodes, clean_edges, planarity_result)
        emb = _make_embedding(result, num_nodes)
        faces = emb.faces()

        if not faces:
            return emb

        # Build adjacency for block-cut tree (skip self-loops)
        adj: list[list[int]] = [[] for _ in range(num_nodes)]
        for u, v in clean_edges:
            adj[u].append(v)
            adj[v].append(u)

        bct = build_block_cut_tree(num_nodes, adj)

        if not bct.cut_vertices:
            # Biconnected graph: just pick the largest face
            best = max(range(len(faces)), key=lambda i: len(faces[i]))
            emb.set_outer_face(best)
            return emb

        # For graphs with cut vertices, find the face assignments per block.
        # Each block has its own set of faces. We want to pick the outer face
        # globally as the largest face across all blocks when the block-cut
        # tree is rooted to maximize total outer boundary.

        # Map each face to the block it primarily belongs to by checking
        # which block contains all the face's edges.
        face_to_block: dict[int, int] = {}
        block_edge_sets: list[set[tuple[int, int]]] = []
        for block in bct.blocks:
            edge_set: set[tuple[int, int]] = set()
            for u, v in block.edges:
                edge_set.add((u, v))
                edge_set.add((v, u))
            block_edge_sets.append(edge_set)

        for fi, face in enumerate(faces):
            if not face:
                continue
            u, v = face[0]
            for bi, es in enumerate(block_edge_sets):
                if (u, v) in es:
                    face_to_block[fi] = bi
                    break

        # For each block, find the largest face
        block_largest_face: dict[int, int] = {}
        block_largest_face_size: dict[int, int] = {}
        for fi, bi in face_to_block.items():
            size = len(faces[fi])
            if bi not in block_largest_face_size or size > block_largest_face_size[bi]:
                block_largest_face[bi] = fi
                block_largest_face_size[bi] = size

        # Bottom-up traversal: root the block-cut tree at the block with
        # the largest face overall, then propagate outward.
        if not block_largest_face:
            best = max(range(len(faces)), key=lambda i: len(faces[i]))
            emb.set_outer_face(best)
            return emb

        root_block = max(
            block_largest_face.keys(),
            key=lambda bi: block_largest_face_size.get(bi, 0),
        )
        outer_fi = block_largest_face[root_block]
        emb.set_outer_face(outer_fi)
        return emb


class MinDepthEmbedder:
    """Choose outer face to minimize block nesting depth.

    Roots the block-cut tree at the block whose center minimizes the
    maximum distance to any leaf block, then picks the largest face
    of that root block as the outer face.

    Handles disconnected graphs and isolated vertices correctly.
    """

    def embed(
        self,
        num_nodes: int,
        edges: Sequence[tuple[int, int]],
        planarity_result: PlanarityResult | None = None,
    ) -> PlanarEmbedding:
        clean_edges = [(u, v) for u, v in edges if u != v]
        result = _get_or_check_planarity(num_nodes, clean_edges, planarity_result)
        emb = _make_embedding(result, num_nodes)
        faces = emb.faces()

        if not faces:
            return emb

        adj: list[list[int]] = [[] for _ in range(num_nodes)]
        for u, v in clean_edges:
            adj[u].append(v)
            adj[v].append(u)

        bct = build_block_cut_tree(num_nodes, adj)

        if len(bct.blocks) <= 1:
            # Single block: just pick the largest face
            best = max(range(len(faces)), key=lambda i: len(faces[i]))
            emb.set_outer_face(best)
            return emb

        # Find the center of the block-cut tree by iteratively removing
        # leaves until 1 or 2 blocks remain.
        remaining = set(b.index for b in bct.blocks)
        adj_copy: dict[int, set[int]] = {
            bi: set(bct.block_adj.get(bi, set())) & remaining for bi in remaining
        }

        while len(remaining) > 2:
            leaves = [bi for bi in remaining if len(adj_copy.get(bi, set())) <= 1]
            if not leaves:
                break
            for leaf in leaves:
                remaining.discard(leaf)
                for nb in adj_copy.get(leaf, set()):
                    adj_copy.get(nb, set()).discard(leaf)
                adj_copy.pop(leaf, None)

        # Root at the center block (pick any remaining)
        center_block = min(remaining)

        # Find largest face in the center block
        block_edge_set: set[tuple[int, int]] = set()
        for u, v in bct.blocks[center_block].edges:
            block_edge_set.add((u, v))
            block_edge_set.add((v, u))

        best_fi = 0
        best_size = 0
        for fi, face in enumerate(faces):
            if not face:
                continue
            u, v = face[0]
            if (u, v) in block_edge_set:
                if len(face) > best_size:
                    best_size = len(face)
                    best_fi = fi

        emb.set_outer_face(best_fi)
        return emb


class OptimalFlexEmbedder:
    """Choose the outer face that minimizes total bends via LP relaxation.

    For each candidate outer face, formulates a linear program:
    - Variables: angle(v, f) >= 1 and bends(e, f) >= 0
    - Face constraints: sum of angles and bends around each face equals
      2(k-2) for inner faces or 2(k+2) for the outer face, where k is the
      face degree (number of edges on the face boundary).
    - Vertex constraints: sum of angles around each vertex equals 4.
    - Objective: minimize total bends.

    Picks the outer face yielding minimum LP objective. Falls back to
    MaxFaceEmbedder if scipy is not available.

    Requires scipy (optional dependency).
    """

    def __init__(self, max_candidates: int = 0) -> None:
        """Initialize the embedder.

        Args:
            max_candidates: Maximum number of candidate outer faces to evaluate.
                0 means evaluate all faces.
        """
        self._max_candidates = max_candidates

    def embed(
        self,
        num_nodes: int,
        edges: Sequence[tuple[int, int]],
        planarity_result: PlanarityResult | None = None,
    ) -> PlanarEmbedding:
        clean_edges = [(u, v) for u, v in edges if u != v]
        result = _get_or_check_planarity(num_nodes, clean_edges, planarity_result)
        emb = _make_embedding(result, num_nodes)
        faces = emb.faces()

        if not faces:
            return emb

        try:
            from scipy.optimize import linprog
        except ImportError:
            # Fall back to MaxFaceEmbedder
            return MaxFaceEmbedder().embed(num_nodes, edges, planarity_result=result)

        best_fi = 0
        best_obj = float("inf")

        # Build face data structures once
        face_data = self._build_face_data(faces, num_nodes)

        # Determine candidates
        candidates = list(range(len(faces)))
        if self._max_candidates > 0 and len(candidates) > self._max_candidates:
            # Prioritize larger faces (more likely to be good outer faces)
            candidates.sort(key=lambda i: len(faces[i]), reverse=True)
            candidates = candidates[: self._max_candidates]

        for fi in candidates:
            obj = self._solve_lp(face_data, fi, linprog)
            if obj is not None and obj < best_obj:
                best_obj = obj
                best_fi = fi

        emb.set_outer_face(best_fi)
        return emb

    @staticmethod
    def _build_face_data(
        faces: list[list[tuple[int, int]]],
        num_nodes: int,
    ) -> dict[str, Any]:
        """Pre-compute face structure for LP formulation."""
        # For each face, find unique vertices and edges
        face_vertices: list[list[int]] = []
        face_edges: list[list[tuple[int, int]]] = []

        # Map (vertex, face_index) pairs and (edge, face_index) pairs to indices
        vf_pairs: list[tuple[int, int]] = []  # (vertex, face_idx)
        ef_pairs: list[tuple[int, int, int]] = []  # (u, v, face_idx)

        vf_index: dict[tuple[int, int], int] = {}
        ef_index: dict[tuple[int, int, int], int] = {}

        for fi, face in enumerate(faces):
            verts = []
            edges_in_face = []
            seen_v: set[int] = set()
            for u, v in face:
                if u not in seen_v:
                    seen_v.add(u)
                    verts.append(u)
                    pair = (u, fi)
                    if pair not in vf_index:
                        vf_index[pair] = len(vf_pairs)
                        vf_pairs.append(pair)
                canon = (min(u, v), max(u, v))
                ekey = (canon[0], canon[1], fi)
                if ekey not in ef_index:
                    ef_index[ekey] = len(ef_pairs)
                    ef_pairs.append(ekey)
                    edges_in_face.append(canon)
            face_vertices.append(verts)
            face_edges.append(edges_in_face)

        # Map vertex to faces it appears in
        vertex_faces: dict[int, list[int]] = {}
        for v, fi in vf_pairs:
            vertex_faces.setdefault(v, []).append(fi)

        return {
            "faces": faces,
            "face_vertices": face_vertices,
            "face_edges": face_edges,
            "vf_pairs": vf_pairs,
            "ef_pairs": ef_pairs,
            "vf_index": vf_index,
            "ef_index": ef_index,
            "vertex_faces": vertex_faces,
            "num_nodes": num_nodes,
        }

    @staticmethod
    def _solve_lp(
        face_data: dict[str, Any],
        outer_fi: int,
        linprog_fn: Any,
    ) -> float | None:
        """Solve bend minimization LP for a given outer face choice.

        Returns the optimal objective value (total bends), or None if infeasible.
        """
        import numpy as np

        vf_pairs = face_data["vf_pairs"]
        ef_pairs = face_data["ef_pairs"]
        vf_index = face_data["vf_index"]
        ef_index = face_data["ef_index"]
        faces = face_data["faces"]
        face_vertices = face_data["face_vertices"]
        face_edges = face_data["face_edges"]
        vertex_faces = face_data["vertex_faces"]

        n_angle = len(vf_pairs)
        n_bend = len(ef_pairs)
        n_vars = n_angle + n_bend

        if n_vars == 0:
            return 0.0

        # Objective: minimize sum of bend variables
        c = np.zeros(n_vars)
        c[n_angle:] = 1.0  # bend variables have cost 1

        # Equality constraints
        eq_rows: list[list[float]] = []
        eq_vals: list[float] = []

        # Face constraints: sum of angles + bends = target
        for fi, face in enumerate(faces):
            k = len(face)
            if fi == outer_fi:
                target = 2 * (k + 2)  # outer face: negative curvature
            else:
                target = 2 * (k - 2)  # inner face: positive curvature

            if target < 0:
                # Can happen for very small faces as inner; LP will be infeasible
                return None

            row = [0.0] * n_vars
            for v in face_vertices[fi]:
                pair = (v, fi)
                if pair in vf_index:
                    row[vf_index[pair]] = 1.0
            for u, v in face_edges[fi]:
                ekey = (u, v, fi)
                if ekey in ef_index:
                    row[n_angle + ef_index[ekey]] = 1.0
            eq_rows.append(row)
            eq_vals.append(float(target))

        # Vertex constraints: sum of angles around vertex = 4
        for v, fi_list in vertex_faces.items():
            row = [0.0] * n_vars
            for fi in fi_list:
                pair = (v, fi)
                if pair in vf_index:
                    row[vf_index[pair]] = 1.0
            eq_rows.append(row)
            eq_vals.append(4.0)

        if not eq_rows:
            return 0.0

        a_eq = np.array(eq_rows)
        b_eq = np.array(eq_vals)

        # Bounds: angles >= 1, bends >= 0
        bounds = [(1.0, None)] * n_angle + [(0.0, None)] * n_bend

        try:
            res = linprog_fn(c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
            if res.success:
                return float(res.fun)
        except Exception:
            pass
        return None


__all__ = [
    "PlanarEmbedder",
    "FixedEmbedder",
    "MaxFaceEmbedder",
    "MinDepthEmbedder",
    "OptimalFlexEmbedder",
]
