"""Planar embedding (rotation system) data structure."""

from __future__ import annotations

from typing import Optional


class PlanarEmbedding:
    """A combinatorial planar embedding stored as a rotation system.

    A rotation system assigns to each vertex a cyclic (clockwise) ordering
    of its incident edges. Together with the graph, this uniquely determines
    a planar drawing up to homeomorphism.

    Attributes:
        rotation: Mapping from vertex to its clockwise neighbor order.
    """

    __slots__ = ("rotation",)

    def __init__(self, rotation: dict[int, list[int]]) -> None:
        self.rotation = rotation

    def faces(self) -> list[list[tuple[int, int]]]:
        """Enumerate all faces as lists of directed edges.

        Each face is a cycle of directed edges (u, v) traversed in a
        consistent orientation. Every directed edge appears in exactly one face.

        Returns:
            List of faces. Each face is a list of (u, v) directed edge tuples.
        """
        # Build next-edge-in-face lookup:
        # For directed edge (u, v), the next edge in the face is determined by
        # finding v's rotation, locating u in it, and taking the *previous*
        # entry (since we traverse faces by turning right = clockwise).
        visited_edges: set[tuple[int, int]] = set()
        face_list: list[list[tuple[int, int]]] = []

        # Build index lookup for each vertex's rotation
        rot_index: dict[int, dict[int, int]] = {}
        for v, neighbors in self.rotation.items():
            rot_index[v] = {w: i for i, w in enumerate(neighbors)}

        for v, neighbors in self.rotation.items():
            for w in neighbors:
                if (v, w) in visited_edges:
                    continue
                # Trace the face starting from (v, w)
                face: list[tuple[int, int]] = []
                u, cur = v, w
                while True:
                    if (u, cur) in visited_edges:
                        break
                    visited_edges.add((u, cur))
                    face.append((u, cur))

                    # Next edge: at vertex 'cur', find 'u' in rotation,
                    # and take the previous neighbor
                    if cur not in rot_index or u not in rot_index[cur]:
                        break
                    idx = rot_index[cur][u]
                    nbrs = self.rotation[cur]
                    next_v = nbrs[(idx - 1) % len(nbrs)]

                    u, cur = cur, next_v

                if face:
                    face_list.append(face)

        return face_list

    def num_faces(self) -> int:
        """Count the number of faces.

        For a connected planar graph, Euler's formula gives F = E - V + 2.
        """
        return len(self.faces())

    def outer_face(self) -> Optional[list[tuple[int, int]]]:
        """Return the outer (largest) face.

        Heuristic: the face with the most edges is typically the outer face.

        Returns:
            The face with the most edges, or None if no faces exist.
        """
        all_faces = self.faces()
        if not all_faces:
            return None
        return max(all_faces, key=len)

    def verify(self) -> bool:
        """Verify basic embedding consistency.

        Checks that every directed edge (u, v) has a corresponding (v, u)
        somewhere in the rotation system.

        Returns:
            True if the embedding is consistent.
        """
        directed_edges: set[tuple[int, int]] = set()
        for v, neighbors in self.rotation.items():
            for w in neighbors:
                directed_edges.add((v, w))

        for u, v in directed_edges:
            if (v, u) not in directed_edges:
                return False

        return True
