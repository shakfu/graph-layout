"""
Spectral layout algorithm.

Uses eigenvectors of the graph Laplacian matrix to position nodes.
This produces layouts that reveal graph structure and clustering.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, cast

import numpy as np

from ..base import StaticLayout
from ..types import (
    Event,
    GroupLike,
    LinkLike,
    NodeLike,
    SizeType,
)


class SpectralLayout(StaticLayout):
    """
    Spectral layout using Laplacian eigenvectors.

    Positions nodes using the eigenvectors corresponding to the smallest
    non-zero eigenvalues of the graph Laplacian. This tends to place
    connected nodes close together and reveal community structure.

    Example:
        layout = SpectralLayout(
            nodes=[{}, {}, {}, {}, {}],
            links=[
                {'source': 0, 'target': 1},
                {'source': 1, 'target': 2},
                {'source': 2, 'target': 3},
                {'source': 3, 'target': 4},
            ],
            size=(800, 600),
        )
        layout.run()
    """

    def __init__(
        self,
        *,
        nodes: Optional[Sequence[NodeLike]] = None,
        links: Optional[Sequence[LinkLike]] = None,
        groups: Optional[Sequence[GroupLike]] = None,
        size: SizeType = (1.0, 1.0),
        random_seed: Optional[int] = None,
        on_start: Optional[Callable[[Optional[Event]], None]] = None,
        on_tick: Optional[Callable[[Optional[Event]], None]] = None,
        on_end: Optional[Callable[[Optional[Event]], None]] = None,
        # Spectral-specific parameters
        dimension: int = 2,
        normalized: bool = True,
    ) -> None:
        """
        Initialize Spectral layout.

        Args:
            nodes: List of nodes
            links: List of links
            groups: List of groups
            size: Canvas size as (width, height)
            random_seed: Random seed for reproducible layouts
            on_start: Callback for start event
            on_tick: Callback for tick event
            on_end: Callback for end event
            dimension: Layout dimension (2 or 3).
            normalized: Whether to use normalized Laplacian. Normalized
                Laplacian often produces better layouts for graphs with
                varying node degrees.
        """
        super().__init__(
            nodes=nodes,
            links=links,
            groups=groups,
            size=size,
            random_seed=random_seed,
            on_start=on_start,
            on_tick=on_tick,
            on_end=on_end,
        )

        # Spectral-specific configuration
        self._dimension: int = max(2, min(3, int(dimension)))
        self._normalized: bool = bool(normalized)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """Get layout dimension (2 or 3)."""
        return self._dimension

    @dimension.setter
    def dimension(self, value: int) -> None:
        """Set layout dimension (clamped to 2-3)."""
        self._dimension = max(2, min(3, int(value)))

    @property
    def normalized(self) -> bool:
        """Get whether normalized Laplacian is used."""
        return self._normalized

    @normalized.setter
    def normalized(self, value: bool) -> None:
        """Set whether to use normalized Laplacian."""
        self._normalized = bool(value)

    # -------------------------------------------------------------------------
    # Laplacian Computation
    # -------------------------------------------------------------------------

    def _compute_adjacency_matrix(self) -> np.ndarray:
        """Compute the adjacency matrix."""
        n = len(self._nodes)
        A = np.zeros((n, n))

        for link in self._links:
            src = self._get_source_index(link)
            tgt = self._get_target_index(link)
            weight = link.weight if link.weight else 1.0
            A[src, tgt] = weight
            A[tgt, src] = weight  # Symmetric for undirected

        return cast(np.ndarray, A)

    def _compute_laplacian(self) -> np.ndarray:
        """Compute the graph Laplacian matrix."""
        A = self._compute_adjacency_matrix()

        # Degree matrix
        D = np.diag(np.sum(A, axis=1))

        # Laplacian: L = D - A
        L = D - A

        if self._normalized:
            # Normalized Laplacian: L_norm = D^(-1/2) * L * D^(-1/2)
            # Equivalent to: I - D^(-1/2) * A * D^(-1/2)
            d = np.diag(D)
            d_inv_sqrt = np.zeros_like(d)
            nonzero = d > 0
            d_inv_sqrt[nonzero] = 1.0 / np.sqrt(d[nonzero])
            D_inv_sqrt = np.diag(d_inv_sqrt)
            L = D_inv_sqrt @ L @ D_inv_sqrt

        return cast(np.ndarray, L)

    # -------------------------------------------------------------------------
    # Layout Computation
    # -------------------------------------------------------------------------

    def _compute(self, **kwargs: Any) -> None:
        """Compute spectral layout positions."""
        n = len(self._nodes)
        if n == 0:
            return

        if n == 1:
            # Single node at center
            self._nodes[0].x = self._canvas_size[0] / 2
            self._nodes[0].y = self._canvas_size[1] / 2
            return

        # Compute Laplacian
        L = self._compute_laplacian()

        # Compute eigenvalues and eigenvectors
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
        except np.linalg.LinAlgError:
            # Fallback to random positions if eigendecomposition fails
            self._initialize_positions(random_init=True)
            return

        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Use eigenvectors corresponding to smallest non-zero eigenvalues
        # The first eigenvector (eigenvalue ~0) is constant, skip it
        dim = min(self._dimension, n - 1)

        # Find first non-trivial eigenvectors
        coords = []
        for i in range(1, min(dim + 1, n)):
            coords.append(eigenvectors[:, i])

        # If we don't have enough eigenvectors, pad with zeros
        while len(coords) < self._dimension:
            coords.append(np.zeros(n))

        # Scale to fit canvas
        padding = 50
        for d in range(len(coords)):
            vec = coords[d]
            if np.max(vec) - np.min(vec) > 1e-10:
                # Normalize to [0, 1]
                vec = (vec - np.min(vec)) / (np.max(vec) - np.min(vec))
            else:
                vec = np.full(n, 0.5)
            coords[d] = vec

        # Apply positions
        canvas_w = self._canvas_size[0] - 2 * padding
        canvas_h = self._canvas_size[1] - 2 * padding

        for i in range(n):
            self._nodes[i].x = padding + coords[0][i] * canvas_w
            if len(coords) > 1:
                self._nodes[i].y = padding + coords[1][i] * canvas_h
            else:
                self._nodes[i].y = self._canvas_size[1] / 2


__all__ = ["SpectralLayout"]
