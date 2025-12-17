"""
Gradient descent for graph layout stress minimization.

This module implements gradient descent with Runge-Kutta integration
to minimize stress in graph layouts with ideal edge lengths.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np


class Locks:
    """
    Manages locks over nodes that should not move during descent.
    """

    def __init__(self) -> None:
        self.locks: dict[int, np.ndarray] = {}

    def add(self, id: int, x: np.ndarray) -> None:
        """
        Add a lock on the node at index id.

        Args:
            id: Index of node to be locked
            x: Required position for node (k-dimensional array)
        """
        self.locks[id] = x

    def clear(self) -> None:
        """Clear all locks."""
        self.locks = {}

    def is_empty(self) -> bool:
        """Check if no locks exist."""
        return len(self.locks) == 0

    def apply(self, f: Callable[[int, np.ndarray], None]) -> None:
        """
        Perform an operation on each lock.

        Args:
            f: Function to apply to each (id, position) pair
        """
        for id, x in self.locks.items():
            f(id, x)


class PseudoRandom:
    """Linear congruential pseudo random number generator."""

    def __init__(self, seed: int = 1):
        self.seed = seed
        self.a = 214013
        self.c = 2531011
        self.m = 2147483648
        self.range = 32767

    def get_next(self) -> float:
        """Get random real between 0 and 1."""
        self.seed = (self.seed * self.a + self.c) % self.m
        return (self.seed >> 16) / self.range

    def get_next_between(self, min_val: float, max_val: float) -> float:
        """Get random real between min and max."""
        return min_val + self.get_next() * (max_val - min_val)


class Descent:
    """
    Uses gradient descent to reduce stress over a graph with ideal edge lengths.

    The standard stress function over graph nodes with position vectors x,y,z is:
    stress = Sum[w[i,j] * (length[i,j] - D[i,j])^2]
    where D is a square matrix of ideal separations between nodes,
    w is matrix of weights for those separations (wij = 1/(Dij^2))
    """

    # Threshold for treating distances as effectively zero.
    # Used to detect coincident nodes and avoid division by zero.
    # Value chosen to be well below typical graph distances while
    # remaining above floating-point noise (~1e-15).
    ZERO_DISTANCE = 1e-10

    # Squared distance epsilon for numerical stability in vectorized operations.
    # Slightly larger than ZERO_DISTANCE^2 to provide robust sqrt() behavior.
    MIN_DIST_SQ = 1e-9

    def __init__(self, x: np.ndarray, D: np.ndarray, G: Optional[np.ndarray] = None):
        """
        Initialize gradient descent.

        Args:
            x: Initial coordinates for nodes (k x n array, k dimensions, n nodes)
            D: Matrix of desired distances between pairs of nodes (n x n)
            G: Optional matrix of weights for goal terms between pairs.
               If G[i][j] > 1 and distance > ideal, no contribution to goal.
               If G[i][j] <= 1, used as weighting on variance contribution.
        """
        self.x = x
        self.D = D
        self.G = G

        self.k = x.shape[0]  # dimensionality
        self.n = x.shape[1]  # number of nodes

        # Convergence threshold
        self.threshold = 0.0001

        # Hessian matrix (k x n x n)
        self.H = np.zeros((self.k, self.n, self.n))

        # Gradient vector (k x n)
        self.g = np.zeros((self.k, self.n))

        # Working arrays
        self.Hd = np.zeros((self.k, self.n))
        self.a = np.zeros((self.k, self.n))
        self.b = np.zeros((self.k, self.n))
        self.c = np.zeros((self.k, self.n))
        self.d = np.zeros((self.k, self.n))
        self.e = np.zeros((self.k, self.n))
        self.ia = np.zeros((self.k, self.n))
        self.ib = np.zeros((self.k, self.n))
        self.xtmp = np.zeros((self.k, self.n))

        # Locks
        self.locks = Locks()

        # Find minimum distance
        self.min_d = float("inf")
        for i in range(self.n):
            for j in range(i + 1, self.n):
                d = float(D[i, j])
                if d > 0 and d < self.min_d:
                    self.min_d = d
        if self.min_d == float("inf"):
            self.min_d = 1.0

        # Grid snap parameters
        self.num_grid_snap_nodes = 0
        self.snap_grid_size = 100.0
        self.snap_strength = 1000.0
        self.scale_snap_by_max_h = False

        # Random number generator
        self.random = PseudoRandom()

        # Projection functions
        self.project: Optional[list[Callable[[np.ndarray, np.ndarray, np.ndarray], None]]] = None

    @staticmethod
    def create_square_matrix(n: int, f: Callable[[int, int], float]) -> np.ndarray:
        """
        Create an n x n matrix using function f(i, j).

        Args:
            n: Matrix size
            f: Function to compute matrix elements

        Returns:
            n x n NumPy array
        """
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                M[i, j] = f(i, j)
        return M

    def offset_dir(self) -> np.ndarray:
        """Generate random offset direction vector."""
        u = np.array([self.random.get_next_between(0.01, 1) - 0.5 for _ in range(self.k)])
        l = np.linalg.norm(u)
        return u * (self.min_d / l)

    def compute_derivatives(self, x: np.ndarray) -> None:
        """
        Compute first and second derivatives, storing results in self.g and self.H.

        Vectorized implementation using NumPy broadcasting for performance.

        Args:
            x: Current positions (k x n)
        """
        n = self.n
        if n < 1:
            return

        # Compute all pairwise differences using broadcasting
        # Shape: (k, n, n) where diff[:, u, v] = x[:, u] - x[:, v]
        diff = x[:, :, np.newaxis] - x[:, np.newaxis, :]  # (k, n, n)

        # Squared differences per dimension: (k, n, n)
        diff_squared = diff**2

        # Sum over dimensions to get squared distances: (n, n)
        dist_squared = np.sum(diff_squared, axis=0)

        # Create mask for diagonal (self-pairs) and near-zero distances
        diagonal_mask = np.eye(n, dtype=bool)

        # Distances: (n, n)
        # Use MIN_DIST_SQ to avoid sqrt(0) and ensure numerical stability
        distances = np.sqrt(np.maximum(dist_squared, self.MIN_DIST_SQ))

        # Get weights from G matrix
        if self.G is None:
            weights = np.ones((n, n))
        else:
            weights = self.G.copy()

        # Normalize weights > 1 to 1.0 (they were just indicators)
        # But first, use them to filter P-stress
        # Ignore long range attractions: weight > 1 and distance > ideal
        p_stress_mask = (weights > 1) & (distances > self.D)
        weights = np.where(weights > 1, 1.0, weights)

        # Filter out diagonal, non-finite ideal distances, and P-stress cases
        valid_mask = ~diagonal_mask & np.isfinite(self.D) & ~p_stress_mask

        # Ideal distances - use 1.0 as safe default to avoid division by zero
        # Note: we must set safe defaults BEFORE computing derived values
        ideal_dist = np.where(valid_mask & (self.D > 0), self.D, 1.0)
        ideal_dist_sq = ideal_dist**2

        # Safe distances for division (use 1.0 where invalid to avoid div-by-zero)
        safe_distances = np.where(valid_mask & (distances > 0), distances, 1.0)
        safe_dist_sq = np.where(valid_mask & (dist_squared > 0), dist_squared, 1.0)
        safe_dist_cubed = safe_dist_sq * safe_distances

        # Gradient scalar: gs = 2 * weight * (distance - ideal) / (ideal^2 * distance)
        # Shape: (n, n)
        # Compute safely then mask - avoids numpy computing invalid values
        gs_denom = ideal_dist_sq * safe_distances
        gs = np.where(valid_mask, 2 * weights * (safe_distances - ideal_dist) / gs_denom, 0.0)

        # Hessian scalar: hs = -2 * weight / (ideal^2 * distance^3)
        hs_denom = ideal_dist_sq * safe_dist_cubed
        hs = np.where(valid_mask, -2 * weights / hs_denom, 0.0)

        # Compute gradients: sum over all v for each u
        # g[i, u] = sum_v(diff[i, u, v] * gs[u, v])
        # Broadcasting: diff (k, n, n) * gs (n, n) -> (k, n, n), then sum over v
        self.g = np.sum(diff * gs[np.newaxis, :, :], axis=2)  # (k, n)

        # Compute Hessian off-diagonal elements
        # H[i, u, v] = hs[u, v] * (2 * dist^3 + ideal * (diff^2[i] - dist^2))
        # Shape: (k, n, n)
        hessian_term = 2 * safe_dist_cubed[np.newaxis, :, :] + ideal_dist[np.newaxis, :, :] * (
            diff_squared - safe_dist_sq[np.newaxis, :, :]
        )
        self.H = np.where(valid_mask[np.newaxis, :, :], hs[np.newaxis, :, :] * hessian_term, 0.0)

        # Compute diagonal elements: Huu[i] = -sum_v(H[i, u, v])
        Huu = -np.sum(self.H, axis=2)  # (k, n)

        # Set diagonal elements
        for i in range(self.k):
            np.fill_diagonal(self.H[i], Huu[i])

        # Max Hessian value for scaling locks and grid snap
        max_h = np.max(np.abs(Huu)) if Huu.size > 0 else 0.0

        # Grid snap forces
        r = self.snap_grid_size / 2
        g = self.snap_grid_size
        w = self.snap_strength
        k_snap = w / (r * r)
        num_nodes = self.num_grid_snap_nodes

        for u in range(num_nodes):
            for i in range(self.k):
                xiu = self.x[i, u]
                m = xiu / g
                f = m % 1
                q = m - f
                a = abs(f)
                if a <= 0.5:
                    dx = xiu - q * g
                else:
                    dx = xiu - ((q + 1) * g if xiu > 0 else (q - 1) * g)

                if -r < dx <= r:
                    if self.scale_snap_by_max_h:
                        self.g[i, u] += max_h * k_snap * dx
                        self.H[i, u, u] += max_h * k_snap
                    else:
                        self.g[i, u] += k_snap * dx
                        self.H[i, u, u] += k_snap

        # Apply locks
        if not self.locks.is_empty():

            def apply_lock(u: int, p: np.ndarray) -> None:
                for i in range(self.k):
                    self.H[i, u, u] += max_h
                    self.g[i, u] -= max_h * (p[i] - x[i, u])

            self.locks.apply(apply_lock)

    def compute_step_size(self, d: np.ndarray) -> float:
        """
        Compute optimal step size to take in direction d.

        Args:
            d: Descent direction (k x n)

        Returns:
            Scalar multiplier for optimal step
        """
        numerator = 0.0
        denominator = 0.0

        for i in range(self.k):
            numerator += np.dot(self.g[i], d[i])
            self.Hd[i] = np.dot(self.H[i], d[i])
            denominator += np.dot(d[i], self.Hd[i])

        if denominator == 0 or not math.isfinite(denominator):
            return 0.0

        return numerator / denominator

    def take_descent_step(self, x: np.ndarray, d: np.ndarray, step_size: float) -> None:
        """
        Take a descent step: x = x - step_size * d.

        Args:
            x: Position vector to update (modified in place)
            d: Descent direction
            step_size: Step size
        """
        x -= step_size * d

    def step_and_project(
        self, x0: np.ndarray, r: np.ndarray, d: np.ndarray, step_size: float
    ) -> None:
        """
        Take a step and project against constraints.

        Args:
            x0: Starting positions (k x n)
            r: Result positions (k x n, modified in place)
            d: Unconstrained descent vector (k x n)
            step_size: Step size
        """
        r[:] = x0

        # Projection only supported for 2D+ layouts
        if self.k >= 2 and self.project:
            # Step in dimension 0
            self.take_descent_step(r[0], d[0], step_size)
            self.project[0](x0[0], x0[1], r[0])

            # Step in dimension 1
            self.take_descent_step(r[1], d[1], step_size)
            self.project[1](r[0], x0[1], r[1])

            # Step in higher dimensions
            for i in range(2, self.k):
                self.take_descent_step(r[i], d[i], step_size)
        else:
            # No projection or 1D case - simple step in all dimensions
            for i in range(self.k):
                self.take_descent_step(r[i], d[i], step_size)

    def compute_next_position(self, x0: np.ndarray, r: np.ndarray) -> None:
        """
        Compute next position using gradient descent with projection.

        Args:
            x0: Current position (k x n)
            r: Result position (k x n, modified in place)
        """
        self.compute_derivatives(x0)
        alpha = self.compute_step_size(self.g)
        self.step_and_project(x0, r, self.g, alpha)

        if self.project:
            self.e[:] = x0 - r
            beta = self.compute_step_size(self.e)
            beta = max(0.2, min(beta, 1.0))
            self.step_and_project(x0, r, self.e, beta)

    def runge_kutta(self) -> float:
        """
        Perform one Runge-Kutta integration step.

        Returns:
            Displacement (squared sum of position changes)
        """
        self.compute_next_position(self.x, self.a)
        self.ia[:] = self.x + (self.a - self.x) / 2.0
        self.compute_next_position(self.ia, self.b)
        self.ib[:] = self.x + (self.b - self.x) / 2.0
        self.compute_next_position(self.ib, self.c)
        self.compute_next_position(self.c, self.d)

        # Compute weighted average
        new_x = (self.a + 2.0 * self.b + 2.0 * self.c + self.d) / 6.0
        diff = self.x - new_x
        disp: float = float(np.sum(diff * diff))
        self.x[:] = new_x

        return disp

    def run(self, iterations: int) -> float:
        """
        Run descent for specified number of iterations or until convergence.

        Args:
            iterations: Maximum number of iterations

        Returns:
            Final stress value
        """
        stress = float("inf")
        converged = False

        while not converged and iterations > 0:
            iterations -= 1
            s = self.runge_kutta()
            converged = abs(stress / s - 1) < self.threshold if s != 0 else False
            stress = s

        return stress

    def reduce_stress(self) -> float:
        """
        Perform one descent step to reduce stress.

        Returns:
            Current stress value
        """
        self.compute_derivatives(self.x)
        alpha = self.compute_step_size(self.g)
        for i in range(self.k):
            self.take_descent_step(self.x[i], self.g[i], alpha)
        return self.compute_stress()

    def compute_stress(self) -> float:
        """
        Compute current stress value.

        Returns:
            Stress (sum of squared normalized distance errors)
        """
        stress = 0.0
        for u in range(self.n - 1):
            for v in range(u + 1, self.n):
                # Compute Euclidean distance
                diff = self.x[:, u] - self.x[:, v]
                l = math.sqrt(np.sum(diff * diff))

                d = self.D[u, v]
                if not math.isfinite(d):
                    continue

                rl = d - l
                d2 = d * d
                stress += rl * rl / d2

        return stress
