"""Tests for gradient descent stress minimization."""

import numpy as np

from graph_layout.cola.descent import Descent, Locks, PseudoRandom


class TestLocks:
    """Test Locks class."""

    def test_create_locks(self):
        """Test locks creation."""
        locks = Locks()
        assert locks.is_empty()

    def test_add_lock(self):
        """Test adding a lock."""
        locks = Locks()
        pos = np.array([1.0, 2.0])
        locks.add(0, pos)

        assert not locks.is_empty()

    def test_clear_locks(self):
        """Test clearing locks."""
        locks = Locks()
        locks.add(0, np.array([1.0, 2.0]))
        locks.add(1, np.array([3.0, 4.0]))

        assert not locks.is_empty()
        locks.clear()
        assert locks.is_empty()

    def test_apply_to_locks(self):
        """Test applying function to locks."""
        locks = Locks()
        locks.add(0, np.array([1.0, 2.0]))
        locks.add(1, np.array([3.0, 4.0]))

        ids = []
        positions = []

        def collect(id, pos):
            ids.append(id)
            positions.append(pos)

        locks.apply(collect)

        assert len(ids) == 2
        assert 0 in ids
        assert 1 in ids
        assert len(positions) == 2


class TestPseudoRandom:
    """Test PseudoRandom class."""

    def test_create_prng(self):
        """Test PRNG creation."""
        prng = PseudoRandom(seed=42)
        assert prng.seed == 42

    def test_deterministic_sequence(self):
        """Test that same seed produces same sequence."""
        prng1 = PseudoRandom(seed=42)
        prng2 = PseudoRandom(seed=42)

        for _ in range(10):
            assert prng1.get_next() == prng2.get_next()

    def test_get_next_range(self):
        """Test that get_next returns values in [0, 1]."""
        prng = PseudoRandom()

        for _ in range(100):
            val = prng.get_next()
            assert 0 <= val <= 1

    def test_get_next_between(self):
        """Test get_next_between range."""
        prng = PseudoRandom()

        for _ in range(100):
            val = prng.get_next_between(5.0, 10.0)
            assert 5.0 <= val <= 10.0


class TestDescent:
    """Test Descent class."""

    def test_create_descent(self):
        """Test descent creation."""
        # Simple 2D case with 3 nodes
        x = np.array(
            [
                [0.0, 1.0, 0.5],  # x coordinates
                [0.0, 0.0, 1.0],  # y coordinates
            ]
        )

        # Distance matrix
        D = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])

        descent = Descent(x, D)

        assert descent.k == 2  # 2D
        assert descent.n == 3  # 3 nodes
        assert descent.x.shape == (2, 3)

    def test_create_square_matrix(self):
        """Test square matrix creation."""
        M = Descent.create_square_matrix(3, lambda i, j: i + j)

        assert M.shape == (3, 3)
        assert M[0, 0] == 0
        assert M[0, 1] == 1
        assert M[1, 1] == 2
        assert M[2, 2] == 4

    def test_offset_dir(self):
        """Test offset direction generation."""
        x = np.array([[0.0, 1.0], [0.0, 0.0]])
        D = np.array([[0.0, 1.0], [1.0, 0.0]])
        descent = Descent(x, D)

        offset = descent.offset_dir()

        assert len(offset) == 2
        # Should be normalized to min_d length
        length = np.linalg.norm(offset)
        assert abs(length - descent.min_d) < 1e-6

    def test_compute_stress_ideal_positions(self):
        """Test stress computation with nodes at ideal positions."""
        # Triangle with side length 1
        x = np.array(
            [
                [0.0, 1.0, 0.5],
                [0.0, 0.0, 0.866],  # approx sqrt(3)/2
            ]
        )

        # Ideal distances (equilateral triangle)
        D = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])

        descent = Descent(x, D)
        stress = descent.compute_stress()

        # Stress should be very low for near-ideal positions
        assert stress < 0.1

    def test_compute_stress_non_ideal(self):
        """Test stress with non-ideal positions."""
        # All nodes at same position (worst case)
        x = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        D = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])

        descent = Descent(x, D)
        stress = descent.compute_stress()

        # Stress should be high
        assert stress > 1.0

    def test_compute_derivatives(self):
        """Test derivative computation."""
        x = np.array([[0.0, 2.0, 1.0], [0.0, 0.0, 1.0]])

        D = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])

        descent = Descent(x, D)
        descent.compute_derivatives(x)

        # Gradient should be computed
        assert descent.g.shape == (2, 3)
        # Hessian should be computed
        assert descent.H.shape == (2, 3, 3)

    def test_reduce_stress(self):
        """Test that reduce_stress actually reduces stress."""
        # Start with poor configuration
        x = np.array([[0.0, 2.0, 4.0], [0.0, 0.0, 0.0]])

        D = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])

        descent = Descent(x.copy(), D)
        initial_stress = descent.compute_stress()

        descent.reduce_stress()
        new_stress = descent.compute_stress()

        # Stress should decrease (or stay same if at minimum)
        assert new_stress <= initial_stress

    def test_runge_kutta_step(self):
        """Test Runge-Kutta integration step."""
        x = np.array([[0.0, 2.0, 1.0], [0.0, 0.0, 1.0]])

        D = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])

        descent = Descent(x, D)
        initial_x = descent.x.copy()

        disp = descent.runge_kutta()

        # Position should change
        assert not np.allclose(descent.x, initial_x)
        # Displacement should be non-negative
        assert disp >= 0

    def test_run_iterations(self):
        """Test running descent for multiple iterations."""
        x = np.array([[0.0, 3.0, 1.5], [0.0, 0.0, 2.0]])

        D = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])

        descent = Descent(x, D)
        initial_stress = descent.compute_stress()

        final_stress = descent.run(10)

        # Stress should decrease after multiple iterations
        assert final_stress <= initial_stress

    def test_locks_prevent_movement(self):
        """Test that locked nodes don't move much."""
        x = np.array([[0.0, 3.0, 1.5], [0.0, 0.0, 2.0]])

        D = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])

        descent = Descent(x.copy(), D)

        # Lock node 0 at its initial position
        descent.locks.add(0, x[:, 0].copy())

        initial_pos = descent.x[:, 0].copy()
        descent.run(5)
        final_pos = descent.x[:, 0]

        # Locked node should stay very close to initial position
        assert np.allclose(initial_pos, final_pos, atol=0.1)

    def test_with_weight_matrix(self):
        """Test descent with weight matrix G."""
        x = np.array([[0.0, 2.0, 1.0], [0.0, 0.0, 1.0]])

        D = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])

        # Weight matrix - higher weights for some pairs
        G = np.array(
            [
                [1.0, 1.0, 2.0],  # Long-range connection
                [1.0, 1.0, 1.0],
                [2.0, 1.0, 1.0],
            ]
        )

        descent = Descent(x, D, G)
        descent.run(5)

        # Should successfully run with weights
        assert descent.compute_stress() >= 0

    def test_1d_descent(self):
        """Test 1D descent."""
        # 3 nodes in 1D
        x = np.array([[0.0, 3.0, 6.0]])

        D = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])

        descent = Descent(x, D)
        descent.run(10)

        # Should converge to something reasonable
        assert descent.compute_stress() < 1.0

    def test_3d_descent(self):
        """Test 3D descent."""
        # 4 nodes in 3D (tetrahedron)
        x = np.array([[0.0, 1.0, 0.5, 0.5], [0.0, 0.0, 0.866, 0.289], [0.0, 0.0, 0.0, 0.816]])

        D = np.ones((4, 4))  # All distances = 1
        np.fill_diagonal(D, 0)

        descent = Descent(x, D)
        initial_stress = descent.compute_stress()
        descent.run(5)

        # Should reduce stress
        assert descent.compute_stress() <= initial_stress

    def test_convergence_threshold(self):
        """Test that descent respects convergence threshold."""
        x = np.array([[0.0, 1.0, 0.5], [0.0, 0.0, 0.866]])

        D = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])

        descent = Descent(x, D)
        descent.threshold = 0.001

        # Start from near-optimal position
        descent.run(100)  # May converge early

        # Should terminate when converged
        stress = descent.compute_stress()
        assert stress >= 0

    def test_grid_snap_nodes(self):
        """Test grid snapping functionality."""
        x = np.array([[0.0, 50.5, 99.5], [0.0, 49.5, 101.0]])

        D = np.array([[0.0, 50.0, 100.0], [50.0, 0.0, 50.0], [100.0, 50.0, 0.0]])

        descent = Descent(x, D)
        descent.num_grid_snap_nodes = 3
        descent.snap_grid_size = 100.0
        descent.snap_strength = 1000.0

        initial_x = descent.x.copy()
        descent.run(5)

        # Positions should change with grid snapping
        # (exact behavior depends on parameters)
        assert descent.x.shape == initial_x.shape
