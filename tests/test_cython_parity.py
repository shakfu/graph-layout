"""Parity between the compiled force kernels and their pure-Python fallbacks.

``graph_layout._speedups`` and the ``compute_*`` methods on the force layouts
are two implementations of the same physics. The compiled one runs whenever a
wheel is installed; the Python one runs on an sdist install with no wheel for
the platform, or after a failed build. Nothing else in the suite exercises the
second, which is how a real divergence went unnoticed once already: the
pure-Python Barnes-Hut path dropped ForceAtlas2's source-side ``(deg_j + 1)``
factor, a different force model on the default code path.

The checks here compare a *single* force evaluation from identical positions.
That is deliberate. The layouts themselves are chaotic -- a last-bit difference
in one force grows by roughly an order of magnitude per few dozen iterations,
so comparing the output of a full multi-hundred-iteration run measures the
Lyapunov exponent, not agreement, and the answer changes with the compiler and
the CPU. Comparing forces has no such amplification and gives the same result
everywhere.

The layout-level tests that remain (here and in the per-algorithm test modules)
therefore run a bounded, small number of iterations.
"""

import math

import numpy as np
import pytest

from graph_layout.force import force_atlas2 as fa2_module
from graph_layout.force import fruchterman_reingold as fr_module
from graph_layout.force import yifan_hu as yh_module
from graph_layout.force._kernel_constants import MIN_ADJUSTED_DIST, MIN_DIST_SQ

pytestmark = pytest.mark.skipif(
    not yh_module._HAS_CYTHON,
    reason="_speedups extension not built; only one path exists to compare",
)

if yh_module._HAS_CYTHON:
    from graph_layout import _speedups


# --------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------

# Both implementations evaluate the same expressions in the same order, so on a
# well-behaved build they agree bit for bit. The tolerance is here only so that
# a build which contracts `a * b + c` into an FMA (see the -ffp-contract=off
# note in CMakeLists.txt) reports the one problem it really has --
# test_naive_repulsion_is_bit_exact -- rather than failing everything.
EXACT_RTOL = 1e-12

# The two Barnes-Hut implementations are separate tree codes that sum the same
# contributions in slightly different orders, so they agree to a few ulp rather
# than exactly. Measured worst case on these graphs is ~4e-16 relative.
BARNES_HUT_RTOL = 1e-9


def _positions(n, seed=1, span=100.0):
    """Deterministic, well-separated positions."""
    rng = np.random.default_rng(seed)
    return rng.random(n) * span, rng.random(n) * span


def _grid_edges(w, h):
    """4-neighbour grid edges over ``w * h`` vertices."""
    sources, targets = [], []
    for r in range(h):
        for c in range(w):
            v = r * w + c
            if c + 1 < w:
                sources.append(v)
                targets.append(v + 1)
            if r + 1 < h:
                sources.append(v)
                targets.append(v + w)
    return (
        np.array(sources, dtype=np.int32),
        np.array(targets, dtype=np.int32),
    )


def _degenerate_positions():
    """Positions that straddle every distance floor in the kernels.

    Nodes 0/1 coincide exactly, 2/3 sit closer than ``sqrt(MIN_DIST_SQ)``, 4/5
    sit just above it, and the rest are ordinary. Only input like this
    distinguishes "clamp the distance" from "skip the pair", which is how the
    two implementations drifted apart.
    """
    below = math.sqrt(MIN_DIST_SQ) / 10.0
    above = math.sqrt(MIN_DIST_SQ) * 10.0
    xs = [0.0, 0.0, 5.0, 5.0 + below, 9.0, 9.0 + above, 20.0, 41.0, 60.5, 83.0]
    ys = [0.0, 0.0, 5.0, 5.0, 9.0, 9.0, 33.0, 12.0, 77.0, 4.5]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def assert_forces_agree(compiled, fallback, rtol, what):
    """Compare two force fields relative to the magnitude of the forces."""
    (cx, cy), (px, py) = compiled, fallback
    scale = float(np.max(np.hypot(cx, cy)))
    worst = float(np.max(np.hypot(cx - px, cy - py)))
    relative = worst / scale if scale > 0 else worst
    assert relative <= rtol, (
        f"{what}: compiled and pure-Python force fields differ by {worst:.6g} "
        f"({relative:.3g} relative to a force scale of {scale:.6g})"
    )


def _zeros(n):
    return np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)


# --------------------------------------------------------------------------
# Build hygiene
# --------------------------------------------------------------------------


class TestBuildHygiene:
    """The extension must not be built with contracted floating-point math."""

    def test_naive_repulsion_is_bit_exact(self):
        """A stray FMA in the kernel silently changes every layout it touches.

        ``dx * dx + dy * dy`` compiles to a single fused multiply-add unless
        contraction is disabled, which rounds once where CPython rounds twice.
        The resulting one-ulp difference is invisible in a force but grows into
        a visibly different drawing over a few hundred iterations, and it
        appears only on targets whose compiler contracts by default -- arm64
        macOS does, x86-64 Linux does not. CMakeLists.txt passes
        ``-ffp-contract=off`` (``/fp:precise`` on MSVC) to prevent it.
        """
        pos_x, pos_y = _positions(80)
        k_sq = 42.0

        compiled = _zeros(80)
        _speedups._compute_repulsive_forces(
            pos_x.copy(), pos_y.copy(), compiled[0], compiled[1], k_sq, 80
        )
        fallback = _zeros(80)
        _yifan_hu().compute_repulsive_naive(pos_x, pos_y, fallback[0], fallback[1], 1.0, k_sq, 80)

        assert np.array_equal(compiled[0], fallback[0]) and np.array_equal(
            compiled[1], fallback[1]
        ), (
            "the compiled repulsion kernel does not reproduce Python float "
            "arithmetic bit for bit; the extension was probably built with "
            "floating-point contraction enabled (see -ffp-contract=off in "
            "CMakeLists.txt)"
        )


# --------------------------------------------------------------------------
# Shared Fruchterman-Reingold / Yifan Hu kernels
# --------------------------------------------------------------------------


def _yifan_hu(theta=0.5):
    layout = yh_module.YifanHuLayout(nodes=[{"x": 0.0, "y": 0.0}], links=[])
    layout.barnes_hut_theta = theta
    return layout


class TestSharedForceKernels:
    """``_compute_repulsive_forces`` / ``_compute_attractive_forces``.

    Both Fruchterman-Reingold and Yifan Hu call these; the Yifan Hu wrappers
    are used to drive the comparison because they take plain arrays.
    """

    @pytest.mark.parametrize("n", [8, 64])
    def test_repulsion_matches(self, n):
        pos_x, pos_y = _positions(n)
        c, k_sq = 0.2, 42.0

        compiled = _zeros(n)
        _speedups._compute_repulsive_forces(
            pos_x.copy(), pos_y.copy(), compiled[0], compiled[1], c * k_sq, n
        )
        fallback = _zeros(n)
        _yifan_hu().compute_repulsive_naive(pos_x, pos_y, fallback[0], fallback[1], c, k_sq, n)

        assert_forces_agree(compiled, fallback, EXACT_RTOL, f"repulsion (n={n})")

    def test_attraction_matches(self):
        n = 100
        pos_x, pos_y = _positions(n)
        sources, targets = _grid_edges(10, 10)
        k = 30.0

        compiled = _zeros(n)
        _speedups._compute_attractive_forces(
            pos_x.copy(),
            pos_y.copy(),
            compiled[0],
            compiled[1],
            sources,
            targets,
            k,
            len(sources),
        )
        fallback = _zeros(n)
        _yifan_hu().compute_attractive(
            pos_x, pos_y, fallback[0], fallback[1], sources, targets, k, len(sources)
        )

        assert_forces_agree(compiled, fallback, EXACT_RTOL, "attraction")

    @pytest.mark.parametrize("theta", [0.0, 0.5, 1.2])
    @pytest.mark.parametrize("n", [60, 200])
    def test_barnes_hut_repulsion_matches(self, n, theta):
        pos_x, pos_y = _positions(n)
        c, k_sq = 0.2, 42.0

        compiled = _zeros(n)
        _speedups._compute_repulsive_forces_barnes_hut(
            pos_x.copy(), pos_y.copy(), compiled[0], compiled[1], c * k_sq, n, theta
        )
        fallback = _zeros(n)
        _yifan_hu(theta).compute_repulsive_barnes_hut(
            pos_x, pos_y, fallback[0], fallback[1], c, k_sq, n
        )

        assert_forces_agree(
            compiled, fallback, BARNES_HUT_RTOL, f"Barnes-Hut repulsion (theta={theta})"
        )

    def test_repulsion_matches_on_degenerate_separations(self):
        pos_x, pos_y = _degenerate_positions()
        n = len(pos_x)
        c, k_sq = 0.2, 42.0

        compiled = _zeros(n)
        _speedups._compute_repulsive_forces(
            pos_x.copy(), pos_y.copy(), compiled[0], compiled[1], c * k_sq, n
        )
        fallback = _zeros(n)
        _yifan_hu().compute_repulsive_naive(pos_x, pos_y, fallback[0], fallback[1], c, k_sq, n)

        assert_forces_agree(compiled, fallback, EXACT_RTOL, "degenerate repulsion")
        assert np.all(np.isfinite(compiled[0])) and np.all(np.isfinite(compiled[1]))

    def test_repulsion_is_bounded_by_the_distance_floor(self):
        """Near-coincident nodes must not produce an unbounded kick.

        The floor is the whole point of clamping rather than dividing by the
        true distance: without it a pair a nanometre apart takes a step of
        ~1e9 * k and leaves the canvas.
        """
        pos_x, pos_y = _degenerate_positions()
        n = len(pos_x)
        c, k_sq = 0.2, 42.0

        fallback = _zeros(n)
        _yifan_hu().compute_repulsive_naive(pos_x, pos_y, fallback[0], fallback[1], c, k_sq, n)

        # Force magnitude is at most c * k_sq / sqrt(MIN_DIST_SQ) per pair, and
        # there are n - 1 pairs per node.
        bound = (n - 1) * c * k_sq / math.sqrt(MIN_DIST_SQ)
        assert float(np.max(np.hypot(*fallback))) <= bound

    def test_attraction_matches_on_degenerate_separations(self):
        pos_x, pos_y = _degenerate_positions()
        n = len(pos_x)
        # Edges over exactly the coincident and near-coincident pairs.
        sources = np.array([0, 2, 4, 6], dtype=np.int32)
        targets = np.array([1, 3, 5, 7], dtype=np.int32)
        k = 30.0

        compiled = _zeros(n)
        _speedups._compute_attractive_forces(
            pos_x.copy(),
            pos_y.copy(),
            compiled[0],
            compiled[1],
            sources,
            targets,
            k,
            len(sources),
        )
        fallback = _zeros(n)
        _yifan_hu().compute_attractive(
            pos_x, pos_y, fallback[0], fallback[1], sources, targets, k, len(sources)
        )

        assert_forces_agree(compiled, fallback, EXACT_RTOL, "degenerate attraction")


# --------------------------------------------------------------------------
# ForceAtlas2 kernels
# --------------------------------------------------------------------------


def _fa2(pos_x, pos_y, *, prevent_overlap=False, sizes=None, degrees=None, scaling=2.0):
    """A ForceAtlas2 layout with its arrays populated, ready for one force pass.

    ``run`` is what allocates the internal arrays, and it always ticks at least
    once (``iterations`` is floored at 1), so the positions are restored
    afterwards. Both copies matter: the naive kernel reads ``_pos_x``/``_pos_y``
    while the Barnes-Hut one builds its tree from the node objects.
    """
    n = len(pos_x)
    nodes = [{"x": float(pos_x[i]), "y": float(pos_y[i])} for i in range(n)]
    layout = fa2_module.ForceAtlas2Layout(
        nodes=nodes,
        links=[],
        size=(400, 400),
        iterations=1,
        prevent_overlap=prevent_overlap,
    )
    layout.run(random_init=False, center_graph=False)
    assert layout._pos_x is not None
    layout._pos_x[:] = pos_x
    layout._pos_y[:] = pos_y
    for i, node in enumerate(layout.nodes):
        node.x = float(pos_x[i])
        node.y = float(pos_y[i])
    layout._scaling = scaling
    if sizes is not None:
        layout._sizes[:] = sizes
    if degrees is not None:
        layout._degrees[:] = degrees
    layout._disp_x.fill(0)
    layout._disp_y.fill(0)
    return layout


class TestForceAtlas2Kernels:
    @pytest.mark.parametrize("n", [8, 64])
    def test_repulsion_matches(self, n):
        pos_x, pos_y = _positions(n)
        degrees = np.arange(n, dtype=np.float64) % 5

        layout = _fa2(pos_x, pos_y, degrees=degrees)
        layout.compute_repulsive_naive()
        fallback = (layout._disp_x.copy(), layout._disp_y.copy())

        compiled = _zeros(n)
        _speedups._compute_fa2_repulsive_forces(
            pos_x.copy(), pos_y.copy(), compiled[0], compiled[1], degrees, 2.0, n
        )

        assert_forces_agree(compiled, fallback, EXACT_RTOL, f"FA2 repulsion (n={n})")

    def test_overlap_repulsion_matches(self):
        n = 40
        pos_x, pos_y = _positions(n, span=60.0)
        degrees = np.arange(n, dtype=np.float64) % 5
        # Radii large enough that many pairs overlap, which is what drives the
        # adjusted-distance floor.
        sizes = np.full(n, 8.0, dtype=np.float64)

        layout = _fa2(pos_x, pos_y, prevent_overlap=True, sizes=sizes, degrees=degrees)
        layout.compute_repulsive_naive()
        fallback = (layout._disp_x.copy(), layout._disp_y.copy())

        compiled = _zeros(n)
        _speedups._compute_fa2_repulsive_forces_overlap(
            pos_x.copy(), pos_y.copy(), compiled[0], compiled[1], degrees, sizes, 2.0, n
        )

        assert_forces_agree(compiled, fallback, EXACT_RTOL, "FA2 overlap repulsion")

    def test_overlap_repulsion_is_bounded_by_the_adjusted_distance_floor(self):
        """Overlapping nodes give a negative border gap, not a huge force."""
        n = 6
        pos_x = np.linspace(0.0, 1.0, n)
        pos_y = np.zeros(n)
        sizes = np.full(n, 20.0, dtype=np.float64)  # every pair deeply overlapping
        degrees = np.zeros(n, dtype=np.float64)

        layout = _fa2(pos_x, pos_y, prevent_overlap=True, sizes=sizes, degrees=degrees)
        layout.compute_repulsive_naive()

        bound = (n - 1) * 2.0 * 1.0 / MIN_ADJUSTED_DIST
        assert float(np.max(np.hypot(layout._disp_x, layout._disp_y))) <= bound
        assert np.all(np.isfinite(layout._disp_x))

    @pytest.mark.parametrize("theta", [0.0, 0.5, 1.2])
    def test_barnes_hut_repulsion_matches(self, theta):
        n = 120
        pos_x, pos_y = _positions(n)
        degrees = np.arange(n, dtype=np.float64) % 5

        layout = _fa2(pos_x, pos_y, degrees=degrees)
        layout.barnes_hut_theta = theta
        layout.compute_repulsive_barnes_hut()
        fallback = (layout._disp_x.copy(), layout._disp_y.copy())

        compiled = _zeros(n)
        _speedups._compute_fa2_repulsive_forces_barnes_hut(
            pos_x.copy(), pos_y.copy(), compiled[0], compiled[1], degrees, 2.0, n, theta
        )

        assert_forces_agree(compiled, fallback, BARNES_HUT_RTOL, f"FA2 Barnes-Hut (theta={theta})")

    def test_repulsion_matches_on_degenerate_separations(self):
        pos_x, pos_y = _degenerate_positions()
        n = len(pos_x)
        degrees = np.arange(n, dtype=np.float64) % 5

        layout = _fa2(pos_x, pos_y, degrees=degrees)
        layout.compute_repulsive_naive()
        fallback = (layout._disp_x.copy(), layout._disp_y.copy())

        compiled = _zeros(n)
        _speedups._compute_fa2_repulsive_forces(
            pos_x.copy(), pos_y.copy(), compiled[0], compiled[1], degrees, 2.0, n
        )

        assert_forces_agree(compiled, fallback, EXACT_RTOL, "FA2 degenerate repulsion")
        assert np.all(np.isfinite(compiled[0])) and np.all(np.isfinite(compiled[1]))


# --------------------------------------------------------------------------
# Layout-level parity over a bounded horizon
# --------------------------------------------------------------------------


def _degenerate_graph():
    """A graph whose nodes sit on top of, and next to, one another."""
    pos_x, pos_y = _degenerate_positions()
    nodes = [
        {"x": float(x), "y": float(y), "width": 10.0, "height": 10.0} for x, y in zip(pos_x, pos_y)
    ]
    links = [{"source": i, "target": i + 1} for i in range(0, len(nodes) - 1, 2)]
    links.append({"source": 0, "target": 9})
    return nodes, links


def _run_both_paths(module, factory):
    """Run ``factory`` with the compiled path and again with the fallback."""
    results = []
    original = module._HAS_CYTHON
    try:
        for use_cython in (True, False):
            module._HAS_CYTHON = use_cython
            layout = factory()
            layout.run(random_init=False, center_graph=False)
            results.append([(n.x, n.y) for n in layout.nodes])
    finally:
        module._HAS_CYTHON = original
    return results


class TestDegenerateLayoutParity:
    """The whole layout, not just one kernel, must survive coincident nodes.

    Each run is capped at a couple of iterations: long enough to exercise every
    guarded branch, short enough that the comparison measures agreement rather
    than the rate at which the dynamics amplify rounding.
    """

    TOLERANCE = 1e-9

    def _assert_agrees(self, results, what):
        worst = max(math.dist(a, b) for a, b in zip(*results))
        assert all(math.isfinite(c) for pos in results for p in pos for c in p), (
            f"{what}: a layout produced a non-finite coordinate"
        )
        assert worst < self.TOLERANCE, (
            f"{what}: compiled and pure-Python paths diverge by {worst:.6g}"
        )

    @pytest.mark.parametrize("use_barnes_hut", [False, True])
    def test_fruchterman_reingold(self, use_barnes_hut):
        nodes, links = _degenerate_graph()
        results = _run_both_paths(
            fr_module,
            lambda: fr_module.FruchtermanReingoldLayout(
                nodes=[dict(n) for n in nodes],
                links=[dict(link) for link in links],
                size=(400, 400),
                iterations=3,
                use_barnes_hut=use_barnes_hut,
            ),
        )
        self._assert_agrees(results, f"FR (barnes_hut={use_barnes_hut})")

    @pytest.mark.parametrize("prevent_overlap", [False, True])
    def test_force_atlas2(self, prevent_overlap):
        nodes, links = _degenerate_graph()
        results = _run_both_paths(
            fa2_module,
            lambda: fa2_module.ForceAtlas2Layout(
                nodes=[dict(n) for n in nodes],
                links=[dict(link) for link in links],
                size=(400, 400),
                iterations=3,
                prevent_overlap=prevent_overlap,
            ),
        )
        self._assert_agrees(results, f"FA2 (prevent_overlap={prevent_overlap})")

    def test_yifan_hu(self):
        nodes, links = _degenerate_graph()
        results = _run_both_paths(
            yh_module,
            lambda: yh_module.YifanHuLayout(
                nodes=[dict(n) for n in nodes],
                links=[dict(link) for link in links],
                size=(400, 400),
                random_seed=1,
                iterations=3,
                level_iterations=1,
            ),
        )
        self._assert_agrees(results, "Yifan Hu")
