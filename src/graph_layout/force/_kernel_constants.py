"""Numerical floors shared by the compiled kernels and their Python fallbacks.

``graph_layout._speedups`` and the pure-Python ``compute_*`` methods in this
package are two implementations of the same force models, and
``tests/test_cython_parity.py`` holds them to producing the same forces from
the same input. Degenerate separations are where such a pair drifts apart most
easily: a guard written as "clamp the distance" and one written as "skip the
pair" agree everywhere except when two nodes nearly coincide, which an ordinary
graph never exercises and a randomly initialised one occasionally does.

These values are the reference. ``_speedups.pyx`` repeats them as literals
because a Cython inner loop cannot cheaply read a Python module constant;
changing one side without the other is what the parity tests catch.
"""

# Floor on the squared separation between two nodes. Below it the repulsive
# force is capped rather than left to grow like 1/d: exactly coincident nodes
# would otherwise divide by zero, and near-coincident ones would take a single
# step large enough to throw a node clear out of the drawing.
MIN_DIST_SQ = 1e-10

# Floor on the overlap-adjusted separation (centre distance minus the two node
# radii) used by ForceAtlas2's overlap-prevention repulsion. Overlapping nodes
# give a negative adjusted distance, so this fixes the sign as well as capping
# the force.
MIN_ADJUSTED_DIST = 0.01
