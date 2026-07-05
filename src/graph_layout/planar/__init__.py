"""Planar straight-line drawing algorithms.

Foundational graph-drawing methods that render a planar graph with straight-line
edges and no crossings on a compact grid or convex frame:

- :class:`SchnyderLayout` -- realizer-based drawing on an ``O(n) x O(n)`` grid.
- :class:`FPPLayout` -- de Fraysseix-Pach-Pollack shift method on the
  ``(2n-4) x (n-2)`` grid.
- :class:`TutteLayout` -- barycentric (spring) embedding with convex faces for
  3-connected planar graphs.
"""

from .fpp import FPPLayout, fpp_coordinates
from .schnyder import SchnyderLayout, schnyder_coordinates
from .tutte import TutteLayout, tutte_coordinates

__all__ = [
    "SchnyderLayout",
    "schnyder_coordinates",
    "FPPLayout",
    "fpp_coordinates",
    "TutteLayout",
    "tutte_coordinates",
]
