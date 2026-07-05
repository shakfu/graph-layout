"""Planar straight-line drawing algorithms.

Foundational graph-drawing methods that render a planar graph with straight-line
edges and no crossings on a compact grid or convex frame:

- :class:`SchnyderLayout` -- realizer-based drawing on the ``(n-1) x (n-1)`` grid.
- :class:`FPPLayout` -- de Fraysseix-Pach-Pollack shift method on the
  ``(2n-4) x (n-2)`` grid.
- :class:`TutteLayout` -- barycentric (spring) embedding with convex faces for
  3-connected planar graphs.
- :class:`PlanarizationLayout` -- straight-line drawing of *non-planar* graphs by
  replacing crossings with dummy vertices.
- :class:`MixedModelLayout` -- visibility-representation drawing with box-vertices
  and bendless port-attached edges (high angular resolution).
"""

from .fpp import FPPLayout, fpp_coordinates
from .mixed_model import MixedModelLayout, visibility_representation
from .planarization import PlanarizationLayout
from .schnyder import SchnyderLayout, schnyder_coordinates
from .tutte import TutteLayout, tutte_coordinates

__all__ = [
    "SchnyderLayout",
    "schnyder_coordinates",
    "FPPLayout",
    "fpp_coordinates",
    "TutteLayout",
    "tutte_coordinates",
    "PlanarizationLayout",
    "MixedModelLayout",
    "visibility_representation",
]
