"""
Setup script for building Cython extensions.

This script builds the _speedups module containing high-performance
Cython implementations of:
- Priority queue (pairing heap)
- Shortest paths (Dijkstra's algorithm)

These extensions are part of the graph_layout package root.

If Cython is not available or compilation fails, the package will still
install and use pure Python or scipy fallbacks.
"""

import sys
import os
import warnings
from setuptools import setup, Extension

# Try to import Cython
try:
    from Cython.Build import cythonize
    HAVE_CYTHON = True
except ImportError:
    HAVE_CYTHON = False
    warnings.warn(
        "Cython not found. Installing without optimized extensions. "
        "For better performance, install Cython and rebuild: "
        "pip install Cython && pip install --no-build-isolation -e .",
        UserWarning
    )

# Try to import numpy for include directories
try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False
    # This should not happen since numpy is a build requirement
    warnings.warn("NumPy not found during build", UserWarning)


def build_extensions():
    """Build Cython extension modules."""
    if not HAVE_CYTHON:
        return []

    # Compiler flags
    extra_compile_args = []
    extra_link_args = []

    if sys.platform == "darwin":
        # macOS specific flags
        extra_compile_args.extend(["-O3", "-std=c99"])
    elif sys.platform == "linux":
        # Linux specific flags
        extra_compile_args.extend(["-O3", "-std=c99"])
    elif sys.platform == "win32":
        # Windows specific flags
        extra_compile_args.extend(["/O2"])

    include_dirs = [np.get_include()] if HAVE_NUMPY else []

    extensions = [
        Extension(
            "graph_layout._speedups",
            sources=["src/graph_layout/_speedups.pyx"],
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
    ]

    # Cythonize with compiler directives
    return cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "embedsignature": True,
        },
        # Build in parallel if possible
        nthreads=os.cpu_count() or 1,
    )


# Only build extensions if Cython is available
ext_modules = build_extensions()

# Let setuptools handle the rest via pyproject.toml
setup(
    ext_modules=ext_modules,
)
