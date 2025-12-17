# graph-layout Build System Documentation

This document describes how graph-layout's Cython extensions are built, distributed, and loaded.

## Overview

graph-layout uses a **hybrid build system** with performance-optimized Cython extensions and automatic fallbacks:

```
Priority Cascade:
1. Cython-compiled C extensions (fastest, default)
2. scipy (fast, optional via pip install graph-layout[fast])
3. Pure Python (slowest, always available)
```

**Key Components:**
- **Cython**: Transpiles `.pyx` files to C code
- **setuptools**: Orchestrates the build process
- **cibuildwheel**: Builds multi-platform wheels for distribution
- **Dynamic loader**: Selects best available implementation at import time

## Build Architecture

### File Structure

```
src/graph_layout/
├── __init__.py             # Main package exports
└── cola/                   # Cola algorithm subpackage
    ├── _pqueue_cy.pyx          # Cython priority queue source
    ├── _pqueue_cy.pxd          # Cython header (for cross-module imports)
    ├── _shortestpaths_cy.pyx   # Cython shortest paths source
    ├── _shortestpaths_py.py    # Pure Python fallback
    └── shortestpaths.py        # Wrapper with priority cascade

    # Generated during build:
    ├── _pqueue_cy.c            # Cython-generated C code (~627KB)
    ├── _shortestpaths_cy.c     # Cython-generated C code (~692KB)
    ├── _pqueue_cy.cpython-313-darwin.so           # Compiled extension (macOS)
    └── _shortestpaths_cy.cpython-313-darwin.so    # Compiled extension (macOS)
```

### Build Flow

```
.pyx files → Cython → .c files → C compiler → .so/.pyd files
```

## Local Development Builds

### Setup Requirements

```bash
# Install development dependencies
uv sync

# Required build tools (installed automatically):
- setuptools>=64
- Cython>=3.0.0
- numpy>=1.20.0
- C compiler (gcc, clang, or MSVC)
```

### Build Process

When you run `uv sync`:

1. **Read Build Configuration** (`pyproject.toml`):
   ```toml
   [build-system]
   requires = [
       "setuptools>=64",
       "wheel",
       "Cython>=3.0.0; platform_python_implementation=='CPython'",
       "numpy>=1.20.0",
   ]
   build-backend = "setuptools.build_meta"
   ```

2. **Install Build Dependencies**:
   - setuptools
   - Cython
   - numpy (for `np.get_include()`)

3. **Execute Build Backend**:
   ```
   setuptools.build_meta.build_editable()
   ↓
   Calls setup.py
   ```

4. **Cythonize Extensions** (`setup.py`):
   ```python
   extensions = [
       Extension(
           "graph_layout.cola._pqueue_cy",
           sources=["src/graph_layout/cola/_pqueue_cy.pyx"],
           extra_compile_args=["-O3", "-std=c99"],
       ),
       Extension(
           "graph_layout.cola._shortestpaths_cy",
           sources=["src/graph_layout/cola/_shortestpaths_cy.pyx"],
           extra_compile_args=["-O3", "-std=c99"],
       ),
   ]

   cythonize(
       extensions,
       compiler_directives={
           "language_level": "3",
           "boundscheck": False,    # Disable bounds checking
           "wraparound": False,      # Disable negative indexing
           "cdivision": True,        # Use C division
       },
   )
   ```

5. **Cython Transpilation**:
   ```
   _pqueue_cy.pyx (4KB) → _pqueue_cy.c (627KB)
   _shortestpaths_cy.pyx (6KB) → _shortestpaths_cy.c (692KB)
   ```

6. **C Compilation**:

   **macOS**:
   ```bash
   clang -O3 -std=c99 -I<numpy_include> _pqueue_cy.c \
       -o _pqueue_cy.cpython-313-darwin.so
   ```

   **Linux**:
   ```bash
   gcc -O3 -std=c99 -I<numpy_include> _pqueue_cy.c \
       -o _pqueue_cy.cpython-313-linux-x86_64.so
   ```

   **Windows**:
   ```bash
   cl.exe /O2 /I<numpy_include> _pqueue_cy.c \
       /link /OUT:_pqueue_cy.cp313-win_amd64.pyd
   ```

7. **Install in Editable Mode**:
   - Extensions built in `src/graph_layout/cola/` directory
   - Python imports directly from source tree
   - Changes to `.py` files take effect immediately
   - Changes to `.pyx` files require rebuild: `uv sync`

### Compiler Optimization Flags

| Platform | Flags | Effect |
|----------|-------|--------|
| macOS/Linux | `-O3` | Maximum optimization |
| macOS/Linux | `-std=c99` | C99 standard |
| Windows | `/O2` | Maximum optimization |
| All | `boundscheck=False` | Remove array bounds checks |
| All | `wraparound=False` | Remove negative index support |
| All | `cdivision=True` | Use C division (faster) |

**Performance Impact**: These flags provide **10-30x speedup** over pure Python.

### Verification

Check if extensions are built:

```bash
# List compiled extensions
ls -lh src/graph_layout/cola/*.so src/graph_layout/cola/*.pyd 2>/dev/null

# Verify they're loaded
uv run python -c "
from graph_layout.cola import shortestpaths
print(f'Implementation: {shortestpaths.get_implementation()}')
"
# Expected output: "cython"

# Check extension details
file src/graph_layout/cola/_pqueue_cy.cpython-313-darwin.so
# macOS output: Mach-O 64-bit bundle arm64
```

## Production Wheel Builds

### cibuildwheel Configuration

For **distribution to end users**, we use GitHub Actions with cibuildwheel to build **pre-compiled wheels** for all platforms.

**Workflow**: `.github/workflows/wheels.yml`

```yaml
jobs:
  build_wheels:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-13, macos-14]

    steps:
      - uses: pypa/cibuildwheel@v2.21
        env:
          CIBW_BUILD: cp39-* cp310-* cp311-* cp312-* cp313-*
          CIBW_SKIP: "*-win32 *-manylinux_i686 *-musllinux_*"
          CIBW_BEFORE_BUILD: pip install numpy Cython
          CIBW_TEST_COMMAND: pytest {package}/tests -v
          CIBW_ARCHS_MACOS: "x86_64 arm64"
          CIBW_ARCHS_LINUX: "x86_64 aarch64"
```

### Build Matrix

cibuildwheel builds **~40 wheel files**:

| Platform | Architectures | Python Versions | Wheels |
|----------|--------------|-----------------|--------|
| Linux (manylinux) | x86_64, aarch64 | 3.9-3.13 | 10 |
| macOS 11+ | x86_64, arm64 | 3.9-3.13 | 10 |
| Windows | amd64 | 3.9-3.13 | 5 |

**Example wheel names**:
```
graph_layout-0.1.2-cp313-cp313-macosx_11_0_arm64.whl
graph_layout-0.1.2-cp313-cp313-manylinux_2_17_x86_64.whl
graph_layout-0.1.2-cp313-cp313-win_amd64.whl
```

### Wheel Contents

Each wheel contains:
- Pre-compiled `.so` (Linux/macOS) or `.pyd` (Windows) files
- Pure Python `.py` files
- Metadata in `*.dist-info/`

**Advantage**: Users don't need a C compiler! Just `pip install graph-layout`.

### Release Process

1. **Tag a release**:
   ```bash
   git tag v0.1.2
   git push origin v0.1.2
   ```

2. **GitHub Actions automatically**:
   - Builds wheels for all platforms
   - Runs tests on each wheel
   - Uploads artifacts

3. **On GitHub Release**:
   - Publishes to PyPI
   - Users can `pip install graph-layout`

## Priority Cascade Implementation

### Runtime Selection Logic

**File**: `src/graph_layout/cola/shortestpaths.py`

```python
# Try Cython implementation first
try:
    from . import _shortestpaths_cy
    _Calculator = _shortestpaths_cy.Calculator
    _IMPLEMENTATION = "cython"
except ImportError:
    # Try scipy as fallback
    try:
        import numpy as np
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import shortest_path
        _IMPLEMENTATION = "scipy"
    except ImportError:
        # Fall back to pure Python
        from ._shortestpaths_py import Calculator as _PyCalculator
        _Calculator = _PyCalculator
        _IMPLEMENTATION = "python"
```

### When Each Implementation Is Used

| Scenario | Implementation | Performance |
|----------|---------------|-------------|
| **Standard install from PyPI** | Cython (wheel) | ⚡⚡⚡ Fastest |
| **Install with `[fast]` extra** | scipy | ⚡⚡⚡ Fastest (similar) |
| **Source install, no compiler** | Pure Python | ⚡ Slowest (100x slower) |
| **PyPy interpreter** | Pure Python | ⚡ Slowest |
| **Unsupported platform** | Pure Python | ⚡ Slowest |

### Advanced Features

The wrapper maintains **full API compatibility** across implementations:

```python
class Calculator:
    def __init__(self, n, edges, get_source, get_target, get_length):
        if _IMPLEMENTATION in ("cython", "python"):
            # Direct delegation
            self._calc = _Calculator(...)
        else:
            # scipy requires building sparse matrix
            self._build_scipy_graph()

        # Always keep pure Python for advanced features
        if _IMPLEMENTATION != "python":
            from ._shortestpaths_py import Calculator as _PyCalculator
            self._py_calc = _PyCalculator(...)

    def path_from_node_to_node_with_prev_cost(self, ...):
        # Advanced feature - always use pure Python
        return self._py_calc.path_from_node_to_node_with_prev_cost(...)
```

**Rationale**: Some features (like custom cost functions) aren't Cython-optimized, so we maintain a pure Python calculator for those cases.

## Troubleshooting

### Build Failures

**Problem**: `ImportError: cannot import name '_shortestpaths_cy'`

**Diagnosis**:
```bash
# Check if extensions exist
ls src/graph_layout/cola/*.so

# Check current implementation
uv run python -c "from graph_layout.cola.shortestpaths import get_implementation; print(get_implementation())"
```

**Solutions**:

1. **Missing C compiler**:
   ```bash
   # macOS
   xcode-select --install

   # Ubuntu/Debian
   sudo apt-get install build-essential

   # Windows
   # Install Visual Studio Build Tools
   ```

2. **Cython not installed**:
   ```bash
   pip install Cython>=3.0.0
   uv sync
   ```

3. **Force rebuild**:
   ```bash
   rm -rf build/ src/graph_layout/cola/*.so src/graph_layout/cola/*.c
   uv sync
   ```

4. **Use scipy instead**:
   ```bash
   pip install graph-layout[fast]
   ```

5. **Disable Cython** (development only):
   ```python
   # Temporarily rename .pyx files
   mv src/graph_layout/cola/_pqueue_cy.pyx src/graph_layout/cola/_pqueue_cy.pyx.bak
   ```

### Performance Issues

**Check which implementation is active**:

```python
from graph_layout.cola import shortestpaths

impl = shortestpaths.get_implementation()
print(f"Using: {impl}")

if impl == "python":
    print("WARNING: Running in pure Python mode (slow)")
    print("Install scipy: pip install graph-layout[fast]")
elif impl == "cython":
    print("✓ Using Cython (optimal)")
elif impl == "scipy":
    print("✓ Using scipy (optimal)")
```

**Benchmark**:

```python
import time
from graph_layout.cola.layout import Layout

nodes = [{'x': 0, 'y': 0} for _ in range(100)]
edges = [{'source': i, 'target': (i+1)%100} for i in range(100)]

layout = Layout().nodes(nodes).links(edges)

start = time.time()
layout.start(50, 0, 0, 0, False)
elapsed = time.time() - start

print(f"Time: {elapsed:.3f}s")
print(f"Expected (Cython): ~0.05s")
print(f"Expected (Python): ~4s")
```

### Platform-Specific Issues

**macOS Apple Silicon (M1/M2/M3)**:
```bash
# Ensure you're using native Python, not Rosetta
python -c "import platform; print(platform.machine())"
# Should output: arm64

# If it says "x86_64", reinstall Python for arm64
```

**Windows MSVC**:
```bash
# Install Visual Studio Build Tools 2019+
# Or use conda which includes a compiler:
conda install -c conda-forge cython numpy
```

**Linux without glibc** (Alpine, musl):
```bash
# Wheels not available for musllinux
# Build from source with:
apk add gcc musl-dev python3-dev
pip install --no-binary :all: graph-layout
```

## Development Workflow

### Making Changes to Cython Code

1. **Edit `.pyx` file**:
   ```bash
   vim src/graph_layout/cola/_pqueue_cy.pyx
   ```

2. **Rebuild extension**:
   ```bash
   uv sync
   # Or force rebuild:
   rm src/graph_layout/cola/_pqueue_cy.c src/graph_layout/cola/_pqueue_cy.*.so
   uv sync
   ```

3. **Test changes**:
   ```bash
   make test
   ```

4. **Verify performance**:
   ```bash
   uv run python scripts/profile_layout.py
   ```

### Making Changes to Pure Python

Changes to `.py` files (including `_shortestpaths_py.py`) take effect immediately in editable mode—no rebuild needed.

### Adding New Cython Modules

1. **Create `.pyx` file**:
   ```bash
   vim src/graph_layout/cola/_newmodule.pyx
   ```

2. **Add to `setup.py`**:
   ```python
   extensions = [
       Extension(
           "graph_layout.cola._newmodule",
           sources=["src/graph_layout/cola/_newmodule.pyx"],
           extra_compile_args=extra_compile_args,
       ),
       # ... existing extensions
   ]
   ```

3. **If cross-importing, create `.pxd`**:
   ```cython
   # _newmodule.pxd
   cdef class MyClass:
       cdef public int value
       cpdef int method(self)
   ```

4. **Build**:
   ```bash
   uv sync
   ```

## Performance Analysis

### Compilation Results

**Source Code Size**:
- `_pqueue_cy.pyx`: 4KB (160 lines)
- `_shortestpaths_cy.pyx`: 6KB (220 lines)

**Generated C Code**:
- `_pqueue_cy.c`: 627KB (Cython boilerplate + optimized code)
- `_shortestpaths_cy.c`: 692KB

**Compiled Binary**:
- `_pqueue_cy.*.so`: 163KB (native machine code)
- `_shortestpaths_cy.*.so`: 167KB

**Speedup**: ~10-30x faster than pure Python for Dijkstra's algorithm.

### What Gets Optimized

Cython compiles these Python operations to C:

```python
# Before (Pure Python)
for i in range(n):
    for j in range(n):
        distance = math.sqrt((x[i] - x[j])**2)
        if distance < min_dist:
            min_dist = distance

# After (Cython)
# → Compiled to C for loop with native types
# → No Python object overhead
# → Direct memory access
# → SIMD vectorization possible
```

**Key optimizations**:
1. **Type specialization**: `int`, `double` instead of Python objects
2. **Bounds check removal**: `boundscheck=False`
3. **Direct memory access**: No Python list/dict overhead
4. **C function calls**: `sqrt()` directly instead of `math.sqrt()`
5. **Inlining**: Small functions inlined by C compiler

## References

- [Cython Documentation](https://cython.readthedocs.io/)
- [setuptools User Guide](https://setuptools.pypa.io/)
- [cibuildwheel Documentation](https://cibuildwheel.readthedocs.io/)
- [PEP 517: Build System Interface](https://peps.python.org/pep-0517/)
- [PEP 518: Build System Dependencies](https://peps.python.org/pep-0518/)

## Summary

PyCola's build system provides:

- [x] **High performance**: 10-30x speedup via Cython
- [x] **User convenience**: Pre-built wheels for all platforms
- [x] **Graceful degradation**: Falls back to pure Python
- [x] **Developer friendly**: Editable installs with hot reload
- [x] **CI/CD automation**: GitHub Actions builds all platforms
- [x] **Zero runtime dependencies**: Cython extensions are self-contained

The priority cascade ensures users get the best performance available on their system without any configuration.
