# TypeScript to Python Translation Guide

Quick reference for translating WebCola TypeScript to Python.

## Common Patterns

### Classes and Constructors

**TypeScript:**
```typescript
export class Variable {
    offset: number = 0;
    block: Block;

    constructor(public desiredPosition: number, public weight: number = 1, public scale: number = 1) {}
}
```

**Python:**
```python
class Variable:
    def __init__(self, desired_position: float, weight: float = 1.0, scale: float = 1.0):
        self.desired_position = desired_position
        self.weight = weight
        self.scale = scale
        self.offset: float = 0.0
        self.block: Optional[Block] = None
```

### Arrays and Matrices

**TypeScript:**
```typescript
var M = new Array(n);
for (var i = 0; i < n; ++i) {
    M[i] = new Array(n);
    for (var j = 0; j < n; ++j) {
        M[i][j] = f(i, j);
    }
}
```

**Python with NumPy:**
```python
import numpy as np

M = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        M[i, j] = f(i, j)

# Or more Pythonic:
M = np.array([[f(i, j) for j in range(n)] for i in range(n)])
```

### Interfaces to Dataclasses/Protocols

**TypeScript:**
```typescript
export interface Point {
    x: number;
    y: number;
}
```

**Python (Option 1 - Dataclass):**
```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
```

**Python (Option 2 - Protocol for duck typing):**
```python
from typing import Protocol

class Point(Protocol):
    x: float
    y: float
```

### Function Types and Callbacks

**TypeScript:**
```typescript
type Comparator<T> = (a: T, b: T) => number;

class RBTree<T> {
    constructor(private _comparator: Comparator<T>) {}
}
```

**Python:**
```python
from typing import TypeVar, Callable

T = TypeVar('T')
Comparator = Callable[[T, T], float]

class RBTree(Generic[T]):
    def __init__(self, comparator: Comparator[T]):
        self._comparator = comparator
```

### Loops and Iteration

**TypeScript:**
```typescript
// while countdown
var i = n;
while (i--) {
    doSomething(i);
}

// forEach
array.forEach((item, index) => {
    process(item);
});
```

**Python:**
```python
# while countdown
for i in range(n - 1, -1, -1):
    do_something(i)

# or using reversed
for i in reversed(range(n)):
    do_something(i)

# forEach equivalent
for index, item in enumerate(array):
    process(item)
```

### Optional/Nullable Types

**TypeScript:**
```typescript
private _descent: Descent = null;

if (this._descent === null) {
    // initialize
}
```

**Python:**
```python
from typing import Optional

_descent: Optional[Descent] = None

if self._descent is None:
    # initialize
```

### Enums

**TypeScript:**
```typescript
export enum EventType { start, tick, end }
```

**Python:**
```python
from enum import Enum, auto

class EventType(Enum):
    START = auto()
    TICK = auto()
    END = auto()
```

### Default Parameters

**TypeScript:**
```typescript
function foo(a: number, b: number = 10, c: number = 20): number {
    return a + b + c;
}
```

**Python:**
```python
def foo(a: float, b: float = 10.0, c: float = 20.0) -> float:
    return a + b + c
```

### Method Chaining (Fluent Interface)

**TypeScript:**
```typescript
class Layout {
    nodes(v: Node[]): this {
        this._nodes = v;
        return this;
    }
}
```

**Python:**
```python
from typing import TypeVar, Self  # Python 3.11+

class Layout:
    def nodes(self, v: list[Node]) -> Self:
        self._nodes = v
        return self

# For Python 3.8-3.10:
from typing import TypeVar

T = TypeVar('T', bound='Layout')

class Layout:
    def nodes(self: T, v: list[Node]) -> T:
        self._nodes = v
        return self
```

### Static Methods

**TypeScript:**
```typescript
class Descent {
    private static dotProd(a: number[], b: number[]): number {
        var x = 0, i = a.length;
        while (i--) x += a[i] * b[i];
        return x;
    }
}
```

**Python with NumPy:**
```python
import numpy as np

class Descent:
    @staticmethod
    def dot_prod(a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b)
```

### Number Constants

**TypeScript:**
```typescript
Number.MAX_VALUE
Number.POSITIVE_INFINITY
```

**Python:**
```python
import math

float('inf')  # or math.inf
math.inf
```

### Array Operations

**TypeScript:**
```typescript
// Map
var doubled = array.map(x => x * 2);

// Filter
var filtered = array.filter(x => x > 10);

// Reduce
var sum = array.reduce((acc, x) => acc + x, 0);

// Slice
var copy = array.slice(0);
```

**Python:**
```python
# Map
doubled = [x * 2 for x in array]
# or
doubled = list(map(lambda x: x * 2, array))

# Filter
filtered = [x for x in array if x > 10]
# or
filtered = list(filter(lambda x: x > 10, array))

# Reduce
from functools import reduce
sum_val = reduce(lambda acc, x: acc + x, array, 0)
# or simpler
sum_val = sum(array)

# Slice (copy)
copy = array[:]
# or
copy = array.copy()
```

### Object/Dictionary Iteration

**TypeScript:**
```typescript
for (var key in object) {
    if (object.hasOwnProperty(key)) {
        doSomething(key, object[key]);
    }
}
```

**Python:**
```python
for key, value in dictionary.items():
    do_something(key, value)

# Or just keys
for key in dictionary:
    do_something(key, dictionary[key])
```

## NumPy Specifics

### Matrix Multiplication

**TypeScript:**
```typescript
// Manual matrix-vector multiply
function rightMultiply(m: number[][], v: number[], r: number[]) {
    var i = m.length;
    while (i--) r[i] = dotProd(m[i], v);
}
```

**Python:**
```python
import numpy as np

def right_multiply(m: np.ndarray, v: np.ndarray) -> np.ndarray:
    return m @ v  # or np.dot(m, v)
```

### Creating Arrays

**TypeScript:**
```typescript
this.H = new Array(this.k);
for (var i = 0; i < this.k; ++i) {
    this.H[i] = new Array(n);
    for (var j = 0; j < n; ++j) {
        this.H[i][j] = new Array(n);
    }
}
```

**Python:**
```python
import numpy as np

self.H = np.zeros((self.k, n, n))
```

## Sorted Containers (RBTree Replacement)

**TypeScript RBTree:**
```typescript
import {RBTree} from './rbtree'

var tree = new RBTree<Event>((a, b) => a.pos - b.pos);
tree.insert(event);
var iter = tree.lowerBound(query);
```

**Python SortedDict:**
```python
from sortedcontainers import SortedDict

tree = SortedDict()
tree[event.pos] = event
idx = tree.bisect_left(query.pos)
```

## Testing Patterns

**TypeScript (Jest/TSDX):**
```typescript
describe('VPSC Solver', () => {
    it('should solve simple constraint', () => {
        const v1 = new Variable(0);
        const v2 = new Variable(1);
        const c = new Constraint(v1, v2, 10);
        const solver = new Solver([v1, v2], [c]);
        solver.solve();
        expect(v2.position() - v1.position()).toBeCloseTo(10);
    });
});
```

**Python (pytest):**
```python
import pytest
from graph_layout.cola.vpsc import Variable, Constraint, Solver

class TestVPSCSolver:
    def test_simple_constraint(self):
        v1 = Variable(0.0)
        v2 = Variable(1.0)
        c = Constraint(v1, v2, 10.0)
        solver = Solver([v1, v2], [c])
        solver.solve()
        assert abs(v2.position() - v1.position() - 10.0) < 1e-6
```

## Naming Conventions

- **Classes**: PascalCase (same in both)
- **Functions/Methods**: camelCase (TS) → snake_case (Python)
- **Constants**: UPPER_CASE (both)
- **Private members**: `_prefixed` (both, but Python convention is stronger)

Examples:
- `getSourceIndex` → `get_source_index`
- `computeDerivatives` → `compute_derivatives`
- `_handleDisconnected` → `_handle_disconnected`
