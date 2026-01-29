# Kandinsky Orthogonal Layout Implementation Plan

## Overview

The Kandinsky model is an orthogonal graph drawing algorithm that handles graphs with vertices of arbitrary degree (unlike GIOTTO which is limited to degree ≤ 4). It uses the **Topology-Shape-Metrics (TSM)** approach with three phases.

## Algorithm Background

### Why Kandinsky?

- Works on **any graph** (not just planar, not limited to degree 4)
- Practical for real-world use cases: UML diagrams, ER diagrams, flowcharts
- Used in commercial tools (yFiles, yEd)
- Well-documented with known approximation algorithms

### Key Papers

- Fößmeier & Kaufmann (1995): Original Kandinsky model
- Tamassia (1987): Min-cost flow for orthogonal drawings
- Eiglsperger (2003): 2-approximation algorithm for Kandinsky bend minimization

## The Three Phases (TSM Approach)

### Phase 1: Planarization

**Goal**: Convert input graph to a planar embedding

**Steps**:
1. If graph is already planar, compute planar embedding
2. If non-planar, insert dummy "crossing vertices" where edges cross
3. Use planarity testing (Boyer-Myrvold or similar)

**Output**: Planar embedding with faces identified

### Phase 2: Orthogonalization

**Goal**: Assign orthogonal shape (angles and bends) to edges

**The Orthogonal Representation**:
- For each edge at each vertex: assign angle (90°, 180°, 270°, or 0° for Kandinsky)
- For each edge: assign number of bends and their directions
- Constraint: angles around each vertex sum to 360°
- Constraint: angles around each face sum to (n-2)×180° for inner faces

**Kandinsky-Specific**:
- Allows 0° angles between consecutive edges (needed for degree > 4)
- Multiple edges can exit from same side of a vertex
- Uses "port" concept: edges grouped by exit side

**Min-Cost Flow Formulation** (Tamassia):
```
- Vertices supply flow: 4 - degree(v)
- Faces consume flow: #vertices_on_face - 4 (inner), +4 (outer)
- Arc (v, F): capacity 2, cost 0 (corner angles)
- Arc (F, F'): capacity 4, cost 1 (bends on edge between faces)
```

**For Kandinsky**: Use 2-approximation algorithm since exact is NP-complete

### Phase 3: Compaction

**Goal**: Assign x,y coordinates minimizing area/edge length

**Approach**:
1. Build constraint graph for horizontal segments
2. Build constraint graph for vertical segments
3. Compute coordinates via longest path or ILP

**Output**: Final (x, y) coordinates for all vertices

## Implementation Plan

### File Structure

```
src/graph_layout/
├── orthogonal/
│   ├── __init__.py
│   ├── kandinsky.py           # Main KandinskyLayout class
│   ├── planarization.py       # Phase 1: Planarization
│   ├── orthogonalization.py   # Phase 2: Orthogonal representation
│   ├── compaction.py          # Phase 3: Coordinate assignment
│   ├── planarity.py           # Planarity testing utilities
│   └── types.py               # OrthogonalRep, Face, Port, etc.
```

### Data Structures

```python
@dataclass
class Port:
    """A port where edges connect to a vertex side."""
    vertex: int
    side: Literal["north", "south", "east", "west"]
    edges: list[int]  # Edge indices using this port

@dataclass
class OrthogonalRep:
    """Orthogonal representation of a graph."""
    # For each vertex: list of (edge, angle) in clockwise order
    vertex_angles: dict[int, list[tuple[int, int]]]  # angle in {0, 90, 180, 270}
    # For each edge: list of bend angles
    edge_bends: dict[int, list[int]]  # +1 = left turn, -1 = right turn

@dataclass
class Face:
    """A face in the planar embedding."""
    vertices: list[int]  # Vertices in order around face
    edges: list[int]     # Edges in order around face
    is_outer: bool

@dataclass
class PlanarEmbedding:
    """Planar embedding with faces."""
    adj_order: dict[int, list[int]]  # Clockwise edge order at each vertex
    faces: list[Face]
    outer_face: int  # Index of outer face
```

### Class: KandinskyLayout

```python
class KandinskyLayout(StaticLayout):
    """
    Kandinsky orthogonal layout algorithm.

    Produces orthogonal drawings where edges use only horizontal
    and vertical segments. Suitable for UML, ER diagrams, flowcharts.

    Parameters:
        node_width: Width of node boxes (default: 40)
        node_height: Height of node boxes (default: 30)
        node_separation: Minimum gap between nodes (default: 40)
        edge_separation: Minimum gap between parallel edges (default: 10)
        bend_cost: Weight for bend minimization (default: 1.0)
        compaction: Compaction strategy ("longest_path" or "ilp")
    """
```

### Implementation Phases

#### Phase 1: MVP (Simplified)

1. **Assume planar input** (skip planarization)
2. **Simple orthogonalization**:
   - Assign ports based on relative positions
   - Use greedy bend assignment
3. **Grid compaction**:
   - Place nodes on integer grid
   - Route edges with simple pathfinding

**Deliverable**: Working layout for simple planar graphs

#### Phase 2: Full Planarization

1. Implement planarity testing (or use existing library)
2. Implement crossing vertex insertion
3. Handle non-planar graphs

#### Phase 3: Optimal Orthogonalization

1. Implement min-cost flow formulation
2. Implement Kandinsky 2-approximation
3. Better bend minimization

#### Phase 4: Advanced Compaction

1. Constraint-based compaction
2. ILP-based optimal compaction (optional)
3. Edge routing improvements

### Algorithm Pseudocode

```
KANDINSKY_LAYOUT(G, width, height):
    # Phase 1: Planarization
    if not is_planar(G):
        G_planar, crossings = planarize(G)
    else:
        G_planar = G
        crossings = []

    embedding = compute_planar_embedding(G_planar)
    faces = compute_faces(embedding)

    # Phase 2: Orthogonalization
    ortho_rep = compute_orthogonal_rep(G_planar, embedding, faces)

    # Phase 3: Compaction
    coordinates = compact(ortho_rep, width, height)

    # Map back crossing vertices to edge bends
    coordinates = restore_crossings(coordinates, crossings)

    return coordinates
```

### Edge Routing

For edges between vertices:
1. Determine exit port (N/S/E/W) based on orthogonal rep
2. Route using only horizontal/vertical segments
3. Insert bends as needed
4. Avoid crossing other edges and nodes

### Test Cases

1. **Simple graphs**: Path, cycle, tree
2. **Planar graphs**: K4, grid, outerplanar
3. **High-degree vertices**: Star graph, complete bipartite
4. **Real-world examples**: Simple UML class diagram, ER diagram

### Dependencies

- May need: NetworkX for planarity testing (optional, can implement)
- Existing: NumPy for coordinate calculations

### Complexity

- Planarization: O(n²) worst case for crossing minimization
- Orthogonalization: O(n²) for flow network
- Compaction: O(n) for longest path, higher for ILP

### Open Questions

1. Should we use NetworkX for planarity testing or implement our own?
2. How to handle disconnected graphs?
3. Should edge labels be considered?
4. Port constraints from user (e.g., "edge must exit from north")?

## References

- [Orthogonal Graph Drawing with Constraints](https://publikationen.uni-tuebingen.de/xmlui/bitstream/handle/10900/49366/pdf/diss.pdf) - Comprehensive thesis
- [Implementing an Algorithm for Orthogonal Graph Layout](https://rtsys.informatik.uni-kiel.de/~biblio/downloads/theses/ocl-bt.pdf) - Implementation guide
- [yFiles Orthogonal Layout](https://www.yfiles.com/the-yfiles-sdk/features/automatic-layouts/orthogonal-layout) - Commercial reference
- [Tom Sawyer Orthogonal Drawing Models](https://blog.tomsawyer.com/orthogonal-drawing-models) - Model comparison
- [OGDF Library](https://ogdf.uos.de/) - Open source C++ implementation

## Milestones

1. **Week 1**: Data structures, simple grid placement (no bends)
2. **Week 2**: Basic orthogonalization with greedy bends
3. **Week 3**: Proper min-cost flow orthogonalization
4. **Week 4**: Planarization for non-planar graphs
5. **Week 5**: Compaction improvements, testing, documentation
