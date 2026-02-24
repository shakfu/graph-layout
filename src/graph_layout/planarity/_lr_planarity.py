"""Core LR-planarity algorithm (de Fraysseix & Rosenstiehl).

Faithful implementation following:
    Brandes, U. (2009). The Left-Right Planarity Test (technical report).

Intervals store back-edge references. The constraint stack tracks which
back edges must go to the left vs right side of the DFS tree path.
"""

from __future__ import annotations

from typing import Optional

from ._types import ConflictPair, Interval, PlanarityResult

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def lr_planarity_test(
    num_nodes: int,
    adj: list[list[int]],
) -> PlanarityResult:
    """Run LR-planarity on a biconnected graph given as adjacency list."""
    if num_nodes <= 2:
        emb = {v: list(adj[v]) for v in range(num_nodes)}
        return PlanarityResult(is_planar=True, embedding=emb)

    m = sum(len(adj[v]) for v in range(num_nodes)) // 2
    if m > 3 * num_nodes - 6:
        witness = _find_kuratowski_subgraph(num_nodes, adj)
        if witness is not None:
            k_type, k_edges = witness
            return PlanarityResult(
                is_planar=False,
                kuratowski_edges=k_edges,
                kuratowski_type=k_type,
            )
        return PlanarityResult(is_planar=False)

    state = _LRState(num_nodes, adj)
    state.phase1_orientation()

    if not state.phase2_testing():
        witness = _find_kuratowski_subgraph(num_nodes, adj)
        if witness is not None:
            k_type, k_edges = witness
            return PlanarityResult(
                is_planar=False,
                kuratowski_edges=k_edges,
                kuratowski_type=k_type,
            )
        return PlanarityResult(is_planar=False)

    embedding = state.phase3_embedding()
    return PlanarityResult(is_planar=True, embedding=embedding)


# ---------------------------------------------------------------------------
# Algorithm state
# ---------------------------------------------------------------------------

# Stack frame types for iterative DFS in phase 2
_ENTER = 0  # process edges of v starting at index idx
_INTEGRATE = 1  # run integration for edge at idx in ordered_adj[v]


class _LRState:
    def __init__(self, n: int, adj: list[list[int]]) -> None:
        self.n = n
        self.adj = adj

        # Phase 1 outputs (per vertex)
        self.height: list[int] = [-1] * n
        self.parent: list[int] = [-1] * n
        self.lowpt_v: list[int] = [0] * n
        self.lowpt2_v: list[int] = [0] * n

        self.tree_children: list[list[int]] = [[] for _ in range(n)]
        self.dfs_order: list[int] = []

        self.nesting_depth: dict[tuple[int, int], int] = {}
        self.ordered_adj: list[list[int]] = [[] for _ in range(n)]

        # Phase 2 data
        self.S: list[ConflictPair] = []
        self.stack_bottom: dict[tuple[int, int], Optional[ConflictPair]] = {}
        self.lowpt_edge: dict[tuple[int, int], tuple[int, int]] = {}
        self.side: dict[tuple[int, int], int] = {}
        self.ref: dict[tuple[int, int], Optional[tuple[int, int]]] = {}

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _lowpt_of_edge(self, e: tuple[int, int]) -> int:
        v, w = e
        if self.parent[w] == v:
            return self.lowpt_v[w]
        else:
            return self.height[w]

    def _conflicting(self, iv: Interval, b: tuple[int, int]) -> bool:
        if iv.empty():
            return False
        assert iv.high is not None
        return self._lowpt_of_edge(iv.high) > self._lowpt_of_edge(b)

    # -------------------------------------------------------------------
    # Phase 1: DFS orientation
    # -------------------------------------------------------------------

    def phase1_orientation(self) -> None:
        self.height[0] = 0
        self.lowpt_v[0] = 0
        self.lowpt2_v[0] = 0
        self.dfs_order.append(0)

        stack: list[tuple[int, int]] = [(0, 0)]

        while stack:
            v, idx = stack[-1]
            if idx < len(self.adj[v]):
                stack[-1] = (v, idx + 1)
                w = self.adj[v][idx]
                if self.height[w] == -1:
                    self.parent[w] = v
                    self.tree_children[v].append(w)
                    self.height[w] = self.height[v] + 1
                    self.lowpt_v[w] = self.height[w]
                    self.lowpt2_v[w] = self.height[w]
                    self.dfs_order.append(w)
                    stack.append((w, 0))
                elif w != self.parent[v]:
                    hw = self.height[w]
                    if hw < self.lowpt_v[v]:
                        self.lowpt2_v[v] = self.lowpt_v[v]
                        self.lowpt_v[v] = hw
                    elif hw > self.lowpt_v[v]:
                        self.lowpt2_v[v] = min(self.lowpt2_v[v], hw)
            else:
                stack.pop()
                if self.parent[v] != -1:
                    pv = self.parent[v]
                    lv = self.lowpt_v[v]
                    if lv < self.lowpt_v[pv]:
                        self.lowpt2_v[pv] = min(self.lowpt_v[pv], self.lowpt2_v[v])
                        self.lowpt_v[pv] = lv
                    elif lv > self.lowpt_v[pv]:
                        self.lowpt2_v[pv] = min(self.lowpt2_v[pv], lv)
                    else:
                        self.lowpt2_v[pv] = min(self.lowpt2_v[pv], self.lowpt2_v[v])

        # Nesting depths
        for v in range(self.n):
            for w in self.adj[v]:
                if w == self.parent[v]:
                    continue
                e = (v, w)
                if self.parent[w] == v:
                    self.nesting_depth[e] = 2 * self.lowpt_v[w]
                    if self.lowpt2_v[w] < self.height[v]:
                        self.nesting_depth[e] += 1
                else:
                    self.nesting_depth[e] = 2 * self.height[w] + 1

        # Sort adjacency: include only tree children and back edges to ancestors.
        # Exclude parent edge and "forward" edges to descendants (those are
        # handled as back edges from the descendant's side).
        for v in range(self.n):
            out: list[int] = []
            for w in self.adj[v]:
                if self.parent[w] == v:
                    # Tree child
                    out.append(w)
                elif w != self.parent[v] and self.height[w] < self.height[v]:
                    # Back edge to ancestor
                    out.append(w)
            self.ordered_adj[v] = sorted(
                out,
                key=lambda w, _v=v: self.nesting_depth.get((_v, w), 2 * self.n),  # type: ignore[misc]
            )

    # -------------------------------------------------------------------
    # Phase 2: LR testing (iterative with explicit return frames)
    # -------------------------------------------------------------------

    def phase2_testing(self) -> bool:
        for v in range(self.n):
            for w in self.ordered_adj[v]:
                self.side[(v, w)] = 1
                self.ref[(v, w)] = None

        # Frame types on the stack:
        #   (_ENTER, v, idx)  -- process ordered_adj[v] starting from idx
        #   (_RETURN, u, v, idx) -- returned from tree edge (u,v); idx is
        #                           position of v in ordered_adj[u]
        #   (_INTEGRATE, v, idx) -- run integration for edge at idx
        #                           in ordered_adj[v]
        root = self.dfs_order[0]
        stack: list[tuple[int, ...]] = [(_ENTER, root, 0)]

        while stack:
            frame = stack[-1]
            ftype = frame[0]

            if ftype == _ENTER:
                _, v, idx = frame
                if idx >= len(self.ordered_adj[v]):
                    # Done with all edges of v
                    stack.pop()
                    # Remove back edges returning to parent of v
                    if self.parent[v] != -1:
                        pe = (self.parent[v], v)
                        self._remove_back_edges(pe)
                    continue

                w = self.ordered_adj[v][idx]
                ei = (v, w)

                # Set stack_bottom for EVERY edge (Brandes step 1)
                self.stack_bottom[ei] = self.S[-1] if self.S else None

                # Advance to next edge; push integrate after processing
                stack[-1] = (_ENTER, v, idx + 1)

                if self.parent[w] == v:
                    # Tree edge: integrate after return, then enter subtree
                    stack.append((_INTEGRATE, v, idx))
                    stack.append((_ENTER, w, 0))
                else:
                    # Back edge: push constraint, then integrate immediately
                    self.lowpt_edge[ei] = ei
                    self.S.append(
                        ConflictPair(
                            left=Interval(),
                            right=Interval(low=ei, high=ei),
                        )
                    )
                    # Integrate for back edge (push frame, it will execute next
                    # before the advance to idx+1)
                    stack.append((_INTEGRATE, v, idx))

            elif ftype == _INTEGRATE:
                _, v, idx = frame
                stack.pop()

                w = self.ordered_adj[v][idx]
                ei = (v, w)

                # Compute lowpt for this edge
                lp = self._lowpt_of_edge(ei)

                if lp < self.height[v]:
                    # Back edges from this edge's subtree go above v
                    parent_edge: Optional[tuple[int, int]] = None
                    if self.parent[v] != -1:
                        parent_edge = (self.parent[v], v)

                    if idx == 0:
                        # First edge of v: inherit lowpt_edge to parent
                        if parent_edge is not None:
                            self.lowpt_edge[parent_edge] = self.lowpt_edge.get(ei, ei)
                    else:
                        # Not first edge: add constraints
                        if parent_edge is not None:
                            if not self._add_constraints(ei, parent_edge):
                                return False

        return True

    def _add_constraints(self, ei: tuple[int, int], e: tuple[int, int]) -> bool:
        cp = ConflictPair(left=Interval(), right=Interval())

        # Phase A: merge return edges of ei
        # Pop everything above stack_bottom[ei], merging single-interval
        # pairs into cp.right, or aligning via ref when lowpt <= lowpt(e).
        sb = self.stack_bottom.get(ei)
        while self.S and self.S[-1] is not sb:
            q = self.S.pop()
            if not q.left.empty():
                q.swap()
            if not q.left.empty():
                return False
            # q.left is empty -- merge q.right
            assert q.right.low is not None
            if self._lowpt_of_edge(q.right.low) > self._lowpt_of_edge(e):
                # Must nest on same side as e
                if cp.right.empty():
                    cp.right = q.right.copy()
                else:
                    assert cp.right.low is not None
                    self.ref[cp.right.low] = q.right.high
                cp.right.low = q.right.low
            else:
                # Align: link to lowpt_edge of parent tree edge
                assert q.right.low is not None
                self.ref[q.right.low] = self.lowpt_edge.get(e, e)

        # Phase B: merge conflicting return edges of previous siblings.
        # No explicit boundary -- loop while top of stack conflicts with ei.
        while self.S and (
            self._conflicting(self.S[-1].left, ei) or self._conflicting(self.S[-1].right, ei)
        ):
            q = self.S.pop()
            if self._conflicting(q.right, ei):
                q.swap()
            if self._conflicting(q.right, ei):
                return False
            # Merge q.right below lowpt(ei) into cp.right
            if cp.right.low is not None:
                self.ref[cp.right.low] = q.right.high
            else:
                cp.right.high = q.right.high
            if q.right.low is not None:
                cp.right.low = q.right.low
            # Merge q.left into cp.left
            if cp.left.empty():
                cp.left = q.left.copy()
            else:
                assert cp.left.low is not None
                self.ref[cp.left.low] = q.left.high
            cp.left.low = q.left.low

        if not cp.left.empty() or not cp.right.empty():
            self.S.append(cp)

        return True

    def _lowest(self, cp: ConflictPair) -> int:
        """Return the lowest lowpoint of a conflict pair's intervals."""
        if cp.left.empty():
            assert cp.right.low is not None
            return self._lowpt_of_edge(cp.right.low)
        if cp.right.empty():
            assert cp.left.low is not None
            return self._lowpt_of_edge(cp.left.low)
        assert cp.left.low is not None and cp.right.low is not None
        return min(self._lowpt_of_edge(cp.left.low), self._lowpt_of_edge(cp.right.low))

    def _remove_back_edges(self, e: tuple[int, int]) -> None:
        u = e[0]

        # Drop entire conflict pairs whose lowest return equals height[u]
        while self.S and self._lowest(self.S[-1]) == self.height[u]:
            pair = self.S.pop()
            if pair.left.low is not None:
                self.side[pair.left.low] = -1

        if self.S:
            pair = self.S.pop()
            # Trim left interval
            while pair.left.high is not None and pair.left.high[1] == u:
                pair.left.high = self.ref.get(pair.left.high)
            if pair.left.high is None and pair.left.low is not None:
                # Left interval just emptied -- link ref and flip side
                self.ref[pair.left.low] = pair.right.low
                self.side[pair.left.low] = -1
                pair.left.low = None
            # Trim right interval
            while pair.right.high is not None and pair.right.high[1] == u:
                pair.right.high = self.ref.get(pair.right.high)
            if pair.right.high is None and pair.right.low is not None:
                # Right interval just emptied -- link ref and flip side
                self.ref[pair.right.low] = pair.left.low
                self.side[pair.right.low] = -1
                pair.right.low = None
            self.S.append(pair)

        # Set ref[e] to the highest return edge still on the stack
        if self._lowpt_of_edge(e) < self.height[u]:
            top = self.S[-1] if self.S else ConflictPair()
            hl = top.left.high
            hr = top.right.high
            if hl is not None and (hr is None or self._lowpt_of_edge(hl) > self._lowpt_of_edge(hr)):
                self.ref[e] = hl
            else:
                self.ref[e] = hr

    # -------------------------------------------------------------------
    # Phase 3: Embedding extraction
    # -------------------------------------------------------------------

    def _resolve_sign(self, e: tuple[int, int]) -> int:
        """Follow ref chain to resolve absolute side for edge e."""
        chain: list[tuple[int, int]] = []
        cur: Optional[tuple[int, int]] = e
        while cur is not None and self.ref.get(cur) is not None:
            chain.append(cur)
            cur = self.ref[cur]
        # cur has no ref -- its side is final
        result = self.side.get(cur, 1) if cur is not None else 1
        for edge in reversed(chain):
            self.side[edge] = self.side.get(edge, 1) * result
            self.ref[edge] = None
            result = self.side[edge]
        return self.side.get(e, 1)

    def phase3_embedding(self) -> dict[int, list[int]]:
        # Resolve signs for all edges (follows ref chains)
        for v in self.dfs_order:
            for w in self.ordered_adj[v]:
                self._resolve_sign((v, w))

        # Re-sort ordered_adj by sign * nesting_depth
        for v in range(self.n):
            self.ordered_adj[v] = sorted(
                self.ordered_adj[v],
                key=lambda w, _v=v: (  # type: ignore[misc]
                    self.side.get((_v, w), 1) * self.nesting_depth.get((_v, w), 0)
                ),
            )

        # Build rotation system via DFS
        rotation: dict[int, list[int]] = {v: [] for v in range(self.n)}
        left_ref: dict[int, int] = {}  # vertex value for left boundary
        right_ref: dict[int, int] = {}  # vertex value for right boundary

        stack: list[tuple[int, int]] = [(self.dfs_order[0], 0)]
        while stack:
            v, idx = stack[-1]
            if idx >= len(self.ordered_adj[v]):
                stack.pop()
                continue
            stack[-1] = (v, idx + 1)
            w = self.ordered_adj[v][idx]
            ei = (v, w)

            if self.parent[w] == v:  # tree edge
                # Add parent link as first entry in child's rotation
                rotation[w].insert(0, v)
                # Add child to parent's rotation
                rotation[v].append(w)
                # Set boundary refs for vertex v
                left_ref[v] = w
                right_ref[v] = w
                # Recurse into child
                stack.append((w, 0))
            else:  # back edge to ancestor w
                # Add ancestor to descendant's rotation
                rotation[v].append(w)
                # Insert descendant v into ancestor w's rotation
                rr = right_ref.get(w)
                lr = left_ref.get(w)
                if self.side.get(ei, 1) == 1:
                    if rr is not None and rr in rotation[w]:
                        pos = rotation[w].index(rr)
                        rotation[w].insert(pos + 1, v)
                    else:
                        rotation[w].append(v)
                else:
                    if lr is not None and lr in rotation[w]:
                        pos = rotation[w].index(lr)
                        rotation[w].insert(pos, v)
                    else:
                        rotation[w].append(v)
                    left_ref[w] = v

        return rotation


# ---------------------------------------------------------------------------
# Kuratowski subgraph extraction
# ---------------------------------------------------------------------------


def _find_kuratowski_subgraph(
    n: int,
    adj: list[list[int]],
) -> Optional[tuple[str, list[tuple[int, int]]]]:
    """Find a K5 or K3,3 subdivision in a non-planar graph.

    Two-phase approach:
    1. Edge deletion: remove edges while maintaining non-planarity to get a
       minimal non-planar subgraph (a subdivision of K5 or K3,3).
    2. Identification: degree-2 vertices are subdivision vertices; smooth them
       to identify the K5 or K3,3 structure.

    Returns:
        ("K5", edges) or ("K3,3", edges), or None if extraction fails.
    """
    # Skip extraction for large graphs (too expensive for O(m^2) edge deletion)
    m = sum(len(adj[v]) for v in range(n)) // 2
    if n > 50 or m > 200:
        return None

    # Build working adjacency
    work_adj: list[set[int]] = [set() for _ in range(n)]
    for v in range(n):
        for w in adj[v]:
            if w > v:
                work_adj[v].add(w)
                work_adj[w].add(v)

    def _is_planar_sub(wadj: list[set[int]], n_total: int) -> bool:
        """Planarity check on the subgraph induced by non-empty adjacencies."""
        verts = [v for v in range(n_total) if wadj[v]]
        if not verts:
            return True
        n_sub = len(verts)
        if n_sub <= 4:
            return True
        local_map = {v: i for i, v in enumerate(verts)}
        local_adj: list[list[int]] = [[] for _ in range(n_sub)]
        for v in verts:
            for w in wadj[v]:
                if w in local_map:
                    local_adj[local_map[v]].append(local_map[w])
        # Sort adjacency lists for deterministic DFS across platforms
        for a in local_adj:
            a.sort()
        m_sub = sum(len(a) for a in local_adj) // 2
        if n_sub >= 3 and m_sub > 3 * n_sub - 6:
            return False
        st = _LRState(n_sub, local_adj)
        st.phase1_orientation()
        return st.phase2_testing()

    # Phase 1: Delete edges to reach a minimal non-planar subgraph.
    # A minimal non-planar subgraph is a subdivision of K5 or K3,3.
    changed = True
    while changed:
        changed = False
        for v in range(n):
            for w in sorted(work_adj[v]):
                if w <= v:
                    continue
                # Try removing edge (v, w)
                work_adj[v].discard(w)
                work_adj[w].discard(v)
                if _is_planar_sub(work_adj, n):
                    # Edge is essential; put it back
                    work_adj[v].add(w)
                    work_adj[w].add(v)
                else:
                    # Edge removed successfully
                    changed = True

    # Phase 2: Identify the K5 or K3,3 structure.
    # Collect edges and vertices of the minimal subgraph.
    sub_edges: list[tuple[int, int]] = []
    degrees: dict[int, int] = {}
    for v in range(n):
        for w in work_adj[v]:
            if w > v:
                sub_edges.append((v, w))
        if work_adj[v]:
            degrees[v] = len(work_adj[v])

    if not sub_edges:
        return None

    # Remove isolated vertices (degree 0) and identify branch vertices
    # In a subdivision of K5: branch vertices have degree 4
    # In a subdivision of K3,3: branch vertices have degree 3
    # Subdivision vertices have degree 2
    branch_verts = [v for v, d in degrees.items() if d >= 3]

    # Smooth degree-2 vertices to find the underlying structure
    # Each degree-2 vertex is on a subdivision path between two branch vertices
    # Build the contracted graph (only branch vertices, with paths between them)
    # Also track which original edges are on each path
    contracted_adj: dict[int, set[int]] = {v: set() for v in branch_verts}
    path_edges: dict[tuple[int, int], list[tuple[int, int]]] = {}

    visited_deg2: set[int] = set()
    for bv in branch_verts:
        for w in sorted(work_adj[bv]):
            if w in visited_deg2:
                continue
            # Follow the path from bv through degree-2 vertices
            path = [(bv, w)]  # original edges on this path
            cur = w
            prev = bv
            while cur in degrees and degrees[cur] == 2:
                visited_deg2.add(cur)
                nbs = sorted(x for x in work_adj[cur] if x != prev)
                if not nbs:
                    break
                nxt = nbs[0]
                path.append((cur, nxt))
                prev = cur
                cur = nxt
            # cur should now be a branch vertex (or we ran out)
            if cur in contracted_adj and bv != cur:
                key = (min(bv, cur), max(bv, cur))
                contracted_adj[bv].add(cur)
                contracted_adj[cur].add(bv)
                if key not in path_edges:
                    path_edges[key] = []
                path_edges[key].extend(path)

    nb = len(branch_verts)
    ne = sum(len(s) for s in contracted_adj.values()) // 2

    # Check for K5: 5 branch vertices, 10 edges
    if nb == 5 and ne == 10:
        all_orig: list[tuple[int, int]] = []
        for key, edges in path_edges.items():
            all_orig.extend(edges)
        return ("K5", _canonicalize_edges(all_orig))

    # Check for K3,3: 6 branch vertices, 9 edges, bipartite
    if nb == 6 and ne == 9:
        # Check bipartiteness
        bv_list = sorted(branch_verts)
        bv_set = set(bv_list)
        # BFS 2-coloring
        color: dict[int, int] = {}
        color[bv_list[0]] = 0
        queue = [bv_list[0]]
        is_bipartite = True
        while queue and is_bipartite:
            v = queue.pop(0)
            for w in contracted_adj[v]:
                if w not in bv_set:
                    continue
                if w not in color:
                    color[w] = 1 - color[v]
                    queue.append(w)
                elif color[w] == color[v]:
                    is_bipartite = False
        if is_bipartite and len(color) == 6:
            all_orig = []
            for key, edges in path_edges.items():
                all_orig.extend(edges)
            return ("K3,3", _canonicalize_edges(all_orig))

    return None


def _canonicalize_edges(
    edges: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Deduplicate and canonicalize edge list."""
    seen: set[tuple[int, int]] = set()
    result: list[tuple[int, int]] = []
    for u, v in edges:
        canon = (min(u, v), max(u, v))
        if canon not in seen:
            seen.add(canon)
            result.append(canon)
    return result


# ---------------------------------------------------------------------------
# Biconnected component decomposition
# ---------------------------------------------------------------------------


def biconnected_components(
    num_nodes: int,
    adj: list[list[int]],
) -> list[tuple[list[int], list[tuple[int, int]]]]:
    """Decompose a connected graph into biconnected components (Tarjan's)."""
    if num_nodes == 0:
        return []
    if num_nodes == 1:
        return [([0], [])]

    disc = [-1] * num_nodes
    low = [0] * num_nodes
    parent = [-1] * num_nodes
    timer = [0]
    edge_stack: list[tuple[int, int]] = []
    components: list[tuple[list[int], list[tuple[int, int]]]] = []

    def _dfs_bcc(root: int) -> None:
        stack: list[tuple[int, int]] = [(root, 0)]
        disc[root] = low[root] = timer[0]
        timer[0] += 1

        while stack:
            v, idx = stack[-1]
            if idx < len(adj[v]):
                stack[-1] = (v, idx + 1)
                w = adj[v][idx]
                if disc[w] == -1:
                    parent[w] = v
                    disc[w] = low[w] = timer[0]
                    timer[0] += 1
                    edge_stack.append((v, w))
                    stack.append((w, 0))
                elif w != parent[v] and disc[w] < disc[v]:
                    edge_stack.append((v, w))
                    low[v] = min(low[v], disc[w])
            else:
                stack.pop()
                if not stack:
                    break
                u = stack[-1][0]
                low[u] = min(low[u], low[v])

                children = sum(1 for x in range(num_nodes) if parent[x] == u)
                if (parent[u] == -1 and children > 1) or (parent[u] != -1 and low[v] >= disc[u]):
                    comp_edges: list[tuple[int, int]] = []
                    comp_verts: set[int] = set()
                    while edge_stack and edge_stack[-1] != (u, v):
                        ee = edge_stack.pop()
                        comp_edges.append(ee)
                        comp_verts.update(ee)
                    if edge_stack:
                        ee = edge_stack.pop()
                        comp_edges.append(ee)
                        comp_verts.update(ee)
                    if comp_edges:
                        components.append((sorted(comp_verts), comp_edges))
                v = stack[-1][0]

    for v in range(num_nodes):
        if disc[v] == -1:
            _dfs_bcc(v)
            if edge_stack:
                comp_verts_set: set[int] = set()
                for ee in edge_stack:
                    comp_verts_set.update(ee)
                components.append((sorted(comp_verts_set), list(edge_stack)))
                edge_stack.clear()

    has_edge: set[int] = set()
    for verts, _ in components:
        has_edge.update(verts)
    for v in range(num_nodes):
        if v not in has_edge:
            components.append(([v], []))

    return components


def test_biconnected(
    num_nodes: int,
    adj: list[list[int]],
) -> PlanarityResult:
    """Test planarity by decomposing into biconnected components."""
    if num_nodes <= 2:
        emb = {v: list(adj[v]) for v in range(num_nodes)}
        return PlanarityResult(is_planar=True, embedding=emb)

    comps = biconnected_components(num_nodes, adj)
    full_embedding: dict[int, list[int]] = {v: [] for v in range(num_nodes)}

    for comp_verts, comp_edges in comps:
        if not comp_edges:
            continue

        local_map = {v: i for i, v in enumerate(comp_verts)}
        n_local = len(comp_verts)
        local_adj: list[list[int]] = [[] for _ in range(n_local)]

        for a, b in comp_edges:
            la, lb = local_map[a], local_map[b]
            local_adj[la].append(lb)
            local_adj[lb].append(la)

        for i in range(n_local):
            seen: set[int] = set()
            unique: list[int] = []
            for w in local_adj[i]:
                if w not in seen:
                    seen.add(w)
                    unique.append(w)
            local_adj[i] = unique

        result = lr_planarity_test(n_local, local_adj)
        if not result.is_planar:
            # Map Kuratowski witness edges back to global vertex IDs
            k_edges = None
            k_type = result.kuratowski_type
            if result.kuratowski_edges is not None:
                k_edges = [(comp_verts[a], comp_verts[b]) for a, b in result.kuratowski_edges]
            return PlanarityResult(
                is_planar=False,
                kuratowski_edges=k_edges,
                kuratowski_type=k_type,
            )

        if result.embedding:
            for local_v, neighbors in result.embedding.items():
                global_v = comp_verts[local_v]
                global_neighbors = [comp_verts[w] for w in neighbors]
                full_embedding[global_v].extend(global_neighbors)

    return PlanarityResult(is_planar=True, embedding=full_embedding)
