"""Block-cut tree for decomposing a graph at articulation points.

A block-cut tree captures the structure of biconnected components (blocks)
and cut vertices in a connected graph. Each block is a maximal biconnected
subgraph, and cut vertices are shared between adjacent blocks.

Used by embedding strategies to optimize face exposure at articulation points.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ._lr_planarity import biconnected_components


@dataclass
class Block:
    """A biconnected component (block) in the block-cut tree.

    Attributes:
        index: Unique identifier for this block.
        vertices: Set of vertices in this block.
        edges: Edges in this block as (u, v) tuples.
    """

    index: int
    vertices: set[int]
    edges: list[tuple[int, int]]


@dataclass
class BlockCutTree:
    """Block-cut tree of a connected graph.

    Attributes:
        blocks: List of blocks (biconnected components).
        cut_vertices: Set of articulation points.
        vertex_to_blocks: Mapping from vertex to indices of blocks containing it.
        block_adj: Adjacency between blocks (two blocks are adjacent if they
            share a cut vertex). Maps block index to set of adjacent block indices.
    """

    blocks: list[Block]
    cut_vertices: set[int]
    vertex_to_blocks: dict[int, list[int]]
    block_adj: dict[int, set[int]] = field(default_factory=dict)


def build_block_cut_tree(
    num_nodes: int,
    adj: list[list[int]],
) -> BlockCutTree:
    """Build the block-cut tree from an adjacency list.

    Decomposes the graph into biconnected components using Tarjan's algorithm
    (via existing ``biconnected_components``), identifies cut vertices, and
    builds the block adjacency structure.

    Args:
        num_nodes: Number of vertices (labeled 0..num_nodes-1).
        adj: Adjacency list for each vertex.

    Returns:
        BlockCutTree with blocks, cut vertices, and adjacency info.
    """
    raw_components = biconnected_components(num_nodes, adj)

    blocks: list[Block] = []
    vertex_to_blocks: dict[int, list[int]] = {}

    for i, (verts, edges) in enumerate(raw_components):
        block = Block(index=i, vertices=set(verts), edges=edges)
        blocks.append(block)
        for v in verts:
            vertex_to_blocks.setdefault(v, []).append(i)

    # Cut vertices appear in more than one block
    cut_vertices: set[int] = set()
    for v, block_indices in vertex_to_blocks.items():
        if len(block_indices) > 1:
            cut_vertices.add(v)

    # Build block adjacency via shared cut vertices
    block_adj: dict[int, set[int]] = {b.index: set() for b in blocks}
    for v in cut_vertices:
        bi = vertex_to_blocks[v]
        for a in bi:
            for b in bi:
                if a != b:
                    block_adj[a].add(b)

    return BlockCutTree(
        blocks=blocks,
        cut_vertices=cut_vertices,
        vertex_to_blocks=vertex_to_blocks,
        block_adj=block_adj,
    )
