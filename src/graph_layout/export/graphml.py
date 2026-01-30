"""
GraphML export for graph layouts.

Generates GraphML format representations, an XML-based format for
graph data interchange.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence
from xml.sax.saxutils import escape

if TYPE_CHECKING:
    from ..base import BaseLayout
    from ..types import Node


def to_graphml(
    layout: BaseLayout,
    *,
    graph_id: str = "G",
    directed: bool = False,
    include_positions: bool = True,
    include_size: bool = True,
    include_weights: bool = True,
    get_node_attrs: Optional[Any] = None,
    get_edge_attrs: Optional[Any] = None,
    extra_node_keys: Optional[dict[str, str]] = None,
    extra_edge_keys: Optional[dict[str, str]] = None,
) -> str:
    """
    Export a layout to GraphML format.

    Args:
        layout: A layout object with nodes and links after run()
        graph_id: ID for the graph element (default "G")
        directed: Whether edges are directed (default False)
        include_positions: Include x, y position data (default True)
        include_size: Include width, height data (default True)
        include_weights: Include edge weights (default True)
        get_node_attrs: Callable(node) -> dict for custom node data
        get_edge_attrs: Callable(link) -> dict for custom edge data
        extra_node_keys: Additional node data keys: {key_id: attr_name}
        extra_edge_keys: Additional edge data keys: {key_id: attr_name}

    Returns:
        GraphML XML string
    """
    nodes = layout.nodes
    links = layout.links

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns"',
        '         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
        '         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns',
        '         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">',
    ]

    # Define keys for node attributes
    key_id = 0

    node_key_map: dict[str, str] = {}

    if include_positions:
        lines.append(f'  <key id="d{key_id}" for="node" attr.name="x" attr.type="double"/>')
        node_key_map["x"] = f"d{key_id}"
        key_id += 1
        lines.append(f'  <key id="d{key_id}" for="node" attr.name="y" attr.type="double"/>')
        node_key_map["y"] = f"d{key_id}"
        key_id += 1

    if include_size:
        lines.append(f'  <key id="d{key_id}" for="node" attr.name="width" attr.type="double"/>')
        node_key_map["width"] = f"d{key_id}"
        key_id += 1
        lines.append(f'  <key id="d{key_id}" for="node" attr.name="height" attr.type="double"/>')
        node_key_map["height"] = f"d{key_id}"
        key_id += 1

    # Label key
    lines.append(f'  <key id="d{key_id}" for="node" attr.name="label" attr.type="string"/>')
    node_key_map["label"] = f"d{key_id}"
    key_id += 1

    # Extra node keys
    if extra_node_keys:
        for key_name, attr_name in extra_node_keys.items():
            lines.append(
                f'  <key id="d{key_id}" for="node" attr.name="{escape(attr_name)}" '
                f'attr.type="string"/>'
            )
            node_key_map[key_name] = f"d{key_id}"
            key_id += 1

    # Edge keys
    edge_key_map: dict[str, str] = {}

    if include_weights:
        lines.append(f'  <key id="d{key_id}" for="edge" attr.name="weight" attr.type="double"/>')
        edge_key_map["weight"] = f"d{key_id}"
        key_id += 1
        lines.append(f'  <key id="d{key_id}" for="edge" attr.name="length" attr.type="double"/>')
        edge_key_map["length"] = f"d{key_id}"
        key_id += 1

    # Extra edge keys
    if extra_edge_keys:
        for key_name, attr_name in extra_edge_keys.items():
            lines.append(
                f'  <key id="d{key_id}" for="edge" attr.name="{escape(attr_name)}" '
                f'attr.type="string"/>'
            )
            edge_key_map[key_name] = f"d{key_id}"
            key_id += 1

    # Graph element
    edge_default = "directed" if directed else "undirected"
    lines.append(f'  <graph id="{escape(graph_id)}" edgedefault="{edge_default}">')

    # Nodes
    for node in nodes:
        node_id = _get_node_id(node)
        lines.append(f'    <node id="{escape(node_id)}">')

        if include_positions:
            lines.append(f'      <data key="{node_key_map["x"]}">{node.x:.6f}</data>')
            lines.append(f'      <data key="{node_key_map["y"]}">{node.y:.6f}</data>')

        if include_size:
            w = node.width if node.width else 1.0
            h = node.height if node.height else 1.0
            lines.append(f'      <data key="{node_key_map["width"]}">{w:.6f}</data>')
            lines.append(f'      <data key="{node_key_map["height"]}">{h:.6f}</data>')

        # Label
        label = _get_node_label(node)
        if label:
            lines.append(f'      <data key="{node_key_map["label"]}">{escape(label)}</data>')

        # Custom attributes
        if get_node_attrs:
            custom = get_node_attrs(node)
            if custom:
                for key, value in custom.items():
                    if key in node_key_map:
                        lines.append(
                            f'      <data key="{node_key_map[key]}">{escape(str(value))}</data>'
                        )

        lines.append("    </node>")

    # Edges
    edge_num = 0
    for link in links:
        src_idx = link.source if isinstance(link.source, int) else link.source.index
        tgt_idx = link.target if isinstance(link.target, int) else link.target.index

        if src_idx is None or tgt_idx is None:
            continue
        if src_idx >= len(nodes) or tgt_idx >= len(nodes):
            continue

        src_node = nodes[src_idx]
        tgt_node = nodes[tgt_idx]

        src_id = _get_node_id(src_node)
        tgt_id = _get_node_id(tgt_node)

        edge_id = f"e{edge_num}"
        edge_num += 1

        lines.append(
            f'    <edge id="{edge_id}" source="{escape(src_id)}" target="{escape(tgt_id)}">'
        )

        if include_weights and link.weight is not None:
            lines.append(f'      <data key="{edge_key_map["weight"]}">{link.weight:.6f}</data>')

        if include_weights and link.length is not None:
            lines.append(f'      <data key="{edge_key_map["length"]}">{link.length:.6f}</data>')

        # Custom attributes
        if get_edge_attrs:
            custom = get_edge_attrs(link)
            if custom:
                for key, value in custom.items():
                    if key in edge_key_map:
                        lines.append(
                            f'      <data key="{edge_key_map[key]}">{escape(str(value))}</data>'
                        )

        lines.append("    </edge>")

    lines.append("  </graph>")
    lines.append("</graphml>")

    return "\n".join(lines)


def to_graphml_orthogonal(
    boxes: Sequence[Any],
    edges: Sequence[Any],
    *,
    graph_id: str = "G",
    directed: bool = False,
    include_bends: bool = True,
) -> str:
    """
    Export an orthogonal layout to GraphML format.

    This includes bend point information for orthogonal edges.

    Args:
        boxes: List of NodeBox objects
        edges: List of OrthogonalEdge objects
        graph_id: ID for the graph element
        directed: Whether edges are directed
        include_bends: Include bend point data (default True)

    Returns:
        GraphML XML string
    """
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns"',
        '         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
        '         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns',
        '         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">',
    ]

    # Node position and size keys
    lines.append('  <key id="x" for="node" attr.name="x" attr.type="double"/>')
    lines.append('  <key id="y" for="node" attr.name="y" attr.type="double"/>')
    lines.append('  <key id="width" for="node" attr.name="width" attr.type="double"/>')
    lines.append('  <key id="height" for="node" attr.name="height" attr.type="double"/>')
    lines.append('  <key id="label" for="node" attr.name="label" attr.type="string"/>')

    # Edge bend key
    if include_bends:
        lines.append('  <key id="bends" for="edge" attr.name="bends" attr.type="string"/>')
        lines.append(
            '  <key id="source_side" for="edge" attr.name="source_side" attr.type="string"/>'
        )
        lines.append(
            '  <key id="target_side" for="edge" attr.name="target_side" attr.type="string"/>'
        )

    # Graph
    edge_default = "directed" if directed else "undirected"
    lines.append(f'  <graph id="{escape(graph_id)}" edgedefault="{edge_default}">')

    # Nodes
    for box in boxes:
        node_id = str(box.index)
        lines.append(f'    <node id="{escape(node_id)}">')
        lines.append(f'      <data key="x">{box.x:.6f}</data>')
        lines.append(f'      <data key="y">{box.y:.6f}</data>')
        lines.append(f'      <data key="width">{box.width:.6f}</data>')
        lines.append(f'      <data key="height">{box.height:.6f}</data>')

        label = str(box.index)
        if hasattr(box, "label"):
            label = str(getattr(box, "label"))
        lines.append(f'      <data key="label">{escape(label)}</data>')

        lines.append("    </node>")

    # Edges
    for i, edge in enumerate(edges):
        src = str(edge.source)
        tgt = str(edge.target)

        lines.append(f'    <edge id="e{i}" source="{escape(src)}" target="{escape(tgt)}">')

        if include_bends:
            # Encode bends as comma-separated coordinate pairs
            if hasattr(edge, "bends") and edge.bends:
                bends_str = ";".join(f"{x:.2f},{y:.2f}" for x, y in edge.bends)
                lines.append(f'      <data key="bends">{bends_str}</data>')

            # Port sides
            if hasattr(edge, "source_port") and edge.source_port:
                lines.append(f'      <data key="source_side">{edge.source_port.side.value}</data>')
            if hasattr(edge, "target_port") and edge.target_port:
                lines.append(f'      <data key="target_side">{edge.target_port.side.value}</data>')

        lines.append("    </edge>")

    lines.append("  </graph>")
    lines.append("</graphml>")

    return "\n".join(lines)


def _get_node_id(node: Node) -> str:
    """Get a unique identifier for a node."""
    if hasattr(node, "id"):
        return str(getattr(node, "id"))
    if hasattr(node, "name"):
        return str(getattr(node, "name"))
    if node.index is not None:
        return str(node.index)
    return "0"


def _get_node_label(node: Node) -> Optional[str]:
    """Get label text for a node."""
    if hasattr(node, "label"):
        return str(getattr(node, "label"))
    if hasattr(node, "name"):
        return str(getattr(node, "name"))
    if node.index is not None:
        return str(node.index)
    return None


__all__ = [
    "to_graphml",
    "to_graphml_orthogonal",
]
