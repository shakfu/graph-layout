"""
DOT (Graphviz) export for graph layouts.

Generates DOT format representations that can be used with Graphviz
tools (dot, neato, fdp, etc.) or imported into other graph tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence

if TYPE_CHECKING:
    from ..base import BaseLayout
    from ..types import Node


def to_dot(
    layout: BaseLayout,
    *,
    name: str = "G",
    directed: bool = False,
    include_positions: bool = True,
    node_shape: str = "ellipse",
    node_color: Optional[str] = None,
    node_fillcolor: Optional[str] = None,
    node_style: Optional[str] = None,
    edge_color: Optional[str] = None,
    graph_attrs: Optional[dict[str, str]] = None,
    node_attrs: Optional[dict[str, str]] = None,
    edge_attrs: Optional[dict[str, str]] = None,
    get_node_label: Optional[Any] = None,
    get_node_attrs: Optional[Any] = None,
    get_edge_attrs: Optional[Any] = None,
) -> str:
    """
    Export a layout to DOT (Graphviz) format.

    Args:
        layout: A layout object with nodes and links after run()
        name: Name of the graph (default "G")
        directed: Whether the graph is directed (default False)
        include_positions: Include pos attributes for nodes (default True)
        node_shape: Default node shape (default "ellipse")
        node_color: Default node border color
        node_fillcolor: Default node fill color
        node_style: Default node style (e.g., "filled")
        edge_color: Default edge color
        graph_attrs: Additional graph-level attributes
        node_attrs: Additional default node attributes
        edge_attrs: Additional default edge attributes
        get_node_label: Callable(node) -> str for custom node labels
        get_node_attrs: Callable(node) -> dict for custom node attributes
        get_edge_attrs: Callable(link) -> dict for custom edge attributes

    Returns:
        DOT format string representation of the graph
    """
    nodes = layout.nodes
    links = layout.links

    graph_type = "digraph" if directed else "graph"
    edge_op = "->" if directed else "--"

    lines = [f"{graph_type} {_quote_id(name)} {{"]

    # Graph attributes
    all_graph_attrs: dict[str, str] = {}
    if graph_attrs:
        all_graph_attrs.update(graph_attrs)

    if all_graph_attrs:
        lines.append(_format_attrs_block("graph", all_graph_attrs))

    # Default node attributes
    all_node_attrs: dict[str, str] = {"shape": node_shape}
    if node_color:
        all_node_attrs["color"] = node_color
    if node_fillcolor:
        all_node_attrs["fillcolor"] = node_fillcolor
        all_node_attrs["style"] = node_style or "filled"
    elif node_style:
        all_node_attrs["style"] = node_style
    if node_attrs:
        all_node_attrs.update(node_attrs)

    if all_node_attrs:
        lines.append(_format_attrs_block("node", all_node_attrs))

    # Default edge attributes
    all_edge_attrs: dict[str, str] = {}
    if edge_color:
        all_edge_attrs["color"] = edge_color
    if edge_attrs:
        all_edge_attrs.update(edge_attrs)

    if all_edge_attrs:
        lines.append(_format_attrs_block("edge", all_edge_attrs))

    lines.append("")

    # Nodes
    for node in nodes:
        node_id = _get_node_id(node)
        node_data: dict[str, str] = {}

        # Label
        if get_node_label:
            node_data["label"] = str(get_node_label(node))
        else:
            label = _get_node_label(node)
            if label:
                node_data["label"] = label

        # Position
        if include_positions:
            # Graphviz pos format: "x,y!" (! means fixed)
            node_data["pos"] = f"{node.x:.2f},{-node.y:.2f}!"

        # Size
        if node.width and node.height:
            # Graphviz uses inches, assume 72 dpi for pixels
            node_data["width"] = f"{node.width / 72:.2f}"
            node_data["height"] = f"{node.height / 72:.2f}"

        # Custom attributes
        if get_node_attrs:
            custom_attrs = get_node_attrs(node)
            if custom_attrs:
                node_data.update(custom_attrs)

        lines.append(f"  {_quote_id(node_id)}{_format_attrs(node_data)};")

    lines.append("")

    # Edges
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

        edge_data: dict[str, str] = {}

        # Weight
        if link.weight is not None:
            edge_data["weight"] = str(link.weight)

        # Length (mapped to len attribute)
        if link.length is not None:
            edge_data["len"] = f"{link.length:.2f}"

        # Custom attributes
        if get_edge_attrs:
            custom_attrs = get_edge_attrs(link)
            if custom_attrs:
                edge_data.update(custom_attrs)

        lines.append(
            f"  {_quote_id(src_id)} {edge_op} {_quote_id(tgt_id)}{_format_attrs(edge_data)};"
        )

    lines.append("}")

    return "\n".join(lines)


def to_dot_orthogonal(
    boxes: Sequence[Any],
    edges: Sequence[Any],
    *,
    name: str = "G",
    directed: bool = False,
    include_positions: bool = True,
    node_shape: str = "box",
    node_color: Optional[str] = None,
    node_fillcolor: Optional[str] = None,
    graph_attrs: Optional[dict[str, str]] = None,
) -> str:
    """
    Export an orthogonal layout to DOT format.

    This preserves rectangular node shapes and includes edge bend points
    as spline control points where applicable.

    Args:
        boxes: List of NodeBox objects from orthogonal layout
        edges: List of OrthogonalEdge objects
        name: Name of the graph
        directed: Whether the graph is directed
        include_positions: Include pos attributes
        node_shape: Node shape (default "box" for orthogonal)
        node_color: Node border color
        node_fillcolor: Node fill color
        graph_attrs: Additional graph attributes

    Returns:
        DOT format string
    """
    graph_type = "digraph" if directed else "graph"
    edge_op = "->" if directed else "--"

    lines = [f"{graph_type} {_quote_id(name)} {{"]

    # Graph attributes for orthogonal
    all_attrs: dict[str, str] = {"splines": "ortho"}
    if graph_attrs:
        all_attrs.update(graph_attrs)
    lines.append(_format_attrs_block("graph", all_attrs))

    # Default node attributes
    node_attrs: dict[str, str] = {"shape": node_shape}
    if node_color:
        node_attrs["color"] = node_color
    if node_fillcolor:
        node_attrs["fillcolor"] = node_fillcolor
        node_attrs["style"] = "filled"
    lines.append(_format_attrs_block("node", node_attrs))

    lines.append("")

    # Nodes (boxes)
    for box in boxes:
        box_data: dict[str, str] = {}
        box_data["label"] = str(box.index)

        if include_positions:
            box_data["pos"] = f"{box.x:.2f},{-box.y:.2f}!"

        # Box size in inches (72 dpi)
        box_data["width"] = f"{box.width / 72:.2f}"
        box_data["height"] = f"{box.height / 72:.2f}"

        if hasattr(box, "label"):
            box_data["label"] = str(getattr(box, "label"))

        lines.append(f"  {box.index}{_format_attrs(box_data)};")

    lines.append("")

    # Edges
    for edge in edges:
        src = edge.source
        tgt = edge.target

        if src >= len(boxes) or tgt >= len(boxes):
            continue

        edge_data: dict[str, str] = {}

        # Include bend information as comment (DOT doesn't support arbitrary bends)
        if hasattr(edge, "bends") and edge.bends:
            edge_data["comment"] = f"bends: {len(edge.bends)}"

        lines.append(f"  {src} {edge_op} {tgt}{_format_attrs(edge_data)};")

    lines.append("}")

    return "\n".join(lines)


def _get_node_id(node: Node) -> str:
    """Get a unique identifier for a node."""
    if hasattr(node, "id"):
        return str(getattr(node, "id"))
    if hasattr(node, "name"):
        return str(getattr(node, "name"))
    if node.index is not None:
        return str(node.index)
    return "node"


def _get_node_label(node: Node) -> Optional[str]:
    """Get label text for a node."""
    if hasattr(node, "label"):
        return str(getattr(node, "label"))
    if hasattr(node, "name"):
        return str(getattr(node, "name"))
    return None


def _quote_id(s: str) -> str:
    """Quote a DOT identifier if necessary."""
    # Check if it needs quoting (contains special chars or starts with digit)
    if not s:
        return '""'

    # Simple identifiers don't need quoting
    if s.isidentifier() or s.isdigit():
        return s

    # Quote and escape
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _format_attrs(attrs: dict[str, str]) -> str:
    """Format attributes as DOT attribute list."""
    if not attrs:
        return ""

    parts = []
    for key, value in attrs.items():
        # Quote value if needed
        if value.startswith('"') and value.endswith('"'):
            parts.append(f"{key}={value}")
        elif " " in value or "," in value or "!" in value:
            parts.append(f'{key}="{value}"')
        else:
            parts.append(f"{key}={_quote_id(value)}")

    return " [" + ", ".join(parts) + "]"


def _format_attrs_block(element: str, attrs: dict[str, str]) -> str:
    """Format a default attributes block."""
    if not attrs:
        return ""

    parts = []
    for key, value in attrs.items():
        if " " in value or "," in value:
            parts.append(f'{key}="{value}"')
        else:
            parts.append(f"{key}={_quote_id(value)}")

    return f"  {element} [{', '.join(parts)}];"


__all__ = [
    "to_dot",
    "to_dot_orthogonal",
]
