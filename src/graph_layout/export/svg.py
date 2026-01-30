"""
SVG export for graph layouts.

Generates SVG representations of graph layouts, supporting both simple
node-edge graphs and orthogonal layouts with bends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence
from xml.sax.saxutils import escape

if TYPE_CHECKING:
    from ..base import BaseLayout
    from ..types import Link, Node


def to_svg(
    layout: BaseLayout,
    *,
    node_radius: float = 20.0,
    node_color: str = "#4a90d9",
    node_stroke: str = "#2c5aa0",
    node_stroke_width: float = 2.0,
    edge_color: str = "#666666",
    edge_width: float = 1.5,
    show_labels: bool = True,
    label_color: str = "#000000",
    font_size: float = 12.0,
    font_family: str = "sans-serif",
    padding: float = 40.0,
    background: Optional[str] = None,
    node_shape: str = "circle",
) -> str:
    """
    Export a layout to SVG format.

    Args:
        layout: A layout object with nodes and links after run()
        node_radius: Radius for circular nodes (default 20)
        node_color: Fill color for nodes (default blue)
        node_stroke: Stroke color for nodes (default darker blue)
        node_stroke_width: Stroke width for nodes (default 2)
        edge_color: Color for edges (default gray)
        edge_width: Width for edges (default 1.5)
        show_labels: Whether to show node labels (default True)
        label_color: Color for labels (default black)
        font_size: Font size for labels (default 12)
        font_family: Font family for labels (default sans-serif)
        padding: Padding around the graph (default 40)
        background: Background color (default None for transparent)
        node_shape: Shape of nodes: "circle" or "rect" (default "circle")

    Returns:
        SVG string representation of the graph
    """
    nodes = layout.nodes
    links = layout.links

    if not nodes:
        return _empty_svg(layout.size[0], layout.size[1], background)

    # Calculate bounding box
    min_x = min(n.x for n in nodes)
    max_x = max(n.x for n in nodes)
    min_y = min(n.y for n in nodes)
    max_y = max(n.y for n in nodes)

    # Add node size to bounds
    for node in nodes:
        node_w = node.width if node.width else node_radius * 2
        node_h = node.height if node.height else node_radius * 2
        min_x = min(min_x, node.x - node_w / 2)
        max_x = max(max_x, node.x + node_w / 2)
        min_y = min(min_y, node.y - node_h / 2)
        max_y = max(max_y, node.y + node_h / 2)

    # Calculate SVG dimensions
    width = max_x - min_x + 2 * padding
    height = max_y - min_y + 2 * padding

    # Offset to center graph with padding
    offset_x = padding - min_x
    offset_y = padding - min_y

    # Build SVG
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width:.1f}" height="{height:.1f}" '
        f'viewBox="0 0 {width:.1f} {height:.1f}">'
    ]

    # Background
    if background:
        svg_parts.append(f'  <rect width="100%" height="100%" fill="{escape(background)}"/>')

    # Edges group
    svg_parts.append('  <g class="edges">')
    for link in links:
        edge_svg = _render_edge(link, nodes, offset_x, offset_y, edge_color, edge_width)
        if edge_svg:
            svg_parts.append(edge_svg)
    svg_parts.append("  </g>")

    # Nodes group
    svg_parts.append('  <g class="nodes">')
    for node in nodes:
        node_svg = _render_node(
            node,
            offset_x,
            offset_y,
            node_radius,
            node_color,
            node_stroke,
            node_stroke_width,
            node_shape,
        )
        svg_parts.append(node_svg)
    svg_parts.append("  </g>")

    # Labels group
    if show_labels:
        svg_parts.append('  <g class="labels">')
        for node in nodes:
            label_svg = _render_label(node, offset_x, offset_y, label_color, font_size, font_family)
            svg_parts.append(label_svg)
        svg_parts.append("  </g>")

    svg_parts.append("</svg>")

    return "\n".join(svg_parts)


def to_svg_orthogonal(
    boxes: Sequence[Any],
    edges: Sequence[Any],
    *,
    node_color: str = "#4a90d9",
    node_stroke: str = "#2c5aa0",
    node_stroke_width: float = 2.0,
    edge_color: str = "#666666",
    edge_width: float = 1.5,
    show_labels: bool = True,
    label_color: str = "#000000",
    font_size: float = 12.0,
    font_family: str = "sans-serif",
    padding: float = 40.0,
    background: Optional[str] = None,
) -> str:
    """
    Export an orthogonal layout to SVG format.

    This function handles orthogonal layouts with rectangular nodes
    and edges that have bends.

    Args:
        boxes: List of NodeBox objects from orthogonal layout
        edges: List of OrthogonalEdge objects with bends
        node_color: Fill color for nodes
        node_stroke: Stroke color for nodes
        node_stroke_width: Stroke width for nodes
        edge_color: Color for edges
        edge_width: Width for edges
        show_labels: Whether to show node labels
        label_color: Color for labels
        font_size: Font size for labels
        font_family: Font family for labels
        padding: Padding around the graph
        background: Background color (None for transparent)

    Returns:
        SVG string representation of the orthogonal graph
    """
    if not boxes:
        return _empty_svg(100, 100, background)

    # Calculate bounding box from boxes
    min_x = min(box.left for box in boxes)
    max_x = max(box.right for box in boxes)
    min_y = min(box.top for box in boxes)
    max_y = max(box.bottom for box in boxes)

    # Include edge bends in bounds
    for edge in edges:
        if hasattr(edge, "bends"):
            for bx, by in edge.bends:
                min_x = min(min_x, bx)
                max_x = max(max_x, bx)
                min_y = min(min_y, by)
                max_y = max(max_y, by)

    # Calculate dimensions
    width = max_x - min_x + 2 * padding
    height = max_y - min_y + 2 * padding
    offset_x = padding - min_x
    offset_y = padding - min_y

    # Build SVG
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width:.1f}" height="{height:.1f}" '
        f'viewBox="0 0 {width:.1f} {height:.1f}">'
    ]

    if background:
        svg_parts.append(f'  <rect width="100%" height="100%" fill="{escape(background)}"/>')

    # Edges with orthogonal routing
    svg_parts.append('  <g class="edges">')
    for edge in edges:
        edge_svg = _render_orthogonal_edge(edge, boxes, offset_x, offset_y, edge_color, edge_width)
        if edge_svg:
            svg_parts.append(edge_svg)
    svg_parts.append("  </g>")

    # Rectangular nodes
    svg_parts.append('  <g class="nodes">')
    for box in boxes:
        box_svg = _render_box(box, offset_x, offset_y, node_color, node_stroke, node_stroke_width)
        svg_parts.append(box_svg)
    svg_parts.append("  </g>")

    # Labels
    if show_labels:
        svg_parts.append('  <g class="labels">')
        for box in boxes:
            label_svg = _render_box_label(
                box, offset_x, offset_y, label_color, font_size, font_family
            )
            svg_parts.append(label_svg)
        svg_parts.append("  </g>")

    svg_parts.append("</svg>")

    return "\n".join(svg_parts)


def _empty_svg(width: float, height: float, background: Optional[str]) -> str:
    """Create an empty SVG."""
    bg = ""
    if background:
        bg = f'\n  <rect width="100%" height="100%" fill="{escape(background)}"/>'
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width:.1f}" height="{height:.1f}" '
        f'viewBox="0 0 {width:.1f} {height:.1f}">{bg}\n</svg>'
    )


def _render_edge(
    link: Link,
    nodes: list[Node],
    offset_x: float,
    offset_y: float,
    color: str,
    width: float,
) -> Optional[str]:
    """Render a simple straight edge."""
    src_idx = link.source if isinstance(link.source, int) else link.source.index
    tgt_idx = link.target if isinstance(link.target, int) else link.target.index

    if src_idx is None or tgt_idx is None:
        return None
    if src_idx >= len(nodes) or tgt_idx >= len(nodes):
        return None

    src = nodes[src_idx]
    tgt = nodes[tgt_idx]

    x1 = src.x + offset_x
    y1 = src.y + offset_y
    x2 = tgt.x + offset_x
    y2 = tgt.y + offset_y

    return (
        f'    <line x1="{x1:.1f}" y1="{y1:.1f}" '
        f'x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{escape(color)}" stroke-width="{width}"/>'
    )


def _render_node(
    node: Node,
    offset_x: float,
    offset_y: float,
    radius: float,
    fill: str,
    stroke: str,
    stroke_width: float,
    shape: str,
) -> str:
    """Render a node."""
    x = node.x + offset_x
    y = node.y + offset_y

    if shape == "rect":
        w = node.width if node.width else radius * 2
        h = node.height if node.height else radius * 2
        return (
            f'    <rect x="{x - w / 2:.1f}" y="{y - h / 2:.1f}" '
            f'width="{w:.1f}" height="{h:.1f}" '
            f'fill="{escape(fill)}" stroke="{escape(stroke)}" '
            f'stroke-width="{stroke_width}" rx="4"/>'
        )
    else:  # circle
        r = radius
        if node.width and node.height:
            r = min(node.width, node.height) / 2
        return (
            f'    <circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" '
            f'fill="{escape(fill)}" stroke="{escape(stroke)}" '
            f'stroke-width="{stroke_width}"/>'
        )


def _render_label(
    node: Node,
    offset_x: float,
    offset_y: float,
    color: str,
    font_size: float,
    font_family: str,
) -> str:
    """Render a node label."""
    x = node.x + offset_x
    y = node.y + offset_y

    # Get label text
    label = str(node.index) if node.index is not None else ""
    if hasattr(node, "label"):
        label = str(getattr(node, "label"))
    elif hasattr(node, "name"):
        label = str(getattr(node, "name"))

    return (
        f'    <text x="{x:.1f}" y="{y:.1f}" '
        f'fill="{escape(color)}" font-size="{font_size}" '
        f'font-family="{escape(font_family)}" '
        f'text-anchor="middle" dominant-baseline="central">'
        f"{escape(label)}</text>"
    )


def _render_orthogonal_edge(
    edge: Any,
    boxes: Sequence[Any],
    offset_x: float,
    offset_y: float,
    color: str,
    width: float,
) -> Optional[str]:
    """Render an orthogonal edge with bends."""
    src_idx = edge.source
    tgt_idx = edge.target

    if src_idx >= len(boxes) or tgt_idx >= len(boxes):
        return None

    # Get source and target port positions
    src_box = boxes[src_idx]
    tgt_box = boxes[tgt_idx]

    if hasattr(edge, "source_port") and edge.source_port:
        src_x, src_y = src_box.get_port_position(edge.source_port.side, edge.source_port.position)
    else:
        src_x, src_y = src_box.x, src_box.y

    if hasattr(edge, "target_port") and edge.target_port:
        tgt_x, tgt_y = tgt_box.get_port_position(edge.target_port.side, edge.target_port.position)
    else:
        tgt_x, tgt_y = tgt_box.x, tgt_box.y

    # Build path
    points = [(src_x + offset_x, src_y + offset_y)]

    if hasattr(edge, "bends"):
        for bx, by in edge.bends:
            points.append((bx + offset_x, by + offset_y))

    points.append((tgt_x + offset_x, tgt_y + offset_y))

    # Create polyline path
    path_data = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in points)

    return (
        f'    <polyline points="{path_data}" '
        f'fill="none" stroke="{escape(color)}" stroke-width="{width}"/>'
    )


def _render_box(
    box: Any,
    offset_x: float,
    offset_y: float,
    fill: str,
    stroke: str,
    stroke_width: float,
) -> str:
    """Render a rectangular node box."""
    x = box.left + offset_x
    y = box.top + offset_y

    return (
        f'    <rect x="{x:.1f}" y="{y:.1f}" '
        f'width="{box.width:.1f}" height="{box.height:.1f}" '
        f'fill="{escape(fill)}" stroke="{escape(stroke)}" '
        f'stroke-width="{stroke_width}" rx="4"/>'
    )


def _render_box_label(
    box: Any,
    offset_x: float,
    offset_y: float,
    color: str,
    font_size: float,
    font_family: str,
) -> str:
    """Render a label for a node box."""
    x = box.x + offset_x
    y = box.y + offset_y

    label = str(box.index)
    if hasattr(box, "label"):
        label = str(getattr(box, "label"))
    elif hasattr(box, "name"):
        label = str(getattr(box, "name"))

    return (
        f'    <text x="{x:.1f}" y="{y:.1f}" '
        f'fill="{escape(color)}" font-size="{font_size}" '
        f'font-family="{escape(font_family)}" '
        f'text-anchor="middle" dominant-baseline="central">'
        f"{escape(label)}</text>"
    )


__all__ = [
    "to_svg",
    "to_svg_orthogonal",
]
