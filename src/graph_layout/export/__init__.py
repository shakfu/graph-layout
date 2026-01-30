"""
Export functionality for graph layouts.

This module provides functions to export graph layouts to various formats:
- SVG: Scalable Vector Graphics for web and print
- DOT: Graphviz format for graph visualization tools
- GraphML: XML-based format for graph data interchange

Example usage:
    from graph_layout import CircularLayout
    from graph_layout.export import to_svg, to_dot, to_graphml

    layout = CircularLayout(
        nodes=[{"index": i} for i in range(5)],
        links=[{"source": i, "target": (i + 1) % 5} for i in range(5)],
        size=(400, 400),
    ).run()

    # Export to SVG
    svg_content = to_svg(layout)
    with open("graph.svg", "w") as f:
        f.write(svg_content)

    # Export to DOT
    dot_content = to_dot(layout)
    with open("graph.dot", "w") as f:
        f.write(dot_content)

    # Export to GraphML
    graphml_content = to_graphml(layout)
    with open("graph.graphml", "w") as f:
        f.write(graphml_content)
"""

from .dot import to_dot, to_dot_orthogonal
from .graphml import to_graphml, to_graphml_orthogonal
from .svg import to_svg, to_svg_orthogonal

__all__ = [
    # SVG export
    "to_svg",
    "to_svg_orthogonal",
    # DOT export
    "to_dot",
    "to_dot_orthogonal",
    # GraphML export
    "to_graphml",
    "to_graphml_orthogonal",
]
