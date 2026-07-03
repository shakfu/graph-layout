#!/usr/bin/env python3
"""Visual showcase of the correctness/feature improvements made in this review.

Renders side-by-side SVG comparisons that make the changes visible:

  * GIOTTO ``bend_optimal`` (Topology-Shape-Metrics) vs the routing heuristic --
    the bend-minimal orthogonal representation now drives the drawing.
  * Cola constraint enforcement (C1) -- overlap avoidance and separation
    constraints, which were previously inert stubs.
  * Sugiyama on a cyclic graph (H3/H4) -- cycle removal + dummy nodes; GIOTTO on
    a cyclic graph (recursion fix).

Usage:
    uv run python tests/demos/improvements_showcase.py
Output:
    build/improvements_showcase.html
"""

from __future__ import annotations

from html import escape
from pathlib import Path

from graph_layout import GIOTTOLayout, Link, Node, SugiyamaLayout
from graph_layout.cola import Layout as ColaLayout

BUILD_DIR = Path(__file__).parent.parent.parent / "build"
W, H = 360, 300
PAD = 28


# --------------------------------------------------------------------------- #
# Small SVG helpers
# --------------------------------------------------------------------------- #


def _fit(points: list[tuple[float, float]], w: int, h: int, pad: int):
    """Return a transform mapping the point bbox into a w x h box with padding."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    span_x = max(maxx - minx, 1e-9)
    span_y = max(maxy - miny, 1e-9)
    scale = min((w - 2 * pad) / span_x, (h - 2 * pad) / span_y)
    ox = (w - scale * span_x) / 2
    oy = (h - scale * span_y) / 2

    def tf(x: float, y: float) -> tuple[float, float]:
        return (ox + (x - minx) * scale, oy + (y - miny) * scale)

    return tf


def _orthogonal_svg(layout, title: str, subtitle: str) -> str:
    """Render a GIOTTO/orthogonal layout (node boxes + orthogonal edges)."""
    boxes = layout.node_boxes
    polylines: list[list[tuple[float, float]]] = []
    pts: list[tuple[float, float]] = []
    for e in layout.orthogonal_edges:
        sp = boxes[e.source].get_port_position(e.source_port.side, e.source_port.position)
        tp = boxes[e.target].get_port_position(e.target_port.side, e.target_port.position)
        line = [sp, *list(e.bends), tp]
        polylines.append(line)
        pts.extend(line)
    for b in boxes:
        pts.append((b.left, b.top))
        pts.append((b.right, b.bottom))
    if not pts:
        pts = [(0, 0), (1, 1)]
    tf = _fit(pts, W, H, PAD)

    parts = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    parts.append(f'<rect width="{W}" height="{H}" fill="#fbfbfd"/>')
    for line in polylines:
        d = " ".join(f"{tf(x, y)[0]:.1f},{tf(x, y)[1]:.1f}" for x, y in line)
        parts.append(f'<polyline points="{d}" fill="none" stroke="#888" stroke-width="1.6"/>')
    # boxes
    box_w = 18.0
    box_h = 14.0
    for i, b in enumerate(boxes):
        cx, cy = tf(b.x, b.y)
        parts.append(
            f'<rect x="{cx - box_w / 2:.1f}" y="{cy - box_h / 2:.1f}" '
            f'width="{box_w}" height="{box_h}" rx="2" '
            f'fill="#4a90d9" stroke="#2c5aa0" stroke-width="1.5"/>'
        )
        parts.append(
            f'<text x="{cx:.1f}" y="{cy + 3:.1f}" text-anchor="middle" '
            f'font-size="9" fill="#fff" font-family="sans-serif">{i}</text>'
        )
    parts.append("</svg>")
    return _card(title, subtitle, "".join(parts))


def _node_svg(
    layout,
    edges: list[tuple[int, int]],
    title: str,
    subtitle: str,
    boxes_wh: tuple[float, float] | None = None,
) -> str:
    """Render a node/edge layout (circles + straight edges), optional boxes."""
    nodes = _nodes_of(layout)
    pts = [(n.x, n.y) for n in nodes]
    if boxes_wh:
        bw, bh = boxes_wh
        for n in nodes:
            pts.append((n.x - bw / 2, n.y - bh / 2))
            pts.append((n.x + bw / 2, n.y + bh / 2))
    tf = _fit(pts, W, H, PAD)
    parts = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    parts.append(f'<rect width="{W}" height="{H}" fill="#fbfbfd"/>')
    for u, v in edges:
        x1, y1 = tf(nodes[u].x, nodes[u].y)
        x2, y2 = tf(nodes[v].x, nodes[v].y)
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#bbb" stroke-width="1.4"/>'
        )
    for i, n in enumerate(nodes):
        cx, cy = tf(n.x, n.y)
        if boxes_wh:
            bw, bh = boxes_wh
            sx = tf(n.x + bw / 2, n.y)[0] - cx
            sy = tf(n.x, n.y + bh / 2)[1] - cy
            parts.append(
                f'<rect x="{cx - sx:.1f}" y="{cy - sy:.1f}" width="{2 * sx:.1f}" '
                f'height="{2 * sy:.1f}" rx="2" fill="#7bb37b" stroke="#3f7a3f" '
                f'stroke-width="1.5"/>'
            )
        else:
            parts.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="8" fill="#4a90d9" '
                f'stroke="#2c5aa0" stroke-width="1.5"/>'
            )
        parts.append(
            f'<text x="{cx:.1f}" y="{cy + 3:.1f}" text-anchor="middle" '
            f'font-size="9" fill="#fff" font-family="sans-serif">{i}</text>'
        )
    parts.append("</svg>")
    return _card(title, subtitle, "".join(parts))


def _card(title: str, subtitle: str, svg: str) -> str:
    return (
        '<div class="card">'
        f'<div class="ct">{escape(title)}</div>'
        f'<div class="cs">{escape(subtitle)}</div>'
        f"{svg}</div>"
    )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def _nodes_of(layout):
    """Cola exposes nodes via a method (fluent API); base layouts via a property."""
    ns = layout.nodes
    return ns() if callable(ns) else ns


def _total_bends(layout) -> int:
    return sum(len(e.bends) for e in layout.orthogonal_edges)


def _min_node_gap(layout) -> float:
    ns = _nodes_of(layout)
    best = float("inf")
    for i in range(len(ns)):
        for j in range(i + 1, len(ns)):
            d = ((ns[i].x - ns[j].x) ** 2 + (ns[i].y - ns[j].y) ** 2) ** 0.5
            best = min(best, d)
    return best


# --------------------------------------------------------------------------- #
# Demo graphs
# --------------------------------------------------------------------------- #

GRAPHS = {
    "K4": (4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),
    "Cube": (
        8,
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ],
    ),
    "Wheel-5": (5, [(1, 2), (2, 3), (3, 4), (4, 1), (0, 1), (0, 2), (0, 3), (0, 4)]),
    "Prism": (6, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)]),
    "3x3 grid": (
        9,
        [
            (0, 1),
            (1, 2),
            (3, 4),
            (4, 5),
            (6, 7),
            (7, 8),
            (0, 3),
            (3, 6),
            (1, 4),
            (4, 7),
            (2, 5),
            (5, 8),
        ],
    ),
}


def _giotto(n, edges, bend_optimal):
    layout = GIOTTOLayout(
        nodes=[{} for _ in range(n)],
        links=[{"source": u, "target": v} for u, v in edges],
        size=(600, 500),
        strict=False,
        bend_optimal=bend_optimal,
    )
    layout.run()
    return layout


def _section_giotto() -> str:
    rows = []
    for name, (n, edges) in GRAPHS.items():
        heur = _giotto(n, edges, False)
        opt = _giotto(n, edges, True)
        rows.append(
            '<div class="pair">'
            + _orthogonal_svg(
                heur, f"{name} - heuristic router", f"{_total_bends(heur)} bends (default)"
            )
            + _orthogonal_svg(
                opt, f"{name} - bend_optimal", f"{_total_bends(opt)} bends (Topology-Shape-Metrics)"
            )
            + "</div>"
        )
    return _block(
        "GIOTTO: bend-optimal drawing now reaches the output",
        "The orthogonalization computes a bend-minimal representation. Before, it "
        "was discarded and edges were routed heuristically. With bend_optimal=True "
        "the representation drives the drawing (falls back safely when out of the "
        "biconnected / max-degree-4 domain).",
        "".join(rows),
    )


def _section_cola() -> str:
    # Overlap avoidance: two big boxes starting on top of each other.
    def cola_overlap(avoid):
        layout = ColaLayout()
        layout.nodes(
            [
                Node(x=0, y=0, width=60, height=60),
                Node(x=8, y=8, width=60, height=60),
                Node(x=120, y=40, width=60, height=60),
            ]
        )
        layout.links([Link(0, 1), Link(1, 2)])
        layout.avoid_overlaps(avoid)
        layout.start(10, 10, 30, 0, False)
        return layout

    off = cola_overlap(False)
    on = cola_overlap(True)
    cards = (
        '<div class="pair">'
        + _node_svg(
            off,
            [(0, 1), (1, 2)],
            "avoid_overlaps=False",
            f"min node gap {_min_node_gap(off):.0f}",
            boxes_wh=(60, 60),
        )
        + _node_svg(
            on,
            [(0, 1), (1, 2)],
            "avoid_overlaps=True",
            f"min node gap {_min_node_gap(on):.0f} (was an inert stub)",
            boxes_wh=(60, 60),
        )
        + "</div>"
    )

    # Separation constraint: node 1 forced >= 120 to the right of node 0.
    sep = ColaLayout()
    sep.nodes([Node(x=0, y=0), Node(x=10, y=0), Node(x=20, y=40)])
    sep.links([Link(0, 1), Link(1, 2)])
    sep.constraints([{"axis": "x", "left": 0, "right": 1, "gap": 120}])
    sep.start(10, 30, 0, 0, False)
    gap = sep.nodes()[1].x - sep.nodes()[0].x
    cards += (
        '<div class="pair">'
        + _node_svg(
            sep,
            [(0, 1), (1, 2)],
            "separation constraint x(1) - x(0) >= 120",
            f"achieved gap {gap:.0f}",
        )
        + "</div>"
    )
    return _block(
        "Cola: constraints and overlap avoidance now actually run (C1)",
        "Layout's VPSC projection was an empty stub, so separation/alignment "
        "constraints and avoid_overlaps had no effect. They are now enforced.",
        cards,
    )


def _section_cyclic() -> str:
    # Sugiyama on a cyclic graph (previously warned + mislayered; GIOTTO crashed).
    n = 6
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)]  # cycle + chord
    sug = SugiyamaLayout(
        nodes=[{} for _ in range(n)],
        links=[{"source": u, "target": v} for u, v in edges],
        size=(600, 500),
    )
    sug.run()
    giotto = _giotto(n, edges, True)
    cards = (
        '<div class="pair">'
        + _node_svg(
            sug,
            edges,
            "Sugiyama on a cyclic graph (H3/H4)",
            "cycle removal + dummy nodes; no crash/warning",
        )
        + _orthogonal_svg(
            giotto, "GIOTTO on a cyclic graph", "recursion fix: cycles no longer hang the layout"
        )
        + "</div>"
    )
    return _block(
        "Cyclic input no longer breaks the hierarchical / orthogonal layouts",
        "Sugiyama now runs cycle removal and inserts dummy nodes for long edges; "
        "GIOTTO's layer assignment no longer infinite-recurses on cycles.",
        cards,
    )


def _block(title: str, desc: str, body: str) -> str:
    return (
        f'<section><h2>{escape(title)}</h2><p class="desc">{escape(desc)}</p>'
        f'<div class="grid">{body}</div></section>'
    )


def main() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    sections = _section_giotto() + _section_cola() + _section_cyclic()
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>graph-layout: review improvements</title>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 0; background: #f2f2f5; color: #222; }}
  header {{ background: #2c3e50; color: #fff; padding: 20px 32px; }}
  header h1 {{ margin: 0 0 4px; font-size: 20px; }}
  header p {{ margin: 0; opacity: .8; font-size: 13px; }}
  section {{ margin: 24px 32px; }}
  h2 {{ font-size: 16px; border-bottom: 2px solid #d0d0d8; padding-bottom: 6px; }}
  .desc {{ font-size: 13px; color: #555; max-width: 900px; }}
  .grid {{ display: flex; flex-direction: column; gap: 14px; }}
  .pair {{ display: flex; flex-wrap: wrap; gap: 14px; }}
  .card {{ background: #fff; border: 1px solid #ddd; border-radius: 8px;
           padding: 8px; box-shadow: 0 1px 2px rgba(0,0,0,.05); }}
  .ct {{ font-size: 13px; font-weight: 600; }}
  .cs {{ font-size: 11px; color: #777; margin-bottom: 4px; }}
</style></head><body>
<header><h1>graph-layout &mdash; review improvements</h1>
<p>Visual evidence for the correctness fixes and the new GIOTTO bend-optimal drawing.</p></header>
{sections}
</body></html>"""
    out = BUILD_DIR / "improvements_showcase.html"
    out.write_text(html)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
