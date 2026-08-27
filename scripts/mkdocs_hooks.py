"""
MkDocs hooks for graph-layout.

Renders ```graph-layout fenced blocks in the documentation into inline SVG by
executing the block against the installed library, so every figure in the docs
is produced by the code shown next to it and cannot drift from the API.

Wire it up in mkdocs.yml:

    hooks:
      - scripts/mkdocs_hooks.py

Fence syntax:

    ```graph-layout title="8-cycle" source node_radius=14
    layout = CircularLayout(
        nodes=[{"index": i} for i in range(8)],
        links=[{"source": i, "target": (i + 1) % 8} for i in range(8)],
        size=(360, 360),
    ).run()
    ```

The block must bind one of:

- `layout` -- any layout object, rendered with its own `.to_svg()`, which
  orthogonal layouts override to draw rectangles and bends
- `boxes` and `edges` -- rendered with `to_svg_orthogonal()`
- `svg` -- a raw SVG string, inlined as-is

Options on the info line:

- `title="..."`   figure caption
- `source`        also show the Python source (`source=above` / `source=below`)
- `seed=N`        RNG seed applied before the block runs (default 0)
- anything else   forwarded to the renderer, e.g. `node_radius=14 padding=20`

The module is importable without mkdocs so the rendering logic can be tested
directly; see tests/test_mkdocs_hooks.py.
"""

from __future__ import annotations

import ast
import math
import random
import re
import shlex
from typing import Any, Dict, List, Optional, Tuple

FENCE_LANGUAGE = "graph-layout"

# Opening fence: at least three backticks or tildes, then the language name.
_OPEN_RE = re.compile(
    r"^(?P<indent>[ ]{0,3})(?P<fence>`{3,}|~{3,})[ ]*"
    + re.escape(FENCE_LANGUAGE)
    + r"(?P<opts>[^\n`]*)$"
)

# Labels default to `currentColor` so they follow the page's text colour in both
# the light and dark themes; the library's own default is opaque black. Only keys
# accepted by both `to_svg` and `to_svg_orthogonal` belong here -- the defaults
# are applied to orthogonal layouts too, which take no `node_radius`.
_RENDER_DEFAULTS: Dict[str, Any] = {
    "label_color": "currentColor",
    "font_size": 11.0,
    "padding": 24.0,
}

_TRUE_FLAGS = {"source"}


class GraphFenceError(Exception):
    """A ```graph-layout block could not be rendered."""


def _build_error(message: str) -> Exception:
    """Raise through mkdocs' PluginError when running under mkdocs."""
    try:
        from mkdocs.exceptions import PluginError
    except ImportError:
        return GraphFenceError(message)
    return PluginError(message)


def _namespace() -> Dict[str, Any]:
    """Names available to a fence body without an explicit import."""
    import graph_layout
    from graph_layout.export import to_svg, to_svg_orthogonal

    ns: Dict[str, Any] = {
        name: getattr(graph_layout, name)
        for name in getattr(graph_layout, "__all__", ())
        if hasattr(graph_layout, name)
    }
    ns.update(
        {
            "__name__": "graph_layout_docs",
            "graph_layout": graph_layout,
            "to_svg": to_svg,
            "to_svg_orthogonal": to_svg_orthogonal,
            "math": math,
            "random": random,
        }
    )
    return ns


def _parse_options(raw: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split a fence info line into fence options and renderer keyword arguments.

    Returns (options, render_kwargs). Values are parsed as Python literals where
    possible, so `show_labels=False` and `padding=20` arrive with the right type.
    """
    options: Dict[str, Any] = {"seed": 0, "source": False, "title": None}
    render_kwargs: Dict[str, Any] = {}

    try:
        tokens = shlex.split(raw.strip())
    except ValueError as exc:
        raise _build_error(f"cannot parse {FENCE_LANGUAGE} options {raw.strip()!r}: {exc}") from exc

    for token in tokens:
        key, sep, value = token.partition("=")
        key = key.strip()
        if not sep:
            if key not in _TRUE_FLAGS:
                raise _build_error(f"unknown {FENCE_LANGUAGE} flag {key!r}")
            options[key] = True
            continue
        parsed: Any
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = value
        if key in options:
            options[key] = parsed
        else:
            render_kwargs[key] = parsed

    if options["source"] not in (False, True, "above", "below"):
        raise _build_error(f"source must be above or below, got {options['source']!r}")
    if options["source"] is True:
        options["source"] = "above"
    return options, render_kwargs


def _seed(value: Any) -> None:
    """Seed both RNGs so force-directed figures are stable across builds."""
    if value is None:
        return
    random.seed(value)
    try:
        import numpy as np
    except ImportError:
        return
    np.random.seed(value)


def _responsive(svg: str) -> str:
    """Let the SVG shrink to the content column while keeping its aspect ratio."""
    if not svg.startswith("<svg"):
        return svg
    return svg.replace(
        "<svg",
        '<svg class="graph-layout-svg" style="max-width:100%;height:auto"',
        1,
    )


def _render_block(code: str, options: Dict[str, Any], render_kwargs: Dict[str, Any]) -> str:
    """Execute one fence body and return its SVG."""
    ns = _namespace()
    _seed(options["seed"])
    exec(compile(code, "<graph-layout fence>", "exec"), ns)

    kwargs = dict(_RENDER_DEFAULTS)
    kwargs.update(render_kwargs)

    if isinstance(ns.get("svg"), str):
        return ns["svg"]
    if "layout" in ns:
        # Orthogonal layouts override to_svg() to route through to_svg_orthogonal.
        return ns["layout"].to_svg(**kwargs)
    if "boxes" in ns and "edges" in ns:
        return ns["to_svg_orthogonal"](ns["boxes"], ns["edges"], **kwargs)
    raise _build_error(
        f"{FENCE_LANGUAGE} block must bind `layout`, `svg`, or both `boxes` and `edges`"
    )


def _figure(svg: str, title: Optional[str]) -> str:
    """Wrap the SVG in a figure. Raw HTML, so python-markdown passes it through."""
    parts = ['<figure class="graph-layout-figure">', _responsive(svg)]
    if title:
        parts.append(f"<figcaption>{title}</figcaption>")
    parts.append("</figure>")
    return "\n".join(parts)


def render_markdown(markdown: str, source_path: str = "<docs>") -> str:
    """Replace every ```graph-layout fence in `markdown` with its rendered figure.

    Args:
        markdown: Page source.
        source_path: Page path, used only in error messages.

    Raises:
        PluginError (or GraphFenceError without mkdocs) if a block fails.
    """
    if FENCE_LANGUAGE not in markdown:
        return markdown

    lines = markdown.split("\n")
    out: List[str] = []
    index = 0
    while index < len(lines):
        match = _OPEN_RE.match(lines[index])
        if match is None:
            out.append(lines[index])
            index += 1
            continue

        fence = match.group("fence")
        close_re = re.compile(r"^[ ]{0,3}" + fence[0] + "{" + str(len(fence)) + r",}[ ]*$")
        end = index + 1
        while end < len(lines) and not close_re.match(lines[end]):
            end += 1
        if end == len(lines):
            raise _build_error(f"{source_path}: unterminated {FENCE_LANGUAGE} block")

        code = "\n".join(lines[index + 1 : end])
        options, render_kwargs = _parse_options(match.group("opts"))
        try:
            svg = _render_block(code, options, render_kwargs)
        except Exception as exc:  # noqa: BLE001 - re-raised with page context
            if isinstance(exc, GraphFenceError):
                raise
            raise _build_error(
                f"{source_path}: {FENCE_LANGUAGE} block failed: {type(exc).__name__}: {exc}\n{code}"
            ) from exc

        figure = _figure(svg, options["title"])
        source = ["```python", *code.split("\n"), "```"] if options["source"] else []

        out.append("")
        if options["source"] == "above":
            out.extend(source)
            out.append("")
        out.append(figure)
        if options["source"] == "below":
            out.append("")
            out.extend(source)
        out.append("")
        index = end + 1

    return "\n".join(out)


def on_page_markdown(markdown: str, page: Any = None, config: Any = None, files: Any = None) -> str:
    """MkDocs hook entry point."""
    path = getattr(getattr(page, "file", None), "src_uri", None) or "<docs>"
    return render_markdown(markdown, source_path=path)
