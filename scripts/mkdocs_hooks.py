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

It may also bind `caption`, a string that replaces `title=` in the figure
caption. Use it to report a measured quantity -- crossings, stress, bend count --
that only exists once the layout has run.

Options on the info line:

- `title="..."`   figure caption
- `class="..."`   extra CSS classes on the figure; `inline` flows figures
                  side by side for comparisons
- `source=...`    where the Python source appears: `below` (the default for a
                  figure on its own row), `above`, `details` (collapsed inside
                  the figure, the default for `inline` figures), or `none`
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
from html import escape
from typing import Any, Dict, Iterator, List, Optional, Tuple

FENCE_LANGUAGE = "graph-layout"

# Opening fence: at least three backticks or tildes, then the language name.
_OPEN_RE = re.compile(
    r"^(?P<indent>[ ]{0,3})(?P<fence>`{3,}|~{3,})[ ]*"
    + re.escape(FENCE_LANGUAGE)
    + r"(?P<opts>[^\n`]*)$"
)

# Any opening fence, ours or not. The scan tracks these so a graph-layout block
# quoted inside a longer fence -- how the docs show the syntax itself -- is left
# as text instead of being executed.
_ANY_FENCE_RE = re.compile(r"^(?P<indent>[ ]{0,3})(?P<fence>`{3,}|~{3,})(?P<info>.*)$")

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
        error: Exception = GraphFenceError(message)
    else:
        error = PluginError(message)
    # The class depends on whether mkdocs is installed, so mark the instance
    # instead: render_markdown uses this to add the page path exactly once.
    error._graph_fence = True  # type: ignore[attr-defined]
    return error


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
    options: Dict[str, Any] = {
        "seed": 0,
        # None means "decide from the figure's class": every figure shows its
        # code unless the page opts out with source=none.
        "source": None,
        "title": None,
        "class": None,
    }
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

    options["source"] = _resolve_source(options["source"], options["class"])
    return options, render_kwargs


def _resolve_source(value: Any, css_class: Optional[str]) -> Optional[str]:
    """Decide where a block's source is shown.

    Unset means shown: `below` for a figure that owns its row, and `details` --
    collapsed inside the figure -- for one flowing in an `inline` comparison row,
    where a full-width code block between figures would break the row apart.
    """
    if value is None:
        return "details" if css_class and "inline" in css_class.split() else "below"
    if value is True:
        return "above"
    if value is False or value == "none":
        return None
    if value in ("above", "below", "details"):
        return value
    raise _build_error(f"source must be above, below, details, or none, got {value!r}")


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


def _render_block(
    code: str, options: Dict[str, Any], render_kwargs: Dict[str, Any]
) -> Tuple[str, Optional[str]]:
    """Execute one fence body and return its SVG and any caption it computed."""
    ns = _namespace()
    _seed(options["seed"])
    exec(compile(code, "<graph-layout fence>", "exec"), ns)

    kwargs = dict(_RENDER_DEFAULTS)
    kwargs.update(render_kwargs)

    caption = ns.get("caption")
    if caption is not None and not isinstance(caption, str):
        raise _build_error(f"caption must be a string, got {type(caption).__name__}")

    if isinstance(ns.get("svg"), str):
        return ns["svg"], caption
    if "layout" in ns:
        # Orthogonal layouts override to_svg() to route through to_svg_orthogonal.
        return ns["layout"].to_svg(**kwargs), caption
    if "boxes" in ns and "edges" in ns:
        return ns["to_svg_orthogonal"](ns["boxes"], ns["edges"], **kwargs), caption
    raise _build_error(
        f"{FENCE_LANGUAGE} block must bind `layout`, `svg`, or both `boxes` and `edges`"
    )


def _highlight(code: str) -> str:
    """Render code as HTML. Pygments where available, plain <pre> otherwise.

    The figure is raw HTML, so python-markdown never sees a fenced block inside
    it and cannot highlight one. Pygments is what pymdownx.highlight uses, and
    `highlight` is the class the Material stylesheet targets, so the result
    matches the code blocks elsewhere on the page.
    """
    try:
        from pygments import highlight
        from pygments.formatters import HtmlFormatter
        from pygments.lexers import PythonLexer
    except ImportError:
        return f"<pre><code>{escape(code)}</code></pre>"
    return highlight(code, PythonLexer(), HtmlFormatter(cssclass="highlight", wrapcode=True))


def _figure(
    svg: str,
    title: Optional[str],
    extra_class: Optional[str] = None,
    source: Optional[str] = None,
) -> str:
    """Wrap the SVG in a figure. Raw HTML, so python-markdown passes it through."""
    classes = "graph-layout-figure"
    if extra_class:
        classes += " " + escape(extra_class, quote=True)
    parts = [f'<figure class="{classes}">', _responsive(svg)]
    if title:
        parts.append(f"<figcaption>{escape(title)}</figcaption>")
    if source is not None:
        parts.append(
            '<details class="graph-layout-source">'
            "<summary>Code for this drawing</summary>"
            f"{_highlight(source)}"
            "</details>"
        )
    parts.append("</figure>")
    return "\n".join(parts)


def _closing_line(lines: List[str], start: int, fence: str) -> int:
    """Index of the line closing the fence opened at `start`, or len(lines).

    A fence closes on the same character repeated at least as many times, with
    nothing after it, so a three-backtick block cannot close a four-backtick one.
    """
    close_re = re.compile(r"^[ ]{0,3}" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}[ ]*$")
    for index in range(start + 1, len(lines)):
        if close_re.match(lines[index]):
            return index
    return len(lines)


def iter_fences(markdown: str) -> Iterator[Tuple[int, str]]:
    """Yield (line number, options) for each graph-layout fence that will render.

    Skips blocks quoted inside a longer fence, matching what render_markdown does.
    """
    lines = markdown.split("\n")
    index = 0
    while index < len(lines):
        fence_match = _ANY_FENCE_RE.match(lines[index])
        if fence_match is None:
            index += 1
            continue
        end = _closing_line(lines, index, fence_match.group("fence"))
        ours = _OPEN_RE.match(lines[index])
        if ours is not None:
            yield index + 1, ours.group("opts")
        index = end + 1


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
        fence_match = _ANY_FENCE_RE.match(lines[index])
        if fence_match is None:
            out.append(lines[index])
            index += 1
            continue

        end = _closing_line(lines, index, fence_match.group("fence"))
        ours = _OPEN_RE.match(lines[index])
        if ours is None:
            # Someone else's fence. Copy it through, closing line included, so a
            # graph-layout block quoted inside it is never executed.
            out.extend(lines[index : min(end + 1, len(lines))])
            index = end + 1
            continue
        if end == len(lines):
            raise _build_error(f"{source_path}: unterminated {FENCE_LANGUAGE} block")

        code = "\n".join(lines[index + 1 : end])
        try:
            options, render_kwargs = _parse_options(ours.group("opts"))
            svg, caption = _render_block(code, options, render_kwargs)
        except Exception as exc:  # noqa: BLE001 - re-raised with page context
            if getattr(exc, "_graph_fence", False):
                raise _build_error(f"{source_path}: {exc}") from exc
            raise _build_error(
                f"{source_path}: {FENCE_LANGUAGE} block failed: {type(exc).__name__}: {exc}\n{code}"
            ) from exc

        where = options["source"]
        figure = _figure(
            svg,
            caption or options["title"],
            options["class"],
            code if where == "details" else None,
        )
        block = ["```python", *code.split("\n"), "```"]

        out.append("")
        if where == "above":
            out.extend(block)
            out.append("")
        out.append(figure)
        if where == "below":
            out.append("")
            out.extend(block)
        out.append("")
        index = end + 1

    return "\n".join(out)


def on_page_markdown(markdown: str, page: Any = None, config: Any = None, files: Any = None) -> str:
    """MkDocs hook entry point."""
    path = getattr(getattr(page, "file", None), "src_uri", None) or "<docs>"
    return render_markdown(markdown, source_path=path)
