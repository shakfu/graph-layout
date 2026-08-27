"""Tests for the MkDocs hook that renders ```graph-layout blocks into SVG.

The hook lives in scripts/mkdocs_hooks.py rather than the package, so it is
loaded by path here. It is importable without mkdocs installed; the one test
that needs a real site build skips when mkdocs is absent.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HOOK_PATH = REPO_ROOT / "scripts" / "mkdocs_hooks.py"
DOCS_DIR = REPO_ROOT / "docs"
MKDOCS_YML = REPO_ROOT / "mkdocs.yml"


def _load_hooks():
    spec = importlib.util.spec_from_file_location("mkdocs_hooks_under_test", HOOK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


hooks = _load_hooks()


def fence(body: str, opts: str = "", lang: str = "graph-layout", char: str = "`") -> str:
    """Build a fenced block for the hook to render."""
    bar = char * 3
    return f"{bar}{lang} {opts}\n{body.strip()}\n{bar}\n"


CYCLE = """
layout = CircularLayout(
    nodes=[{"index": i} for i in range(6)],
    links=[{"source": i, "target": (i + 1) % 6} for i in range(6)],
    size=(200, 200),
).run()
"""


class TestPassthrough:
    def test_markdown_without_fences_is_unchanged(self):
        text = "# Title\n\nSome prose.\n\n```python\nprint(1)\n```\n"
        assert hooks.render_markdown(text) == text

    def test_other_languages_are_left_alone(self):
        text = fence("print(1)", lang="python")
        assert hooks.render_markdown(text) == text

    def test_indented_language_name_is_not_a_fence(self):
        # Four spaces makes it an indented code block, not a fence.
        text = "    ```graph-layout\n    layout = None\n    ```\n"
        assert hooks.render_markdown(text) == text


class TestRendering:
    def test_layout_binding_produces_inline_svg_figure(self):
        out = hooks.render_markdown(fence(CYCLE))
        assert '<figure class="graph-layout-figure">' in out
        assert '<svg class="graph-layout-svg"' in out
        assert "</figure>" in out
        assert "graph-layout" not in out.split("<figure")[0]

    def test_svg_is_made_responsive(self):
        out = hooks.render_markdown(fence(CYCLE))
        assert 'style="max-width:100%;height:auto"' in out

    def test_title_becomes_a_figcaption(self):
        out = hooks.render_markdown(fence(CYCLE, 'title="A six cycle"'))
        assert "<figcaption>A six cycle</figcaption>" in out

    def test_no_title_means_no_figcaption(self):
        assert "<figcaption>" not in hooks.render_markdown(fence(CYCLE))

    def test_tilde_fences_are_supported(self):
        out = hooks.render_markdown(fence(CYCLE, char="~"))
        assert '<figure class="graph-layout-figure">' in out

    def test_surrounding_markdown_is_preserved(self):
        out = hooks.render_markdown("# Heading\n\n" + fence(CYCLE) + "\nAfter.\n")
        assert out.startswith("# Heading")
        assert "After." in out

    def test_two_blocks_on_one_page_both_render(self):
        out = hooks.render_markdown(fence(CYCLE) + "\n" + fence(CYCLE))
        assert out.count('<figure class="graph-layout-figure">') == 2

    def test_raw_svg_binding_is_inlined_verbatim(self):
        out = hooks.render_markdown(fence("svg = \"<svg><circle r='1'/></svg>\""))
        assert "<circle r='1'/>" in out

    def test_boxes_and_edges_binding_uses_orthogonal_renderer(self):
        body = """
_layout = KandinskyLayout(
    nodes=[{"index": i} for i in range(4)],
    links=[{"source": i, "target": (i + 1) % 4} for i in range(4)],
    size=(200, 200),
).run()
boxes = _layout.node_boxes
edges = _layout.orthogonal_edges
"""
        out = hooks.render_markdown(fence(body))
        assert "<rect" in out

    def test_orthogonal_layout_dispatches_through_its_own_to_svg(self):
        body = """
layout = KandinskyLayout(
    nodes=[{"index": i} for i in range(4)],
    links=[{"source": i, "target": (i + 1) % 4} for i in range(4)],
    size=(200, 200),
).run()
"""
        # to_svg_orthogonal takes no node_radius, so a rect here proves the
        # shared defaults did not leak an unsupported keyword.
        assert "<rect" in hooks.render_markdown(fence(body))


class TestOptions:
    def test_render_kwargs_reach_the_renderer(self):
        out = hooks.render_markdown(fence(CYCLE, "node_radius=5"))
        assert 'r="5.0"' in out

    def test_boolean_literals_are_parsed(self):
        with_labels = hooks.render_markdown(fence(CYCLE, "show_labels=True"))
        without = hooks.render_markdown(fence(CYCLE, "show_labels=False"))
        assert '<g class="labels">' in with_labels
        assert '<g class="labels">' not in without

    def test_labels_default_to_currentcolor_for_theme_switching(self):
        assert 'fill="currentColor"' in hooks.render_markdown(fence(CYCLE))

    def test_source_shows_the_code_above_the_figure(self):
        out = hooks.render_markdown(fence(CYCLE, "source"))
        assert "```python" in out
        assert out.index("```python") < out.index("<figure")

    def test_source_below_shows_the_code_after_the_figure(self):
        out = hooks.render_markdown(fence(CYCLE, "source=below"))
        assert out.index("<figure") < out.index("```python")

    def test_source_is_omitted_by_default(self):
        assert "```python" not in hooks.render_markdown(fence(CYCLE))

    def test_quoted_titles_with_spaces_survive_parsing(self):
        out = hooks.render_markdown(fence(CYCLE, "title='one two three' node_radius=7"))
        assert "<figcaption>one two three</figcaption>" in out
        assert 'r="7.0"' in out


class TestDeterminism:
    def test_seeded_force_layout_renders_identically_across_builds(self):
        body = """
layout = FruchtermanReingoldLayout(
    nodes=[{"index": i} for i in range(8)],
    links=[{"source": i, "target": (i + 1) % 8} for i in range(8)],
    size=(200, 200),
    random_seed=3,
).run()
"""
        first = hooks.render_markdown(fence(body))
        second = hooks.render_markdown(fence(body))
        assert first == second

    def test_fence_seed_controls_randomness_used_by_the_block(self):
        body = """
import random

layout = CircularLayout(
    nodes=[{"index": i} for i in range(6)],
    links=[{"source": i, "target": random.randrange(6)} for i in range(6)],
    size=(200, 200),
).run()
"""
        assert hooks.render_markdown(fence(body, "seed=1")) == hooks.render_markdown(
            fence(body, "seed=1")
        )
        assert hooks.render_markdown(fence(body, "seed=1")) != hooks.render_markdown(
            fence(body, "seed=2")
        )


class TestFailures:
    def test_block_binding_nothing_is_an_error(self):
        with pytest.raises(Exception, match="must bind"):
            hooks.render_markdown(fence("x = 1"))

    def test_exception_in_block_reports_page_and_source(self):
        with pytest.raises(Exception) as info:
            hooks.render_markdown(fence("raise ValueError('boom')"), source_path="gallery.md")
        message = str(info.value)
        assert "gallery.md" in message
        assert "ValueError: boom" in message
        assert "raise ValueError" in message

    def test_unterminated_fence_is_an_error(self):
        with pytest.raises(Exception, match="unterminated"):
            hooks.render_markdown("```graph-layout\nlayout = None\n")

    def test_unknown_flag_is_an_error(self):
        with pytest.raises(Exception, match="unknown"):
            hooks.render_markdown(fence(CYCLE, "wiggle"))

    def test_bad_source_value_is_an_error(self):
        with pytest.raises(Exception, match="source must be"):
            hooks.render_markdown(fence(CYCLE, "source=sideways"))

    def test_unsupported_render_keyword_is_an_error(self):
        with pytest.raises(Exception, match="TypeError"):
            hooks.render_markdown(fence(CYCLE, "not_a_real_option=1"))


class TestMkDocsEntryPoint:
    def test_on_page_markdown_uses_the_page_path_in_errors(self):
        class File:
            src_uri = "guides/example.md"

        class Page:
            file = File()

        with pytest.raises(Exception, match="guides/example.md"):
            hooks.on_page_markdown(fence("x = 1"), page=Page())

    def test_on_page_markdown_tolerates_a_missing_page(self):
        out = hooks.on_page_markdown(fence(CYCLE))
        assert "<figure" in out


class TestSiteConfiguration:
    def test_hook_is_registered_in_mkdocs_yml(self):
        assert "scripts/mkdocs_hooks.py" in MKDOCS_YML.read_text()

    def test_every_page_in_the_nav_exists(self):
        config = MKDOCS_YML.read_text()
        nav = config.split("nav:", 1)[1].split("\nvalidation:", 1)[0]
        pages = re.findall(r"([\w./-]+\.md)", nav)
        assert pages, "no pages found in the mkdocs nav"
        missing = [page for page in pages if not (DOCS_DIR / page).is_file()]
        assert missing == []

    def test_every_docs_page_renders(self):
        for page in sorted(DOCS_DIR.rglob("*.md")):
            hooks.render_markdown(page.read_text(), source_path=str(page.relative_to(REPO_ROOT)))

    def test_site_builds(self, tmp_path):
        pytest.importorskip("mkdocs", reason="mkdocs is in the optional docs group")
        from mkdocs.commands.build import build
        from mkdocs.config import load_config

        site_dir = tmp_path / "site"
        build(load_config(str(MKDOCS_YML), site_dir=str(site_dir)))
        rendered = (site_dir / "gallery" / "index.html").read_text()
        assert rendered.count('<figure class="graph-layout-figure">') > 1
