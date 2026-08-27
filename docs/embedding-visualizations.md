# Embedding Visualizations

Figures in these docs are generated at build time. A fenced block tagged
`graph-layout` is executed against the installed library and replaced with the
resulting inline SVG. The implementation is a single MkDocs hook,
`scripts/mkdocs_hooks.py`, wired up by `hooks:` in `mkdocs.yml`.

Live rendering over checked-in images means an example that no longer runs fails
the build instead of leaving a stale picture in place.

## Writing a block

````markdown
```graph-layout title="8-cycle" source node_radius=14
layout = CircularLayout(
    nodes=[{"index": i} for i in range(8)],
    links=[{"source": i, "target": (i + 1) % 8} for i in range(8)],
    size=(300, 300),
).run()
```
````

produces

```graph-layout title="8-cycle" node_radius=14
layout = CircularLayout(
    nodes=[{"index": i} for i in range(8)],
    links=[{"source": i, "target": (i + 1) % 8} for i in range(8)],
    size=(300, 300),
).run()
```

## What the block must bind

The body runs as an ordinary Python module. Exactly one of the following must
exist when it finishes:

| Binding | Rendered with |
|---------|---------------|
| `layout` | `layout.to_svg(**options)` |
| `boxes` and `edges` | `to_svg_orthogonal(boxes, edges, **options)` |
| `svg` | inlined verbatim, for hand-built SVG |

Everything in `graph_layout.__all__` is already in scope, along with `to_svg`,
`to_svg_orthogonal`, `math`, and `random`. Any other import works normally.

## Options

Options go on the fence's info line and are parsed as Python literals.

| Option | Effect |
|--------|--------|
| `title="..."` | Figure caption |
| `source` | Also show the Python source; `source=above` (the default) or `source=below` |
| `seed=N` | Seeds `random` and `numpy.random` before the block runs (default `0`) |
| anything else | Forwarded to the renderer, e.g. `node_radius=14 show_labels=False padding=20` |

Three renderer defaults differ from the library's: `label_color="currentColor"`
so labels follow the light and dark themes, `font_size=11`, and `padding=24`.
Only options accepted by both `to_svg` and `to_svg_orthogonal` may be defaulted
this way, which is why `node_radius` is set per block rather than globally.

## Reproducibility

Iterative layouts start from random positions. Two controls keep a figure stable
across builds:

- `random_seed=` on the layout constructor, which is what the algorithms
  themselves use.
- `seed=` on the fence, which seeds the global `random` and `numpy.random` state
  for any randomness the block itself introduces.

Deterministic algorithms -- spectral, Schnyder, Reingold-Tilford, circular --
need neither.

## Failures

A block that raises fails the build with the page path, the exception, and the
offending source. There is no fallback to a partial page, so a broken example
blocks the build rather than reaching the published site.

## Building the site

```bash
make docs-serve    # live reload at http://127.0.0.1:8000
make docs          # build into site/
make docs-strict   # build with warnings promoted to errors
make docs-deploy   # strict build, then push to the gh-pages branch
```

Publishing is manual. `.github/workflows/docs.yml` holds the same strict build
for on-demand runs, gated on `workflow_dispatch`.

MkDocs and the Material theme live in the `docs` dependency group, kept out of
the default `dev` group so the test environment stays lean.
