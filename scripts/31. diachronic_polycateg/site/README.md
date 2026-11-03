# Fuzzy Boundaries over Time — website

A static site to explore the diachronic POS-overlap results. Three pages:

- **index.html** — the method, from parsed sentence to feature matrix to movement measures.
- **explorer.html** — every word in a searchable, sortable table; click for feature deltas and plots.
- **categories.html** — the two category-overlap heatmaps.

It is fully static (HTML + CSS + a little JavaScript), so it runs on GitHub Pages with no server.

## 1. Generate the data

The site reads files produced by `build_site.py`, which reuses your existing modules. Make sure these are importable from the same folder:
`diachronic_common.py`, `word_level.py`, `confirm_constructions.py`.

In `build_site.py`, set the two cache paths in `RESOLUTIONS` (your Pass-1 output for year and for decade). Then:

```bash
python build_site.py
```

This writes, for each resolution:

```
site/data/annee/words.json        site/data/decennie/words.json
site/data/annee/plots/*.png        site/data/decennie/plots/*.png
site/data/annee/overlap_*.png      site/data/decennie/overlap_*.png
```

It generates detailed plots only for words above the per-resolution bin threshold
(`DETAIL_MIN`), capped at `MAX_DETAIL`; every word still appears in the table with its scores.

The heatmaps are copied from your `category_level.py` output — run that first so
`overlap_variation.png` and `overlap_signed.png` exist in each cache dir.

## 2. Add the poster image

Save your pipeline poster as `site/assets/pipeline.png`. It appears on the method page.
(If it's missing, the page just hides that slot.)

## 3. Preview locally

GitHub Pages serves over HTTP, and the explorer uses `fetch`, which browsers block on
`file://`. So preview with a tiny local server, not by double-clicking the file:

```bash
cd site
python -m http.server 8000
# open http://localhost:8000
```

## 4. Deploy to GitHub Pages

1. Create a repo and push **the contents of `site/`** to it (so `index.html` is at the repo root).
2. Repo → **Settings → Pages** → Source: *Deploy from a branch* → branch `main`, folder `/ (root)`.
3. Wait a minute; your site is at `https://<username>.github.io/<repo>/`.

If you'd rather keep the Python in the same repo, push everything but set Pages to serve
from a `/docs` folder and move the `site/` contents there, or use a `gh-pages` branch.

## Notes

- The sample `words.json` files let the explorer render before you build; `build_site.py`
  overwrites them with the real data.
- Fonts load from Google Fonts (needs internet); everything else is self-contained.
- The colour code — parent (blue), previous (pink), following (green), child (orange) —
  is the same throughout, including the left edge of each feature row in the detail panel.
