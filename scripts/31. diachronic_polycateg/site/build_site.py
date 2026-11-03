"""
build_site.py
=============
Builds the static data the website reads. GitHub Pages cannot run Python, so we
compute everything ONCE here and write plain files the browser can load:

  site/data/<resolution>/words.json      <- table rows + per-word detail
  site/data/<resolution>/plots/*.png     <- PCA, usage, construction images
  site/data/<resolution>/overlap_*.png   <- the two category heatmaps (copied)

Run it for both resolutions (year and decade). It reuses your existing modules
(diachronic_common, word_level, confirm_constructions) so the numbers match
exactly what those scripts produce.

  python build_site.py
"""

import os
import json
import shutil
import matplotlib
matplotlib.use("Agg")          # no display needed; just save files
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import sys
sys.path.insert(1, "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/31. diachronic_polycateg")
import diachronic_common as common
import word_level as wl
import confirm_constructions as cc

# --- where things are ------------------------------------------------------
SITE_DIR = os.path.join(os.path.dirname(__file__), "site")

# one entry per resolution: (label, the Pass-1 cache for that resolution)
RESOLUTIONS = [
    ("annee",    "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/31. diachronic_polycateg/output"),
    ("decennie", "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/31. diachronic_polycateg/output/decennie"),
]

# a word gets full detail (feature deltas + plots) only above this many bins
DETAIL_MIN = {"annee": 9, "decennie": 3}
# cap the number of detailed words (the rest still appear in the table)
MAX_DETAIL = 300

# heatmaps written by category_level.py, copied into each resolution's data dir
HEATMAPS = ["overlap_variation.png", "overlap_signed.png"]


# --- construction sanity checks (only for these specific words) ------------
# raw (accented) lemma is used for matching; the filename key is accent-stripped
# to match the cleaned lemma the explorer shows.

def _is_de(tok):
    return tok is not None and (tok["lemma"].lower() == "de"
                                or tok["form"].lower() in {"de", "du", "des", "d'"})

def m_de_before(sent, i):
    prev = sent[i - 1] if i > 0 else None
    return _is_de(prev)

def m_a_before(sent, i):          # "<noun> à venir": à directly before, nothing after
    prev = sent[i - 1] if i > 0 else None
    return prev is not None and (prev["lemma"].lower() in {"à", "a"}
                                 or prev["form"].lower() in {"à", "a"})

def m_dans_cadre_de(sent, i):
    before = [sent[j]["lemma"].lower() for j in range(max(0, i - 3), i)]
    nxt = sent[i + 1] if i + 1 < len(sent) else None
    return any(l in {"dans", "en"} for l in before) and _is_de(nxt)

CONSTRUCTIONS = [
    ("partenariat", "NOUN", m_de_before,     "X de partenariat"),
    ("coopération", "NOUN", m_de_before,     "X de coopération"),
    ("partir",      "VERB", None,            "à partir de"),     # uses cc.m_a_partir_de
    ("cadre",       "NOUN", m_dans_cadre_de, "dans le cadre de"),
    ("venir",       "VERB", m_a_before,      "à venir"),
]


def feature_slot(feature_name):
    """Which context slot a feature belongs to (drives the colour code)."""
    if ":parent:" in feature_name:
        return "parent"
    if ":child:" in feature_name:
        return "child"
    if ":prev:" in feature_name:
        return "prev"
    if ":next:" in feature_name:
        return "next"
    return "self"


def movement_for(years, matrix, weights):
    """displacement, jitter, path length, max distance, directness, score."""
    n = len(matrix)
    half = n // 2
    early_c = common.weighted_mean(matrix[:half] if half > 0 else matrix,
                                   weights[:half] if half > 0 else weights)
    late_c = common.weighted_mean(matrix[half:], weights[half:])
    overall_c = common.weighted_mean(matrix, weights)

    displacement = common.euclidean(early_c, late_c)
    jitter = sum(common.euclidean(matrix[r], overall_c) for r in range(n)) / n

    path_length = sum(common.euclidean(matrix[r - 1], matrix[r]) for r in range(1, n))

    max_dist = 0.0
    for a in range(n):
        for b in range(a + 1, n):
            d = common.euclidean(matrix[a], matrix[b])
            if d > max_dist:
                max_dist = d

    score = (displacement / (jitter + 1e-9)) * (n ** 0.5)
    directness = displacement / (path_length + 1e-9)
    return dict(displacement=displacement, jitter=jitter, path_length=path_length,
                max_dist=max_dist, directness=directness, score=score)


def plot_usage(bins, occ_by_bin, tokens_by_bin, lemma, pos, save):
    """How often the word occurs over time: raw count and per-1k-tokens."""
    occ = [occ_by_bin.get(b, 0) for b in bins]
    per1k = [(occ_by_bin.get(b, 0) / tokens_by_bin[b] * 1000) if tokens_by_bin.get(b) else 0
             for b in bins]
    fig, ax1 = plt.subplots(figsize=(8, 4.2))
    ax1.bar(bins, occ, color="#c9d3df", label="occurrences")
    ax1.set_ylabel("occurrences")
    ax2 = ax1.twinx()
    ax2.plot(bins, per1k, color="#353f6b", marker="o", ms=3, label="per 1k tokens")
    ax2.set_ylabel("per 1k tokens")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xlabel("bin")
    ax1.set_title(f"Usage over time: {lemma} / {pos}")
    fig.tight_layout()
    fig.savefig(save, dpi=130)
    plt.close(fig)


def plot_construction(df, title, save):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(df["bin"], df["rate"], marker="o", ms=3, color="#e08a3c",
            label="share of the word's uses")
    ax.plot(df["bin"], df["per_1k_tokens"], marker="s", ms=3, color="#353f6b",
            label="per 1k tokens")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("bin"); ax.set_ylabel("frequency")
    ax.set_title(title); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save, dpi=130)
    plt.close(fig)


def build(resolution, cache_dir):
    print(f"\n=== building {resolution} ===")
    data_dir = os.path.join(SITE_DIR, "data", resolution)
    plots_dir = os.path.join(data_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    bins = common.list_bins(cache_dir)
    all_dfs = wl.load_all_bins(cache_dir, bins)
    presence = common.collect_unit_bins(cache_dir, bins)

    # occurrences per (lemma,pos) per bin, and total tokens per bin
    occ_by_word = {}
    for b in bins:
        df = all_dfs[b]
        for lemma, pos, count in zip(df["lemma"], df["pos"], df["count"]):
            occ_by_word.setdefault((lemma, pos), {})[b] = int(count)

    tokens_by_bin = {}
    for label, paths in cc.build_bin_groups(by=resolution):
        if label in bins:
            tokens_by_bin[label] = sum(len(s) for p in paths for s in cc.iter_sentences(p))

    # construction plots (keyed by accent-stripped lemma so they match the table)
    construction_keys = {}
    for raw_lemma, pos, matcher, label in CONSTRUCTIONS:
        m = matcher if matcher is not None else cc.m_a_partir_de
        df = cc.construction_frequency(cc.build_bin_groups(by=resolution), raw_lemma, pos, m)
        key = f"{common.normalize_lemma(raw_lemma)}_{pos}"
        save = os.path.join(plots_dir, f"{key}_construction.png")
        plot_construction(df, f"{raw_lemma} : {label}", save)
        construction_keys[key] = f"data/{resolution}/plots/{key}_construction.png"

    # rank candidates by displacement so MAX_DETAIL keeps the most interesting
    ranked = []
    for (lemma, pos), bins_present in presence.items():
        if len(bins_present) < 2:
            continue
        years, feats, matrix, weights = wl.gather_unit_vectors(all_dfs, bins_present, lemma, pos)
        m = movement_for(years, matrix, weights)
        ranked.append((m["displacement"], lemma, pos, bins_present, m))
    ranked.sort(reverse=True)

    detail_min = DETAIL_MIN[resolution]
    detail_budget = MAX_DETAIL
    words = []
    for displacement, lemma, pos, bins_present, m in ranked:
        n = len(bins_present)
        key = f"{lemma}_{pos}"
        row = {
            "lemma": lemma, "pos": pos, "n_bins": n,
            "first": bins_present[0], "last": bins_present[-1],
            "displacement": round(m["displacement"], 4),
            "jitter": round(m["jitter"], 4),
            "path_length": round(m["path_length"], 4),
            "max_dist": round(m["max_dist"], 4),
            "directness": round(m["directness"], 4),
            "score": round(m["score"], 4),
            "detail": False, "feature_delta": None, "plots": None,
        }

        give_detail = n >= detail_min and detail_budget > 0
        if give_detail:
            detail_budget -= 1
            row["detail"] = True

            fc = wl.feature_change(all_dfs, presence, lemma, pos)
            row["feature_delta"] = [
                {"feature": r["feature"], "slot": feature_slot(r["feature"]),
                 "first": round(r["first"], 4), "last": round(r["last"], 4),
                 "delta": round(r["delta"], 4)}
                for _, r in fc.iterrows()
            ]

            pca_path = os.path.join(plots_dir, f"{key}_pca.png")
            wl.plot_pca_cloud(all_dfs, presence, lemma, pos, pca_path)
            usage_path = os.path.join(plots_dir, f"{key}_usage.png")
            plot_usage(bins, occ_by_word.get((lemma, pos), {}), tokens_by_bin,
                       lemma, pos, usage_path)

            row["plots"] = {
                "pca": f"data/{resolution}/plots/{key}_pca.png",
                "usage": f"data/{resolution}/plots/{key}_usage.png",
            }
            if key in construction_keys:
                row["plots"]["construction"] = construction_keys[key]

        words.append(row)

    # copy the two category heatmaps
    for name in HEATMAPS:
        src = os.path.join(cache_dir, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(data_dir, name))

    out = {"resolution": resolution, "bins": bins, "n_words": len(words), "words": words}
    with open(os.path.join(data_dir, "words.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False)
    print(f"  {len(words)} words written, {MAX_DETAIL - detail_budget} with detail")


if __name__ == "__main__":
    for resolution, cache_dir in RESOLUTIONS:
        build(resolution, cache_dir)
    print("\nDone. Open site/index.html locally, or push the site/ folder to GitHub Pages.")
