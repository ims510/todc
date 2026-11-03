"""
category_level.py
=================
Category-level diachronic change, built on a stable PANEL of words
(words present in (almost) every bin), so we watch the same words age
rather than the vocabulary turning over.

Outputs:
  1. For each category, the context features whose centroid moved most
     between the first and last bin.
  2. Two heatmaps over category pairs:
       - variation: std of the pair's centroid-similarity across bins
         (darker = that relationship changed most over time);
       - signed change: similarity in the last bin minus the first bin
         (positive = the two categories converged, negative = diverged).

Run after Pass 1. Set OUT_DIR in diachronic_common.py first.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import diachronic_common as common

# A word joins the panel if it appears in at least this fraction of bins.
# 1.0 = present in every bin (strict). If that leaves too few words, the
# script automatically retries at 0.8.
PRESENCE_FRACTION = 1.0
MIN_PANEL_SIZE = 50
TOP_FEATURES = 12


def build_panel(presence, n_bins, fraction):
    """Set of (lemma, pos) present in at least `fraction` of the bins."""
    needed = fraction * n_bins
    panel = set()
    for key, bins_present in presence.items():
        if len(bins_present) >= needed:
            panel.add(key)
    return panel


def category_centroids_per_bin(out_dir, bins, panel):
    """
    For each bin and each category, the token-weighted average profile of
    the panel words in that category.

    Returns a dict:  category -> { bin -> pandas Series (feature -> value) }
    Each Series is in that bin's own feature space.
    """
    centroids = {}
    for b in bins:
        df = common.load_bin_profiles(out_dir, b)

        # keep only panel words
        keys = list(zip(df["lemma"], df["pos"]))
        in_panel = [k in panel for k in keys]
        df_panel = df[in_panel]

        feat_cols = common.feature_columns(df_panel)
        for pos_value, group in df_panel.groupby("pos"):
            values = group[feat_cols].values
            weights = group["count"].values
            centroid = common.weighted_mean(values, weights)
            series = pd.Series(centroid, index=feat_cols)
            centroids.setdefault(pos_value, {})[b] = series
    return centroids


def top_moving_features(centroids, top_n=TOP_FEATURES):
    """For each category, rank features by |last bin - first bin|."""
    report = {}
    for cat, per_bin in centroids.items():
        bins_present = sorted(per_bin.keys())
        if len(bins_present) < 2:
            continue
        first_bin = bins_present[0]
        last_bin = bins_present[-1]

        # align first and last centroid to the union of their features
        all_feats = sorted(set(per_bin[first_bin].index) |
                            set(per_bin[last_bin].index))
        first_vec = per_bin[first_bin].reindex(all_feats, fill_value=0.0)
        last_vec = per_bin[last_bin].reindex(all_feats, fill_value=0.0)

        delta = (last_vec - first_vec)
        table = pd.DataFrame({"feature": all_feats,
                              "first": first_vec.values,
                              "last": last_vec.values,
                              "delta": delta.values})
        table["abs_delta"] = table["delta"].abs()
        table = table.sort_values("abs_delta", ascending=False).head(top_n)
        report[cat] = table.reset_index(drop=True)
    return report


def overlap_matrices(centroids, bins):
    """
    For every pair of categories, the time series of cosine similarity
    between their centroids (computed within each bin). Returns the sorted
    category list plus two square matrices: variation (std) and signed change.
    """
    cats = sorted(centroids.keys())

    # collect similarity per pair per bin
    pair_sims = {}
    for b in bins:
        cats_here = [c for c in cats if b in centroids[c]]
        for x in range(len(cats_here)):
            for y in range(x + 1, len(cats_here)):
                cA = cats_here[x]
                cB = cats_here[y]
                sim = common.cosine(centroids[cA][b].values,
                                    centroids[cB][b].values)
                pair_sims.setdefault((cA, cB), {})[b] = sim

    size = len(cats)
    variation = np.full((size, size), np.nan)
    signed = np.full((size, size), np.nan)
    for (cA, cB), per_bin in pair_sims.items():
        ordered_bins = sorted(per_bin.keys())
        values = [per_bin[b] for b in ordered_bins]
        iA = cats.index(cA)
        iB = cats.index(cB)
        variation[iA, iB] = variation[iB, iA] = np.std(values)
        signed[iA, iB] = signed[iB, iA] = values[-1] - values[0]
    return cats, variation, signed


def draw_heatmap(matrix, labels, title, save):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    # write the number in each cell
    for i in range(len(labels)):
        for j in range(len(labels)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}",
                        ha="center", va="center", fontsize=7, color="white")
    fig.colorbar(im)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save, dpi=150)
    plt.show()
    plt.close()

def shared_feature_changes(centroids, catA, catB, top_n=15):
    """
    Features two categories started or stopped SHARING between the first and
    last bin they co-occur in. 'Shared mass' on a feature = the product of the
    two category centroids there (high only when BOTH occupy it).
      positive delta -> they came to share it more  (pulls them together)
      negative delta -> they share it less          (pushes them apart)
    """
    common_bins = sorted(set(centroids[catA].keys()) & set(centroids[catB].keys()))
    if len(common_bins) < 2:
        return None
    first_bin = common_bins[0]
    last_bin = common_bins[-1]

    def shared_mass(b):
        a = centroids[catA][b]
        c = centroids[catB][b]
        feats = a.index.union(c.index)
        a = a.reindex(feats, fill_value=0.0)
        c = c.reindex(feats, fill_value=0.0)
        return a * c                       # element-wise shared mass per feature

    first_shared = shared_mass(first_bin)
    last_shared = shared_mass(last_bin)
    feats = first_shared.index.union(last_shared.index)
    first_shared = first_shared.reindex(feats, fill_value=0.0)
    last_shared = last_shared.reindex(feats, fill_value=0.0)

    table = pd.DataFrame({"feature": feats,
                          "shared_first": first_shared.values,
                          "shared_last": last_shared.values,
                          "delta": (last_shared - first_shared).values})
    table["abs_delta"] = table["delta"].abs()
    table = table.sort_values("abs_delta", ascending=False).head(top_n)
    return table.reset_index(drop=True)

if __name__ == "__main__":
    bins = common.list_bins()
    n_bins = len(bins)
    print(f"Bins: {bins}")

    presence = common.collect_unit_bins(common.OUT_DIR, bins)

    panel = build_panel(presence, n_bins, PRESENCE_FRACTION)
    if len(panel) < MIN_PANEL_SIZE and PRESENCE_FRACTION == 1.0:
        print(f"Strict panel too small ({len(panel)} words); retrying at 0.8")
        panel = build_panel(presence, n_bins, 0.8)
    print(f"Panel size: {len(panel)} words")

    centroids = category_centroids_per_bin(common.OUT_DIR, bins, panel)

    # 1. top moving features per category
    report = top_moving_features(centroids)
    for cat in sorted(report.keys()):
        print(f"\n=== {cat}: features that moved most (first vs last bin) ===")
        print(report[cat].to_string(index=False))

    # 2. overlap heatmaps
    cats, variation, signed = overlap_matrices(centroids, bins)
    draw_heatmap(variation, cats,
                 "How much each category pair's similarity varies over time",
                 os.path.join(common.OUT_DIR, "overlap_variation.png"))
    draw_heatmap(signed, cats,
                 "Signed change in similarity (last bin - first bin)",
                 os.path.join(common.OUT_DIR, "overlap_signed.png"))
    # 3. for the category pairs that moved most, show which features drove it
    pairs = []
    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            if not np.isnan(signed[i, j]):
                pairs.append((cats[i], cats[j], signed[i, j]))
    pairs.sort(key=lambda triple: abs(triple[2]), reverse=True)  # biggest change first

    for catA, catB, change in pairs[:5]:
        direction = "converged" if change > 0 else "diverged"
        print(f"\n=== {catA} <-> {catB}: {direction} (signed change {change:+.3f}) ===")
        table = shared_feature_changes(centroids, catA, catB)
        if table is not None:
            print(table.to_string(index=False))