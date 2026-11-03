"""
word_level.py
=============
Word-level diachronic change. For each word (lemma, pos) we treat its
per-bin profile vectors as a little cloud of points in feature space and ask:
how far did the cloud travel, relative to how much it naturally wobbles?

  displacement (D) = distance between the average of the EARLY-half points
                     and the average of the LATE-half points  (travel)
  jitter       (J) = average distance of each point to the cloud's centre
                     (wobble / noise)
  score        = (D / J) * sqrt(n)        with n = number of bins the word
                                          appears in

D/J is "travel measured in units of the word's own wobble", so a noisy word
that bounces in place scores low. sqrt(n) trusts a word seen in many bins
more than one seen in few, without letting ubiquity dominate.

For the top movers we also produce: the features that changed most (net,
signed), a PCA cloud of the yearly vectors, and the neighbourhood stacked
area chart.

Run after Pass 1. Set OUT_DIR in diachronic_common.py first.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import diachronic_common as common

MIN_BINS = 9        # a word must appear in at least this many bins (use 3 for decades)
TOP_K = 15          # how many discovered movers to make detail plots for
TOP_FEATURES = 15   # features listed per word
CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}

def load_all_bins(out_dir, bins):
    """Load every bin's profile DataFrame once, into a dict bin -> DataFrame."""
    all_dfs = {}
    for b in bins:
        all_dfs[b] = common.load_bin_profiles(out_dir, b)
    return all_dfs


def gather_unit_vectors(all_dfs, bins_present, lemma, pos):
    """
    Collect a word's profile across the bins it appears in, aligned to the
    union of all features it ever uses (missing features filled with 0).

    Returns: years (list), feature names (list), matrix (years x features),
             weights (token count per year).
    """
    vectors = {}
    weights = {}
    for b in bins_present:
        row = common.get_unit_row(all_dfs[b], lemma, pos)
        if row is not None:
            feats = common.feature_columns(all_dfs[b])
            vectors[b] = row[feats]            # a Series indexed by feature
            weights[b] = float(row["count"])

    years = sorted(vectors.keys())
    all_feats = set()
    for b in years:
        all_feats.update(vectors[b].index)
    all_feats = sorted(all_feats)

    matrix = np.zeros((len(years), len(all_feats)))
    weight_vector = np.zeros(len(years))
    for r, b in enumerate(years):
        aligned = vectors[b].reindex(all_feats, fill_value=0.0)
        matrix[r, :] = aligned.values
        weight_vector[r] = weights[b]
    return years, all_feats, matrix, weight_vector


# def displacement_jitter_score(matrix, weights):
#     """Compute D, J, score, and the early/late centroids for one word."""
#     n = len(matrix)
#     half = n // 2

#     early_centroid = common.weighted_mean(matrix[:half] if half > 0 else matrix,
#                                           weights[:half] if half > 0 else weights)
#     late_centroid = common.weighted_mean(matrix[half:], weights[half:])
#     overall_centroid = common.weighted_mean(matrix, weights)

#     displacement = common.euclidean(early_centroid, late_centroid)

#     distances = []
#     for r in range(n):
#         distances.append(common.euclidean(matrix[r], overall_centroid))
#     jitter = float(np.mean(distances))

#     score = (displacement / (jitter + 1e-9)) * np.sqrt(n)
#     return displacement, jitter, score, early_centroid, late_centroid

def movement_measures(years, matrix, weights):
    """All movement numbers for one word's cloud of yearly vectors."""
    n = len(matrix)
    half = n // 2

    early_c = common.weighted_mean(matrix[:half] if half > 0 else matrix,
                                   weights[:half] if half > 0 else weights)
    late_c  = common.weighted_mean(matrix[half:], weights[half:])
    overall_c = common.weighted_mean(matrix, weights)

    displacement = common.euclidean(early_c, late_c)
    jitter = float(np.mean([common.euclidean(matrix[r], overall_c) for r in range(n)]))

    # path length: sum of step distances in chronological order
    path_length = 0.0
    for r in range(1, n):
        path_length += common.euclidean(matrix[r - 1], matrix[r])

    # max distance: the two most separated yearly points (cloud diameter)
    max_dist = 0.0
    for a in range(n):
        for b in range(a + 1, n):
            d = common.euclidean(matrix[a], matrix[b])
            if d > max_dist:
                max_dist = d

    score = (displacement / (jitter + 1e-9)) * np.sqrt(n)
    directness = displacement / (path_length + 1e-9)   # ~1 = straight march
    return {"displacement": displacement, "jitter": jitter,
            "path_length": path_length, "max_dist": max_dist,
            "directness": directness, "score": score,
            "early_c": early_c, "late_c": late_c}

def rank_movers(all_dfs, presence, bins):
    """
    Score every CONTENT word present in at least MIN_BINS bins.
    Ranked by DISPLACEMENT (net travel); jitter and score are still reported
    so you can see whether a high displacement is backed by a tight cloud.
    """
    rows = []
    for (lemma, pos), bins_present in presence.items():
        if pos not in CONTENT_POS:
            continue
        if len(bins_present) < MIN_BINS:
            continue
        years, feats, matrix, weights = gather_unit_vectors(
            all_dfs, bins_present, lemma, pos)
        m = movement_measures(years, matrix, weights)
        rows.append({"lemma": lemma, "pos": pos, "n_bins": len(years),
                     "first": years[0], "last": years[-1],
                     "displacement": round(m["displacement"], 4),
                     "jitter": round(m["jitter"], 4),
                     "path_length": round(m["path_length"], 4),
                     "max_dist": round(m["max_dist"], 4),
                     "directness": round(m["directness"], 4),
                     "score": round(m["score"], 4)})
    table = pd.DataFrame(rows).sort_values("displacement", ascending=False)
    return table.reset_index(drop=True)


def feature_change(all_dfs, presence, lemma, pos, top_n=TOP_FEATURES):
    """The features that moved most for one word (late centroid - early centroid)."""
    years, feats, matrix, weights = gather_unit_vectors(
        all_dfs, presence[(lemma, pos)], lemma, pos)
    dico = movement_measures(years, matrix, weights)
    delta = dico["late_c"] - dico["early_c"]
    table = pd.DataFrame({"feature": feats,
                          "first": dico["early_c"], "last": dico["late_c"], "delta": delta})
    table["abs_delta"] = table["delta"].abs()
    table = table.sort_values("abs_delta", ascending=False).head(top_n)
    return table.reset_index(drop=True)


def pca_2d(matrix):
    """Project rows onto their first two principal components (plain numpy)."""
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    # SVD gives the principal directions; U*S are the coordinates
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    coords = U[:, :2] * S[:2]
    return coords


def plot_pca_cloud(all_dfs, presence, lemma, pos, save):
    years, feats, matrix, weights = gather_unit_vectors(
        all_dfs, presence[(lemma, pos)], lemma, pos)
    coords = pca_2d(matrix)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=years, cmap="viridis", s=80)
    for i, year in enumerate(years):
        ax.annotate(str(year), (coords[i, 0], coords[i, 1]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    fig.colorbar(sc, label="Bin")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(f"Profile cloud over time: {lemma} / {pos}\n"
                 "(consistent smear = drift; shapeless scatter = noise)")
    fig.tight_layout()
    fig.savefig(save, dpi=150)
    plt.show()
    plt.close()


def plot_neighbourhood_area(out_dir, bins, lemma, pos, save, min_share=0.03):
    rows = []          # one dict per present bin
    present_bins = []
    for b in bins:
        dist_df = common.load_distribution(out_dir, b)
        r = common.get_unit_row(dist_df, lemma, pos)
        if r is None:
            continue
        present_bins.append(b)
        shares = {c[2:]: float(r[c]) for c in dist_df.columns
                  if c.startswith("p_") and c not in ("p_SYM", "p_X", "p_PUNCT")}
        rows.append(shares)

    if len(present_bins) < 2:
        print(f"{lemma}/{pos}: not enough bins.")
        return

    # tidy frame: rows = bins, columns = categories, missing = 0
    frame = pd.DataFrame(rows, index=present_bins).fillna(0.0)

    # split into big bands + one "other" band so the stack always fills up
    big = [c for c in frame.columns if frame[c].max() > min_share]
    small = [c for c in frame.columns if c not in big]
    plot_frame = frame[big].copy()
    if small:
        plot_frame["other"] = frame[small].sum(axis=1)

    plt.figure(figsize=(11, 6))
    plt.stackplot(plot_frame.index, plot_frame.T.values,
                  labels=list(plot_frame.columns))
    plt.xlabel("Bin"); plt.ylabel("Neighbourhood share (p_norm)")
    plt.title(f"Neighbourhood composition: {lemma} / {pos}")
    plt.legend(ncol=4, fontsize=8, loc="upper left")
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150)
    plt.show()
    plt.close()

def movers_by_category(movers, top_n=10):
    """Print the top movers within each POS separately."""
    for pos_value, group in movers.groupby("pos"):
        print(f"\n### Top {pos_value} movers (by displacement) ###")
        print(group.sort_values("displacement", ascending=False)
              .head(top_n).to_string(index=False))

if __name__ == "__main__":
    bins = common.list_bins()
    print(f"Bins: {bins}")

    all_dfs = load_all_bins(common.OUT_DIR, bins)
    presence = common.collect_unit_bins(common.OUT_DIR, bins)

    # 1. rank movers and save
    movers = rank_movers(all_dfs, presence, bins)
    movers.to_csv(os.path.join(common.OUT_DIR, "word_movers.csv"), index=False)
    # print("\nTop movers by (displacement / jitter) * sqrt(n):")
    print("\nTop movers by displacement (content words only):")
    print(movers.head(20).to_string(index=False))

    print("\nTop movers by POS category:")
    movers_by_category(movers, top_n=10)
    # 2. detail plots for the top movers plus the watchlist
    detail = list(zip(movers["lemma"].head(TOP_K), movers["pos"].head(TOP_K)))
    for pair in common.WATCHLIST:
        if pair not in detail:
            detail.append(pair)

    for lemma, pos in detail:
        if (lemma, pos) not in presence:
            continue
        if len(presence[(lemma, pos)]) < MIN_BINS:
            continue
        print(f"\n=== {lemma} / {pos}: features that moved most ===")
        print(feature_change(all_dfs, presence, lemma, pos).to_string(index=False))
        plot_pca_cloud(all_dfs, presence, lemma, pos,
                       os.path.join(common.OUT_DIR, f"pca_{lemma}_{pos}.png"))
        plot_neighbourhood_area(common.OUT_DIR, bins, lemma, pos,
                                os.path.join(common.OUT_DIR, f"area_{lemma}_{pos}.png"))