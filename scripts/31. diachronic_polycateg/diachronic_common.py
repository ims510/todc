"""
diachronic_common.py
====================
Shared helpers used by both category_level.py and word_level.py.

It reads the per-bin cache produced by Pass 1 (one folder per bin, each
containing profiles.npz and units.csv) and:
  - drops the categories we don't trust (SYM, X, PUNCT), both as units
    (rows) and as context features (columns);
  - hands back simple pandas DataFrames and a couple of distance functions.

Point OUT_DIR at the cache you want to analyse (the decade one or the
yearly one).
"""

import os
import numpy as np
import pandas as pd
import unicodedata

# Point this at the Pass 1 output you want to analyse:
OUT_DIR = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/31. diachronic_polycateg/output"

# Categories we drop everywhere.
DROP_POS = {"SYM", "X", "PUNCT"}

# Words we always want detailed plots for, on top of the discovered movers.
# WATCHLIST = [("cadre", "NOUN"), ("niveau", "NOUN"),
#              ("partir", "VERB"), ("compte", "NOUN")]
WATCHLIST = [("cadre", "NOUN"), ("expliquer", "VERB"), ("souligner", "VERB")]


def list_bins(out_dir=OUT_DIR):
    """Return the sorted list of bin labels (folder names that are numbers)."""
    bins = []
    for name in os.listdir(out_dir):
        full = os.path.join(out_dir, name)
        if name.isdigit() and os.path.isdir(full):
            bins.append(int(name))
    return sorted(bins)


def _is_bad_feature(feature_name):
    """True if a feature describes a SYM / X / PUNCT context token."""
    for bad in ("upos=SYM", "upos=PUNCT", "upos=X"):
        if bad in feature_name:
            return True
    return False

def normalize_lemma(lemma):
    """Strip accents so e.g. 'université' and 'universite' become one word."""
    if not isinstance(lemma, str):
        return lemma
    decomposed = unicodedata.normalize("NFD", lemma)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def load_bin_profiles(out_dir, b, min_pos_share=0.10):
    """
    Load one bin's profiles as a DataFrame.
    Columns: 'lemma', 'pos', 'count', then one column per context feature.

    Cleaning (so every analysis inherits it):
      1. drop SYM / X / PUNCT units and context features;
      2. (above);
      3. strip accents and merge units that become identical
         (token-weighted average of their profiles, counts summed);
      4. drop any (lemma, pos) whose count is below `min_pos_share` of that
         lemma's total count across all POS in the bin (removes tagging
         scatter like Mme/ADP, Mme/VERB sitting next to Mme/PROPN).
    """
    path = os.path.join(out_dir, str(b), "profiles.npz")
    data = np.load(path, allow_pickle=True)

    feature_names = list(data["feature_names"])
    df = pd.DataFrame(data["matrix"], columns=feature_names)
    df.insert(0, "pos", list(data["poss"]))
    df.insert(0, "lemma", list(data["lemmas"]))
    if "counts" in data.files:
        df.insert(2, "count", list(data["counts"]))
    else:
        df.insert(2, "count", 1)

    # 1. drop unwanted unit categories (rows)
    df = df[~df["pos"].isin(DROP_POS)].copy()

    # 2. drop unwanted context-feature columns
    good_feats = [c for c in feature_names if not _is_bad_feature(c)]
    df = df[["lemma", "pos", "count"] + good_feats]

    # 3. strip accents, then merge units that are now identical.
    #    weight each profile by its count, sum within (lemma,pos), divide back
    df["lemma"] = df["lemma"].apply(normalize_lemma)
    df[good_feats] = df[good_feats].multiply(df["count"], axis=0)
    df = df.groupby(["lemma", "pos"], as_index=False).sum()
    df[good_feats] = df[good_feats].divide(df["count"], axis=0)

    # 4. drop POS that are a tiny minority of the lemma in this bin
    lemma_total = df.groupby("lemma")["count"].transform("sum")
    df = df[df["count"] >= min_pos_share * lemma_total].copy()

    return df[["lemma", "pos", "count"] + good_feats]


def feature_columns(df):
    """The feature columns of a profile DataFrame (everything but the labels)."""
    return [c for c in df.columns if c not in ("lemma", "pos", "count")]


def get_unit_row(df, lemma, pos):
    """Return the single row for (lemma, pos), or None if absent."""
    rows = df[(df["lemma"] == lemma) & (df["pos"] == pos)]
    if len(rows) == 0:
        return None
    return rows.iloc[0]


def collect_unit_bins(out_dir, bins):
    """For each (lemma, pos), the sorted list of bins it appears in."""
    presence = {}
    for b in bins:
        df = load_bin_profiles(out_dir, b)
        for lemma, pos in zip(df["lemma"], df["pos"]):
            key = (lemma, pos)
            if key not in presence:
                presence[key] = []
            presence[key].append(b)
    for key in presence:
        presence[key] = sorted(presence[key])
    return presence


def load_distribution(out_dir, b):
    """
    Load one bin's units.csv (purity/entropy + p_<category> shares).
    Accents stripped so lookups match load_bin_profiles. Note: the p_norm
    values were computed in Pass 1 over the unfiltered neighbour pool, so
    this view is illustrative only and is not re-merged after stripping.
    """
    df = pd.read_csv(os.path.join(out_dir, str(b), "units.csv"))
    df = df[~df["pos"].isin(DROP_POS)].copy()
    df["lemma"] = df["lemma"].apply(normalize_lemma)
    df = df.drop_duplicates(subset=["lemma", "pos"], keep="first")
    return df


# --- distance / averaging helpers ------------------------------------------

def cosine(u, v):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def euclidean(u, v):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    return float(np.sqrt(np.sum((u - v) ** 2)))


def weighted_mean(rows, weights):
    """Weighted average of a set of row-vectors. rows: 2D array, weights: 1D."""
    rows = np.asarray(rows, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.sum() == 0:
        weights = np.ones(len(rows))
    return np.average(rows, axis=0, weights=weights)