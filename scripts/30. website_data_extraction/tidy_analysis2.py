"""
tidy_analysis.py
================
Shared analysis functions used by both export_data.py (syntactic-only
treebanks) and export_prosody.py (prosody-capable treebanks).

The functions take generic numpy/dict inputs so they don't depend on any
particular corpus-loading library.
"""

import os
import json
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import entropy as shannon_entropy

# ── Tunables (kept in one place for consistency) ───────────────────────────
SIM_THRESHOLD     = 0.7
MAX_NEIGHBOURS    = 20
EXCLUDED_TAGS     = {"X", "SYM"}
BRIDGE_THRESHOLD  = 0.05
TOP_BRIDGE_WORDS  = 20
TOP_PULLING_FEATS = 15


# ── Core helpers ───────────────────────────────────────────────────────────

def global_category_freqs(idx2lexunit):
    all_tags = [lu[1] for lu in idx2lexunit.values()]
    total = len(all_tags)
    return {cat: count / total for cat, count in Counter(all_tags).items()}


def _get_neighbours(unit_idx, sim_row):
    mask = sim_row >= SIM_THRESHOLD
    mask[unit_idx] = False
    eligible = np.where(mask)[0]
    if len(eligible) == 0:
        return eligible
    if len(eligible) > MAX_NEIGHBOURS:
        top_local = np.argpartition(sim_row[eligible], -MAX_NEIGHBOURS)[-MAX_NEIGHBOURS:]
        return eligible[top_local]
    return eligible


def _freq_normalised_probs(raw_weights, total_weight, category_freqs):
    norm = {cat: (w / total_weight) / category_freqs[cat]
            for cat, w in raw_weights.items()}
    s = sum(norm.values())
    return {cat: v / s for cat, v in norm.items()}


# ── Per-word scores ────────────────────────────────────────────────────────

def compute_scores(idx2lexunit, similarity_matrix, category_freqs):
    scores = {}
    for unit_idx, lex_unit in idx2lexunit.items():
        current_cat = lex_unit[1]
        sim_row     = similarity_matrix[unit_idx]
        neighbours  = _get_neighbours(unit_idx, sim_row)
        if len(neighbours) == 0:
            continue

        raw_weights, total_weight = {}, 0.0
        for n_idx in neighbours:
            n_cat = idx2lexunit[n_idx][1]
            sim   = float(sim_row[n_idx])
            raw_weights[n_cat] = raw_weights.get(n_cat, 0.0) + sim
            total_weight += sim

        p_norm = _freq_normalised_probs(raw_weights, total_weight, category_freqs)
        purity = p_norm.get(current_cat, 0.0)
        ent    = float(shannon_entropy(list(p_norm.values()), base=2))

        rival_cat, rival_p = None, 0.0
        for cat, p in p_norm.items():
            if cat != current_cat and p > rival_p:
                rival_cat, rival_p = cat, p

        scores[lex_unit] = {
            "purity":         purity,
            "entropy":        ent,
            "rival_category": rival_cat,
            "rival_p":        float(rival_p),
        }
    return scores


# ── Stability / Intruder matrices ──────────────────────────────────────────

def compute_stability_intruder_matrices(idx2lexunit, similarity_matrix, category_freqs):
    rows = []
    for unit_idx, lex_unit in idx2lexunit.items():
        current_cat = lex_unit[1]
        sim_row     = similarity_matrix[unit_idx]
        neighbours  = _get_neighbours(unit_idx, sim_row)
        if len(neighbours) == 0:
            continue

        raw_weights, total_weight = {}, 0.0
        for n_idx in neighbours:
            n_cat = idx2lexunit[n_idx][1]
            sim   = float(sim_row[n_idx])
            raw_weights[n_cat] = raw_weights.get(n_cat, 0.0) + sim
            total_weight += sim

        p_norm = _freq_normalised_probs(raw_weights, total_weight, category_freqs)
        for cat, prob in p_norm.items():
            rows.append({"Original": current_cat, "Neighbour": cat, "Weight": prob})

    df       = pd.DataFrame(rows)
    all_cats = sorted([
        c for c in (set(df["Original"]) | set(df["Neighbour"]))
        if c not in EXCLUDED_TAGS
    ])
    matrix = (df.groupby(["Original", "Neighbour"])["Weight"]
                .sum().unstack(fill_value=0)
                .reindex(index=all_cats, columns=all_cats, fill_value=0))

    row_sums  = matrix.sum(axis=1).replace(0, 1)
    stability = matrix.div(row_sums, axis=0)

    intruder_raw = stability.copy()
    np.fill_diagonal(intruder_raw.values, 0.0)
    row_sums_i = intruder_raw.sum(axis=1).replace(0, 1)
    intruder   = intruder_raw.div(row_sums_i, axis=0)

    return stability, intruder


def weighted_average_stability(stability, category_freqs):
    cats    = stability.index
    diag    = pd.Series(np.diag(stability), index=cats)
    weights = pd.Series(category_freqs).reindex(cats).fillna(0)
    total   = weights.sum()
    return float((diag * weights).sum() / total) if total > 0 else 0.0


# ── Bridge words & pulling features ─────────────────────────────────────────

def get_top_bridge_words(scores, original_cat, rival_cat, top_n=TOP_BRIDGE_WORDS):
    matches = [
        {"word": word, "purity": d["purity"], "entropy": d["entropy"]}
        for (word, o_cat), d in scores.items()
        if o_cat == original_cat and d["rival_category"] == rival_cat
    ]
    matches.sort(key=lambda x: 0.7 * x["purity"] - 0.3 * x["entropy"])
    return matches[:top_n]


def get_pulling_features(lex_unit, rival_cat,
                         feature_matrix, similarity_matrix,
                         idx2lexunit, lexunit2idx, idx2feature, feature_mask=None,
                         top_k=TOP_PULLING_FEATS):
    unit_idx     = lexunit2idx[lex_unit]
    sim_row      = similarity_matrix[unit_idx]
    unit_vec     = feature_matrix[unit_idx]
    neighbours   = _get_neighbours(unit_idx, sim_row)
    rival_idxs   = [i for i in neighbours if idx2lexunit[i][1] == rival_cat]
    if not rival_idxs:
        return []

    centroid      = np.median(feature_matrix[rival_idxs], axis=0)
    diffs         = np.abs(unit_vec - centroid)
    # Features the unit doesn't use at all can't be pulling anything.
    pulling_score = np.where(unit_vec > 0, unit_vec / (diffs + 1e-6), 0.0)

    if feature_mask is not None:
        pulling_score = np.where(feature_mask, pulling_score, 0.0)
        
    top_indices   = np.argsort(pulling_score)[::-1][:top_k]
    results = [
        {
            "feature":    idx2feature[f_idx],
            "unit_value": round(float(unit_vec[f_idx]), 4),
            "rival_avg":  round(float(centroid[f_idx]), 4),
            "diff":       round(float(diffs[f_idx]), 4),
        }
        for f_idx in top_indices
        if pulling_score[f_idx] > 0
    ]
    results.sort(key=lambda x: x["unit_value"], reverse=True)
    return results


# ── Main export function ───────────────────────────────────────────────────

def export_space(output_dir,
                 idx2lexunit, idx2feature, lexunit2idx,
                 feature_matrix, similarity_matrix):
    """
    Compute and write the three JSON files (stability, intruder, bridge_pairs)
    for one feature space, into `output_dir`.

    Returns (was_score, n_categories).
    """
    os.makedirs(output_dir, exist_ok=True)
    category_freqs = global_category_freqs(idx2lexunit)

    stability, intruder = compute_stability_intruder_matrices(
        idx2lexunit, similarity_matrix, category_freqs
    )
    was = weighted_average_stability(stability, category_freqs)

    def df_to_json(df):
        return {
            "categories": list(df.index),
            "values":     [[round(v, 4) for v in row] for row in df.values.tolist()],
        }

    with open(os.path.join(output_dir, "stability.json"), "w") as f:
        json.dump(df_to_json(stability), f)
    with open(os.path.join(output_dir, "intruder.json"), "w") as f:
        json.dump(df_to_json(intruder), f)

    # Bridge pairs — precompute words + pulling features for every pair
    # whose intruder overlap exceeds the threshold.
    scores      = compute_scores(idx2lexunit, similarity_matrix, category_freqs)
    cats        = list(stability.index)
    bridge_data = {}

    for orig in cats:
        for rival in cats:
            if orig == rival:
                continue
            if intruder.loc[orig, rival] < BRIDGE_THRESHOLD:
                continue
            bridges = get_top_bridge_words(scores, orig, rival)
            if not bridges:
                continue

            enriched = []
            for entry in bridges:
                lex_unit = (entry["word"], orig)
                if lex_unit not in lexunit2idx:
                    continue
                feats = get_pulling_features(
                    lex_unit, rival,
                    feature_matrix, similarity_matrix,
                    idx2lexunit, lexunit2idx, idx2feature,
                )
                enriched.append({
                    "word":     entry["word"],
                    "purity":   round(entry["purity"],  4),
                    "entropy":  round(entry["entropy"], 4),
                    "features": feats,
                })
            if enriched:
                bridge_data[f"{orig}→{rival}"] = enriched

    with open(os.path.join(output_dir, "bridge_pairs.json"), "w") as f:
        json.dump(bridge_data, f, ensure_ascii=False)

    return round(was, 4), len(cats)


# ── Global index management ────────────────────────────────────────────────

def update_index(index_path, treebank_name, spaces_info, default_space):
    """
    Add or replace this treebank's entry in the global index.json.

    spaces_info : {space_name: {"was": float, "n_categories": int}}
    default_space: which space's WAS is used on the landing-page chart
    """
    index = []
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)

    # Strip any existing entry for this treebank
    index = [e for e in index if e["name"] != treebank_name]

    default = spaces_info[default_space]
    index.append({
        "name":          treebank_name,
        "was":           default["was"],
        "n_categories":  default["n_categories"],
        "spaces":        list(spaces_info.keys()),
        "default_space": default_space,
        "was_by_space":  {s: info["was"] for s, info in spaces_info.items()},
    })
    index.sort(key=lambda x: x["was"], reverse=True)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)