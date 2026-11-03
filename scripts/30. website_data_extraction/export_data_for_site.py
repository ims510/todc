"""
export_data.py
==============
Run this once on your machine to precompute all treebank data and write
it into a directory structure ready to be deployed as a static website.

Output layout
-------------
  site/
    data/
      index.json                   ← global ranking + treebank list
      <Treebank-Name>/
       syntactic/
        stability.json             ← stability matrix
        intruder.json              ← intruder matrix
        bridge_pairs.json          ← bridge words + pulling features
                                     for every pair with overlap > BRIDGE_THRESHOLD
For prosody-capable treebanks (Rhapsodie), use export_prosody.py — it writes
additional space subdirectories alongside syntactic/.
Usage
-----
  python export_data.py

Edit the CONFIG section below before running.
"""

import os
import json
import sys
import traceback
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import entropy as shannon_entropy
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

sys.path.insert(1, "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/tod")
import tod.corpus

# =============================================================================
# CONFIG — edit these before running
# =============================================================================
TREEBANKS_FOLDER   = "/Users/madalina/ud-treebanks-v2.18"
PATTERNS_TEXT_FILE = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/3. probability_matrix/patterns_all_nodes.txt"
GREW_PATTERN       = "pattern{X[upos<>PUNCT]}"
OUTPUT_DIR         = "/Users/madalina/Documents/PHD/code/category_overlap_site/data"          # will be created if it doesn't exist

# One treebank per language. We keep only the *largest* treebank for each
# language whose total token count is at least this threshold. Set
# MIN_TOKENS = 0 to disable the size filter (still one per language).
MIN_TOKENS = 10_000

# Hand-pick the treebank to use for specific languages, overriding the
# "largest one above MIN_TOKENS" rule. Keys are language names as they
# appear after "UD_" (e.g. "French" in "UD_French-GSD"). Values are the
# full directory name to force.
TREEBANK_OVERRIDES = {
    # "French": "UD_French-Sequoia",
}

# Neighbourhood
SIM_THRESHOLD  = 0.7
MAX_NEIGHBOURS = 20

# Tags to remove from matrices entirely
EXCLUDED_TAGS = {"X", "SYM"}

# For bridge word export: only save pairs where intruder overlap > this value
BRIDGE_THRESHOLD = 0.05   # 5 % of the intruder mass

# Per pair, how many bridge words and how many pulling features to store
TOP_BRIDGE_WORDS   = 20
TOP_PULLING_FEATS  = 15

# Feature patterns (same as your existing script)
EXCLUDED_FEATURE_PATTERNS = [r"own"]
INCLUDED_FEATURE_PATTERNS = [
    r"upos=",      r"position=",  r"rel_shallow=",
    r"Aspect=",    r"Animacy=",   r"Case=",
    r"Clusivity=", r"Definite=",  r"Deixis=",     r"DeixisRef",
    r"Evident=",   r"Negation=",  r"Number=",     r"Gender=",
    r"Degree=",    r"ExtPos=",    r"Foreign=",    r"Mood",
    r"NounClass=", r"NumType=",   r"Person=",     r"Polarity=",
    r"Polite=",    r"Poss=",      r"PronType=",   r"Reflex=",
    r"Tense=",     r"Typo=",      r"VerbForm=",   r"Voice=",
]


# =============================================================================
# Core computation helpers (same logic as should_split.py)
# =============================================================================

def _global_category_freqs(corpus):
    all_tags = [lu[1] for lu in corpus._idx2lexunit.values()]
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
    norm = {
        cat: (w / total_weight) / category_freqs[cat]
        for cat, w in raw_weights.items()
    }
    s = sum(norm.values())
    return {cat: v / s for cat, v in norm.items()}


def compute_stability_matrix(corpus, similarity_matrix, category_freqs):
    """Return (stability_df, intruder_df)."""
    all_data = []

    for unit_idx, lex_unit in corpus._idx2lexunit.items():
        current_cat = lex_unit[1]
        sim_row     = similarity_matrix[unit_idx]
        neighbours  = _get_neighbours(unit_idx, sim_row)
        if len(neighbours) == 0:
            continue

        raw_weights  = {}
        total_weight = 0.0
        for n_idx in neighbours:
            n_cat = corpus._idx2lexunit[n_idx][1]
            sim   = float(sim_row[n_idx])
            raw_weights[n_cat] = raw_weights.get(n_cat, 0.0) + sim
            total_weight += sim

        p_norm = _freq_normalised_probs(raw_weights, total_weight, category_freqs)
        for cat, prob in p_norm.items():
            all_data.append({"Original": current_cat, "Neighbour": cat, "Weight": prob})

    df = pd.DataFrame(all_data)
    all_cats = sorted(
        [c for c in (set(df["Original"]) | set(df["Neighbour"])) if c not in EXCLUDED_TAGS]
    )

    matrix = (
        df.groupby(["Original", "Neighbour"])["Weight"]
        .sum()
        .unstack(fill_value=0)
        .reindex(index=all_cats, columns=all_cats, fill_value=0)
    )
    row_sums   = matrix.sum(axis=1).replace(0, 1)
    stability  = matrix.div(row_sums, axis=0)

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


def compute_scores(corpus, similarity_matrix, category_freqs):
    """Return dict {(word, pos): {...}} with per-unit purity/entropy/rival."""
    scores = {}
    for unit_idx, lex_unit in corpus._idx2lexunit.items():
        current_cat = lex_unit[1]
        sim_row     = similarity_matrix[unit_idx]
        neighbours  = _get_neighbours(unit_idx, sim_row)
        if len(neighbours) == 0:
            continue

        raw_weights  = {}
        total_weight = 0.0
        for n_idx in neighbours:
            n_cat = corpus._idx2lexunit[n_idx][1]
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


def get_top_bridge_words(scores, original_cat, rival_cat, top_n=TOP_BRIDGE_WORDS):
    matches = [
        {"word": word, "purity": data["purity"], "entropy": data["entropy"]}
        for (word, o_cat), data in scores.items()
        if o_cat == original_cat and data["rival_category"] == rival_cat
    ]
    matches.sort(key=lambda x: 0.7 * x["purity"] - 0.3 * x["entropy"])
    return matches[:top_n]


def get_pulling_features(lex_unit, rival_cat, corpus, similarity_matrix,
                         top_k=TOP_PULLING_FEATS):
    unit_idx     = corpus._lexunit2idx[lex_unit]
    sim_row      = similarity_matrix[unit_idx]
    unit_vec     = corpus.feature_matrix[unit_idx]
    neighbours   = _get_neighbours(unit_idx, sim_row)
    rival_idxs   = [i for i in neighbours if corpus._idx2lexunit[i][1] == rival_cat]
    if not rival_idxs:
        return []

    centroid      = np.median(corpus.feature_matrix[rival_idxs], axis=0)
    diffs         = np.abs(unit_vec - centroid)
    pulling_score = unit_vec / (diffs + 1e-6)
    top_indices   = np.argsort(pulling_score)[::-1][:top_k]

    results = [
        {
            "feature":    corpus._idx2feature[f_idx],
            "unit_value": round(float(unit_vec[f_idx]), 4),
            "rival_avg":  round(float(centroid[f_idx]), 4),
            "diff":       round(float(diffs[f_idx]), 4),
        }
        for f_idx in top_indices
    ]
    results.sort(key=lambda x: x["unit_value"], reverse=True)
    return results


# =============================================================================
# Per-treebank export
# =============================================================================

def export_treebank(treebank_path, treebank_name):
    """
    Process one treebank and write three JSON files to OUTPUT_DIR/<treebank_name>/.
    Returns the WAS score (float) or None on failure.
    """
    out_dir = os.path.join(OUTPUT_DIR, treebank_name)
    os.makedirs(out_dir, exist_ok=True)

    # --- Load ----------------------------------------------------------------
    corpus = tod.corpus.Corpus(
        treebank_path            = treebank_path,
        grew_pattern             = GREW_PATTERN,
        patterns_text_file       = PATTERNS_TEXT_FILE,
        use_sud                  = True,
        matrix_type              = "coverage",
        excluded_feature_patterns= EXCLUDED_FEATURE_PATTERNS,
        included_feature_patterns= INCLUDED_FEATURE_PATTERNS,
    )
    similarity_matrix = cosine_similarity(corpus.feature_matrix)
    category_freqs    = _global_category_freqs(corpus)

    # --- Matrices ------------------------------------------------------------
    stability, intruder = compute_stability_matrix(corpus, similarity_matrix, category_freqs)
    was                 = weighted_average_stability(stability, category_freqs)

    # Round for compact JSON
    def df_to_json(df):
        return {
            "categories": list(df.index),
            "values":     [[round(v, 4) for v in row] for row in df.values.tolist()],
        }

    with open(os.path.join(out_dir, "stability.json"), "w") as f:
        json.dump(df_to_json(stability), f)

    with open(os.path.join(out_dir, "intruder.json"), "w") as f:
        json.dump(df_to_json(intruder), f)

    # --- Bridge pairs --------------------------------------------------------
    scores      = compute_scores(corpus, similarity_matrix, category_freqs)
    cats        = list(stability.index)
    bridge_data = {}   # "ORIG→RIVAL": [{word, purity, entropy, features: [...]}]

    for orig in cats:
        for rival in cats:
            if orig == rival:
                continue
            # Only export pairs with meaningful intruder overlap
            if rival not in intruder.columns or orig not in intruder.index:
                continue
            if intruder.loc[orig, rival] < BRIDGE_THRESHOLD:
                continue

            bridges = get_top_bridge_words(scores, orig, rival)
            if not bridges:
                continue

            enriched = []
            for entry in bridges:
                lex_unit = (entry["word"], orig)
                if lex_unit not in corpus._lexunit2idx:
                    continue
                feats = get_pulling_features(lex_unit, rival, corpus, similarity_matrix)
                enriched.append({
                    "word":    entry["word"],
                    "purity":  round(entry["purity"],  4),
                    "entropy": round(entry["entropy"], 4),
                    "features": feats,
                })

            if enriched:
                bridge_data[f"{orig}→{rival}"] = enriched

    with open(os.path.join(out_dir, "bridge_pairs.json"), "w") as f:
        json.dump(bridge_data, f, ensure_ascii=False)

    return was


# =============================================================================
# Main loop
# =============================================================================

# =============================================================================
# Treebank selection (one per language, above token threshold)
# =============================================================================

def count_tokens(treebank_path):
    """
    Total syntactic-word count across all .conllu files in a treebank.

    Counts data rows (numeric ID), excluding MWT range rows (e.g. '1-2')
    and ellipsis rows (e.g. '1.1'), matching how UD reports treebank sizes.
    Returns 0 if the directory is unreadable or empty (covers iCloud-evicted
    stubs that error on read).
    """
    total = 0
    try:
        files = [f for f in os.listdir(treebank_path) if f.endswith(".conllu")]
    except OSError:
        return 0
    for fname in files:
        path = os.path.join(treebank_path, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line or line[0] == "#" or line == "\n":
                        continue
                    first = line.split("\t", 1)[0]
                    if first.isdigit():
                        total += 1
        except (OSError, UnicodeDecodeError):
            # Treat unreadable files as zero so the precheck still gives a
            # useful answer for the rest of the treebank.
            pass
    return total


def select_treebanks(folder, min_tokens=MIN_TOKENS, overrides=None):
    """
    Walk `folder`, count tokens for every UD_* subdirectory, and keep one
    treebank per language: the override if present, otherwise the largest
    one whose token count is >= min_tokens.

    Returns a list of (treebank_dir, language, n_tokens), sorted by name.
    """
    overrides = overrides or {}

    candidates = sorted([
        d for d in os.listdir(folder)
        if d.startswith("UD_") and os.path.isdir(os.path.join(folder, d))
    ])

    print(f"Scanning {len(candidates)} treebanks for token counts …")
    by_lang = {}   # language -> (dir, n_tokens)
    for d in tqdm(candidates):
        # "UD_French-GSD" -> "French"
        try:
            language = d[3:].split("-", 1)[0]
        except IndexError:
            continue
        n_tokens = count_tokens(os.path.join(folder, d))
        if language in overrides:
            if d == overrides[language]:
                by_lang[language] = (d, n_tokens)
            continue
        if n_tokens < min_tokens:
            continue
        if language not in by_lang or n_tokens > by_lang[language][1]:
            by_lang[language] = (d, n_tokens)

    selected = sorted(
        [(d, lang, n) for lang, (d, n) in by_lang.items()],
        key=lambda x: x[0],
    )

    # Warn about overrides that didn't resolve (typo, missing folder)
    seen_langs = set(by_lang.keys())
    for lang, want in overrides.items():
        if lang not in seen_langs:
            print(f"  ⚠️  override for {lang!r} → {want!r} not found")

    return selected


# =============================================================================
# Main loop
# =============================================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    selected = select_treebanks(TREEBANKS_FOLDER, MIN_TOKENS, TREEBANK_OVERRIDES)
    print(f"\nSelected {len(selected)} treebanks "
          f"(one per language, ≥{MIN_TOKENS:,} tokens)\n")

    global_index = []     # successfully processed
    failed = []           # name + first line of the error

    for treebank_dir, language, n_tokens in tqdm(selected, desc="Exporting"):
        full_path     = os.path.join(TREEBANKS_FOLDER, treebank_dir)
        treebank_name = "-".join(treebank_dir.split("_")[1:])
        print(f"  {treebank_name}  ({n_tokens:,} tokens) …", end=" ", flush=True)

        try:
            was = export_treebank(full_path, treebank_name)
            stab_path = os.path.join(OUTPUT_DIR, treebank_name, "stability.json")
            with open(stab_path) as f:
                n_cats = len(json.load(f)["categories"])
            global_index.append({
                "name":         treebank_name,
                "language":     language,
                "was":          round(was, 4),
                "n_categories": n_cats,
                "n_tokens":     n_tokens,
            })
            print(f"WAS={was:.3f}  ({n_cats} categories)")
        except Exception as e:
            msg = str(e).strip().splitlines()[0][:200]
            failed.append({"name": treebank_name, "language": language,
                           "tokens": n_tokens, "error": msg})
            print(f"FAILED — {msg}")
            # Full traceback still useful for debugging the first few
            traceback.print_exc()

    # Index file for the website
    global_index.sort(key=lambda x: x["was"], reverse=True)
    index_path = os.path.join(OUTPUT_DIR, "index.json")
    with open(index_path, "w") as f:
        json.dump(global_index, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print(f"Successfully processed: {len(global_index)} / {len(selected)}")
    print(f"Failed:                 {len(failed)}")

    if failed:
        print("\nFailed treebanks:")
        for fb in failed:
            print(f"  • {fb['name']:<35} {fb['error']}")
        failed_path = os.path.join(OUTPUT_DIR, "failed.json")
        with open(failed_path, "w") as f:
            json.dump(failed, f, indent=2)
        print(f"\nDetails written to {failed_path}")

    print(f"\nIndex written to {index_path}")
    print(f"Run 'python -m http.server 8000 --directory site' to preview locally.")