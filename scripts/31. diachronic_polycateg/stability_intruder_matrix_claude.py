"""
should_split.py
===============
Polycategoriality analysis of a UD/SUD treebank.

Main outputs
------------
  stability_matrix  - DataFrame (rows/cols = POS tags).
                      The diagonal holds each category's stability score;
                      off-diagonal cells show how much weight leaks to other tags.
  intruder_matrix   - Same shape, diagonal zeroed and rows re-normalised.
                      Shows the *relative* pull of every rival category.

Key functions
-------------
  build_matrices()                  - returns (stability_matrix, intruder_matrix)
  compute_scores()                  - per-lexical-unit purity / entropy / rival info
  get_top_bridge_words()            - bridge words between two categories
  get_pulling_features()            - features pulling one word toward a rival category
  get_aggregate_pulling_features()  - same, aggregated over the top-k bridge words
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.stats import entropy as shannon_entropy
from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.path.insert(1, "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/tod")
import tod.corpus

# ---------------------------------------------------------------------------
# Configuration - edit these paths / patterns before running
# ---------------------------------------------------------------------------
TREEBANK_PATH = "/Users/madalina/Documents/PHD/code/data/test/UD_French-GSD"         # e.g. "UD_French-GSD"
GREW_PATTERN  = "pattern{X[upos<>PUNCT]}"
PATTERNS_TEXT_FILE = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/3. probability_matrix/patterns_all_nodes.txt"
OUTPUT_DIR         = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/29. polycategoriality/"            # where PNGs are saved
# Cosine-similarity neighbourhood parameters
SIM_THRESHOLD = 0.7   # minimum similarity to be considered a neighbour
MAX_NEIGHBOURS = 20   # hard cap on neighbourhood size

# Optionally exclude / include specific feature patterns
EXCLUDED_FEATURE_PATTERNS = [r"own"]
INCLUDED_FEATURE_PATTERNS = [
    r"upos=",      r"position=",  r"rel_shallow=",
    r"Abbr=",      r"Aspect=",    r"Animacy=",    r"Case=",
    r"Clusivity=", r"Definite=",  r"Deixis=",     r"DeixisRef",
    r"Evident=",   r"Negation=",  r"Number=",     r"Gender=",
    r"Degree=",    r"ExtPos=",    r"Foreign=",    r"Mood",
    r"NounClass=", r"NumType=",   r"Person=",     r"Polarity=",
    r"Polite=",    r"Poss=",      r"PronType=",   r"Reflex=",
    r"Tense=",     r"Typo=",      r"VerbForm=",   r"Voice=",
]

# Tags to exclude from the output matrices (e.g. catch-all / noise tags)
EXCLUDED_TAGS = []   # e.g. ["X", "SYM"]

# ===========================================================================
# Step 1 - Load corpus & compute similarity matrix
# ===========================================================================

def load_corpus():
    """Load the treebank and return a tod Corpus object."""
    corpus = tod.corpus.Corpus(
        treebank_path=TREEBANK_PATH,
        grew_pattern=GREW_PATTERN,
        patterns_text_file=PATTERNS_TEXT_FILE,
        use_sud=False,
        matrix_type="coverage",
        excluded_feature_patterns=EXCLUDED_FEATURE_PATTERNS,
        included_feature_patterns=INCLUDED_FEATURE_PATTERNS,
    )
    return corpus


def build_similarity_matrix(corpus):
    """Compute a pairwise cosine-similarity matrix over the feature matrix."""
    return cosine_similarity(corpus.feature_matrix)


# ===========================================================================
# Step 2 - Per-word scores (purity, entropy, rival category)
# ===========================================================================

def _global_category_freqs(corpus):
    """Return a dict {category: relative_frequency} over all lexical units."""
    all_tags = [lu[1] for lu in corpus._idx2lexunit.values()]
    total = len(all_tags)
    return {cat: count / total for cat, count in Counter(all_tags).items()}


def _get_neighbours(unit_idx, similarity_matrix, threshold=SIM_THRESHOLD,
                    max_n=MAX_NEIGHBOURS):
    """Return the indices of the top-k similar neighbours above *threshold*."""
    above = np.where(similarity_matrix[unit_idx] >= threshold)[0]
    return sorted(
        [i for i in above if i != unit_idx],
        key=lambda x: similarity_matrix[unit_idx][x],
        reverse=True,
    )[:max_n]


def _freq_normalised_probs(raw_weights, total_weight, category_freqs):
    """
    Divide each category's relative weight by its global frequency, then
    re-normalise so the resulting distribution sums to 1.
    """
    norm = {
        cat: (w / total_weight) / category_freqs[cat]
        for cat, w in raw_weights.items()
    }
    total_norm = sum(norm.values())
    return {cat: v / total_norm for cat, v in norm.items()}


def compute_scores(corpus, similarity_matrix):
    """
    Compute per-lexical-unit statistics.

    Returns
    -------
    dict  {(word, pos): {"purity", "entropy", "rival_category",
                         "rival_weight_ratio", "distribution"}}

    Notes
    -----
    * purity            - normalised probability mass on the unit's own tag
    * entropy           - Shannon entropy (base 2) of the neighbourhood distribution
    * rival_category    - tag with the highest normalised mass other than own tag
    * rival_weight_ratio - normalised probability of that rival tag
    * distribution      - human-readable string of the full profile (tags with p ≥ 0.05)
    """
    category_freqs = _global_category_freqs(corpus)
    scores = {}

    for idx, lex_unit in corpus._idx2lexunit.items():
        current_cat = lex_unit[1]
        neighbours = _get_neighbours(idx, similarity_matrix)
        if not neighbours:
            continue

        # Weighted categorical counts
        raw_weights = {}
        total_weight = 0.0
        for n_idx in neighbours:
            n_cat = corpus._idx2lexunit[n_idx][1]
            sim   = float(similarity_matrix[idx][n_idx])
            raw_weights[n_cat] = raw_weights.get(n_cat, 0.0) + sim
            total_weight += sim

        # Frequency-normalised probability distribution
        p_norm = _freq_normalised_probs(raw_weights, total_weight, category_freqs)

        purity = p_norm.get(current_cat, 0.0)
        ent    = shannon_entropy(list(p_norm.values()), base=2)

        # Rival: the non-own category with the highest normalised mass
        rival_cat, rival_p = None, 0.0
        for cat, p in p_norm.items():
            if cat != current_cat and p > rival_p:
                rival_cat, rival_p = cat, p

        # Readable distribution string (only categories with p ≥ 0.05)
        sorted_dist = sorted(p_norm.items(), key=lambda x: x[1], reverse=True)
        dist_str = ", ".join(
            f"{cat}: {prob:.2f}" for cat, prob in sorted_dist if prob >= 0.05
        )

        scores[lex_unit] = {
            "purity":             purity,
            "entropy":            ent,
            "rival_category":     rival_cat,
            "rival_weight_ratio": rival_p,
            "distribution":       dist_str,
        }

    return scores


# ===========================================================================
# Step 3 - Stability & Intruder matrices
# ===========================================================================

def build_matrices(corpus, similarity_matrix, excluded_tags=None):
    """
    Build the stability matrix and the intruder matrix.

    Parameters
    ----------
    corpus            : tod Corpus object
    similarity_matrix : pairwise cosine similarity (n_units x n_units)
    excluded_tags     : list of POS tags to drop from the output (default: none)

    Returns
    -------
    stability_matrix : pd.DataFrame  - rows sum to 1; diagonal = stability score
    intruder_matrix  : pd.DataFrame  - diagonal zeroed, rows re-normalised
    """
    category_freqs = _global_category_freqs(corpus)
    excluded_tags  = set(excluded_tags or [])
    all_data = []

    for idx, lex_unit in corpus._idx2lexunit.items():
        current_cat = lex_unit[1]
        neighbours  = _get_neighbours(idx, similarity_matrix)
        if not neighbours:
            continue

        raw_weights  = {}
        total_weight = 0.0
        for n_idx in neighbours:
            n_cat = corpus._idx2lexunit[n_idx][1]
            sim   = float(similarity_matrix[idx][n_idx])
            raw_weights[n_cat] = raw_weights.get(n_cat, 0.0) + sim
            total_weight += sim

        p_norm = _freq_normalised_probs(raw_weights, total_weight, category_freqs)

        for cat, prob in p_norm.items():
            all_data.append({
                "Original": current_cat,
                "Neighbour": cat,
                "Weight": prob,
            })

    df = pd.DataFrame(all_data)

    # Aggregate and force-square the matrix
    all_cats = sorted(set(df["Original"]) | set(df["Neighbour"]))
    matrix   = (
        df.groupby(["Original", "Neighbour"])["Weight"]
        .sum()
        .unstack(fill_value=0)
        .reindex(index=all_cats, columns=all_cats, fill_value=0)
    )

    # Row-normalise → stability matrix
    row_sums = matrix.sum(axis=1).replace(0, 1)
    stability_matrix = matrix.div(row_sums, axis=0)

    # Intruder matrix: zero the diagonal, re-normalise rows
    intruder_raw = stability_matrix.copy()
    np.fill_diagonal(intruder_raw.values, 0.0)
    row_sums_int = intruder_raw.sum(axis=1).replace(0, 1)
    intruder_matrix = intruder_raw.div(row_sums_int, axis=0)

    # Drop unwanted tags
    if excluded_tags:
        keep = [c for c in all_cats if c not in excluded_tags]
        stability_matrix = stability_matrix.loc[keep, keep]
        intruder_matrix  = intruder_matrix.loc[keep, keep]

    return stability_matrix, intruder_matrix


# ===========================================================================
# Step 4 - Bridge-word analysis
# ===========================================================================

def get_top_bridge_words(scores, original_cat, rival_cat, top_n=100):
    """
    Find words from *original_cat* that are most pulled toward *rival_cat*.

    The ranking uses (0.7 x purity - 0.3 x entropy) in ascending order
    so the words with the *lowest* purity and *lowest* entropy come first
    (these are the clearest, most focused bridge words).

    Parameters
    ----------
    scores       : dict returned by compute_scores()
    original_cat : the annotated category to search within (e.g. "ADJ")
    rival_cat    : the competing category (e.g. "VERB")
    top_n        : maximum number of results

    Returns
    -------
    list of dicts with keys: word, purity, entropy
    """
    matches = [
        {
            "word":    word,
            "purity":  data["purity"],
            "entropy": data["entropy"],
        }
        for (word, o_cat), data in scores.items()
        if o_cat == original_cat and data["rival_category"] == rival_cat
    ]

    matches.sort(key=lambda x: 0.7 * x["purity"] - 0.3 * x["entropy"])
    return matches[:top_n]


def get_pulling_features(lex_unit, rival_category, corpus, similarity_matrix,
                         similarity_threshold=SIM_THRESHOLD,
                         n_neighbors=MAX_NEIGHBOURS,
                         top_k_features=20):
    """
    Identify the features that 'pull' a lexical unit toward a rival category.

    The score for each feature is:  unit_value / (|unit - rival_avg| + ε)
    High score → the feature is strongly expressed by the unit AND closely
    matches the rival category's prototype.

    Parameters
    ----------
    lex_unit              : tuple (word, pos)
    rival_category        : the competing POS tag
    corpus                : tod Corpus object
    similarity_matrix     : pairwise cosine similarity matrix
    similarity_threshold  : minimum similarity for a neighbour (default 0.7)
    n_neighbors           : neighbourhood cap (default 20)
    top_k_features        : number of top features to return (default 20)

    Returns
    -------
    list of dicts sorted by unit_value (descending):
        feature, unit_value, rival_avg, difference
    Returns [] if no rival neighbours are found.
    """
    unit_idx       = corpus._lexunit2idx[lex_unit]
    lex_unit_vec   = corpus.feature_matrix[unit_idx]

    # Neighbours of the unit above threshold, capped at n_neighbors
    neighbours = _get_neighbours(unit_idx, similarity_matrix,
                                 threshold=similarity_threshold,
                                 max_n=n_neighbors)

    # Keep only those that belong to the rival category
    rival_idxs = [i for i in neighbours if corpus._idx2lexunit[i][1] == rival_category]

    if not rival_idxs:
        print(f"Warning: no neighbours found in '{rival_category}' "
              f"above threshold {similarity_threshold} for {lex_unit}")
        return []

    # Rival prototype (median for robustness)
    rival_centroid  = np.median(corpus.feature_matrix[rival_idxs], axis=0)
    feature_diffs   = np.abs(lex_unit_vec - rival_centroid)

    # Pulling score: how strongly does this unit share the feature with the rival?
    eps           = 1e-6
    pulling_score = lex_unit_vec / (feature_diffs + eps)
    top_indices   = np.argsort(pulling_score)[::-1][:top_k_features]

    results = [
        {
            "feature":     corpus._idx2feature[f_idx],
            "unit_value":  float(lex_unit_vec[f_idx]),
            "rival_avg":   float(rival_centroid[f_idx]),
            "difference":  float(feature_diffs[f_idx]),
        }
        for f_idx in top_indices
    ]

    results.sort(key=lambda x: x["unit_value"], reverse=True)
    return results


def get_aggregate_pulling_features(scores, original_cat, rival_cat, corpus,
                                   similarity_matrix, top_k_words=10,
                                   top_k_features=20,
                                   similarity_threshold=SIM_THRESHOLD,
                                   n_neighbors=MAX_NEIGHBOURS):
    """
    Aggregate pulling features across the top-k bridge words.

    For each of the top-k bridge words (original_cat → rival_cat),
    collect the features returned by get_pulling_features(), then
    summarise them with mean, median, std, and occurrence count.

    Parameters
    ----------
    scores           : dict returned by compute_scores()
    original_cat     : annotated category (e.g. "PART")
    rival_cat        : rival category    (e.g. "PRON")
    corpus           : tod Corpus object
    similarity_matrix: pairwise cosine similarity matrix
    top_k_words      : number of bridge words to analyse (default 10)
    top_k_features   : number of top features to return  (default 20)

    Returns
    -------
    pd.DataFrame with columns:
        feature, avg_unit_value, median_unit_value, std_unit_value, count_words
    Sorted by avg_unit_value descending.
    Returns None if no bridge words are found.
    """
    bridge_words = get_top_bridge_words(scores, original_cat, rival_cat,
                                        top_n=top_k_words)
    if not bridge_words:
        print(f"No bridge words found from '{original_cat}' to '{rival_cat}'")
        return None

    print(f"Analysing {len(bridge_words)} bridge words: "
          f"{[w['word'] for w in bridge_words]}\n")

    feature_values = {}   # feature_name → [unit_value, ...]

    for entry in bridge_words:
        lex_unit = (entry["word"], original_cat)
        feats    = get_pulling_features(
            lex_unit             = lex_unit,
            rival_category       = rival_cat,
            corpus               = corpus,
            similarity_matrix    = similarity_matrix,
            similarity_threshold = similarity_threshold,
            n_neighbors          = n_neighbors,
            top_k_features       = top_k_features * 2,   # cast wider net before aggregating
        )
        for f in feats:
            feature_values.setdefault(f["feature"], []).append(f["unit_value"])

    rows = [
        {
            "feature":            name,
            "avg_unit_value":     float(np.mean(vals)),
            "median_unit_value":  float(np.median(vals)),
            "std_unit_value":     float(np.std(vals)),
            "count_words":        len(vals),
        }
        for name, vals in feature_values.items()
    ]

    rows.sort(key=lambda x: x["avg_unit_value"], reverse=True)
    return pd.DataFrame(rows[:top_k_features])


# ===========================================================================
# Step 5 - Optional: Weighted Average Stability (single summary scalar)
# ===========================================================================

def weighted_average_stability(stability_matrix, category_freqs):
    """
    Compute the Weighted Average Stability (WAS) of a language.

    Each category's diagonal (stability score) is weighted by its global
    frequency, giving a single scalar in [0, 1].

    Parameters
    ----------
    stability_matrix : pd.DataFrame (output of build_matrices)
    category_freqs   : dict {category: relative_frequency}

    Returns
    -------
    float
    """
    cats   = stability_matrix.index
    diag   = pd.Series(np.diag(stability_matrix), index=cats)
    weights = pd.Series(category_freqs).reindex(cats).fillna(0)
    total   = weights.sum()
    if total == 0:
        return 0.0
    return float((diag * weights).sum() / total)


# ===========================================================================
# Step 6 - Plots
# ===========================================================================

def _heatmap(matrix, title, output_path):
    """Draw a heatmap, save it to *output_path*, and display it."""
    treebank_name = TREEBANK_PATH.rstrip("/").split("/")[-1]
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        matrix,
        annot=True,
        cmap="YlOrRd",
        fmt=".2f",
    )
    plt.title(f"{title} - {treebank_name}")
    plt.ylabel("Original Category (from Treebank)")
    plt.xlabel("Rival Category (The 'Intruder')")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()          # blocks until the window is closed
    plt.close()


def make_plots(stability_matrix, intruder_matrix):
    """
    Save and display the stability and intruder heatmaps.

    The plots are written to OUTPUT_DIR and also shown interactively so
    you can inspect them before entering category pairs in the REPL.
    Close each window to continue.
    """
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    treebank_name = TREEBANK_PATH.rstrip("/").split("/")[-1]

    _heatmap(
        stability_matrix,
        title="Stability Matrix",
        output_path=os.path.join(OUTPUT_DIR, f"{treebank_name}_stability_matrix.png"),
    )
    _heatmap(
        intruder_matrix,
        title="Intruder Matrix",
        output_path=os.path.join(OUTPUT_DIR, f"{treebank_name}_intruder_matrix.png"),
    )


# ===========================================================================
# Step 7 - Interactive exploration loop
# ===========================================================================

_HELP = """
Commands
--------
  ORIG RIVAL          - bridge words for ORIG → RIVAL  (e.g.  ADJ VERB)
  ORIG RIVAL word     - pulling features for a specific word  (e.g.  ADJ VERB censé)
  agg ORIG RIVAL [k]  - aggregate pulling features, optionally top-k words (default 5)
  cats                - list available categories
  quit / exit         - quit
"""

def _available_cats(scores):
    return sorted({lu[1] for lu in scores})


def _run_interactive(scores, corpus, similarity_matrix):
    """
    Read category pairs (and optional words) from stdin and print results.
    Runs until the user types 'quit' or 'exit'.
    """
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.float_format", "{:.3f}".format)

    print(_HELP)
    print("Available categories:", ", ".join(_available_cats(scores)))

    while True:
        try:
            raw = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not raw:
            continue

        parts = raw.split()
        cmd   = parts[0].lower()

        # ── quit ───────────────────────────────────────────────────────────
        if cmd in ("quit", "exit"):
            print("Exiting.")
            break

        # ── cats ───────────────────────────────────────────────────────────
        if cmd == "cats":
            print("Available categories:", ", ".join(_available_cats(scores)))
            continue

        # ── agg ORIG RIVAL [k] ─────────────────────────────────────────────
        if cmd == "agg":
            if len(parts) < 3:
                print("Usage: agg ORIG RIVAL [k]")
                continue
            orig, rival = parts[1], parts[2]
            k = int(parts[3]) if len(parts) >= 4 else 5
            df = get_aggregate_pulling_features(
                scores            = scores,
                original_cat      = orig,
                rival_cat         = rival,
                corpus            = corpus,
                similarity_matrix = similarity_matrix,
                top_k_words       = k,
                top_k_features    = 20,
            )
            if df is not None:
                print(df.to_string(index=False))
            continue

        # ── ORIG RIVAL [word] ──────────────────────────────────────────────
        if len(parts) >= 2:
            orig, rival = parts[0], parts[1]
            cats = _available_cats(scores)

            if orig not in cats:
                print(f"Unknown category '{orig}'. Available: {', '.join(cats)}")
                continue
            if rival not in cats:
                print(f"Unknown category '{rival}'. Available: {', '.join(cats)}")
                continue

            # Optional specific word
            if len(parts) >= 3:
                word = parts[2]
                lex_unit = (word, orig)
                if lex_unit not in corpus._lexunit2idx:
                    print(f"'{word}' not found with tag '{orig}' in the corpus.")
                    continue
                print(f"\n── Pulling features: ('{word}', '{orig}') → '{rival}' ──")
                feats = get_pulling_features(
                    lex_unit          = lex_unit,
                    rival_category    = rival,
                    corpus            = corpus,
                    similarity_matrix = similarity_matrix,
                    top_k_features    = 20,
                )
                if feats:
                    print(pd.DataFrame(feats).to_string(index=False))
            else:
                # No specific word → show bridge words
                print(f"\n── Bridge words: '{orig}' → '{rival}' ──")
                bridges = get_top_bridge_words(scores, orig, rival, top_n=20)
                if bridges:
                    print(pd.DataFrame(bridges).to_string(index=False))
                    print(
                        "\nTip: to inspect a specific word, type:  "
                        f"{orig} {rival} <word>"
                    )
                else:
                    print("No bridge words found for this pair.")
            continue

        print("Unrecognised command. Type 'quit' to exit or see commands above.")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Loading corpus …")
    corpus            = load_corpus()
    similarity_matrix = build_similarity_matrix(corpus)
    category_freqs    = _global_category_freqs(corpus)

    # ------------------------------------------------------------------
    # 2. Build matrices
    # ------------------------------------------------------------------
    print("Building stability & intruder matrices …")
    stability_matrix, intruder_matrix = build_matrices(
        corpus, similarity_matrix, excluded_tags=EXCLUDED_TAGS
    )

    was = weighted_average_stability(stability_matrix, category_freqs)
    print(f"Weighted Average Stability: {was:.4f}")

    # ------------------------------------------------------------------
    # 3. Show plots  (close each window to continue)
    # ------------------------------------------------------------------
    print("\nShowing plots - close each window to continue …")
    make_plots(stability_matrix, intruder_matrix)

    # ------------------------------------------------------------------
    # 4. Compute per-word scores (needed for the interactive loop)
    # ------------------------------------------------------------------
    print("Computing per-word scores …")
    scores = compute_scores(corpus, similarity_matrix)

    # ------------------------------------------------------------------
    # 5. Interactive exploration
    # ------------------------------------------------------------------
    _run_interactive(scores, corpus, similarity_matrix)