"""
hausa_dialect_stability.py
==========================
Polycategoriality analysis of several UD/SUD treebanks of the same language
(here: the four Hausa dialect treebanks), compared in a single figure.

Main outputs
------------
  Per dialect (unchanged from the single-treebank script):
    stability_matrix  - diagonal = stability score, off-diagonal = leakage
    intruder_matrix   - diagonal zeroed, rows re-normalised

  Combined:
    <name>_dialects_stability_matrix.png
    <name>_dialects_intruder_matrix.png
        Each cell is split into four triangles, one per dialect
        (top = Northern, right = Eastern, bottom = Southern, left = Western).
        Colour = value, on one shared scale for all dialects.
        Thickness of the black frame = standard deviation of the four values.
        Grey triangle = no data (tag absent, or too few lexical units).
    <name>_dialects_cells.csv
        One row per (original, rival) pair: the four values for each matrix
        plus their SD, sorted by stability SD (most divergent cells first).
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import os
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.stats import entropy as shannon_entropy
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(1, "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/tod")
import tod.corpus

# ---------------------------------------------------------------------------
# Configuration - edit these paths / patterns before running
# ---------------------------------------------------------------------------

UD_DIR = "/Users/madalina/Documents/PHD/code/data/test"

# One entry per dialect. The key decides where the dialect sits inside each
# cell of the combined matrix: N = top, E = right, S = bottom, W = left.
# Check the folder names against your local copy of UD.
DIALECTS = {
    "N": {"label": "Northern (Ader)",  "path": f"/Users/madalina/Downloads/UD_Hausa-NorthernAutogramm-master"},
    "E": {"label": "Eastern (Kano)",   "path": f"/Users/madalina/Downloads/UD_Hausa-EasternAutogramm-master"},
    "S": {"label": "Southern (Zaria)", "path": f"/Users/madalina/Downloads/UD_Hausa-SouthernAutogramm-master"},
    "W": {"label": "Western (Gobir)",  "path": f"/Users/madalina/Downloads/UD_Hausa-WesternAutogramm-master"},
}
DIALECT_ORDER = ["N", "E", "S", "W"]
LANGUAGE_NAME = "Hausa"            # used in titles and file names

GREW_PATTERN       = "pattern{X[upos<>PUNCT]}"
PATTERNS_TEXT_FILE = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/3. probability_matrix/patterns_all_nodes.txt"
OUTPUT_DIR         = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/29. polycategoriality/33. haussa/"
USE_SUD            = False

# Cosine-similarity neighbourhood parameters
SIM_THRESHOLD  = 0.7   # minimum similarity to be considered a neighbour
MAX_NEIGHBOURS = 20    # hard cap on neighbourhood size

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
EXCLUDED_TAGS = ["X"]   # e.g. ["X", "SYM"]

# Combined-plot parameters
MIN_UNITS_PER_CAT = 3     # a dialect's row is greyed out if fewer lexical units carry the tag
ANNOT_MIN         = 0.01  # values below this are not printed (keeps the figure readable)
MAX_LINEWIDTH     = 4.0   # frame width (pt) given to the cell with the highest SD
MIN_SD_FOR_FRAME  = 0.02  # no frame below this SD
SAVE_SINGLE_DIALECT_PLOTS = True   # also save one ordinary heatmap per dialect
SD_LW_MIN         = 0.6   # frame width (pt) at MIN_SD_FOR_FRAME
SD_LW_MAX         = 2.5   # frame width (pt) at the largest SD
CELL_INCHES       = 0.9   # side of one matrix cell, in inches

# ===========================================================================
# Step 1 - Load corpus & compute similarity matrix
# ===========================================================================

def load_corpus(treebank_path):
    """Load one treebank and return a tod Corpus object."""
    return tod.corpus.Corpus(
        treebank_path=treebank_path,
        grew_pattern=GREW_PATTERN,
        patterns_text_file=PATTERNS_TEXT_FILE,
        # use_sud=USE_SUD,
        matrix_type="coverage",
        min_occurrences=0,
        excluded_feature_patterns=EXCLUDED_FEATURE_PATTERNS,
        included_feature_patterns=INCLUDED_FEATURE_PATTERNS,
    )


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


def _unit_distribution(idx, corpus, similarity_matrix, category_freqs):
    """
    Frequency-normalised category distribution of one unit's neighbourhood,
    or None if the unit has no neighbour above the threshold.
    (Shared by compute_scores and build_matrices.)
    """
    neighbours = _get_neighbours(idx, similarity_matrix)
    if not neighbours:
        return None

    raw_weights, total_weight = {}, 0.0
    for n_idx in neighbours:
        n_cat = corpus._idx2lexunit[n_idx][1]
        sim   = float(similarity_matrix[idx][n_idx])
        raw_weights[n_cat] = raw_weights.get(n_cat, 0.0) + sim
        total_weight += sim

    return _freq_normalised_probs(raw_weights, total_weight, category_freqs)


def compute_scores(corpus, similarity_matrix):
    """
    Compute per-lexical-unit statistics.

    Returns
    -------
    dict  {(word, pos): {"purity", "entropy", "rival_category",
                         "rival_weight_ratio", "distribution"}}
    """
    category_freqs = _global_category_freqs(corpus)
    scores = {}

    for idx, lex_unit in corpus._idx2lexunit.items():
        current_cat = lex_unit[1]
        p_norm = _unit_distribution(idx, corpus, similarity_matrix, category_freqs)
        if p_norm is None:
            continue

        purity = p_norm.get(current_cat, 0.0)
        ent    = shannon_entropy(list(p_norm.values()), base=2)

        rival_cat, rival_p = None, 0.0
        for cat, p in p_norm.items():
            if cat != current_cat and p > rival_p:
                rival_cat, rival_p = cat, p

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
# Step 3 - Stability & Intruder matrices (one treebank)
# ===========================================================================

def build_matrices(corpus, similarity_matrix, excluded_tags=None):
    """
    Build the stability matrix and the intruder matrix for one treebank.

    Returns
    -------
    stability_matrix : pd.DataFrame  - rows sum to 1; diagonal = stability score
    intruder_matrix  : pd.DataFrame  - diagonal zeroed, rows re-normalised
    n_units          : pd.Series     - number of lexical units behind each row
    """
    category_freqs = _global_category_freqs(corpus)
    excluded_tags  = set(excluded_tags or [])
    all_data = []
    unit_counts = Counter()

    for idx, lex_unit in corpus._idx2lexunit.items():
        current_cat = lex_unit[1]
        p_norm = _unit_distribution(idx, corpus, similarity_matrix, category_freqs)
        if p_norm is None:
            continue
        unit_counts[current_cat] += 1
        for cat, prob in p_norm.items():
            all_data.append({"Original": current_cat, "Neighbour": cat, "Weight": prob})

    df = pd.DataFrame(all_data)

    all_cats = sorted(set(df["Original"]) | set(df["Neighbour"]))
    matrix   = (
        df.groupby(["Original", "Neighbour"])["Weight"]
        .sum()
        .unstack(fill_value=0)
        .reindex(index=all_cats, columns=all_cats, fill_value=0)
    )

    row_sums = matrix.sum(axis=1).replace(0, 1)
    stability_matrix = matrix.div(row_sums, axis=0)

    intruder_raw = stability_matrix.copy()
    np.fill_diagonal(intruder_raw.values, 0.0)
    row_sums_int = intruder_raw.sum(axis=1).replace(0, 1)
    intruder_matrix = intruder_raw.div(row_sums_int, axis=0)

    n_units = pd.Series(unit_counts).reindex(all_cats, fill_value=0)

    if excluded_tags:
        keep = [c for c in all_cats if c not in excluded_tags]
        stability_matrix = stability_matrix.loc[keep, keep]
        intruder_matrix  = intruder_matrix.loc[keep, keep]
        n_units          = n_units.loc[keep]

    return stability_matrix, intruder_matrix, n_units


# ===========================================================================
# Step 4 - Bridge-word analysis (unchanged)
# ===========================================================================

def get_top_bridge_words(scores, original_cat, rival_cat, top_n=100):
    """
    Find words from *original_cat* that are most pulled toward *rival_cat*.
    Ranking: ascending (0.7 x purity - 0.3 x entropy).
    """
    matches = [
        {"word": word, "purity": data["purity"], "entropy": data["entropy"]}
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
    Score per feature: unit_value / (|unit - rival_median| + eps)
    """
    unit_idx     = corpus._lexunit2idx[lex_unit]
    lex_unit_vec = corpus.feature_matrix[unit_idx]

    neighbours = _get_neighbours(unit_idx, similarity_matrix,
                                 threshold=similarity_threshold,
                                 max_n=n_neighbors)
    rival_idxs = [i for i in neighbours if corpus._idx2lexunit[i][1] == rival_category]

    if not rival_idxs:
        print(f"Warning: no neighbours found in '{rival_category}' "
              f"above threshold {similarity_threshold} for {lex_unit}")
        return []

    rival_centroid = np.median(corpus.feature_matrix[rival_idxs], axis=0)
    feature_diffs  = np.abs(lex_unit_vec - rival_centroid)

    eps           = 1e-6
    pulling_score = lex_unit_vec / (feature_diffs + eps)
    top_indices   = np.argsort(pulling_score)[::-1][:top_k_features]

    results = [
        {
            "feature":    corpus._idx2feature[f_idx],
            "unit_value": float(lex_unit_vec[f_idx]),
            "rival_avg":  float(rival_centroid[f_idx]),
            "difference": float(feature_diffs[f_idx]),
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
    """Aggregate pulling features across the top-k bridge words."""
    bridge_words = get_top_bridge_words(scores, original_cat, rival_cat,
                                        top_n=top_k_words)
    if not bridge_words:
        print(f"No bridge words found from '{original_cat}' to '{rival_cat}'")
        return None

    print(f"Analysing {len(bridge_words)} bridge words: "
          f"{[w['word'] for w in bridge_words]}\n")

    feature_values = {}
    for entry in bridge_words:
        feats = get_pulling_features(
            lex_unit             = (entry["word"], original_cat),
            rival_category       = rival_cat,
            corpus               = corpus,
            similarity_matrix    = similarity_matrix,
            similarity_threshold = similarity_threshold,
            n_neighbors          = n_neighbors,
            top_k_features       = top_k_features * 2,
        )
        for f in feats:
            feature_values.setdefault(f["feature"], []).append(f["unit_value"])

    rows = [
        {
            "feature":           name,
            "avg_unit_value":    float(np.mean(vals)),
            "median_unit_value": float(np.median(vals)),
            "std_unit_value":    float(np.std(vals)),
            "count_words":       len(vals),
        }
        for name, vals in feature_values.items()
    ]
    rows.sort(key=lambda x: x["avg_unit_value"], reverse=True)
    return pd.DataFrame(rows[:top_k_features])


# ===========================================================================
# Step 5 - Weighted Average Stability (unchanged)
# ===========================================================================

def weighted_average_stability(stability_matrix, category_freqs):
    """Frequency-weighted mean of the diagonal, in [0, 1]."""
    cats    = stability_matrix.index
    diag    = pd.Series(np.diag(stability_matrix), index=cats)
    weights = pd.Series(category_freqs).reindex(cats).fillna(0)
    total   = weights.sum()
    if total == 0:
        return 0.0
    return float((diag * weights).sum() / total)


# ===========================================================================
# Step 6 - Align the dialects
# ===========================================================================

def stack_dialects(matrices, n_units, min_units=MIN_UNITS_PER_CAT):
    """
    Put the per-dialect matrices on a shared tag set.

    Parameters
    ----------
    matrices : {dialect_key: DataFrame}
    n_units  : {dialect_key: Series}   lexical units per row category

    Returns
    -------
    cats : list of tags (union over dialects)
    cube : array (n_dialects, n_cats, n_cats), dialects in DIALECT_ORDER.
           NaN where a tag is absent from a dialect (row or column), and on
           rows backed by fewer than *min_units* lexical units.
    sd   : array (n_cats, n_cats), SD over the dialects that have a value;
           NaN when fewer than two do.
    """
    cats = sorted(set().union(*(m.index for m in matrices.values())))
    cube = np.full((len(DIALECT_ORDER), len(cats), len(cats)), np.nan)

    for d_i, d in enumerate(DIALECT_ORDER):
        m = matrices[d].reindex(index=cats, columns=cats)   # absent tags -> NaN
        too_few = n_units[d].reindex(cats).fillna(0) < min_units
        m.loc[too_few.values, :] = np.nan
        cube[d_i] = m.values

    n_valid = np.sum(~np.isnan(cube), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sd = np.nanstd(cube, axis=0)
    sd[n_valid < 2] = np.nan
    return cats, cube, sd


def export_cell_table(cats, stab_cube, intr_cube, stab_sd, intr_sd, output_path):
    """One row per (original, rival) pair; sorted by stability SD."""
    rows = []
    for i, orig in enumerate(cats):
        for j, rival in enumerate(cats):
            row = {"original": orig, "rival": rival}
            for d_i, d in enumerate(DIALECT_ORDER):
                row[f"stab_{d}"] = stab_cube[d_i, i, j]
            row["stab_sd"] = stab_sd[i, j]
            for d_i, d in enumerate(DIALECT_ORDER):
                row[f"intr_{d}"] = intr_cube[d_i, i, j]
            row["intr_sd"] = intr_sd[i, j]
            rows.append(row)
    df = pd.DataFrame(rows).sort_values("stab_sd", ascending=False, na_position="last")
    df.to_csv(output_path, index=False)
    return df


# ===========================================================================
# Step 7 - Plots
# ===========================================================================

# Text positions inside a cell, kept away from the cell edge so that
# numbers are not covered by the SD frame.
_TEXT_OFFSET = {"N": (0.5, 0.22), "E": (0.75, 0.5), "S": (0.5, 0.78), "W": (0.25, 0.5)}
NO_DATA_COLOUR = "#cfcfcf"
# Font size configuration for plot annotations
CELL_ANNOT_FONTSIZE = 8.5      # in-cell triangle annotation (values)
KEY_LABEL_FONTSIZE = 12.0       # labels inside the key cell
HEATMAP_ANNOT_FONTSIZE = 8.0   # single-treebank heatmap annotations
KEY_TITLE_FONTSIZE = 12.0      # title above the dialect key
LEGEND_FONTSIZE = 9.0          # legend handle label size
LEGEND_TITLE_FONTSIZE = 10.0   # legend title font size
COLORBAR_TITLE_FONTSIZE = 12.0 # colourbar title font size
 
 
def _triangle(d, x, y):
    """Vertices of dialect *d*'s triangle in the cell whose top-left corner is (x, y)."""
    c = (x + 0.5, y + 0.5)
    return {
        "N": [(x, y),     (x + 1, y),     c],
        "E": [(x + 1, y), (x + 1, y + 1), c],
        "S": [(x, y + 1), (x + 1, y + 1), c],
        "W": [(x, y),     (x, y + 1),     c],
    }[d]
 
 
def _fmt(v):
    s = f"{v:.2f}"
    return s[1:] if s.startswith("0") else s     # ".95" instead of "0.95"
 
 
def _sd_linewidth(sd, sd_max):
    """
    Frame width on a log scale between MIN_SD_FOR_FRAME and sd_max:
    each doubling of the SD adds the same amount of thickness.
    """
    lo = MIN_SD_FOR_FRAME
    if sd_max <= lo:
        return SD_LW_MAX
    t = np.log(sd / lo) / np.log(sd_max / lo)
    return SD_LW_MIN + (SD_LW_MAX - SD_LW_MIN) * float(np.clip(t, 0, 1))
 
 
def _draw_key(ax):
    """A single cell showing which triangle is which dialect."""
    polys = [_triangle(d, 0, 0) for d in DIALECT_ORDER]
    ax.add_collection(PolyCollection(polys, facecolors="#f2f2f2",
                                     edgecolors="grey", linewidths=0.8))
    for d in DIALECT_ORDER:
        dx, dy = _TEXT_OFFSET[d]
        label = DIALECTS[d]["label"].replace(" (", "\n(")
        ax.text(dx, dy, label, ha="center", va="center", fontsize=KEY_LABEL_FONTSIZE)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(1.02, -0.02)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Dialect key", fontsize=KEY_TITLE_FONTSIZE)
 
 
def plot_dialect_matrix(cats, cube, sd, title, output_path, show=True):
    """
    Combined matrix: each cell split into one triangle per dialect,
    frame thickness = SD across dialects (log scale).
    """
    n = len(cats)
    cmap, norm = plt.get_cmap("YlOrRd"), Normalize(0, 1)
 
    side = max(CELL_INCHES * n, 6)
    fig = plt.figure(figsize=(side + 3.2, side + 1.0))
    gs  = fig.add_gridspec(3, 2, width_ratios=[side, 3.0],
                           height_ratios=[1.2, 1, 1], wspace=0.08)
    ax     = fig.add_subplot(gs[:, 0])
    key_ax = fig.add_subplot(gs[0, 1])
    leg_ax = fig.add_subplot(gs[1, 1]); leg_ax.axis("off")
    cb_ax  = fig.add_subplot(gs[2, 1]); cb_ax.axis("off")
 
    # --- triangles -------------------------------------------------------
    polys, colours, missing = [], [], []
    for i in range(n):                  # row    = original category
        for j in range(n):              # column = rival category
            for d_i, d in enumerate(DIALECT_ORDER):
                v    = cube[d_i, i, j]
                tri  = _triangle(d, j, i)
                if np.isnan(v):
                    missing.append(tri)
                    continue
                polys.append(tri)
                colours.append(cmap(norm(v)))
                if v >= ANNOT_MIN:
                    dx, dy = _TEXT_OFFSET[d]
                    ax.text(j + dx, i + dy, _fmt(v), ha="center", va="center",
                            fontsize=CELL_ANNOT_FONTSIZE, zorder=4,
                            color="white" if v > 0.6 else "black")
 
    ax.add_collection(PolyCollection(polys, facecolors=colours,
                                     edgecolors="white", linewidths=0.3, zorder=2))
    if missing:
        ax.add_collection(PolyCollection(missing, facecolors=NO_DATA_COLOUR,
                                         edgecolors="white", linewidths=0.3, zorder=2))
 
    # white grid between cells
    for k in range(n + 1):
        ax.axhline(k, color="white", linewidth=1.5, zorder=3)
        ax.axvline(k, color="white", linewidth=1.5, zorder=3)
 
    # --- axes (set before the frames: the frame inset needs the final cell size)
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    ax.apply_aspect()
    cell_pts = ax.get_position().width * fig.get_figwidth() * 72 / n   # points per cell
 
    # --- SD frames -------------------------------------------------------
    # Each frame hugs the inside edge of its cell, so it never spills into
    # the neighbouring cell and stays clear of the numbers.
    sd_filled = np.nan_to_num(sd, nan=0.0)
    sd_max    = sd_filled.max()
    for i, j in zip(*np.where(sd_filled >= MIN_SD_FOR_FRAME)):
        lw  = _sd_linewidth(sd_filled[i, j], sd_max)
        pad = (lw / 2 + 0.75) / cell_pts          # 0.75 pt = half the white grid line
        ax.add_patch(Rectangle(
            (j + pad, i + pad), 1 - 2 * pad, 1 - 2 * pad,
            fill=False, edgecolor="black", linewidth=lw, zorder=5,
            joinstyle="miter",
        ))
 
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels(cats, rotation=90)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels(cats)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_ylabel("Original Category (from Treebank)")
    ax.set_xlabel("Rival Category (The 'Intruder')")
    ax.set_title(title)
 
    # --- key, legend, colour bar ------------------------------------------
    _draw_key(key_ax)
 
    handles = []
    if sd_max >= MIN_SD_FOR_FRAME:
        lo = MIN_SD_FOR_FRAME
        for s in (lo, np.sqrt(lo * sd_max), sd_max):   # evenly spaced on the log scale
            handles.append(Line2D([0], [0], color="black",
                                  linewidth=_sd_linewidth(s, sd_max),
                                  label=f"SD = {s:.2f}"))
    handles.append(Patch(facecolor=cmap(0.0), edgecolor="lightgrey",
                         label="no number: value < .01"))
    if missing:
        handles.append(Patch(facecolor=NO_DATA_COLOUR, label="no data"))
    leg_ax.legend(
        handles=handles,
        loc="center",
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        title=f"Across-dialect SD (log scale;\nno frame below {MIN_SD_FOR_FRAME:.2f})",
        title_fontsize=LEGEND_TITLE_FONTSIZE,
        handlelength=2.5,
    )
 
    cax = cb_ax.inset_axes([0.08, 0.55, 0.84, 0.1])
    sm  = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, cax=cax, orientation="horizontal")
    cax.set_title("Stability score", fontsize=COLORBAR_TITLE_FONTSIZE)
    cax.tick_params(labelsize=8)
 
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()      # blocks until the window is closed
    plt.close(fig)
 
 
def _heatmap(matrix, title, output_path, show=False):
    """Ordinary single-treebank heatmap (as in the published paper)."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, cmap="YlOrRd", fmt=".2f", vmin=0, vmax=1,
                annot_kws={"fontsize": HEATMAP_ANNOT_FONTSIZE})
    plt.title(title)
    plt.ylabel("Original Category (from Treebank)")
    plt.xlabel("Rival Category (The 'Intruder')")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close()


# ===========================================================================
# Step 8 - Interactive exploration loop
# ===========================================================================

_HELP = """
Commands
--------
  use D               - switch dialect (D = one of {keys})
  cell ORIG RIVAL     - the four dialect values for one cell of both matrices
  ORIG RIVAL          - bridge words for ORIG -> RIVAL in the active dialect
  ORIG RIVAL word     - pulling features for a specific word
  agg ORIG RIVAL [k]  - aggregate pulling features, top-k words (default 5)
  cats                - list categories of the active dialect
  quit / exit         - quit
"""


def _available_cats(scores):
    return sorted({lu[1] for lu in scores})


def _run_interactive(results, cell_table):
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.float_format", "{:.3f}".format)

    active = DIALECT_ORDER[0]
    print(_HELP.format(keys=", ".join(DIALECT_ORDER)))

    while True:
        r = results[active]
        scores, corpus, sim = r["scores"], r["corpus"], r["sim"]
        try:
            raw = input(f"\n[{DIALECTS[active]['label']}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not raw:
            continue

        parts = raw.split()
        cmd   = parts[0].lower()

        if cmd in ("quit", "exit"):
            print("Exiting.")
            break

        if cmd == "use":
            if len(parts) == 2 and parts[1].upper() in DIALECTS:
                active = parts[1].upper()
            else:
                print(f"Usage: use D   (D = one of {', '.join(DIALECT_ORDER)})")
            continue

        if cmd == "cats":
            print("Available categories:", ", ".join(_available_cats(scores)))
            continue

        if cmd == "cell":
            if len(parts) != 3:
                print("Usage: cell ORIG RIVAL")
                continue
            sel = cell_table[(cell_table["original"] == parts[1]) &
                             (cell_table["rival"] == parts[2])]
            if sel.empty:
                print("No such cell.")
            else:
                print(sel.T.to_string(header=False))
            continue

        if cmd == "agg":
            if len(parts) < 3:
                print("Usage: agg ORIG RIVAL [k]")
                continue
            k  = int(parts[3]) if len(parts) >= 4 else 5
            df = get_aggregate_pulling_features(scores, parts[1], parts[2],
                                                corpus, sim, top_k_words=k)
            if df is not None:
                print(df.to_string(index=False))
            continue

        if len(parts) >= 2:
            orig, rival = parts[0], parts[1]
            cats = _available_cats(scores)
            if orig not in cats or rival not in cats:
                print(f"Unknown category. Available: {', '.join(cats)}")
                continue

            if len(parts) >= 3:
                lex_unit = (parts[2], orig)
                if lex_unit not in corpus._lexunit2idx:
                    print(f"'{parts[2]}' not found with tag '{orig}'.")
                    continue
                print(f"\n── Pulling features: {lex_unit} → '{rival}' ──")
                feats = get_pulling_features(lex_unit, rival, corpus, sim)
                if feats:
                    print(pd.DataFrame(feats).to_string(index=False))
            else:
                print(f"\n── Bridge words: '{orig}' → '{rival}' ──")
                bridges = get_top_bridge_words(scores, orig, rival, top_n=20)
                if bridges:
                    print(pd.DataFrame(bridges).to_string(index=False))
                else:
                    print("No bridge words found for this pair.")
            continue

        print("Unrecognised command.")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}

    # ------------------------------------------------------------------
    # 1. One pass per dialect
    # ------------------------------------------------------------------
    for d in DIALECT_ORDER:
        info = DIALECTS[d]
        print(f"\n=== {info['label']} ===")
        corpus = load_corpus(info["path"])
        sim    = build_similarity_matrix(corpus)
        freqs  = _global_category_freqs(corpus)

        stab, intr, n_units = build_matrices(corpus, sim, excluded_tags=EXCLUDED_TAGS)

        print(f"Lexical units: {len(corpus._idx2lexunit)}  "
              f"(with >= 1 neighbour: {int(n_units.sum())})")
        print("Units per category:", n_units.to_dict())
        print(f"Weighted Average Stability: {weighted_average_stability(stab, freqs):.4f}")

        results[d] = {
            "corpus": corpus, "sim": sim, "stab": stab, "intr": intr,
            "n_units": n_units, "scores": compute_scores(corpus, sim),
        }

        if SAVE_SINGLE_DIALECT_PLOTS:
            tb = os.path.basename(info["path"].rstrip("/"))
            _heatmap(stab, f"Stability Matrix - {tb}",
                     os.path.join(OUTPUT_DIR, f"{tb}_stability_matrix.png"))
            _heatmap(intr, f"Intruder Matrix - {tb}",
                     os.path.join(OUTPUT_DIR, f"{tb}_intruder_matrix.png"))

    # ------------------------------------------------------------------
    # 2. Align dialects, export, plot
    # ------------------------------------------------------------------
    n_units = {d: results[d]["n_units"] for d in DIALECT_ORDER}
    cats, stab_cube, stab_sd = stack_dialects({d: results[d]["stab"] for d in DIALECT_ORDER}, n_units)
    _,    intr_cube, intr_sd = stack_dialects({d: results[d]["intr"] for d in DIALECT_ORDER}, n_units)

    cell_table = export_cell_table(
        cats, stab_cube, intr_cube, stab_sd, intr_sd,
        os.path.join(OUTPUT_DIR, f"{LANGUAGE_NAME}_dialects_cells.csv"),
    )
    print("\nCells where the dialects differ most (stability matrix):")
    print(cell_table.head(10).to_string(index=False, float_format="{:.2f}".format))

    print("\nShowing plots - close each window to continue …")
    plot_dialect_matrix(cats, stab_cube, stab_sd,
                        f"Stability Matrix - {LANGUAGE_NAME} dialects",
                        os.path.join(OUTPUT_DIR, f"{LANGUAGE_NAME}_dialects_stability_matrix.png"))
    plot_dialect_matrix(cats, intr_cube, intr_sd,
                        f"Intruder Matrix - {LANGUAGE_NAME} dialects",
                        os.path.join(OUTPUT_DIR, f"{LANGUAGE_NAME}_dialects_intruder_matrix.png"))

    # ------------------------------------------------------------------
    # 3. Interactive exploration
    # ------------------------------------------------------------------
    _run_interactive(results, cell_table)
