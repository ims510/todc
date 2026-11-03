"""
analyze_results.py
==================

Visualisations for the cached output of parser_comparison_multitb.py.

Reads ./multi_treebank_cache/<treebank>/{spec.json, tokens.csv} for every
treebank and produces, in --output-dir:

  analysis_summary.csv     all per-treebank stats in one table
  forest_plot.png          purity coef + 95% CI per language, family-coloured
  rival_heatmap.png        rival hit rate by language x (purity x entropy) cell
  size_vs_effect_log.png    train tokens (log) vs purity-effect strength
  size_vs_effect_linear.png train tokens (linear) vs purity-effect strength
  morph_vs_effect.png       avg included features per token vs effect strength
  qualitative_panel_<name>.png  donut-pair panel for one or more treebanks

Cheap to re-run; tweak any plot and execute again. No re-parsing required.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter

import conllu
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CACHE_DIR  = "./multi_treebank_cache"
OUTPUT_DIR = "./multi_treebank_output/plots"

RIVAL_PROB_THRESHOLD = 0.05

INCLUDED_FEATURE_PATTERNS = [
    r"upos=",
    r"Abbr=",      r"Aspect=",    r"Animacy=",    r"Case=",
    r"Clusivity=", r"Definite=",  r"Deixis=",     r"DeixisRef",
    r"Evident=",   r"Negation=",  r"Number=",     r"Gender=",
    r"Degree=",    r"ExtPos=",    r"Foreign=",    r"Mood",
    r"NounClass=", r"NumType=",   r"Person=",     r"Polarity=",
    r"Polite=",    r"Poss=",      r"PronType=",   r"Reflex=",
    r"Tense=",     r"Typo=",      r"VerbForm=",   r"Voice=",
]
_INCLUDED_FEATURE_RE = re.compile("|".join(INCLUDED_FEATURE_PATTERNS))

# Language -> (display name, family). Underscores in lookup keys are
# normalised to spaces, so 'Old_Church_Slavonic' matches 'Old Church Slavonic'.
LANG_INFO = {
    # Romance
    "French": ("French", "Romance"),
    "Italian": ("Italian", "Romance"),
    "Spanish": ("Spanish", "Romance"),
    "Portuguese": ("Portuguese", "Romance"),
    "Catalan": ("Catalan", "Romance"),
    "Romanian": ("Romanian", "Romance"),
    "Galician": ("Galician", "Romance"),
    "Latin": ("Latin", "Romance"),
    "Ligurian": ("Ligurian", "Romance"),
    "Old French": ("Old French", "Romance"),
    # Germanic
    "English": ("English", "Germanic"),
    "German": ("German", "Germanic"),
    "Dutch": ("Dutch", "Germanic"),
    "Swedish": ("Swedish", "Germanic"),
    "Danish": ("Danish", "Germanic"),
    "Norwegian": ("Norwegian", "Germanic"),
    "Icelandic": ("Icelandic", "Germanic"),
    "Faroese": ("Faroese", "Germanic"),
    "Afrikaans": ("Afrikaans", "Germanic"),
    "Gothic": ("Gothic", "Germanic"),
    "Low Saxon": ("Low Saxon", "Germanic"),
    "Swiss German": ("Swiss German", "Germanic"),
    # Slavic
    "Russian": ("Russian", "Slavic"),
    "Polish": ("Polish", "Slavic"),
    "Czech": ("Czech", "Slavic"),
    "Slovak": ("Slovak", "Slavic"),
    "Slovenian": ("Slovenian", "Slavic"),
    "Croatian": ("Croatian", "Slavic"),
    "Serbian": ("Serbian", "Slavic"),
    "Ukrainian": ("Ukrainian", "Slavic"),
    "Belarusian": ("Belarusian", "Slavic"),
    "Bulgarian": ("Bulgarian", "Slavic"),
    "Macedonian": ("Macedonian", "Slavic"),
    "Old Church Slavonic": ("OCS", "Slavic"),
    "Old East Slavic": ("Old East Slavic", "Slavic"),
    "Upper Sorbian": ("Upper Sorbian", "Slavic"),
    "Pomak": ("Pomak", "Slavic"),
    # Indic
    "Hindi": ("Hindi", "Indic"),
    "Urdu": ("Urdu", "Indic"),
    "Bengali": ("Bengali", "Indic"),
    "Marathi": ("Marathi", "Indic"),
    "Sanskrit": ("Sanskrit", "Indic"),
    "Bhojpuri": ("Bhojpuri", "Indic"),
    "Maithili": ("Maithili", "Indic"),
    "Kangri": ("Kangri", "Indic"),
    # Iranian
    "Persian": ("Persian", "Iranian"),
    "Kurdish": ("Kurdish", "Iranian"),
    # Celtic
    "Irish": ("Irish", "Celtic"),
    "Welsh": ("Welsh", "Celtic"),
    "Scottish Gaelic": ("Scottish Gaelic", "Celtic"),
    "Breton": ("Breton", "Celtic"),
    "Manx": ("Manx", "Celtic"),
    "Old Irish": ("Old Irish", "Celtic"),
    # Baltic
    "Latvian": ("Latvian", "Baltic"),
    "Lithuanian": ("Lithuanian", "Baltic"),
    # Hellenic
    "Greek": ("Greek", "Hellenic"),
    "Ancient Greek": ("Ancient Greek", "Hellenic"),
    # Albanian / Armenian
    "Albanian": ("Albanian", "Albanian"),
    "Armenian": ("Armenian", "Armenian"),
    "Western Armenian": ("Western Armenian", "Armenian"),
    # Uralic
    "Finnish": ("Finnish", "Uralic"),
    "Estonian": ("Estonian", "Uralic"),
    "Hungarian": ("Hungarian", "Uralic"),
    "Erzya": ("Erzya", "Uralic"),
    "Karelian": ("Karelian", "Uralic"),
    "Komi Permyak": ("Komi-Permyak", "Uralic"),
    "Komi Zyrian": ("Komi-Zyrian", "Uralic"),
    "Livvi": ("Livvi", "Uralic"),
    "Moksha": ("Moksha", "Uralic"),
    "North Sami": ("North Sami", "Uralic"),
    "Skolt Sami": ("Skolt Sami", "Uralic"),
    # Turkic
    "Turkish": ("Turkish", "Turkic"),
    "Uzbek": ("Uzbek", "Turkic"),
    "Kazakh": ("Kazakh", "Turkic"),
    "Kyrgyz": ("Kyrgyz", "Turkic"),
    "Tatar": ("Tatar", "Turkic"),
    "Uyghur": ("Uyghur", "Turkic"),
    "Yakut": ("Yakut", "Turkic"),
    "Old Turkish": ("Old Turkish", "Turkic"),
    "Turkish German": ("Turkish-German", "Turkic"),
    # Semitic
    "Arabic": ("Arabic", "Semitic"),
    "Hebrew": ("Hebrew", "Semitic"),
    "Amharic": ("Amharic", "Semitic"),
    "Maltese": ("Maltese", "Semitic"),
    "Akkadian": ("Akkadian", "Semitic"),
    "Coptic": ("Coptic", "Afro-Asiatic"),
    # Sinitic
    "Chinese": ("Chinese", "Sinitic"),
    "Cantonese": ("Cantonese", "Sinitic"),
    "Classical Chinese": ("Classical Chinese", "Sinitic"),
    # Japonic / Koreanic
    "Japanese": ("Japanese", "Japonic"),
    "Korean": ("Korean", "Koreanic"),
    # Dravidian
    "Tamil": ("Tamil", "Dravidian"),
    "Telugu": ("Telugu", "Dravidian"),
    "Malayalam": ("Malayalam", "Dravidian"),
    # Austronesian / Austroasiatic / Tai-Kadai
    "Indonesian": ("Indonesian", "Austronesian"),
    "Tagalog": ("Tagalog", "Austronesian"),
    "Vietnamese": ("Vietnamese", "Austroasiatic"),
    "Thai": ("Thai", "Tai-Kadai"),
    # Niger-Congo
    "Wolof": ("Wolof", "Niger-Congo"),
    "Yoruba": ("Yoruba", "Niger-Congo"),
    # Other
    "Basque": ("Basque", "Isolate"),
    "Naija": ("Naija", "Creole"),
}

FAMILY_COLORS = {
    "Romance":       "#e41a1c",
    "Germanic":      "#377eb8",
    "Slavic":        "#4daf4a",
    "Indic":         "#984ea3",
    "Iranian":       "#a65628",
    "Celtic":        "#f781bf",
    "Baltic":        "#999999",
    "Hellenic":      "#dede00",
    "Albanian":      "#66c2a5",
    "Armenian":      "#fc8d62",
    "Uralic":        "#8da0cb",
    "Turkic":        "#e78ac3",
    "Semitic":       "#a6d854",
    "Afro-Asiatic":  "#bf812d",
    "Sinitic":       "#ffd92f",
    "Japonic":       "#e5c494",
    "Koreanic":      "#b3b3b3",
    "Dravidian":     "#1b9e77",
    "Austronesian":  "#d95f02",
    "Austroasiatic": "#7570b3",
    "Tai-Kadai":     "#e7298a",
    "Niger-Congo":   "#66a61e",
    "Isolate":       "#000000",
    "Creole":        "#bc80bd",
    "Other":         "#cccccc",
}

# Stable colour per UD POS tag, used in the qualitative donut panel
UD_POS_COLORS = {
    "NOUN":  "#1f77b4", "VERB":  "#d62728", "ADJ":   "#2ca02c",
    "ADV":   "#ff7f0e", "PRON":  "#9467bd", "DET":   "#8c564b",
    "ADP":   "#e377c2", "AUX":   "#7f7f7f", "CCONJ": "#bcbd22",
    "SCONJ": "#17becf", "PART":  "#aec7e8", "NUM":   "#ffbb78",
    "PROPN": "#98df8a", "INTJ":  "#ff9896", "PUNCT": "#c5b0d5",
    "SYM":   "#c49c94", "X":     "#f7b6d2",
    "_":     "#dddddd",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def parse_distribution(dist_str):
    """'ADJ: 0.45, NOUN: 0.30' -> {'ADJ': 0.45, 'NOUN': 0.30}."""
    if not dist_str or pd.isna(dist_str):
        return {}
    out = {}
    for piece in str(dist_str).split(","):
        piece = piece.strip()
        if ":" not in piece:
            continue
        cat, prob = piece.rsplit(":", 1)
        try:
            out[cat.strip()] = float(prob.strip())
        except ValueError:
            continue
    return out


def lookup_lang(lang_raw):
    """Return (display_name, family) for a UD-style language string."""
    if not lang_raw:
        return ("?", "Other")
    norm = lang_raw.replace("_", " ").strip()
    if norm in LANG_INFO:
        return LANG_INFO[norm]
    # try without trailing dialect markers etc.
    return (norm, "Other")


def load_treebank(tb_cache):
    """Load one cached treebank: returns dict or None."""
    spec_p   = os.path.join(tb_cache, "spec.json")
    tokens_p = os.path.join(tb_cache, "tokens.csv")
    if not (os.path.exists(spec_p) and os.path.exists(tokens_p)):
        return None
    with open(spec_p) as f:
        spec = json.load(f)
    df = pd.read_csv(tokens_p)
    display, family = lookup_lang(spec.get("language", ""))
    return {
        "name":        spec["name"],
        "language":    display,
        "family":      family,
        "spec":        spec,
        "df":          df,
    }


def load_all(cache_dir):
    out = []
    for name in sorted(os.listdir(cache_dir)):
        tb_cache = os.path.join(cache_dir, name)
        if not os.path.isdir(tb_cache):
            continue
        tb = load_treebank(tb_cache)
        if tb is not None:
            out.append(tb)
    return out


# ---------------------------------------------------------------------------
# Per-treebank statistics
# ---------------------------------------------------------------------------

def fit_regression(df):
    sub = df.dropna(subset=["purity"]).copy()
    if len(sub) < 50 or sub["pos_err"].sum() < 5:
        return None
    try:
        m  = smf.logit("pos_err ~ purity + log_freq", data=sub).fit(disp=0)
        ci = m.conf_int().loc["purity"]
        return {
            "coef":    float(m.params["purity"]),
            "se":      float(m.bse["purity"]),
            "ci_low":  float(ci.iloc[0]),
            "ci_high": float(ci.iloc[1]),
            "p":       float(m.pvalues["purity"]),
            "n_reg":   len(sub),
            "n_err":   int(sub["pos_err"].sum()),
        }
    except Exception:
        return None


def rival_alignment_per_cell(df, threshold=RIVAL_PROB_THRESHOLD):
    sub = df.dropna(subset=["distribution", "purity", "entropy"]).copy()
    if len(sub) == 0:
        return {}

    def in_rivals(row):
        d = parse_distribution(row["distribution"])
        rivals = {c for c, p in d.items()
                  if p >= threshold and c != row["gold_upos"]}
        return row["pred_upos"] in rivals

    sub["pred_in_rivals"] = sub.apply(in_rivals, axis=1)
    pur_med = sub["purity"].median()
    ent_med = sub["entropy"].median()
    sub["pur_lvl"] = np.where(sub["purity"]  <  pur_med, "low_pur",  "high_pur")
    sub["ent_lvl"] = np.where(sub["entropy"] >= ent_med, "high_ent", "low_ent")
    sub["cell"]    = sub["pur_lvl"] + " x " + sub["ent_lvl"]

    errs = sub[sub["pos_err"] == 1]
    out = {"hit_overall": (float(errs["pred_in_rivals"].mean())
                           if len(errs) else np.nan),
           "n_err_total": int(len(errs))}
    for cell in ["low_pur x low_ent",  "low_pur x high_ent",
                 "high_pur x low_ent", "high_pur x high_ent"]:
        ce = errs[errs["cell"] == cell]
        out[f"hit_{cell}"] = (float(ce["pred_in_rivals"].mean())
                              if len(ce) else np.nan)
        out[f"n_{cell}"]   = int(len(ce))
    return out


def morph_richness(spec):
    """Average number of morphological features per token in train file.
    Counts only features whose 'Key=Value' string matches a pattern in
    INCLUDED_FEATURE_PATTERNS, so the richness descriptor reflects the
    same feature set the framework uses to score lex units (otherwise we
    would be counting features that the purity computation ignores)."""
    train_path = os.path.join(spec["path"], spec["train_file"])
    if not os.path.exists(train_path):
        return np.nan
    n_tok = n_feat = 0
    try:
        with open(train_path, encoding="utf-8") as f:
            for sent in conllu.parse_incr(f):
                for tok in sent:
                    if not isinstance(tok["id"], int):
                        continue
                    n_tok += 1
                    feats = tok.get("feats")
                    if not feats:
                        continue
                    for k, v in feats.items():
                        if _INCLUDED_FEATURE_RE.search(f"{k}={v}"):
                            n_feat += 1
    except Exception as e:
        print(f"  morph_richness({spec['name']}): {e}")
        return np.nan
    return n_feat / n_tok if n_tok else np.nan


def build_stats(all_data, compute_morph=True):
    """One row per treebank: regression, rival alignment, descriptive stats."""
    rows = []
    for tb in all_data:
        df  = tb["df"]
        row = {"treebank":      tb["name"],
               "language":      tb["language"],
               "family":        tb["family"],
               "n_train":       int(tb["spec"].get("n_train", 0)),
               "n_test":        int(tb["spec"].get("n_test", 0)),
               "n_test_tokens": len(df),
               "n_pos_errors":  int(df["pos_err"].sum()),
               "err_rate":      float(df["pos_err"].mean()) if len(df) else np.nan}
        reg = fit_regression(df)
        if reg: row.update(reg)
        ra = rival_alignment_per_cell(df)
        if ra: row.update(ra)
        if compute_morph:
            row["morph_richness"] = morph_richness(tb["spec"])
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot 1: forest plot
# ---------------------------------------------------------------------------

def make_forest_plot(stats_df, output_path):
    valid = stats_df.dropna(subset=["coef"]).sort_values("coef").copy()
    if len(valid) == 0:
        print("  forest plot: no valid regressions")
        return
    fig, ax = plt.subplots(figsize=(9, max(3, 0.20 * len(valid))))
    y = np.arange(len(valid))
    fam_colors = [FAMILY_COLORS.get(f, FAMILY_COLORS.get("Other", "black"))
                  for f in valid["family"]]
    sig_colours = ["green" if hi < 0 else "red" if lo > 0 else "gray"
                for lo, hi in zip(valid["ci_low"], valid["ci_high"])]
    
    for yi, (xi, lo, hi, fam_c, sig_c) in enumerate(zip(valid["coef"],
                                                        valid["ci_low"],
                                                        valid["ci_high"],
                                                        fam_colors,
                                                        sig_colours)):
        ax.errorbar(xi, yi, xerr=[[xi - lo], [hi - xi]],
                    fmt='o', color=fam_c, ecolor=sig_c, capsize=3, markersize=6)

    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(valid["language"])
    
    # Optional but highly recommended: Color the language names (y-ticks) 
    # by family too. It makes spotting typological tendencies much easier!
    for tick_label, color in zip(ax.get_yticklabels(), fam_colors):
        tick_label.set_color(color)

    # 4. MARGINS: Remove the large empty gaps at the top and bottom of the plot
    ax.set_ylim(-0.5, len(valid) - 0.5)

    ax.set_xlabel("Purity coefficient (95% CI), controlling for log_freq")
    ax.set_title("Effect of stability on POS-tagging error across languages")
    
    families_present = [f for f in FAMILY_COLORS if f in set(valid["family"])]
    handles = [mpatches.Patch(color=FAMILY_COLORS[f], label=f)
               for f in families_present]
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              title="Family", title_fontsize=9, frameon=True)
              
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  saved {output_path}")


# ---------------------------------------------------------------------------
# Plot 2: rival hit-rate heatmap
# ---------------------------------------------------------------------------
def _wilson_ci(k, n, z=1.96):
    """Wilson 95% CI for a binomial proportion. Returns (lo, hi)."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half   = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def make_rival_heatmap(stats_df, output_path, min_cell_n=0, title=None, row_label_col="language"):
    cells = ["low_pur x low_ent",  "low_pur x high_ent",
             "high_pur x low_ent", "high_pur x high_ent"]
    pretty_cells = ["low purity\nlow entropy",
                    "low purity\nhigh entropy",
                    "high purity\nlow entropy",
                    "high purity\nhigh entropy"]
    rate_cols = [f"hit_{c}" for c in cells]
    n_cols    = [f"n_{c}"   for c in cells]
    sub = stats_df.dropna(subset=rate_cols, how="all").copy()
    if len(sub) == 0:
        print("  rival heatmap: no rival data")
        return
    sub = sub.sort_values(["family", "language"])
    M = sub[rate_cols].to_numpy(dtype=float)
    N = sub[n_cols].to_numpy(dtype=float)

    mask = (np.isnan(M)) | (np.isnan(N)) | (N < min_cell_n)
    M_masked = np.ma.masked_array(M, mask=mask)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#dddddd")  # light gray for missing cells

    fig, ax = plt.subplots(figsize=(7, max(3, 0.38 * len(sub))))
    im = ax.imshow(M_masked, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(pretty_cells, fontsize=9)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub[row_label_col])

    default_title = "Rival-alignment hit rate across (purity x entropy) cells"
    if min_cell_n > 0:
        default_title += f"  (cells with n<{min_cell_n} suppressed)"
    ax.set_title(title or default_title)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]; n = N[i, j]
            if mask[i, j]:
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, hatch="///", fill=False, edgecolor="#888888", linewidth=0.0, alpha=0.6))
                label = f"n={int(n)}" if not np.isnan(n) else "n=0"
                ax.text(j, i, label, ha="center", va="center", fontsize=7, color="#444444")
            else:
                txt_color = "white" if v < 0.5 else "black"
                ax.text(j, i, f"{v:.0%}\nn={int(n)}", ha="center", va="center", fontsize=7, color=txt_color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Hit rate (errors landing in framework rivals)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  saved {output_path}")

def pick_family_representatives(stats_df):
    """
    One treebank per family: the one with the largest |coef|/SE among
    treebanks where the regression converged. That metric combines effect
    size and precision -- exactly what 'strongest result with tightest CI'
    means, and it's symmetric, so we don't only pick negative coefs.
    """
    sub = stats_df.dropna(subset=["coef", "se"]).copy()
    sub = sub[sub["se"] > 0]
    if len(sub) == 0:
        return sub
    sub["t_abs"] = (sub["coef"] / sub["se"]).abs() # t-statistic = effect size / standard error
    sub = sub.sort_values("t_abs", ascending=False)
    rep = sub.drop_duplicates(subset=["family"], keep="first")
    return rep

def make_family_representative_heatmap(stats_df, output_path, min_cell_n=0):
    rep = pick_family_representatives(stats_df)
    if len(rep) == 0:
        print("  family-representative heatmap: no eligible treebanks")
        return
    rep = rep.copy()
    #annotate the family name in the row labels, to make it clear this is a family-level plot
    rep["row_label"] = rep["language"] + " (" + rep["family"] + ")"
    title = ("Rival-alignment hit rate -- one representative per family "
             "(largest |coef|/SE)")
    make_rival_heatmap(rep, output_path, min_cell_n=min_cell_n, title=title, row_label_col="row_label")

def compute_pooled_cells(all_data, threshold=RIVAL_PROB_THRESHOLD):
    """
    Pool errors across treebanks per (purity x entropy) cell.
 
    Each treebank's median split is computed within the treebank, so a
    "low_pur" lex unit means "below this language's median purity"
    consistently with the per-treebank analysis. We then sum hits and Ns
    across languages -- a sample-size-weighted aggregate, not a naive
    mean of rates.
    """

    cells = ["low_pur x low_ent",  "low_pur x high_ent",
             "high_pur x low_ent", "high_pur x high_ent"]
    totals = {c: {"hits": 0, "n": 0, "n_langs": 0} for c in cells}

    for tb in all_data:
        df = tb["df"]
        sub = df.dropna(subset=["distribution", "purity", "entropy"]).copy()
        if len(sub) == 0:
            continue

        def in_rivals(row):
            d = parse_distribution(row["distribution"])
            rivals = {c for c, p in d.items()
                      if p >= threshold and c != row["gold_upos"]}
            return row["pred_upos"] in rivals
        

        sub["pred_in_rivals"] = sub.apply(in_rivals, axis=1)
        pur_med = sub["purity"].median()
        ent_med = sub["entropy"].median()
        sub["pur_lvl"] = np.where(sub["purity"]  <  pur_med, "low_pur",  "high_pur")
        sub["ent_lvl"] = np.where(sub["entropy"] >= ent_med, "high_ent", "low_ent")
        sub["cell"]    = sub["pur_lvl"] + " x " + sub["ent_lvl"]
 
        errs = sub[sub["pos_err"] == 1]
        for cell in cells:
            ce = errs[errs["cell"] == cell]
            if len(ce) > 0:
                totals[cell]["hits"]    += int(ce["pred_in_rivals"].sum())
                totals[cell]["n"]       += int(len(ce))
                totals[cell]["n_langs"] += 1
    return totals

def make_pooled_4cell_heatmap(all_data, output_path):
    """2x2 grid of pooled hit rates across all treebanks, with Wilson 95% CI.
    Wilson 95% CI is like a fairness filter, it adjusts the confidence based
    on the sample size, the larger the sample size the more confident. so if the
     parser gets 5/5 it's not 95% sure the parser is 100% correct, it says that true accuracy
      is between 56% and 100% based on this tiny sample size """
    
    totals = compute_pooled_cells(all_data)
    layout = [["low_pur x low_ent",  "low_pur x high_ent"],
              ["high_pur x low_ent", "high_pur x high_ent"]]
    M  = np.full((2, 2), np.nan)
    Ns = np.zeros((2, 2), dtype=int)
    Hs = np.zeros((2, 2), dtype=int)
    Ls = np.zeros((2, 2), dtype=int)
    CI = [[(np.nan, np.nan)] * 2 for _ in range(2)]
    for i, row in enumerate(layout):
        for j, cell in enumerate(row):
            t = totals[cell]
            Ns[i, j] = t["n"]; Hs[i, j] = t["hits"]; Ls[i,j]=t["n_langs"]
            if t["n"] > 0:
                M[i, j] = t["hits"] / t["n"]
                CI[i][j] = _wilson_ci(t["hits"], t["n"])

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#dddddd")
    M_masked = np.ma.masked_invalid(M)
 
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    im = ax.imshow(M_masked, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["low entropy", "high entropy"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["low purity", "high purity"])
    ax.set_title("Pooled rival-alignment hit rate across all treebanks\n"
                 "(weighted by error count; Wilson 95% CI)")
    
    for i in range(2):
        for j in range(2):
            n = Ns[i, j]; k = Hs[i, j]; v = M[i, j]
            lo, hi = CI[i][j]
            n_langs = Ls[i, j]
            if n == 0:
                ax.text(j, i, "no data\n(n=0)",
                        ha="center", va="center", fontsize=10, color="#444")
                continue
            txt_color = "white" if v < 0.5 else "black"
            ax.text(j, i,
                    f"{v:.1%}\n"
                    f"[{lo:.0%}, {hi:.0%}]\n"
                    f"n={n} errors\n"
                    f"({n_langs} languages)",
                    ha="center", va="center",
                    fontsize=10, color=txt_color)
 
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("Pooled hit rate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  saved {output_path}")
# ---------------------------------------------------------------------------
# Plot 3: scatter of size / morph richness vs effect strength
# ---------------------------------------------------------------------------

def _scatter_with_fit(ax, x, y, families, languages, xlabel, ylabel, title):
    for i in range(len(x)):
        c = FAMILY_COLORS.get(families.iloc[i], FAMILY_COLORS["Other"])
        ax.scatter(x.iloc[i], y.iloc[i], color=c, s=60,
                   edgecolor="black", linewidth=0.5, zorder=3)
        ax.annotate(languages.iloc[i],
                    (x.iloc[i], y.iloc[i]),
                    xytext=(4, 3), textcoords="offset points",
                    fontsize=7, alpha=0.85)
    # OLS line for reference
    mask = (~x.isna()) & (~y.isna())
    if mask.sum() >= 3:
        xv = x[mask].to_numpy(); yv = y[mask].to_numpy()
        b1, b0 = np.polyfit(xv, yv, 1)
        xs = np.linspace(xv.min(), xv.max(), 50)
        ax.plot(xs, b0 + b1 * xs, color="gray", linestyle="--",
                linewidth=1, zorder=1, label=f"OLS slope={b1:.2f}")
        # Pearson r
        if xv.std() > 0 and yv.std() > 0:
            r = np.corrcoef(xv, yv)[0, 1]
            ax.text(0.02, 0.97, f"r = {r:+.2f}  (n={mask.sum()})",
                    transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white",
                              edgecolor="gray", alpha=0.8))
    ax.axhline(0, color="black", linestyle=":", linewidth=0.7, zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def make_scatter_plots(stats_df, out_size_log, out_size_linear, out_morph):
    sub = stats_df.dropna(subset=["coef"]).copy()
    if len(sub) < 3:
        print("  scatter: not enough valid regressions")
        return
    # 'effect strength' = -coef. Negative purity coef means high purity
    # protects against errors, which is the expected direction; flipping
    # the sign gives a y-axis where bigger = stronger expected effect.
    sub["effect"] = -sub["coef"]
    sub["log_n_train"] = np.log10(sub["n_train"].replace(0, np.nan))
    sub["n_train_k"]   = sub["n_train"] / 1_000.0

    fams = [f for f in FAMILY_COLORS if f in set(sub["family"])]
    handles = [mpatches.Patch(color=FAMILY_COLORS[f], label=f) for f in fams]

    # --- Size vs effect, log axis ---
    # UD treebanks span ~2-3 orders of magnitude (PDTC ~1.5M, smaller ones ~10k).
    # On a linear axis they pile up on the left; log spreads them out for
    # readability. The linear plot below answers the same question on the
    # raw scale, in case any non-linearity matters.
    fig, ax = plt.subplots(figsize=(8, 6))
    _scatter_with_fit(
        ax, sub["log_n_train"], sub["effect"],
        sub["family"], sub["language"],
        xlabel="log10(train tokens)",
        ylabel=r"Purity effect strength  ($-\beta_{\mathrm{purity}}$)",
        title="Treebank size vs strength of purity effect (log axis)",
    )
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              title="Family", title_fontsize=9, frameon=True)
    fig.tight_layout(); fig.savefig(out_size_log, dpi=150); plt.close(fig)
    print(f"  saved {out_size_log}")
 
    # --- Size vs effect, linear axis (raw token counts in thousands) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    _scatter_with_fit(
        ax, sub["n_train_k"], sub["effect"],
        sub["family"], sub["language"],
        xlabel="train tokens (thousands)",
        ylabel=r"Purity effect strength  ($-\beta_{\mathrm{purity}}$)",
        title="Treebank size vs strength of purity effect (linear axis)",
    )
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              title="Family", title_fontsize=9, frameon=True)
    fig.tight_layout(); fig.savefig(out_size_linear, dpi=150); plt.close(fig)
    print(f"  saved {out_size_linear}")

    # --- Morph richness vs effect ---
    if "morph_richness" in sub.columns and sub["morph_richness"].notna().sum() >= 3:
        fig, ax = plt.subplots(figsize=(8, 6))
        _scatter_with_fit(
            ax, sub["morph_richness"], sub["effect"],
            sub["family"], sub["language"],
            xlabel="Average UD features per token (train)",
            ylabel=r"Purity effect strength  ($-\beta_{\mathrm{purity}}$)",
            title="Morphological richness vs strength of purity effect",
        )
        ax.legend(handles=handles, loc="lower right", fontsize=8,
                  title="Family", title_fontsize=9, frameon=True)
        fig.tight_layout(); fig.savefig(out_morph, dpi=150); plt.close(fig)
        print(f"  saved {out_morph}")


# ---------------------------------------------------------------------------
# Plot 4: qualitative donut panel
# ---------------------------------------------------------------------------

def select_interesting_lex_units(df, n=5,
                                 max_purity=0.85,
                                 min_test_n=4,
                                 min_errors=2):
    """
    Pick lex units that are:
      - genuinely mixed in the framework (purity below threshold),
      - frequent enough in test to be informative,
      - actually erred on by the parser (otherwise the parser donut is dull).
    Sorted to surface the meatiest cases first.
    """
    sub = df.dropna(subset=["purity", "distribution"]).copy()
    grouped = (sub.groupby(["lemma", "gold_upos"])
                  .agg(n_test=("pos_err", "size"),
                       n_err=("pos_err", "sum"),
                       purity=("purity", "first"),
                       distribution=("distribution", "first"))
                  .reset_index())
    cand = grouped[(grouped["purity"] <= max_purity)
                   & (grouped["n_test"] >= min_test_n)
                   & (grouped["n_err"] >= min_errors)]
    # rank: more errors first, then lower purity
    cand = cand.sort_values(["n_err", "purity"], ascending=[False, True])
    return cand.head(n)


def _draw_donut(ax, dist, title, hole_radius=0.55):
    """dist: dict cat -> mass (will be normalised). Stable POS colours."""
    if not dist:
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                transform=ax.transAxes, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        return
    items  = sorted(dist.items(), key=lambda kv: -kv[1])
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    total  = sum(values) or 1.0
    pct    = [v / total for v in values]
    colors = [UD_POS_COLORS.get(l, "#cccccc") for l in labels]
    wedges, _ = ax.pie(values, colors=colors, startangle=90,
                       wedgeprops=dict(width=1 - hole_radius,
                                       edgecolor="white", linewidth=1))
    # Label only wedges that take >= 6% of the donut, to avoid clutter
    for w, lab, p in zip(wedges, labels, pct):
        if p < 0.06:
            continue
        ang = (w.theta2 + w.theta1) / 2
        x = 0.78 * np.cos(np.deg2rad(ang))
        y = 0.78 * np.sin(np.deg2rad(ang))
        ax.text(x, y, f"{lab}\n{p:.0%}",
                ha="center", va="center", fontsize=7.5,
                fontweight="bold", color="white")
    ax.set_title(title, fontsize=10)


def make_qualitative_panel(tb, output_path, n=5):
    """Two donuts (framework vs parser) per chosen lex unit, in a 5x2 grid."""
    df = tb["df"]
    units = select_interesting_lex_units(df, n=n)
    if len(units) == 0:
        print(f"  qualitative panel ({tb['name']}): no qualifying lex units")
        return

    rows = len(units)
    fig, axes = plt.subplots(rows, 2, figsize=(8, 3.2 * rows))
    if rows == 1:
        axes = np.array([axes])

    for i, u in enumerate(units.itertuples(index=False)):
        # Framework distribution from training (cached on each token row)
        framework = parse_distribution(u.distribution)

        # Parser predictions for this lex unit in test
        rows_tu = df[(df["lemma"] == u.lemma) & (df["gold_upos"] == u.gold_upos)]
        parser  = dict(Counter(rows_tu["pred_upos"].dropna()))

        title_l = (f"{u.lemma!r} ({u.gold_upos})  "
                   f"framework  |  purity={u.purity:.2f}")
        title_r = (f"{u.lemma!r} ({u.gold_upos})  "
                   f"parser  |  {int(u.n_err)} err / {int(u.n_test)} occ")
        _draw_donut(axes[i, 0], framework, title_l)
        _draw_donut(axes[i, 1], parser,    title_r)

    fig.suptitle(f"{tb['language']} ({tb['name']}): "
                 f"low-purity lex units -- framework vs parser",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  saved {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cache-dir",  default=CACHE_DIR)
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--examples-langs", nargs="+", default=None,
                   help="Treebank name(s) for the qualitative donut panel "
                        "(e.g. fr_gsd cs_pdt). Defaults to the treebank with "
                        "the most low-purity errors.")
    p.add_argument("--n-examples", type=int, default=5,
                   help="How many lex units to show in each qualitative panel.")
    p.add_argument("--no-morph", action="store_true",
                   help="Skip morphological-richness computation (faster, "
                        "skips the morph_vs_effect plot).")
    p.add_argument("--min-cell-n", type=int, default=0,
                   help="Suppress heatmap cells with fewer than this many "
                        "errors (rendered as gray hatch). Default 0 = show all.")
    p.add_argument("--skip", nargs="+", default=[],
                   choices=["forest", "heatmap", "family_heatmap",
                            "pooled_heatmap", "scatter", "qualitative"],
                   help="Plots to skip.")
    return p.parse_args()


def pick_default_example(stats_df, all_data):
    """Treebank with the most errors among low-purity test tokens."""
    best, best_score = None, -1
    by_name = {tb["name"]: tb for tb in all_data}
    for _, row in stats_df.iterrows():
        tb = by_name.get(row["treebank"])
        if tb is None: continue
        df = tb["df"]
        sub = df.dropna(subset=["purity"])
        score = int(((sub["purity"] <= 0.85) & (sub["pos_err"] == 1)).sum())
        if score > best_score:
            best_score, best = score, tb["name"]
    return best


def main():
    args = parse_args()
    if not os.path.isdir(args.cache_dir):
        print(f"No cache at {args.cache_dir}", file=sys.stderr); sys.exit(1)
    os.makedirs(args.output_dir, exist_ok=True)
 
    print(f"Loading cached treebanks from {args.cache_dir}...")
    all_data = load_all(args.cache_dir)
    if not all_data:
        print("No cached treebanks found.", file=sys.stderr); sys.exit(1)
    print(f"  loaded {len(all_data)} treebanks")
    for tb in all_data:
        print(f"    {tb['name']:<22} {tb['language']:<22} {tb['family']}")
 
    print("\nComputing per-treebank statistics...")
    stats_df = build_stats(all_data, compute_morph=not args.no_morph)
    summary_path = os.path.join(args.output_dir, "analysis_summary.csv")
    stats_df.to_csv(summary_path, index=False)
    print(f"  saved {summary_path}")
 
    print("\nGenerating plots...")
    if "forest" not in args.skip:
        make_forest_plot(stats_df,
                         os.path.join(args.output_dir, "forest_plot.png"))
    if "heatmap" not in args.skip:
        make_rival_heatmap(stats_df,
                           os.path.join(args.output_dir, "rival_heatmap.png"),
                           min_cell_n=args.min_cell_n)
    if "family_heatmap" not in args.skip:
        make_family_representative_heatmap(
            stats_df,
            os.path.join(args.output_dir, "rival_heatmap_family_reps.png"),
            min_cell_n=args.min_cell_n)
    if "pooled_heatmap" not in args.skip:
        make_pooled_4cell_heatmap(
            all_data,
            os.path.join(args.output_dir, "rival_heatmap_pooled.png"))
    if "scatter" not in args.skip:
        make_scatter_plots(stats_df,
                           os.path.join(args.output_dir, "size_vs_effect_log.png"),
                           os.path.join(args.output_dir, "size_vs_effect_linear.png"),
                           os.path.join(args.output_dir, "morph_vs_effect.png"))
    if "qualitative" not in args.skip:
        targets = args.examples_langs
        if not targets:
            d = pick_default_example(stats_df, all_data)
            targets = [d] if d else []
        by_name = {tb["name"]: tb for tb in all_data}
        for name in targets:
            tb = by_name.get(name)
            if tb is None:
                print(f"  qualitative: '{name}' not found in cache; skipping")
                continue
            out = os.path.join(args.output_dir,
                               f"qualitative_panel_{name}.png")
            make_qualitative_panel(tb, out, n=args.n_examples)


if __name__ == "__main__":
    main()